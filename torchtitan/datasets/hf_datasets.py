# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pickle
from typing import Any, Dict, List, Optional

import torch
from torch.distributed.checkpoint.stateful import Stateful
from torch.utils.data import IterableDataset
import numpy as np

try:
    from torchdata.stateful_dataloader import StatefulDataLoader
except ImportError as e:
    raise ImportError(
        "Please install the latest torchdata nightly to use StatefulDataloader via:"
        "pip3 install --pre torchdata --index-url https://download.pytorch.org/whl/nightly"
    ) from e

from copy import deepcopy
from torchtitan.datasets.tokenizer import Tokenizer
from torchtitan.logging_utils import logger

from datasets import load_dataset
from datasets.distributed import split_dataset_by_node
from torchtitan.datasets.image_utils import convert_image_base64_to_patches
# map from dataset name to a local directory, or
# a dataset repository on the HF hub
_supported_datasets = {
    "c4_mini": "torchtitan/datasets/c4_mini",
    "c4": "allenai/c4",
    'imagenet': "/shared/nas/data/m1/yangyic3/Multimodal-Mistral/data/processed/imagenet/",
    "imagenet+dclm": "/shared/nas/data/m1/yangyic3/Multimodal-Mistral/data/processed/imagenet+dclm/",
    "capfusion+dclm":"",
    "mix+dclm": ""
}


class HuggingFaceDatasetVL(IterableDataset, Stateful):
    """PyTorch Representation of the HuggingFace Dataset.

    Args:
        dataset_name (str): name of the dataset to load
        dataset_path (Optional[str]):
            Path to the dataset in the file system. If provided, data will be loaded
            from this path instead of downloaded.
        tokenizer (Tokenizer):
            Tokenizer used to encode data. Tokenize must implement an `encode` and `decode` method.
        seq_len (int): max sequence length
        world_size (int): number of data parallel processes participating in training
        rank (int): rank of the current data parallel process
        infinite (bool): whether to loop infinitely over the dataset

    We currently support the c4 dataset and a subset of it:
    c4_mini (45K training entries)
    c4 (177M training entries - this dataset is streamed due to the size)

    >> c4 (EN) <<:
    c4 cleaned, English version
    Data input format (c4):
    {
    'url': 'https://klyq.com/beginners-bbq-class-taking-place-in-missoula/',
    'text': 'Beginners BBQ Class Taking Place in Missoula!\nDo you want to get better at ...',
    'timestamp': '2019-04-25T12:57:54Z'
    }

    Example use (c4):
    >>> ds = HuggingFaceDataset(dataset_name="c4", dataset_path=None, tokenizer=tokenizer)
    >>> for batch in Dataloader(ds, batch_size=8):
            print(f"Batch size: {len(batch)}")
        Batch size: 8
    """ 
        


    def __init__(
        self,
        dataset_name: str,
        dataset_path: Optional[str],
        tokenizer: Tokenizer,
        seq_len: int = 2048,
        world_size: int = 1,
        rank: int = 0,
        infinite: bool = False,
    ) -> None:
        # allow user to pass in a (local or HF hub) path to use unsupported datasets
        if dataset_name not in _supported_datasets:
            if dataset_path:
                logger.warning(
                    f"Dataset {dataset_name} is not tested or verfied. "
                    f"Recommended datasets are: {list(_supported_datasets.keys())}."
                )
            else:
                raise ValueError(
                    f"Dataset {dataset_name} is not supported. "
                    f"Supported datasets are: {list(_supported_datasets.keys())}."
                )

        if not dataset_path:
            dataset_path = _supported_datasets[dataset_name]
        logger.info(f"Preparing {dataset_name} dataset from {dataset_path}")

        if dataset_name == "c4":
            ds = load_dataset(dataset_path, name="en", split="train", streaming=True)
        elif dataset_name == 'imagenet':
            ds = load_dataset(dataset_path, split="train", streaming=True)
        elif dataset_name == 'imagenet+dclm':
            ds = load_dataset("/shared/nas/data/m1/yangyic3/Multimodal-Mistral/data/processed/imagenet/", split="train", streaming=True)
            ds_2 = load_dataset("torchtitan/datasets/processed/dclm/", split="train", streaming=True)  
        elif dataset_name == "capfusion+dclm":
            ds = load_dataset("/shared/nas/data/m1/yangyic3/Multimodal-Mistral/data/processed/capfusion/", split="train", streaming=True)
            ds_2 = load_dataset("torchtitan/datasets/processed/dclm/", split="train", streaming=True)
        elif dataset_name == 'mix+dclm':
            ds_2 = load_dataset("torchtitan/datasets/processed/dclm/", split="train", streaming=True)
            
            # caption
            ds_a = load_dataset("/shared/nas/data/m1/yangyic3/Multimodal-Mistral/data/processed/recaption/", split="train", streaming=True)
            ds_b = load_dataset("/shared/nas/data/m1/yangyic3/Multimodal-Mistral/data/processed/pixmo_cap/", split="train", streaming=True)
            # digital
            ds_c = load_dataset("/shared/nas/data/m1/yangyic3/Multimodal-Mistral/data/processed/pixmo_doc", split="train", streaming=True)
            ds_d = load_dataset("/shared/nas/data/m1/yangyic3/Multimodal-Mistral/data/processed/digital/", split="train", streaming=True)
            # ocr
            ds_e = load_dataset("/shared/nas/data/m1/yangyic3/Multimodal-Mistral/data/processed/recaption_ocr/", split="train", streaming=True)
            
            
            self._data_a = split_dataset_by_node(ds_a, rank, world_size)
            self._data_b = split_dataset_by_node(ds_b, rank, world_size)
            self._data_c = split_dataset_by_node(ds_c, rank, world_size)
            self._data_d = split_dataset_by_node(ds_d, rank, world_size)
            self._data_e = split_dataset_by_node(ds_e, rank, world_size)
            
        else:
            ds = load_dataset(dataset_path, split="train")

        # TODO: support shuffling and checkpointing
        self.dataset_name = dataset_name
        
        self._data_2 = split_dataset_by_node(ds_2, rank, world_size)
        self._tokenizer = tokenizer
        self.seq_len = seq_len
        self.infinite = infinite
        self.language_data_ratio = 0.5 #0.5
        self.img_after_prob = 0.5 # 0.66

        # variables for checkpointing
        self._sample_idx = 0
        
        # language-based image generation
        self._all_tokens: List[int] = []
        self._all_vision_patches: List[np.ndarray] = []
        self._all_vision_patches_indices: List[int] = []
        self._all_labels: List[int] = []
        self._all_noise: List[np.ndarray] = []
        self._all_noise_patches_indices: List[int] = []
        
        
        
        # language-only data
        self._all_tokens_language: List[int] = []
        self._all_vision_patches_language: List[np.ndarray] = []
        self._all_vision_patches_indices_language: List[int] = []
        self._all_labels_language: List[int] = []
        self._all_noise_language: List[np.ndarray] = []
        self._all_noise_patches_indices_language: List[int] = []
        
        
        # image-conditioned language generation (e.g., image classification, image captioning, visual instruction following)
        self._all_tokens_vl: List[int] = []
        self._all_vision_patches_vl: List[np.ndarray] = []
        self._all_vision_patches_indices_vl: List[int] = []
        self._all_labels_vl: List[int] = []
        self._all_noise_vl: List[np.ndarray] = []
        self._all_noise_patches_indices_vl: List[int] = []
        
        
        
        
        
        
        
    def add_vl_data(self, sample):
        NON_VISION_TOKEN = -1
        
        content = sample['content']
        content_bef = content[0]
        content_aft = content[1]
        if content_bef['type'] == 'text':
            image = content_aft['image_url']['url']
        else:
            image = content_bef['image_url']['url']
        patches = convert_image_base64_to_patches(image)
        n_rows, n_cols = patches.shape[:2]
        n_patches = n_rows * n_cols
        patches = patches.view(n_patches, -1)[0].unsqueeze(0) 
        assert patches.shape[0] == 1, f"{patches.shape[0]} != 1"
        
        
        img_tokens = ["<vpatch>"]
        cur_patch_indices =[len(self._all_vision_patches_vl) + 0]
        cur_noise_patch_indices = [len(self._all_noise_vl) + 0] 
        cur_tokens = self._tokenizer.encode(''.join(img_tokens), bos=False, eos=False)
        # cur_tokens = self._tokenizer.convert_tokens_to_ids(img_tokens) # return a list of int
        assert len(cur_tokens) == len(cur_patch_indices), f"{len(cur_tokens)} != {len(cur_patch_indices)}"
        self._all_tokens_vl.extend(cur_tokens)
        self._all_vision_patches_indices_vl.extend(cur_patch_indices)
        self._all_noise_patches_indices_vl.extend(cur_noise_patch_indices)
        noisy_image_patches = patches
        self._all_vision_patches_vl.extend(noisy_image_patches.numpy())
        self._all_noise_vl.extend(noisy_image_patches.numpy())
        self._all_labels_vl.extend([-100] * len(cur_tokens))        
        

        content = sample['content']
        content_bef = content[0]
        content_aft = content[1]
        if content_bef['type'] == 'text':
            sample_text = content_bef['text']
            image = content_aft['image_url']['url']
        else:
            sample_text = content_aft['text']
            image = content_bef['image_url']['url']
        sample_text = sample_text.replace("_", " ", 1000).replace("-", " ", 1000)
        
        patches = convert_image_base64_to_patches(image)
        n_rows, n_cols = patches.shape[:2]
        n_patches = n_rows * n_cols

        patches = patches.view(n_patches, -1) # shape: (w_patch_num * h_patch_num, patch_size * patch_size * 3)
        # ---
        img_tokens = ["<vision>"]
        cur_patch_indices = [NON_VISION_TOKEN]
        for row_idx in range(n_rows):
            for col_idx in range(n_cols):
                if row_idx != 0 and col_idx == 0: # when new row starts
                    img_tokens.append(f"<vrow_sep>")
                    cur_patch_indices.append(NON_VISION_TOKEN)
                img_tokens.append(f"<vpatch>")
                cur_patch_indices.append(len(self._all_vision_patches_vl) + row_idx * n_cols + col_idx)
        img_tokens.append("<vision>")
        cur_patch_indices.append(NON_VISION_TOKEN)
        
        
        
        cur_tokens = self._tokenizer.encode(''.join(img_tokens), bos=False, eos=False)
        # cur_tokens = self._tokenizer.convert_tokens_to_ids(img_tokens) # return a list of int
        assert len(cur_tokens) == len(cur_patch_indices), f"{len(cur_tokens)} != {len(cur_patch_indices)}"
        
        self._all_tokens_vl.extend(cur_tokens)
        self._all_vision_patches_indices_vl.extend(cur_patch_indices)
        # self._all_noise_patches_indices_vl.extend(cur_noise_patch_indices)
        self._all_vision_patches_vl.extend(patches.numpy())
        # self._all_noise_vl.extend(patches.numpy())
        self._all_labels_vl.extend([-100] * len(cur_tokens))
        
        prefix_tokens = self._tokenizer.encode("<IMG_UND>", bos=True, eos=False)
        self._all_tokens_vl.extend(prefix_tokens)
        self._all_vision_patches_indices_vl.extend([NON_VISION_TOKEN] * len(prefix_tokens))
        # self._all_noise_patches_indices_vl.extend([NON_VISION_TOKEN] * len(prefix_tokens))
        self._all_labels_vl.extend([-100] * len(prefix_tokens))  
    
        sample_tokens = self._tokenizer.encode(sample_text, bos=False, eos=True)
        self._all_tokens_vl.extend(sample_tokens)
        self._all_vision_patches_indices_vl.extend([NON_VISION_TOKEN] * len(sample_tokens))
        # self._all_noise_patches_indices_vl.extend([NON_VISION_TOKEN] * len(sample_tokens))
        self._all_labels_vl.extend(sample_tokens)
    

    def add_generation_data(self, sample):
        NON_VISION_TOKEN = -1
        content = sample['content']
        content_bef = content[0]
        content_aft = content[1]
        if content_bef['type'] == 'text':
            sample_text = content_bef['text']
            image = content_aft['image_url']['url']
        else:
            sample_text = content_aft['text']
            image = content_bef['image_url']['url']
        # remove all "_" and "-" in the sample_text
        sample_text = sample_text.replace("_", " ", 1000).replace("-", " ", 1000)
        
        patches = convert_image_base64_to_patches(image)
        n_rows, n_cols = patches.shape[:2]
        n_patches = n_rows * n_cols
        patches = patches.view(n_patches, -1) # shape: (w_patch_num * h_patch_num, patch_size * patch_size * 3)
        
        sample_tokens = self._tokenizer.encode("<IMG_GEN>" + sample_text, bos=True, eos=True)
        self._all_tokens.extend(sample_tokens)
        self._all_vision_patches_indices.extend([NON_VISION_TOKEN] * len(sample_tokens))
        self._all_labels.extend([-100] * len(sample_tokens))
        self._all_noise_patches_indices.extend([NON_VISION_TOKEN] * len(sample_tokens))
        
        # ---
        img_tokens = ["<vision>"]
        cur_patch_indices = [NON_VISION_TOKEN]
        cur_noise_patch_indices = [len(self._all_noise)]
        for row_idx in range(n_rows):
            for col_idx in range(n_cols):
                if row_idx != 0 and col_idx == 0: # when new row starts
                    img_tokens.append(f"<vrow_sep>")
                    cur_patch_indices.append(NON_VISION_TOKEN)
                    cur_noise_patch_indices.append(len(self._all_noise) + row_idx * n_cols)
                img_tokens.append(f"<vpatch>")
                cur_patch_indices.append(len(self._all_vision_patches) + row_idx * n_cols + col_idx)
                if col_idx == (n_cols - 1):
                    cur_noise_patch_indices.append(NON_VISION_TOKEN)
                else:
                    cur_noise_patch_indices.append(len(self._all_noise) + 1 + row_idx * n_cols + col_idx)
        img_tokens.append("<vision>")
        cur_patch_indices.append(NON_VISION_TOKEN)
        cur_noise_patch_indices.append(NON_VISION_TOKEN)
        
        cur_tokens = self._tokenizer.encode(''.join(img_tokens), bos=False, eos=False)
        assert len(cur_tokens) == len(cur_patch_indices), f"{len(cur_tokens)} != {len(cur_patch_indices)}"
        self._all_tokens.extend(cur_tokens)
        self._all_vision_patches_indices.extend(cur_patch_indices)
        self._all_noise_patches_indices.extend(cur_noise_patch_indices)
        
        noisy_image_patches = deepcopy(patches.numpy())
        self._all_vision_patches.extend(patches.numpy())
        self._all_noise.extend(noisy_image_patches)
        self._all_labels.extend([-100] * len(cur_tokens))        
        
        
        
    
    def add_language_data(self, sample, sample_img):
        ### add dummy data for the image generation
        NON_VISION_TOKEN = -1
        content = sample_img['content']
        content_bef = content[0]
        content_aft = content[1]
        if content_bef['type'] == 'text':
            sample_text = content_bef['text']
            image = content_aft['image_url']['url']
        else:
            sample_text = content_aft['text']
            image = content_bef['image_url']['url']
        patches = convert_image_base64_to_patches(image)
        n_rows, n_cols = patches.shape[:2]
        n_patches = n_rows * n_cols
        patches = patches.view(n_patches, -1)[0].unsqueeze(0) 
        assert patches.shape[0] == 1, f"{patches.shape[0]} != 1"
        img_tokens = ["<vpatch>"]
        cur_patch_indices =[len(self._all_vision_patches_language) + 0]
        cur_noise_patch_indices = [len(self._all_noise_language) + 0] 
        cur_tokens = self._tokenizer.encode(''.join(img_tokens), bos=False, eos=False)
        # cur_tokens = self._tokenizer.convert_tokens_to_ids(img_tokens) # return a list of int
        assert len(cur_tokens) == len(cur_patch_indices), f"{len(cur_tokens)} != {len(cur_patch_indices)}"
        self._all_tokens_language.extend(cur_tokens)
        self._all_vision_patches_indices_language.extend(cur_patch_indices)
        self._all_noise_patches_indices_language.extend(cur_noise_patch_indices)
        noisy_image_patches = patches.numpy()
        self._all_vision_patches_language.extend(noisy_image_patches)
        self._all_noise_language.extend(noisy_image_patches)
        self._all_labels_language.extend([-100] * len(cur_tokens))        
        
        #### add language data
        content = sample['content']
        sample_text = content[0]['text']
        sample_tokens = self._tokenizer.encode(sample_text, bos=True, eos=True)
        if len(sample_tokens) >= self.seq_len: 
            sample_tokens = sample_tokens[:self.seq_len-1] + [self._tokenizer.eos_id]
        self._all_tokens_language.extend(sample_tokens)
        self._all_vision_patches_indices_language.extend([NON_VISION_TOKEN] * len(sample_tokens))
        self._all_labels_language.extend(sample_tokens)
        self._all_noise_patches_indices_language.extend([NON_VISION_TOKEN] * len(sample_tokens))
        

    

    def __iter__(self):
        max_buffer_token_len = 1 + self.seq_len
        data_iter_1 = self._get_data_iter()
        data_iter_2 = self._get_data_iter_2()
        
        
        while True:
            while len(self._all_tokens_language) < max_buffer_token_len:
                sample = next(data_iter_2)
                try:
                    sample_img = next(data_iter_1)
                except StopIteration:
                    data_iter_1 = self._get_data_iter()
                    sample_img = next(data_iter_1)
                self.add_language_data(sample, sample_img)
            
            while len(self._all_tokens) < max_buffer_token_len:
                try:
                    sample = next(data_iter_1)
                except StopIteration:
                    data_iter_1 = self._get_data_iter()
                    sample = next(data_iter_1)
                
                self.add_generation_data(sample)
        
            while len(self._all_tokens_vl) < max_buffer_token_len:
                try:
                    sample = next(data_iter_1)
                except StopIteration:
                    data_iter_1 = self._get_data_iter()
                    sample = next(data_iter_1)
                self.add_vl_data(sample)
            
            
            
            
            if np.random.rand() < self.language_data_ratio:
                x = torch.LongTensor(self._all_tokens_language[:max_buffer_token_len])
                input_ids = x[:-1]
                x = torch.LongTensor(self._all_labels_language[:max_buffer_token_len])
                label = x[1:]
                indices = torch.LongTensor(self._all_vision_patches_indices_language[:max_buffer_token_len])
                noise_indices = torch.LongTensor(self._all_noise_patches_indices_language[:max_buffer_token_len])
                
                # get the max number from indices
                max_idx = indices.max() + 1
                max_idx_noise = noise_indices.max() + 1
                
                indices = indices[:-1]
                noise_indices = noise_indices[:-1]
                
                vision_patches = torch.FloatTensor(np.array(self._all_vision_patches_language[:max_idx]))
                noise_patches = torch.FloatTensor(np.array(self._all_noise_language[:max_idx_noise]))
                
                
                # update tokens to the remaining tokens
                self._all_tokens_language = self._all_tokens_language[max_buffer_token_len:]
                self._all_labels_language = self._all_labels_language[max_buffer_token_len:]
                self._all_vision_patches_indices_language = self.modify_numbers_numpy(self._all_vision_patches_indices_language[max_buffer_token_len:], max_idx.item())
                self._all_noise_patches_indices_language = self.modify_numbers_numpy(self._all_noise_patches_indices_language[max_buffer_token_len:], max_idx_noise.item())
                
                self._all_vision_patches_language = self._all_vision_patches_language[max_idx:]
                self._all_noise_language = self._all_noise_language[max_idx_noise:]
                
                # logger.info(f"vision_patches: {vision_patches.shape}, max_idx: {max_idx}")
                yield input_ids, label, indices, vision_patches, noise_patches, noise_indices
                
                
                
                    
            else:
                if np.random.rand() < self.img_after_prob:
                    # doing language-conditioned image generation
                    x = torch.LongTensor(self._all_tokens[:max_buffer_token_len])
                    input_ids = x[:-1]
                    x = torch.LongTensor(self._all_labels[:max_buffer_token_len])
                    label = x[1:]
                    indices = torch.LongTensor(self._all_vision_patches_indices[:max_buffer_token_len])
                    noise_indices = torch.LongTensor(self._all_noise_patches_indices[:max_buffer_token_len])
                                    
                    # get the max number from indices
                    max_idx = indices.max() + 1
                    max_idx_noise = noise_indices.max() + 1
                    
                    indices = indices[:-1]
                    noise_indices = noise_indices[:-1]
                    
                    vision_patches = torch.FloatTensor(np.array(self._all_vision_patches[:max_idx]))
                    noise_patches = torch.FloatTensor(np.array(self._all_noise[:max_idx_noise]))
                    
                    
                    # update tokens to the remaining tokens
                    self._all_tokens = []
                    self._all_labels = []
                    self._all_vision_patches_indices = []
                    self._all_noise_patches_indices = []
                    self._all_vision_patches = []
                    self._all_noise = []
        
                    yield input_ids, label, indices, vision_patches, noise_patches, noise_indices
                
                else:
                    # doing image-conditioned language generation (also with image construction)
                    x = torch.LongTensor(self._all_tokens_vl[:max_buffer_token_len])
                    input_ids = x[:-1]
                    x = torch.LongTensor(self._all_labels_vl[:max_buffer_token_len])
                    label = x[1:]
                    indices = torch.LongTensor(self._all_vision_patches_indices_vl[:max_buffer_token_len])
                    noise_indices = torch.LongTensor(self._all_noise_patches_indices_vl[:max_buffer_token_len])
                                    
                    # get the max number from indices
                    # logger.info(indices)
                    max_idx = indices.max() + 1
                    max_idx_noise = noise_indices.max() + 1
                    
                    indices = indices[:-1]
                    noise_indices = noise_indices[:-1]
                    
                    vision_patches = torch.FloatTensor(np.array(self._all_vision_patches_vl[:max_idx]))
                    noise_patches = torch.FloatTensor(np.array(self._all_noise_vl[:max_idx_noise])) 
                    
                    # update tokens to the remaining tokens
                    self._all_tokens_vl = []
                    self._all_labels_vl = []
                    self._all_vision_patches_indices_vl = []
                    self._all_noise_patches_indices_vl = []
                    self._all_vision_patches_vl = []
                    self._all_noise_vl = []

                    yield input_ids, label, indices, vision_patches, noise_patches, noise_indices




    
    
    
    
    def modify_numbers_numpy(self, nums, max_idx):
        arr = np.array(nums)
        arr[arr != -1] -= max_idx
        return arr.tolist()


    
    
    def _get_data_iter(self):
        class CustomIter:
            def __init__(self, data_a, data_b, data_c, data_d, data_e):
                self.data_iter_a = iter(data_a)
                self.data_iter_b = iter(data_b)
                self.data_iter_c = iter(data_c)
                self.data_iter_d = iter(data_d)
                self.data_iter_e = iter(data_e)
                self.data_list = [data_a, data_b, data_c, data_d, data_e]
                self.iter_list = [self.data_iter_a, self.data_iter_b, self.data_iter_c, self.data_iter_d, self.data_iter_e]
                self.probability_list = [0.4, 0.2, 0.03, 0.07, 0.3]
            
            
            def __next__(self):
                # randomly select one of the data_iter according to the probability
                idx = np.random.choice(len(self.iter_list), p=self.probability_list)
                tgt_iter = self.iter_list[idx]
                try:
                    sample = next(tgt_iter)
                except StopIteration:
                    self.iter_list[idx] = iter(self.data_list[idx])
                    tgt_iter = self.iter_list[idx]
                    sample = next(tgt_iter)
                return sample
                    
        
        return CustomIter(self._data_a, self._data_b, self._data_c, self._data_d, self._data_e)
    
    




    def _get_data_iter_2(self):
        return iter(self._data_2)
        
        

    
    
    def load_state_dict(self, state_dict):
        self._sample_idx = state_dict["sample_idx"]
        self._all_tokens = state_dict["token_buffer"]

    def state_dict(self):
        return {"token_buffer": self._all_tokens, "sample_idx": self._sample_idx}


class DPAwareDataLoader(StatefulDataLoader, Stateful):
    """
    A wrapper around the StatefulDataLoader that ensures that the state is stored only once per DP rank.
    """

    def __init__(self, dp_rank: int, hf_ds: IterableDataset, batch_size: int):
        super().__init__(hf_ds, batch_size)
        self._dp_rank = dp_rank
        self._rank_id = f"dp_rank_{dp_rank}"

    def state_dict(self) -> Dict[str, Any]:
        # Store state only for dp rank to avoid replicating the same state across other dimensions
        return {self._rank_id: pickle.dumps(super().state_dict())}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        # State being empty is valid, don't log a warning
        if not state_dict:
            return

        if self._rank_id not in state_dict:
            logger.warning(
                f"DataLoader state is empty for dp rank {self._dp_rank}, expected key {self._rank_id}."
            )
            return
        super().load_state_dict(pickle.loads(state_dict[self._rank_id]))




def build_hf_data_loader(
    dataset_name: str,
    dataset_path: Optional[str],
    tokenizer: Tokenizer,
    batch_size: int,
    seq_len: int,
    world_size,
    rank,
    infinite: bool = True,
):
    hf_ds = HuggingFaceDatasetVL(
        dataset_name, dataset_path, tokenizer, seq_len, world_size, rank, infinite
    )

    return DPAwareDataLoader(rank, hf_ds, batch_size=batch_size)

