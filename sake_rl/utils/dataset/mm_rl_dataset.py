import base64
import copy
import os
import pickle
from collections import defaultdict
from typing import List, Optional, Union

import numpy as np
import pandas as pd
import torch
import verl.utils.torch_functional as verl_F
from omegaconf import ListConfig
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin
from verl.utils.model import compute_position_id_with_mask


def collate_fn(data_list: list[dict]) -> dict:
    tensors = defaultdict(list)
    non_tensors = defaultdict(list)

    for data in data_list:
        for key, val in data.items():
            if isinstance(val, torch.Tensor):
                tensors[key].append(val)
            else:
                non_tensors[key].append(val)

    for key, val in tensors.items():
        tensors[key] = torch.stack(val, dim=0)

    for key, val in non_tensors.items():
        non_tensors[key] = np.array(val, dtype=object)

    return {**tensors, **non_tensors}


def process_image(image: dict, max_pixels: int = 672 * 672 * 2, min_pixels: int = 512 * 512):
    """
    Process image with resize and return processed image along with size information.
    
    Returns:
        processed_image: PIL Image object
        original_size: (width, height) tuple of original image
        processed_size: (width, height) tuple of processed image
    """
    import math
    from io import BytesIO

    from PIL import Image

    if isinstance(image, dict):
        image = Image.open(BytesIO(image['bytes']))

    # Store original size
    original_width, original_height = image.width, image.height
    original_size = (original_width, original_height)

    # Resize if too large
    if (image.width * image.height) > max_pixels:
        resize_factor = math.sqrt(max_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height), resample=Image.Resampling.NEAREST)

    # Resize if too small
    if (image.width * image.height) < min_pixels:
        resize_factor = math.sqrt(min_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height), resample=Image.Resampling.NEAREST)

    if image.mode != 'RGB':
        image = image.convert('RGB')

    processed_size = (image.width, image.height)
    
    return image, original_size, processed_size


class RLHFDataset(Dataset):
    """
    We assume the dataset contains a column that contains prompts and other information
    """

    def __init__(
        self,
        parquet_files: Union[str, List[str]],
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin] = None,
        prompt_key='prompt',
        image_key='images',
        max_prompt_length=1024,
        filter_prompts=True,
        cache_dir='~/.cache/verl/rlhf',
        chat_template_func=None,
        return_raw_chat=False,
        truncation='error',
        user_prompt_round_1=None,
    ):
        if not isinstance(parquet_files, (List, ListConfig)):
            parquet_files = [parquet_files]

        self.parquet_files = copy.deepcopy(parquet_files)
        self.original_parquet_files = copy.deepcopy(parquet_files)  # use for resume
        self.cache_dir = os.path.expanduser(cache_dir)
        self.tokenizer = tokenizer
        self.processor = processor

        self.prompt_key = prompt_key
        self.image_key = image_key
        self.max_prompt_length = max_prompt_length
        self.filter_prompts = filter_prompts

        self.return_raw_chat = return_raw_chat
        self.chat_template_func = chat_template_func
        self.truncation = truncation

        self.user_prompt_round_1 = user_prompt_round_1

        # whether to store the dataset in state_dict()
        # default not store
        self.serialize_dataset = False
        self._download()
        self._read_files_and_tokenize()

    def _download(self, use_origin_parquet=False):
        from verl.utils.fs import copy_to_local

        parquet_files = self.parquet_files if not use_origin_parquet else self.original_parquet_files
        for i, parquet_file in enumerate(parquet_files):
            self.parquet_files[i] = copy_to_local(src=parquet_file, cache_dir=self.cache_dir)

    def _read_files_and_tokenize(self):
        dataframes = []
        for parquet_file in self.parquet_files:
            # read parquet files and cache
            dataframe = pd.read_parquet(parquet_file)
            dataframes.append(dataframe)
        self.dataframe = pd.concat(dataframes)

        print(f'original dataset len: {len(self.dataframe)}')

        # filter out too long prompts
        tokenizer = self.tokenizer
        prompt_key = self.prompt_key

        # HACK: update prompt (hard rule for qwen-vl model)
        if self.user_prompt_round_1 is not None:
            with open(
                self.user_prompt_round_1, 'rb'
            ) as file:  # NOTE: we use pickle since strings with "\n" can't be correctly passed don't know why
                user_prompt_round_1 = pickle.load(file)
            print(f"[Prompt Set] Round-1 User Prompt: {user_prompt_round_1}")
            self.dataframe[prompt_key] = self.dataframe[prompt_key].apply(
                lambda x: np.array([{'content': user_prompt_round_1 + x[0]['content'], 'role': 'user'}])
            )

        self.dataframe = self.dataframe[
            self.dataframe.apply(
                lambda doc: len(tokenizer.apply_chat_template(doc[prompt_key], add_generation_prompt=True))
                <= self.max_prompt_length,
                axis=1,
            )
        ]

        print(f'filter dataset len: {len(self.dataframe)}')

    def resume_dataset_state(self):
        self.serialize_dataset = False if hasattr(self, 'original_parquet_files') else True
        # resume dataframe if not it's serialized in data.pt
        if not self.serialize_dataset:
            self._download(use_origin_parquet=True)  # download and resume from original parquet files
            self._read_files_and_tokenize()
        else:
            print(r'old dataloader ckpt file is used, please train from scratch for better ckpt performance')

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, item):
        """
        Note that we also return the raw_input_ids so that it can be combined with other chat template
        """
        row_dict: dict = self.dataframe.iloc[item].to_dict()

        chat = row_dict.pop(self.prompt_key)

        prompt_with_chat_template = self.tokenizer.apply_chat_template(chat, add_generation_prompt=True, tokenize=False)

        is_multi_modal = self.image_key in row_dict
        if is_multi_modal:  # expand image token
            raw_prompt = prompt_with_chat_template.replace('<image>', '<|vision_start|><|image_pad|><|vision_end|>')
            # Convert base64 string to bytes if needed
            images = row_dict.pop(self.image_key)
            
            # Extract image paths for search cache lookup before processing
            image_paths = []
            for img in images:
                if isinstance(img, dict):
                    if 'path' in img:
                        image_paths.append(img['path'])
                    if 'bytes' in img and isinstance(img['bytes'], str):
                        # Decode base64 string to bytes
                        img['bytes'] = base64.b64decode(img['bytes'])
            
            # Store image paths for rollout search cache lookup
            # img_id can be extracted from path: path.split("/")[-1].split(".jpg")[0]
            if image_paths:
                row_dict['image_urls'] = image_paths  # Store paths for search cache
            
            # Process images and collect size information
            processed_images = []
            image_sizes = []  # List of (original_size, processed_size) tuples
            for image in images:
                processed_img, orig_size, proc_size = process_image(image)
                processed_images.append(processed_img)
                image_sizes.append({
                    'original_size': orig_size,  # (width, height)
                    'processed_size': proc_size  # (width, height)
                })
            
            row_dict['multi_modal_data'] = {'image': processed_images}
            # Store image size information for bbox scaling
            row_dict['image_size_info'] = image_sizes
            image_inputs = self.processor.image_processor(row_dict['multi_modal_data']['image'], return_tensors='pt')
            image_grid_thw = image_inputs['image_grid_thw']
            # NOTE: We don't need to store image_inputs in row_dict for memory saving
            # row_dict['multi_modal_inputs'] = {key: val for key, val in image_inputs.items()}

            if image_grid_thw is not None:
                merge_length = self.processor.image_processor.merge_size**2
                index = 0
                while '<image>' in prompt_with_chat_template:
                    prompt_with_chat_template = prompt_with_chat_template.replace(
                        '<image>',
                        '<|vision_start|>'
                        + '<|placeholder|>' * (image_grid_thw[index].prod() // merge_length)
                        + '<|vision_end|>',
                        1,
                    )
                    index += 1

                prompt_with_chat_template = prompt_with_chat_template.replace(
                    '<|placeholder|>', self.processor.image_token
                )
        else:
            raw_prompt = prompt_with_chat_template

        input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(
            prompt=prompt_with_chat_template,
            tokenizer=self.tokenizer,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )

        if is_multi_modal:
            from verl.models.transformers.qwen2_vl import get_rope_index

            position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids[0],
                image_grid_thw=image_grid_thw,
                attention_mask=attention_mask[0],
            )  # (3, seq_len)
        else:
            position_ids = compute_position_id_with_mask(attention_mask)

        row_dict['input_ids'] = input_ids[0]
        row_dict['attention_mask'] = attention_mask[0]
        row_dict['position_ids'] = position_ids[0]
        row_dict['raw_prompt_ids'] = self.tokenizer.encode(raw_prompt, add_special_tokens=False)

        # encode prompts without chat template
        if self.return_raw_chat:
            row_dict['raw_prompt'] = chat.tolist()

        # add index for each prompt
        index = row_dict.get("extra_info", {}).get("index", 0)
        row_dict["index"] = index

        return row_dict

    def __getstate__(self):
        if not self.serialize_dataset:
            state = self.__dict__.copy()

            if 'dataframe' in state:
                del state['dataframe']
            return state
        return self.__dict__.copy()
