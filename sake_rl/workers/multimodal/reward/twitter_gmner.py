import json

import numpy as np
import torch
from verl import DataProto

from sake_rl.utils.reward_score_mm import _default_compute_score


class TwitterGMNER_RewardManager:
    """The reward manager for Twitter-GMNER task."""

    def __init__(self, tokenizer, num_examine, compute_score=None) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or _default_compute_score

    def extract_responses_list(
        self,
        tokenizer,
        input_ids: torch.Tensor,  # User Prompt + All Responses
        multi_turn_response_mask: torch.Tensor,  # 0,0,0,...,1,1,1,...,0,0,0,...,1,1,1
    ) -> list:
        diff = torch.diff(multi_turn_response_mask, prepend=torch.tensor([0], device=multi_turn_response_mask.device))
        starts = torch.where(diff == 1)[0]
        mask_appended = torch.cat(
            [multi_turn_response_mask, torch.tensor([0], device=multi_turn_response_mask.device)], dim=0
        )
        diff_end = torch.diff(mask_appended)
        ends = torch.where(diff_end == -1)[0] - 1
        segments = []
        for s, e in zip(starts, ends):
            segments.append(input_ids[s : e + 1].tolist())

        # Decode each segment
        decoded_responses = tokenizer.batch_decode(segments, skip_special_tokens=True)
        return decoded_responses

    def __call__(self, data: DataProto):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        # shape: (B*R, response_length_total)
        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)

        already_print_data_sources = {}

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            # Get valid prompt_ids w/o padding tokens
            prompt_ids = data_item.batch['prompts']
            prompt_length = prompt_ids.shape[-1]
            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            # Get valid response_ids w/o padding tokens
            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            response_str = [response_str]
            # For multi turn, we maybe need `response_str` in a list format
            if 'multi_turn_response_mask' in data_item.batch:
                # `response_str` is a list now
                response_str = self.extract_responses_list(
                    self.tokenizer, data_item.batch['input_ids'], data_item.batch['multi_turn_response_mask']
                )

            # We need `ground_truth` to be a list to support multiple candidate answers
            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']
            if isinstance(ground_truth, str):
                ground_truth = [ground_truth]
            if 'candidate_answers' in data_item.non_tensor_batch['reward_model']:
                candidate_answers = data_item.non_tensor_batch['reward_model']['candidate_answers']
                if isinstance(candidate_answers, list):
                    ground_truth += candidate_answers
                elif isinstance(candidate_answers, str):
                    ground_truth += json.loads(candidate_answers)
                else:
                    raise TypeError(f"candidate_answers must be a list or a string, but got {type(candidate_answers)}")
            ground_truth = [g for g in ground_truth if isinstance(g, str)]
            data_source = data_item.non_tensor_batch['data_source']

            # Get extra_info from data_item, handling both dict and numpy array cases
            extra_info_raw = data_item.non_tensor_batch.get('extra_info', None)
            if extra_info_raw is None:
                extra_info = {}
            elif isinstance(extra_info_raw, dict):
                extra_info = extra_info_raw.copy()
            else:
                # Handle numpy array case: convert to dict if it's a numpy object
                if hasattr(extra_info_raw, 'item'):  # numpy scalar
                    extra_info = extra_info_raw.item()
                    if not isinstance(extra_info, dict):
                        extra_info = {}
                else:
                    extra_info = {}
            
            # Add image size information for bbox scaling
            if 'image_size_info' in data_item.non_tensor_batch:
                extra_info['image_size_info'] = data_item.non_tensor_batch['image_size_info']

            score = self.compute_score(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
            )
            reward_tensor[i, valid_response_length - 1] = score

            # Store updated extra_info back to data for F1 metrics extraction
            # This ensures that F1 metrics computed in compute_score are available in compute_data_metrics
            # Note: non_tensor_batch values must be numpy arrays for verl protocol compatibility
            if 'extra_info' not in data.non_tensor_batch:
                # Initialize as numpy array of objects (dicts)
                data.non_tensor_batch['extra_info'] = np.array([{} for _ in range(len(data))], dtype=object)
            
            # Update extra_info at index i
            # Directly modify the numpy array element (verl requires numpy arrays)
            extra_info_array = data.non_tensor_batch['extra_info']
            extra_info_array[i] = extra_info

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[ground_truth]", ground_truth)
                print("[score]", score)
                
        return reward_tensor

