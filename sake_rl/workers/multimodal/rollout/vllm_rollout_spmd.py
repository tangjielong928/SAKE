"""
The vllm_rollout that can be applied in different backend
When working with FSDP:
- Use DTensor weight loader (recommended) or HF weight loader
- Utilize state_dict from the FSDP to synchronize the weights among tp ranks in vLLM
When working with Megatron:
- Use Megatron weight loader
- During training, only the current pp stage holds the parameters
- Before inference, broadcast the parameters of the current pp rank to all other pp ranks (all pp ranks holds all the parameters)
- Bind the parameters to the inference engine
- Do inference in tp. pp is treated as additional dp
- After inference, all the parameters that doesn't belong to this pp rank is freed.
"""

import copy
import json
import os
import pickle
import re
import threading
from concurrent.futures import (  # parallelize search call
    ThreadPoolExecutor,
    as_completed,
)
from copy import deepcopy
from typing import Any, List, Union

import base64
import numpy as np
import torch
from io import BytesIO
from PIL import Image
from omegaconf import DictConfig
from tensordict import TensorDict
from tqdm import tqdm
from verl import DataProto
from verl.third_party.vllm import vllm_version
from verl.utils import hf_processor
from verl.utils.torch_functional import pad_sequence_to_length
from verl.workers.rollout.vllm_rollout.vllm_rollout_spmd import (
    _repeat_interleave,
    vLLMRollout,
)

from sake_rl.utils.tools.image_search import call_image_search
from sake_rl.utils.tools.text_search import call_text_search
from sake_rl.utils.torch_functional import get_final_eos_mask

# TODO
# 1. support pp in vllm
# 2. passing tokenizer is not necessary? no encoding/decoding is happending here
# 3. simplify init logics

def pad_to_max_stack(tensor_list: List[torch.Tensor], pad_token_id: int, dim: int) -> torch.Tensor:
    assert all([t.ndim == 1 for t in tensor_list])
    max_len = max([t.size(0) for t in tensor_list])
    padded_tensor_list = []
    for t in tensor_list:
        padded_tensor_list.append(
            torch.cat([t, torch.tensor([pad_token_id] * (max_len - t.size(0)), device=t.device, dtype=t.dtype)], dim=0)
        )
    return torch.stack(padded_tensor_list, dim=dim)


class vLLMRollout_MultiTurn_SAKE_RL(vLLMRollout):

    def __init__(self, model_path: str, config: DictConfig, tokenizer, model_hf_config, **kwargs):

        super().__init__(model_path, config, tokenizer, model_hf_config, **kwargs)

        # add tokenizer
        self.tokenizer = tokenizer
        # add processor
        self.processor = hf_processor(model_path)


        self.user_prompt_after_image_search = None
        self.user_prompt_after_text_search = None
        try:
            with open(config.search.user_prompt_after_image_search, 'rb') as file:
                self.user_prompt_after_image_search = pickle.load(file)
        except Exception as e:
            print(f"Error: {e} | user_prompt_after_image_search default to None")
        try:
            with open(config.search.user_prompt_after_text_search, 'rb') as file:
                self.user_prompt_after_text_search = pickle.load(file)
        except Exception as e:
            print(f"Error: {e} | user_prompt_after_text_search default to None")

        print(f"[Prompt Set] user_prompt_after_text_search: {self.user_prompt_after_text_search}")
        print(f"[Prompt Set] user_prompt_after_image_search: {self.user_prompt_after_image_search}")
        
        # Cache path for offline search results
        self.search_cache_path = getattr(config.search, 'search_cache_path', None)
        print(f"[Search Cache] search_cache_path: {self.search_cache_path}")

    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:

        print(f">>> vllm_rollout_spmd Rollout Starts ...")

        # rebuild vllm cache engine
        if vllm_version in ('0.3.1', '0.4.2', '0.5.4', '0.6.3') and self.config.free_cache_engine:
            self.inference_engine.init_cache_engine()

        # All for 1st USER prompt
        idx = prompts.batch['input_ids']  # (B'*R, max_prompt_length), left padding with |end_of_text|
        batch_size = idx.size(0)  # B'
        # for logit_log_prob & loss computation
        attention_mask = prompts.batch['attention_mask']  # (B'*R, max_prompt_length), left padding 0
        position_ids = prompts.batch['position_ids']  # (B'*R, max_prompt_length), left padding 0
        # used to construct attention_mask
        eos_token_id = prompts.meta_info['eos_token_id']  # [151645, 151643] -> ｜im_end｜, |end_of_text|
        input_prompt_generation_mask = torch.zeros_like(
            idx, dtype=attention_mask.dtype, device=attention_mask.device
        )  # (B'*R, max_prompt_length), all 0

        non_tensor_batch = prompts.non_tensor_batch
        if 'raw_prompt_ids' not in non_tensor_batch:
            non_tensor_batch['raw_prompt_ids'] = np.array(
                [_pre_process_inputs(self.pad_token_id, idx[i]) for i in range(batch_size)], dtype=object
            )

        if batch_size != len(non_tensor_batch['raw_prompt_ids']):
            raise RuntimeError('vllm sharding manager is not work properly.')

        do_sample = prompts.meta_info.get('do_sample', True)
        if not do_sample:
            kwargs = {
                'best_of': 1,
                'top_p': 1.0,
                'top_k': -1,
                'min_p': 0.0,
                'temperature': 0,
                'n': 1,  # if greedy, only 1 response
            }

        n = 1 if prompts.meta_info.get('validate', False) else self.config.n  # TODO: for validate, do_sample=False

        ##### Initialization #####
        vllm_inputs = (
            []
        )  # B*R, list of dict, into -> vllm.engine, each dict with keys: 'prompt_token_ids', 'multi_modal_data', the values are 'raw_prompt_ids' and [PIL.Image]
        multi_turn_response_mask = []  # B*R, list of list of Tensor, for distinguish 'USER tokens' & 'ASSISTANT tokens'
        prefix_prompt_lengths = []  # B*R, list of int, record first round prompt of all trajs
        search_tool_return_images = []

        # We manually repeart trajs for rollout, since some trajs need multi-round self.inference_engine.generate() with `sampling_n=1`
        if 'multi_modal_data' in non_tensor_batch:
            _multi_modal_data_list = non_tensor_batch['multi_modal_data']
            for raw_prompt_ids, multi_modal_data in zip(non_tensor_batch.pop('raw_prompt_ids'), _multi_modal_data_list):
                prefix_length = len(raw_prompt_ids)
                for _ in range(n):
                    # NOTE: use deepcopy to seperate variables
                    vllm_inputs.append(
                        {
                            'prompt_token_ids': deepcopy(raw_prompt_ids),
                            'multi_modal_data': deepcopy(multi_modal_data),
                        }  # raw_prompt_ids: list
                    )
                    multi_turn_response_mask.append(
                        [
                            torch.zeros(prefix_length, dtype=attention_mask.dtype, device=attention_mask.device)
                        ]  # USER, Mark as 0
                    )  # [torch.Tensor(prefix_length,)]
                    prefix_prompt_lengths.append(prefix_length)
                    search_tool_return_images.append([])  # init as empty lists

        # We need 'image_urls' for search, the shape should be aligned with B*R
        if 'image_urls' in non_tensor_batch.keys() and not prompts.meta_info.get('validate', False):
            non_tensor_batch['image_urls'] = _repeat_interleave(non_tensor_batch['image_urls'], self.config.n)
        
        # Extract img_ids for cached search lookup
        # img_id comes from sample["images"][0]["path"].split("/")[-1].split(".jpg")[0]
        img_ids = []
        if 'image_urls' in non_tensor_batch.keys():
            for img_url_item in non_tensor_batch['image_urls']:
                img_id = None
                if img_url_item:
                    # Handle case where img_url_item is a list of paths
                    if isinstance(img_url_item, (list, tuple)) and len(img_url_item) > 0:
                        img_url = img_url_item[0]  # Take the first path
                    else:
                        img_url = img_url_item
                    
                    if img_url:
                        # Extract img_id from path (e.g., "/path/to/123456.jpg" -> "123456")
                        img_id = str(img_url).split("/")[-1].split(".jpg")[0].split(".png")[0].split(".jpeg")[0]
                img_ids.append(img_id)
        else:
            # Fallback: try to get from multi_modal_data paths if available
            if 'multi_modal_data' in non_tensor_batch.keys():
                for mm_data in non_tensor_batch['multi_modal_data']:
                    img_id = None
                    if isinstance(mm_data, dict) and 'image' in mm_data:
                        images = mm_data['image']
                        if images and len(images) > 0:
                            # Try to get path from the first image
                            first_img = images[0]
                            if hasattr(first_img, 'filename') and first_img.filename:
                                img_id = str(first_img.filename).split("/")[-1].split(".jpg")[0].split(".png")[0]
                    for _ in range(n):
                        img_ids.append(img_id)
            else:
                img_ids = [None] * (batch_size * n)
        
        # Ensure img_ids length matches batch_size * n
        if len(img_ids) < batch_size * n:
            img_ids = img_ids + [None] * (batch_size * n - len(img_ids))
        elif len(img_ids) > batch_size * n:
            img_ids = img_ids[:batch_size * n]
        
        print(f"[Search Cache] Extracted {len([x for x in img_ids if x])} img_ids for cache lookup")

        ##### Loop Setting #####
        to_generate = list(range(batch_size * n))  # B*R, all trajs' index
        worker_trajs_count = len(to_generate)
        id_image_gen_cnt = [0] * (batch_size * n)
        max_image_gen_round = self.config.search.image_search_limit # Image Search Constraint
        id_text_gen_cnt = [0] * (batch_size * n)
        max_text_gen_round = self.config.search.text_search_limit # Text Search Constraint
        
        # Save original images for each trajectory (to be reused in subsequent rounds)
        original_images_per_traj = []  # B*R, list of list of PIL.Image
        for i in range(batch_size * n):
            if 'multi_modal_data' in vllm_inputs[i] and 'image' in vllm_inputs[i]['multi_modal_data']:
                original_images_per_traj.append(deepcopy(vllm_inputs[i]['multi_modal_data']['image']))
            else:
                original_images_per_traj.append([])
        # Add pbar for better monitoring
        with tqdm(total=worker_trajs_count, desc="Worker Rollout Progress", unit="task") as pbar:
            current_iteration = 0
            max_iterations = self.config.max_gen_round
            while current_iteration < max_iterations and len(to_generate) > 0:
                # Prepare prompts to generation
                idx_to_gen = []  # list of vllm_inputs, at first the length is B'*R
                for i in to_generate:
                    idx_to_gen.append(vllm_inputs[i])

                print(
                    f"[Round #{current_iteration} Rollout START] For THIS round, We hava {len(idx_to_gen)} trajs to complete ..."
                )

                # users can customize different sampling_params at different run
                with self.update_sampling_params(n=1):  # TODO: for validate, do_sample=False
                    outputs = self.inference_engine.generate(
                        prompts=idx_to_gen, sampling_params=self.sampling_params, use_tqdm=False  # list of dict
                    )

                response = []  # list of tuple, B'*R, valid(no-pad) response_ids with unequal length
                for output in outputs:
                    for sample_id in range(len(output.outputs)):
                        # HACK: filter > (voc_size+specidal_token_num) token_ids, 151664 for qwen model
                        _token_ids = output.outputs[sample_id].token_ids
                        filtered_token_ids = [token_id for token_id in _token_ids if token_id <= 151664]
                        if 151645 not in filtered_token_ids:
                            # replace the last token with <|im_end|> if no <|im_end|> in response,
                            # this is to ensure successful execution of get_final_eos_mask in multi-turn scenario
                            filtered_token_ids[-1] = 151645

                        response.append(filtered_token_ids)

                # attach model responses to vllm_inputs
                assert len(to_generate) == len(response)

                idx_to_remove = []
                id_search_query_mapping = {}
                for i_gen, response_ in zip(to_generate, response):
                    # update conversation
                    response_ = list(response_)
                    vllm_inputs[i_gen]['prompt_token_ids'] += response_
                    multi_turn_response_mask[i_gen].append(
                        torch.ones(len(response_), dtype=attention_mask.dtype, device=attention_mask.device)
                    )  # ASSISTANT, Mark as 1

                    # [SEARCH TRIGGER] We check model's last turn response, if not any <xxx_search>, then remove this traj from to_generate
                    decoded_resp_ = self.tokenizer.decode(response_, skip_special_tokens=True)
                    
                    # Check for <answer> tag - if found, this trajectory is complete
                    if re.search(r'<answer>.*</answer>', decoded_resp_, re.DOTALL):
                        idx_to_remove.append(i_gen)
                        continue
                    
                    # Need to call text search - new format: <text_search>[{"query": "..."}, ...]</text_search>
                    # Priority: text_search > image_search (as per prompt instructions)
                    text_search_match = re.search(r'<text_search>(.*?)</text_search>', decoded_resp_, re.DOTALL)
                    if text_search_match:
                        assert str(i_gen) not in id_search_query_mapping.keys()
                        if (
                            id_text_gen_cnt[i_gen] >= max_text_gen_round or current_iteration == max_iterations - 1
                        ):
                            idx_to_remove.append(i_gen)
                            print(f"{i_gen} has reached max_text_gen_round {max_text_gen_round}")
                            continue
                        try:
                            queries_json_str = text_search_match.group(1).strip()
                            queries = json.loads(queries_json_str)
                            
                            # Normalize queries: handle cases where queries might be strings instead of dicts
                            if not isinstance(queries, list):
                                queries = [queries]
                            
                            normalized_queries = []
                            for query_item in queries:
                                if isinstance(query_item, str):
                                    normalized_queries.append({"query": query_item})
                                elif isinstance(query_item, dict):
                                    if "query" not in query_item:
                                        # If dict but no "query" key, try to extract a string value
                                        query_str = ""
                                        for key, value in query_item.items():
                                            if isinstance(value, str) and value:
                                                query_str = value
                                                break
                                        if not query_str:
                                            query_str = str(query_item)
                                        normalized_queries.append({"query": query_str})
                                    else:
                                        normalized_queries.append(query_item)
                                else:
                                    normalized_queries.append({"query": str(query_item)})
                            
                            queries = normalized_queries
                            
                            if isinstance(queries, list) and len(queries) > 0:
                                # Get original text from context if available
                                all_context = self.tokenizer.decode(
                                    vllm_inputs[i_gen]['prompt_token_ids'], skip_special_tokens=True
                                )
                                # Try to extract original text (adjust pattern as needed)
                                original_text = None
                                if "Here are the image and original social media post for your task:" in all_context:
                                    try:
                                        original_text = all_context.split("Here are the image and original social media post for your task:\n")[1].split("assistant")[0].strip()
                                    except:
                                        pass
                                
                                # Add original_text to each query if not present
                                for query_obj in queries:
                                    if isinstance(query_obj, dict) and "original_text" not in query_obj and original_text:
                                        query_obj["original_text"] = original_text
                                
                                id_search_query_mapping[str(i_gen)] = {"type": "text", "content": queries, "original_text": original_text}
                                id_text_gen_cnt[i_gen] += 1
                            else:
                                print(f"[Round #{current_iteration} Rollout ERROR] Invalid text_search queries for traj {i_gen}, will be removed!!!")
                                idx_to_remove.append(i_gen)
                        except json.JSONDecodeError as e:
                            print(f"[Round #{current_iteration} Rollout ERROR] Failed to parse text_search JSON for traj {i_gen}: {e}, will be removed!!!")
                            idx_to_remove.append(i_gen)
                    
                    # Need to call image search - new format: <image_search>[{"query": "..."}, ...]</image_search>
                    elif re.search(r'<image_search>.*</image_search>', decoded_resp_, re.DOTALL):
                        image_search_match = re.search(r'<image_search>(.*?)</image_search>', decoded_resp_, re.DOTALL)
                        if image_search_match:
                            assert str(i_gen) not in id_search_query_mapping.keys()
                            if (
                                id_image_gen_cnt[i_gen] >= max_image_gen_round or current_iteration == max_iterations - 1
                            ):
                                idx_to_remove.append(i_gen)
                                print(f"{i_gen} has reached max_image_gen_round {max_image_gen_round}")
                                continue
                            try:
                                # Parse JSON from image_search tag
                                queries_json_str = image_search_match.group(1).strip()
                                queries = json.loads(queries_json_str)
                                
                                # Normalize queries: handle cases where queries might be strings instead of dicts
                                if not isinstance(queries, list):
                                    queries = [queries]
                                
                                normalized_queries = []
                                for query_item in queries:
                                    if isinstance(query_item, str):
                                        normalized_queries.append({"query": query_item})
                                    elif isinstance(query_item, dict):
                                        if "query" not in query_item:
                                            # If dict but no "query" key, try to extract a string value
                                            query_str = ""
                                            for key, value in query_item.items():
                                                if isinstance(value, str) and value:
                                                    query_str = value
                                                    break
                                            if not query_str:
                                                query_str = str(query_item)
                                            normalized_queries.append({"query": query_str})
                                        else:
                                            normalized_queries.append(query_item)
                                    else:
                                        normalized_queries.append({"query": str(query_item)})
                                
                                queries = normalized_queries
                                
                                if isinstance(queries, list) and len(queries) > 0:
                                    id_search_query_mapping[str(i_gen)] = {"type": "image", "content": queries}
                                    id_image_gen_cnt[i_gen] += 1
                                else:
                                    print(f"[Round #{current_iteration} Rollout ERROR] Invalid image_search queries for traj {i_gen}, will be removed!!!")
                                    idx_to_remove.append(i_gen)
                            except json.JSONDecodeError as e:
                                print(f"[Round #{current_iteration} Rollout ERROR] Failed to parse image_search JSON for traj {i_gen}: {e}, will be removed!!!")
                                idx_to_remove.append(i_gen)
                    
                    # Direct Answer or no search tag found
                    else:
                        # remove this traj from to_generate
                        idx_to_remove.append(i_gen)
                        # NOTE: to_generate.remove(i_gen) # DO NOT .remove() in for loop

                print(
                    f"[Round #{current_iteration} Rollout Search Trigger] For THIS round, we need to conduct search for: {id_search_query_mapping} ..."
                )

                # update 'to_generate'
                for x in idx_to_remove:
                    to_generate.remove(x)

                print(
                    f"[Round #{current_iteration} Rollout END] For THIS round, We hava completed {len(idx_to_remove)} trajs ..."
                )
                print(
                    f"[Round #{current_iteration} Rollout END] For NEXT round, We hava {len(to_generate)} trajs to complete ..."
                )

                # [Call Search Tool] Conduct Search as-needed
                search_result = []

                if not self.config.search.parallel_tool_call:
                    ########################################## sequential implementation #############################################
                    for i_todo in tqdm(to_generate, desc=f"[Round #{current_iteration} Searching Progress]"):
                        tool_returned_images = []
                        assert str(i_todo) in id_search_query_mapping.keys()
                        _type = id_search_query_mapping[str(i_todo)]["type"]
                        _content = id_search_query_mapping[str(i_todo)]["content"]  # Now it's a list of query objects
                        _original_text = id_search_query_mapping[str(i_todo)].get("original_text", None)
                        _img_id = img_ids[i_todo] if i_todo < len(img_ids) else None
                        # print(f"[Round #{current_iteration} Search START] Call search tool | Type: {_type} | Content: {_content} | img_id: {_img_id} ...")
                        if _type == "text":
                            tool_returned_str, tool_stat = call_text_search(
                                queries=_content,
                                img_id=_img_id,
                                cache_path=self.search_cache_path,
                                original_text=_original_text,
                            )
                        elif _type == "image":
                            tool_returned_str, tool_returned_images, tool_stat = call_image_search(
                                queries=_content,
                                img_id=_img_id,
                                cache_path=self.search_cache_path,
                            )
                        else:
                            raise ValueError(f"[Round #{current_iteration} Search ERROR] Unknown Search Type: {_type}")
                        # print(f"[Round #{current_iteration} Search END] Search tool return:\n {tool_returned_str} ...")
                        search_result.append((tool_returned_str, tool_returned_images, tool_stat))
                    ########################################## sequential implementation #############################################
                else:
                    ############################################## parallel implementation #############################################
                    def tool_helper(i_todo):
                        tool_returned_images = []
                        assert str(i_todo) in id_search_query_mapping.keys()
                        _type = id_search_query_mapping[str(i_todo)]["type"]
                        _content = id_search_query_mapping[str(i_todo)]["content"]  # Now it's a list of query objects
                        _original_text = id_search_query_mapping[str(i_todo)].get("original_text", None)
                        _img_id = img_ids[i_todo] if i_todo < len(img_ids) else None
                        thread_id = threading.current_thread().ident
                        print(
                            f"[Round #{current_iteration} Search START][Thread{thread_id}] Call search tool | Type: {_type} | Content: {_content} | img_id: {_img_id} ..."
                        )
                        if _type == "text":
                            tool_returned_str, tool_stat = call_text_search(
                                queries=_content,
                                img_id=_img_id,
                                cache_path=self.search_cache_path,
                                original_text=_original_text,
                            )
                        elif _type == "image":
                            tool_returned_str, tool_returned_images, tool_stat = call_image_search(
                                queries=_content,
                                img_id=_img_id,
                                cache_path=self.search_cache_path,
                            )
                        else:
                            raise ValueError(
                                f"[Round #{current_iteration} Search ERROR][Thread{thread_id}] Unknown Search Type: {_type}"
                            )
                        print(
                            f"[Round #{current_iteration} Search END][Thread{thread_id}] Search tool return:\n {tool_returned_str} ..."
                        )
                        return (tool_returned_str, tool_returned_images, tool_stat)

                    search_call_futures = []
                    with ThreadPoolExecutor(self.config.search.parallel_tool_call_threads) as pool:
                        for i_todo in to_generate:
                            assert str(i_todo) in id_search_query_mapping.keys()
                            search_call_futures.append(pool.submit(tool_helper, i_todo))
                        for _ in tqdm(
                            as_completed(search_call_futures),
                            desc=f"[MT][Round #{current_iteration} Searching Progress]",
                        ):
                            pass
                    search_result = [f.result() for f in search_call_futures]
                    ############################################## parallel implementation #############################################

                # [Process Search Results]
                to_generate_ = to_generate.copy()  # make a copy since we will be modifying to_generate
                assert len(to_generate_) == len(
                    search_result
                ), f"Current Itr: {current_iteration} | len(to_generate_): {len(to_generate_)} | len(search_result): {len(search_result)}"
                for i_gen_, search_result_ in zip(to_generate_, search_result):

                    search_result_txt, search_result_img, tool_stat = search_result_
                    search_type = id_search_query_mapping[str(i_gen_)]["type"]

                    # Parse JSON results and format for model
                    try:
                        search_results_json = json.loads(search_result_txt)
                        formatted_result = ""
                        
                        if search_type == "text":
                            # Format text search results
                            formatted_result = "[Text Search Results] Below are the search results for each query:\n"
                            for result in search_results_json:
                                query = result.get("query", "")
                                status = result.get("status", "unknown")
                                if status == "success":
                                    summary = result.get("result", {}).get("summary", "")
                                    formatted_result += f"\nQuery: {query}\nSummary: {summary}\n"
                                else:
                                    error_msg = result.get("result", {}).get("summary", "No information found.")
                                    formatted_result += f"\nQuery: {query}\nSummary: {error_msg}\n"
                            search_result_txt = formatted_result
                        
                        elif search_type == "image":
                            # Format image search results - images already loaded in call_image_search
                            # search_result_img already contains PIL Images, just format the text
                            formatted_result = "[Image Search Results] Below are the visual references for each query:\n"
                            total_images_expected = 0
                            
                            for result in search_results_json:
                                query = result.get("query", "")
                                status = result.get("status", "unknown")
                                kg_image_path = result.get("kg_image_path", None)
                                
                                if status == "success":
                                    images = result.get("result", {}).get("images", [])
                                    # kg_image_path如果存在会在tool_returned_images的第一位
                                    num_images = len(images)
                                    if kg_image_path and os.path.basename(kg_image_path):
                                        num_images += 1  # kg图片会额外添加
                                    
                                    total_images_expected += num_images
                                    formatted_result += f"\nQuery: {query}\nFound {num_images} reference image(s):\n"
                                    
                                    # 格式化输出每张图片
                                    for index in range(num_images):
                                        formatted_result += f"  Image {index+1}: <|vision_start|><|image_pad|><|vision_end|>\n"
                                    
                                    if num_images == 0:
                                        formatted_result += "  No images found.\n"
                                else:
                                    formatted_result += f"\nQuery: {query}\nNo images found.\n"
                            
                            # 验证图像数量一致性
                            assert len(search_result_img) == total_images_expected, (
                                f"[Round #{current_iteration} Image Count Warning] Expected {total_images_expected} images but got {len(search_result_img)} from tool"
                            )
                            
                            search_result_txt = formatted_result
                    
                    except json.JSONDecodeError as e:
                        print(f"[Round #{current_iteration} Search Result Parse ERROR] Failed to parse JSON result: {e}")
                        # Fallback to original text
                        if search_type == "text":
                            search_result_txt = "[Text Search Results] " + search_result_txt
                        elif search_type == "image":
                            search_result_txt = "[Image Search Results] " + search_result_txt

                    # init search_result_message
                    search_result_message = search_result_txt
                    # print(f"search_result_message: {search_result_message}")

                    # Get original images for this trajectory
                    orig_images = original_images_per_traj[i_gen_]
                    num_orig_images = len(orig_images)
                    
                    # Build original image placeholders (to be inserted before org_query)
                    orig_image_placeholders = ""
                    for _ in range(num_orig_images):
                        orig_image_placeholders += "<|vision_start|><|image_pad|><|vision_end|>\n"

                    # Construct Next Round Prompt
                    # Use after_image_search_prompt and after_text_search_prompt to differentiate the two cases
                    if search_type == "text":
                        # Text Search Performed
                        if self.user_prompt_after_text_search is not None:
                            all_context = self.tokenizer.decode(
                                vllm_inputs[i_gen_]['prompt_token_ids'], skip_special_tokens=True
                            )
                            # Try to extract original text
                            org_query = None
                            try:
                                if "Here are the image and original social media post for your task:" in all_context:
                                    org_query = (
                                        all_context.split("Here are the image and original social media post for your task:\n")[1]
                                        .split("assistant")[0]
                                        .strip()
                                    )
                            except:
                                pass

                            if org_query:
                                search_result_message = (
                                    "Searched results: <information>"
                                    + search_result_txt
                                    + "</information>\n"
                                    + f"Here are the image and original social media post for your task:\n{orig_image_placeholders}{org_query}\n"
                                    + self.user_prompt_after_text_search
                                )
                            else:
                                search_result_message = (
                                    "Searched results: <information>"
                                    + search_result_txt
                                    + "</information>\n"
                                    + self.user_prompt_after_text_search
                                )
                    
                    elif search_type == "image":
                        # Image Search Performed
                        if self.user_prompt_after_image_search is not None:
                            all_context = self.tokenizer.decode(
                                vllm_inputs[i_gen_]['prompt_token_ids'], skip_special_tokens=True
                            )
                            # Try to extract original text
                            org_query = None
                            try:
                                if "Here are the image and original social media post for your task:" in all_context:
                                    org_query = (
                                        all_context.split("Here are the image and original social media post for your task:\n")[1]
                                        .split("assistant")[0]
                                        .strip()
                                    )
                            except:
                                pass
                            if org_query:
                                search_result_message = (
                                    "Searched results: <information>"
                                    + search_result_txt
                                    + "</information>\n"
                                    + f"Here are the image and original social media post for your task:\n{orig_image_placeholders}{org_query}\n"
                                    + self.user_prompt_after_image_search
                                )
                            else:
                                search_result_message = (
                                    "Searched results: <information>"
                                    + search_result_txt
                                    + "</information>\n"
                                    + self.user_prompt_after_image_search
                                )

                    search_result_message = (
                        "<|im_start|>user\n" + search_result_message + "<|im_end|>\n<|im_start|>assistant\n"
                    )
                    next_turn_prompt_ids = self.tokenizer.encode(search_result_message)

                    # update conversation
                    vllm_inputs[i_gen_][
                        'prompt_token_ids'
                    ] += next_turn_prompt_ids  # this might go over response length, but we will cut it later by 'max_response_length_total'
                    
                    # Add images to vllm_inputs in order: search result images first, then original images
                    # This order matches the placeholders in search_result_message:
                    #   1. Search result images (inside <information>...</information>)
                    #   2. Original images (before org_query)
                    if orig_images and num_orig_images > 0:
                        search_result_img += orig_images
                        vllm_inputs[i_gen_]['multi_modal_data']['image'] += search_result_img
                        search_tool_return_images[i_gen_] += search_result_img
                        print(f"Added {len(search_result_img)} image(s) to trajectory {i_gen_} (search: {len(search_result_img) - num_orig_images}, original: {num_orig_images})")
                    
                    multi_turn_response_mask[i_gen_].append(
                        torch.zeros(len(next_turn_prompt_ids), dtype=attention_mask.dtype, device=attention_mask.device)
                    )  # USER, Mark as 0

                # update pbar
                pbar.update(worker_trajs_count - len(to_generate))

                # update iteration count
                current_iteration += 1

        # re-build response
        response = []  # B'*R, torch.Tensors with unequal lengths
        response_generation_mask = []  # B'*R, torch.Tensors with unequal lengths but align with 'response'
        # process search tool returned images
        for i_ in range(batch_size * n):  # 0~15, 4*4
            # for each traj, we skip first-round prompt_ids/attention_mask
            all_response_masks = torch.cat(multi_turn_response_mask[i_][1:], dim=0)
            resp_mask_device = all_response_masks.device

            first_round_prompt_length = prefix_prompt_lengths[i_]
            response_after_prompt = vllm_inputs[i_]['prompt_token_ids'][first_round_prompt_length:]

            # NOTE: [For Multi-Image] Update response_after_prompt(list of token_ids) and all_response_masks if search tool returned images
            if search_tool_return_images[i_]:
                # process PIL.Images to get 'pixel_values' and 'image_grid_thw'
                searched_image_inputs = self.processor.image_processor(
                    search_tool_return_images[i_], return_tensors='pt'
                )  # dict_keys(['pixel_values', 'image_grid_thw'])
                searched_image_grid_thw = searched_image_inputs['image_grid_thw']
                # print(f"searched_image_grid_thw shape: {searched_image_grid_thw.shape}")
                # print(f"searched_image_grid_thw: {searched_image_grid_thw}")
                if searched_image_grid_thw is not None:
                    merge_length = self.processor.image_processor.merge_size**2
                    index, image_pad_token, magic_num = 0, 151655, 654321
                    all_response_masks = all_response_masks.tolist()  # for convenient modification
                    while image_pad_token in response_after_prompt:
                        # find pos of <|image_pad|>
                        pos = response_after_prompt.index(image_pad_token)
                        replicate_count = searched_image_grid_thw[index].prod() // merge_length
                        # update response_after_prompt
                        response_after_prompt[pos : pos + 1] = [magic_num] * replicate_count
                        # update all_response_masks
                        all_response_masks[pos : pos + 1] = [0] * replicate_count
                        index += 1
                    response_after_prompt = [image_pad_token if x == magic_num else x for x in response_after_prompt]
                    all_response_masks = torch.tensor(all_response_masks, dtype=torch.int64, device=resp_mask_device)

            response_generation_mask.append(all_response_masks)  # at least we have single-turn conversation
            all_response = torch.tensor(response_after_prompt, device=idx.device, dtype=idx.dtype)
            response.append(all_response)
            assert (
                response[i_].shape[0] == response_generation_mask[i_].shape[0]
            ), f"shape mismatched | response[i_]: {response[i_].shape[0]} | response_generation_mask[i_]: {response_generation_mask[i_].shape[0]}"
        assert len(response) == len(
            response_generation_mask
        ), "length mismatched between response and response_generation_mask!"

        # attention_mask:       prompt           response
        #                 [0,0,0,0,1,1,1,1, | 1,1,1,0,0,0,0,0]
        response = pad_to_max_stack(
            response, self.pad_token_id, dim=0
        )  # Tensor, (B'*R, padded_length), padded_length is the max length of samples in list
        response_generation_mask = pad_to_max_stack(response_generation_mask, 0, dim=0)  # Tensor, (B'*R, padded_length)
        assert all([response.size(dim) == response_generation_mask.size(dim) for dim in range(response.ndim)])

        # cut or pad to max length
        # all should be (B*R, self.config.response_length)
        if response.shape[1] > self.config.response_length_total:
            response = response[:, : self.config.response_length_total]
            response_generation_mask = response_generation_mask[:, : self.config.response_length_total]
        elif response.shape[1] < self.config.response_length_total:
            response = pad_sequence_to_length(response, self.config.response_length_total, self.pad_token_id)
            response_generation_mask = pad_sequence_to_length(
                response_generation_mask, self.config.response_length_total, 0
            )

        # All for 1st USER prompt
        if self.config.n > 1 and do_sample:
            idx = _repeat_interleave(idx, self.config.n)  # (B, max_prompt_length) -> (B*R, max_prompt_length)
            attention_mask = _repeat_interleave(attention_mask, self.config.n)
            position_ids = _repeat_interleave(position_ids, self.config.n)
            batch_size = batch_size * self.config.n
            # NOTE: We repeat 'multi_modal_data'
            if 'multi_modal_data' in non_tensor_batch.keys():
                repeated = []
                _index_br = 0
                for item in non_tensor_batch['multi_modal_data']:
                    for _ in range(self.config.n):
                        new_item = copy.deepcopy(item)
                        if search_tool_return_images[_index_br]:
                            new_item['image'] += search_tool_return_images[_index_br]
                        repeated.append(new_item)
                        _index_br += 1
                non_tensor_batch['multi_modal_data'] = np.array(repeated)
            # we also need to repeat 'input_prompt_generation_mask'
            input_prompt_generation_mask = _repeat_interleave(
                input_prompt_generation_mask, self.config.n
            )  # (B, max_prompt_length) -> (B*R, max_prompt_length), all 0

        seq = torch.cat([idx, response], dim=-1)  # (B*R, max_prompt_length+max_response_length_total)

        response_length = response.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).expand(batch_size, -1)
        if position_ids.dim() == 3:  # qwen2vl mrope
            delta_position_id = delta_position_id.view(batch_size, 1, -1).expand(batch_size, 3, -1)

        # TODO(sgm): fix position_ids on right_pad
        # prompt: left pad + response: right pad
        # attention_mask: [0,0,0,0,1,1,1,1, | 1,1,1,0,0,0,0,0]
        # position_ids:   [0,0,0,0,0,1,2,3, | 4,5,6,7,8,9,10,11]
        response_position_ids = position_ids[:, -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)

        response_attention_mask = get_final_eos_mask(
            response_id=response, eos_token=[151645], dtype=attention_mask.dtype
        )  # HACK: for qwen, |im_end| is 151645
        # attention_mask: (...,0,0,0,1,1,1), response_attention_mask: (1,1,1,0,0,0,...)
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)
        multi_turn_response_mask = torch.cat([input_prompt_generation_mask, response_generation_mask], dim=-1)

        # all the tp ranks should contain the same data here. data in all ranks are valid
        # NOTE: .contiguous() for broadcast
        batch = TensorDict(
            {
                'prompts': idx.contiguous(),
                'responses': response.contiguous(),
                'input_ids': seq.contiguous(),  # here input_ids become the whole sentences
                # 'old_log_probs': log_probs, # we will recompute old log prob with actor
                'attention_mask': attention_mask.contiguous(),
                'position_ids': position_ids.contiguous(),
                'multi_turn_response_mask': multi_turn_response_mask.contiguous(),
            },
            batch_size=batch_size,
        )

        # free vllm cache engine
        if vllm_version in ('0.3.1', '0.4.2', '0.5.4', '0.6.3') and self.config.free_cache_engine:
            self.inference_engine.free_cache_engine()

        print(f">>> vllm_rollout_spmd Rollout Ends ...")
        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)
