#!/bin/bash

model=Qwen2.5-VL-7B-Instruct
B="7B"
datasets=("train" "dev" "test")
tasks=("gmner" "fmnerg")

for task in "${tasks[@]}"; do
    for ds in "${datasets[@]}"; do
        for i in {5..6}; do
            export CUDA_VISIBLE_DEVICES=0,1,2,3 \
            # QWENVL_BBOX_FORMAT='new' \
            swift infer \
                --seed ${i} \
                --model Qwen/${model} \
                --infer_backend pt \
                --temperature 1 \
                --max_new_tokens 2048 \
                --val_dataset xxxcold_start/get_is_search_tag/data/swift_input_${ds}.jsonl \
                --max_batch_size 16 \
                --use_hf true \
                --system xxxcold_start/get_is_search_tag/data/system_prompt_${task}.txt \
                --result_path xxxcold_start/get_is_search_tag/result/4times/${model}/${task}/${ds}/qwen_${B}_${task}_${ds}_${i}.jsonl
        done
    done
done
# for i in {1..4}; do
#     export CUDA_VISIBLE_DEVICES=0,1 \
#     # QWENVL_BBOX_FORMAT='new' \
#     swift infer \
#         --seed ${i} \
#         --model Qwen/Qwen2.5-VL-3B-Instruct \
#         --infer_backend pt \
#         --temperature 1 \
#         --max_new_tokens 2048 \
#         --val_dataset xxxcold_start/get_is_search_tag/data/swift_input_test.jsonl \
#         --max_batch_size 16 \
#         --use_hf true \
#         --system xxxcold_start/get_is_search_tag/data/system_prompt.txt \
#         --result_path xxxcold_start/get_is_search_tag/result/4times/Qwen2.5-VL-3B-Instruct/gmner/test/qwen_3B_gmner_test_${i}.jsonl
# done

# for i in {1..4}; do
#     CUDA_VISIBLE_DEVICES=0,1,2,3 \
#     # QWENVL_BBOX_FORMAT='new' \
#     swift infer \
#         --seed ${i} \
#         --model Qwen/Qwen2.5-VL-3B-Instruct \
#         --infer_backend pt \
#         --temperature 1 \
#         --max_new_tokens 2048 \
#         --val_dataset xxxcold_start/get_is_search_tag/data/swift_input_test.jsonl \
#         --max_batch_size 16 \
#         --use_hf true \
#         --system xxxcold_start/get_is_search_tag/data/system_prompt_FMNERG.txt \
#         --result_path xxxcold_start/get_is_search_tag/result/Qwen2.5-VL-3B-Instruct/fmnerg/test/qwen_3B_fmnerg_test_${i}.jsonl
# done


# CUDA_VISIBLE_DEVICES=0,1,2,3 \
# # QWENVL_BBOX_FORMAT='new' \
# swift infer \
#     --model Qwen/Qwen2.5-VL-7B-Instruct \
#     --infer_backend pt \
#     --temperature 0 \
#     --max_new_tokens 2048 \
#     --val_dataset xxxcold_start/get_is_search_tag/data/swift_input_dev.jsonl \
#     --use_hf true \
#     --system xxxcold_start/get_is_search_tag/data/system_prompt_FMNERG.txt \
#     --result_path xxxcold_start/get_is_search_tag/result_fmnerg/Qwen2.5-VL-7B-Instruct/qwen_7B_fmnerg_dev.jsonl


# CUDA_VISIBLE_DEVICES=0,1,2,3 \
# # QWENVL_BBOX_FORMAT='new' \
# swift infer \
#     --model Qwen/Qwen2.5-VL-7B-Instruct \
#     --infer_backend pt \
#     --temperature 0 \
#     --max_new_tokens 2048 \
#     --val_dataset xxxcold_start/get_is_search_tag/data/swift_input_test.jsonl \
#     --use_hf true \
#     --system xxxcold_start/get_is_search_tag/data/system_prompt_FMNERG.txt \
#     --result_path xxxcold_start/get_is_search_tag/result_fmnerg/Qwen2.5-VL-7B-Instruct/qwen_7B_fmnerg_test.jsonl