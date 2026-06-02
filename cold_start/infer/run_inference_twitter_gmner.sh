#!/bin/bash

# Twitter GMNER inference

# 设置项目根目录
PROJECT_ROOT="xxx/cold_start/infer"
cd ${PROJECT_ROOT}

# 设置 PYTHONPATH
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

MODEL_PATH="xxx/baseline_experiment/output/Qwen2.5-VL-3B-Instruct/v8-20251118-054355/checkpoint-6132"  # 修改为你的模型路径
TEST_DATA="xxx/baseline_experiment/Twitter-GMNER/twitter_gmner_test.json"  # 修改为你的测试数据路径
IMAGE_ROOT="xxx/baseline_experiment/Twitter-GMNER/whole_image"
SEARCH_CACHE_PATH="xxx/cold_start/cot_data/twitter_gmner_test"
OUTPUT_FILE="predictions.json"

# Prompt paths
ROUND_1_PROMPT="xxx/cold_start/infer/round_1_user_prompt_1.txt"
AFTER_TEXT_SEARCH_PROMPT="xxx/cold_start/infer/after_text_search_prompt_1.txt"
AFTER_IMAGE_SEARCH_PROMPT="xxx/cold_start/infer/after_image_search_prompt_1.txt"

CUDA_VISIBLE_DEVICES=1 python3 inference_twitter_gmner.py \
    --model_path ${MODEL_PATH} \
    --test_data ${TEST_DATA} \
    --image_root ${IMAGE_ROOT} \
    --search_cache_path ${SEARCH_CACHE_PATH} \
    --output_file ${OUTPUT_FILE} \
    --round_1_prompt ${ROUND_1_PROMPT} \
    --after_text_search_prompt ${AFTER_TEXT_SEARCH_PROMPT} \
    --after_image_search_prompt ${AFTER_IMAGE_SEARCH_PROMPT} \
    --start_idx 0 \
    --end_idx 10 \
    --max_new_tokens 2048

