#!/bin/bash

# Twitter GMNER inference
# Supports two modes: transformers (local model) or openai (API)

# Set project root
PROJECT_ROOT="/data/cl/SAKE-Dev/SAKE"
cd ${PROJECT_ROOT}

# Set PYTHONPATH
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"


# ================== Common configuration ==================
TEST_DATA="./sake_rl/data/twitter_gmner_gt.json"
IMAGE_ROOT="./sake_rl/data/Twitter-GMNER/whole_image"
SEARCH_CACHE_PATH="./sake_rl/data/twitter_cached_search/twitter_gmner_test"
ONLINE_SEARCH=false  # Enable online search; false uses the offline cache at SEARCH_CACHE_PATH
OUTPUT_FILE="prediction/test0602.json"
TASK_TYPE="gmner"

# ================== Evaluation configuration ==================
GT_PATH="./sake_rl/data/twitter_gmner_gt.json"  # Ground-truth file path
ENABLE_EVALUATION=true  # Whether to enable evaluation

# Prompt paths
ROUND_1_PROMPT="./sake_rl/prompts/Twitter-GMNER-Prompt/round_1_user_prompt_1.txt"
AFTER_TEXT_SEARCH_PROMPT="./sake_rl/prompts/Twitter-GMNER-Prompt/after_text_search_prompt_1.txt"
AFTER_IMAGE_SEARCH_PROMPT="./sake_rl/prompts/Twitter-GMNER-Prompt/after_image_search_prompt_1.txt"

# ================== Select inference mode ==================
# Set to "transformers" or "openai"
INFERENCE_MODE="openai"

# ================== Transformers mode configuration ==================
MODEL_PATH="xxx/SAKE/SAKE-7B/20260112_step_4800"

# ================== OpenAI/vLLM mode configuration ==================
# vLLM service configuration, matching the server started by vllm_start.sh
export OPENAI_API_KEY="EMPTY"  # vLLM does not require a real API key
API_BASE="http://localhost:10091/v1"  # vLLM service address
MODEL_NAME="SAKE"  # Must match --served-model-name in vllm_start.sh
TEMPERATURE=0.0

# ================== Run inference ==================
# Build evaluation arguments
EVAL_ARGS=""
if [ "$ENABLE_EVALUATION" = "true" ] && [ -n "$GT_PATH" ]; then
    EVAL_ARGS="--evaluate --gt_path ${GT_PATH}"
fi

ONLINE_SEARCH_ARGS=""
if [ "$ONLINE_SEARCH" = "true" ]; then
    ONLINE_SEARCH_ARGS="--online_search"
fi

if [ "$INFERENCE_MODE" = "transformers" ]; then
    echo ">>> Running in Transformers mode"
    CUDA_VISIBLE_DEVICES=0 python3 sake_rl/scripts/inference_twitter_gmner.py \
        --inference_mode transformers \
        --model_path ${MODEL_PATH} \
        --test_data ${TEST_DATA} \
        --image_root ${IMAGE_ROOT} \
        --search_cache_path ${SEARCH_CACHE_PATH} \
        ${ONLINE_SEARCH_ARGS} \
        --output_file ${OUTPUT_FILE} \
        --round_1_prompt ${ROUND_1_PROMPT} \
        --after_text_search_prompt ${AFTER_TEXT_SEARCH_PROMPT} \
        --after_image_search_prompt ${AFTER_IMAGE_SEARCH_PROMPT} \
	--task_type ${TASK_TYPE} \
        --start_idx 0 \
        --end_idx 1500 \
        --max_new_tokens 8192 \
        ${EVAL_ARGS}

elif [ "$INFERENCE_MODE" = "openai" ]; then
    echo ">>> Running in OpenAI mode"
    
    # Build optional arguments
    OPTIONAL_ARGS=""
    if [ -n "$API_BASE" ]; then
        OPTIONAL_ARGS="$OPTIONAL_ARGS --api_base ${API_BASE}"
    fi
    
    python3 sake_rl/scripts/inference_twitter_gmner.py \
        --inference_mode openai \
        --model_name ${MODEL_NAME} \
        --temperature ${TEMPERATURE} \
        ${OPTIONAL_ARGS} \
        --test_data ${TEST_DATA} \
        --image_root ${IMAGE_ROOT} \
        --search_cache_path ${SEARCH_CACHE_PATH} \
        ${ONLINE_SEARCH_ARGS} \
        --output_file ${OUTPUT_FILE} \
        --round_1_prompt ${ROUND_1_PROMPT} \
        --after_text_search_prompt ${AFTER_TEXT_SEARCH_PROMPT} \
        --after_image_search_prompt ${AFTER_IMAGE_SEARCH_PROMPT} \
	--task_type ${TASK_TYPE} \
        --start_idx 0 \
        --end_idx 1500 \
        --max_new_tokens 8192 \
        ${EVAL_ARGS}
else
    echo "Error: INFERENCE_MODE must be 'transformers' or 'openai'"
    exit 1
fi
