#!/bin/bash

# Twitter GMNER inference
# Supports two modes: transformers (local model) or openai (API)

# 设置项目根目录
PROJECT_ROOT="/root/work/filestorage/xxx/sake_rl/code"
cd ${PROJECT_ROOT}

# 设置 PYTHONPATH
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

# ================== 通用配置 ==================
TEST_DATA="/root/work/filestorage/xxx/sake_rl/code/sake_rl/data/twitter_fmnerg_gt.json"
IMAGE_ROOT="/root/work/filestorage/xxx/sake_rl/dataset/dataset/whole_image"
SEARCH_CACHE_PATH="/root/work/filestorage/xxx/sake_rl/dataset/twitter_gmner_test"
OUTPUT_FILE="prediction/predictions_SAKE_fmnerg_7B_1500_step_230.json"
TASK_TYPE="fmnerg"

# ================== 评估配置 ==================
GT_PATH="/root/work/filestorage/xxx/sake_rl/code/sake_rl/data/twitter_fmnerg_gt.json"  # GT 文件路径
ENABLE_EVALUATION=true  # 是否启用评估

# Prompt paths
ROUND_1_PROMPT="/root/work/filestorage/xxx/sake_rl/code/sake_rl/prompts/Twitter-FMNERG-Prompt/round_1_user_prompt_1_fmnerg.txt"
AFTER_TEXT_SEARCH_PROMPT="/root/work/filestorage/xxx/sake_rl/code/sake_rl/prompts/Twitter-FMNERG-Prompt/after_text_search_prompt_1.txt"
AFTER_IMAGE_SEARCH_PROMPT="/root/work/filestorage/xxx/sake_rl/code/sake_rl/prompts/Twitter-FMNERG-Prompt/after_image_search_prompt_1.txt"

# ================== 选择推理模式 ==================
# 设置为 "transformers" 或 "openai"
INFERENCE_MODE="openai"

# ================== Transformers 模式配置 ==================
MODEL_PATH="xxx/SAKE/SAKE-7B/20260112_step_4800"

# ================== OpenAI/vLLM 模式配置 ==================
# vLLM 服务配置（对应 vllm_start.sh 启动的服务）
export OPENAI_API_KEY="EMPTY"  # vLLM 不需要真实 API key
API_BASE="http://localhost:10095/v1"  # vLLM 服务地址
MODEL_NAME="SAKE"  # 与 vllm_start.sh 中 --served-model-name 一致
TEMPERATURE=0.0

# ================== 运行推理 ==================
# 构建评估参数
EVAL_ARGS=""
if [ "$ENABLE_EVALUATION" = "true" ] && [ -n "$GT_PATH" ]; then
    EVAL_ARGS="--evaluate --gt_path ${GT_PATH}"
fi

if [ "$INFERENCE_MODE" = "transformers" ]; then
    echo ">>> Running in Transformers mode"
    CUDA_VISIBLE_DEVICES=0 python3 sake_rl/scripts/inference_twitter_gmner.py \
        --inference_mode transformers \
        --model_path ${MODEL_PATH} \
        --test_data ${TEST_DATA} \
        --image_root ${IMAGE_ROOT} \
        --search_cache_path ${SEARCH_CACHE_PATH} \
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
    
    # 构建可选参数
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

