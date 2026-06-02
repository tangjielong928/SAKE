
CUDA_VISIBLE_DEVICES=1 python -m vllm.entrypoints.openai.api_server \
    --port 10091 \
    --served-model-name SAKE \
    --model /data/cl/SAKE-Dev/SAKE/checkpoints/checkpoint-2604 \
    --gpu-memory-utilization 0.90 \
    --max-model-len 32768 \
    --trust-remote-code
