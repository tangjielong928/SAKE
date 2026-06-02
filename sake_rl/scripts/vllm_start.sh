
CUDA_VISIBLE_DEVICES=4 python -m vllm.entrypoints.openai.api_server \
    --port 10095 \
    --served-model-name SAKE \
    --model /root/work/filestorage/xxx/sake_rl/code/checkpoints/Twitter-GMNER/Qwen2.5-VL-7B_grpo_search_ckpt282_20260112/global_step_4800/actor \
    --gpu-memory-utilization 0.90 \
    --max-model-len 32768 \
    --trust-remote-code
