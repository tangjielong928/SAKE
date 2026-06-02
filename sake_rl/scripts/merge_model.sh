#!/bin/bash

python sake_rl/scripts/legacy_model_merger.py merge \
    --backend fsdp \
    --local_dir xxx/code/checkpoints/Twitter-GMNER/Qwen2.5-VL-7B_grpo_search_ckpt282_20260112/global_step_4800/actor \
    --target_dir ./SAKE/SAKE-7B/20260112_step_4800 \
    --hf_model_path xxx/Qwen2.5-VL-7B-Instruct
