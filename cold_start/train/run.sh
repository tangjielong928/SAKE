#!/bin/bash
#  --save_strategy epoch \
export CUDA_VISIBLE_DEVICES=1,2

torchrun --nproc_per_node=2 \
  $(which swift) sft \
  --model xxx/models/Qwen/Qwen2.5-VL-7B-Instruct \
  --use_hf true \
  --dataset ./cold_start/train/sake_secot.jsonl \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --num_train_epochs 3 \
   --save_strategy epoch   \
  --output_dir output_cot_7B_secot \
  --warmup_ratio 0.05 \
  --train_type full \
  --gradient_checkpointing true \
  --deepspeed zero3

# swift sft \
#   --model xxx/models/Qwen/Qwen2.5-VL-3B-Instruct \
#   --use_hf true \
#   --dataset ./cold_start/train/test_sample.jsonl \
#   --per_device_train_batch_size 2 \
#   --num_train_epochs 3 \
#   --save_steps 2 \
#   --output_dir output_cot \
#   --train_type full \