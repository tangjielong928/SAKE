CUDA_VISIBLE_DEVICES=0 swift export \
  --ckpt_dir xxx/output_cot_3B/v5-20251222-054027/checkpoint-748 \
  --output_dir xxx/output_cot_3B/merged \
  --merge_lora false