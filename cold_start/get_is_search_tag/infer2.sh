for i in {1..4}; do
    export CUDA_VISIBLE_DEVICES=2,3 \
    # QWENVL_BBOX_FORMAT='new' \
    swift infer \
        --seed ${i} \
        --model Qwen/Qwen2.5-VL-7B-Instruct \
        --infer_backend pt \
        --temperature 1 \
        --max_new_tokens 2048 \
        --val_dataset xxx/cold_start/get_is_search_tag/data/swift_input_test.jsonl \
        --max_batch_size 16 \
        --use_hf true \
        --system xxx/cold_start/get_is_search_tag/data/system_prompt.txt \
        --result_path xxx/cold_start/get_is_search_tag/result/4times/Qwen2.5-VL-7B-Instruct/gmner/test/qwen_7B_gmner_test_${i}.jsonl
done