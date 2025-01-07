#!/bin/bash

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# python eval.py \
#     --model_path="/mnt/workspace/zpf/.cache/Qwen2.5-7B-Instruct" \
#     --dataset_dir="/mnt/workspace/zpf/FS/YM-66K/ym-66k" \
#     --save_dir="./qwen25_results/" \
#     --model_type="Qwen" \
#     --eval_type="text" \
#     --partition="test_Spanish" \
#     --batch_size=500 \
#     --max_length=1024

# python eval.py \
#     --model_path="/mnt/workspace/zpf/.cache/Qwen2.5-7B-Instruct" \
#     --dataset_dir="/mnt/workspace/zpf/FS/YM-66K/ym-66k" \
#     --save_dir="./qwen25_results/" \
#     --eval_type="text" \
#     --model_type="Qwen" \
#     --partition="test_American" \
#     --batch_size=500 \
#     --max_length=1024

# python eval.py \
#     --model_path="/mnt/workspace/zpf/.cache/Qwen2.5-7B-Instruct" \
#     --dataset_dir="/mnt/workspace/zpf/FS/YM-66K/ym-66k" \
#     --save_dir="./qwen25_results/" \
#     --eval_type="text" \
#     --model_type="Qwen" \
#     --partition="test_Chinese" \
#     --batch_size=500 \
#     --max_length=1024

# python eval.py \
#     --model_path="/mnt/workspace/zpf/.cache/Qwen2.5-7B-Instruct" \
#     --dataset_dir="/mnt/workspace/zpf/FS/YM-66K/ym-66k" \
#     --save_dir="./qwen25_results/" \
#     --eval_type="text" \
#     --model_type="Qwen" \
#     --partition="test_French" \
#     --batch_size=500 \
#     --max_length=1024

# python eval.py \
#     --model_path="/mnt/workspace/zpf/.cache/Qwen2.5-7B-Instruct" \
#     --dataset_dir="/mnt/workspace/zpf/FS/YM-66K/ym-66k" \
#     --save_dir="./qwen25_results/" \
#     --eval_type="text" \
#     --model_type="Qwen" \
#     --partition="test_Indian" \
#     --batch_size=500 \
#     --max_length=1024

# python eval.py \
#     --model_path="/mnt/workspace/zpf/.cache/Qwen2.5-7B-Instruct" \
#     --dataset_dir="/mnt/workspace/zpf/FS/YM-66K/ym-66k" \
#     --save_dir="./qwen25_results/" \
#     --eval_type="text" \
#     --model_type="Qwen" \
#     --partition="test_Italian" \
#     --batch_size=500 \
#     --max_length=1024

# python eval.py \
#     --model_path="/mnt/workspace/zpf/.cache/Qwen2.5-7B-Instruct" \
#     --dataset_dir="/mnt/workspace/zpf/FS/YM-66K/ym-66k" \
#     --save_dir="./qwen25_results/" \
#     --eval_type="text" \
#     --model_type="Qwen" \
#     --partition="test_Japanese" \
#     --batch_size=500 \
#     --max_length=1024


# python eval.py \
#     --model_path="/mnt/workspace/zpf/FS/Qwen2-VL/models/foodsky_lora_sft_Spanish" \
#     --dataset_dir="/mnt/workspace/zpf/FS/YM-66K/ym-66k" \
#     --save_dir="./results/" \
#     --eval_type="text_foodsky" \
#     --model_type="Qwen" \
#     --partition="test_Spanish" \
#     --batch_size=500 \
#     --max_length=10240

# python eval.py \
#     --model_path="/mnt/workspace/zpf/FS/Qwen2-VL/models/foodsky_lora_sft_American" \
#     --dataset_dir="/mnt/workspace/zpf/FS/YM-66K/ym-66k" \
#     --save_dir="./results/" \
#     --eval_type="text_foodsky" \
#     --model_type="Qwen" \
#     --partition="test_American" \
#     --batch_size=500 \
#     --max_length=1024

# python eval.py \
#     --model_path="/mnt/workspace/zpf/FS/Qwen2-VL/models/foodsky_lora_sft_Chinese" \
#     --dataset_dir="/mnt/workspace/zpf/FS/YM-66K/ym-66k" \
#     --save_dir="./results/" \
#     --eval_type="text_foodsky" \
#     --model_type="Qwen" \
#     --partition="test_Chinese" \
#     --batch_size=500 \
#     --max_length=1024

# python eval.py \
#     --model_path="/mnt/workspace/zpf/FS/Qwen2-VL/models/foodsky_lora_sft_French" \
#     --dataset_dir="/mnt/workspace/zpf/FS/YM-66K/ym-66k" \
#     --save_dir="./results/" \
#     --eval_type="text_foodsky" \
#     --model_type="Qwen" \
#     --partition="test_French" \
#     --batch_size=500 \
#     --max_length=1024

# python eval.py \
#     --model_path="/mnt/workspace/zpf/FS/Qwen2-VL/models/foodsky_lora_sft_Indian" \
#     --dataset_dir="/mnt/workspace/zpf/FS/YM-66K/ym-66k" \
#     --save_dir="./results/" \
#     --eval_type="text_foodsky" \
#     --model_type="Qwen" \
#     --partition="test_Indian" \
#     --batch_size=500 \
#     --max_length=1024

# python eval.py \
#     --model_path="/mnt/workspace/zpf/FS/Qwen2-VL/models/foodsky_lora_sft_Italian" \
#     --dataset_dir="/mnt/workspace/zpf/FS/YM-66K/ym-66k" \
#     --save_dir="./results/" \
#     --eval_type="text_foodsky" \
#     --model_type="Qwen" \
#     --partition="test_Italian" \
#     --batch_size=500 \
#     --max_length=1024

# python eval.py \
#     --model_path="/mnt/workspace/zpf/FS/Qwen2-VL/models/foodsky_lora_sft_Japanese" \
#     --dataset_dir="/mnt/workspace/zpf/FS/YM-66K/ym-66k" \
#     --save_dir="./results/" \
#     --eval_type="text_foodsky" \
#     --model_type="Qwen" \
#     --partition="test_Japanese" \
#     --batch_size=500 \
#     --max_length=1024
