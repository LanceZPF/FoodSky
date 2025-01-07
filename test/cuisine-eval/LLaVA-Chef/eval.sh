#!/bin/bash

python eval.py \
    --model_path="/mnt/workspace/zpf/.cache/Qwen2-VL-7B-Instruct" \
    --dataset_dir="/mnt/workspace/zpf/FS/YM-66K/ym-66k" \
    --save_dir="./qwen2vl_results/" \
    --eval_type="text" \
    --model_type="Qwen" \
    --partition="test_American" \
    --start=0 \
    --end=-1 \
    --batch_size=500

python eval.py \
    --model_path="/mnt/workspace/zpf/.cache/Qwen2-VL-7B-Instruct" \
    --dataset_dir="/mnt/workspace/zpf/FS/YM-66K/ym-66k" \
    --save_dir="./qwen2vl_results/" \
    --eval_type="text" \
    --model_type="Qwen" \
    --partition="test_Chinese" \
    --start=0 \
    --end=-1 \
    --batch_size=500

python eval.py \
    --model_path="/mnt/workspace/zpf/.cache/Qwen2-VL-7B-Instruct" \
    --dataset_dir="/mnt/workspace/zpf/FS/YM-66K/ym-66k" \
    --save_dir="./qwen2vl_results/" \
    --eval_type="text" \
    --model_type="Qwen" \
    --partition="test_Indian" \
    --start=0 \
    --end=-1 \
    --batch_size=500

python eval.py \
    --model_path="/mnt/workspace/zpf/.cache/Qwen2-VL-7B-Instruct" \
    --dataset_dir="/mnt/workspace/zpf/FS/YM-66K/ym-66k" \
    --save_dir="./qwen2vl_results/" \
    --eval_type="text" \
    --model_type="Qwen" \
    --partition="test_Italian" \
    --start=0 \
    --end=-1 \
    --batch_size=500

python eval.py \
    --model_path="/mnt/workspace/zpf/.cache/Qwen2-VL-7B-Instruct" \
    --dataset_dir="/mnt/workspace/zpf/FS/YM-66K/ym-66k" \
    --save_dir="./qwen2vl_results/" \
    --eval_type="text" \
    --model_type="Qwen" \
    --partition="test_Japanese" \
    --start=0 \
    --end=-1 \
    --batch_size=500

