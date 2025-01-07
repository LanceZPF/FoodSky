llamafactory-cli train qwen2.5-7b-lora-sft.yaml

# DISTRIBUTED_ARGS="
#     --nproc_per_node 8 \
#     --nnodes 1 \
#     --node_rank 0 \
#     --master_addr 127.0.0.1 \
#     --master_port 12446
# "

# torchrun $DISTRIBUTED_ARGS src/train.py \
#     --deepspeed /mnt/workspace/zpf/FS/LLaMA-Factory/examples/deepspeed/ds_z3_config.json \
#     --stage sft \
#     --do_train \
#     --use_fast_tokenizer \
#     --flash_attn \
#     --model_name_or_path /mnt/workspace/zpf/.cache/Qwen2.5-7B-Instruct \
#     --dataset alpaca_foodsky_zh, alpaca_zh_demo, glaive_toolcall_zh_demo \
#     --template qwen \
#     --finetuning_type lora \
#     --lora_target q_proj,v_proj \
#     --output_dir /mnt/workspace/zpf/FS/outputs/qwen2.5-7b-instruct-sft \
#     --overwrite_cache \
#     --overwrite_output_dir \
#     --warmup_steps 100 \
#     --weight_decay 0.1 \
#     --per_device_train_batch_size 4 \
#     --gradient_accumulation_steps 4 \
#     --ddp_timeout 9000 \
#     --learning_rate 5e-5 \
#     --lr_scheduler_type cosine \
#     --logging_steps 1 \
#     --cutoff_len 10240 \
#     --save_steps 1000 \
#     --plot_loss \
#     --num_train_epochs 3 \
#     --bf16