# llamafactory-cli export \
#     --model_name_or_path /mnt/workspace/zpf/.cache/Qwen2.5-14B-Instruct \
#     --adapter_name_or_path /mnt/workspace/zpf/FS/LLaMA-Factory/saves/foodsky-14b/lora/sft0/checkpoint-100 \
#     --template qwen \
#     --finetuning_type lora \
#     --export_dir foodsky-14b-1230-100 \
#     --export_size 2 \
#     --export_legacy_format False

# llamafactory-cli export \
#     --model_name_or_path /mnt/workspace/zpf/.cache/Qwen2.5-14B-Instruct \
#     --adapter_name_or_path /mnt/workspace/zpf/FS/LLaMA-Factory/saves/foodsky-14b/lora/sft0/checkpoint-500 \
#     --template qwen \
#     --finetuning_type lora \
#     --export_dir foodsky-14b-1230-500 \
#     --export_size 2 \
#     --export_legacy_format False

llamafactory-cli export \
    --model_name_or_path /mnt/workspace/zpf/.cache/Qwen2.5-14B-Instruct \
    --adapter_name_or_path /mnt/workspace/zpf/FS/LLaMA-Factory/saves/foodsky-14b/lora/sft0/checkpoint-1000 \
    --template qwen \
    --finetuning_type lora \
    --export_dir foodsky-14b-1230-1000 \
    --export_size 2 \
    --export_legacy_format False

# llamafactory-cli export \
#     --model_name_or_path /mnt/workspace/zpf/.cache/Qwen2.5-14B-Instruct \
#     --adapter_name_or_path /mnt/workspace/zpf/FS/LLaMA-Factory/saves/foodsky-14b/lora/sft0 \
#     --template qwen \
#     --finetuning_type lora \
#     --export_dir foodsky-14b-1230-fn \
#     --export_size 2 \
#     --export_legacy_format False
