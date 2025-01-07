# Quick Start Guide

## Evaluation

### NLP Metrics (BLEU, GLEU, ROUGE)
The code for evaluating BLEU, GLEU and ROUGE scores is located in the `inference_nlp` directory. You can refer to `inference_nlp/all_test.sh` for example usage.

### CDE (Code Execution) Testing
The CDE evaluation code is in the `test` directory. Use `test/cde-eval/test.sh` to run the evaluation.

### GPT Evaluations
You can leverage GPT models to evaluate benchmarks. See examples in the `test/gpt-eval` directory.

## Training

For training, you can refer to the detailed instructions in `train/train_scripts/README.md`, which covers:

- Resource optimization: Full-tuning, LoRA and QLoRA with various quantization options
- Advanced algorithms: GaLore, BAdam, Adam-mini, DoRA, LongLoRA, etc.
- Practical tricks: FlashAttention-2, Unsloth, Liger Kernel, RoPE scaling, NEFTune and rsLoRA
- Experiment monitoring: LlamaBoard, TensorBoard, Wandb, MLflow, etc.

Training scripts are provided in the `train` directory. You can use these as reference for your own training:

- LoRA finening example: `train/train_scripts/train.sh`
- LoRA merge example: `train/train_scripts/merge_lora.sh`
- The cuisine evaluation example: `test\cuisine-eval\LLMs\eval.sh`
- Other training examples can be found in the training directory

## Other Directory Structure

- The cuisine finetuning data from Yummly-66K is provided in the `train\data` directory.
- Cuisine finetuning example: `train\train_scripts\cuisine_finetune`
- The old training codes for FoodSky-Cl is provided in the `train\train_code_old` directory.
