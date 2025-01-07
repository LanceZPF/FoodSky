
import argparse

# from llava.eval.eval_recipe import EvalModel
from qwenvl import QwenVLEvalModel

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="./checkpoints/llava-7b-finetune")
    parser.add_argument("--mm_project_path", type=str, default=None)
    parser.add_argument("--dataset_dir", type=str, default="/data/datasets/Food/Recipe1M/")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--save_dir", type=str, default="results/")
    parser.add_argument("--model_type", type=str, default="LLaVAChef")
    parser.add_argument("--eval_type", type=str, default="recipe1m")
    parser.add_argument("--partition", type=str, default="test_1kimage")
    parser.add_argument("--prompt_type", type=str, default="i__instruct")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=-1)

    parser.add_argument("--batch_size", type=int, default=50)

    args = parser.parse_args()

    # model = EvalModel(args)

    model = QwenVLEvalModel(args)

    if args.batch_size == 1:
        model.predict()
    else:
        model.predict_batch()