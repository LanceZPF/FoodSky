
'''
- Use latest version of transformers

'''


import argparse

from models.instructBLIP import InstructBlip
from models.gpt2 import GTP2
from models.phi2 import Phi2
from models.mistral import Mistral
from models.llama import LLAMA
from models.qwen import Qwen

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="/mnt/workspace/zpf/.cache/Qwen2.5-7B-Instruct")
    parser.add_argument("--dataset_dir", type=str, default="/mnt/workspace/zpf/FS/YM-66K/ym-66k")
    parser.add_argument("--save_dir", type=str, default="results/")
    parser.add_argument("--max_length", type=int, default=32)
    parser.add_argument("--model_type", type=str, default="qwen")
    parser.add_argument("--eval_type", type=str, default="yummly")
    parser.add_argument("--partition", type=str, default="test_Spanish")
    parser.add_argument("--prompt_type", type=str, default="instruct")

    parser.add_argument("--batch_size", type=int, default=1)

    args = parser.parse_args()

    model = eval(args.model_type)(args)

    model.predict()
