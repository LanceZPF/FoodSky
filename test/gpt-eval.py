# This code is modified from C-Eval Project: https://github.com/SJTU-LIT/ceval

import re
from tqdm import tqdm
import numpy as np

import os
import argparse
import pandas as pd
import json

import openai

import time

from nltk.translate.bleu_score import sentence_bleu

datadir = 'newdata'

DEFAULT_SYSTEM_PROMPT_CHOICE = "你是一个回答考试选择题的助手，请根据问题正确回答A、B、C或D。"

DEFAULT_SYSTEM_PROMPT_QA = """You are a helpful assistant. 你是一个乐于助人的助手。"""

openai.api_key = 'XXX'

choices = ["A", "B", "C", "D"]

def gpt_output_choice(question):

    # 设置你的 OpenAI API 密钥

    input_text = question
    # 调用 OpenAI API
    response = openai.chat.completions.create(
        # model="gpt-4-turbo-preview",
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": DEFAULT_SYSTEM_PROMPT_CHOICE},
            {"role": "user", "content": input_text}
        ],
        max_tokens=1,
        seed=42
    )

    return response.choices[0].message.content

def gpt_output_qa(question):

    # 设置你的 OpenAI API 密钥

    input_text = question
    # 调用 OpenAI API
    response = openai.chat.completions.create(
        # model="gpt-4-turbo-preview",
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": DEFAULT_SYSTEM_PROMPT_QA},
            {"role": "user", "content": input_text}
        ],
        seed=42
    )

    return response.choices[0].message.content

# This code is modified from C-Eval Project: https://github.com/SJTU-LIT/ceval

def format_example(line, include_answer=True, cot=False, with_prompt=False):
    example = line['question']
    for choice in choices:
        example += f'\n{choice}. {line[f"{choice}"]}'
    if include_answer:
        if cot:
            example += "\n答案：让我们一步一步思考，\n" + \
                line["explanation"] + f"\n所以答案是{line['answer']}。\n\n"
        else:
            example += '\n答案：' + line["answer"] + '\n\n'
    else:
        if with_prompt is False:
            if cot:
                example += "\n答案：让我们一步一步思考，\n1."
        else:
            if cot:
                example += "\n答案是什么？让我们一步一步思考，\n1."
    return example

def main_choice(args, evaluator,  take):
    if evaluator == 'GPT':
        assert os.path.exists(f"{datadir}/subject_mapping.json"), "subject_mapping.json not found!"
        with open(f"{datadir}/subject_mapping.json", encoding='utf-8') as f:
            subject_mapping = json.load(f)
        filenames = os.listdir(f"{datadir}/val")
        subject_list = [val_file.replace("_val.csv","") for val_file in filenames]
        accuracy, summary = {}, {}

        run_date=time.strftime('%Y-%m-%d_%H-%M-%S',time.localtime(time.time()))
        output_dir = args.output_dir
        save_result_dir=os.path.join(output_dir,f"take{take}")
        if not os.path.exists(save_result_dir):
            os.makedirs(save_result_dir,exist_ok=True)

        all_answers = {}
        for index, subject_name in enumerate(subject_list):
            print(f"{subject_name} Inference starts!")
            val_file_path=os.path.join(f'{datadir}/val',f'{subject_name}_val.csv')
            dev_file_path=os.path.join(f'{datadir}/dev',f'{subject_name}_dev.csv')
            test_file_path=os.path.join(f'{datadir}/test',f'{subject_name}_test.csv')

            val_df=pd.read_csv(val_file_path) if args.do_test is False else pd.read_csv(test_file_path)
            dev_df=pd.read_csv(dev_file_path) if args.few_shot else None

            answers = ['NA'] * len(val_df) if args.do_test is True else list(val_df['answer'])

            correct_num = 0

            for row_index, row in tqdm(val_df.iterrows(), total=len(val_df)):

                question = format_example(row, include_answer=False, cot=args.cot, with_prompt=args.with_prompt)

                inputs = question
                ans = gpt_output_choice(inputs)
                if ans == answers[row_index]:
                    correct_num += 1
                else:
                    print()
                    print(row_index)
                    print(ans)
                    print(answers[row_index])

            correct_ratio = 100*correct_num/len(answers)
            print(f"Subject: {subject_name}")
            print(f"Acc: {correct_ratio}")
            accuracy[subject_name] = correct_ratio
            summary[subject_name] = {"score":correct_ratio,
                                    "num":len(val_df),
                                    "correct":correct_ratio*len(val_df)/100}
            all_answers[subject_name] = answers

        json.dump(all_answers,open(save_result_dir+'/submission.json','w'),ensure_ascii=False,indent=4)

        print("Accuracy:")
        for k, v in accuracy.items():
            print(k, ": ", v)

        total_num = 0
        total_correct = 0
        summary['grouped'] = {
            "cook": {"correct": 0.0, "num": 0}, 
            "nutrition": {"correct": 0.0, "num": 0}
            }
        for subj, info in subject_mapping.items():
            group = info[2]
            summary['grouped'][group]["num"]   += summary[subj]['num']
            summary['grouped'][group]["correct"] += summary[subj]['correct']
        for group, info in summary['grouped'].items():
            info['score'] = info["correct"] / info["num"]
            total_num += info["num"]
            total_correct += info["correct"]
        summary['All'] = {"score": total_correct / total_num, "num": total_num, "correct": total_correct}

        print(summary['All'])

        json.dump(summary,open(save_result_dir+'/summary.json','w'),ensure_ascii=False,indent=2)

def evaluate_bleu(data_file):
    with open(data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    total_bleu = 0
    for item in data:
        question = item['instruction']+item['input']
        reference = item['output'] # 取第一句作为参考答案
        print('ground truth')
        print(reference)
        
        # 调用 OpenAI API 获取模型生成的答案
        response = gpt_output_qa(question)
        print('model output')
        print(response)
        
        # 计算 BLEU 分数
        reference_tokens = list(reference)
        response_tokens = list(response)
        bleu = sentence_bleu([reference_tokens], response_tokens)
        
        total_bleu += bleu
        
    avg_bleu = total_bleu / len(data)
    return avg_bleu

def main_qa():
    data_file = 'bleugleu\cookdiet_qa.json'
    bleu_score = evaluate_bleu(data_file)
    print(f"Average BLEU score: {bleu_score:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cot",choices=["False","True"], default="False")
    parser.add_argument("--few_shot", choices=["False","True"], default="True")
    parser.add_argument("--ntrain", "-k", type=int, default=5)
    parser.add_argument("--with_prompt", choices=["False","True"], default="True")
    parser.add_argument("--constrained_decoding", choices=["False","True"], default="True")
    parser.add_argument("--temperature",type=float,default=0.2)
    parser.add_argument("--n_times", default=1,type=int)
    parser.add_argument("--do_save_csv", choices=["False","True"], default="False")
    parser.add_argument("--output_dir", type=str, default='result')
    parser.add_argument("--do_test", choices=["False","True"], default="False")
    parser.add_argument("--verbose", action="store_true", help="Print detailed information of each example.")

    args = parser.parse_args()

    args.cot = args.cot == "True"
    args.few_shot = args.few_shot == "True"
    args.with_prompt = args.with_prompt == "True"
    args.constrained_decoding = args.constrained_decoding == "True"
    args.do_test = args.do_test == "True"
    args.do_save_csv = args.do_save_csv == "True"
    if args.constrained_decoding is True:
        args.n_times=max(args.n_times,1)
    print(args)
    
    # print(gpt_output('s','s'))
    for i in range(args.n_times):
        main_choice(args,evaluator='GPT',take=i)
