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
import jieba  # 导入 jieba 库

DEFAULT_SYSTEM_PROMPT_QA = """You are a helpful assistant. 你是一个乐于助人的助手。"""

openai.api_key = 'XXX'

def gpt_output_qa(question):

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

def evaluate_bleu(data_file, n):
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
        reference_tokens = list(jieba.lcut(reference))  # 使用 jieba 分词
        response_tokens = list(jieba.lcut(response))
        
        weights = [0] * 4
        for i in range(n):
            weights[i] = 1.0 / n
        
        bleu = sentence_bleu([reference_tokens], response_tokens, weights=weights)
        
        total_bleu += bleu
        
    avg_bleu = total_bleu / len(data)
    return avg_bleu

def main_qa():
    data_file = 'bleugleu/cookdiet_qa.json'
    output_file = 'bleugleu/bleu_results.txt'

    f = open(output_file, 'w', encoding='utf-8')
    f.write("BLEU Results\n\n")

    for n in range(1, 5):
        bleu_score = evaluate_bleu(data_file, n)
        print(f"Average BLEU-{n} score: {bleu_score:.4f}")
        f.write(f"Average BLEU-{n} score: {bleu_score:.4f}\n")
    
    f.close()

if __name__ == "__main__":
    main_qa()
