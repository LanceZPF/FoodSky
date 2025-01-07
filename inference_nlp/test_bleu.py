import re
from tqdm import tqdm
import numpy as np

import os
import argparse
import pandas as pd
import json

import time

from nltk.translate.bleu_score import sentence_bleu
import jieba

def evaluate_bleu(data_file, answer_file, n, type=None):
    with open(data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 读取生成的答案文件
    with open(answer_file, 'r', encoding='utf-8') as f:
        answer_data = f.read()
    
    # 使用正则表达式，以 0 为分隔符拆分答案
    answers = re.split(r'\n0\n', answer_data)
    
    total_bleu = 0
    count = 0
    for i, item in enumerate(data):
        # 如果指定了type,则只评估该type的问题
        if type is not None and item.get('type') != type:
            continue
            
        count += 1
        question = item['instruction']+item['input']
        reference = item['instruction']+item['input']+item['output']
        
        response = answers[i]
        
        # 计算 BLEU 分数
        reference_tokens = list(jieba.lcut(reference))
        response_tokens = list(jieba.lcut(response))
        
        weights = [0] * 4
        for i in range(n):
            weights[i] = 1.0 / n
        
        bleu = sentence_bleu([reference_tokens], response_tokens, weights=weights)
        total_bleu += bleu
        
    avg_bleu = total_bleu / count if count > 0 else 0
    return avg_bleu

def main_qa():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_file', default='cookdiet_qa22.json', type=str)
    parser.add_argument('--answer_file', default='food_answer_1.csv', type=str)
    parser.add_argument('--output_file', default='output_bleu.txt', type=str)
    args = parser.parse_args()

    data_file = args.data_file
    answer_file = args.answer_file
    output_file = args.output_file

    f = open(output_file, 'a', encoding='utf-8')
    f.write("BLEU Results\n\n")

    # 评估所有问题
    f.write("Overall Results:\n")
    for n in range(1, 5):
        bleu_score = evaluate_bleu(data_file, answer_file, n)
        print(f"Average BLEU-{n} score: {bleu_score:.4f}")
        f.write(f"Average BLEU-{n} score: {bleu_score:.4f}\n")
    
    # 评估culinary类型问题
    f.write("\nCulinary Type Results:\n")
    for n in range(1, 5):
        bleu_score = evaluate_bleu(data_file, answer_file, n, type='culinary')
        print(f"Culinary BLEU-{n} score: {bleu_score:.4f}")
        f.write(f"Culinary BLEU-{n} score: {bleu_score:.4f}\n")

    # 评估dietetic类型问题  
    f.write("\nDietetic Type Results:\n")
    for n in range(1, 5):
        bleu_score = evaluate_bleu(data_file, answer_file, n, type='dietetic')
        print(f"Dietetic BLEU-{n} score: {bleu_score:.4f}")
        f.write(f"Dietetic BLEU-{n} score: {bleu_score:.4f}\n")

    f.close()

if __name__ == "__main__":
    main_qa()