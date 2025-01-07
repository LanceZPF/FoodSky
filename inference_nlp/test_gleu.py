import re
from tqdm import tqdm
import numpy as np
import os
import argparse
import pandas as pd
import json
import time
from nltk.translate.gleu_score import sentence_gleu
from nltk.translate.gleu_score import corpus_gleu
import jieba

def distinct_n(text, n):
    tokens = list(jieba.cut(text))
    ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
    unique_ngrams = set(ngrams)
    distinct_n = len(unique_ngrams) / len(ngrams)
    return distinct_n

def evaluate_metrics(data_file, answer_file, output_file, type=None):
    with open(data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 读取生成的答案文件
    with open(answer_file, 'r', encoding='utf-8') as f:
        answer_data = f.read()
    
    # 使用正则表达式，以 0 为分隔符拆分答案
    answers = re.split(r'\n0\n', answer_data)
        
    total_gleu = 0
    total_distinct_1 = 0
    total_distinct_2 = 0
    references = []
    responses = []
    count = 0

    with open(output_file, 'a', encoding='utf-8') as f:
        for i, item in enumerate(data):
            # 如果指定了type,则只评估该type的问题
            if type is not None and item.get('type') != type:
                continue
                
            count += 1
            question = item['instruction'] + item['input']
            reference = item['instruction']+item['input']+item['output']
            
            response = answers[i]  # 获取对应的答案
            
            references.append(jieba.lcut(reference))
            responses.append(jieba.lcut(response))
            
            gleu = sentence_gleu([reference], response)
            distinct_1 = distinct_n(response, 1)
            distinct_2 = distinct_n(response, 2)
            
            total_gleu += gleu
            total_distinct_1 += distinct_1
            total_distinct_2 += distinct_2
            
    avg_gleu = total_gleu / count
    avg_distinct_1 = total_distinct_1 / count
    avg_distinct_2 = total_distinct_2 / count
    corpus_gleu_score = corpus_gleu(references, responses)
    
    with open(output_file, 'a', encoding='utf-8') as f:
        if type:
            f.write(f"\n{type.capitalize()} Type Results:\n")
        f.write(f"Average Sentence GLEU score: {avg_gleu:.4f}\n")
        f.write(f"Corpus GLEU score: {corpus_gleu_score:.4f}\n")
        f.write(f"Average Distinct-1: {avg_distinct_1:.4f}\n")
        f.write(f"Average Distinct-2: {avg_distinct_2:.4f}\n")
        
    return avg_gleu, corpus_gleu_score, avg_distinct_1, avg_distinct_2

def main_qa():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_file', default='cookdiet_qa22.json', type=str)
    parser.add_argument('--answer_file', default='food_answer_1.csv', type=str)
    parser.add_argument('--output_file', default='output_gleu.txt', type=str)
    args = parser.parse_args()

    data_file = args.data_file
    answer_file = args.answer_file
    output_file = args.output_file
    
    with open(output_file, 'a', encoding='utf-8') as f:
        f.write("Evaluation Metrics Results\n\n")
    
    # 评估所有问题
    print("Overall Results:\n")
    avg_gleu, corpus_gleu, avg_distinct_1, avg_distinct_2 = evaluate_metrics(data_file, answer_file, output_file)
    print(f"Average Sentence GLEU score: {avg_gleu:.4f}")
    print(f"Corpus GLEU score: {corpus_gleu:.4f}")
    print(f"Average Distinct-1: {avg_distinct_1:.4f}")
    print(f"Average Distinct-2: {avg_distinct_2:.4f}")

    # 评估culinary类型问题
    avg_gleu, corpus_gleu, avg_distinct_1, avg_distinct_2 = evaluate_metrics(data_file, answer_file, output_file, type='culinary')
    print(f"\nCulinary Type Results:")
    print(f"Average Sentence GLEU score: {avg_gleu:.4f}")
    print(f"Corpus GLEU score: {corpus_gleu:.4f}")
    print(f"Average Distinct-1: {avg_distinct_1:.4f}")
    print(f"Average Distinct-2: {avg_distinct_2:.4f}")

    # 评估dietetic类型问题
    avg_gleu, corpus_gleu, avg_distinct_1, avg_distinct_2 = evaluate_metrics(data_file, answer_file, output_file, type='dietetic')
    print(f"\nDietetic Type Results:")
    print(f"Average Sentence GLEU score: {avg_gleu:.4f}")
    print(f"Corpus GLEU score: {corpus_gleu:.4f}")
    print(f"Average Distinct-1: {avg_distinct_1:.4f}")
    print(f"Average Distinct-2: {avg_distinct_2:.4f}")

if __name__ == "__main__":
    main_qa()