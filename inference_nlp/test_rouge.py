import re 
from tqdm import tqdm
import numpy as np
import os
import argparse
import pandas as pd
import json
import time
from rouge import Rouge
import jieba

def segment_text(text):
    return ' '.join(jieba.cut(text))

def evaluate_rouge(data_file, answer_file, output_file, type=None):
    with open(data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 读取生成的答案文件
    with open(answer_file, 'r', encoding='utf-8') as f:
        answer_data = f.read()
    
    # 使用正则表达式，以 0 为分隔符拆分答案
    answers = re.split(r'\n0\n', answer_data)
        
    rouge = Rouge()
    total_scores = {'rouge-1': 0, 'rouge-2': 0, 'rouge-l': 0}
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
            
            # 分词
            segmented_reference = segment_text(reference)
            segmented_response = segment_text(response)
            
            scores = rouge.get_scores(segmented_response, segmented_reference)[0]
            
            total_scores['rouge-1'] += scores['rouge-1']['f']
            total_scores['rouge-2'] += scores['rouge-2']['f']
            total_scores['rouge-l'] += scores['rouge-l']['f']
            
    avg_scores = {metric: score / count for metric, score in total_scores.items()}
    
    with open(output_file, 'a', encoding='utf-8') as f:
        if type:
            f.write(f"\n{type.capitalize()} Type Results:\n")
        f.write(f"Average ROUGE Scores:\n")
        f.write(f"ROUGE-1: {avg_scores['rouge-1']:.4f}\n")
        f.write(f"ROUGE-2: {avg_scores['rouge-2']:.4f}\n")
        f.write(f"ROUGE-L: {avg_scores['rouge-l']:.4f}\n")
        
    return avg_scores

def main_qa():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_file', default='cookdiet_qa22.json', type=str)
    parser.add_argument('--answer_file', default='food_answer_1.csv', type=str)
    parser.add_argument('--output_file', default='output_rouge.txt', type=str)
    args = parser.parse_args()

    data_file = args.data_file
    answer_file = args.answer_file
    output_file = args.output_file
    
    with open(output_file, 'a', encoding='utf-8') as f:
        f.write("ROUGE Results\n\n")
    
    # 评估所有问题
    print("Overall Results:")
    avg_scores = evaluate_rouge(data_file, answer_file, output_file)
    print(f"Average ROUGE Scores:")
    print(f"ROUGE-1: {avg_scores['rouge-1']:.4f}")
    print(f"ROUGE-2: {avg_scores['rouge-2']:.4f}")
    print(f"ROUGE-L: {avg_scores['rouge-l']:.4f}")

    # 评估culinary类型问题
    print("\nCulinary Type Results:")
    avg_scores = evaluate_rouge(data_file, answer_file, output_file, type='culinary')
    print(f"Average ROUGE Scores:")
    print(f"ROUGE-1: {avg_scores['rouge-1']:.4f}")
    print(f"ROUGE-2: {avg_scores['rouge-2']:.4f}")
    print(f"ROUGE-L: {avg_scores['rouge-l']:.4f}")

    # 评估dietetic类型问题
    print("\nDietetic Type Results:")
    avg_scores = evaluate_rouge(data_file, answer_file, output_file, type='dietetic')
    print(f"Average ROUGE Scores:")
    print(f"ROUGE-1: {avg_scores['rouge-1']:.4f}")
    print(f"ROUGE-2: {avg_scores['rouge-2']:.4f}")
    print(f"ROUGE-L: {avg_scores['rouge-l']:.4f}")

if __name__ == "__main__":
    main_qa()