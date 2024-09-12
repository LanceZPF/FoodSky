import re 
from tqdm import tqdm
import numpy as np
import os
import argparse
import pandas as pd
import json
import openai
import time
from rouge import Rouge
import jieba

DEFAULT_SYSTEM_PROMPT_QA = """You are a helpful assistant. 你是一个乐于助人的助手。"""
openai.api_key = 'XXX'

def gpt_output_qa(question):
    input_text = question
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": DEFAULT_SYSTEM_PROMPT_QA},
            {"role": "user", "content": input_text}
        ],
        seed=42
    )
    return response.choices[0].message.content

def segment_text(text):
    return ' '.join(jieba.cut(text))

def evaluate_rouge(data_file, output_file):
    with open(data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    rouge = Rouge()
    total_scores = {'rouge-1': 0, 'rouge-2': 0, 'rouge-l': 0}

    with open(output_file, 'a', encoding='utf-8') as f:
        for item in data:
            question = item['instruction'] + item['input']
            reference = item['output']
            
            response = gpt_output_qa(question)
            
            # 分词
            segmented_reference = segment_text(reference)
            segmented_response = segment_text(response)
            
            scores = rouge.get_scores(segmented_response, segmented_reference)[0]
            
            total_scores['rouge-1'] += scores['rouge-1']['f']
            total_scores['rouge-2'] += scores['rouge-2']['f']
            total_scores['rouge-l'] += scores['rouge-l']['f']
            
            f.write(f"Question: {question}\n")
            f.write(f"Reference: {reference}\n")
            f.write(f"Response: {response}\n")
            f.write(f"ROUGE-1: {scores['rouge-1']['f']:.4f}\n")
            f.write(f"ROUGE-2: {scores['rouge-2']['f']:.4f}\n")
            f.write(f"ROUGE-L: {scores['rouge-l']['f']:.4f}\n\n")
            
    avg_scores = {metric: score / len(data) for metric, score in total_scores.items()}
    
    with open(output_file, 'a', encoding='utf-8') as f:
        f.write(f"Average ROUGE Scores:\n")
        f.write(f"ROUGE-1: {avg_scores['rouge-1']:.4f}\n")
        f.write(f"ROUGE-2: {avg_scores['rouge-2']:.4f}\n")
        f.write(f"ROUGE-L: {avg_scores['rouge-l']:.4f}\n")
        
    return avg_scores

def main_qa():
    data_file = 'bleugleu/cookdiet_qa.json'
    output_file = 'bleugleu/rouge_results.txt'
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("ROUGE Results\n\n")
    
    avg_scores = evaluate_rouge(data_file, output_file)
    print(f"Average ROUGE Scores:")
    print(f"ROUGE-1: {avg_scores['rouge-1']:.4f}")
    print(f"ROUGE-2: {avg_scores['rouge-2']:.4f}")
    print(f"ROUGE-L: {avg_scores['rouge-l']:.4f}")

if __name__ == "__main__":
    main_qa()