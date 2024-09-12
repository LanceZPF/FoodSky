import re
from tqdm import tqdm
import numpy as np
import os
import argparse
import pandas as pd
import json
import openai
import time
from nltk.translate.gleu_score import sentence_gleu
from nltk.translate.gleu_score import corpus_gleu
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

def distinct_n(text, n):
    tokens = list(jieba.cut(text))
    ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
    unique_ngrams = set(ngrams)
    distinct_n = len(unique_ngrams) / len(ngrams)
    return distinct_n

def evaluate_metrics(data_file, output_file):
    with open(data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    total_gleu = 0
    total_distinct_1 = 0
    total_distinct_2 = 0
    references = []
    responses = []

    with open(output_file, 'a', encoding='utf-8') as f:
        for item in data:
            question = item['instruction'] + item['input']
            reference = item['output']
            
            response = gpt_output_qa(question)
            
            references.append(jieba.lcut(reference))
            responses.append(jieba.lcut(response))
            
            gleu = sentence_gleu([reference], response)
            distinct_1 = distinct_n(response, 1)
            distinct_2 = distinct_n(response, 2)
            
            total_gleu += gleu
            total_distinct_1 += distinct_1
            total_distinct_2 += distinct_2
            
            f.write(f"Question: {question}\n")
            f.write(f"Reference: {reference}\n")
            f.write(f"Response: {response}\n")
            f.write(f"GLEU: {gleu:.4f}\n")
            f.write(f"Distinct-1: {distinct_1:.4f}\n")
            f.write(f"Distinct-2: {distinct_2:.4f}\n\n")
            
    avg_gleu = total_gleu / len(data)
    avg_distinct_1 = total_distinct_1 / len(data)
    avg_distinct_2 = total_distinct_2 / len(data)
    corpus_gleu_score = corpus_gleu(references, responses)
    
    with open(output_file, 'a', encoding='utf-8') as f:
        f.write(f"Average Sentence GLEU score: {avg_gleu:.4f}\n")
        f.write(f"Corpus GLEU score: {corpus_gleu_score:.4f}\n")
        f.write(f"Average Distinct-1: {avg_distinct_1:.4f}\n")
        f.write(f"Average Distinct-2: {avg_distinct_2:.4f}\n")
        
    return avg_gleu, corpus_gleu_score, avg_distinct_1, avg_distinct_2

def main_qa():
    data_file = 'bleugleu/cookdiet_qa.json'
    output_file = 'bleugleu/metrics_results.txt'
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("Evaluation Metrics Results\n\n")
    
    avg_gleu, corpus_gleu, avg_distinct_1, avg_distinct_2 = evaluate_metrics(data_file, output_file)
    print(f"Average Sentence GLEU score: {avg_gleu:.4f}")
    print(f"Corpus GLEU score: {corpus_gleu:.4f}")
    print(f"Average Distinct-1: {avg_distinct_1:.4f}")
    print(f"Average Distinct-2: {avg_distinct_2:.4f}")

if __name__ == "__main__":
    main_qa()