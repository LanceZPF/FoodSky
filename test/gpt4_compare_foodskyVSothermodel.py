import openai
import json
import pandas as pd
# from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM  # 0417 for run Qwen-7b-chat
import os
from openai import OpenAI
# import torch
# from transformers import GenerationConfig
# from transformers import BitsAndBytesConfig
# from peft import  PeftModel
# import sys
import re

# Set the OpenAI API key
openai.api_key = 'XXX'

def evaluate_with_gpt4(question, answer, ground_truth):
    """Evaluate an answer using GPT-4 based on a specific criterion."""
    evaluations = {}
    criteria = {
        "Logic": "回答是否足够凝练简短且正确？请就回答的逻辑正确性给一个0-100之间的数字评分：",
        "Professional": "回答是否足够凝练简短且专业？请就回答的知识专业性给一个0-100之间的数字评分：",
        "Informative": "回答是否足够凝练简短且言之有物？请就回答的信息密度给一个0-100之间的数字评分：",
        "Fluent": "回答是否足够凝练简短且流程？请就回答的表达流畅度给一个0-100之间的数字评分：?"
    }
    for criterion, prompt in criteria.items():
        eval_prompt = f"问题是：{question}\n参考答案是: {ground_truth}\n模型回答是: {answer}\n{prompt}"
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": """You are a helpful assistant. 你是一个乐于助人的助手，请你作为裁判，对特定问题的回答好坏直接给出评分。"""},
                {"role": "user", "content": eval_prompt}
            ],
            max_tokens=100,
            temperature=0.7,
            n=1,
            stop=None
        )
        evaluations[criterion] = response.choices[0].message.content.strip()
    print(evaluations)
    return evaluations

def compare_evaluations(eval_A, eval_B, win_counts, tie_counts, lose_counts):
    """Compare evaluations and return the winner based on numeric scores."""
    ourA_total_score = 0
    B_total_score = 0
    
    for criterion in eval_A.keys():
        A_score = int(re.findall(r'\d+', eval_A[criterion])[0])
        B_score = int(re.findall(r'\d+', eval_B[criterion])[0])
        
        if A_score > B_score:
            win_counts[criterion] += 1
        elif A_score < B_score:
            lose_counts[criterion] += 1
        else:
            tie_counts[criterion] += 1
        
        ourA_total_score += A_score
        B_total_score += B_score
    
    if ourA_total_score > B_total_score:
        return "We win!"
    elif ourA_total_score < B_total_score:
        return "We lose." 
    else:
        return "Tie~"
   
def main():
    questions_path = 'cookdiet_qa.json'
    food_answer_path = 'foodsky-food_answer.csv'  # ferry
    other_answer_path = "qwen-food_answer.csv"  # 11111111111
    with open(questions_path,'r',encoding='utf-8') as f:
        questions = json.load(f)

    # 读取两个 CSV 文件
    df_food = pd.read_csv(food_answer_path, header=None)
    df_other = pd.read_csv(other_answer_path, header=None)

    # 初始化计数器
    win_counts = {criterion: 0 for criterion in ["Logic", "Professional", "Informative", "Fluent"]}
    tie_counts = {criterion: 0 for criterion in ["Logic", "Professional", "Informative", "Fluent"]}
    lose_counts = {criterion: 0 for criterion in ["Logic", "Professional", "Informative", "Fluent"]}

    for i,question in enumerate(questions):
        print('------------', i, question)

        results = []

        answer_i = i*2 + 1
        answer_food = df_food.iloc[answer_i, 0]
        answer_other = df_other.iloc[answer_i, 0]

        eval_A = evaluate_with_gpt4(question['instruction']+question['input'], answer_food, question['output'])
        eval_B = evaluate_with_gpt4(question['instruction']+question['input'], answer_other, question['output'])
        
        winner = compare_evaluations(eval_A, eval_B, win_counts, tie_counts, lose_counts)

        result = {
            "Question": question['instruction'],
            "Winner": winner,
            "Logic Score A": eval_A["Logic"],
            "Professional Score A": eval_A["Professional"],
            "Informative Score A": eval_A["Informative"],
            "Fluent Score A": eval_A["Fluent"],
            "Logic Score B": eval_B["Logic"],
            "Professional Score B": eval_B["Professional"],
            "Informative Score B": eval_B["Informative"],
            "Fluent Score B": eval_B["Fluent"]
        }

        results.append(result)
        # Convert results to DataFrame and save as CSV or directly to a PDF
        df = pd.DataFrame(results)

        df.to_csv('evaluation_results_Qwen-7B-Chat.csv', mode='a', index=False)
        print("Results saved to evaluation_results.csv.")

    # 计算并打印获胜百分比
    total_counts = {criterion: win_counts[criterion] + tie_counts[criterion] + lose_counts[criterion] 
                    for criterion in win_counts.keys()}
    win_percentages = {criterion: win_counts[criterion] / total_counts[criterion] * 100 
                       for criterion in win_counts.keys()}
    
    print("Win counts:", win_counts)
    print("Tie counts:", tie_counts) 
    print("Lose counts:", lose_counts)
    print("Win percentages:", win_percentages)

if __name__ == "__main__":
    main()