import openai
import json
import pandas as pd

# Set the OpenAI API key
openai.api_key = 'XXX'

def generate_answer(model, question):
    """Generate an answer using a specified GPT model."""
    response = openai.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": question}
        ],
        max_tokens=150,
        temperature=0.7,
        n=1,
        stop=None
    )
    return response.choices[0].message.content.strip()

def evaluate_with_gpt4(question, answer):
    """Evaluate an answer using GPT-4 based on a specific criterion."""
    evaluations = {}
    criteria = {
        "Logic": "Is this answer logically correct?",
        "Professional": "Does this answer demonstrate professional knowledge?",
        "Informative": "Is this answer informative enough?",
        "Fluent": "Is this answer written fluently?"
    }
    for criterion, prompt in criteria.items():
        eval_prompt = f"{question}\nAnswer: {answer}\n{prompt}"
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": eval_prompt}
            ],
            max_tokens=60,
            temperature=0.7,
            n=1,
            stop=None
        )
        evaluations[criterion] = response.choices[0].message.content.strip()
    return evaluations

def compare_evaluations(eval_35, eval_35_turbo):
   """Compare evaluations and return a binary result."""
   gpt35_score = 0
   gpt35_turbo_score = 0
   for criterion in eval_35.keys():
       if eval_35[criterion].lower().startswith('yes'):
           gpt35_score += 1
       if eval_35_turbo[criterion].lower().startswith('yes'):
           gpt35_turbo_score += 1
   if gpt35_score > gpt35_turbo_score:
       return "GPT-3.5"
   elif gpt35_score < gpt35_turbo_score:
       return "GPT-3.5 Turbo"
   else:
       return "Tie"

def main():
    questions = ["What is the impact of climate change on polar bears?"]  # Example question
    results = []

    for question in questions:
        answer_35 = generate_answer("gpt-3.5-turbo-0613", question)
        answer_35_turbo = generate_answer("gpt-3.5-turbo", question)
        eval_35 = evaluate_with_gpt4(question, answer_35)
        eval_35_turbo = evaluate_with_gpt4(question, answer_35_turbo)
        
        winner = compare_evaluations(eval_35, eval_35_turbo)
        result = {
            "Question": question,
            "Winner": winner
        }
        results.append(result)

    # Convert results to DataFrame and save as CSV or directly to a PDF
    df = pd.DataFrame(results)
    df.to_csv('evaluation_results.csv', index=False)
    print("Results saved to evaluation_results.csv.")

if __name__ == "__main__":
    main()
