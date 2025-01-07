import json
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer  
import os
# os.environ['CUDA_VISIBLE_DEVICES']='5'
import torch
from transformers import GenerationConfig
import argparse

class foodGPT():
    global DEFAULT_SYSTEM_PROMPT
    global TEMPLATE
    DEFAULT_SYSTEM_PROMPT = """You are a helpful assistant. 你是一个乐于助人的助手。"""

    TEMPLATE = (
        "[INST] <<SYS>>\n"
        "{system_prompt}\n"
        "<</SYS>>\n\n"
        "{instruction} [/INST]"
    )

    def __init__(self,
                max_tokens,
                gpus,
                base_model=None,
                tokenizer_path=None,
                only_cpu=False,
                load_in_8bit=False,
                with_prompt=True,
                interactive=False,
                ):
        self.base_model = base_model
        self.tokenizer_path = tokenizer_path
        self.max_tokens = max_tokens
        self.gpus = gpus
        self.only_cpu = only_cpu
        self.load_in_8bit = load_in_8bit
        self.with_prompt = with_prompt
        self.interactive = interactive
        
        if self.only_cpu is True:
            self.gpus = ""
        # os.environ["CUDA_VISIBLE_DEVICES"] = self.gpus
        self.device = torch.device("cuda:" + str(self.gpus) if torch.cuda.is_available() else "cpu")
        self.setup()

    def setup(self):
        # 设置生成配置
        # self.generation_config = GenerationConfig(
        #     temperature=0.3,
        #     top_k=40,
        #     top_p=0.9,
        #     do_sample=True,
        #     num_beams=1,
        #     repetition_penalty=1.2,
        #     max_new_tokens=400
        # )

        self.generation_config = GenerationConfig(
            temperature=0.6,  # 稍微提高随机性,以增加答案的多样性
            top_k=50,  # 从概率最高的 50 个标记中采样,以确保答案的相关性
            top_p=0.95,  # 从累积概率超过 0.95 的标记中采样,以进一步增加答案的多样性
            do_sample=True,  # 通过随机采样生成标记,以获得更自然的答案
            num_beams=5,  # 使用宽度为 5 的波束搜索,以在速度和质量之间取得平衡
            repetition_penalty=1.2,  # 略微增加重复惩罚,以减少答案中的重复
            max_new_tokens=self.max_tokens  # 将最大新标记数设为 100,以生成相对简洁的答案
        )

        load_type = torch.float16
        torch.cuda.set_device(self.device)
        if self.tokenizer_path is None:
            self.tokenizer_path = self.base_model
            
        # 加载tokenizer和模型    
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path, legacy=True)
        base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            torch_dtype=load_type,
            low_cpu_mem_usage=True,
            device_map='auto',
            load_in_8bit=self.load_in_8bit,
            trust_remote_code=True
        )

        model_vocab_size = base_model.get_input_embeddings().weight.size(0)
        tokenizer_vocab_size = len(self.tokenizer)
        print(f"Vocab of the base model: {model_vocab_size}")
        print(f"Vocab of the tokenizer: {tokenizer_vocab_size}")
        # assert model_vocab_size==tokenizer_vocab_size
        self.model = base_model
        if self.device==torch.device('cpu'):
            self.model.float()
        self.model.eval()

    def generate_prompt(self,instruction, system_prompt=DEFAULT_SYSTEM_PROMPT):
        # 生成prompt
        return TEMPLATE.format_map({'system_prompt': system_prompt, 'instruction': instruction})
    
    def generate_prompt2(self, instruction, input_text):
        return f"Instruction: {instruction}\nInput: {input_text}\nOutput:"
        
    def start_qa(self,raw_input_text):
        with torch.no_grad():

            messages = [
                {"role": "system", "content": "你是一个有用的助手"},
                {"role": "user", "content": raw_input_text}
            ]
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
            #print("model_inputs",model_inputs)
            #pred, history = model.chat(tokenizer, question, history=[])
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=512
                ) 
            #print("generated_ids",generated_ids)
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
                ]
            #print("generated_ids",generated_ids)
            answer_35_turbo = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            print("question:",raw_input_text)
            print("answer_35_turbo:",answer_35_turbo)
            return answer_35_turbo

            # if self.interactive:
            #     if len(raw_input_text.strip())==0:
            #         print("你是一个健康饮食助手。")
            #     if self.with_prompt:
            #         input_text = self.generate_prompt(instruction=raw_input_text)
            #     else:
            #         input_text = raw_input_text
                
            #     # 对输入进行编码    
            #     inputs = self.tokenizer(input_text,return_tensors="pt")
            #     # 生成回答
            #     generation_output = self.model.generate(
            #         input_ids = inputs["input_ids"].to(self.device),
            #         attention_mask = inputs['attention_mask'].to(self.device),
            #         eos_token_id=self.tokenizer.eos_token_id,
            #         pad_token_id=self.tokenizer.pad_token_id,
            #         generation_config = self.generation_config
            #     )
            #     s = generation_output[0]
            #     output = self.tokenizer.decode(s,skip_special_tokens=True)
            #     if self.with_prompt:
            #         response = output.split("[/INST]")[-1]
            #     else:
            #         response = output
                    
            #     print("Response: ",response)
            #     print("\n")
            #     return response
            # else:
            #     input_text = raw_input_text
            #     inputs = self.tokenizer(input_text, return_tensors="pt")
            #     generation_output = self.model.generate(
            #         input_ids=inputs["input_ids"].to(self.device),
            #         attention_mask=inputs['attention_mask'].to(self.device),
            #         eos_token_id=self.tokenizer.eos_token_id,
            #         pad_token_id=self.tokenizer.pad_token_id,
            #         generation_config=self.generation_config
            #     )
            #     output = self.tokenizer.decode(generation_output[0], skip_special_tokens=True)
            #     print("Response: ",output)
            #     print("\n")

            #     response = output
            #     return response

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_model', default='/mnt/workspace/zpf/FS/LLaMA-Factory/foodsky-7b-1229-500', type=str)
    parser.add_argument('--data_path', default='cookdiet_qa22.json', type=str)
    parser.add_argument('--output_path', default='output.csv', type=str)
    args = parser.parse_args()

    questions_path = args.data_path
    model_path = args.base_model

    # 读取问题
    with open(questions_path,'r',encoding='utf-8') as f:
        questions = json.load(f)

    maxlen = 0
    for i in questions:
        leninput = len(i['output'])
        if leninput > maxlen:
            maxlen = leninput

    print('max_length:')
    print(maxlen)

    # 初始化模型
    model = foodGPT(base_model = model_path, max_tokens=maxlen, gpus='0')

    for i, question in enumerate(questions):
        food_generate_answers = []
        print('+++++++++++++now question:', i, question)
        print('+++++++++++++foodsky:')
        # 生成回答
        answer_35 = model.start_qa(question['instruction'] + question['input'])

        food_generate_answers.append(answer_35)

        # print(food_generate_answers)

        # 保存回答
        df_food = pd.DataFrame(food_generate_answers)
        df_food.to_csv(args.output_path, mode='a', index=False)

if __name__ == "__main__":
    main()