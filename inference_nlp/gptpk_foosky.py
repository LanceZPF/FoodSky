import json
import pandas as pd
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM  # 0417 for run Qwen-7b-chat
import os
import torch
from transformers import GenerationConfig
from transformers import BitsAndBytesConfig
from peft import  PeftModel
import sys

import argparse
from transformers import AutoModelForCausalLM, LlamaForCausalLM, LlamaTokenizer

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
                gpus,
                base_model=None,
                lora_model=None,
                tokenizer_path=None,
                predictions_file = './predictions.json',
                user_inputs = None,
                alpha = 1.0,
                system_prompt=DEFAULT_SYSTEM_PROMPT,
                negative_prompt=None,
                guidance_scale=1.0,
                draft_k=-1,
                draft_base_model=None,
                draft_lora_model=None,
                only_cpu=False,
                speculative_sampling=False,
                draft_model_load_in_8bit=False,
                draft_model_load_in_4bit=False,
                load_in_8bit=False,
                load_in_4bit=False,
                use_flash_attention_2=False,
                use_ntk=False,
                with_prompt=True,
                interactive=True,
                use_vllm=True,
                ):
        #init parameter
        self.base_model = base_model
        self.lora_model = lora_model
        self.tokenizer_path = tokenizer_path
        self.predictions_file = predictions_file
        self.gpus = gpus
        self.user_inputs = user_inputs
        self.alpha = alpha 
        self.system_prompt = system_prompt
        self.negative_prompt = negative_prompt
        self.guidance_scale = guidance_scale
        self.draft_k = draft_k
        self.draft_base_model = draft_base_model
        self.draft_lora_model = draft_lora_model
        self.only_cpu = only_cpu
        self.speculative_sampling = speculative_sampling
        self.draft_model_load_in_8bit = draft_model_load_in_8bit
        self.draft_model_load_in_4bit = draft_model_load_in_4bit
        self.load_in_8bit = load_in_8bit
        self.load_in_4bit = load_in_4bit
        self.use_flash_attention_2 = use_flash_attention_2
        self.use_ntk = use_ntk
        self.with_prompt = with_prompt
        self.interactive = interactive
        self.use_vllm = use_vllm
        if self.guidance_scale > 1:
            try:
                from transformers.generation import UnbatchedClassifierFreeGuidanceLogitsProcessor
                self.UnbatchedClassifierFreeGuidanceLogitsProcessor = UnbatchedClassifierFreeGuidanceLogitsProcessor
            except ImportError:
                raise ImportError("Please install the latest transformers (commit equal or later than d533465) to enable CFG sampling.")

        if self.use_vllm:
            from vllm import LLM, SamplingParams
            self.LLM=LLM
            self.SamplingParams=SamplingParams
            if self.lora_model is not None:
                raise ValueError("vLLM currently does not support LoRA, please merge the LoRA weights to the base model.")
            if self.load_in_8bit or self.load_in_4bit:
                raise ValueError("vLLM currently does not support quantization, please use fp16 (default) or unuse --use_vllm.")
            if self.only_cpu:
                raise ValueError("vLLM requires GPUs with compute capability not less than 7.0. If you want to run only on CPU, please unuse --use_vllm.")
            if self.guidance_scale > 1:
                raise ValueError("guidance_scale > 1, but vLLM does not support CFG sampling. Please unset guidance_scale. ")
            if self.speculative_sampling:
                raise ValueError("speculative_sampling is set, but vLLM does not support speculative sampling. Please unset speculative_sampling. ")
        if self.load_in_8bit and self.load_in_4bit:
            raise ValueError("Only one quantization method can be chosen for inference. Please check your arguments")
        if self.only_cpu is True:
            self.gpus = ""
            if self.load_in_8bit or self.load_in_4bit:
                raise ValueError("Quantization is unavailable on CPU.")
        os.environ["CUDA_VISIBLE_DEVICES"] = self.gpus
        self.device = torch.device("cuda:" + str(self.gpus) if torch.cuda.is_available() else "cpu")
        self.setup()

    def setup(self):
        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        sys.path.append(parent_dir)
        if not self.only_cpu:
            if self.use_flash_attention_2:
                from flash_attn_patch_for_inference import replace_llama_attn_with_flash_attn
                replace_llama_attn_with_flash_attn()
            else:
                from attn_and_long_ctx_patches import apply_attention_patch
                apply_attention_patch(use_memory_efficient_attention=True)
        if self.use_ntk:
            from attn_and_long_ctx_patches import apply_ntk_scaling_patch
            apply_ntk_scaling_patch(self.alpha)
        if self.speculative_sampling:
            if self.draft_base_model == None:
                raise ValueError("Speculative sampling requires a draft model. Please specify the draft model.")
            if self.draft_model_load_in_8bit and self.draft_model_load_in_4bit:
                raise ValueError("Only one quantization method can be chosen for inference. Please check your arguments")
            from speculative_sample import speculative_sample
        if self.use_vllm:
            self.generation_config = dict(
                temperature=0.2,
                top_k=40,
                top_p=0.9,
                max_tokens=400,
                presence_penalty=1.0,
            )
        else:
            self.generation_config = GenerationConfig(
                temperature=0.2,
                top_k=40,
                top_p=0.9,
                do_sample=True,
                num_beams=1,
                repetition_penalty=1.1,
                max_new_tokens=400
            )
        sample_data = ["为什么要减少污染，保护环境？"]
        load_type = torch.float16
        torch.cuda.set_device(self.device)
        if self.tokenizer_path is None:
            self.tokenizer_path = self.lora_model
            if self.lora_model is None:
                self.tokenizer_path = self.base_model
        if self.use_vllm:
            self.model = self.LLM(model = self.base_model,
                tokenizer = self.tokenizer_path,
                gpu_memory_utilization = 0.9,
                tokenizer_mode = 'auto',
                enforce_eager = 'True',
                swap_space = 2,
                tensor_parallel_size=1)
            self.tokenizer = LlamaTokenizer.from_pretrained(self.tokenizer_path, legacy=True)
        else:
            
            self.tokenizer = LlamaTokenizer.from_pretrained(self.tokenizer_path, legacy=True)
            if self.load_in_4bit or self.load_in_8bit:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=self.load_in_4bit,
                    load_in_8bit=self.load_in_8bit,
                    bnb_4bit_compute_dtype=load_type,
                )
            base_model = AutoModelForCausalLM.from_pretrained(
                self.base_model,
                torch_dtype=load_type,
                low_cpu_mem_usage=True,
                device_map='auto',
                load_in_4bit=self.load_in_4bit,
                load_in_8bit=self.load_in_8bit,
                quantization_config=quantization_config if (self.load_in_4bit or self.load_in_8bit) else None,
                trust_remote_code=True
            )
            if self.speculative_sampling:
                if self.load_in_4bit or self.load_in_8bit:
                    draft_quantization_config = BitsAndBytesConfig(
                        load_in_4bit=self.draft_model_load_in_4bit,
                        load_in_8bit=self.draft_model_load_in_8bit,
                        bnb_4bit_compute_dtype=load_type,
                    )
                draft_base_model = LlamaForCausalLM.from_pretrained(
                    self.draft_base_model,
                    torch_dtype=load_type,
                    low_cpu_mem_usage=True,
                    device_map='auto',
                    load_in_4bit=self.draft_model_load_in_4bit,
                    load_in_8bit=self.draft_model_load_in_8bit,
                    quantization_config=draft_quantization_config if (self.draft_model_load_in_4bit or self.draft_model_load_in_8bit) else None
                )
            model_vocab_size = base_model.get_input_embeddings().weight.size(0)
            tokenizer_vocab_size = len(self.tokenizer)
            print(f"Vocab of the base model: {model_vocab_size}")
            print(f"Vocab of the tokenizer: {tokenizer_vocab_size}")
            if model_vocab_size!=tokenizer_vocab_size:
                print("Resize model embeddings to fit tokenizer")
                base_model.resize_token_embeddings(tokenizer_vocab_size)
            if self.speculative_sampling:
                draft_model_vocab_size = draft_base_model.get_input_embeddings().weight.size(0)
                print(f"Vocab of the draft base model: {draft_model_vocab_size}")
                if draft_model_vocab_size!=tokenizer_vocab_size:
                    print("Resize draft model embeddings to fit tokenizer")
                    draft_base_model.resize_token_embeddings(tokenizer_vocab_size)
            if self.lora_model is not None:
                print("loading peft model")
                self.model = PeftModel.from_pretrained(base_model, self.lora_model,torch_dtype=load_type,device_map='auto',).half()
            else:
                self.model = base_model
            if self.speculative_sampling:
                if self.draft_lora_model is not None:
                    print("loading peft draft model")
                    draft_model = PeftModel.from_pretrained(draft_base_model, self.draft_lora_model,torch_dtype=load_type,device_map='auto',).half()
                else:
                    draft_model = draft_base_model

            if self.device==torch.device('cpu'):
                self.model.float()
            self.model.eval()
            if self.speculative_sampling:
                if self.device==torch.device('cpu'):
                    draft_model.float()
                draft_model.eval()

        examples = sample_data
    

    def generate_prompt(self,instruction, system_prompt=DEFAULT_SYSTEM_PROMPT):
        return TEMPLATE.format_map({'instruction': instruction,'system_prompt': system_prompt})
    def start_qa(self,raw_input_text):
        with torch.no_grad():
            if self.interactive:
                '''
                print("Start inference with instruction mode.")

                print('='*85)
                print("+ 该模式下仅支持单轮问答，无多轮对话能力。\n"
                    "+ 如要进行多轮对话，请使用llama.cpp或本项目中的gradio_demo.py。")
                print('-'*85)
                print("+ This mode only supports single-turn QA.\n"
                    "+ If you want to experience multi-turn dialogue, please use llama.cpp or gradio_demo.py.")
                print('='*85)
                '''
                #raw_input_text = self.user_inputs
                #while True:
                #raw_input_text是输入值，靠input输入
                if len(raw_input_text.strip())==0:
                    print("我是一个健康饮食助手，请向我提问。")
                if self.with_prompt:
                    input_text = self.generate_prompt(instruction=raw_input_text, system_prompt=self.system_prompt)
                    negative_text = None if self.negative_prompt is None \
                        else self.generate_prompt(instruction=raw_input_text, system_prompt=self.negative_prompt)
                else:
                    input_text = raw_input_text
                    negative_text = self.negative_prompt

                if self.use_vllm:
                    output = self.model.generate([input_text], self.SamplingParams(**self.generation_config), use_tqdm=False)
                    response = output[0].outputs[0].text
                else:
                    inputs = self.tokenizer(input_text,return_tensors="pt")  #add_special_tokens=False ?
                    if self.guidance_scale ==1:
                        if not self.speculative_sampling:
                            generation_output = self.model.generate(
                                input_ids = inputs["input_ids"].to(self.device),
                                attention_mask = inputs['attention_mask'].to(self.device),
                                eos_token_id=self.tokenizer.eos_token_id,
                                pad_token_id=self.tokenizer.pad_token_id,
                                generation_config = self.generation_config
                            )
                        else: # enable speculative sampling
                            generation_output = speculative_sample(
                                input_ids=inputs["input_ids"].to(self.device),
                                target_model=self.model,
                                draft_model=draft_model,
                                draft_k=self.draft_k,
                                generation_config=self.generation_config,
                                eos_token_id=self.tokenizer.eos_token_id,
                                pad_token_id=self.tokenizer.pad_token_id,
                            )
                    else: # enable CFG sampling
                        if negative_text is None:
                            negative_prompt_ids = None
                            negative_prompt_attention_mask = None
                        else:
                            negative_inputs = self.tokenizer(negative_text,return_tensors="pt")
                            negative_prompt_ids = negative_inputs["input_ids"].to(self.device)
                            negative_prompt_attention_mask = negative_inputs["attention_mask"].to(self.device)
                        if not self.speculative_sampling:
                            generation_output = self.model.generate(
                                input_ids = inputs["input_ids"].to(self.device),
                                attention_mask = inputs['attention_mask'].to(self.device),
                                eos_token_id=self.tokenizer.eos_token_id,
                                pad_token_id=self.tokenizer.pad_token_id,
                                generation_config = self.generation_config,
                                guidance_scale = self.guidance_scale,
                                negative_prompt_ids = negative_prompt_ids,
                                negative_prompt_attention_mask = negative_prompt_attention_mask
                            )
                        else: # enable speculative sampling
                            generation_output = speculative_sample(
                                input_ids=inputs["input_ids"].to(self.device),
                                target_model=self.model,
                                draft_model=draft_model,
                                draft_k=self.draft_k,
                                generation_config=self.generation_config,
                                eos_token_id=self.tokenizer.eos_token_id,
                                pad_token_id=self.tokenizer.pad_token_id,
                                guidance_scale=self.guidance_scale,
                                negative_prompt_ids=negative_prompt_ids,
                                negative_prompt_attention_mask=negative_prompt_attention_mask,
                            )
                    s = generation_output[0]
                    output = self.tokenizer.decode(s,skip_special_tokens=True)
                    if self.with_prompt:
                        response = output.split("[/INST]")[-1].strip()
                    else:
                        response = output
                #response就是回答的输出
                print("Response: ",response)
                print("\n")
                return response
            else:
                print("Start inference.")
                results = []
                if self.use_vllm:
                    if self.with_prompt is True:
                        inputs = [self.generate_prompt(example, system_prompt=self.system_prompt) for example in examples]
                    else:
                        inputs = examples
                    outputs = self.model.generate(inputs, SamplingParams(**self.generation_config))

                    for index, (example, output) in enumerate(zip(examples, outputs)):
                        response = output.outputs[0].text

                        print(f"======={index}=======")
                        print(f"Input: {example}\n")
                        print(f"Output: {response}\n")

                        results.append({"Input":example,"Output":response})

                else:
                    for index, example in enumerate(examples):
                        if self.with_prompt:
                            input_text = self.generate_prompt(instruction=example, system_prompt=self.system_prompt)
                            negative_text = None if self.negative_prompt is None else \
                                self.generate_prompt(instruction=example, system_prompt=self.negative_prompt)
                        else:
                            input_text = example
                            negative_text = self.negative_prompt
                        inputs = self.tokenizer(input_text,return_tensors="pt")  #add_special_tokens=False ?
                        if self.guidance_scale == 1:
                            if not self.speculative_sampling:
                                generation_output = self.model.generate(
                                    input_ids = inputs["input_ids"].to(self.device),
                                    attention_mask = inputs['attention_mask'].to(self.device),
                                    eos_token_id=self.tokenizer.eos_token_id,
                                    pad_token_id=self.tokenizer.pad_token_id,
                                    generation_config = self.generation_config
                                )
                            else: # enable speculative sampling
                                generation_output = speculative_sample(
                                    input_ids=inputs["input_ids"].to(self.device),
                                    target_model=self.model,
                                    draft_model=draft_model,
                                    draft_k=self.draft_k,
                                    generation_config=self.generation_config,
                                    eos_token_id=self.tokenizer.eos_token_id,
                                    pad_token_id=self.tokenizer.pad_token_id,
                                )
                        else: # enable CFG sampling
                            if negative_text is None:
                                negative_prompt_ids = None
                                negative_prompt_attention_mask = None
                            else:
                                negative_inputs = self.tokenizer(negative_text,return_tensors="pt")
                                negative_prompt_ids = negative_inputs["input_ids"].to(self.device)
                                negative_prompt_attention_mask = negative_inputs["attention_mask"].to(self.device)
                            if not self.speculative_sampling:
                                generation_output = self.model.generate(
                                    input_ids = inputs["input_ids"].to(self.device),
                                    attention_mask = inputs['attention_mask'].to(self.device),
                                    eos_token_id=self.tokenizer.eos_token_id,
                                    pad_token_id=self.tokenizer.pad_token_id,
                                    generation_config = self.generation_config,
                                    guidance_scale = self.guidance_scale,
                                    negative_prompt_ids = negative_prompt_ids,
                                    negative_prompt_attention_mask = negative_prompt_attention_mask
                                )
                            else: # enable speculative sampling
                                generation_output = speculative_sample(
                                    input_ids=inputs["input_ids"].to(self.device),
                                    target_model=self.model,
                                    draft_model=draft_model,
                                    draft_k=self.draft_k,
                                    generation_config=self.generation_config,
                                    eos_token_id=self.tokenizer.eos_token_id,
                                    pad_token_id=self.tokenizer.pad_token_id,
                                    guidance_scale=self.guidance_scale,
                                    negative_prompt_ids=negative_prompt_ids,
                                    negative_prompt_attention_mask=negative_prompt_attention_mask,
                                )
                        s = generation_output[0]
                        output = self.tokenizer.decode(s,skip_special_tokens=True)
                        if self.with_prompt:
                            response = output.split("[/INST]")[1].strip()
                        else:
                            response = output
                        print(f"======={index}=======")
                        print(f"Input: {example}\n")
                        print(f"Output: {response}\n")

                        results.append({"Input":input_text,"Output":response})

                dirname = os.path.dirname(self.predictions_file)
                os.makedirs(dirname,exist_ok=True)
                with open(self.predictions_file,'w') as f:
                    json.dump(results,f,ensure_ascii=False,indent=2)
                if self.use_vllm:
                    with open(dirname+'/generation_config.json','w') as f:
                        json.dump(self.generation_config,f,ensure_ascii=False,indent=2)
                else:
                    self.generation_config.save_pretrained('./')



def main():
    questions_path = '/home/meish/Chinese-LLaMA-Alpaca-2-main/scripts/inference/cookdiet_qa.json'
    model_path = '/home/meish/Chinese-LLaMA-Alpaca-2.1/Chinese-LLaMA-Alpaca-food/FoodGPT-T6G-7B'  # ferry

    with open(questions_path,'r',encoding='utf-8') as f:
        questions = json.load(f)

    model = foodGPT(base_model = model_path, gpus='0')

    for i, question in enumerate(questions):

        results = []
        food_generate_answers = []

        # ferry
        print('+++++++++++++now question:', i, question)
        print('+++++++++++++foodsky:')
        answer_35 = model.start_qa(question['instruction']+ question['input'])

        food_generate_answers.append(answer_35)


        df_food = pd.DataFrame(food_generate_answers)
        df_food.to_csv('food_answer.csv', mode='a', index=False)

if __name__ == "__main__":
    main()
