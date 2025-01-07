import json, os
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import DataLoader, Dataset

from .dataset import Yummly

class Qwen():
    def __init__(self, args) -> None:
        self.args = args
        self.save_dir = os.path.join(self.args.save_dir, f"{args.model_type}_{args.eval_type}_{args.partition}.json")
        self.recipe = Yummly(args.dataset_dir, partition=args.partition)
        print("Results will be saved at:", self.save_dir)

        kwargs = {"device_map": "auto"}

        print(kwargs)
        
        self.model = AutoModelForCausalLM.from_pretrained(args.model_path, trust_remote_code=True, **kwargs).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
        
        print("test partition:", args.partition)
        print("test dataset size:", len(self.recipe))
        print("save dir:", self.save_dir)

    def save_results(self, results, file_name):
        with open(file_name, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)

    def predict_batch(self, batch):
        outputs = []

        # print(list(batch[0]))
        
        batch2 = self.tokenizer(list(batch[0]), padding=True, truncation=True, return_tensors="pt").to(self.model.device)
        # batch2 = self.tokenizer(list(batch[0]), padding=True, padding_side='left', truncation=True, return_tensors="pt").to(self.model.device)
        # batch2 = self.tokenizer(list(batch[0]), return_tensors="pt").to(self.model.device)
        input_len = batch2['input_ids'].shape[1]

        # with torch.inference_mode():
        output_ids = self.model.generate(
            **batch2,
            max_new_tokens=self.args.max_length
        )

        for i in range(output_ids.shape[0]):
            output = self.tokenizer.batch_decode(output_ids[i][input_len:].unsqueeze(0), clean_up_tokenization_spaces=False, skip_special_tokens=True)
            outputs.append({
                "id": batch[-1][i],
                "gt": batch[1][i],
                "pred": output[0].split("\n\n")[-1],
                "prompt": batch[0][i]
            })

        return outputs

    def predict(self):
        self.outputs = []
        self.done = []
        if os.path.exists(self.save_dir):
            self.outputs = json.load(open(self.save_dir))
            self.done = [o['id'] for o in self.outputs]
        test_dataloader = DataLoader(self.recipe, batch_size=self.args.batch_size, shuffle=False)
        
        batch_id = 0
        print("Number of already processed files:", len(self.done))

        for batch in tqdm(test_dataloader):
            batch_id += 1
            ids = batch[-1]
            done_counter = 0
            for id in ids:
                if id in self.done:
                    done_counter += 1
            if done_counter == len(ids):  # skip if whole batch is already done
                continue

            outputs = self.predict_batch(batch)
            
            for o in outputs:
                self.outputs.append(o)

            if batch_id % 10 == 0:
                self.save_results(self.outputs, self.save_dir)
        
        self.save_results(self.outputs, self.save_dir)
