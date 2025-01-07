import torch
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from PIL import Image
import json, os
from tqdm import tqdm
import requests
from io import BytesIO
from qwen_vl_utils import process_vision_info


def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image


def load_images(image_files):
    out = []
    for image_file in image_files:
        image = load_image(image_file)
        out.append(image)
    return out


class Yummly():
    def __init__(self, dataset_dir, partition="test_Spanish", input_type="im_t_ing"):
        self.dataset_dir = dataset_dir
        self.partition = partition
        self.data = json.load(open(os.path.join(dataset_dir, f"{partition}.json")))
        self.ids = list(range(len(self.data)))
        self.input_type = input_type

    def __len__(self):
        return len(self.ids)

    def get_sample(self, idx):
        sample = self.data[idx]
        if len(sample['images']) > 0:
            sample['image_path'] = os.path.join(self.dataset_dir, sample['images'][0])
        else:
            sample['image_path'] = None
        return sample

    def conversation(self, idx):
        sample = self.get_sample(idx)
        chat = {
            "id": f"{self.partition}_{idx}",
            "conversations": []
        }

        q = sample['conversations'][0]['value']
        t = sample['conversations'][1]['value']

        if "image_path" in sample and sample['image_path'] is not None:
            if os.path.isfile(sample['image_path']):
                chat['image'] = sample['image_path']

        chat["conversations"].append({
            "from": "human", 
            "value": q
        })

        chat["conversations"].append({
            "from": "gpt",
            "value": t
        })

        return chat


class QwenVLModel:
    def __init__(self, model_path) -> None:
        self.model_path = model_path
        self.max_new_tokens = 512
        self.build_model()
    
    def build_model(self):
        self.processor = AutoProcessor.from_pretrained(self.model_path)
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_path,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True
        ).eval()

    def step_batch(self, queries, image_files):
        images = load_images(image_files)
        messages = [
            {
                "role": "user", 
                "content": [
                    {"type": "image", "image": img} for img in images
                ] + [
                    {"type": "text", "text": query}
                ]
            }
            for query in queries
        ]
        
        texts = [
            self.processor.apply_chat_template(
                [msg], tokenize=False, add_generation_prompt=True
            )
            for msg in messages
        ]

        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        )
        inputs = inputs.to("cuda")

        generated_ids = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_texts = self.processor.batch_decode(
            generated_ids_trimmed, 
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )
        
        return output_texts


class QwenVLEvalModel:
    def __init__(self, args) -> None:
        self.args = args
        self.save_dir = os.path.join(self.args.save_dir, f"{args.model_type}_{args.eval_type}.json")
        
        self.recipe = Yummly(args.dataset_dir, partition=args.partition, input_type=self.args.eval_type)
        self.model = QwenVLModel(args.model_path)
        
        print("test partition:", args.partition)
        print("test dataset size:", len(self.recipe))
        print("save dir:", self.save_dir)

    def save_results(self, results, file_name):
        with open(file_name, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)

    def predict_loop_batch(self, batch_indices, batch_size):
        for i in tqdm(range(self.args.start, self.args.end, batch_size)):
            batch_indices = range(i, min(i + batch_size, self.args.end))
            chats = [self.recipe.conversation(idx) for idx in batch_indices]

            # Filter out already processed samples
            chats = [chat for chat in chats if chat['id'] not in self.done and "image" in chat]
            if not chats:
                continue

            image_files = [chat['image'] for chat in chats]
            queries = [chat['conversations'][0]['value'] for chat in chats]
            targets = [chat['conversations'][1]['value'] for chat in chats]

            try:
                outputs = self.model.step_batch(queries, image_files)
            except Exception as e:
                print(f"Error processing batch starting at index {i}: {e}")
                continue

            for chat, target, output in zip(chats, targets, outputs):
                self.outputs[chat['id']] = {
                    "id": chat['id'],
                    "gt": target,
                    "pred": output,
                    "q": chat['conversations'][0]['value']
                }

            if i % (10 * batch_size) == 0:
                self.save_results(self.outputs, self.save_dir)

        return self.outputs

    def predict_batch(self):
        self.outputs = {}
        self.done = []
        if self.args.end < 1:
            self.args.end = len(self.recipe)
        self.args.end = min(self.args.end, len(self.recipe))
        print(f"Processing from {self.args.start} to {self.args.end} index of dataset size: {len(self.recipe)}")

        if self.args.start > self.args.end:
            return

        if os.path.exists(self.save_dir):
            self.outputs = json.load(open(self.save_dir))
            self.done = list(self.outputs.keys())

        print("Already processed samples:", len(self.done))
        
        self.predict_loop_batch(range(self.args.start, self.args.end), self.args.batch_size)
        self.save_results(self.outputs, self.save_dir)


    def predict_loop(self):
        for i in tqdm(range(self.args.start, self.args.end)):
            chat = self.recipe.conversation(i)
            
            if chat['id'] in self.done:
                continue
                
            image_path = chat['image'] if "image" in chat.keys() else None
            
            if image_path is None:
                print("No image found. Skip this sample.")
                continue
            
            q = chat['conversations'][0]['value']
            t = chat['conversations'][1]['value']

            output = self.model.step(q, [image_path])
            
            self.outputs[chat['id']] = {
                "id": chat['id'],
                "gt": t,
                "pred": output,
                "q": q
            }
            
            if i % 10 == 0:
                self.save_results(self.outputs, self.save_dir)

        return self.outputs
    def predict(self):
        # Existing single-sample prediction
        self.outputs = {}
        self.done = []
        if self.args.end < 1:
            self.args.end = len(self.recipe)
        self.args.end = min(self.args.end, len(self.recipe))
        print(f"Processing from {self.args.start} to {self.args.end} index of dataset size: {len(self.recipe)}")

        if self.args.start > self.args.end:
            return
            
        if os.path.exists(self.save_dir):
            self.outputs = json.load(open(self.save_dir))
            self.done = list(self.outputs.keys())

        print("Already processed samples:", len(self.done))
        
        results = self.predict_loop()
        self.save_results(results, self.save_dir)
