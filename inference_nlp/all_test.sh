# python save_inference.py --base_model /mnt/workspace/zpf/FS/LLaMA-Factory/foodsky-7b-1229-500 --data_path cookdiet_qa22.json --output_path newfood_answer.csv
python test_bleu.py --data_file cookdiet_qa22.json --answer_file newfood_answer.csv --output_file output_bleu.txt
python test_gleu.py --data_file cookdiet_qa22.json --answer_file newfood_answer.csv --output_file output_gleu.txt
python test_rouge.py --data_file cookdiet_qa22.json --answer_file newfood_answer.csv --output_file output_rouge.txt


# python save_inference.py --base_model /mnt/workspace/zpf/FS/LLaMA-Factory/foodsky-14b-1230-fn --data_path cookdiet_qa22.json --output_path newfood_answer-14b.csv
python test_bleu.py --data_file cookdiet_qa22.json --answer_file newfood_answer-14b.csv --output_file output_bleu.txt
python test_gleu.py --data_file cookdiet_qa22.json --answer_file newfood_answer-14b.csv --output_file output_gleu.txt
python test_rouge.py --data_file cookdiet_qa22.json --answer_file newfood_answer-14b.csv --output_file output_rouge.txt


# python save_inference.py --base_model /mnt/workspace/zpf/FS/LLaMA-Factory/foodsky-7b-1229-500 --data_path cookdiet_qa25.json --output_path newfood_answer-foodqa25-7b.csv 
# python save_inference.py --base_model /mnt/workspace/zpf/FS/LLaMA-Factory/foodsky-14b-1230-fn --data_path cookdiet_qa25.json --output_path newfood_answer-foodqa25-14b.csv

# python save_inference.py --base_model /mnt/workspace/zpf/.cache/Qwen2.5-7B-Instruct --data_path cookdiet_qa25.json --output_path qwen25_answer-foodqa25-7b.csv 
# python save_inference.py --base_model /mnt/workspace/zpf/.cache/Qwen2.5-14B-Instruct --data_path cookdiet_qa25.json --output_path qwen25_answer-foodqa25-14b.csv 