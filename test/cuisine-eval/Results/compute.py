import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import json, sys, torch
from metrics import compute_metrics, compute_metrics_hf
from metrics import save_results

# ===================================================
# ===================================================

# type = "text_foodsky"
type = "foodsky_test_Spanish"
dir = "/mnt/workspace/zpf/FS/LLaVA-Chef-main/LLMs/results"
results = []

files = os.listdir(dir)
for fi in files:
    file = os.path.join(dir, f"{fi}")
    if (file.endswith(".json") and type in fi):
        print(f'processing------{type}')
        try:
            result = compute_metrics(file)
            result_hf = compute_metrics_hf(file)
            result.update(result_hf)
            result["method"] = fi
            results.append(result)
        except Exception as e:
            print(e)
            continue

save_results(results, f"results_qwen_{type}.json")

'''
# ===================================================
# ===================================================

type = "ing_i_t__instruct"
dir = f"results1k/{type}/"
results = []

files = os.listdir (dir)
for fi in files:
    file = dir + f"{fi}"
    if (file.endswith (".json")):
        if (file.endswith (".json") and "llavachef_" in fi):
            result = compute_metrics (file)
            result_hf = compute_metrics_hf (file)
            result.update (result_hf)
            result["method"] = fi
            results.append (result)
save_results (results, f"results1k/results1k_llavachefv1_{type}.json")

# ===================================================
# ===================================================


'''