import json
import random

all_outputs = []

json_path1 = "/aifs4su/gov/models/baichuan-moe-hf-128k-chat-guhao-HKyue1-dataset-output/HK_yue_qa_T0.7_N1024_S42_10000.json"
json_path2 = "/aifs4su/gov/models/baichuan-moe-hf-128k-chat-guhao-wikitext-dataset-output/wikitext_T0.7_N1024_S42_10000.json"
json_path3 = "./datasets/hf-llama-2-7b/alpaca_T0.7_N1024_S42_5000.json"

with open(json_path1, 'r') as f:
    dataset_for_eval = f.readlines()
for line in dataset_for_eval:
    json_data = json.loads(line)
    all_outputs.append(json_data)

with open(json_path2, 'r') as f:
    dataset_for_eval = f.readlines()
for line in dataset_for_eval:
    json_data = json.loads(line)
    all_outputs.append(json_data)

with open(json_path3, 'r') as f:
    dataset_for_eval = f.readlines()
for line in dataset_for_eval:
    json_data = json.loads(line)
    all_outputs.append(json_data)

random.shuffle(all_outputs)

with open('./datasets/hf-llama-2-7b/mix_wiki_alpaca_8000.json', 'w') as f:
    for item in all_outputs:
        f.write(json.dumps(item) + '\n')
