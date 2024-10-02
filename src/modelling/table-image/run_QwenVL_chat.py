from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
import torch
import argparse
import json
import random
from tqdm import tqdm
import os
#Define the argument parser
parser = argparse.ArgumentParser(description="Run your Python script with arguments")

dne_ids = []
# Add required arguments
parser.add_argument("--input_file", help="file to process")
parser.add_argument("--output_file", help="file to write results to")
# Parse arguments
args = parser.parse_args()

torch.manual_seed(1234)
cache_dir = "/home/vivek/kunal/models_cache/"
# Note: The default behavior now has injection attack prevention off.
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True, cache_dir=cache_dir)

# use bf16
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", device_map="auto", trust_remote_code=True, bf16=True).eval()
# use fp16
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", device_map="auto", trust_remote_code=True, fp16=True).eval()
# use cpu only
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", device_map="cpu", trust_remote_code=True).eval()
# use cuda device
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", device_map={"": "cuda:0"}, trust_remote_code=True,cache_dir=cache_dir).eval()
BASE_IMAGE_DIR = "/home/vivek/kunal/table-images-final/HybridQA_old"
# Specify hyperparameters for generation (No need to do this if you are using transformers>=4.32.0)
model.generation_config = GenerationConfig.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True)

# 1st dialogue turn
print(torch.cuda.is_available())  # Should return True if GPU is available
print(torch.cuda.current_device())
print(torch.cuda.get_device_name(torch.cuda.current_device()))  # Should return the name of the GPU

with open(args.input_file,'r') as f:
    prompts = json.load(f)
prompt_list = list(prompts.items())    
random.shuffle(prompt_list)
prompts = dict(prompt_list)
# print(prompts)
errors = {}
for idx,(key,val) in tqdm(enumerate(prompts.items())):
    # print(key)
    # print(val["image"])
    try:
       # TODO: Change code format for not hyb
        val["prompt"][1] = os.path.join(BASE_IMAGE_DIR,val["prompt"][1].split("/")[-1])
        # table_image_path = val["image"]
        # if not os.path.exists(table_image_path):
        #     print("mkc",table_image_path)
        #     dne_ids.append(((table_image_path.split("/")[-1]).split(".jpg")[0]).replace("_","/"))
        query = tokenizer.from_list_format([
            {'image': val["prompt"][1]},
            {'text': val["prompt"][0]},
        ])
        response, history = model.chat(tokenizer, query=query, history=None)
        result_dict = {}
        result_dict[key] = {"image":val["prompt"][1],"text":val["prompt"][0],"response":response,"gold_ans":val["gold_ans"]}
        with open(args.output_file, "a") as f:
            f.write(json.dumps(result_dict) + '\n')
    except Exception as e:
        # print(key,val)
        print(e)
        # errors[key] = (val,e)
        # print("mkc")
    # break
    # print(response)
# with open(os.path.join(args.input_file.split("/qid")[0],"errors.json"),"w") as f:
#     (json.dump(errors,f,indent=4))
# print(dne_ids)
