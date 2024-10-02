import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer
import argparse
import random
import json
from tqdm import tqdm
#Define the argument parser
parser = argparse.ArgumentParser(description="Run your Python script with arguments")
import os
# Add required arguments
parser.add_argument("--input_file", help="file to process")
parser.add_argument("--output_file", help="file to write results to")
# Parse arguments
args = parser.parse_args()

torch.manual_seed(1234)

cache_dir = "/media/vivek/c33fd89b-a307-4208-a045-64d021572535/kunal"
# parser = argparse.ArgumentParser()
# parser.add_argument("--quant", choices=[4], type=int, default=None, help='quantization bits')
# parser.add_argument("--from_pretrained", type=str, default="THUDM/cogagent-chat-hf", help='pretrained ckpt')
# parser.add_argument("--local_tokenizer", type=str, default="lmsys/vicuna-7b-v1.5", help='tokenizer path')
# parser.add_argument("--fp16", action="store_true")
# parser.add_argument("--bf16", action="store_true")
BASE_IMAGE_DIR = "/home/vivek/kunal/table-images-final/HybridQA_old"
# Define the argument parser

torch.set_grad_enabled(False)

# Initialize model and tokenizer
model = AutoModel.from_pretrained(
    'internlm/internlm-xcomposer2-4khd-7b', 
    torch_dtype=torch.bfloat16, 
    trust_remote_code=True,
    cache_dir=cache_dir,
    device_map="cuda"
).eval()
tokenizer = AutoTokenizer.from_pretrained(
    'internlm/internlm-xcomposer2-4khd-7b', 
    trust_remote_code=True,
    cache_dir=cache_dir,
    device_map="cuda"
)

print(torch.cuda.is_available())  # Should return True if GPU is available
print(torch.cuda.current_device())
print(torch.cuda.get_device_name(torch.cuda.current_device()))  # Should return the name of the GPU

# Read existing results and keep track of processed keys
processed_keys = set()
if os.path.exists(args.output_file):
    with open(args.output_file, 'r') as f:
        for line in f:
            result = json.loads(line.strip())
            processed_keys.update(result.keys())

with open(args.input_file, 'r') as f:
    prompts = json.load(f)

prompt_list = list(prompts.items())
random.shuffle(prompt_list)
prompts = dict(prompt_list)

errors = {}
for idx, (key, val) in tqdm(enumerate(prompts.items())):
    if key in processed_keys:
        print(f"Skipping {key} as it is already processed.")
        continue

    try:
        val["prompt"][1] = os.path.join(BASE_IMAGE_DIR, val["prompt"][1].split("/")[-1])
        query1 = "<ImageHere>" + val["prompt"][0]
        image = val["prompt"][1]
        
        with torch.cuda.amp.autocast():
            response, his = model.chat(
                tokenizer, 
                query=query1, 
                image=image, 
                hd_num=1, 
                history=[], 
                do_sample=False, 
                num_beams=3, 
                temperature=0
            )
        
        result_dict = {
            key: {
                "image": val["prompt"][1],
                "text": val["prompt"][0],
                "response": response,
                "gold_ans": val["gold_ans"]
            }
        }
        
        with open(args.output_file, "a") as f:
            f.write(json.dumps(result_dict) + '\n')
    
    except torch.cuda.OutOfMemoryError as e:
        print(f"CUDA OOM Error for {key}: {e}")
        errors[key] = {"value": val, "error": str(e)}
        
        # Clear the cache and move on to the next iteration
        torch.cuda.empty_cache()
    
    except Exception as e:
        print(f"Error for {key}: {e}")
        errors[key] = {"value": val, "error": str(e)}

# Optionally save the errors to a file
error_file = os.path.join(os.path.dirname(args.input_file), "errors.json")
with open(error_file, "w") as f:
    json.dump(errors, f, indent=4)
