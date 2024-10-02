import torch
from PIL import Image
from transformers import AutoModelForCausalLM, LlamaTokenizer
import argparse
import random
import json
from tqdm import tqdm
import os
#Define the argument parser
parser = argparse.ArgumentParser(description="Run your Python script with arguments")

# Add required arguments
parser.add_argument("--input_file", help="file to process")
parser.add_argument("--output_file", help="file to write results to")
# Parse arguments
args = parser.parse_args()

torch.manual_seed(1234)

cache_dir = "/media/vivek/c33fd89b-a307-4208-a045-64d021572535/kunal/CogAgent-VQA"
# parser = argparse.ArgumentParser()
# parser.add_argument("--quant", choices=[4], type=int, default=None, help='quantization bits')
# parser.add_argument("--from_pretrained", type=str, default="THUDM/cogagent-chat-hf", help='pretrained ckpt')
# parser.add_argument("--local_tokenizer", type=str, default="lmsys/vicuna-7b-v1.5", help='tokenizer path')
# parser.add_argument("--fp16", action="store_true")
# parser.add_argument("--bf16", action="store_true")
BASE_IMAGE_DIR = "/home/vivek/kunal/table-images-final/HybridQA_old/"

# args = parser.parse_args()
MODEL_PATH = "THUDM/cogagent-chat-hf"
TOKENIZER_PATH = "lmsys/vicuna-7b-v1.5"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

tokenizer = LlamaTokenizer.from_pretrained(TOKENIZER_PATH,cache_dir=cache_dir)

torch_type = torch.float16

print("========Use torch type as:{} with device:{}========\n\n".format(torch_type, DEVICE))

model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch_type,
        trust_remote_code=True,
        cache_dir=cache_dir,
device_map={"": "cuda:0"}
    ).eval()

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
    # print(val["prompt"][0])
    try:
            
            val["prompt"][1] = os.path.join(BASE_IMAGE_DIR,val["prompt"][1].split("/")[-1])
            image_path = val["prompt"][1]
            if image_path == "stop":
                break

            image = Image.open(image_path).convert('RGB')
            # history = []
            query = val["prompt"][0]
            if query == "clear":
                break
            input_by_model = model.build_conversation_input_ids(tokenizer, query=query, images=[image])
            inputs = {
                'input_ids': input_by_model['input_ids'].unsqueeze(0).to(DEVICE),
                'token_type_ids': input_by_model['token_type_ids'].unsqueeze(0).to(DEVICE),
                'attention_mask': input_by_model['attention_mask'].unsqueeze(0).to(DEVICE),
                'images': [[input_by_model['images'][0].to(DEVICE).to(torch_type)]],
            }
            if 'cross_images' in input_by_model and input_by_model['cross_images']:
                inputs['cross_images'] = [[input_by_model['cross_images'][0].to(DEVICE).to(torch_type)]]

            # add any transformers params here.
            gen_kwargs = {"max_length": 12000,
                        "temperature": 0.0,
                        "do_sample": False}
            with torch.no_grad():
                outputs = model.generate(**inputs, **gen_kwargs)
                outputs = outputs[:, inputs['input_ids'].shape[1]:]
                response = tokenizer.decode(outputs[0])
                response = response.split("</s>")[0]
                # print("\nCog:", response)
                result_dict = {}
                result_dict[key] = {"image":val["prompt"][1],"text":val["prompt"][0],"response":response,"gold_ans":val["gold_ans"]}
                with open(args.output_file, "a") as f:
                    f.write(json.dumps(result_dict) + '\n')
                # print(r   esult_dict)
                # history.append((query, response)
               
    except Exception as e:
        print(key,val)
        print(e)
        print("mkc")
        # errors[key] = (val,e)
    # break
    # print(response)
    
# with open(os.path.join(args.input_file.split("/qid")[0],"errors.json"),"w") as f:
#     (json.dump(errors,f,indent=4))

