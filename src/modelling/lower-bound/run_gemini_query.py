api_keys = [""]
kunal_api_keys= []
import asyncio
import google.generativeai as genai
from PIL import Image
import copy
import argparse
import json
import random
import json
import time
import os
from tqdm.asyncio import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Generate text from images in parallel")
    parser.add_argument("--input_file", help="file to process")
    parser.add_argument("--output_file", help="file to write results to")
    # a true flag to indicate whether to use the GPU
    parser.add_argument("--mm" , action="store_true", help="Use images_also", default=False)
    parser.add_argument("--keyno", help="API key number to use", default=1, type=int)
    parser.add_argument("--is_array", default=False)
    return parser.parse_args()

def convert_prompt_images(prompt):
    modified_prompt = []
    for _ in prompt:
        if os.path.isfile(_):
            _ = Image.open(_)
            if _.mode in ['RGBA', 'P']:
                _ = _.convert('RGB')
        
        modified_prompt.append(_)
    
    return modified_prompt

async def get_output(model, prompt, is_list=False):
    if type(prompt) == dict:
        prompt = prompt['prompt']
    if type(prompt) == str:
        prompt = [prompt]
    
    try:
        prompt = convert_prompt_images(prompt)

        # c = 0
        # print(c)
    # if True:
    # The Gemini models only support HARM_CATEGORY_HARASSMENT, HARM_CATEGORY_HATE_SPEECH, HARM_CATEGORY_SEXUALLY_EXPLICIT, and HARM_CATEGORY_DANGEROUS_CONTENT
        r = await model.generate_content_async(prompt, safety_settings = {
            genai.types.HarmCategory.HARM_CATEGORY_HATE_SPEECH: genai.types.HarmBlockThreshold.BLOCK_NONE,
            genai.types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: genai.types.HarmBlockThreshold.BLOCK_NONE,
            genai.types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: genai.types.HarmBlockThreshold.BLOCK_NONE,
            genai.types.HarmCategory.HARM_CATEGORY_HARASSMENT: genai.types.HarmBlockThreshold.BLOCK_NONE
        })
        result = r._result  # Access the result attribute
        candidates = result.candidates  # Get the list of candidates
        first_candidate = candidates[0]  # Assuming you want the first candidate
        content = first_candidate.content  # Access the content of the first candidate
        parts = content.parts  # Get the parts of the content
        first_part = parts[0]  # Assuming you want the first part
        desired_text = first_part.text  # Extract the text from the first part
        return desired_text
    except Exception as e:
        print(type(prompt))
        print(f"Error processing prompt: {e}")
        # print(r)
        return "Error occurred."

async def process_job(model, args, prompt, index, key, is_list=False):
    await asyncio.sleep(index*5)  # Adjusted sleep to use index directly
    result = await get_output(model, prompt, is_list)
    if type(prompt) == str:
        prompt = {
            "prompt": prompt
        }
    result_dict = copy.deepcopy(prompt)
    result_dict['key'] = key
    result_dict['response'] = result
    # Open the output file in append mode and write the result
    with open(args.output_file, "a") as f:
        f.write(json.dumps(result_dict) + '\n')

async def main_async_fun():
    # random.seed(420)
    random.seed(69)
    
    args = parse_args()
    with open(args.input_file, "r") as f:
        prompts = json.load(f)

    if os.path.exists(args.output_file):
        print("Output file already exists. Exiting. Theek kar code bhadwe! Remove the file and run again.")
        exit(0)
    
    prompt_list = list(prompts.items())
    random.shuffle(prompt_list)
    prompts = dict(prompt_list)
    # prompts = dict(list(prompts)) # Limit to 10 for testing

    genai.configure(api_key=kunal_api_keys[int(args.keyno)])
    if args.mm:
        model = genai.GenerativeModel('gemini-1.5-flash-latest', generation_config = genai.GenerationConfig(temperature=0.0))
    else:
        model = genai.GenerativeModel('gemini-1.5-flash-latest', generation_config = genai.GenerationConfig(temperature=0.0))


    # Ensure the output file is empty before starting
    # open(args.output_file, 'w').close()

    tasks = [process_job(model, args, prompt, index, key, args.is_array) for index, (key, prompt) in enumerate(prompts.items())]

    # Use tqdm to track progress of tasks completion
    for task in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
        await task

    print("Done")

if __name__ == "__main__":
        asyncio.run(main_async_fun())
        
