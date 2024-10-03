
import json
import os
import base64

def load_prompts(prompts_path):
    """Load prompts from a given JSON file."""
    with open(prompts_path, "r") as f:
        return json.load(f)

def encode_image(image_path):
    """Encode the image from the provided path into base64 format."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def process_prompts(dataset, question_type, prompts_path, output_dir):
    """Process prompts and generate final requests."""
    prompts = load_prompts(prompts_path)
    final_requests = []
    qid_to_gold = {}

    for qid, prompt in prompts.items():
        qid_to_gold[qid] = prompt['gold_ans']
        image_path = prompt['prompt'][1]
        
        # Check if image exists, encode it if yes
        if os.path.exists(image_path):
            encoded_image = encode_image(image_path)
        else:
            print(f"Image not found: {image_path}")
            encoded_image = None
        
        final_requests.append({
            'qid': qid,
            'prompt': prompt['prompt'][0],
            'image': encoded_image,
            'gold_answer': prompt['gold_ans']
        })
    
    # Save final results
    output_file = os.path.join(output_dir, f"{dataset}_{question_type}_final_requests.json")
    with open(output_file, 'w') as f:
        json.dump(final_requests, f, indent=2)
    
    print(f"Processed {len(prompts)} prompts. Results saved to {output_file}.")
