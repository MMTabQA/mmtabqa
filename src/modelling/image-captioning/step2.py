# %%
import os
import json
import random
import re
import argparse

def get_all_entity_image_tags(text):
    pattern = r"\{ENTITY_IMAGE-.*?\}"
    matches = re.findall(pattern, text)
    return matches


random.seed(42069)

# Define the argument parser
parser = argparse.ArgumentParser(description="Run your Python script with arguments")

# Add required arguments
parser.add_argument("--dataset", type=str, required=True, help="Dataset to use (e.g., WikiTableQuestions)")
parser.add_argument("--question-type", type=str, required=True, help="Question type (e.g., answer)")

# Parse arguments
args = parser.parse_args()

# Access parsed arguments
DATASET = args.dataset
QUESTION_TYPE = args.question_type

print(f"Running script with dataset: {DATASET} and question type: {QUESTION_TYPE}")


BASE_DIR = f"/home/suyash/final_repo/[temp]train_test_questions/{DATASET}"
OG_BASE_DIR = f"/home/suyash/final_repo/{DATASET}"
IMAGE_BASE_DIR = "/home/suyash/final_repo/final_images"

PROMPT_FILE = f"/home/suyash/final_repo/modelling/baseline_2/Results/{DATASET}/imagekey_to_prompt.json"
OUTPTUT_FILE = f"/home/suyash/final_repo/modelling/baseline_2/Results/{DATASET}/imagekey_to_output.jsonl"
#%%
tables_dict = {}
train_questions = {}
test_questions = {}



for q_type in ["explicit", "answer", "implicit"]:
    with open(os.path.join(BASE_DIR, f"test_{q_type}_questions.jsonl")) as f:
        lines = f.readlines()
        for line in lines:
            line_json = json.loads(line)
            test_questions[line_json["question_id"]] = line_json
# with open(os.path.join(BASE_DIR, f"test_{QUESTION_TYPE}_questions.jsonl")) as f:
#     lines = f.readlines()
#     for line in lines:
#         line_json = json.loads(line)
#         test_questions[line_json["question_id"]] = line_json

with open(os.path.join(OG_BASE_DIR, "experiment_ready_dataset", "tables.jsonl")) as f:
    lines = f.readlines()
    for line in lines:
        line_json = json.loads(line)
        tables_dict[line_json["table_id"]] = line_json

with open(os.path.join(BASE_DIR, f"train_{QUESTION_TYPE}_questions.jsonl")) as f:
    lines = f.readlines()
    for line in lines:
        line_json = json.loads(line)
        train_questions[line_json["question_id"]] = line_json

with open(os.path.join(OG_BASE_DIR, "experiment_ready_dataset", "image_id_to_image_path.json")) as f:
    image_id_to_image_path = json.load(f)

print("Train and test questions:", len(train_questions), len(test_questions))

with open(os.path.join(BASE_DIR, "tables_split.json")) as f:
    split = json.load(f)

train_tables = split['train']
train_exclusive_questions = [train_questions[qid] for qid in train_questions if train_questions[qid]["table_context"] in train_tables]
random.shuffle(train_exclusive_questions)

with open(os.path.join(OG_BASE_DIR, "experiment_ready_dataset", "image_id_to_original_string.json")) as f:
    image_id_to_original_string = json.load(f)

#%%
def parse_response(text):
    text_lines = text.split("\n")
    ans = {}
    for line in text_lines:
        line = line.strip()
        if "->" not in line:
            continue
        try:
            entity_tag, prediction = line.split("->")
        except:
            entity_tag1, entity_tag2, prediction = line.split("->")
            ans[entity_tag1.strip()] = prediction.strip()
            ans[entity_tag2.strip()] = prediction.strip()
            continue
        
        if len(get_all_entity_image_tags(entity_tag))==0:
            print("ONO", entity_tag)
            print(text)
            print("----------------")
            entity_tag = f"{{{entity_tag.strip()}}}"
        
        entity_tag_list = set(get_all_entity_image_tags(entity_tag))
        
        if len(entity_tag_list)!=1:
            print("WTFFFF", line)
            print(entity_tag_list)
            # print(line)
        prediction = prediction.strip()
        for entity_tag in entity_tag_list:
            ans[entity_tag] = prediction
    return ans

imagekey_entity_to_prediction = {}
imagekey_to_response = {} 

with open(OUTPTUT_FILE) as f:
    for line in f.readlines():
        line = json.loads(line)
        imagekey = line['key']
        response_text = line['response']
        response_text = response_text.split("Step 3:")[-1]
        entityid_to_prediction = parse_response(response_text)
        for entity_id, prediction in entityid_to_prediction.items():
            imagekey_entity_to_prediction[(imagekey, entity_id)] = prediction
        imagekey_to_response[imagekey] = line['response']

# %%
len(imagekey_entity_to_prediction)


def generate_table_string(table_array):
    table_string = ""

    for row_idx, row in enumerate(table_array):
        for col_idx, cell in enumerate(row):
            cell = cell.replace("\t", " ").replace("\n", " ").replace("\\n", " ").replace("\\t", " ").replace("|", " ")
            table_string = table_string  + cell + " | "
        table_string = table_string + "\n"

    return table_string

def get_all_image_tags(text):
    pattern = r"\{IMG-\{.*?\}\}"
    matches = re.findall(pattern, text)
    return matches


# %%
import sys
sys.path.append("/home/suyash/final_repo/evaluation_metrics")
from exact_match import EvaluationMetrics

test_tableids = [question_data["table_context"] for question_id, question_data in list(test_questions.items())]

with open(f"/home/suyash/final_repo/modelling/baseline_2/Results/{DATASET}/imageid_to_imagekey_entityid.json") as f:
    IMAGEID_TO_IMAGEKEY_ENTITYID = json.load(f)

imageid_to_prediction = {}
imagekey_entity_to_gold = {}
imageid_to_gold = {}

total_image_tags = 0
image_tags_hits = 0
for table_id in test_tableids:
    table_array = tables_dict[table_id]["table_array"]
    table_string = generate_table_string(table_array)
    image_tags = get_all_image_tags(table_string)

    for image_id in image_tags:
        total_image_tags += 1
        gold_ans = image_id_to_original_string[image_id]

        pred_ans_key = tuple(IMAGEID_TO_IMAGEKEY_ENTITYID[image_id])
        imagekey_entity_to_gold[pred_ans_key] = gold_ans
        imageid_to_gold[image_id] = gold_ans
        try:
            imageid_to_prediction[image_id] = imagekey_entity_to_prediction[pred_ans_key]
            image_tags_hits += 1
        except:
            imageid_to_prediction[image_id] = "error"

print("Total image tags:", total_image_tags)
print("Image tags hits:", image_tags_hits)
print(len(imagekey_entity_to_prediction))

exact_match = 0
substring_match = 0
llm_match = 0
incorrect = 0
total = 0
f1_scores = []

llm_evaluations = []
evaluator = EvaluationMetrics()

for image_id, gold_ans in imageid_to_gold.items():
    prediction = imageid_to_prediction[image_id]

    if evaluator.compute_exact_match(gold_ans, prediction):
        exact_match += 1
        substring_match += 1
        llm_match += 1
    else:
        if evaluator.gold_ans_in_prediction(prediction, gold_ans):
            substring_match += 1
        else:
            pass

    f1_scores.append(evaluator.compute_f1_score(gold_ans, prediction))

mean_f1_score = sum(f1_scores) / len(f1_scores)
accuracy = exact_match / len(f1_scores)
substring_accuracy = substring_match / len(f1_scores)

print(f"{DATASET} {QUESTION_TYPE} Results:")
print(f"Exact match: {accuracy*100}")
print(f"Substring match: {substring_accuracy*100}")
print(f"Mean F1 score: {mean_f1_score}")

exit(0)

# %%
