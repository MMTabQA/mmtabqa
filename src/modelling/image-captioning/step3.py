from collections import defaultdict
import re
import os
import json
import random
import argparse

random.seed(42069)

def get_all_entity_image_tags(text):
    pattern = r"\{ENTITY_IMAGE-.*?\}"
    matches = re.findall(pattern, text)
    return matches

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

ENTITY_OUTPTUT_FILE = f"/home/suyash/final_repo/modelling/baseline_2/Results/{DATASET}/imagekey_to_output.jsonl"

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

with open(ENTITY_OUTPTUT_FILE) as f:
    for line in f.readlines():
        line = json.loads(line)
        imagekey = line['key']
        response_text = line['response']
        response_text = response_text.split("Step 3:")[-1]
        entityid_to_prediction = parse_response(response_text)
        for entity_id, prediction in entityid_to_prediction.items():
            imagekey_entity_to_prediction[(imagekey, entity_id)] = prediction
        imagekey_to_response[imagekey] = line['response']


with open(f"/home/suyash/final_repo/modelling/baseline_2/Results/{DATASET}/imageid_to_imagekey_entityid.json") as f:
    IMAGEID_TO_IMAGEKEY_ENTITYID = json.load(f)

tables_dict = {}
train_questions = {}
test_questions = {}

with open(os.path.join(BASE_DIR, f"test_{QUESTION_TYPE}_questions.jsonl")) as f:
    lines = f.readlines()
    for line in lines:
        line_json = json.loads(line)
        test_questions[line_json["question_id"]] = line_json


with open(os.path.join(BASE_DIR, f"train_{QUESTION_TYPE}_questions.jsonl")) as f:
    lines = f.readlines()
    for line in lines:
        line_json = json.loads(line)
        train_questions[line_json["question_id"]] = line_json


with open(os.path.join(OG_BASE_DIR, "experiment_ready_dataset", "tables.jsonl")) as f:
    lines = f.readlines()
    for line in lines:
        line_json = json.loads(line)
        tables_dict[line_json["table_id"]] = line_json

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


def get_all_image_tags(text):
    pattern = r"\{IMG-\{.*?\}\}"
    matches = re.findall(pattern, text)
    return matches

def generate_table_string(table_array):
    table_string = ""

    for row_idx, row in enumerate(table_array):
        for col_idx, cell in enumerate(row):
            cell = cell.replace("\t", " ").replace("\n", " ").replace("\\n", " ").replace("\\t", " ").replace("|", " ")
            table_string = table_string  + cell + " | "
        table_string = table_string + "\n"

    return table_string


def make_array_qa_ready(table_array):
    curr_id = 1
    imagepath_to_imageid = defaultdict(list)
    imageid_to_entityid = {}


    for row_idx, row in enumerate(table_array):
        for col_idx, cell in enumerate(row):
            image_ids = get_all_image_tags(cell)
            for image_id in image_ids:
                if image_id not in imageid_to_entityid:
                    imageid_to_entityid[image_id] = f"{{ENTITY-{curr_id}}}"
                    curr_id += 1

    modified_table_array = []

    entityid_to_imageid = {}
    
    for row_idx, row in enumerate(table_array):
        modified_row = []
        for col_idx, cell in enumerate(row):
            image_ids = get_all_image_tags(cell)
            for image_id in image_ids:
                rep_entity = imageid_to_entityid[image_id]
                imageid_to_entityid[image_id] = rep_entity
                entityid_to_imageid[rep_entity] = image_id
                cell = cell.replace(image_id, rep_entity)
            modified_row.append(cell)
        modified_table_array.append(modified_row)
    return modified_table_array, imageid_to_entityid, entityid_to_imageid

################WTQ_specific####################
import pandas as pd
with open("/home/suyash/final_repo/WikiTableQuestions/WikiTableQuestions/misc/table-metadata.tsv") as f:
    WTQ_METADATA = pd.read_csv(f, sep="\t")

def get_wtq_metadata(table_id):
    table_metadata = WTQ_METADATA[WTQ_METADATA["contextId"] == table_id.replace("WTQ/", "")]

    page_title = table_metadata["title"].values[0]
    headers = table_metadata['headers'].values[0]
    return page_title, headers
################################################
def create_metdata_prompt_sentence(page_title, table_headers):
    if pd.isna(table_headers):
        prompt = f"{page_title}."
    else:
        prompt = f"{table_headers} of {page_title}."
    return prompt



# %%
def get_all_entity_tags(text):
    entity_tags = re.findall(r'\{ENTITY-[0-9]+\}', text)
    return entity_tags

QUESTION_PROMPT_TEMPLATE_WITH_ANSWER = """Table context: {table_metadata}

Table:
{table}

Entities predicted:
{image_ids_string}

Question: {question}

Answer:
{answer}
"""

QUESTION_PROMPT_TEMPLATE_WO_ANSWER = """Table context: {table_metadata}

Table:
{table}

Entities predicted:
{image_ids_string}

Question: {question}

Now, Steps 1-2 are below for the given question:

Step 1 Question Answering Reasoning:
"""


example_question_ids = train_exclusive_questions[:2]

# example_reasons = {
#     'nt-3923': "We know that {ENTITY-1} corresponds to the World Junior Championship 2002. The question asks for the competition listed after the World Junior Championships 2002. The competition listed after the World Junior Championships 2002 from the table is the Asian Games. Thus, the answer is 'Asian Games'.",
#     'nt-12173': "We know that {ENTITY-1} corresponds to 'Loose Women'. The question asks for the number of consecutive years that Loose Women ran most recently. From the table, we can see that it ran consecutively from 1999-2002, 2010, and 2012-13. Thus, the number of consecutive years that Loose Women ran was 4, 1, and 2. Therefore, the answer is the most recent of all these consecutive runs. Thus, the answer is '2'."
# }

def create_question_prompt(table_array, table_id, question, entityid_to_imageid=None, imageid_to_prediction=None, qa_reason=None, answer=None):
    table_string = generate_table_string(table_array)
    if DATASET == "WikiTableQuestions":
        page_title, headers = get_wtq_metadata(table_id)
    elif DATASET == "WikiSQL":
        page_title = tables_dict[table_id]["page_title"]
        headers = tables_dict[table_id]["section_title"]
    elif DATASET == "fetaqa_MM_cleaned":
        page_title = tables_dict[table_id]["table_page_title"]
        headers = tables_dict[table_id]["table_section_title"]

    table_metadata = create_metdata_prompt_sentence(page_title, headers)
    
    image_ids_string = ""
    
    entity_ids = get_all_entity_tags(table_string)
    
    for entity_id in entity_ids:
        image_id = entityid_to_imageid[entity_id]
        prediction = imageid_to_prediction[image_id]
        if "error" in prediction.lower():
            image_ids_string = image_ids_string + f"{entity_id} -> Couldn't predict entity for this tag\n"
        else:
            image_ids_string = image_ids_string + f"{entity_id} -> {prediction}\n"
    
    if answer:
        question_prompt = QUESTION_PROMPT_TEMPLATE_WITH_ANSWER.format(table_metadata=table_metadata, table=table_string, question=question, image_ids_string=image_ids_string, answer=answer)
    else:
        question_prompt = QUESTION_PROMPT_TEMPLATE_WO_ANSWER.format(table_metadata=table_metadata, table=table_string, question=question, image_ids_string=image_ids_string)
    return question_prompt

FINAL_EXAMPLES = []

for ex_ques in example_question_ids:
    table_id = ex_ques["table_context"]
    table_array = tables_dict[table_id]["table_array"]
    question = ex_ques["question"]
    answer = ex_ques["answer"]
    uniformed_table_array, imageid_to_entityid, entityid_to_imageid = make_array_qa_ready(table_array)
    
    question_prompt = create_question_prompt(uniformed_table_array, table_id, question, entityid_to_imageid, image_id_to_original_string, qa_reason=None, answer=answer)

    print(question_prompt)
    FINAL_EXAMPLES.append(question_prompt)


# %%
QUESTION_ANSWERING_PROMPT_TEMPLATE = """You are given a table in which some entities in various table cells have been replaced by tokens of the type '{{ENTITY-<entity_id>}}'. Further, you are given some predictions about the entities corresponding to these tokens, which might be correct or incorrect. Each row of the table is in separate lines, and the columns are separated by ' | '. Based upon the context of the table while using real-world knowledge, your task is to use the entity predictions given to answer the question. You must perform this task in the following steps

Step 1: Reason about what should be the answer to the question based upon the table and the predicted entities corresponding to the entity tags. The reasoning should be detailed and should be based upon the context of the table and the question, using the predicted entities given and real-world knowledge for answering the question. You must also account that some entity predictions might be incorrect, and you must reason about the correct entity based upon the context provided by the table if any entity appears incorrect. IMPORTANT: You must explore any kind of reasoning -- numerical, logical, knowledge-based needed for performing the task. 
Step 2: Based upon the reasoning provided, provide the answer to the question.

IMPORTANT: You must use the provided predictions as loose guidelines and reason about the correct answer based upon the context provided by the table and the question. You must also use your real-world knowledge to answer the question to the best possible extent. It is really important that you use the context given by the question to fix any incorrect predictions and answer the question correctly. You MUST provide an answer that is the best correct for the question.

You are also provided with some examples for better understanding the task:

Example 1:
{example_1}

Example 2:
{example_2}

Now, based upon the examples given above, you must follow the steps 1-2 to answer the question corresponding to the table. It is IMPORTANT that you perform all the both the steps to the best possible extent to get the correct answer. IMPORTANT: You must give the answer in the format 'Step 2:\n<answer>'.

{main_task}"""


# %%
def get_imageid_to_predictions(imageids):
    imageid_to_predictions = {}
    for image_id in imageids:
        image_key = tuple(IMAGEID_TO_IMAGEKEY_ENTITYID[image_id])
        if image_key in imagekey_entity_to_prediction:
            imageid_to_predictions[image_id] = imagekey_entity_to_prediction[image_key]
        else:
            imageid_to_predictions[image_id] = "Couldn't predict entity for this tag"
    return imageid_to_predictions

qid_to_prompt = {}

for qid, ques in test_questions.items():
    table_id = ques["table_context"]
    table_array = tables_dict[table_id]["table_array"]
    imageid_to_predictions = get_imageid_to_predictions(get_all_image_tags(generate_table_string(table_array)))
    uniform_table_array, imageid_to_entityid, entityid_to_imageid = make_array_qa_ready(table_array)
    question = ques["question"]
    gold_answer = ques["answer"]
    main_task = create_question_prompt(uniform_table_array, table_id, question, entityid_to_imageid, imageid_to_predictions, qa_reason=None, answer=None)
    final_prompt = QUESTION_ANSWERING_PROMPT_TEMPLATE.format(example_1=FINAL_EXAMPLES[0], example_2=FINAL_EXAMPLES[1], main_task=main_task)
    qid_to_prompt[qid] = {
        "prompt": final_prompt,
        "gold_answer": gold_answer
    }

# %%
print(list(qid_to_prompt.items())[0][1]['prompt'])

# %%
OUTPUT_DIR = f"/home/suyash/final_repo/modelling/baseline_2/Results/{DATASET}_{QUESTION_TYPE}"
try:
    os.mkdir(OUTPUT_DIR)
except:
    pass

PROMPT_PATH = os.path.join(OUTPUT_DIR, f"qa_prompt.jsonl")
OUTPUT_PATH = os.path.join(OUTPUT_DIR, f"qa_output.jsonl")
with open(os.path.join(PROMPT_PATH), 'w') as f:
    json.dump(qid_to_prompt, f)

# %%
print(f"python3 /home/suyash/final_repo/Common_codes/run_gemini_query.py --input_file {PROMPT_PATH} --output_file {OUTPUT_PATH}")
exit(0)

# %%
import jsonlines
qid_to_response = {}
qid_to_goldans = {}
correct = 0
incorrect = 0
with jsonlines.open(OUTPUT_PATH, 'r') as f:
    for line in f:
        prediction = (line['response'])
        prediction = prediction.split("Step 2")[-1].strip()
        gold_ans = line['gold_answer']
        
        if compute_exact_match(gold_ans, prediction):
            correct += 1
            print("---------------")
            # print("Prompt: ", line['prompt'].split("Now, based upon the examples given above, you must follow the steps 1-2 to answer the question corresponding to the table.")[-1])
            # print("Response: ", line['response'])
            print("Prediction: ", prediction)
            print("Gold ans: ", gold_ans)
            print("---------------")
        elif gold_ans_in_prediction(prediction, gold_ans):
            correct += 1
            print("---------------")
            # print("Prompt: ", line['prompt'].split("Now, based upon the examples given above, you must follow the steps 1-2 to answer the question corresponding to the table.")[-1])
            # print("Response: ", line['response'])
            print("Prediction: ", prediction)
            print("Gold ans: ", gold_ans)
            print("---------------")
        else:
            incorrect += 1
        f1_score = compute_f1_score(gold_ans, prediction)
        f1_scores.append(f1_score)
        
        # print("---------------")
        
print("Accuracy:", correct/(correct+incorrect))
print("F1 Score:", sum(f1_scores)/len(f1_scores))

# With reasoning in few-shot examples
# Accuracy: 0.39849624060150374
# F1 Score: 0.5144053574135438

# Without reasoning in few-shot examples
# Accuracy: 0.4398496240601504
# F1 Score: 0.4185913782673272

# %%
with jsonlines.open(OUTPUT_PATH, 'r') as f:
    for line in f:
        prediction = (line['response'])
        prediction = prediction.split("Step 2 Answer:")[-1].strip()
        gold_ans = line['gold_answer']
        
        if compute_exact_match(gold_ans, prediction):
            correct += 1
        elif gold_ans_in_prediction(prediction, gold_ans):
            correct += 1
        else:
            incorrect += 1
        print("---------------")
        print("Prompt: ", line['prompt'])
        print("Response: ", line['response'])
        print("")
        print("Prediction: ", prediction)
        print("Gold ans: ", gold_ans)
        print("---------------")
        f1_score = compute_f1_score(gold_ans, prediction)
        f1_scores.append(f1_score)
