import os
import json
import random
import re
import argparse

random.seed(42069)

# Define the argument parser
parser = argparse.ArgumentParser(description="Run your Python script with arguments")

# Add required arguments
parser.add_argument("--dataset", type=str, required=True, help="Dataset to use (e.g., WikiTableQuestions)")
parser.add_argument("--question-type", type=str, required=True, help="Question type (e.g., answer)")

TODO: rsync
with open("/home/suyash/final_repo/HybridQA_Tab_MM/Hybrid_QA_MM/experiment_ready_dataset/mm_passages.json","r") as f:
    mm_passages=json.load(f)
with open("/home/suyash/final_repo/HybridQA_Tab_MM/Hybrid_QA_MM/experiment_ready_dataset/text_passages.json","r") as f:
    text_passages = json.load(f)
    
    
# Parse arguments
args = parser.parse_args()

# Access parsed arguments
DATASET = args.dataset
QUESTION_TYPE = args.question_type

print(f"Running script with dataset: {DATASET} and question type: {QUESTION_TYPE}")


BASE_DIR = f"/home/suyash/final_repo/redone_samples_700cap/{DATASET}"
OG_BASE_DIR = f"/home/suyash/final_repo/{DATASET}"
IMAGE_BASE_DIR = "/home/suyash/final_repo/final_images"
OG_BASE_DIR_HYB = f"/home/suyash/final_repo/HybridQA_Tab_MM/Hybrid_QA_MM"
def get_all_entity_image_tags(text):
    pattern = r"\{ENTITY_IMAGE-.*?\}"
    matches = re.findall(pattern, text)
    return matches
# %%
tables_dict = {}
train_questions = {}
test_questions = {}

for q_type in ["explicit", "answer", "implicit"]:
    with open(os.path.join(BASE_DIR, f"test_{q_type}_questions.jsonl")) as f:
        lines = f.readlines()
        for line in lines:
            line_json = json.loads(line)
            test_questions[line_json["question_id"]] = line_json

with open(os.path.join(OG_BASE_DIR_HYB, "experiment_ready_dataset", "tables.jsonl")) as f:
    lines = f.readlines()
    for line in lines:
        line_json = json.loads(line)
        tables_dict[line_json["table_id"]] = line_json
TODO: Rename the path below
with open("/home/suyash/final_repo/Refactor/outputs/image_id_to_refined_path.json") as f:
    image_id_to_image_path = json.load(f)

print("Train and test questions:", len(train_questions), len(test_questions))

# with open(os.path.join(BASE_DIR, "tables_split.json")) as f:
#     split = json.load(f)

# train_tables = split['train']
# train_exclusive_questions = [train_questions[qid] for qid in train_questions if train_questions[qid]["table_context"] in train_tables]
# random.shuffle(train_exclusive_questions)

with open(os.path.join(OG_BASE_DIR_HYB, "experiment_ready_dataset", "image_id_to_original_string.json")) as f:
    image_id_to_original_string = json.load(f)

# %%
from collections import defaultdict
import re
from PIL import Image
import num2words
import copy
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


def generate_imagekey(table_id, cell_text, column_index):
    return f"{table_id}+{cell_text}+{column_index}"

def generate_other_rows(table_array, row_idx, col_idx):
    other_rows = []
    for row in table_array:
        if row == table_array[row_idx]:
            continue
        
        if len(get_all_image_tags(row[col_idx])) == 0:
            other_rows.append(row[col_idx])
    
    return other_rows

def find_other_valid_rows(table_array, curr_row_idx, curr_col_idx):
    other_rows = []
    for row_idx, row in enumerate(table_array):
        if len(get_all_image_tags(row[col_idx])) == 0 and row_idx!=0 and row_idx!=curr_row_idx:
            other_rows.append(row_idx)
    
    if len(other_rows) == 0:
        print("Yo WTF 0 other rows containing text for this column")
        return []
    
    return other_rows

def generate_table_from_row(table_array, row_idx):
    table_headers = table_array[0]
    table_row = table_array[row_idx]
    
    infobox_strings = [f"{k}: {v}" for k, v in zip(table_headers, table_row)]
    infobox_string = "\n".join(infobox_strings)
    return infobox_string

def make_imageids_uniform(table_array):
    curr_id = 1
    imagepath_to_imageid = defaultdict(list)
    imagepath_to_entityid = {}
    
    for row_idx, row in enumerate(table_array):
        for col_idx, cell in enumerate(row):
            image_ids = get_all_image_tags(cell)
            for image_id in image_ids:
                imagepath = image_id_to_image_path[image_id]
                imagepath_to_imageid[imagepath].append(image_id)
                if imagepath not in imagepath_to_entityid:
                    imagepath_to_entityid[imagepath] = f"{{ENTITY_IMAGE-{curr_id}}}"
                    curr_id += 1
    
    modified_table_array = []
    imageid_to_entityid = {}
    
    entityid_to_imageids = defaultdict(list)
    
    for row_idx, row in enumerate(table_array):
        modified_row = []
        for col_idx, cell in enumerate(row):
            image_ids = get_all_image_tags(cell)
            for image_id in image_ids:
                imagepath = image_id_to_image_path[image_id]
                rep_entity = imagepath_to_entityid[imagepath]
                imageid_to_entityid[image_id] = rep_entity
                entityid_to_imageids[rep_entity].append(image_id)
                cell = cell.replace(image_id, rep_entity)
            modified_row.append(cell)
        modified_table_array.append(modified_row)
    return modified_table_array, imageid_to_entityid, entityid_to_imageids

imagekey_to_prompt = {}
multi_image_cell = 0
single_image_cell = 0
imagekey_to_originalcell = {}
imagekey_to_tablecell = {}

IMAGEID_TO_IMAGEKEY_ENTITYID = {} # image_id -> (image_key, entity_id)

test_tableids = [question_data["table_context"] for question_id, question_data in list(test_questions.items())]

# test_tableids = train_exclusive_tables
for table_id in test_tableids:
    
    table_data = tables_dict[table_id]
    table_array = table_data["table_array"]
    
    imagekey_to_cellids = defaultdict(list)
    headers = table_array[0]
    
    uniform_table_array, imageid_to_entityid, entityid_to_imageids = make_imageids_uniform(table_array)
    if DATASET == "Hybrid_QA_MM":
        passage_arr = tables_dict[table_id]["cells_to_link"]
    
    passage_dict = {}
    
    for elements in passage_arr:
        tup = tuple(elements[2])
        if tup in passage_dict.keys():
            passage_dict[tup].append((elements[0],elements[1]))
        else:
            passage_dict[tup]= [(elements[0],elements[1])]
            # print("mkc3")
    # table_metadata=table_metadata.replace("{","""{{""")
    # table_metadata=table_metadata.replace("}","""}}""")
    # question=question.replace("{","""{{""")
    # question=question.replace("}","""}}""")
    # answer=answer.replace("{","""{{""")
    # answer=answer.replace("}","""}}""")
    for row_idx, row in enumerate(uniform_table_array):
        for col_idx, cell in enumerate(row):
            entity_ids = get_all_entity_image_tags(cell)
            if len(entity_ids)>0:
                imagekey = generate_imagekey(table_id, cell, col_idx)
                imagekey_to_cellids[imagekey].append((row_idx, col_idx))
                for entity_id in entity_ids:
                    for image_id in entityid_to_imageids[entity_id]:
                        IMAGEID_TO_IMAGEKEY_ENTITYID[image_id] = (imagekey, entity_id)

    if DATASET == "WikiTableQuestions":
        page_title, section_title = get_wtq_metadata(table_id)
    elif DATASET == "WikiSQL":
        page_title = tables_dict[table_id]["page_title"]
        section_title = tables_dict[table_id]["section_title"]
    elif DATASET == "fetaqa_MM_cleaned":
        page_title = tables_dict[table_id]["table_page_title"]
        section_title = tables_dict[table_id]["table_section_title"]
    else:
        page_title = ""
        section_title=tables_dict[table_id]["url"].split("/")[-1]
        
    for imagekey, cellids in imagekey_to_cellids.items():
        
        curr_col_idx = cellids[0][1]
        curr_row_idx = cellids[0][0]
        passage_info = ""
        specific_value = (curr_row_idx,curr_col_idx)
    
        # for k,val in passage_dict.items():
        #   elements[0]=elements[0].replace("{","""{{""")
        #   elements[1]=elements[1].replace("}","""}}""")
        passage_info+=f"Passage related to entity in "
        #   print("mkc1")
        # for vals in val:
        passage_info += f"row {curr_row_idx} and column {curr_col_idx}, "  
        keys = [key for key, value_array in passage_dict.items() if specific_value in value_array]

        # for k,val in passage_dict.items():   
        for k in keys:
         for links in k:  
            if links in mm_passages.keys():
                mm_passages[links]=mm_passages[links].replace("{","""{{""").replace("}","""}}""")
    
                passage_info+=f"{mm_passages[links]}. "
                # print("mkc2")
            elif links in text_passages.keys():
                text_passages[links]=text_passages[links].replace("{","""{{""").replace("}","""}}""")
                passage_info+=f"{text_passages[links]}. "
        
        col_header = headers[curr_col_idx]
        
        cell_text = uniform_table_array[curr_row_idx][curr_col_idx]
        entity_tags = get_all_entity_image_tags(cell_text)
        entity_tags_string = ", ".join(entity_tags)
        
        other_valid_rows = find_other_valid_rows(uniform_table_array, curr_row_idx, curr_col_idx)
        
        infobox_1 = generate_table_from_row(uniform_table_array, other_valid_rows[0])
        infobox_1 = "Table 1:\n" + infobox_1
        
        if len(other_valid_rows) > 1:
            infobox_2 = generate_table_from_row(uniform_table_array, other_valid_rows[1])
            infobox_2 = "\n\nTable 2:\n" + infobox_2
        else:
            infobox_2 = ""
        
        main_infobox = [generate_table_from_row(uniform_table_array, cellid[0]) for cellid in cellids]
        main_infobox_string = ""
        for idx, infobox_table in enumerate(main_infobox):
            main_infobox_string = main_infobox_string + f"Table {idx+3}:\n{infobox_table}\n"
            main_infobox_string = main_infobox_string + "\n"
        
        total_tables = 2 + len(main_infobox)
        
        num_tables = num2words.num2words(total_tables)
        first_table_index = 3 if len(infobox_2)>0 else 2
        # other_rows = generate_other_rows(table_array, curr_row_idx, curr_col_idx)
        # if len(other_rows) > 0:
        #     other_rows_string = ", ".join(other_rows)
        #     other_rows_text = f"Some other similar {col_header} are {other_rows_string}. "
        # else:
        #     other_rows_text = ""

        if len(main_infobox) > 1:
            focus_tables = f"Table {first_table_index} through Table {num_tables}"
            same_entry_string = " (which is the same entry across all the table(s))"
            table_or_tables = f"{num2words.num2words(total_tables-2)} tables"
        else:
            focus_tables = f"Table {first_table_index}"
            same_entry_string = ""
            table_or_tables = "table"
        
        TEXT_PROMPT = f"""You are given certain infobox tables centred around a common topic. Each table lists its entries in separate lines, formatted as 'column_name: cell_content'. However, few entities in some of the tables have been replaced by images of those entities, denoted by tags of the format '{{ENTITY_IMAGE-<entity_id>}}' that represent those entities. Your task is to, based upon the context provided by the given table and the related prior tables, reason, predict and replace certain {{ENTITY_IMAGE}} tags by their original representative entity names and also provide their visual descriptions. Some example responses are given to better illustrate the task, and each new example starts with a line of '#'s. NOTE: YOU ARE ONLY PROVIDED IMAGES FOR THE MAIN TASK, NOT FOR THE EXAMPLE TASKS:

#####################
EXAMPLE 1:

Table context: Solo discography|Singles of Gladys Knight

Table 1:
Year: 1978
Single: I'm Coming Home Again
Peak chart positions US: —
Peak chart positions US R&B: 54
Peak chart positions US A/C: —
Peak chart positions UK: —

Table 2:
Year: 1979
Single: Am I Too Late
Peak chart positions US: —
Peak chart positions US R&B: 45
Peak chart positions US A/C: —
Peak chart positions UK: —

Now, based upon the examples above and the table(s) given below, your specific task is to replace the image tag(s) {{ENTITY_IMAGE-10}}, {{ENTITY_IMAGE-4}}, {{ENTITY_IMAGE-11}} mentioned in Single column of the following table with their original entities. In order to help you perform this task, you are also provided images corresponding to the tag(s) in the order {{ENTITY_IMAGE-10}}, {{ENTITY_IMAGE-4}}, {{ENTITY_IMAGE-11}}. You must do this in the following steps:

Step 1: You must describe the relevant information about the entity that can be inferred from the given table context related to Solo discography|Singles of Gladys Knight. Ensure that this is as detailed as possible, and ONLY uses the information provided in the three tables given above. Use your real-world knowledge to make as many inferences and form relationships as possible from the information provided in the table. IMPORTANT: It is of utmost importance that you DO NOT include information from the image in this description.

Step 2: You must visually describe the image(s) in complete detail, highlighting the important aspects based upon the context of the tables provided and the description obtained in Step 1. The visual descriptions must be based upon the fact that the image(s) occur in the entry '{{ENTITY_IMAGE-10}} (with {{ENTITY_IMAGE-4}} & {{ENTITY_IMAGE-11}})' representing a Single in the context of Solo discography|Singles of Gladys Knight. Ensure that you consider other entries of Single in other tables as well to make the visual description as accurate as possible. Using these visual descriptions, you must also identify the entities that are depicted using the different images in the context of the table.

Step 3: You must combine the information from Step 1 and Step 2 along with other attributes in the image, the context of the table(s) provided and real-world knowledge to provide the actual entry corresponding to '{{ENTITY_IMAGE-10}} (with {{ENTITY_IMAGE-4}} & {{ENTITY_IMAGE-11}})' of the Single column from the table above.

Step 4: Based upon the response in Step 3, you must output the entities corresponding to the entity tag(s) present in the entry for Single in the format '{{ENTITY_IMAGE-<entity_id>}} -> <entity_string>'. It is VERY IMPORTANT that you follow this format while providing the output. You MUST list every entity tag in a separate line.

Step 5: Give a concise visual description corresponding to the images for the entity tag(s).
IMPORTANT: You must use your real-world knowledge, the other entires given in the table(s) below and the example tables provided above to find the replacement for the string.

Table 3:
Year: 1991
Single: {{ENTITY_IMAGE-10}} (with {{ENTITY_IMAGE-4}} & {{ENTITY_IMAGE-11}})
Peak chart positions US: —
Peak chart positions US R&B: 19
Peak chart positions US A/C: —
Peak chart positions UK: —


Step 1 Table description:
Based on the context of the table, the entry for Single in the table is related to Solo discography|Singles of Gladys Knight. The entry is likely to be a song title, as suggested by the other entries in the column. Also, we know that the single peaked at 19th position in US R&B charts in 1991. This information suggests that the song was a popular R&B track in the early 90s. Based upon the entry in the table '{{ENTITY_IMAGE-10}} (with {{ENTITY_IMAGE-4}} & {{ENTITY_IMAGE-11}})', we can infer that {{ENTITY_IMAGE-10}} corresponds to the song title, while {{ENTITY_IMAGE-4}} and {{ENTITY_IMAGE-11}} are likely to be images of the artists who featured in the song.

Step 2 Image Description:
{{ENTITY_IMAGE-10}} is an image of a vinyl record with a label that reads "Superwoman", which is name of a popular song by Gladys Knight. The record is black and has a silver label. The text "Superwoman" is written in blue on the label. Since this entity should be a song title, "Superwoman" is the song and corresponds to {{ENTITY_IMAGE-10}}. {{ENTITY_IMAGE-4}} is an image of a woman with short blonde hair, who must be one of the artists featuring on the song Superwoman. She can be identified as the artist Dionne Warwick, as her description matches with the features visible in the image. {{ENTITY_IMAGE-11}} is an image of a woman with long black hair, who must be another artist featuring on the song Superwoman. She can be identified as the artist Patti LaBelle as her description matches with her features. 

**Step 3:**
From the entities identified above, the entry of the cell must correspond to 'Superwoman (with Dionne Warwick & Patti LaBelle)'.

**Step 4:**
{{ENTITY_IMAGE-10}} -> Superwoman
{{ENTITY_IMAGE-4}} -> Dionne Warwick
{{ENTITY_IMAGE-11}} -> Patti LaBelle
################################

**Step 5:**
{{ENTITY_IMAGE-10}} -> A lady wearing a Black and Red coloured dress with a S written on her torso.
{{ENTITY_IMAGE-4}} -> A middle aged Black woman
{{ENTITY_IMAGE-11}} -> A middle aged woman with short hair

MAIN TASK:
Table context: {section_title} of {page_title}

{infobox_1}

{infobox_2}

{passage_info}

Now, based upon the examples above and the table(s) given below, your specific task is to replace the image tag(s) {entity_tags_string} mentioned in {col_header} column of the following {table_or_tables} with their original entities{same_entry_string}. In order to help you perform this task, you are also provided images corresponding to the tag(s) in the order {entity_tags_string}. You are also given relevant passages related ti the cell if there are any. Use them also. You must do this in the following steps:

Step 1: You must describe the relevant information about the entity that can be inferred from the given table context related to {section_title} of {page_title}. Ensure that this is as detailed as possible, and ONLY uses the information provided in the {num_tables} tables given above. Use your real-world knowledge to make as many inferences and form relationships as possible from the information provided in the image. IMPORTANT: It is of utmost importance that you DO NOT include information from the image in this description.

Step 2: You must visually describe the image(s) in complete detail, highlighting the important aspects based upon the context of the tables provided and the description obtained in Step 1. The visual descriptions must be based upon the fact that the image(s) occur in the entry {cell_text} representing a {col_header} in the context of {section_title} of {page_title}. Ensure that you consider other entries of {col_header} in other tables as well to make the visual description as accurate as possible. Using these visual descriptions, you must also identify the entities that are depicted using the different images in the context of the table.

Step 3: You must combine the information from Step 1 and Step 2 along with other attributes in the image, the context of the table(s) provided and real-world knowledge to provide the actual entry corresponding to {cell_text} of the {col_header} column from the table above.

Step 4: Based upon the response in Step 3, you must output the entities corresponding to the entity tag(s) present in the entry for Single in the format '{{ENTITY_IMAGE-<entity_id>}} -> <entity_string>'. It is VERY IMPORTANT that you follow this format while providing the output. You MUST list every entity tag in a separate line.

Step 5: Give a concise visual description corresponding to the images for the entity tag(s).
IMPORTANT: You must use your real-world knowledge, the other entires given in the table(s) below and the example tables provided above to find the replacement for the string.

{main_infobox_string}

Step 1 Table description: """


        images_path = [image_id_to_image_path[entityid_to_imageids[entity_id][0]] for entity_id in entity_tags]
        images = [(os.path.join(IMAGE_BASE_DIR, image_path)) for image_path in images_path]
        
        imagekey_to_prompt[imagekey] = {"prompt": [TEXT_PROMPT] + images,
        }

# %%
import os
try:
    os.mkdir(f"/home/suyash/final_repo/modelling/baseline_2/Results/{DATASET}")
except:
    pass

PROMPT_FILE = f"/home/suyash/final_repo/modelling/baseline_2/Results/{DATASET}/imagekey_to_prompt.json"
with open(PROMPT_FILE, 'w') as f:
    json.dump(imagekey_to_prompt, f)

with open(f"/home/suyash/final_repo/modelling/baseline_2/Results/{DATASET}/imageid_to_imagekey_entityid.json", 'w') as f:
    json.dump(IMAGEID_TO_IMAGEKEY_ENTITYID, f)

OUTPTUT_FILE = f"/home/suyash/final_repo/modelling/baseline_2/Results/{DATASET}/imagekey_to_output.jsonl"

# %%
print(f"python3 /home/suyash/final_repo/Common_codes/run_gemini_query.py --input_file {PROMPT_FILE} --output_file {OUTPTUT_FILE} --mm")

# %%
