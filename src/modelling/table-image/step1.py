# %%
import os
import json
import random
import argparse
import sys
sys.path.append('/home/suyash/final_repo/modelling/table-image-approach/scripts')

from temp_gemini import run_gemini

random.seed(42069)
dne=[]
PROMPT_OUT_FILE = "/home/suyash/final_repo/modelling/table-image-approach/few_shot.json"

with open("/home/suyash/final_repo/HybridQA_Tab_MM/Hybrid_QA_MM/experiment_ready_dataset/mm_passages.json","r") as f:
    mm_passages=json.load(f)
with open("/home/suyash/final_repo/HybridQA_Tab_MM/Hybrid_QA_MM/experiment_ready_dataset/text_passages.json","r") as f:
    text_passages = json.load(f)

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

c = 0
ctr = 0

BASE_DIR = f"/home/suyash/final_repo/gpt_test_samples/HybridQA_Tab_MM/{DATASET}"
OG_BASE_DIR = f"/home/suyash/final_repo/HybridQA_Tab_MM/{DATASET}"
IMAGE_PATH_DIR_TRAIN = f"/home/suyash/final_repo/temp_train_table_images_new/{DATASET}"
IMAGE_PATH_DIR_TEST = f"/home/suyash/final_repo/temp_test_table_images_new/{DATASET}"
BASE_DIR_1 = f"/home/suyash/final_repo/[temp]train_test_questions/{DATASET}"
BASE_DIR_700=f"/home/suyash/final_repo/redone_samples_700cap/{DATASET}"
BASE_DIR_GPT_HYB = f"/home/suyash/final_repo/gpt_test_samples/HybridQA_Tab_MM/Hybrid_QA_MM"
# %%
tables_dict = {}
train_questions = {}
test_questions = {}

with open(os.path.join(BASE_DIR_GPT_HYB, f"test_{QUESTION_TYPE}_questions.jsonl")) as f:
    lines = f.readlines()
    for line in lines:
        line_json = json.loads(line)
        test_questions[line_json["question_id"]] = line_json

with open(os.path.join(OG_BASE_DIR, "experiment_ready_dataset", "tables.jsonl")) as f:
    lines = f.readlines()
    for line in lines:
        line_json = json.loads(line)
        tables_dict[line_json["table_id"]] = line_json

# with open(os.path.join(BASE_DIR, f"train_{QUESTION_TYPE}_questions.jsonl")) as f:
#     lines = f.readlines()
#     for line in lines:
#         line_json = json.loads(line)
#         train_questions[line_json["question_id"]] = line_json

with open(os.path.join(OG_BASE_DIR, "experiment_ready_dataset", "image_id_to_image_path.json")) as f:
    image_id_to_image_path = json.load(f)

with open(os.path.join(OG_BASE_DIR, "experiment_ready_dataset", "image_id_to_original_string.json")) as f:
    image_id_to_original_string = json.load(f)

# %%
print("Train and test questions:", len(train_questions), len(test_questions))

with open(os.path.join(BASE_DIR_1, "tables_split.json")) as f:
    split = json.load(f)

train_tables = split['train']
train_exclusive_questions = [train_questions[qid] for qid in train_questions if train_questions[qid]["table_context"] in train_tables]
random.shuffle(train_exclusive_questions)

with open(os.path.join(OG_BASE_DIR, "experiment_ready_dataset", "image_id_to_original_string.json")) as f:
    image_id_to_original_string = json.load(f)

# %%
len(train_questions)

# %%
def create_metdata_prompt_sentence(page_title, table_headers):
    if pd.isna(table_headers):
        prompt = f"Table related to the {page_title}."
    else:
        prompt = f"Table related to {table_headers}."
    return prompt

def generate_table_string(table_array):
    table_string = ""

    for row_idx, row in enumerate(table_array):
        for col_idx, cell in enumerate(row):
            cell = cell.replace("\t", " ").replace("\n", " ").replace("\\n", " ").replace("\\t", " ").replace("|", " ")
            table_string = table_string  + cell + " | "
        table_string = table_string + "\n"

    return table_string


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


# %%
import re
import copy

def get_all_image_tags(text):
    pattern = r"\{IMG-\{.*?\}\}"
    matches = re.findall(pattern, text)
    return matches

def normalize_image_tags(table_string):
    table_string_copy = copy.deepcopy(table_string)
    image_ids = get_all_image_tags(table_string)
    image_ids = sorted(list(set(image_ids)))
    
    for image_idx, image_id in enumerate(image_ids):
        table_string_copy = table_string_copy.replace(image_id, f"{{ENTITY-{image_idx+1}}}")
    return table_string_copy

def convert_table_to_prompt(relevant_nodes,table_id, table_array, question, answer = None):
    if DATASET == "WikiTableQuestions":
        page_title, headers = get_wtq_metadata(table_id)
    elif DATASET == "WikiSQL":
        page_title = tables_dict[table_id]["page_title"]
        headers = tables_dict[table_id]["section_title"]
    elif DATASET == "fetaqa_MM_cleaned":
        page_title = tables_dict[table_id]["table_page_title"]
        headers = tables_dict[table_id]["table_section_title"]
    elif DATASET == "Hybrid_QA_MM":
        page_title = ""
        headers=(tables_dict[table_id]["url"].split("/")[-1]).replace("_"," ")
        

    table_metadata = create_metdata_prompt_sentence(page_title, headers)

    table_string = generate_table_string(table_array)
    
    table_string = normalize_image_tags(table_string)

    
    TABLE_TEXT = """Table context: {table_metadata}

Question: {question}
Answer: """
    # if answer is not None:
    #     TABLE_TEXT = TABLE_TEXT + """{answer}"""

    passage_dict = {}
    for elements in relevant_nodes:
        tup = elements[2]
        if tup in passage_dict.keys():
            passage_dict[tup].append((elements[1][0],elements[1][1]))
        else:
            passage_dict[tup]= [(elements[1][0],elements[1][1])]
    # print(passage_dict)
    for k,val in passage_dict.items():
    #   elements[0]=elements[0].replace("{","""{{""")
    #   elements[1]=elements[1].replace("}","""}}""")
      TABLE_TEXT+=f"Passage related to entity in "
    #   print("mkc1")
      for vals in val:
          TABLE_TEXT += f"row {vals[0]} and column {vals[1]}, "     
          
      if k in mm_passages.keys():
            mm_passages[k]=mm_passages[k].replace("{","""{{""").replace("}","""}}""")

            TABLE_TEXT+=f"{mm_passages[k]}. "
            # print("mkc2")
      elif k in text_passages.keys():
            text_passages[k]=text_passages[k].replace("{","""{{""").replace("}","""}}""")
            TABLE_TEXT+=f"{text_passages[k]}. "
            # print("mkc3")
    table_metadata=table_metadata.replace("{","""{{""")
    table_metadata=table_metadata.replace("}","""}}""")
    question=question.replace("{","""{{""")
    question=question.replace("}","""}}""")
    # answer=answer.replace("{","""{{""")
    # answer=answer.replace("}","""}}""")
    # print(TABLE_TEXT)
    return TABLE_TEXT.format(table_metadata=table_metadata, question=question)


# %%
import re
from collections import defaultdict

def make_imageids_uniform(table_array):
    imagepath_to_imageid = defaultdict(list)
    
    for row_idx, row in enumerate(table_array):
        for col_idx, cell in enumerate(row):
            image_ids = get_all_image_tags(cell)
            for image_id in image_ids:
                imagepath = image_id_to_image_path[image_id]
                imagepath_to_imageid[imagepath].append(image_id)
    
    modified_table_array = []
    for row_idx, row in enumerate(table_array):
        modified_row = []
        for col_idx, cell in enumerate(row):
            image_ids = get_all_image_tags(cell)
            for image_id in image_ids:
                imagepath = image_id_to_image_path[image_id]
                rep_imageid = imagepath_to_imageid[imagepath][0]
                cell = cell.replace(image_id, rep_imageid)
            modified_row.append(cell)
        modified_table_array.append(modified_row)
    return modified_table_array

def get_all_image_tags(text):
    pattern = r"\{IMG-\{.*?\}\}"
    matches = re.findall(pattern, text)
    return matches


# %%
examples = []
example_table_images = []
ctr = 0
# for ques in train_exclusive_questions:
#     ctr+=1
#     if ctr==20:
#         break
#     table_id = ques["table_context"]
#     table_array = tables_dict[table_id]["table_array"]
#     uniform_table_array = make_imageids_uniform(table_array)
#     passage_arr = tables_dict[table_id]["cells_to_link"]
#     flag = False
#     ans = ques["answer-text"]
#     if DATASET == "WikiSQL":
#         for x in ques["answer"]:
#             if not isinstance(x,str):
#                 flag = True
#                 break
#         ansArr=ques["answer"]
#         if flag is True:
#             ansArr = [str(num) for num in ques["answer"]]
#         ans = ", ".join(ansArr)
    
    
        
#     prompt = convert_table_to_prompt(passage_arr,table_id, uniform_table_array, ques["question"], ans)
#     # if ("team" in prompt.lower()) or ("country" in prompt.lower()) or ("city" in prompt.lower()) or ("competiton" in prompt.lower()): 
    
#     file_name = table_id.replace("/", "_")
    
    
#     table_image_path = os.path.join(IMAGE_PATH_DIR_TRAIN, f"{file_name}.jpg")
#     # if not os.path.exists(table_image_path):
#     #     print("mkc")
#     if not os.path.exists(table_image_path):
#             # print("mkc",table_image_path)
#             continue
#     example_table_images.append(table_image_path)
#     examples.append(prompt)
    
    
    
# # random.shuffle(examples)
# zipped_list = list(zip(examples, example_table_images))

# # Step 2: Shuffle the zipped list
# random.shuffle(zipped_list)

# Step 3: Unzip the shuffled list back into two lists
# shuffled_list1, shuffled_list2 = zip(*zipped_list)

# # Convert the zipped lists back to lists (if you need them as lists)
# examples = list(shuffled_list1)
# example_table_images = list(shuffled_list2)
# ans_dict = run_gemini(examples,example_table_images)
# for ans in ((ans_dict)):
 
#     print("Text: ",ans["text"])
#     print("Image: ",ans["image"])
#     print("Answer: ",ans["response"])

# with open(PROMPT_OUT_FILE,"w") as f:
#     json.dump(ans_dict,f)
# print(examples[0:20])
# print(example_table_images[0:20])   

# %%

qid_to_prompts = {}
train_saved = []
with open("/home/suyash/final_repo/modelling/table-image-approach/few_shot_1.json",'r') as f:
    train_saved = json.load(f)
    # for line in f:
    #     try:
    #         json_obj = json.loads(line)
    #         train_saved.append(json_obj)
    #     except Exception as e:
    #         print(e)

# print(train_saved[0])  
ex_foramts = [ex["text"].split("Answer: ")[0] + "Step 1 : " + ex["response"] + " Step 2 : "+ (ex["text"].split("Answer: ")[-1]).split("##")[0] for ex in train_saved]
 
for qid, ques in test_questions.items():
    table_id = ques["table_context"]
    table_array = tables_dict[table_id]["table_array"]
    uniform_table_array = make_imageids_uniform(table_array)
#     TEXT_PROMPT = """You are given a table in the form of an image. In the table, some entities (mentioned in text form originally) have been replaced by images that represent them. Based upon the context of the table while using real-world knowledge, your task is to identify the entities corresponding to the images in the table and answer the question. You must perform this task in the following steps:

# Step 1: Reason about what should be the answer to the question by identifying the relevant entities represented by images using the context of the table and the question. The reasoning should be detailed and should be based upon the context of the table and the question, using real-world knowledge for answering the question. IMPORTANT: You must explore any kind of reasoning -- numerical, logical, knowledge-based needed for disambiguating the entities and answering the question.
# Step 2: Based upon the reasoning provided, provide the answer to the question.

# You are also provided with some question-answer examples for better understanding the format of providing the answer:

# Example 1:
# Table context: Table related to Early computer characteristics in context of History of computing hardware.\n\nQuestion: what was the next operational computer after the modified eniac?\nStep 1: To determine the next operational computer after the modified ENIAC, we look for computers that became operational after mid-1948. The Manchester Small-Scale Experimental Machine (Baby) was operational in June 1948, and its development led to the Manchester Mark 1, which became operational in April 1949. Therefore, the next significant computer after the modified ENIAC was the Manchester Mark 1.\n\nStep 2:\nManchester Mark 1

# Example 2:
# Table context: Table related to Junction list|MD 144WB in context of Maryland Route 144. \n\nQuestion: which ramps are not signed?\nStep 1: First, look across the Notes column to find out which rows have unsigned/not-signed mentioned. Then look for thes destinations in the Destinations column to get the relevant destinations. The following 3 destinations have unsigned ramp written in their notes - Berkeley Springs, WV|Limestone Road north|Virginia Avenue to US 522 north.\n\nStep 2:\nUS 522 south - Berkeley Springs, WV|Limestone Road north|Virginia Avenue to US 522 north

# Example 3:
# Table context: Table related to the 1982 Toronto Blue Jays season.\n\nQuestion: leal won on july 23, but when was the blue jays previous win?\nStep 1 : The table shows the win-loss record for the Toronto Blue Jays throughout the 1982 season [evidenced by the header "W" and "L" at the top of the rightmost column]. Dates are listed along the leftmost side of the table [evident by dates written on the left side of the table]. Under the "W" column, there are marks indicating wins [represented by "I" symbols]. Looking at July, we can see that the Blue Jays have wins marked on July 17, 20 and 23 [referring to the "I" symbols under the "W" column next to "17", "20" and "23" on the left side of the table]. Since there are no wins marked before July 20, we can conclude that the Blue Jays previous win was on July 20.\n\nStep 2:\nJuly 20

# Example 4:
# Table context: Table related to Fifth round proper in context of 1975–76 FA Cup.\n\nQuestion: how many games played by sunderland are listed here?\nStep 1 : We can conclude Sunderland played in 2 games. The table shows teams listed under "Home team" and "Away team" columns [column headers provide this information].  Looking across the rows, Sunderland's logo, which comprises of 2 horses to the side and a Black&White sheild in between, is listed under one of these columns twice [in the 2nd and 3rd row]. Therefore, Sunderland participated in two games.\n\nStep 2:\n2

# Example 5:
# Table context: Table related to Ships in class in context of Leander-class cruiser (1931).\n\nQuestion: which ships have a pennant number higher than 50?\nStep 1 :The table  has a "Pennant" column showing the pennant number of various ships. By examining its values over various rows, we filter out the rows which have value greater than 50. Then under the Namesake column, the names of the ships corresponding to the filtered rows are - Leander|Orion|Achilles|Apollo.\n\nStep 2:\nLeander|Orion|Achilles|Apollo

# Example 6:
# Table context: Table related to Complete Formula One World Championship results in context of Playlife.\n\nQuestion: when was the benetton b198 chassis used?\nStep 1: The table shows Formula One results with a context of Playlife, possibly a constructor. As we can see, the Benetton b198 Chassis is the blue coloured supporting structure, as seen in the chassis column of the table. In the same row, there is the column year, which gives us the answer as 1998.\n\nStep 2: 1998.  

# Example 7:
# Table context: Table related to Defunct railroads in context of List of Washington, D.C., railroads.\n\nQuestion: was the pennsylvania railroad under the prr or the rf&p?\nStep 1 : The table shows defunct railroads in Washington D.C. The "Pennsylvania Railroad" is the golden background picture represented in the 11th row with trains visible in it. In the same row, another column named "Mark" has the abbreviation as "PRR". Thus, the Pennsylvania Railroad operated under PRR since "PRR" is its short name.\n\nStep 2:PRR

# Example 8:
# Table context: Table related to Schedule in context of 1996 Frankfurt Galaxy season.\n\nQuestion: how many games were played t waldstadion?\nStep 1 : The table shows the Frankfurt Galaxy's 1996 season schedule, likely indicating locations or stadiums. Look for a "Location" or "Stadium" column and then find entries mentioning "Waldstadion" in that column. Counting these entries will reveal the number of games played at Waldstadion since the schedule shows where each game took place.\n\nStep 2: 5

# Now, based upon the examples given above, you must understand the image given and follow the steps 1-2 to answer the question corresponding to the table represented in the image. It is IMPORTANT that you perform all the both the steps to the best possible extent to get the correct answer. You must follow the format of answers as demonstrated by the examples above. IMPORTANT: You must give the answer in the format 'Step 2:\n<answer>'.

# {main_part}
# """
    TEXT_PROMPT_FETA = """You are given a table in the form of an image. In the table, some entities (mentioned in text form originally) have been replaced by images that represent them. Based upon the context of the table while using real-world knowledge, your task is to identify the entities corresponding to the images in the table and answer the question. You must perform this task in the following steps:

Step 1: Reason about what should be the answer to the question by identifying the relevant entities represented by images using the context of the table and the question. The reasoning should be detailed and should be based upon the context of the table and the question, using real-world knowledge for answering the question. IMPORTANT: You must explore any kind of reasoning -- numerical, logical, knowledge-based needed for disambiguating the entities and answering the question.
Step 2: Based upon the reasoning provided, provide the answer to the question.

You are also provided with some question-answer examples for better understanding the format of providing the answer:

Example 1:
Table context: Table related to NHL awards in context of 2013–14 NHL season.\n\nQuestion: Which teams were competing for the Stanley Cup in the 2013-14 NHL season?\nStep 1 : The Stanley Cup is the silver-coloured cup represented in the first row, Award column. In the same row under the reciepient's column, we can see a Black-coloured logo with LA written on it, which is the logo for <>.Also in the runners-up column, we can see a Blue-coloured logo with "New-York Rangers" written on it. Thus, we can conclude that The Los Angeles Kings won the Stanley Cup, defeating the New York Rangers.\n\nStep 2:\nThe Los Angeles Kings won the Stanley Cup, defeating the New York Rangers.

Example 2:
Table context: Table related to International competitions in context of Debbie Marti.\n\nQuestion: In which city was the 1991 World Championships held and what distance did Debbie Marti achieve to qualify?\nStep 1: As we can see in the 5th row, the Competiton represented in the Competitions column by a Blue-coloured logo is the World Championships. In the same row, under the venue column, we can see a collage of pictures of the prominent buildings from Tokyo. Thus We can conclude that the venue of the competiton was Tokyo. Also, in the column "Notes", we can see that Debbie Marti qualified with 1.86m.\n\nStep 2: At the 1991 World Championships in Tokyo, Debbie Marti qualified with 1.86 m.  

Example 3:
Table context: Table related to Awards and nominations in context of Project Gutenberg (film).\n\nQuestion: What awards did Project Gutenberg win at the 38th Hong Kong Film Awards?\nStep 1 : The answer to the question can be found by looking at the column titled "Award" in the table. We can infer that there are seven rows in the table, each corresponding to an award won by Project Gutenberg. The categories listed are (Best Film, Best Director, Best Screenplay, Best Cinematography, Best Film Editing, Best Art Direction, and Best Costume Make Up Design) exactly match up to the categories listed in the "Award" column. So, to find the answer, you would need to look for each of these categories in the "Award" column and see which movie title is listed next to it.\n\nStep 2:Project Gutenberg won seven awards at the 38th Hong Kong Film Awards, in the categories Best Film, Best Director, Best Screenplay, Best Cinematography, Best Film Editing, Best Art Direction, and Best Costume Make Up Design.

Example 4:
Table context: Table related to Awards and nominations in context of Mike Cahill (director).\n\nQuestion: What film won the Alfred P. Sloan Prize at the Sundance Film Festival in 2014?\nStep 1 : Look at the "Year" column and find the year 2014. Then, look at the "Award" column for that row. If it says "Alfred P. Sloan Prize", then the movie title in the "Film" column for that row is the answer. In the table you described, on the row where "Year" is 2014, "Award" is "Alfred P. Sloan Prize", and "Film" is "I Origins".\n\nStep 2:\nCahill's film I Origins again won the Alfred P. Sloan Prize at the 2014 Sundance Film Festival, his second time receiving the award.

Example 5:
Table context: Table related to Home attendances in context of 2012–13 Everton F.C. season.\n\nQuestion: How did Everton F.C. do against Manchester United and Tottenham Hotspur during their 2012-13 season?\nStep 1 : Look for Manchester United and Tottenham Hotspur on the “Opponent” column.  Look at the corresponding “Score” for each team. For Manchester United, the score is  1-0 in favor of Everton. For Tottenham Hotspur, the score is 2-1 in favor of Everton. Therefore, Everton won against both Manchester United and Tottenham Hotspur.\n\nStep 2:\nEverton F.C. won over Manchester United in the first game of the season with 1–0, defeated Tottenham Hotspur 2–1, and defeated Manchester City 2–0 in the Premier League.

Example 6:
Table context: Table related to International competitions in context of Süreyya Ayhan.\n\nQuestion: How did Sureyya Ayhan fare at the 2003 World Championships?\nStep 1 : Look for "2003" in the "Year" column. Look across that row to the "Competition" column. It should say "World Championships". In the "Event" column, it shows "1500 m". Finally, under the "Position" column, it shows "2nd", indicating that Süreyya Ayhan won a silver medal.\n\nStep 2:\nSüreyya Ayhan won a silver medal in the 1500 m of the 2003 World Championships.

Example 7:
Table context: Table related to Grammy Awards in context of Roberta Flack.\n\nQuestion: When and for which songs did the singer Roberta Flack win Grammy Awards for Record of they Year?\nStep 1 : Looking at the table under the "Year" column, you can see 1973 listed twice.  In the corresponding rows under "Award" it says "Record of the Year" each time.  Looking at the "Nominee / work" column for those two rows, it shows "The First Time Ever I Saw Your Face" in 1973 and "Killing Me Softly With His Song" in 1974.  This confirms that Flack won the award for these two songs in consecutive years.\n\nStep 2:\nFlack won the Grammy Award for Record of the Year on two consecutive years: "The First Time Ever I Saw Your Face" won at the 1973 Grammys as did "Killing Me Softly with His Song" at the 1974 Grammys.

Example 8:
Table context: Table related to Television series in context of Kim Jung-hyun (actor, born 1990).\n\nQuestion: What did Kim Jung-hyun do in KBS2 in 2017?\nStep 1 : Look for the year "2017" in the "Year" column. Look across that row to the "Network" column. It should say "KBS2". In the "Title" column, it shows "School 2017". This indicates that Kim Jung-hyun played in that drama in 2017 on KBS2.\n\nStep 2:\In 2017, Kim Jung-hyun played in KBS2's School 2017.

Now, based upon the examples given above, you must understand the image given and follow the steps 1-2 to answer the question corresponding to the table represented in the image. It is IMPORTANT that you perform all the both the steps to the best possible extent to get the correct answer. You must follow the format of answers as demonstrated by the examples above. IMPORTANT: You must give the answer in the format 'Step 2:\n<answer>'.

{main_part}"""
    TEXT_PROMPT_WTQ = """You are given a table in the form of an image. In the table, some entities (mentioned in text form originally) have been replaced by images that represent them. Based upon the context of the table while using real-world knowledge, your task is to identify the entities corresponding to the images in the table and answer the question. You must perform this task in the following steps:

Step 1: Reason about what should be the answer to the question by identifying the relevant entities represented by images using the context of the table and the question. The reasoning should be detailed and should be based upon the context of the table and the question, using real-world knowledge for answering the question. IMPORTANT: You must explore any kind of reasoning -- numerical, logical, knowledge-based needed for disambiguating the entities and answering the question.
Step 2: Based upon the reasoning provided, provide the answer to the question.

You are also provided with some question-answer examples for better understanding the format of providing the answer:

Example 1:
Table context: Table related to Fifth round proper in context of 1975–76 FA Cup.\n\nQuestion: how many games played by sunderland are listed here?\nStep 1 : We can conclude Sunderland played in 2 games. The table shows teams listed under "Home team" and "Away team" columns [column headers provide this information].  Looking across the rows, Sunderland's logo, which comprises of 2 horses to the side and a Black&White sheild in between, is listed under one of these columns twice [in the 2nd and 3rd row]. Therefore, Sunderland participated in two games.\n\nStep 2:\n2

Example 2:
Table context: Table related to Complete Formula One World Championship results in context of Playlife.\n\nQuestion: when was the benetton b198 chassis used?\nStep 1: The table shows Formula One results with a context of Playlife, possibly a constructor. As we can see, the Benetton b198 Chassis is the blue coloured supporting structure, as seen in the chassis column of the table. In the same row, there is the column year, which gives us the answer as 1998.\n\nStep 2: 1998.  

Example 3:
Table context: Table related to Defunct railroads in context of List of Washington, D.C., railroads.\n\nQuestion: was the pennsylvania railroad under the prr or the rf&p?\nStep 1 : The table shows defunct railroads in Washington D.C. The "Pennsylvania Railroad" is the golden background picture represented in the 11th row with trains visible in it. In the same row, another column named "Mark" has the abbreviation as "PRR". Thus, the Pennsylvania Railroad operated under PRR since "PRR" is its short name.\n\nStep 2:PRR

Example 4:
Table context: Table related to Schedule and results in context of 2013–14 Chicago State Cougars women's basketball team.\n\nQuestion: how many games were played against grand canyon?\nStep 1 : We can see that there are 2 instances of the grand canyon in the opponent column. One in the 20th row, where there is a purple coloured logo which says GCC, which refers to the Grand Canyon College. Another is in the 26th row. Thus, we can conclude that 2 matches were played against the grand canyon.\n\nStep 2:\n2

Example 5:
Table context: Table related to Roster|Letter winners in context of 1915 Michigan Wolverines football team.\n\nQuestion: how many players were taller and weighed more than frank millard?\nStep 1 : Frank Millard is the clean-shaved, short haired guy visible in the 5th row. His height is 5'7 and weight is 212. Thus clearly, there are only 2 players whose height and weight is more than his, one in the 2nd row and other in the 8th row.\n\nStep 2:\n2

Example 6:
Table context: Table related to Racing record|Career summary in context of Conor Daly.\n\nQuestion: the two teams who raced in 2011 are carlin motorsport and what other team?\nStep 1 : In the year column there are 2 rows which have a mention of 2011. Apart from Carlin motorsport, the other one has a green car with the logo Schmidt Motorsports on it.\n\nStep 2:\nSchmidt Motorsports

Example 7:
Table context: Table related to Regular season|Schedule in context of 1995 New York Jets season.\n\nQuestion: team that scored more than 40 points against the jets that is not the miami dolphins\nStep 1 : As clearly visible, the opponent mentioned in the 4th row, which has a Black-coloured logo written as "RAIDERS" on it scored 47 goals against the jets. The logo is of the team Oakland Raiders. Thus, We can conclude that Oakland Raiders is the other team that scored 47 goals against the jets.\n\nStep 2:\nOakland Raiders

Example 8:
Table context: Table related to Winners|By Country in context of EHF Cup Winners' Cup.\n\nQuestion: did france or croatia have a larger finals total?\nStep 1 : Under the country column, in the 5th row we can see a Blue-coloured chicken logo, with FFHANDBALL written under it. That is the logo for France's handball federation. In the 8th row, we can see and Red-Blue coloured handall logo, with the Croatia Handball federation written underneath it, which represents Croatia. Thus, we can conclude that France had more Finals Total, 4, than Croatia, 1.\n\nStep 2:\nFrance

Now, based upon the examples given above, you must understand the image given and follow the steps 1-2 to answer the question corresponding to the table represented in the image. It is IMPORTANT that you perform all the both the steps to the best possible extent to get the correct answer. You must follow the format of answers as demonstrated by the examples above. IMPORTANT: You must give the answer in the format 'Step 2:\n<answer>'.

{main_part}
"""
    TEXT_PROMPT_WIKISQL = """You are given a table in the form of an image. In the table, some entities (mentioned in text form originally) have been replaced by images that represent them. Based upon the context of the table while using real-world knowledge, your task is to identify the entities corresponding to the images in the table and answer the question. You must perform this task in the following steps:

Step 1: Reason about what should be the answer to the question by identifying the relevant entities represented by images using the context of the table and the question. The reasoning should be detailed and should be based upon the context of the table and the question, using real-world knowledge for answering the question. IMPORTANT: You must explore any kind of reasoning -- numerical, logical, knowledge-based needed for disambiguating the entities and answering the question.
Step 2: Based upon the reasoning provided, provide the answer to the question.

You are also provided with some question-answer examples for better understanding the format of providing the answer:

Example 1:
Table related to Teams and venues in context of 2004 Belarusian Premier League.\n\nQuestion:  What is the Location for Belshina?\nStep 1 : The table shows the teams and their corresponding locations. To find the location for Belshina, we need to locate Belshina in the 'Team' column. Belshina is located in row 9, and its location is given in the 'Location' column as 'Bobruisk'. Therefore, the location for Belshina is Bobruisk.\n\nStep 2 : bobruisk

Example 2:
Table related to Made the cut in context of 2009 U.S. Open (golf).\n\nQuestion:  what is the to par for retief goosen?\nStep 1 : The question asks for the "to par" for Retief Goosen. Looking at the table, we can see that Retief Goosen is listed on row 1, and the "To par" column for row 1 is +3. Therefore, the answer is +3.\n\nStep 2 : +3

Example 3:
Table related to Round 16 in context of 1969 VFL season.\n\nQuestion:  If the Venue was kardinia park what was the highest Crowd attended?\nStep 1 : The question asks for the highest crowd that attended a game at Kardinia Park. Looking at the table, we see that Kardinia Park is listed as the venue for round 3.  The crowd for that game was 16,211. The question asked for the highest crowd.  Looking at the rest of the table, the only higher crowd is 21,025.  Therefore, 21,025 is the answer.\n\nStep 2 : 21,025

Example 4:
Table related to Prime ministers in context of Interwar unemployment and poverty in the United Kingdom.\n\nQuestion:  What is the birth place of the prime minister who served George V and entered office on 23 October 1922?\nStep 1 : The answer can be arrived by locating the row where the prime minister entered office on 23 October 1922. This is the row corresponding to Andrew Bonar Law. Looking at the 'Birth Place' column for this row, the answer is given as "Rexton, Kent County, New Brunswick, Canada".\n\nStep 2 : rexton, kent county, new brunswick, canada

Example 5:
Table related to Round 12 in context of 1976 VFL season.\n\nQuestion:  What date did North Melbourne score 22.14 (146) as the home team?\nStep 1 : The question asks what date North Melbourne scored 22.14 (146) as the home team. We can find the row that corresponds to North Melbourne by finding the row with their logo. Row 5 shows their logo, which shows a Blue-coloured kangaroo with "North Melbourne" written on it, and their home score, which is 22.14 (146). In the last column, the date is 19 June 1976. Therefore, the answer to the question is 19 June 1976.\n\nStep 2 : 19 june 1976

Example 6:
Table related to Game Log in context of 1982 Atlanta Braves season.\n\nQuestion:  Where was the game on Thursday, April 29, and they played the Chicago Cubs?\nStep 1 : The table shows the 1982 Atlanta Braves game log. We need to find the game on Thursday, April 29. We see this date on row 19. Looking at the "Opponent" column, we see the Chicago Cubs. It is represented by a Purple background logo with a Cwritten in Red colour. Therefore, the game was played at the location listed in the "Site" column for row 19: Atlanta-Fulton County Stadium.\n\nStep 2 : atlanta-fulton county stadium

Example 7:
Table related to Medal table in context of 2007 Military World Games.\n\nQuestion:  Which nation has a Silver of 1, a Gold of 0, and a Total of 1?\nStep 1 : The question asks for the nation with 1 silver medal, 0 gold medals, and a total of 1 medal.Looking at the table, we can see that Bulgaria, Cameroon and Hungary, which are represented by a White-Green_Red striped flag, Green-Red-Tellow flag with a star, and a Red-White Green striped flag respectively, have 1 silver medal, 0 gold medals and a total of 1 medal.\n\nStep 2 : bulgaria, cameroon, hungary

Example 8:
Table related to Round 4 in context of 1956 VFL season.\n\nQuestion:  Which home team scored 12.15 (87)?\nStep 1 : The table shows the scores of different teams in round 4 of the 1956 VFL season. The row with the home team score of 12.15 (87) is the second row which corresponds to Collingwood. Therefore, the home team that scored 12.15 (87) is Collingwood.\n\nStep 2 : collingwood

Now, based upon the examples given above, you must understand the image given and follow the steps 1-2 to answer the question corresponding to the table represented in the image. It is IMPORTANT that you perform all the both the steps to the best possible extent to get the correct answer. You must follow the format of answers as demonstrated by the examples above. IMPORTANT: You must give the answer in the format 'Step 2:\n<answer>'.

{main_part}"""

    TEXT_PROMPT_WTQ_Visual = """You are given a table in the form of an image. In the table, some entities (mentioned in text form originally) have been replaced by images that represent them. Based upon the context of the table while using real-world knowledge, your task is to identify the entities corresponding to the images in the table and answer the question. You must perform this task in the following steps:

Step 1: Reason about what should be the answer to the question by identifying the relevant entities represented by images using the context of the table and the question. The reasoning should be detailed and should be based upon the context of the table and the question, using real-world knowledge for answering the question. IMPORTANT: You must explore any kind of reasoning -- numerical, logical, knowledge-based needed for disambiguating the entities and answering the question.
Step 2: Based upon the reasoning provided, provide the answer to the question.

You are also provided with some question-answer examples for better understanding the format of providing the answer:

Example 1:
Table context: Table related to Fifth round proper in context of 1975–76 FA Cup.\n\nQuestion: how many games played by the team represented by a red logo with two lions by its side are listed here?\nStep 1 : We can conclude that the team represented by the Red logo, with two lions by its side, played in 2 games. The table shows teams listed under "Home team" and "Away team" columns [column headers provide this information].  Looking across the rows, Sunderland's logo, which comprises of 2 horses to the side and a Black&White sheild in between, is listed under one of these columns twice [in the 2nd and 3rd row]. Therefore, the team represented by the Red logo, with two lions by its side, participated in two games.\n\nStep 2:\n2

Example 2:
Table context: Table related to Complete Formula One World Championship results in context of Playlife.\n\nQuestion: when was the benetton b198 chassis used?\nStep 1: The table shows Formula One results with a context of Playlife, possibly a constructor. As we can see, the Benetton b198 Chassis is the blue coloured supporting structure, as seen in the chassis column of the table. In the same row, there is the column year, which gives us the answer as 1998.\n\nStep 2: 1998.  

Example 3:
Table context: Table related to Defunct railroads in context of List of Washington, D.C., railroads.\n\nQuestion: was the pennsylvania railroad under the prr or the rf&p?\nStep 1 : The table shows defunct railroads in Washington D.C. The "Pennsylvania Railroad" is the golden background picture represented in the 11th row with trains visible in it. In the same row, another column named "Mark" has the abbreviation as "PRR". Thus, the Pennsylvania Railroad operated under PRR since "PRR" is its short name.\n\nStep 2:PRR

Example 4:
Table context: Table related to Schedule and results in context of 2013–14 Chicago State Cougars women's basketball team.\n\nQuestion: how many games were played against grand canyon?\nStep 1 : We can see that there are 2 instances of the grand canyon in the opponent column. One in the 20th row, where there is a purple coloured logo which says GCC, which refers to the Grand Canyon College. Another is in the 26th row. Thus, we can conclude that 2 matches were played against the grand canyon.\n\nStep 2:\n2

Example 5:
Table context: Table related to Roster|Letter winners in context of 1915 Michigan Wolverines football team.\n\nQuestion: how many players were taller and weighed more than frank millard?\nStep 1 : Frank Millard is the clean-shaved, short haired guy visible in the 5th row. His height is 5'7 and weight is 212. Thus clearly, there are only 2 players whose height and weight is more than his, one in the 2nd row and other in the 8th row.\n\nStep 2:\n2

Example 6:
Table context: Table related to Racing record|Career summary in context of Conor Daly.\n\nQuestion: the two teams who raced in 2011 are carlin motorsport and what other team?\nStep 1 : In the year column there are 2 rows which have a mention of 2011. Apart from Carlin motorsport, the other one has a green car with the logo Schmidt Motorsports on it.\n\nStep 2:\nSchmidt Motorsports

Example 7:
Table context: Table related to Regular season|Schedule in context of 1995 New York Jets season.\n\nQuestion: team that scored more than 40 points against the jets that is not the miami dolphins\nStep 1 : As clearly visible, the opponent mentioned in the 4th row, which has a Black-coloured logo written as "RAIDERS" on it scored 47 goals against the jets. The logo is of the team Oakland Raiders. Thus, We can conclude that Oakland Raiders is the other team that scored 47 goals against the jets.\n\nStep 2:\nOakland Raiders

Example 8:
Table context: Table related to Winners|By Country in context of EHF Cup Winners' Cup.\n\nQuestion: did the country with Blue-coloured chicken logo, with FFHANDBALL written under it or the country with Red-Blue coloured handall logo, with the Croatia Handball federation written underneath it have a larger finals total?\nStep 1 : Under the country column, in the 5th row we can see a Blue-coloured chicken logo, with FFHANDBALL written under it. That is the logo for France's handball federation. In the 8th row, we can see and Red-Blue coloured handall logo, with the Croatia Handball federation written underneath it, which represents Croatia. Thus, we can conclude that France had more Finals Total, 4, than Croatia, 1.\n\nStep 2:\nFrance

Now, based upon the examples given above, you must understand the image given and follow the steps 1-2 to answer the question corresponding to the table represented in the image. It is IMPORTANT that you perform all the both the steps to the best possible extent to get the correct answer. You must follow the format of answers as demonstrated by the examples above. IMPORTANT: You must give the answer in the format 'Step 2:\n<answer>'.

{main_part}
"""
    TEXT_PROMPT_FETA_visual = """You are given a table in the form of an image. In the table, some entities (mentioned in text form originally) have been replaced by images that represent them. Based upon the context of the table while using real-world knowledge, your task is to identify the entities corresponding to the images in the table and answer the question. You must perform this task in the following steps:

Step 1: Reason about what should be the answer to the question by identifying the relevant entities represented by images using the context of the table and the question. The reasoning should be detailed and should be based upon the context of the table and the question, using real-world knowledge for answering the question. IMPORTANT: You must explore any kind of reasoning -- numerical, logical, knowledge-based needed for disambiguating the entities and answering the question.
Step 2: Based upon the reasoning provided, provide the answer to the question.

You are also provided with some question-answer examples for better understanding the format of providing the answer:

Example 1:
Table context: Table related to NHL awards in context of 2013–14 NHL season.\n\nQuestion: Which teams were competing for the Stanley Cup in the 2013-14 NHL season?\nStep 1 : The Stanley Cup is the silver-coloured cup represented in the first row, Award column. In the same row under the reciepient's column, we can see a Black-coloured logo with LA written on it, which is the logo for <>.Also in the runners-up column, we can see a Blue-coloured logo with "New-York Rangers" written on it. Thus, we can conclude that The Los Angeles Kings won the Stanley Cup, defeating the New York Rangers.\n\nStep 2:\nThe Los Angeles Kings won the Stanley Cup, defeating the New York Rangers.

Example 2:
Table context: Table related to International competitions in context of Debbie Marti.\n\nQuestion: In which city was the competiton represented by the Blue-coloured logo  held and what distance did Debbie Marti achieve to qualify?\nStep 1: As we can see in the 5th row, the Competiton represented in the Competitions column by a Blue-coloured logo is the World Championships. In the same row, under the venue column, we can see a collage of pictures of the prominent buildings from Tokyo. Thus We can conclude that the venue of the competiton was Tokyo. Also, in the column "Notes", we can see that Debbie Marti qualified with 1.86m.\n\nStep 2: At the 1991 World Championships in Tokyo, Debbie Marti qualified with 1.86 m.  

Example 3:
Table context: Table related to Awards and nominations in context of Project Gutenberg (film).\n\nQuestion: What awards did Project Gutenberg win at the 38th Hong Kong Film Awards?\nStep 1 : The answer to the question can be found by looking at the column titled "Award" in the table. We can infer that there are seven rows in the table, each corresponding to an award won by Project Gutenberg. The categories listed are (Best Film, Best Director, Best Screenplay, Best Cinematography, Best Film Editing, Best Art Direction, and Best Costume Make Up Design) exactly match up to the categories listed in the "Award" column. So, to find the answer, you would need to look for each of these categories in the "Award" column and see which movie title is listed next to it.\n\nStep 2:Project Gutenberg won seven awards at the 38th Hong Kong Film Awards, in the categories Best Film, Best Director, Best Screenplay, Best Cinematography, Best Film Editing, Best Art Direction, and Best Costume Make Up Design.

Example 4:
Table context: Table related to Awards and nominations in context of Mike Cahill (director).\n\nQuestion: What film won the Alfred P. Sloan Prize at the Sundance Film Festival in 2014?\nStep 1 : Look at the "Year" column and find the year 2014. Then, look at the "Award" column for that row. If it says "Alfred P. Sloan Prize", then the movie title in the "Film" column for that row is the answer. In the table you described, on the row where "Year" is 2014, "Award" is "Alfred P. Sloan Prize", and "Film" is "I Origins".\n\nStep 2:\nCahill's film I Origins again won the Alfred P. Sloan Prize at the 2014 Sundance Film Festival, his second time receiving the award.

Example 5:
Table context: Table related to Home attendances in context of 2012–13 Everton F.C. season.\n\nQuestion: How did the team with the blue-coloured logo wiith a hut drawn over it, do against the team with a Red-coloured logo and a devil drawn over it, and the team with a white background logo, with a Blue coloured bird standing on a ball during their 2012-13 season?\nStep 1 : Looking for the Red logo and Whiite logo (which represent Manchester United and Tottenham Hotspur resp.) on the “Opponent” column.  Look at the corresponding “Score” for each team. For Manchester United, the score is  1-0 in favor of Everton. For Tottenham Hotspur, the score is 2-1 in favor of Everton. Therefore, Everton won against both Manchester United and Tottenham Hotspur.\n\nStep 2:\nEverton F.C. won over Manchester United in the first game of the season with 1–0, defeated Tottenham Hotspur 2–1, and defeated Manchester City 2–0 in the Premier League.

Example 6:
Table context: Table related to International competitions in context of Süreyya Ayhan.\n\nQuestion: How did Sureyya Ayhan fare at the 2003 World Championships?\nStep 1 : Look for "2003" in the "Year" column. Look across that row to the "Competition" column. It should say "World Championships". In the "Event" column, it shows "1500 m". Finally, under the "Position" column, it shows "2nd", indicating that Süreyya Ayhan won a silver medal.\n\nStep 2:\nSüreyya Ayhan won a silver medal in the 1500 m of the 2003 World Championships.

Example 7:
Table context: Table related to Grammy Awards in context of Roberta Flack.\n\nQuestion: When and for which songs did the singer Roberta Flack win Grammy Awards for Record of they Year?\nStep 1 : Looking at the table under the "Year" column, you can see 1973 listed twice.  In the corresponding rows under "Award" it says "Record of the Year" each time.  Looking at the "Nominee / work" column for those two rows, it shows "The First Time Ever I Saw Your Face" in 1973 and "Killing Me Softly With His Song" in 1974.  This confirms that Flack won the award for these two songs in consecutive years.\n\nStep 2:\nFlack won the Grammy Award for Record of the Year on two consecutive years: "The First Time Ever I Saw Your Face" won at the 1973 Grammys as did "Killing Me Softly with His Song" at the 1974 Grammys.

Example 8:
Table context: Table related to Television series in context of Kim Jung-hyun (actor, born 1990).\n\nQuestion: What did Kim Jung-hyun do in KBS2 in 2017?\nStep 1 : Look for the year "2017" in the "Year" column. Look across that row to the "Network" column. It should say "KBS2". In the "Title" column, it shows "School 2017". This indicates that Kim Jung-hyun played in that drama in 2017 on KBS2.\n\nStep 2:\In 2017, Kim Jung-hyun played in KBS2's School 2017.

Now, based upon the examples given above, you must understand the image given and follow the steps 1-2 to answer the question corresponding to the table represented in the image. It is IMPORTANT that you perform all the both the steps to the best possible extent to get the correct answer. You must follow the format of answers as demonstrated by the examples above. IMPORTANT: You must give the answer in the format 'Step 2:\n<answer>'.

{main_part}"""
    TEXT_PROMPT_WIKISQL_visual = """You are given a table in the form of an image. In the table, some entities (mentioned in text form originally) have been replaced by images that represent them. Based upon the context of the table while using real-world knowledge, your task is to identify the entities corresponding to the images in the table and answer the question. You must perform this task in the following steps:

Step 1: Reason about what should be the answer to the question by identifying the relevant entities represented by images using the context of the table and the question. The reasoning should be detailed and should be based upon the context of the table and the question, using real-world knowledge for answering the question. IMPORTANT: You must explore any kind of reasoning -- numerical, logical, knowledge-based needed for disambiguating the entities and answering the question.
Step 2: Based upon the reasoning provided, provide the answer to the question.

You are also provided with some question-answer examples for better understanding the format of providing the answer:

Example 1:
Table related to Teams and venues in context of 2004 Belarusian Premier League.\n\nQuestion:  What is the Location for Belshina?\nStep 1 : The table shows the teams and their corresponding locations. To find the location for Belshina, we need to locate Belshina in the 'Team' column. Belshina is located in row 9, and its location is given in the 'Location' column as 'Bobruisk'. Therefore, the location for Belshina is Bobruisk.\n\nStep 2 : bobruisk

Example 2:
Table related to Made the cut in context of 2009 U.S. Open (golf).\n\nQuestion:  what is the to par for retief goosen?\nStep 1 : The question asks for the "to par" for Retief Goosen. Looking at the table, we can see that Retief Goosen is listed on row 1, and the "To par" column for row 1 is +3. Therefore, the answer is +3.\n\nStep 2 : +3

Example 3:
Table related to Round 16 in context of 1969 VFL season.\n\nQuestion:  If the Venue was kardinia park what was the highest Crowd attended?\nStep 1 : The question asks for the highest crowd that attended a game at Kardinia Park. Looking at the table, we see that Kardinia Park is listed as the venue for round 3.  The crowd for that game was 16,211. The question asked for the highest crowd.  Looking at the rest of the table, the only higher crowd is 21,025.  Therefore, 21,025 is the answer.\n\nStep 2 : 21,025

Example 4:
Table related to Prime ministers in context of Interwar unemployment and poverty in the United Kingdom.\n\nQuestion:  What is the birth place of the prime minister who served George V and entered office on 23 October 1922?\nStep 1 : The answer can be arrived by locating the row where the prime minister entered office on 23 October 1922. This is the row corresponding to Andrew Bonar Law. Looking at the 'Birth Place' column for this row, the answer is given as "Rexton, Kent County, New Brunswick, Canada".\n\nStep 2 : rexton, kent county, new brunswick, canada

Example 5:
Table related to Round 12 in context of 1976 VFL season.\n\nQuestion:  What date did the team with a Blue-coloured kangaroo with "North Melbourne" written on it score 22.14 (146) as the home team?\nStep 1 : The question asks what date North Melbourne scored 22.14 (146) as the home team. We can find the row that corresponds to North Melbourne by finding the row with their logo. Row 5 shows their logo, which shows a Blue-coloured kangaroo with "North Melbourne" written on it, and their home score, which is 22.14 (146). In the last column, the date is 19 June 1976. Therefore, the answer to the question is 19 June 1976.\n\nStep 2 : 19 june 1976

Example 6:
Table related to Game Log in context of 1982 Atlanta Braves season.\n\nQuestion:  Where was the game on Thursday, April 29, and they played the team with Purple background logo with a C written in Red colour?\nStep 1 : The table shows the 1982 Atlanta Braves game log. We need to find the game on Thursday, April 29. We see this date on row 19. Looking at the "Opponent" column, we see the Chicago Cubs. It is represented by a Purple background logo with a Cwritten in Red colour. Therefore, the game was played at the location listed in the "Site" column for row 19: Atlanta-Fulton County Stadium.\n\nStep 2 : atlanta-fulton county stadium

Example 7:
Table related to Medal table in context of 2007 Military World Games.\n\nQuestion:  Which nation has a Silver of 1, a Gold of 0, and a Total of 1?\nStep 1 : The question asks for the nation with 1 silver medal, 0 gold medals, and a total of 1 medal.Looking at the table, we can see that Bulgaria, Cameroon and Hungary, which are represented by a White-Green_Red striped flag, Green-Red-Tellow flag with a star, and a Red-White Green striped flag respectively, have 1 silver medal, 0 gold medals and a total of 1 medal.\n\nStep 2 : bulgaria, cameroon, hungary

Example 8:
Table related to Round 4 in context of 1956 VFL season.\n\nQuestion:  Which home team scored 12.15 (87)?\nStep 1 : The table shows the scores of different teams in round 4 of the 1956 VFL season. The row with the home team score of 12.15 (87) is the second row which corresponds to Collingwood. Therefore, the home team that scored 12.15 (87) is Collingwood.\n\nStep 2 : collingwood

Now, based upon the examples given above, you must understand the image given and follow the steps 1-2 to answer the question corresponding to the table represented in the image. It is IMPORTANT that you perform all the both the steps to the best possible extent to get the correct answer. You must follow the format of answers as demonstrated by the examples above. IMPORTANT: You must give the answer in the format 'Step 2:\n<answer>'.

{main_part}"""

    TEXT_PROMPT_HYBRIDQA = """You are given a table in the form of an image. In the table, some entities (mentioned in text form originally) have been replaced by images that represent them. Based upon the context of the table while using real-world knowledge, your task is to identify the entities corresponding to the images in the table and answer the question. You must perform this task in the following steps:

Step 1: Reason about what should be the answer to the question by identifying the relevant entities represented by images using the context of the table and the question. The reasoning should be detailed and should be based upon the context of the table and the question, using real-world knowledge for answering the question. You are also given some extra information of a few entities in the form of passages. You have to use these passages along with the images to answer the question. IMPORTANT: You must explore any kind of reasoning -- numerical, logical, knowledge-based needed for disambiguating the entities and answering the question.
Step 2: Based upon the reasoning provided, provide the answer to the question.

You are also provided with some question-answer examples for better understanding the format of providing the answer:

Example 1:
{example_1}
Example 2:
{example_2}
Example 3:
{example_3}
Example 4:
{example_4}
Now, based upon the examples given above, you must understand the image given and follow the steps 1-2 to answer the question corresponding to the table represented in the image. It is IMPORTANT that you perform all the both the steps to the best possible extent to get the correct answer. You must follow the format of answers as demonstrated by the examples above. IMPORTANT: You must give the answer in the format 'Step 2:\n<answer>'.

{main_part}"""

    # TEXT_PROMPT = """You are given a table in which some entities in various table cells have been replaced by tokens of the type '{{ENTITY-<entity_id>}}. Each row of the table is in separate lines, and the columns are separated by '|'. Based upon the context of the table and using real-world knowledge, your task is to answer a question based upon the table by guessing the replaced entities of the table. You are given some examples to better illustrate the task:

# # Example 1:
# # {example_1}

# # Example 2:
# # {example_2}

# # Example 3:
# # {example_3}

# # Example 4:
# # {example_4}

# # Example 5:
# # {example_5}

# # Example 6:
# # {example_6}

# # Example 7:
# # {example_7}

# # Example 8:
# # {example_8}

# # Now, using the above examples as context, answer the question given:
# # {main_part}"""
    passage_arr = tables_dict[table_id]["cells_to_link"]
    flag = False
    ans = None
    relevant_nodes = []
    for node in ques['answer-node']:
        if node[3] == 'passage' and node[2] is not None:
            relevant_nodes.append(node)

    if DATASET == "WikiSQL":
        for x in ques["answer"]:
            if not isinstance(x,str):
                flag = True
                break
        ansArr=ques["answer"]
        if flag is True:
            ansArr = [str(num) for num in ques["answer"]]
        ans = ", ".join(ansArr)
    
    
    # print(relevant_nodes)
    prompt = convert_table_to_prompt(relevant_nodes,table_id, uniform_table_array, ques["question"], ans)
    # prompt = convert_table_to_prompt(table_id, uniform_table_array, ques["question"])
    prompt = (prompt.split("Answer: ")[0] + " "+prompt.split("Answer: ")[1])
    # final_prompt = TEXT_PROMPT.format(example_1=examples[0], example_2=examples[1], example_3 = examples[2],example_4 = examples[3],example_5 = examples[4],example_6 = examples[5],example_7 = examples[6],example_8 = examples[7], main_part=prompt)
    TEXT_PROMPT = TEXT_PROMPT_WTQ_Visual
    if DATASET == "fetaqa_MM_cleaned":
        TEXT_PROMPT = TEXT_PROMPT_FETA_visual
    elif DATASET == "WikiSQL":
        TEXT_PROMPT=TEXT_PROMPT_WIKISQL_visual
    elif DATASET == "Hybrid_QA_MM":
        TEXT_PROMPT = TEXT_PROMPT_HYBRIDQA.format(example_1=ex_foramts[0],example_2=ex_foramts[1],example_3=ex_foramts[2],example_4=ex_foramts[3],main_part=prompt)   
    final_prompt = TEXT_PROMPT

    file_name = table_id.replace("/", "_")
    
    table_image_path = os.path.join(IMAGE_PATH_DIR_TEST, f"{file_name}.jpg")
    if not os.path.exists(table_image_path):
            print("mkc",table_image_path)
            dne.append(((table_image_path.split("/")[-1]).split(".jpg")[0]).replace("_","/"))
            # break
    
    # qid_to_prompts[qid] = {"prompt": [final_prompt, table_image_path,example_table_images[0],example_table_images[1],example_table_images[2],example_table_images[3],example_table_images[4],example_table_images[5],example_table_images[6],example_table_images[7]], "gold_ans": ques["answer"]}
    ######Uncomment the above if you do not want to hardcode the few shots.
    # if ((ctr%2)!=0 and ("|" in ques["answer"])):
    #     ctr+=1
    #     qid_to_prompts[qid] = {"prompt": [final_prompt, table_image_path], "gold_ans": ques["answer"]}
    # elif (ctr%2==0):
    #     ctr+=1
    #     qid_to_prompts[qid] = {"prompt": [final_prompt, table_image_path], "gold_ans": ques["answer"]}
    if DATASET == "WikiSQL":
        flag = False
        for x in ques["answer"]:
            if not isinstance(x,str):
                flag = True
                break
        ansArr=ques["answer"]
        if flag is True:
            ansArr = [str(num) for num in ques["answer"]]

    
    
        ans = ", ".join(ansArr)
    ans = ques["answer-text"]
    qid_to_prompts[qid] = {"prompt": [final_prompt, table_image_path], "gold_ans": ans}

    # c+=1
    # if c==10:
    #     break
    # break
    # if ctr==20:
    #     break
    # break
# %%
DATASET = BASE_DIR.split("/")[-1]
RESULTS_BASE_DIR = "/home/suyash/final_repo/modelling/table-image-approach/Results/final/"

print(dne)
try:
    os.mkdir(os.path.join(RESULTS_BASE_DIR, f"{DATASET}_{QUESTION_TYPE}_oracle_gpt_final"))
except:
    pass
PROMPT_FILE = os.path.join(RESULTS_BASE_DIR, f"{DATASET}_{QUESTION_TYPE}_oracle_gpt_final", "qid_to_prompts.json")
RESULT_FILE = os.path.join(RESULTS_BASE_DIR, f"{DATASET}_{QUESTION_TYPE}_oracle_gpt_final", "results.json")

with open(PROMPT_FILE, "w") as f:
    json.dump(qid_to_prompts, f)

print(f"python3 /home/suyash/final_repo/Common_codes/run_gemini_query.py --input_file {PROMPT_FILE} --output_file {RESULT_FILE} --mm")

