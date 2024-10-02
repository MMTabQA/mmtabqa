# %%
import os
import json
import random
import argparse

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


if DATASET == "fetaqa_MM_cleaned":
    BASE_DIR_700 = f"/home/suyash/final_repo/redone_samples_700cap/FetaQA"
else:
    BASE_DIR_700 = f"/home/suyash/final_repo/redone_samples_700cap/{DATASET}"
BASE_DIR = f"/home/suyash/final_repo/[temp]train_test_questions/{DATASET}"
OG_BASE_DIR = f"/home/suyash/final_repo/{DATASET}"

# %%
tables_dict = {}
train_questions = {}
test_questions = {}

with open(os.path.join(BASE_DIR_700, f"test_{QUESTION_TYPE}_questions.jsonl")) as f:
    lines = f.readlines()
    for line in lines:
        line_json = json.loads(line)
        test_questions[line_json["question_id"]] = line_json

if DATASET == "WikiSQL":
    with open(os.path.join(OG_BASE_DIR, "experiment_ready_dataset", "temporary_ogtable_tables.jsonl")) as f:
        lines = f.readlines()
        for line in lines:
            line_json = json.loads(line)
            # print((line_json)[1])
            # print("HAHAHAH")
            # break
            tables_dict[line_json["table_id"]] = line_json
else:
    with open(os.path.join(OG_BASE_DIR, "experiment_ready_dataset", "tables.jsonl")) as f:
        lines = f.readlines()
        for line in lines:
            line_json = json.loads(line)
            # print((line_json)[1])
            # print("HAHAHAH")
            # break
            tables_dict[line_json["table_id"]] = line_json

train_questions = {}
for qtype in ["explicit", "answer", "implicit"]:
    with open(os.path.join(BASE_DIR, f"train_{qtype}_questions.jsonl")) as f:
        lines = f.readlines()
        for line in lines:
            line_json = json.loads(line)
            train_questions[line_json["question_id"]] = line_json

with open(os.path.join(OG_BASE_DIR, "experiment_ready_dataset", "image_id_to_image_path.json")) as f:
    image_id_to_image_path = json.load(f)

with open(os.path.join(OG_BASE_DIR, "experiment_ready_dataset", "image_id_to_original_string.json")) as f:
    image_id_to_original_string = json.load(f)

# %%
print("Train and test questions:", len(train_questions), len(test_questions))

with open(os.path.join(BASE_DIR, "tables_split.json")) as f:
    split = json.load(f)

train_tables = split['train']
train_exclusive_questions = [train_questions[qid] for qid in train_questions if train_questions[qid]["table_context"] in train_tables]
random.shuffle(train_exclusive_questions)
if DATASET == "WikiTableQuestions":
    train_exclusive_questions_ids = ['nt-5062', 'nt-10074', 'nt-6789', 'nt-6388', 'nt-2995', 'nt-8791', 'nt-11346', 'nt-11905']
    train_exclusive_questions = [train_questions[qid] for qid in train_exclusive_questions_ids]
elif DATASET == "fetaqa_MM_cleaned":
    train_exclusive_questions_ids = [154, 18263, 1635, 20961, 9841, 18269, 21287, 17634]
    train_exclusive_questions = [train_questions[qid] for qid in train_exclusive_questions_ids]

with open(os.path.join(OG_BASE_DIR, "experiment_ready_dataset", "image_id_to_original_string.json")) as f:
    image_id_to_original_string = json.load(f)

# %%
len(train_questions)

# %%
def create_metdata_prompt_sentence(page_title, table_headers):
    if pd.isna(table_headers):
        prompt = f"Table related to the {page_title}."
    else:
        prompt = f"Table related to {table_headers} of {page_title}."
    return prompt

def generate_table_string(table_array):
    table_string = ""

    for row_idx, row in enumerate(table_array):
        for col_idx, cell in enumerate(row):
            cell=str(cell)
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

def convert_table_to_prompt(table_id, table_array, question, answer = None, reason=None):
    if DATASET == "WikiTableQuestions":
        page_title, headers = get_wtq_metadata(table_id)
    elif DATASET == "WikiSQL":
        page_title = tables_dict[table_id]["page_title"]
        headers = tables_dict[table_id]["section_title"]
    elif DATASET == "fetaqa_MM_cleaned":
        page_title = tables_dict[table_id]["table_page_title"]
        headers = tables_dict[table_id]["table_section_title"]

    table_metadata = create_metdata_prompt_sentence(page_title, headers)

    table_string = generate_table_string(table_array)
    
    
    TABLE_TEXT = """Table context: {table_metadata}

Table:
{table}

Question: {question}
Step 1: """
    if answer is not None:
        TABLE_TEXT = TABLE_TEXT + """{reason}

Step 2: {answer}"""

    return TABLE_TEXT.format(table_metadata=table_metadata, table=table_string, question=question, answer=answer, reason=reason)


# %%
import re
from collections import defaultdict

# %%
qid_to_reason_exemplar = {
    "nt-5062": "To determine the next operational computer after the modified ENIAC, we look for computers that became operational after mid-1948. The Manchester Small-Scale Experimental Machine (Baby) was operational in June 1948, and its development led to the Manchester Mark 1, which became operational in April 1949. Therefore, the next significant computer after the modified ENIAC was the Manchester Mark 1",
    # "nt-4988": "To determine the seating capacity of the Holon City Arena in Holon from the table, locate the row labeled \"Holon City Arena\" under the \"Stadium\" column. Then, look across to the \"Capacity\" column in the same row, which shows the number of seats as 2,850.",
    "nt-10074": "The table shows the score in the 'Score' column, where the first number denotes the goals Scotland scored. This number is 0 for the games held on 27 May 1951 and 13 December 1950. Therefore, the two games are 27 May 1951 and 13 December 1950.",
    "nt-6789": "The table shows the win-loss record for the Toronto Blue Jays throughout the 1982 season [evidenced by the header \"W\" and \"L\" at the top of the rightmost column]. Dates are listed along the leftmost side of the table [evident by dates written on the left side of the table]. Under the \"W\" column, there are marks indicating wins [represented by \"I\" symbols]. Looking at July, we can see that the Blue Jays have wins marked on July 17, 20 and 23 [referring to the \"I\" symbols under the \"W\" column next to \"17\", \"20\" and \"23\" on the left side of the table]. Since there are no wins marked before July 20, we can conclude that the Blue Jays previous win was on July 20.",
    "nt-6388": "We can conclude Sunderland played in 2 games. The table shows teams listed under \"Home team\" and \"Away team\" columns [column headers provide this information].  Looking across the rows, Sunderland is listed under one of these columns twice [scanning for \"Sunderland\" within these columns reveals it twice]. Therefore, Sunderland participated in two games.",
    "nt-2995": 'The table  has a "Home/Away" column showing where games were played. Examine the first row of the table, as "top" refers to the first game.  This row\'s entry in "Home/Away" is "Home".',
    # "nt-10979": 'The table shows the the number of medals won by different countries. Locate the "Bronze" column which indicates the number of Bronze medals won by countries. We can see this number is 0 for North Korea and Japan. Therefore the countries with 0 bronze medals are North Korea and Japan.',
    "nt-8791": 'To determine the counties whose capital names are not the same as their name, we look at the "County" and "Capital" columns. We find that the capital name is different from the county name for "Tana River" and "Taita-Taveta". Therefore, the counties whose capital names are not the same as their name are Tana River and Taita-Taveta.',
    "nt-223": 'The table shows Formula One results with a context of Playlife, possibly a constructor. Look for a "Chassis" column listing car models. Find "Benetton B198" in that column. In the same row, there is the column year, which gives us the answer as 1998.',
    "nt-11346": 'The table shows defunct railroads in Washington D.C. Look for "Pennsylvania Railroad" or "PRR". In the same row, another column (likely "Mark" or "Reporting Mark") should have an abbreviation. If this abbreviation is "PRR", then the Pennsylvania Railroad operated under PRR since "PRR" is its short name.',
    "nt-11905": 'The table shows the Frankfurt Galaxy\'s 1996 season schedule, likely indicating locations or stadiums. Look for a "Location" or "Stadium" column and then find entries mentioning "Waldstadion" in that column. Counting these entries will reveal the number of games played at Waldstadion since the schedule shows where each game took place.',
    154: 'The table shows the NHL awards for the 2013-14 season. Look for the "Award" column and find the "Stanley Cup" entry. Look at the entry in the corresponding "Recepient(s)" column and "Runner(s)-up/Finalists" column to find the teams competing for the Stanley Cup. The teams are the Los Angeles Kings and the New York Rangers.',
    # In which city was the 1991 World Championships held and what distance did Debbie Marti achieve to qualify?
    # "answer": "At the 1991 World Championships in Tokyo, Debbie Marti qualified with 1.86 m.",
    18263: 'Look for the 1991 World Championships in the "Competition" colun, whose venue is given in the "Venue" column as Tokyo. The "Notes" column indicates that Debbie Marti qualified with a distance of 1.86 m.',
    # What awards did Project Gutenberg win at the 38th Hong Kong Film Awards?
    # "Project Gutenberg won seven awards at the 38th Hong Kong Film Awards, in the categories Best Film, Best Director, Best Screenplay, Best Cinematography, Best Film Editing, Best Art Direction, and Best Costume Make Up Design."
    1635: 'Look at all the entries of 38th Hong Kong Film Awards in the "Ceremony" column which have Won in the "Result" column. Identify all such awards from the "Category" column, which are Best Film, Best Director, Best Screenplay, Best Cinematography, Best Film Editing, Best Art Direction, and Best Costume Make Up Design.',
    # What film won the Alfred P. Sloan Prize at the Sundance Film Festival in 2014?
    # Cahill's film I Origins again won the Alfred P. Sloan Prize at the 2014 Sundance Film Festival, his second time receiving the award.
    20961: 'The table shows the Awards and Nominations of the director Mike Cahill. To determine the films by that won the Alfred P. Sloan Prize at the Sundance Film Festival in 2014, look for the "Year" column and find the entry for 2014. Then, look for the "Award" column and find the "Alfred P. Sloan Prize" entry. The film that won the prize is "I Origins", which is Cahill\'s second time receiving the award.',
    # How did Everton F.C. do against Manchester United and Tottenham Hotspur during their 2012-13 season?
    # Everton F.C. won over Manchester United in the first game of the season with 1\u20130, defeated Tottenham Hotspur 2\u20131, and defeated Manchester City 2\u20130 in the Premier League.
    9841: 'Look for Manchester United and Tottenham Hotspur on the “Opponent” column.  Look at the corresponding “Score” for each team. In the first game of the season with Manchester For Manchester United, the score is  1-0 in favor of Everton. For Tottenham Hotspur, the score is 2-1 in favor of Everton. Therefore, Everton won against both Manchester United and Tottenham Hotspur.',
    # 9841: 'Look for Manchester United and Tottenham Hotspur on the “Opponent” column.  Look at the corresponding “Score” for each team. In the first game of the season with Manchester For Manchester United, the score is  1-0 in favor of Everton. For Tottenham Hotspur, the score is 2-1 in favor of Everton. In a subsequent game, the score is  Therefore, Everton won against both Manchester United and Tottenham Hotspur.',
    18269: 'Look for "2003" in the "Year" column. Look across that row to the "Competition" column. It should say "World Championships". In the "Event" column, it shows "1500 m". Finally, under the "Position" column, it shows "2nd", indicating that Süreyya Ayhan won a silver medal.',
    21287: 'Looking at the table under the "Year" column, you can see 1973 listed twice.  In the corresponding rows under "Award" it says "Record of the Year" each time.  Looking at the "Nominee / work" column for those two rows, it shows "The First Time Ever I Saw Your Face" in 1973 and "Killing Me Softly With His Song" in 1974.  This confirms that Flack won the award for these two songs in consecutive years.',
    17634: 'Look for the year "2017" in the "Year" column. Look across that row to the "Network" column. It should say "KBS2". In the "Title" column, it shows "School 2017". This indicates that Kim Jung-hyun played in that drama in 2017 on KBS2.'
}

#################Uncomment to build exemplars##########################
examples = ["hahah" for i in range(20)]
# for ques in train_exclusive_questions:
#     table_id = ques["table_context"]
#     table_array = tables_dict[table_id]["original_table_array"]
#     uniform_table_array = table_array
#     reason = qid_to_reason_exemplar[ques["question_id"]]
#     prompt = convert_table_to_prompt(table_id, uniform_table_array, ques["question"], ques['answer'], reason)
#     examples.append(prompt)
# random.shuffle(examples)


# %%

qid_to_prompts = {}
for qid, ques in test_questions.items():
    table_id = ques["table_context"]
    table_array = tables_dict[table_id]["original_table_array"]
    uniform_table_array = table_array
    
    TEXT_PROMPT = """You are given a table in text format where each row of the table is in separate lines, and the columns are separated by '|'. Based upon the context of the table and using real-world knowledge, your task is to answer a question based upon the table provided. You must perform this task in the following steps:

Step 1: Reason about what should be the answer to the question based upon the context of the table. The reasoning should be detailed and should be based upon the context of the table and the question, using real-world knowledge for answering the question. IMPORTANT: You must explore any kind of reasoning -- numerical, logical, knowledge-based needed for answering the question.
Step 2: Based upon the reasoning provided, provide the answer to the question.

You are given some examples to better illustrate the task:

Step 1: Reason about what should be the answer to the question by identifying the relevant entities represented by images using the context of the table and the question. The reasoning should be detailed and should be based upon the context of the table and the question, using real-world knowledge for answering the question. IMPORTANT: You must explore any kind of reasoning -- numerical, logical, knowledge-based needed for disambiguating the entities and answering the question.
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

Example 5:
{example_5}

Example 6:
{example_6}

Example 7:
{example_7}

Example 8:
{example_8}

Now, using the above examples as context, reason and answer the question given:
{main_part}"""

    TEXT_PROMPT_WIKISQL_UB = """You are given a table in the form of an image. In the table, some entities (mentioned in text form originally) have been replaced by images that represent them. Based upon the context of the table while using real-world knowledge, your task is to identify the entities corresponding to the images in the table and answer the question. You must perform this task in the following steps:

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
Table related to Round 12 in context of 1976 VFL season.\n\nQuestion:  What date did North Melbourne score 22.14 (146) as the home team?\nStep 1 : The question asks what date North Melbourne scored 22.14 (146) as the home team. We can find the row that corresponds to North Melbourne by finding the row with their name. Row 5 shows their name, and their home score, which is 22.14 (146). In the last column, the date is 19 June 1976. Therefore, the answer to the question is 19 June 1976.\n\nStep 2 : 19 june 1976

Example 6:
Table related to Game Log in context of 1982 Atlanta Braves season.\n\nQuestion:  Where was the game on Thursday, April 29, and they played the Chicago Cubs?\nStep 1 : The table shows the 1982 Atlanta Braves game log. We need to find the game on Thursday, April 29. We see this date on row 19. Looking at the "Opponent" column, we see the Chicago Cubs. Therefore, the game was played at the location listed in the "Site" column for row 19: Atlanta-Fulton County Stadium.\n\nStep 2 : atlanta-fulton county stadium

Example 7:
Table related to Medal table in context of 2007 Military World Games.\n\nQuestion:  Which nation has a Silver of 1, a Gold of 0, and a Total of 1?\nStep 1 : The question asks for the nation with 1 silver medal, 0 gold medals, and a total of 1 medal.Looking at the table, we can see that Bulgaria, Cameroon and Hungary, have 1 silver medal, 0 gold medals and a total of 1 medal.\n\nStep 2 : bulgaria, cameroon, hungary

Example 8:
Table related to Round 4 in context of 1956 VFL season.\n\nQuestion:  Which home team scored 12.15 (87)?\nStep 1 : The table shows the scores of different teams in round 4 of the 1956 VFL season. The row with the home team score of 12.15 (87) is the second row which corresponds to Collingwood. Therefore, the home team that scored 12.15 (87) is Collingwood.\n\nStep 2 : collingwood

Now, based upon the examples given above, you must understand the image given and follow the steps 1-2 to answer the question corresponding to the table represented in the image. It is IMPORTANT that you perform all the both the steps to the best possible extent to get the correct answer. You must follow the format of answers as demonstrated by the examples above. IMPORTANT: You must give the answer in the format 'Step 2:\n<answer>'.

{main_part}"""
    prompt = convert_table_to_prompt(table_id, uniform_table_array, ques["question"])
    
    
    final_prompt = TEXT_PROMPT.format(example_1=examples[0], example_2=examples[1], example_3=examples[2], example_4=examples[3], example_5=examples[4], example_6=examples[5], example_7=examples[6], example_8=examples[7], main_part=prompt)
    
    qid_to_prompts[qid] = {"prompt": final_prompt, "gold_ans": ques["answer"]}
    if DATASET=="WikiSQL":
        final_prompt = TEXT_PROMPT_WIKISQL_UB.format(main_part = prompt)
        flag = False
        for x in ques["answer"]:
            if not isinstance(x,str):
                flag = True
                break
        ansArr=ques["answer"]
        if flag is True:
            ansArr = [str(num) for num in ques["answer"]]

        ans = ", ".join(ansArr)
        qid_to_prompts[qid] = {"prompt": final_prompt, "gold_ans": ans}
    
        
        

    # qid_to_prompts[qid] = {"prompt": [final_prompt, table_image_path], "gold_ans": ans}

# %%
DATASET = BASE_DIR.split("/")[-1]
RESULTS_BASE_DIR = "/home/suyash/final_repo/modelling/upper_bound/Results"

try:
    os.mkdir(os.path.join(RESULTS_BASE_DIR, f"{DATASET}_{QUESTION_TYPE}"))
except:
    pass
PROMPT_FILE = os.path.join(RESULTS_BASE_DIR, f"{DATASET}_{QUESTION_TYPE}", "qid_to_prompts.json")
RESULT_FILE = os.path.join(RESULTS_BASE_DIR, f"{DATASET}_{QUESTION_TYPE}", "results.json")

with open(PROMPT_FILE, "w") as f:
    json.dump(qid_to_prompts, f)

print(f"python3 /home/suyash/final_repo/Common_codes/run_gemini_query.py --input_file {PROMPT_FILE} --output_file {RESULT_FILE}")

