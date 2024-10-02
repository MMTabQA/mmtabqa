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
if DATASET == "fetaqa_MM_cleaned":
    CAP_700_SAMPLES = f"/home/suyash/final_repo/redone_samples_700cap/FetaQA/"
else:
    CAP_700_SAMPLES = f"/home/suyash/final_repo/redone_samples_700cap/{DATASET}/"

print(f"Running script with dataset: {DATASET} and question type: {QUESTION_TYPE}")


BASE_DIR = f"/home/suyash/final_repo/[temp]train_test_questions/{DATASET}"
OG_BASE_DIR = f"/home/suyash/final_repo/{DATASET}"

# %%
tables_dict = {}
train_questions = {}
test_questions = {}

with open(os.path.join(CAP_700_SAMPLES, f"test_{QUESTION_TYPE}_questions.jsonl")) as f:
    lines = f.readlines()
    for line in lines:
        line_json = json.loads(line)
        test_questions[line_json["question_id"]] = line_json

with open(os.path.join(OG_BASE_DIR, "experiment_ready_dataset", "tables.jsonl")) as f:
    lines = f.readlines()
    for line in lines:
        line_json = json.loads(line)
        tables_dict[line_json["table_id"]] = line_json

train_questions = {}
for qtype in ["explicit", "answer", "implicit", "visual"]:
# for qtype in ["visual"]:
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

with open(os.path.join(OG_BASE_DIR, "experiment_ready_dataset", "image_id_to_original_string.json")) as f:
    image_id_to_original_string = json.load(f)

# %%
if args.dataset == "WikiTableQuestions":
    # train_exclusive_questions = ["nt-5463", "nu-1628", "nu-2495", "nt-1454", "nt-8539", "nt-10664", "nt-5933", "nu-2806"]
    train_exclusive_questions = ["nt-5463", "nt-1454", "nt-8539", "nt-10664", "nt-5933", "nu-2806", "nt-1555:Chicago Cardinals", "nt-3613:Argentina"]
    train_exclusive_questions = [train_questions[qid] for qid in train_exclusive_questions]
elif args.dataset == "WikiSQL":
    train_exclusive_questions = ["WSQL-74751", "WSQL-29267", "WSQL-56037", "WSQL-26020", "WSQL-23863", "WSQL-61633", "WSQL-67435", "WSQL-78329"]
    train_exclusive_questions = [train_questions[qid] for qid in train_exclusive_questions]
elif args.dataset == "fetaqa_MM_cleaned":
    train_exclusive_questions = [20919, 15854, 18189, 8341, "16849:norway", "10446:france", 16422, 2282]
    train_exclusive_questions = [train_questions[qid] for qid in train_exclusive_questions]
print(train_exclusive_questions)

# test_questions = train_exclusive_questions[110:110+30]
# test_questions = {test_question['question_id']: test_question for test_question in test_questions}
# print(test_questions)

# %%
def create_metdata_prompt_sentence(page_title, table_headers):
    if pd.isna(table_headers):
        prompt = f"Table related to the {page_title}."
    else:
        prompt = f"Table related to {table_headers} in context of {page_title}."
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
    
    table_string = normalize_image_tags(table_string)
    
    if answer:
        if type(answer) == list:
            answer = ", ".join([str(_) for _ in answer])
        TABLE_TEXT = """Table context: {table_metadata}

Table:
{table}

Question: {question}
Step 1: {reason}
Step 2: {answer}"""
        return TABLE_TEXT.format(table_metadata=table_metadata, table=table_string, question=question, answer=answer, reason=reason)

    TABLE_TEXT = """Table context: {table_metadata}

Table:
{table}

Question: {question}
Step 1: """
    return TABLE_TEXT.format(table_metadata=table_metadata, table=table_string, question=question)



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
qid_to_reason = {
    "nt-5463": "The table shows the medal count for various countries in the 1973 Asian Athletics Championships. We need to find the country/countries that have the same total medals as Thailand. We identify {ENTITY-7} as Thailand, which has a total of 4 medals (2 gold, 2 silver, and 0 bronze). Looking at the table, we see that {ENTITY-9} which is Iran and {ENTITY-2} which is Malayasia also have a total of 4 medals. Therefore, the answer is Iran and Malaysia.",
    "nu-1628": "The question asks for the player who earned less than $200 but more than $100 besides Ben Hogan. We know Ben Hogan is {ENTITY-9} from the table. Looking at the table, we see that the only player besides Ben Hogan who earned less than $200 but more than $100 is the player in the 8th place. The player in the 8th place is {ENTITY-8}, who is Henry Picard.",
    "nu-2495": "The table shows the schedule of the 1972 Minnesota Vikings season. We need to find out how many times the Vikings played at Three Rivers Stadium. To do this, we need to identify which team plays at Three Rivers Stadium. Three Rivers Stadium was the home stadium of the Pittsburgh Steelers. We need to find the rows in the table where the opponent is the Pittsburgh Steelers. The opponent on November 26 is Pittsburg Steelers, which is {ENTITY-2} and {ENTITY-3} represents the Three Rivers Stadium. Therefore, the answer is 1.",
    "nt-1454": "The table shows the number of goals scored by different players in the 2010–11 PFC Levski Sofia season, categorized by competition. We need to find the row corresponding to Ismail Isa and then look the goals in the \"Total\" column. The row corresponding to Ismail Isa is the 3rd row and the total goals scored by him is 8. Therefore, the answer is 8.",
    "nt-8539": "The table shows the winners and runners-up of the men's and women's tournaments in the Old Four tournament.  We are looking for the winner of the women's tournament in 2003.  The table shows that the women's winner in 2003 was {ENTITY-2}.  We need to figure out what school {ENTITY-2} represents.  Looking at the table, we see that {ENTITY-2} is the winner of the men's tournament in 2003, 2004, 2006 and 2007 and runner-up in 2012 and 2013.  We also see that {ENTITY-2} is the runner-up in the women's tournament in 2004, 2006 and 2012.  This suggests that {ENTITY-2} is a school that is consistently competitive in the Old Four tournament.  Based on this information, we can guess that {ENTITY-2} represents the University of Western Ontario. We answer 'Western' based upon other entires of 'Toronto' and 'London' in the table.",
    'nt-10664': """ The table shows the release history of the album "Fables of the Reconstruction". We need to find the number of releases in compact disc format and the number of releases in cassette tape format. Then we need to subtract the number of cassette tape releases from the number of compact disc releases to find the difference. Looking at the table, we can see only {ENTITY-1} and {ENTITY-2} as the entity tokens in the Format column, which correspond to LP and cassette tape respectively. Looking at the format column, we can see 6 releases in Compact Disc format and 1 release in Cassette Tape format. Therefore, the answer is 6-1 = 5.""",
    'nt-5933': """The table shows the schedule of the 1974 Kansas City Chiefs season. We need to find the date when they played the Broncos and lost. Looking at the table, we see that the Chiefs played the Broncos in week 4, on October 6, 1974. The result column shows that they lost the game. Thus, the date is October 6, 1974.""",
    'nu-2806': """The question asks for the manufacturer(s) that appear the least on the chart. To answer this, we need to count how many times each manufacturer appears in the "Manufacturer" column. {ENTITY-1} only occurs twice and it corresponds to "New Flyer", while {ENTITY-4} also occurs twice and it corresponds to "Gillig". Therefore, the answer is New Flyer and Gillig.""",
    "nt-1555:Chicago Cardinals": """The question asks for the number of games played against the team whose logo features a red cardinal. We need to identify the team with a red cardinal logo from the table.  Real-world knowledge tells us that the Chicago Cardinals have a red cardinal logo.  We need to find the team name in the table that corresponds to the Chicago Cardinals.  Looking at the table, we see that the team name "Chicago Cardinals" is not explicitly mentioned. However, we can infer that the team with the red cardinal logo is the team that plays in Chicago. From real-world knowledge of game schedules and results in the 1952 season, we can determine that {ENTITY-6} is the Chicago Cardinals and is mentioned twice in the table.""",
    "nt-3613:Argentina": """he question asks for the total medals won by the country whose flag is composed of white and yellow stripes. Looking at the table, we need to identify the country with a flag of white and yellow stripes.  Real-world knowledge tells us that the flag of Argentina is composed of white and yellow stripes.  We need to find the row corresponding to Argentina in the table and then find the value in the "Total" column for that row. From real-world knowledge of individual gold/silver/bronze medal wins in 2011 Pan American Games, we can identify that Argentina is represented by {ENTITY-6} in the table.  The table shows that Argentina won a total of 7 medals in the 2011 Pan American Games.""",
    'WSQL-74751': """The question asks for the home team that scored 12.6 (78). Looking at the table, we need to find the row where the "Home team score" column has the value "12.6 (78)".  We can then identify the corresponding "Home team" from that row. This corresponds to the sixth row, where the Home Team is {ENTITY-9}, which corresponds to hawthorn. Therefore, the answer is hawthorn.""",
    "WSQL-29267": """The table shows Carlo Simionato's achievements in various competitions. We need to find the Time for the European Cup competition held in Moscow. Looking at the table, we see that there are three rows with "European Cup" as the Competition. The first row has London as the Venue, and the second row has {ENTITY-4} as the Venue. We need to figure out what {ENTITY-4} represents. From the year column, we can see that this European Cup happened in 1985. From the real-world knowledge, we know that this European Cup was held in Moscow. Therefore, the time corresponding to this is 38.88.""",
    "WSQL-56037": """The table shows the Members of Parliament for the North Staffordshire constituency in the UK Parliament. The columns are Election, 1st Member, 1st Party, 2nd Member, and 2nd Party. The question asks for the 1st Party in the election of 1865. Looking at the row for 1865, we see that the 1st Party is listed as {ENTITY-7}. We need to guess what party this entity represents.  Since the table shows various political parties like Whig, Conservative, and {ENTITY-4}, it's likely that {ENTITY-7} also represents a political party. From real-world knowledge and based upon the 1st Member "Sir Edward Manningham-Buller, Bt" we can conclude that {ENTITY-7} corresponds to liberal.""",
    "WSQL-26020": """The table shows Margarita Ponomaryova's achievements in various competitions. We need to find the competition where she achieved 1st position with a note of 57.03. Looking at the table, we can see that the only row with a 1st position and a note of 57.03 is the one for the year 1989. The competition in that row is '{ENTITY-8}' and the venue is '{ENTITY-9} , {ENTITY-10}'. From real-world knowledge, we know that Margarita Ponomaryova won the 1989 World Student Games (Universiade) in the 400m hurdles event. Therefore, the answer is World Student Games (Universiade).""",
    "WSQL-23863": """The table shows the results of the promotion round for the 2nd Bundesliga. The first column shows the season, the second column shows the 16th placed team in the 2nd Bundesliga, the third column shows the 3rd placed team in the 3rd Liga, and the last two columns show the results of the two games played between the two teams. We are asked to find the 3rd Liga team from the game 1 of 0-1 between the years 2008-09. Looking at the table, we can see that the game 1 of 0-1 occurred in the 2008-09 season. The 3rd Liga team in that season is {ENTITY-2}. From real-world knowledge, we know that the 3rd Liga team in the 2008-09 season with score 0-1 in both game 1 and game 2 is SC Paderborn 07. Therefore, the answer is SC Paderborn 07.""",
    "WSQL-61633": """The table shows Cristina Fink's achievements in various competitions, including the Olympic Games. We need to find the venue where she achieved a DNQ (Did Not Qualify) position in the Olympic Games. Looking at the table, we see that she achieved DNQ in the 1992 Olympic Games. The venue for the 1992 Olympic Games is listed as {ENTITY-6}.  Based on real-world knowledge, we know that the 1992 Summer Olympics were held in Barcelona, Spain. Therefore, we can guess that {ENTITY-6} represents "Barcelona, Spain".""",
    "WSQL-67435": """The table shows information about the 1935 VFL season, specifically Round 13. We need to find the venue where South Melbourne played as the away team. Using real-world knowledge, we can guess that South Melbourne played Footscray in this round and corresponds to {ENTITY-2}. The corresponding venue from the table is Western Oval.""",
    "WSQL-78329": """The table shows the 2000 NFL Draft for the Baltimore Ravens.  We are looking for the round where Southern Mississippi is listed as the school/club team.  The table shows that Jamal Lewis was drafted in the first round from the University of Tennessee.  The next player listed is Travis Taylor from Florida.  The third round shows that the player was drafted from the Louisville.  The fifth round shows Miami (FL) as the school/club team.  The sixth round shows that the Linebacker player was drafted from the South Mississippi, which is listed as {ENTITY-5} in the table.  Therefore, Southern Mississippi must be the school/club team for the player drafted in the sixth round. Therefore, the answer is 6.""",
    20919: """The table shows the Prime Ministers of Qatar from 1970 to present.  The fourth row shows that {ENTITY-2} was appointed as Prime Minister on 3 April 2007.  The table does not provide a reason for his appointment, but we can infer that he was appointed because the previous Prime Minister, Abdullah bin Khalifa Al Thani, resigned. From real-world knowledge, we know that {ENTITY-2} who is the Prime Minister of Qatar after 2007 is Hamad bin Jassim bin Jaber Al Thani.""",
    15854: """The table shows the results of the United Bowl (IFL) games. We need to find the row corresponding to the game between Sioux Falls Storm and Tri-Cities Fever on July 14, 2012.  The table shows that the Sioux Falls Storm played against the Tri-Cities Fever on July 14, 2012, and the Sioux Falls Storm won with a score of 59 to 32.""",
    18189: """The table shows Joshua Grommen's club history, including the season, league, and number of appearances and goals. In 2018, he played for {ENTITY-3} in the {ENTITY-2} league. We can infer that {ENTITY-3} is a club and {ENTITY-2} is a league based on the table's structure and the context of the question.  We can also infer that {ENTITY-2} is likely a professional league, as it is listed alongside other leagues like NPL Queensland. Based on real-world knowledge, we can guess that {ENTITY-3} is Davao Aguilas FC and {ENTITY-2} is the Philippines Football League (PFL).""",
    8341: """The table lists Gwen Verdon's filmography, including the year, title, role, and notes. To answer the question, we need to find the row corresponding to the movie "Walking Across Egypt" and the row corresponding to the year 2000.  

* **"Walking Across Egypt":** Based upon real-world knowledge, we know that this movie came out in 1999. We need to find the row where the "Title" column is "Walking Across Egypt" and the "Year" column is 1999. This row will tell us the role Gwen Verdon played in this movie. From the table, this role is Alora.
* **2000:** We need to find the row where the "Year" column is 2000. This row will tell us the title of the movie she appeared in that year. The movie corresponds to {ENTITY-14} where she played the role of MRs. Drago. Based upon real-world knowledge, we know that this is the movie Bruno, released in 2000.""",
    "16849:norway": """The question asks for the number of goals scored by Simon against a team whose flag has blue and white stripes at the 2011 FIFA Women's World Cup. Based upon real-world knowledge we know that this country is Norway. Looking at the table, we can see that Simon scored two goals against Norway, which is the team from the country of {ENTITY-28}. The table also shows that these goals were scored in the 2011 FIFA Women's World Cup, which is represented by {ENTITY-29}.""",
    "10446:france": """The question asks which country scored better between Canada and the country whose flag has blue stripes. Looking at the table, Canada is listed as the third-place finisher with a total score of 97.357. The country whose flag has blue stripes is likely France, which corresponds to 4th rank due to French swimmers like Cinthia Bouhier, Charlotte Fabre, Myriam Glez. Canada scored better than France, finishing ahead of France by almost a full point (96.467).""",
    16422: """The table shows Peter McKennan's career statistics, including the number of appearances and goals scored for various clubs. The question asks for the total number of goals scored in 121 appearances for Patrick Thistle. We can find this information by looking at the rows for Patrick Thistle, which is represented by the token {ENTITY-4}. The table shows the number of appearances and goals scored for each season, and the total for all seasons. We need to find the total goals scored in 121 appearances. Finding the sum, we see that the total number of goals scored in 121 appearances for Patrick Thistle is 70.""",
    2282: """The table shows the career statistics of Yohan Betsch, a French professional footballer. The table lists the clubs he played for, the seasons, and the leagues. We need to find the clubs he played for during the 2011-2013 seasons. Looking at the table, we see that he played for {ENTITY-5} during the 2011-12 season and {ENTITY-1} during the 2012-13 season. From real-world knowledge we know that {ENTITY-5} is FC Metz and {ENTITY-1} is Ligue 2 side Laval. Therefore, Betsch was at Metz in the 2011–12 season, then he joined Ligue 2 side Laval in the 2012-13 season.""",
    
}

examples = []
for ques in train_exclusive_questions:
    table_id = ques["table_context"]
    table_array = tables_dict[table_id]["table_array"]
    uniform_table_array = make_imageids_uniform(table_array)
    # prompt = convert_table_to_prompt(table_id, uniform_table_array, ques["question"], ques['answer'], qid_to_reason[ques["question_id"]])
    prompt = convert_table_to_prompt(table_id, uniform_table_array, ques["question"], ques['answer'], qid_to_reason[ques["question_id"]])
    examples.append(prompt)
random.shuffle(examples)

# %%

qid_to_prompts = {}
for qid, ques in test_questions.items():
    table_id = ques["table_context"]
    table_array = tables_dict[table_id]["table_array"]
    uniform_table_array = make_imageids_uniform(table_array)
    
    TEXT_PROMPT = """You are given a table in which some entities in various table cells have been replaced by tokens of the type '{{ENTITY-<entity_id>}}. Each row of the table is in separate lines, and the columns are separated by '|'. Based upon the context of the table and using real-world knowledge, your task is to answer a question based upon the table by guessing the replaced entities of the table. You must perform this task in the following steps:

Step 1: Reason about what should be the answer to the question based upon the context of the table. The reasoning should be detailed and should be based upon the context of the table and the question, using real-world knowledge for answering the question and guessing various entities involved in finding the answer. IMPORTANT: You must explore any kind of reasoning -- numerical, logical, knowledge-based needed for answering the question.
Step 2: Based upon the reasoning provided, provide the answer to the question

You are given some question-answer samples to better format for providing the answer. IMPORTANT: You must give the answer in the format "Step 2: <answer>".:

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

Now, using the above examples as context, answer the question given:
{main_part}"""
    prompt = convert_table_to_prompt(table_id, uniform_table_array, ques["question"])
    
    final_prompt = TEXT_PROMPT.format(example_1=examples[0], example_2=examples[1], example_3=examples[2], example_4=examples[3], example_5=examples[4], example_6=examples[5], example_7=examples[6], example_8=examples[7], main_part=prompt)
    
    qid_to_prompts[qid] = {"prompt": final_prompt, "gold_ans": ques["answer"]}

# %%
DATASET = BASE_DIR.split("/")[-1]
RESULTS_BASE_DIR = "/home/suyash/final_repo/modelling/lower_bound_no_cot/Results"

try:
    os.mkdir(os.path.join(RESULTS_BASE_DIR, f"{DATASET}_{QUESTION_TYPE}"))
except:
    pass
PROMPT_FILE = os.path.join(RESULTS_BASE_DIR, f"{DATASET}_{QUESTION_TYPE}", "qid_to_prompts.json")
RESULT_FILE = os.path.join(RESULTS_BASE_DIR, f"{DATASET}_{QUESTION_TYPE}", "results.json")

with open(PROMPT_FILE, "w") as f:
    json.dump(qid_to_prompts, f, indent=4)

print(f"python3 /home/suyash/final_repo/Common_codes/run_gemini_query.py --input_file {PROMPT_FILE} --output_file {RESULT_FILE}")

