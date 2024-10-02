
import os
import json
import random
import argparse
from tqdm import tqdm
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

RESULTS_BASE_DIR = "/home/suyash/final_repo/modelling/table-image-approach/Results/final/"

regex_match_cnt = 0
gpt_evaluator_prompts = {}
GPT_EVALUATOR_PROMPT_TEMPLATE = """You are given a question, along with its gold answer, and a candidate which represents a model generated answer. Your task here is to work as an evaluator and determine whether the candidate and the answer refer to the same entity or not.
You are also given a few examples to help you understand more

Example 1:
Table context: Table related to Summary of rounds of play|Third round in context of 2010 U.S. Open (golf).

Question: who was the top player?
Answer:
Dustin Johnson

Candidate: The player in the first row.
Is the candidate correct?

No, the candidate does not refer to "Dustin Johnson"

Example 2:
Table context: Table related to Career|International goals in context of Khalid Al Romaihi.

Question: which date and opponent is next after june 12,1989
Answer:
February 22, 1989|Portugal

Candidate: February 22, 1989, United Arab Emirates
Is the candidate correct?

No, the correct answer is February 22, 1989 against Portugal.

Example 3:
Table context: Table related to Production in context of Niobium.

Question: which country produced 50 in 2002 but only 13 in 2003?
Answer:
Congo D.R.

Candidate: Democratic Republic of the Congos
Is the candidate correct?

Yes, the candidate "Democratic Republic of the Congo" is correct.

Example 4:
Table context: Table related to Classification|Race in context of 2009 Chinese Grand Prix.

Question: who was the last driver to actually finish this race?
Answer:
Nelson Piquet, Jr.

Candidate: 7
Is the candidate correct?

No, the candidate 7 does not refer to "Nelson Piquet, Jr."

{question}{answer}\n\nCandidate: {candidate}\n\nIs the candidate correct?\n\n"""
c=0
try:
    os.mkdir(os.path.join(RESULTS_BASE_DIR, f"{DATASET}_{QUESTION_TYPE}"))
except:
    pass
PROMPT_FILE = os.path.join(RESULTS_BASE_DIR, f"{DATASET}_{QUESTION_TYPE}", "qid_to_prompts.json")
RESULT_FILE = os.path.join(RESULTS_BASE_DIR, f"{DATASET}_{QUESTION_TYPE}", "results.json")

qid_to_response = {}
with open(RESULT_FILE) as f:
    for line in f.readlines():
        line = json.loads(line)
        qid_to_response[line['key']] = line

# %%
import sys
import ast

def format_prompt(prompt):
    parts=prompt.split("#")
    prompt = parts[0] + parts[1]
    return prompt

sys.path.append("/home/suyash/final_repo/evaluation_metrics/")
from exact_match import EvaluationMetrics

if DATASET == "fetaqa_MM_cleaned":
    ground_truths = []
    predictions = []

    for qid, response in tqdm(qid_to_response.items()):
        gold_ans = response['gold_ans']

        prediction = response['response']
        prediction = prediction.split("Step 2:")[-1].strip()

        ground_truths.append(gold_ans)
        predictions.append(prediction)

    evaluator = EvaluationMetrics()
    bleu_score = evaluator.get_bleu_score(predictions, ground_truths)
    rouge_score = evaluator.get_rouge_score(predictions, ground_truths)
    bleurt_score = evaluator.get_bleurt_score(predictions, ground_truths)

else:

    exact_match = 0
    substring_match = 0
    llm_match = 0
    incorrect = 0
    total = 0
    f1_scores = []

    llm_evaluations = []
    evaluator = EvaluationMetrics()

    for qid, response in tqdm(qid_to_response.items()):
        gold_ans = response['gold_ans']
        # TODO: Fix the # after answer in the prompt
        prompt = response["prompt"][0].split("You must follow the format of answers as demonstrated by the examples above. IMPORTANT: You must give the answer in the format 'Step 2:\n<answer>'.\n\n")[-1]
        # prompt = format_prompt(prompt)
        if type(gold_ans) == list:
            gold_ans = ", ".join([str(_) for _ in gold_ans])

        prediction = response['response']

        if prediction == "Error occurred.": #Temporarily ignoring error responses from gemini
            continue
        prediction = prediction.split("Step 2")[-1].strip()
        if prediction.startswith(":\n"):
            prediction = prediction[2:]

        if prediction.startswith("['"):
            prediction = prediction[2:-2]
        if len(prediction)>0 and prediction[0]=='[':
            prediction = prediction[1:-1]
        if c<100:
         c+=1
         gpt_evaluator_prompts[qid] = GPT_EVALUATOR_PROMPT_TEMPLATE.format(question = prompt,answer = gold_ans,candidate=prediction)
        #  print(gpt_evaluator_prompts[qid])

        if evaluator.regex_match(gold_ans, prediction):
            regex_match_cnt+=1

        if evaluator.compute_exact_match(gold_ans, prediction,DATASET):
            exact_match += 1
            substring_match += 1
            llm_match += 1
        else:
            if evaluator.gold_ans_in_prediction(prediction, gold_ans):
                substring_match += 1
                # print("----------------------")
                # print(gold_ans)
                # print(prediction)

            # llm_evaluations.append({
            #     "qid": qid,
            #     "gold_ans": gold_ans,
            #     "prediction": prediction,
            #     "question": test_questions[qid]["question"]
            # })
            else:
                pass
                # print("----------------")
                # print(prompt)
                print("################")

                print(gold_ans)
                # print("################")

                print(prediction)
                # print("################")

                # print(response["prompt"][1])
                # print("----------------")

        f1_scores.append(evaluator.compute_f1_score(gold_ans, prediction))

    mean_f1_score = sum(f1_scores) / len(f1_scores)
    accuracy = exact_match / len(f1_scores)
    substring_accuracy = substring_match / len(f1_scores)
    regex_accuracy = regex_match_cnt / len(f1_scores)

    print(f"{DATASET} {QUESTION_TYPE} Results:")
    print(f"Exact match: {accuracy*100}")
    print(f"Substring match: {substring_accuracy*100}")
    print(f"Mean F1 score: {mean_f1_score}")
    print(f"Regex accuracy: {regex_accuracy}")

    filename = '/home/suyash/final_repo/modelling/table-image-approach/Results/gpt_evaluator_prompts_random.json'

# Open the file in write mode and use json.dump to write the dictionary to the file
# with open(filename, 'w') as file:
#         json.dump(gpt_evaluator_prompts, file, indent=4)

# print(f"Dictionary saved to {filename}")

    # # %%
    # len(llm_evaluations)

    # # %%
    # qid_to_llmprompt = {}
    # for eval in llm_evaluations:
    #     qid = eval["qid"]
    #     gold_ans = eval["gold_ans"]
    #     prediction = eval["prediction"]
    #     question = eval["question"]
    #     prompt = llm_entity_matching_prompt(question, gold_ans, prediction)

    #     qid_to_llmprompt[qid] = prompt

    # PROMPT_PATH = "/tmp/llm_evaluations_prompt.json"
    # with open(PROMPT_PATH, "w") as f:
    #     json.dump(qid_to_llmprompt, f)

    # RESULTS_PATH = "/tmp/llm_evaluations_results.json"

    # # %%
    # qid_to_llmprompt

    # # %%
    # print(f"python3 /home/suyash/final_repo/Common_codes/run_gemini_query.py --input_file {PROMPT_PATH} --output_file {RESULTS_PATH}")

    # # %%
    # with open(RESULTS_PATH) as f:
    #     for line in f.readlines():
    #         line = json.loads(line)
    #         if line['response'] == 'yes':
    #             print(line['response'])
    #             print(line['prompt'])

    # %%
