
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

RESULTS_BASE_DIR = "/home/suyash/final_repo/modelling/lower_bound_no_cot/Results"

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

sys.path.append("/home/suyash/final_repo/evaluation_metrics/")
from exact_match import EvaluationMetrics

if DATASET == "fetaqa_MM_cleaned":
    ground_truths = []
    predictions = []
    
    for qid, response in qid_to_response.items():
        if "error" in response['response'].lower():
            print("SAD :((")
            print(response['response'])
            response['response'] = ""
        gold_ans = response['gold_ans']

        prediction = response['response'].split("Step 2:")[-1].strip()

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

    for qid, response in qid_to_response.items():
        gold_ans = response['gold_ans']
        if type(gold_ans) == list:
            gold_ans = ", ".join([str(_) for _ in gold_ans])

        if response['response'] == "An error occured.":
            print("SAD :((")
            continue
        prediction = response['response'].split("Step 2:")[-1].strip()
        
        if prediction.startswith("['"):
            prediction = prediction[2:-2]
        if prediction[0]=='[':
            prediction = prediction[1:-1]

        if evaluator.compute_exact_match(gold_ans, prediction, args.dataset):
            exact_match += 1
            substring_match += 1
            llm_match += 1
        else:
            if evaluator.gold_ans_in_prediction(prediction, gold_ans):
                substring_match += 1

            # llm_evaluations.append({
            #     "qid": qid,
            #     "gold_ans": gold_ans,
            #     "prediction": prediction,
            #     "question": test_questions[qid]["question"]
            # })
            else:
                pass
                # print("----------------")
                # print(gold_ans)
                # print(prediction)
                # print("----------------")

        f1_scores.append(evaluator.compute_f1_score(gold_ans, prediction))

    mean_f1_score = sum(f1_scores) / len(qid_to_response)
    accuracy = exact_match / len(qid_to_response)
    substring_accuracy = substring_match / len(qid_to_response)

    print(f"{DATASET} {QUESTION_TYPE} Results:")
    print(f"Exact match: {accuracy*100}")
    print(f"Substring match: {substring_accuracy*100}")
    print(f"Mean F1 score: {mean_f1_score}")

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



