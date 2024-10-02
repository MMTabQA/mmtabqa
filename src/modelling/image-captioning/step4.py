
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

RESULTS_BASE_DIR = "/home/suyash/final_repo/modelling/baseline_2/Results"

PROMPT_FILE = os.path.join(RESULTS_BASE_DIR, f"{DATASET}_{QUESTION_TYPE}", "qa_prompt.json")
RESULT_FILE = os.path.join(RESULTS_BASE_DIR, f"{DATASET}_{QUESTION_TYPE}", "qa_output.jsonl")

qid_to_prediction = {}
qid_to_gold_ans = {}

with open(RESULT_FILE) as f:
    for line in f.readlines():
        line = json.loads(line)
        qid = line['key']
        prediction = (line['response'])
        prediction = prediction.split("Step 2")[-1].strip()
        if prediction.startswith(":\n"):
            prediction = prediction[2:]
        gold_ans = line['gold_answer']
        qid_to_gold_ans[qid] = gold_ans
        qid_to_prediction[qid] = prediction

# %%
import sys
import ast

sys.path.append("/home/suyash/final_repo/evaluation_metrics/")
from exact_match import EvaluationMetrics

if DATASET == "fetaqa_MM_cleaned":
    ground_truths = []
    predictions = []
    
    for qid, prediction in qid_to_prediction.items():
        gold_ans = qid_to_gold_ans[qid]

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

    for qid, prediction in qid_to_prediction.items():
        gold_ans = qid_to_gold_ans[qid]
        if type(gold_ans) == list:
            gold_ans = ", ".join([str(_) for _ in gold_ans])

        if prediction.startswith("['"):
            prediction = prediction[2:-2]
        if len(prediction) > 0 and prediction[0]=='[':
            prediction = prediction[1:-1]

        if evaluator.compute_exact_match(gold_ans, prediction):
            exact_match += 1
            substring_match += 1
            llm_match += 1
        else:
            if evaluator.gold_ans_in_prediction(prediction, gold_ans):
                substring_match += 1
                print("----------------------")
                print("Gold:", gold_ans)
                print("Prediction:", prediction)
                print("----------------------")
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
