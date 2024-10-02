
# ML Model Prompt Creation and Evaluation

## Overview
This repository contains two scripts that assist in machine learning (ML) model-related tasks. The first script is for creating prompts based on a dataset, while the second is for evaluating the model's performance using different metrics like exact match, substring match, and F1 score.

## 1. Step 1: Prompt Creation (`step1.py`)

### Description
This script generates structured prompts based on the dataset and question type for model input.

### Usage
```bash
python step1.py --dataset <dataset_name> --question-type <question_type>
```

### Arguments:
- `--dataset`: Specifies the dataset to use (e.g., WikiTableQuestions, WikiSQL, FetaQA).
- `--question-type`: Specifies the type of question (e.g., answer, explicit, implicit).

### Functionality:
- Processes training and test questions from the dataset.
- Generates table-related prompts and outputs them in a predefined format.
- Saves the generated prompts to a specified directory for further use.

---

## 2. Step 2: Evaluation (`step2.py`)

### Description
This script evaluates the model-generated responses against gold-standard answers by calculating various metrics such as exact match, substring match, BLEU, ROUGE, and F1 scores.

### Usage
```bash
python step2.py --dataset <dataset_name> --question-type <question_type>
```

### Arguments:
- `--dataset`: The dataset used for creating the prompts.
- `--question-type`: The type of question used in prompt creation.

### Functionality:
- Reads model-generated responses and compares them with gold answers.
- Evaluates the model's performance based on multiple metrics.
- Outputs the results and saves them to a directory.

---

## Directory Structure
- **Input Data**: Expected datasets are stored in specific directories, such as `/home/suyash/final_repo/`.
- **Output**: The results from evaluations are saved in the `Results` directory as `Results/<dataset>_<question_type>/`.

## Dependencies
- Python 3.x
- Libraries: `argparse`, `json`, `pandas`, `random`, `tqdm`

## Example Commands
```bash
python step1.py --dataset WikiTableQuestions --question-type answer
python step2.py --dataset WikiTableQuestions --question-type answer
```

These scripts automate the process of generating prompts and evaluating the model's outputs, enabling easier training and assessment of machine learning models for question-answering tasks.
