import re
# import evaluate
import numpy as np
import evaluate
class EvaluationMetrics():
    def __init__(self):
        pass
        

    def clean_string(self, text):
        text = text.strip().lower().replace("\n", " ").replace("\\n", " ").replace(",", "").replace("-", " ").replace(":", " ").strip()
        text = re.sub(r'\s+', ' ', text)
        return text

    def compute_f1_score(self, prediction, ground_truth):
        # Convert lists to sets for efficient operations
        prediction = self.clean_string(prediction)
        ground_truth = self.clean_string(ground_truth)
        
        tokens_true = ground_truth.split()
        tokens_pred = prediction.split()
        
        set_true = set(tokens_true)
        set_pred = set(tokens_pred)
        
        tp = len(set_true.intersection(set_pred))
        fp = len(set_pred - set_true)
        fn = len(set_true - set_pred)
        
        # Calculate precision and recall
        precision = tp / (tp + fp) if tp + fp > 0 else 0
        recall = tp / (tp + fn) if tp + fn > 0 else 0
        
        # Calculate F1 score
        f1_score = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
        
        return f1_score

    def compute_exact_match(self, prediction, ground_truth,dataset):
        prediction = self.clean_string(prediction)

        ground_truth = self.clean_string(ground_truth)
        
        resp = (prediction == ground_truth)
        if resp:
            return resp
        
        if dataset =="WikiSQL":
            try:
                num1 = float(prediction)
                num2 = float(ground_truth)
                # Perform operations on the number
                resp2 = (num1==num2)
                diff = abs(num1-num2)
                if (diff<1e-6):
                    return True
            except ValueError:
                pass

        return (resp)

    def gold_ans_in_prediction(self, prediction, ground_truth):
        prediction = self.clean_string(prediction)
        ground_truth = self.clean_string(ground_truth)
        
        return ground_truth in prediction

    def llm_entity_matching_prompt(self, question, entity1, entity2):
        PROMPT_TEMPLATE = f"""Using common-sense knowledge and knowledge of the real world, your task is to tell if the two possible answers to a question refer to the same entity/value. You must answer 'yes' if the two refer to the same entity/value in context of the question, or 'no' if they don't refer to the same entity/value in context of the question. If the possible answers are a numerical value or a sum of money or a date or time duration, you must compare the value they represent and not the exact text. IMPORTANT: YOU MUST ONLY OUTPUT yes OR no. DO NOT ADD ANY ADDITIONAL TEXT.
    Question: {question}
    Answer 1: {entity1}
    Answer 2: {entity2}

    Response: """

        PROMPT_TEMPLATE = f"""Using common-sense knowledge and knowledge of the real world, your task is to tell if the two possible answers to a question refer to the same entity/value. You must answer 'yes' if the two refer to the same entity/value in context of the question, or 'no' if they don't refer to the same entity/value in context of the question. If the possible answers are a numerical value or a sum of money or a date or time duration, you must compare the value they represent and not the exact text. IMPORTANT: YOU MUST ONLY OUTPUT yes OR no. DO NOT ADD ANY ADDITIONAL TEXT.
    Answer 1: {entity1}
    Answer 2: {entity2}

    Response: """

        return PROMPT_TEMPLATE

    def get_bleu_score(self, predictions, ground_truths):
        import evaluate
        bleu = evaluate.load("sacrebleu")
        # self.bert_score = evaluate.load("bertscore")
        bleu_results = bleu.compute(predictions = predictions, references = [[elem] for elem in ground_truths])
        print("====BLEU RESULTS====")
        print(bleu_results)
    
    def get_rouge_score(self, predictions, ground_truths):
        rouge = evaluate.load("rouge")
        rouge_results = rouge.compute(predictions = predictions, references = [[elem] for elem in ground_truths])
        print("=====ROUGE RESULTS=====")
        print(rouge_results)
    
    def get_bleurt_score(self, predictions, ground_truths):
        import evaluate
        bleurt = evaluate.load("bleurt")
        bleurt_results = bleurt.compute(predictions = predictions, references = ground_truths)
        mean_value = np.mean(bleurt_results['scores'])
        print("======BLEURT RESULTS========")
        print(mean_value)
        
    def regex_match(self, gold_answer, generated_answer):
        # Escape special characters in the gold answer to match it literally
        escaped_gold_answer = re.escape(gold_answer)
        
        # Compile the escaped gold answer into a regex pattern
        pattern = re.compile(escaped_gold_answer)
        
        # Search for the pattern in the generated answer
        match = pattern.search(generated_answer)
        
        if match:
            return True
        else:
            return False