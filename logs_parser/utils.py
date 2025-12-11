import os
import json
import re

def load_answers(base_dir):
    answers = {}
    # Try looking for answers/ in the current working directory first (base_dir)
    answers_dir = os.path.join(base_dir, "answers")
    
    if not os.path.isdir(answers_dir):
        # Fallback: look in the directory where this script resides (logs_parser/)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        answers_dir = os.path.join(script_dir, "answers")
        
        if not os.path.isdir(answers_dir):
             return answers

    for filename in os.listdir(answers_dir):
        if filename.endswith(".json"):
            task_id = filename.split(".")[0]
            filepath = os.path.join(answers_dir, filename)
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    if "test" in data:
                        answers[task_id] = [t.get("output") for t in data["test"]]
            except Exception as e:
                print(f"Warning: Could not read answer file {filename}: {e}")
    return answers

def normalize_model_name(raw_name):
    # 1. Check for standard suffix: model_name_123
    match = re.match(r"^(.*)_\d+$", raw_name)
    if match:
        return match.group(1)
    
    # 2. Check for format: extraction (model_name) or transformation (model_name)
    match = re.search(r"\(([^)]+)\)$", raw_name)
    if match:
        return match.group(1)

    # 3. Check for format: Judge (Type) - model_name
    match = re.search(r" - (.*)$", raw_name)
    if match:
        return match.group(1)

    return raw_name