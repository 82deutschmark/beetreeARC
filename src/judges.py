import json
import re
from src.models import call_model

def run_judge(judge_name, prompt, judge_model, openai_client, anthropic_client, google_keys, data_holder):
    print(f"\n[pick_solution_v2] Running {judge_name} Judge ({judge_model})...")
    try:
        response_obj = call_model(openai_client, anthropic_client, google_keys, prompt, judge_model)
        data_holder["response"] = response_obj.text
        
        json_match = re.search(r"\{{.*\}}", response_obj.text, re.DOTALL)
        if json_match:
            parsed_json = json.loads(json_match.group(0))
            data_holder["parsed"] = parsed_json
            return parsed_json
    except Exception as e:
        print(f"[pick_solution_v2] {judge_name} Judge Error: {e}")
        data_holder["error"] = str(e)
    return None
