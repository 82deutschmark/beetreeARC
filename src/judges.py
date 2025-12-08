import json
import re
from src.models import call_model

def extract_json(text):
    """
    Robustly extract a JSON object from text.
    Prioritizes objects containing 'candidates' key.
    """
    if not text:
        return None
    text = text.strip()
    
    # 1. Try to find JSON block within markdown fences
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        try:
            obj = json.loads(match.group(1))
            if isinstance(obj, dict) and "candidates" in obj:
                return obj
        except json.JSONDecodeError:
            pass

    # 2. Scan for any '{' and try to decode a valid JSON object
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        start_idx = match.start()
        try:
            # raw_decode parsing stops at the end of the valid object
            obj, _ = decoder.raw_decode(text, idx=start_idx)
            if isinstance(obj, dict) and "candidates" in obj:
                return obj
        except json.JSONDecodeError:
            continue
            
    return None

def run_judge(judge_name, prompt, judge_model, openai_client, anthropic_client, google_keys, data_holder):
    print(f"\n[pick_solution_v2] Running {judge_name} Judge ({judge_model})...")
    try:
        response_obj = call_model(openai_client, anthropic_client, google_keys, prompt, judge_model)
        data_holder["response"] = response_obj.text
        
        parsed_json = extract_json(response_obj.text)
        
        if parsed_json:
            data_holder["parsed"] = parsed_json
            return parsed_json
        else:
            print(f"[pick_solution_v2] {judge_name} Judge: Could not parse JSON. Response start: {response_obj.text[:500]}")
            
    except Exception as e:
        print(f"[pick_solution_v2] {judge_name} Judge Error: {e}")
        data_holder["error"] = str(e)
    return None