import re

try:
    from parsing_utils.common import create_call_info
except ImportError:
    from logs_parser.parsing_utils.common import create_call_info

def parse_generic_step(content, task_id, test_id, answers):
    result = {
        "calls": [],
        "solved": False
    }

    if isinstance(content, dict) and content.get("is_solved") is True:
         result["solved"] = True

    if not isinstance(content, dict):
        return result

    for call_key, call_val in content.items():
        if call_key == "is_solved": continue 

        cleaned_name = re.sub(r'_\d+(\.\d+)?$', '', call_key)
        
        result["calls"].append(create_call_info(cleaned_name, call_val, task_id, test_id, answers, run_id=call_key))
        
    return result