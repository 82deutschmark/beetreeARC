import json

try:
    # Try local import (when running from logs_parser dir)
    from parsing_utils.common import check_correctness, create_call_info
    from parsing_utils.finish import parse_finish_step
    from parsing_utils.nested import parse_nested_step
    from parsing_utils.generic import parse_generic_step
except ImportError:
    # Fallback to absolute import (when running from root)
    from logs_parser.parsing_utils.common import check_correctness, create_call_info
    from logs_parser.parsing_utils.finish import parse_finish_step
    from logs_parser.parsing_utils.nested import parse_nested_step
    from logs_parser.parsing_utils.generic import parse_generic_step

def parse_log_file(filepath, task_id, test_id, step_name, answers):
    """
    Parses a single log file and returns a structured dictionary of results.
    """
    try:
        with open(filepath, 'r') as f:
            content = json.load(f)
    except Exception as e:
        print(f"Warning: Could not read {filepath}: {e}")
        return None

    if step_name == "finish":
        return {"type": "finish", "data": parse_finish_step(content)}
    elif step_name == "5":
        return {"type": "nested", "data": parse_nested_step(content, task_id, test_id, answers)}
    else:
        return {"type": "generic", "data": parse_generic_step(content, task_id, test_id, answers)}