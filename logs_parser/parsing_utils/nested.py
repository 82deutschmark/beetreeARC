import re

try:
    from parsing_utils.common import create_call_info
except ImportError:
    from logs_parser.parsing_utils.common import create_call_info

def parse_nested_step(content, task_id, test_id, answers):
    result = {
        "steps": {}, # sub-steps
        "solved": False
    }
    
    if isinstance(content, dict) and content.get("is_solved") is True:
        result["solved"] = True

    # Handle nested structure for step 5
    # content is { "sub-step": { "call": ... }, ... }
    for sub_step, calls_dict in content.items():
        if not isinstance(calls_dict, dict):
            continue 

        new_step_name = f"5-{sub_step}"
        cleaned_calls = []
        
        nested_containers = ["gemini_gen", "opus_gen"]
        
        for call_key, call_val in calls_dict.items():
            if call_key == "hint_generation" and isinstance(call_val, dict):
                # Handle hint_generation (direct call object)
                model = call_val.get("model", "")
                name = "Hint Generation"
                if model:
                    name += f" ({model})"
                
                # Check for cost/duration directly in call_val or if they are missing
                # Some logs might put stats in a wrapper, but schema says it's direct.
                cleaned_calls.append(create_call_info(name, call_val, task_id, test_id, answers, run_id=call_key))
                continue

            if call_key in nested_containers and isinstance(call_val, dict):
                # Determine generator
                generator_name = None
                if call_key == "gemini_gen":
                    generator_name = "Gemini"
                elif call_key == "opus_gen":
                    generator_name = "Opus"

                # Process nested calls
                for inner_call, inner_val in call_val.items():
                    if not isinstance(inner_val, dict):
                        continue
                        
                    cleaned_name = re.sub(r'_\d+(\.\d+)?$', '', inner_call)
                    
                    model = inner_val.get("model", "")
                    if model:
                        cleaned_name += f" ({model})"
                    
                    cleaned_calls.append(create_call_info(cleaned_name, inner_val, task_id, test_id, answers, generator=generator_name, run_id=inner_call))
            else:
                # Process normal call
                cleaned_name = re.sub(r'_\d+(\.\d+)?$', '', call_key)
                
                # Try to infer generator from call key string for objects_pipeline variants
                gen_name = None
                if "gemini_gen" in call_key:
                    gen_name = "Gemini"
                elif "opus_gen" in call_key:
                    gen_name = "Opus"
                
                cleaned_calls.append(create_call_info(cleaned_name, call_val, task_id, test_id, answers, generator=gen_name, run_id=call_key))
        
        result["steps"][new_step_name] = cleaned_calls
        
    return result