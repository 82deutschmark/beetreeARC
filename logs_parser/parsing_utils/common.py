import json
import re

def check_correctness(call_val, task_id, test_id, answers):
    is_correct = call_val.get("is_correct")
    # Fallback to ground truth check ONLY if is_correct is missing
    if is_correct is None and "Extracted grid" in call_val:
        extracted = call_val["Extracted grid"]
        if task_id in answers:
            idx = int(test_id) - 1
            if 0 <= idx < len(answers[task_id]):
                correct_grid = answers[task_id][idx]
                if extracted == correct_grid:
                    is_correct = True
                else:
                    is_correct = False
    return is_correct

def create_call_info(name, data, task_id, test_id, answers, generator=None, run_id=None):
    duration = 0
    cost = 0
    input_tokens = 0
    output_tokens = 0
    cached_tokens = 0
    status_str = ""
    timing_breakdown = []
    
    if isinstance(data, dict):
        duration = data.get("duration_seconds", 0)
        cost = data.get("total_cost", 0)
        input_tokens = data.get("input_tokens", 0)
        output_tokens = data.get("output_tokens", 0)
        cached_tokens = data.get("cached_tokens", 0)
        timing_breakdown = data.get("timing_breakdown", [])
        is_correct = check_correctness(data, task_id, test_id, answers)
        
        if is_correct is True:
            status_str = "PASS"
        elif is_correct is False:
            status_str = "FAIL"
            
    extracted_grid_failed = False
    bad_grid = False
    verification_details = {}
    llm_response = ""
    extracted_grid = None
    if isinstance(data, dict):
        llm_response = data.get("Full raw LLM response", "")
        verification_details = data.get("verification_details", {})
        if "Extracted grid" in data:
            extracted_grid = data["Extracted grid"]
            if extracted_grid is None:
                extracted_grid_failed = True
            elif isinstance(extracted_grid, list):
                height = len(extracted_grid)
                width = 0
                if height > 0 and isinstance(extracted_grid[0], list):
                    width = len(extracted_grid[0])
                
                if height == 1 or width == 1:
                    bad_grid = True

    return {
        "name": name,
        "run_id": run_id if run_id else name,
        "duration": duration,
        "cost": cost,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cached_tokens": cached_tokens,
        "timing_breakdown": timing_breakdown,
        "status": status_str,
        "generator": generator,
        "extracted_grid_failed": extracted_grid_failed,
        "bad_grid": bad_grid,
        "verification_details": verification_details,
        "llm_response": llm_response,
        "extracted_grid": extracted_grid
    }
