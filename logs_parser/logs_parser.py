import argparse
import os
import re
import json
import statistics

def load_answers(base_dir):
    answers = {}
    # Try looking for answers/ in the current working directory first
    answers_dir = os.path.join(base_dir, "answers")
    if not os.path.isdir(answers_dir):
        # Fallback: look in the script's directory
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

def parse_logs(directory):
    if not os.path.isdir(directory):
        print(f"Error: Directory '{directory}' does not exist.")
        return

    # Load answers relative to current working directory
    answers = load_answers(os.getcwd())

    files = os.listdir(directory)
    # Regex to capture task_id, test_id, and step from .json files
    # Matches: ...<task>_<test>_step_<step_name>.json OR ...<task>_<test>_step_finish.json
    pattern = re.compile(r'([a-f0-9]{8})_(\d+)_step_([a-zA-Z0-9]+)\.json$')

    # Structure: {(task, test): {"steps": {step_name: [calls]}, "finish_data": {...}, "model_statuses": {}, "step_statuses": {}}}
    task_data = {}

    for filename in files:
        match = pattern.search(filename)
        if match:
            task_id = match.group(1)
            test_id = match.group(2)
            step_name = match.group(3) # '1', '3', '5', 'finish' etc.
            
            # Skip steps 2 and 4
            if step_name in ["2", "4"]:
                continue

            key = (task_id, int(test_id))
            if key not in task_data:
                task_data[key] = {"steps": {}, "finish_data": None, "model_statuses": {}, "step_statuses": {}}
            
            filepath = os.path.join(directory, filename)
            try:
                with open(filepath, 'r') as f:
                    content = json.load(f)
                    
                    if step_name == "finish":
                        task_data[key]["finish_data"] = content
                        
                        if isinstance(content, dict):
                            # Check for root-level result
                            if "result" in content:
                                task_data[key]["finish_status"] = content["result"]
                        
                        # Extract judges info for display
                        cleaned_calls = []
                        if isinstance(content, dict) and "selection_details" in content:
                            sel_details = content["selection_details"]
                            if isinstance(sel_details, dict) and "judges" in sel_details:
                                judges = sel_details["judges"]
                                if isinstance(judges, dict):
                                    for judge_name, judge_data in judges.items():
                                        if isinstance(judge_data, dict):
                                            duration = judge_data.get("duration_seconds", 0)
                                            cost = judge_data.get("total_cost", 0)
                                            model = judge_data.get("model", "")
                                            
                                            display_name = f"Judge ({judge_name.capitalize()})"
                                            if model:
                                                display_name += f" - {model}"
                                                
                                            cleaned_calls.append({
                                                "name": display_name,
                                                "duration": duration,
                                                "cost": cost,
                                                "status": ""
                                            })
                        
                        task_data[key]["steps"]["finish"] = cleaned_calls
                        continue # Don't process finish as a regular step
                    
                    if step_name == "5":
                        if isinstance(content, dict) and content.get("is_solved") is True:
                             task_data[key]["step_statuses"]["5"] = True

                        # Handle nested structure for step 5
                        # content is { "sub-step": { "call": ... }, ... }
                        for sub_step, calls_dict in content.items():
                            if not isinstance(calls_dict, dict):
                                continue # Skip non-dict sub-step data

                            new_step_name = f"5-{sub_step}"
                            cleaned_calls = []
                            
                            # Define keys that act as nested containers
                            nested_containers = ["hint_generation", "gemini_gen", "opus_gen"]
                            
                            for call_key, call_val in calls_dict.items():
                                if call_key in nested_containers and isinstance(call_val, dict):
                                    # Process nested calls
                                    for inner_call, inner_val in call_val.items():
                                        if not isinstance(inner_val, dict):
                                            continue
                                            
                                        if "_step_" in inner_call:
                                            cleaned_name = inner_call.split("_step_")[0]
                                        else:
                                            cleaned_name = inner_call
                                        
                                        model = inner_val.get("model", "")
                                        if model:
                                            cleaned_name += f" ({model})"
                                        
                                        duration = inner_val.get("duration_seconds", 0)
                                        cost = inner_val.get("total_cost", 0)
                                        
                                        is_correct = inner_val.get("is_correct")
                                        
                                        # Fallback to ground truth check ONLY if is_correct is missing
                                        if is_correct is None and "Extracted grid" in inner_val:
                                            extracted = inner_val["Extracted grid"]
                                            if task_id in answers:
                                                idx = int(test_id) - 1
                                                if 0 <= idx < len(answers[task_id]):
                                                    correct_grid = answers[task_id][idx]
                                                    if extracted == correct_grid:
                                                        is_correct = True
                                                    else:
                                                        is_correct = False

                                        status_str = ""
                                        if is_correct is True:
                                            status_str = "PASS"
                                        elif is_correct is False:
                                            status_str = "FAIL"
                                        
                                        cleaned_calls.append({
                                            "name": cleaned_name,
                                            "duration": duration,
                                            "cost": cost,
                                            "status": status_str
                                        })
                                else:
                                    # Process normal call
                                    if "_step_" in call_key:
                                        cleaned_name = call_key.split("_step_")[0]
                                    else:
                                        cleaned_name = call_key
                                    
                                    if isinstance(call_val, dict):
                                        duration = call_val.get("duration_seconds", 0)
                                        cost = call_val.get("total_cost", 0)
                                        
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

                                        status_str = ""
                                        if is_correct is True:
                                            status_str = "PASS"
                                        elif is_correct is False:
                                            status_str = "FAIL"
                                    else:
                                        duration = 0
                                        cost = 0
                                        status_str = ""
                                    
                                    cleaned_calls.append({
                                        "name": cleaned_name,
                                        "duration": duration,
                                        "cost": cost,
                                        "status": status_str
                                    })
                            
                            task_data[key]["steps"][new_step_name] = cleaned_calls
                    else:
                        # Handle flat structure for other steps (1, 3, etc.)
                        if isinstance(content, dict) and content.get("is_solved") is True:
                             task_data[key]["step_statuses"][step_name] = True

                        cleaned_calls = []
                        for call_key, call_val in content.items():
                            if call_key == "is_solved": continue # Skip the status key itself

                            if "_step_" in call_key:
                                cleaned_name = call_key.split("_step_")[0]
                            else:
                                cleaned_name = call_key
                            
                            if isinstance(call_val, dict): # Defensive check
                                duration = call_val.get("duration_seconds", 0)
                                cost = call_val.get("total_cost", 0)
                                
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

                                status_str = ""
                                if is_correct is True:
                                    status_str = "PASS"
                                elif is_correct is False:
                                    status_str = "FAIL"
                            else: # Not a dict, set defaults
                                duration = 0
                                cost = 0
                                status_str = ""
                            
                            cleaned_calls.append({
                                "name": cleaned_name,
                                "duration": duration,
                                "cost": cost,
                                "status": status_str
                            })
                        
                        task_data[key]["steps"][step_name] = cleaned_calls

            except json.JSONDecodeError:
                print(f"Warning: Could not decode JSON from {filename}")
            except Exception as e:
                print(f"Warning: Could not read {filename}: {e}")

    # Determine max name length for pretty printing
    max_name_len = 0
    for key in task_data:
        for step_name in task_data[key]["steps"]:
            for call_info in task_data[key]["steps"][step_name]:
                if len(call_info["name"]) > max_name_len:
                    max_name_len = len(call_info["name"])
    # Add some padding
    max_name_len = max(max_name_len, 20) + 2 # Ensure a minimum width and add padding


    # Sort pairs: task (string), test (int)
    sorted_keys = sorted(task_data.keys(), key=lambda x: (x[0], x[1]))

    total_tasks_count = 0
    solved_tasks_count = 0
    vote_only_solved_count = 0
    score_only_solved_count = 0

    for task, test in sorted_keys:
        total_tasks_count += 1
        
        # --- Standard Strategy (already implemented) ---
        status = "FAILED" # Default status
        finish_data = task_data[(task, test)]["finish_data"]
        
        # Check explicit result from finish file first
        if "finish_status" in task_data[(task, test)]:
             status_val = task_data[(task, test)]["finish_status"]
             if status_val == "PASS":
                 status = "SOLVED"
             elif status_val == "FAIL":
                 status = "FAILED"
             else:
                 status = status_val # Use raw value if unknown
        # Fallback to candidates check
        elif finish_data and isinstance(finish_data, dict) and "candidates_object" in finish_data:
            candidates_obj = finish_data["candidates_object"]
            if isinstance(candidates_obj, dict):
                for candidate_key, candidate_val in candidates_obj.items():
                    if isinstance(candidate_val, dict) and candidate_val.get("is_correct") is True:
                        status = "SOLVED"
                        break
        
        if status == "SOLVED":
            solved_tasks_count += 1

        # --- Vote only Strategy ---
        vote_only_solved = False
        if finish_data and isinstance(finish_data, dict) and "candidates_object" in finish_data:
            candidates_obj = finish_data["candidates_object"]
            if isinstance(candidates_obj, dict):
                sorted_candidates = sorted(
                    [val for val in candidates_obj.values() if isinstance(val, dict)],
                    key=lambda x: x.get("count", 0),
                    reverse=True
                )
                top_two_candidates = sorted_candidates[:2]
                for candidate in top_two_candidates:
                    if candidate.get("is_correct") is True:
                        vote_only_solved = True
                        break
        if vote_only_solved:
            vote_only_solved_count += 1

        # --- Score only Strategy ---
        score_only_solved = False
        if finish_data and isinstance(finish_data, dict) and "selection_details" in finish_data:
            sel_details = finish_data["selection_details"]
            if isinstance(sel_details, dict) and "selection_process" in sel_details:
                sel_process = sel_details["selection_process"]
                if isinstance(sel_process, dict) and "candidates_summary" in sel_process:
                    candidates_summary = sel_process["candidates_summary"]
                    if isinstance(candidates_summary, list):
                        sorted_scored_candidates = sorted(
                            [c for c in candidates_summary if isinstance(c, dict) and "score" in c],
                            key=lambda x: x.get("score", 0),
                            reverse=True
                        )
                        top_two_scored_candidates = sorted_scored_candidates[:2]
                        
                        # Need to link scored candidate back to the main candidates_object to check is_correct
                        if finish_data and "candidates_object" in finish_data:
                            all_candidates_obj = finish_data["candidates_object"]
                            for scored_candidate in top_two_scored_candidates:
                                candidate_grid_str = json.dumps(scored_candidate.get("candidate_grid_as_tuple")) # Stored as tuple in summary
                                if candidate_grid_str in all_candidates_obj:
                                    if all_candidates_obj[candidate_grid_str].get("is_correct") is True:
                                        score_only_solved = True
                                        break
        if score_only_solved:
            score_only_solved_count += 1
            
        print(f"{task}:{test} {status}")
        
        steps_dict = task_data[(task, test)]["steps"]
        step_statuses = task_data[(task, test)]["step_statuses"]
        
        # Sort steps: integers first, then strings (like 'finish')
        sorted_steps = sorted(steps_dict.keys(), key=lambda s: (0, int(s)) if s.isdigit() else (1, s))
        
        for step in sorted_steps:
            step_solved_mark = ""
            # Check if this step (or the parent step for substeps) is marked as solved
            # For 5-image, 5-hint etc, we look for "5" in step_statuses
            lookup_step = step
            if "-" in step:
                lookup_step = step.split("-")[0]
            
            if step_statuses.get(lookup_step) is True:
                step_solved_mark = " [SOLVED]"

            print(f"  {step}{step_solved_mark}")
            for call_info in steps_dict[step]:
                name = call_info["name"]
                duration = call_info["duration"]
                cost = call_info["cost"]
                status_val = call_info["status"]
                
                print(f"    {name:<{max_name_len}} {duration:8.2f}s  ${cost:9.4f}  {status_val}")

    print("-" * 80)
    print("Model Summary")
    print("-" * 80)

    model_stats = {}

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

    for key in task_data:
        steps = task_data[key]["steps"]
        for step_name, calls in steps.items():
            for call in calls:
                raw_name = call["name"]
                model_name = normalize_model_name(raw_name)

                if model_name not in model_stats:
                    model_stats[model_name] = {
                        "solver_attempts": 0, 
                        "total_calls": 0,
                        "passes": 0, 
                        "durations": [], 
                        "costs": []
                    }

                model_stats[model_name]["total_calls"] += 1

                status = call["status"]
                if status in ["PASS", "FAIL"]:
                    model_stats[model_name]["solver_attempts"] += 1
                    if status == "PASS":
                        model_stats[model_name]["passes"] += 1
                
                # Collect valid durations
                if isinstance(call.get("duration"), (int, float)) and call["duration"] > 0:
                     model_stats[model_name]["durations"].append(call["duration"])

                # Collect valid costs
                if isinstance(call.get("cost"), (int, float)):
                     model_stats[model_name]["costs"].append(call["cost"])

    max_model_len = 0
    for m in model_stats:
        max_model_len = max(max_model_len, len(m))
    max_model_len = max(max_model_len, 10)

    print(f"{'Model':<{max_model_len}}  {'Solver Attempts':<15}  {'Total Calls':<12}  {'Passed':<8}  {'Pass Rate'}")

    sorted_models = sorted(model_stats.keys())
    for m in sorted_models:
        stats = model_stats[m]
        attempts = stats["solver_attempts"]
        total = stats["total_calls"]
        passed = stats["passes"]
        rate = (passed / attempts) * 100 if attempts > 0 else 0
        print(f"{m:<{max_model_len}}  {attempts:<15}  {total:<12}  {passed:<8}  {rate:6.2f}%")

    print("\n" + "-" * 80)
    print("Model Timing Statistics")
    print("-" * 80)
    
    print(f"{'Model':<{max_model_len}}  {'Avg (s)':<10}  {'95% (s)':<10}  {'Max (s)'}")
    
    for m in sorted_models:
        stats = model_stats[m]
        durations = sorted(stats["durations"])
        
        if not durations:
            print(f"{m:<{max_model_len}}  {'-':<10}  {'-':<10}  {'-'}")
            continue
            
        avg_time = statistics.mean(durations)
        max_time = max(durations)
        
        # Calculate 95th percentile
        k = 0.95 * (len(durations) - 1)
        f = int(k)
        c = int(k) + 1
        if c >= len(durations):
            p95 = durations[-1]
        elif f == c:
             p95 = durations[int(k)]
        else:
             p95 = durations[f] * (c - k) + durations[c] * (k - f)

        print(f"{m:<{max_model_len}}  {avg_time:<10.2f}  {p95:<10.2f}  {max_time:.2f}")

    print("\n" + "-" * 80)
    print("Model Cost Statistics")
    print("-" * 80)

    grand_total_cost = sum(sum(stats["costs"]) for stats in model_stats.values())

    print(f"{'Model':<{max_model_len}}  {'Avg Cost':<10}  {'Total Cost':<12}  {'% of Total'}")

    for m in sorted_models:
        stats = model_stats[m]
        costs = stats["costs"]
        
        if not costs:
             print(f"{m:<{max_model_len}}  {'-':<10}  {'-':<12}  {'-'}")
             continue
        
        avg_cost = statistics.mean(costs)
        total_cost = sum(costs)
        
        percentage_of_total = (total_cost / grand_total_cost) * 100 if grand_total_cost > 0 else 0
        
        print(f"{m:<{max_model_len}}  ${avg_cost:<9.4f}  ${total_cost:<11.4f}  {percentage_of_total:8.2f}%")

    print("\n" + "-" * 80)
    print("Strategy Performance")
    print("-" * 80)

    print(f"{'Strategy':<20}  {'Solved':<8}  {'Failed':<8}  {'Success Rate'}")
    
    # Standard Strategy
    failed_count = total_tasks_count - solved_tasks_count
    rate = (solved_tasks_count / total_tasks_count) * 100 if total_tasks_count > 0 else 0
    print(f"{'Standard':<20}  {solved_tasks_count:<8}  {failed_count:<8}  {rate:6.2f}%")

    # Vote only Strategy
    vote_only_failed = total_tasks_count - vote_only_solved_count
    vote_only_rate = (vote_only_solved_count / total_tasks_count) * 100 if total_tasks_count > 0 else 0
    print(f"{'Vote only':<20}  {vote_only_solved_count:<8}  {vote_only_failed:<8}  {vote_only_rate:6.2f}%")

    # Score only Strategy
    score_only_failed = total_tasks_count - score_only_solved_count
    score_only_rate = (score_only_solved_count / total_tasks_count) * 100 if total_tasks_count > 0 else 0
    print(f"{'Score only':<20}  {score_only_solved_count:<8}  {score_only_failed:<8}  {score_only_rate:6.2f}%")

def main():
    parser = argparse.ArgumentParser(description="Parse log files to extract task and test IDs.")
    parser.add_argument("directory", help="Path to the logs directory")
    args = parser.parse_args()

    parse_logs(args.directory)

if __name__ == "__main__":
    main()