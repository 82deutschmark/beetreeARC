from datetime import datetime
import json
import sys
from pathlib import Path

def generate_submission(final_results, submission_dir_path: str, run_timestamp: str):
    submission_dir = Path(submission_dir_path)
    submission_dir.mkdir(parents=True, exist_ok=True)
    submission_file = submission_dir / "submission.json"
    
    submission_data = {}
    
    for task_id, test_idx, preds in final_results:
        if not preds:
            continue
        
        # Save individual task/test result
        individual_file = submission_dir / f"{run_timestamp}_{task_id}_{test_idx}.json"
        try:
            with open(individual_file, "w") as f:
                json.dump(preds, f, indent=2)
        except Exception as e:
            print(f"Error saving individual result for {task_id}:{test_idx}: {e}", file=sys.stderr)
        
    # Re-process to format correctly
    formatted_submission = {}
    
    # Group by task_id
    task_results = {}
    for task_id, test_idx, preds in final_results:
            if task_id not in task_results:
                task_results[task_id] = {}
            task_results[task_id][test_idx] = preds

    for task_id, tests in task_results.items():
        # We need to determine the number of tests. 
        # We can assume the max test_idx found is the number of tests? 
        # Or just use the indices we have.
        # If we are running a subset, we can't produce a valid full submission file anyway.
        # So let's just output what we have, sorted by test_index.
        
        max_idx = max(tests.keys())
        formatted_submission[task_id] = []
        task_aggregated_data = []
        
    for task_id, tests in task_results.items():
        # We need to determine the number of tests. 
        # We can assume the max test_idx found is the number of tests? 
        # Or just use the indices we have.
        # If we are running a subset, we can't produce a valid full submission file anyway.
        # So let's just output what we have, sorted by test_index.
        
        max_idx = max(tests.keys())
        formatted_submission[task_id] = []
        task_aggregated_data = []
        
        for i in range(1, max_idx + 1):
            preds_raw = tests.get(i)
            
            solutions = preds_raw
            usage_stats = None
            
            # Unpack if tuple (solutions, usage_stats)
            if isinstance(preds_raw, tuple) and len(preds_raw) == 2 and isinstance(preds_raw[1], dict):
                solutions, usage_stats = preds_raw
            
            attempt_1 = [[0]] # Default empty/fail
            attempt_2 = [[0]]
            correct_1 = False
            correct_2 = False
            reasoning_1 = None
            reasoning_2 = None
            
            if solutions:
                candidates = []
                # solutions is expected to be a list of candidate dicts from pick_solution_v2
                # e.g. [{"grid": ..., "is_correct": ...}, ...]
                if isinstance(solutions, list):
                    for p in solutions:
                        if isinstance(p, dict) and "grid" in p:
                            candidates.append(p)
                # Fallback legacy checks just in case (though we expect list of dicts now)
                elif isinstance(solutions, tuple) and len(solutions) == 2 and isinstance(solutions[0], list):
                        pass
                        
                if candidates:
                    c1 = candidates[0]
                    attempt_1 = c1["grid"]
                    # explicit check for True to handle None/False safely
                    correct_1 = c1.get("is_correct") is True
                    reasoning_1 = c1.get("reasoning_summary")
                    
                    if len(candidates) > 1:
                        c2 = candidates[1]
                        attempt_2 = c2["grid"]
                        correct_2 = c2.get("is_correct") is True
                        reasoning_2 = c2.get("reasoning_summary")
                    else:
                        # duplicate attempt 1 if no attempt 2
                        attempt_2 = attempt_1
                        correct_2 = correct_1
                        reasoning_2 = reasoning_1
            
            formatted_submission[task_id].append({
                "attempt_1": attempt_1,
                "attempt_2": attempt_2
            })
            
            end_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            
            if usage_stats:
                def halve(v, is_int=False):
                    if v is None: return None
                    return v // 2 if is_int else v / 2

                usage_data = {
                    "prompt_tokens": halve(usage_stats.get("prompt_tokens"), True),
                    "completion_tokens": halve(usage_stats.get("completion_tokens"), True),
                    "total_tokens": halve(usage_stats.get("total_tokens"), True),
                    "completion_tokens_details": {
                        "reasoning_tokens": halve(usage_stats.get("reasoning_tokens", 0), True),
                        "accepted_prediction_tokens": halve(usage_stats.get("accepted_prediction_tokens"), True),
                        "rejected_prediction_tokens": halve(usage_stats.get("rejected_prediction_tokens", 0), True)
                    }
                }
                cost_data = {
                    "prompt_cost": halve(usage_stats.get("prompt_cost")),
                    "completion_cost": halve(usage_stats.get("completion_cost")),
                    "reasoning_cost": halve(usage_stats.get("reasoning_cost")),
                    "total_cost": halve(usage_stats.get("total_cost"))
                }
            
            metadata_template_1 = {
                "model": "Johan_Land_Solver_V6",
                "provider": "Johan_Land",
                "start_timestamp": run_timestamp,
                "end_timestamp": end_timestamp,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "user",
                            "content": "NA"
                        }
                    },
                    {
                        "index": 1,
                        "message": {
                            "role": "assistant",
                            "content": "NA"
                        }
                    }
                ],
                "reasoning_summary": reasoning_1,
                "kwargs": {
                    "background": "mixed",
                    "stream": "mixed",
                    "reasoning": {
                        "effort": "max with fallbacks for latency, cost and stability"
                    },
                    "max_output_tokens": "max less delta for stability"
                },
                "usage": usage_data,
                "cost": cost_data,
                "task_id": task_id,
                "pair_index": i - 1,
                "test_id": "Johan_Land_Solver_V6_Eval_2_Full_Run"
            }
            
            metadata_template_2 = metadata_template_1.copy()
            metadata_template_2["reasoning_summary"] = reasoning_2

            task_aggregated_data.append({
                "attempt_1": {
                    "answer": attempt_1, 
                    "correct": correct_1,
                    "metadata": metadata_template_1
                },
                "attempt_2": {
                    "answer": attempt_2, 
                    "correct": correct_2,
                    "metadata": metadata_template_2
                }
            })

        # Save task-level aggregated file (without timestamp)
        task_file = submission_dir / f"{task_id}.json"
        try:
            with open(task_file, "w") as f:
                json.dump(task_aggregated_data, f, indent=2)
        except Exception as e:
            print(f"Error saving task file {task_file}: {e}", file=sys.stderr)

    with open(submission_file, "w") as f:
        json.dump(formatted_submission, f)
        
    print(f"Submission file saved to: {submission_file}")
