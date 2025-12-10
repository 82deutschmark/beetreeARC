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
        
        for i in range(1, max_idx + 1):
            preds = tests.get(i)
            attempt_1 = [[0]] # Default empty/fail
            attempt_2 = [[0]]
            
            if preds:
                candidates = []
                if isinstance(preds, list):
                    for p in preds:
                        if isinstance(p, dict) and "grid" in p:
                            candidates.append(p["grid"])
                elif isinstance(preds, tuple) and len(preds) == 2 and isinstance(preds[0], list):
                        # Handle potential differnt return format if any
                        pass
                        
                if candidates:
                    attempt_1 = candidates[0]
                    attempt_2 = candidates[1] if len(candidates) > 1 else candidates[0]
            
            formatted_submission[task_id].append({
                "attempt_1": attempt_1,
                "attempt_2": attempt_2
            })

    with open(submission_file, "w") as f:
        json.dump(formatted_submission, f)
        
    print(f"Submission file saved to: {submission_file}")
