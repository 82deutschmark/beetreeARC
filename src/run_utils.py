import re
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from src.audit_prompts import build_logic_prompt, build_consistency_prompt
from src.judges import run_judge

def find_task_path(task_id: str) -> Path:
    if task_id.endswith(".json"):
        p = Path(task_id)
        if p.exists():
            return p
        task_id = p.stem
    candidate = Path("data/arc-agi-2-evaluation") / f"{task_id}.json"
    if candidate.exists():
        return candidate
    candidate_training = Path("data/arc-agi-2-training") / f"{task_id}.json"
    if candidate_training.exists():
        return candidate_training
    raise FileNotFoundError(f"Task file for '{task_id}' not found in data/arc-agi-2-evaluation/ or data/arc-agi-2-training/.")

def is_solved(candidates_object) -> bool:
    if not candidates_object:
        return False

    total_model_runs = sum(group['count'] for group in candidates_object.values())
    if total_model_runs == 0:
        return False

    sorted_groups = sorted(candidates_object.values(), key=lambda g: g['count'], reverse=True)
    top_group = sorted_groups[0]
    max_count = top_group['count']
    
    percentage = (max_count / total_model_runs)
    
    # Condition 1: count > 25%
    # Condition 2: count >= 4
    if not (percentage > 0.25 and max_count >= 4):
        return False
        
    # Condition 3: all other groups have exactly count=1
    for group in sorted_groups[1:]:
        if group['count'] != 1:
            return False
            
    return True

def pick_solution(candidates_object):
    # Model priority mapping (higher number = higher priority)
    MODEL_PRIORITY = {
        "claude-opus-4.5-thinking-60000": 4,
        "gemini-3-high": 3,
        "gpt-5.1-high": 2,
        "claude-sonnet-4.5-thinking-60000": 1
    }

    def get_group_priority(group):
        max_priority = 0
        for run_id in group['models']:
            # run_id format is typically "model-name_count_step"
            # We try to match the longest possible prefix in our priority map
            for model_name, priority in MODEL_PRIORITY.items():
                if run_id.startswith(model_name):
                    if priority > max_priority:
                        max_priority = priority
        return max_priority

    # Sort by count (descending) and then by priority (descending)
    sorted_groups = sorted(
        candidates_object.values(), 
        key=lambda g: (g['count'], get_group_priority(g)), 
        reverse=True
    )
    
    print("\n" + "="*40)
    print("FINAL OUTCOME")
    print("="*40)
    
    is_solved_flag = False
    unknown_status = False
    top_groups = sorted_groups[:2]
    
    if len(top_groups) > 0:
        if top_groups[0].get("is_correct") is None:
            unknown_status = True
        elif top_groups[0]["is_correct"]:
            is_solved_flag = True
        elif len(top_groups) > 1 and top_groups[1]["is_correct"]:
            is_solved_flag = True
        
    if unknown_status:
        print("Outcome: SUBMITTED (No Ground Truth)")
    elif is_solved_flag:
        print("Outcome: SOLVED")
    else:
        print("Outcome: FAILED")

    print("\n--- Debug Info ---")
    if not top_groups:
        print("No solutions generated.")
    else:
        for i, group in enumerate(top_groups):
            correctness = group.get('is_correct')
            c_str = "Unknown" if correctness is None else str(correctness)
            priority = get_group_priority(group)
            print(f"Group {i+1}: Count={group['count']}, Priority={priority}, Correct={c_str}")
            print(f"  Models: {', '.join(group['models'])}")

    # Check for other correct groups
    other_correct = []
    for i, group in enumerate(sorted_groups):
        if i < 2:
            continue
        if group.get('is_correct') is True:
            other_correct.append((i + 1, group))
            
    if other_correct:
        print("\n--- Other Correct Groups ---")
        for rank, group in other_correct:
            print(f"Group {rank}: Count={group['count']}, Correct={group['is_correct']}")
            print(f"  Models: {', '.join(group['models'])}")
            
    return top_groups, is_solved_flag, {}

def pick_solution_v2(candidates_object, reasoning_store, task, test_index, openai_client, anthropic_client, google_keys, judge_model="gemini-3-high"):
    """
    Advanced solution picker using TWO LLM Judges (Logic & Consistency).
    Replicates methodology from solution_handler.py:
    - Attempt 1: Consensus (Vote Count)
    - Attempt 2: Auditor Choice (Max Score of Logic/Consistency)
    """
    print("\n[pick_solution_v2] Starting Advanced Solution Picker (Multi-Judge)")
    
    # 0. Preparation
    train_examples = task.train
    test_input = task.test[test_index-1].input if task.test and len(task.test) >= test_index else None
    
    # Flatten candidates for easy indexing
    candidates_list = []
    for idx, (grid_tuple, val) in enumerate(candidates_object.items()):
        candidates_list.append({
            "id": idx,
            "grid": val.get("grid"),
            "models": val.get("models"),
            "count": val.get("count"),
            "is_correct": val.get("is_correct"),
            "reasoning": {} 
        })
    
    print(f"[pick_solution_v2] Total unique candidates found: {len(candidates_list)}")

    # Init Metadata
    selection_metadata = {
        "judges": {},
        "selection_process": {}
    }

    if not candidates_list:
        print("[pick_solution_v2] No candidates found.")
        return [], False, selection_metadata

    # 1. Extract Reasoning
    for cand in candidates_list:
        for model_id in cand["models"]:
            if model_id in reasoning_store:
                cand["reasoning"][model_id] = reasoning_store[model_id]

    # 3. Build Prompts
    full_prompt_logic = build_logic_prompt(train_examples, test_input, candidates_list)
    full_prompt_cons = build_consistency_prompt(train_examples, test_input, candidates_list)

    # 4. Run Judges
    scores = {c['id']: 0.0 for c in candidates_list}

    # Data Holders
    logic_data = { "prompt": full_prompt_logic, "response": None, "parsed": None }
    cons_data = { "prompt": full_prompt_cons, "response": None, "parsed": None }

    with ThreadPoolExecutor(max_workers=2) as executor:
        future_logic = executor.submit(run_judge, "Logic", full_prompt_logic, judge_model, openai_client, anthropic_client, google_keys, logic_data)
        future_cons = executor.submit(run_judge, "Consistency", full_prompt_cons, judge_model, openai_client, anthropic_client, google_keys, cons_data)
        
        logic_res = future_logic.result()
        cons_res = future_cons.result()

    # Update Scores based on results
    if logic_res and "candidates" in logic_res:
        for c in logic_res["candidates"]:
            cid = c.get("candidate_id")
            if cid in scores:
                scores[cid] = max(scores[cid], c.get("score", 0))
    
    if cons_res and "candidates" in cons_res:
        for c in cons_res["candidates"]:
            cid = c.get("candidate_id")
            if cid in scores:
                scores[cid] = max(scores[cid], c.get("score", 0))

    # 2. Determine Attempt 1 (Consensus)
    # Sort by count desc, then by score desc
    candidates_by_consensus = sorted(candidates_list, key=lambda c: (c['count'], scores[c['id']]), reverse=True)
    attempt_1_candidate = candidates_by_consensus[0]
    
    # Determine if it was a tie-break
    is_tie_break = False
    if len(candidates_by_consensus) > 1:
        second_best = candidates_by_consensus[1]
        if second_best['count'] == attempt_1_candidate['count']:
            is_tie_break = True
            
    label = "Consensus (Tie-Break by Score)" if is_tie_break else "Consensus (Vote)"
    print(f"[pick_solution_v2] Attempt 1 ({label}): Candidate {attempt_1_candidate['id']} (Votes: {attempt_1_candidate['count']}, Score: {scores[attempt_1_candidate['id']]})")

    # 5. Determine Attempt 2 (Auditor)
    # Sort candidates by Max Score
    sorted_by_score = sorted(candidates_list, key=lambda c: scores[c['id']], reverse=True)
    
    attempt_2_candidate = None
    # Pick the best scoring candidate that is NOT Attempt 1
    for cand in sorted_by_score:
        if cand['id'] != attempt_1_candidate['id']:
            attempt_2_candidate = cand
            break
            
    final_selection = [attempt_1_candidate]
    if attempt_2_candidate:
        print(f"[pick_solution_v2] Attempt 2 (Auditor): Candidate {attempt_2_candidate['id']} (Max Score: {scores[attempt_2_candidate['id']]})")
        final_selection.append(attempt_2_candidate)
    else:
        print("[pick_solution_v2] Attempt 2 (Auditor): No distinct candidate found. Using Consensus only.")
    
    # Debug Scoring
    print("\n[pick_solution_v2] Final Scores:")
    for cand in candidates_list:
        print(f"  ID {cand['id']}: Score={scores[cand['id']]}, Votes={cand['count']}")

    # 6. Populate Metadata
    selection_metadata["judges"]["logic"] = logic_data
    selection_metadata["judges"]["consistency"] = cons_data
    selection_metadata["selection_process"] = {
        "candidates_summary": [
            {"id": c['id'], "votes": c['count'], "score": scores[c['id']]} 
            for c in candidates_list
        ],
        "attempt_1": {"type": "Consensus", "candidate_id": attempt_1_candidate['id'], "vote_count": attempt_1_candidate['count']},
        "attempt_2": {
            "type": "Auditor", 
            "candidate_id": attempt_2_candidate['id'] if attempt_2_candidate else None, 
            "audit_score": scores[attempt_2_candidate['id']] if attempt_2_candidate else None
        }
    }

    # 7. Construct Return Output
    top_groups = []
    for cand in final_selection:
        grid_tuple = tuple(tuple(row) for row in cand['grid'])
        top_groups.append(candidates_object[grid_tuple])

    # 8. Final Success Check
    print("\n" + "="*40)
    print("FINAL OUTCOME (V2 - Multi-Judge)")
    print("="*40)
    
    is_solved_flag = False
    unknown_status = False
    
    for i, group in enumerate(top_groups):
        correctness = group.get("is_correct")
        role = "Auditor"
        if i == 0:
            role = label

        print(f"Attempt {i+1} ({role}) Correctness: {correctness}")
        
        if correctness is None:
            unknown_status = True
        elif correctness:
            is_solved_flag = True
            
    if unknown_status:
        print("Outcome: SUBMITTED (No Ground Truth)")
    elif is_solved_flag:
        print("Outcome: SOLVED")
    else:
        print("Outcome: FAILED")

    return top_groups, is_solved_flag, selection_metadata
