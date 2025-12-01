import re
from pathlib import Path

MODEL_WEIGHTS = {
    "claude-opus-4.5-thinking-60000": 16,
    "gemini-3-high": 15,
    "claude-opus-4.5-thinking-16000": 14,
    "gpt-5.1-high": 13,
    "claude-sonnet-4.5-thinking-60000": 12,
    "claude-opus-4.5-thinking-4000": 11,
    "claude-opus-4.5-thinking-1024": 10,
    "claude-opus-4.5-no-thinking": 9,
    "claude-sonnet-4.5-thinking-16000": 8,
    "gemini-3-low": 7,
    "gpt-5.1-medium": 6,
    "claude-sonnet-4.5-thinking-4000": 5,
    "claude-sonnet-4.5-thinking-1024": 4,
    "claude-sonnet-4.5-no-thinking": 3,
    "gpt-5.1-low": 2,
    "gpt-5.1-none": 1
}

def find_task_path(task_id: str) -> Path:
    if task_id.endswith(".json"):
        p = Path(task_id)
        if p.exists():
            return p
        task_id = p.stem
    candidate = Path("data/arc-agi-2-evaluation") / f"{task_id}.json"
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"Task file for '{task_id}' not found in data/arc-agi-2-evaluation/.")

def get_group_sort_key(group):
    count = group['count']
    max_weight = 0
    for run_id in group['models']:
        # Strip the suffix like _1_step_1 to get the base model name
        base_model = re.sub(r'_\d+_.*$', '', run_id)
        weight = MODEL_WEIGHTS.get(base_model, 0)
        if weight > max_weight:
            max_weight = weight
    return (count, max_weight)

def is_solved(candidates_object) -> float:
    return 0.8

def pick_solution(candidates_object):
    sorted_groups = sorted(candidates_object.values(), key=get_group_sort_key, reverse=True)
    
    print("\n" + "="*40)
    print("FINAL OUTCOME")
    print("="*40)
    
    is_solved_flag = False
    top_groups = sorted_groups[:2]
    
    if len(top_groups) > 0 and top_groups[0]["is_correct"]:
        is_solved_flag = True
    elif len(top_groups) > 1 and top_groups[1]["is_correct"]:
        is_solved_flag = True
        
    if is_solved_flag:
        print("Outcome: SOLVED")
    else:
        print("Outcome: FAILED")

    print("\n--- Debug Info ---")
    if not top_groups:
        print("No solutions generated.")
    else:
        for i, group in enumerate(top_groups):
            print(f"Group {i+1}: Count={group['count']}, Correct={group['is_correct']}")
            print(f"  Models: {', '.join(group['models'])}")
            
    return top_groups, is_solved_flag
