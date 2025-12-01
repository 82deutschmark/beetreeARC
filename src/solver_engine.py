import sys
import time
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import httpx
from openai import OpenAI
from anthropic import Anthropic
from google import genai


# Import from existing project modules
from src.config import get_api_keys
from src.tasks import load_task, build_prompt
from src.image_generation import generate_and_save_image
from src.hint_generation import generate_hint
from src.logging import setup_logging
from src.parallel import run_single_model
from src.run_utils import find_task_path, pick_solution, is_solved

def run_models_in_parallel(models_to_run, run_id_counts, step_name, prompt, test_example, openai_client, anthropic_client, google_client, verbose, image_path=None):
    all_results = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        
        # Generate unique run IDs
        run_list = []
        for model_name in models_to_run:
            count = run_id_counts.get(model_name, 0) + 1
            run_id_counts[model_name] = count
            run_id = f"{model_name}_{count}_{step_name}"
            run_list.append({"name": model_name, "run_id": run_id})

        future_to_run_id = {
            executor.submit(run_single_model, run["name"], run["run_id"], prompt, test_example, openai_client, anthropic_client, google_client, verbose, image_path): run["run_id"]
            for run in run_list
        }

        for future in as_completed(future_to_run_id):
            res = future.result()
            if res:
                all_results.append(res)
    return all_results

def run_solver_mode(task_id: str, test_index: int, verbose: bool):
    print("Solver mode activated.")
    setup_logging(verbose)

    def write_step_log(step_name: str, data: dict):
        log_path = Path("logs") / f"{task_id}_{test_index}_{step_name}.json"
        with open(log_path, "w") as f:
            json.dump(data, f, indent=4, default=lambda o: '<not serializable>')
        if verbose:
            print(f"Saved log for {step_name} to {log_path}")

    try:
        task_path = find_task_path(task_id)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    openai_key, claude_key, google_key = get_api_keys()
    http_client = httpx.Client(timeout=3600.0)
    openai_client = OpenAI(api_key=openai_key, http_client=http_client) if openai_key else None
    anthropic_client = Anthropic(api_key=claude_key, http_client=http_client) if claude_key else None
    google_client = genai.Client(api_key=google_key) if google_key else None

    try:
        task = load_task(task_path)
    except Exception as e:
        print(f"Error loading task: {e}", file=sys.stderr)
        http_client.close()
        sys.exit(1)

    test_idx = test_index - 1
    if test_idx < 0 or test_idx >= len(task.test):
        print(f"Error: Test index {test_index} is out of range.", file=sys.stderr)
        http_client.close()
        sys.exit(1)
    test_example = task.test[test_idx]
    
    run_id_counts = {}
    candidates_object = {}

    def process_results(results, step_log):
        nonlocal candidates_object
        initial_solutions = len(candidates_object)
        for res in results:
            if res:
                run_key = f"{res['run_id']}_{time.time()}"
                step_log[run_key] = {
                    "Full raw LLM call": res["prompt"],
                    "Full raw LLM response": res["full_response"],
                    "Extracted grid": res["grid"],
                }
                if res["grid"] is not None:
                    grid_tuple = tuple(tuple(row) for row in res["grid"])
                    if grid_tuple not in candidates_object:
                        candidates_object[grid_tuple] = {"grid": res["grid"], "count": 0, "models": [], "is_correct": res["is_correct"]}
                    candidates_object[grid_tuple]["count"] += 1
                    candidates_object[grid_tuple]["models"].append(res["run_id"])
        new_solutions = len(candidates_object) - initial_solutions
        print(f"Found {new_solutions} new unique solutions.")

    # STEP 1
    print("\n--- STEP 1: Initial model run ---")
    step_1_log = {}
    models_step1 = ["claude-sonnet-4.5-no-thinking", "gpt-5.1-none", "gpt-5.1-none"]
    print(f"Running {len(models_step1)} models...")
    prompt_step1 = build_prompt(task.train, test_example)
    results_step1 = run_models_in_parallel(models_step1, run_id_counts, "step_1", prompt_step1, test_example, openai_client, anthropic_client, google_client, verbose)
    process_results(results_step1, step_1_log)
    write_step_log("step_1", step_1_log)

    # STEP 2
    print("\n--- STEP 2: First check ---")
    solved_prob = is_solved(candidates_object)
    step_2_log = {"candidates_object": {str(k): v for k, v in candidates_object.items()}, "is_solved_prob": solved_prob}
    write_step_log("step_2", step_2_log)
    if solved_prob > 0.9:
        print("is_solved() > 0.9, moving to STEP FINISH.")
        picked_solutions, result = pick_solution(candidates_object)
        finish_log = {"candidates_object": {str(k): v for k, v in candidates_object.items()}, "picked_solutions": picked_solutions, "result": "PASS" if result else "FAIL", "correct_solution": test_example.output}
        write_step_log("step_finish", finish_log)
        http_client.close()
        return

    # STEP 3
    print("\n--- STEP 3: Second model run ---")
    step_3_log = {}
    models_step3 = ["claude-sonnet-4.5-no-thinking", "gpt-5.1-none"]
    print(f"Running {len(models_step3)} models...")
    prompt_step3 = build_prompt(task.train, test_example)
    results_step3 = run_models_in_parallel(models_step3, run_id_counts, "step_3", prompt_step3, test_example, openai_client, anthropic_client, google_client, verbose)
    process_results(results_step3, step_3_log)
    write_step_log("step_3", step_3_log)

    # STEP 4
    print("\n--- STEP 4: Second check ---")
    solved_prob = is_solved(candidates_object)
    step_4_log = {"candidates_object": {str(k): v for k, v in candidates_object.items()}, "is_solved_prob": solved_prob}
    write_step_log("step_4", step_4_log)
    if solved_prob > 0.9:
        print("is_solved() > 0.9, moving to STEP FINISH.")
        picked_solutions, result = pick_solution(candidates_object)
        finish_log = {"candidates_object": {str(k): v for k, v in candidates_object.items()}, "picked_solutions": picked_solutions, "result": "PASS" if result else "FAIL", "correct_solution": test_example.output}
        write_step_log("step_finish", finish_log)
        http_client.close()
        return

    # STEP 5
    print("\n--- STEP 5: Final model runs ---")
    step_5_log = {"trigger-deep-thinking": {}, "image": {}, "generate-hint": {}}
    models_step5 = ["claude-sonnet-4.5-no-thinking", "gpt-5.1-none"]
    
    print(f"Running {len(models_step5)} models with deep thinking...")
    prompt_deep = build_prompt(task.train, test_example, trigger_deep_thinking=True)
    results_deep = run_models_in_parallel(models_step5, run_id_counts, "step_5_deep_thinking", prompt_deep, test_example, openai_client, anthropic_client, google_client, verbose)
    process_results(results_deep, step_5_log["trigger-deep-thinking"])
    
    print(f"Running {len(models_step5)} models with image...")
    image_path = f"logs/{task_id}_{test_index}_step5_image.png"
    generate_and_save_image(task, image_path)
    prompt_image = build_prompt(task.train, test_example, image_path=image_path)
    results_image = run_models_in_parallel(models_step5, run_id_counts, "step_5_image", prompt_image, test_example, openai_client, anthropic_client, google_client, verbose, image_path=image_path)
    process_results(results_image, step_5_log["image"])

    print(f"Running {len(models_step5)} models with generated hint...")
    hint_image_path = f"logs/{task_id}_{test_index}_step5_generate_hint.png"
    hint_data = generate_hint(task, hint_image_path, "gpt-5.1-none", verbose)
    if hint_data and hint_data["hint"]:
        step_5_log["generate-hint"]["hint_generation"] = {
            "Full raw LLM call": hint_data["prompt"],
            "Full raw LLM response": hint_data["full_response"],
            "Extracted hint": hint_data["hint"],
        }
        prompt_hint = build_prompt(task.train, test_example, strategy=hint_data["hint"])
        results_hint = run_models_in_parallel(models_step5, run_id_counts, "step_5_generate_hint", prompt_hint, test_example, openai_client, anthropic_client, google_client, verbose)
        process_results(results_hint, step_5_log["generate-hint"])
    
    write_step_log("step_5", step_5_log)

    # STEP FINISH
    print("\n--- STEP FINISH: Pick and print solution ---")
    picked_solutions, result = pick_solution(candidates_object)
    finish_log = {"candidates_object": {str(k): v for k, v in candidates_object.items()}, "picked_solutions": picked_solutions, "result": "PASS" if result else "FAIL", "correct_solution": test_example.output}
    write_step_log("step_finish", finish_log)
        
    http_client.close()
