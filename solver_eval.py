import sys
import os
from pathlib import Path
from typing import List, Optional
import json

# Add current directory to path so we can import from src
sys.path.append(os.getcwd())

from src.tasks.loading import load_task
from src.grid import grid_to_string

# ========================================== 
# SOLVER RESOLUTION
# ========================================== 

def default_solver(input_grid: List[List[int]]) -> List[List[int]]:
    """Default placeholder solver (Identity)."""
    return input_grid

def get_solver():
    """Detects if a solver is piped via stdin; otherwise returns default."""
    if not sys.stdin.isatty():
        try:
            code = sys.stdin.read()
            if not code.strip():
                return default_solver
            
            # Mirror the Sandbox environment from src/sandbox.py
            import math
            import itertools
            import copy
            from collections import Counter, deque, defaultdict
            from typing import List, Optional, Tuple, Any, Dict, Set
            
            try:
                import numpy as np
            except ImportError:
                np = None

            try:
                import scipy
                import scipy.ndimage
            except ImportError:
                scipy = None

            try:
                import cv2
            except ImportError:
                cv2 = None

            namespace = {
                "np": np,
                "cv2": cv2,
                "scipy": scipy,
                "Counter": Counter,
                "deque": deque,
                "defaultdict": defaultdict,
                "List": List,
                "Optional": Optional,
                "Tuple": Tuple,
                "Any": Any,
                "Dict": Dict,
                "Set": Set,
                "copy": copy,
                "deepcopy": copy.deepcopy,
                "gcd": math.gcd,
                "math": math,
                "itertools": itertools,
                "Grid": List[List[int]]
            }
            
            # Execute the piped code
            exec(code, namespace)
            
            if 'solver' in namespace:
                return namespace['solver']
            else:
                print("Warning: Stdin provided but no 'solver' function found in the code. Using default identity solver.")
                return default_solver
        except Exception as e:
            print(f"Error loading solver from stdin: {e}")
            return default_solver
    return default_solver

# ========================================== 
# EVALUATION LOGIC
# ========================================== 

def find_task_path(task_id: str) -> Path:
    """Finds the JSON file for a given task ID in common data directories."""
    if task_id.endswith(".json"):
        p = Path(task_id)
        if p.exists():
            return p
        task_id = p.stem

    search_dirs = [
        "data/evaluation-arc-agi-1",
        "data/evaluation-arc-agi-2",
        "data/training-arc-agi-2",
        "data/training",
        "data/evaluation",
    ]
    
    for d in search_dirs:
        candidate = Path(d) / f"{task_id}.json"
        if candidate.exists():
            return candidate
            
    raise FileNotFoundError(f"Could not find task '{task_id}' in data directories.")

def print_grid_comparison(name: str, expected: List[List[int]], actual: List[List[int]]):
    print(f"\n--- {name} ---")
    
    exp_str = grid_to_string(expected).splitlines()
    act_str = grid_to_string(actual).splitlines()
    
    max_h = max(len(exp_str), len(act_str))
    exp_str += [""] * (max_h - len(exp_str))
    act_str += [""] * (max_h - len(act_str))
    
    width = max(len(s) for s in exp_str) if exp_str else 0
    
    print(f"{ 'EXPECTED':<{width}}   {'ACTUAL'}")
    for e, a in zip(exp_str, act_str):
        print(f"{e:<{width}} | {a}")
    print("----------------")

def evaluate_solver(task_str: str, solver_func):
    try:
        if ":" in task_str:
            task_id, test_idx_str = task_str.split(":")
            test_idx = int(test_idx_str)
        else:
            task_id = task_str
            test_idx = 1
            
        print(f"Loading task: {task_id}")
        task_path = find_task_path(task_id)
        print(f"Found at: {task_path}")
        
        task = load_task(task_path)
        
        print(f"\nEvaluating Training Examples ({len(task.train)}):")
        train_correct = 0
        for i, example in enumerate(task.train):
            print(f"  Train {i+1}: ", end="")
            try:
                result = solver_func(example.input)
                
                # Convert numpy arrays to lists
                if hasattr(result, 'tolist'):
                    result = result.tolist()
                    
                if result == example.output:
                    print("✅ PASS")
                    train_correct += 1
                else:
                    print("❌ FAIL")
                    print_grid_comparison(f"Train {i+1} Failure", example.output, result)
            except Exception as e:
                print(f"❌ ERROR: {e}")

        print(f"\nTraining Score: {train_correct}/{len(task.train)}")
        
        print(f"\nEvaluating Test Example {test_idx}:")
        if test_idx < 1 or test_idx > len(task.test):
            print(f"❌ Invalid test index {test_idx}. Task has {len(task.test)} test examples.")
            return

        test_example = task.test[test_idx - 1]
        
        try:
            result = solver_func(test_example.input)
            
            # Convert numpy arrays to lists
            if hasattr(result, 'tolist'):
                result = result.tolist()
            
            if test_example.output is not None:
                if result == test_example.output:
                    print("✅ PASS")
                else:
                    print("❌ FAIL")
                    print_grid_comparison(f"Test {test_idx} Failure", test_example.output, result)
            else:
                print("❓ DONE (No Ground Truth for Test)")
                print("\nGenerated Output:")
                print(grid_to_string(result))
                
        except Exception as e:
             print(f"❌ ERROR: {e}")

    except Exception as e:
        print(f"Fatal Error: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python solver_eval.py <task_id>[:<test_index>]")
        print("Example: python solver_eval.py 007bbfb7:1")
        print("Example (with pipe): cat my_solver.py | python solver_eval.py 007bbfb7:1")
        sys.exit(1)
        
    active_solver = get_solver()
    evaluate_solver(sys.argv[1], active_solver)
