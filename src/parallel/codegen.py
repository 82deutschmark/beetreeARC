import re
import sys
import copy

def extract_and_run_solver(llm_code: str, test_input_grid: list, train_examples: list = None) -> tuple[list | None, dict | None]:
    """
    Extracts Python code from LLM response, executes it, calls solver(), 
    and returns the predicted grid.
    
    If train_examples is provided, it verifies the solver against all training pairs first.
    Returns: (predicted_grid, verification_log)
    """
    verification_log = {"train_results": [], "status": "UNKNOWN"}
    
    code = llm_code
    # Try to extract from markdown block
    pattern = r"```python(.*?)```"
    match = re.search(pattern, llm_code, re.DOTALL)
    if match:
        code = match.group(1).strip()
    else:
        # Heuristic: If no markdown, try to find the start of the function definition
        if "def solver" in llm_code:
            lines = llm_code.splitlines()
            start_idx = -1
            for i, line in enumerate(lines):
                if "def solver" in line:
                    start_idx = i
                    break
            if start_idx != -1:
                code = "\n".join(lines[start_idx:])
    
    local_scope = {}
    
    # Inject common utilities into the execution scope
    import math
    import itertools
    import copy
    import numpy as np
    import scipy
    from collections import Counter, deque, defaultdict
    from typing import List, Optional, Tuple, Any, Dict, Set
    
    local_scope = {
        "np": np,
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
        "copy": copy.copy,
        "deepcopy": copy.deepcopy,
        "gcd": math.gcd,
        "math": math,
        "itertools": itertools,
        "Grid": List[List[int]]
    }

    try:
        # Execute the code definition
        exec(code, {}, local_scope)
    except Exception as e:
        print(f"DEBUG: Code Definition FAILED: {e}", file=sys.stderr)
        print(f"--- FAILED CODE ---\n{code}\n-------------------", file=sys.stderr)
        verification_log["status"] = "FAIL_DEFINE"
        verification_log["error"] = str(e)
        return None, verification_log
        
    if "solver" not in local_scope:
        print("DEBUG: Solver function 'solver' not found in generated code.", file=sys.stderr)
        verification_log["status"] = "FAIL_NO_SOLVER"
        return None, verification_log
        
    solver_func = local_scope["solver"]
    if not callable(solver_func):
        print("DEBUG: 'solver' was defined but is not callable.", file=sys.stderr)
        verification_log["status"] = "FAIL_SOLVER_NOT_CALLABLE"
        return None, verification_log

    # Verification Step
    if train_examples:
        for i, ex in enumerate(train_examples):
            entry = {
                "index": i, 
                "status": "UNKNOWN",
                # Store as plain lists/data for JSON logging
                "input": ex.input, 
                "expected": ex.output,
                "actual": None
            }
            try:
                # Deepcopy input to prevent mutation side-effects affecting subsequent runs
                input_copy = copy.deepcopy(ex.input)
                
                # Run solver on training input
                res = solver_func(input_copy)
                entry["actual"] = res
                
                # Check against training output
                if res != ex.output:
                    # Explicitly NOT printing for FAIL_VERIFICATION as per instructions
                    entry["status"] = "FAIL"
                    verification_log["train_results"].append(entry)
                    verification_log["status"] = "FAIL_VERIFICATION"
                    verification_log["failed_example_index"] = i
                    return None, verification_log
                else:
                    entry["status"] = "PASS"
                    verification_log["train_results"].append(entry)
                    
            except Exception as e:
                print(f"DEBUG: Solver CRASHED on Train Example {i+1}: {e}", file=sys.stderr)
                import traceback
                traceback.print_exc(file=sys.stderr)
                entry["status"] = "CRASH"
                entry["error"] = str(e)
                verification_log["train_results"].append(entry)
                verification_log["status"] = "FAIL_CRASH"
                verification_log["failed_example_index"] = i
                return None, verification_log
        
        verification_log["status"] = "PASS"
        
    try:
        # Run the solver against the test input
        # Note: test_input_grid is already a list of lists (Python object)
        result = solver_func(test_input_grid)
        
        # Basic validation: must be list of lists
        if isinstance(result, list):
             # Ensure it's not a flat list? Or just robustly handle
             if len(result) > 0 and isinstance(result[0], list):
                 return result, verification_log
             # Handle empty grid case or other variations if needed
             if len(result) == 0:
                 return result, verification_log
        
        print("DEBUG: Solver returned invalid type (not list of lists).", file=sys.stderr)
        verification_log["test_run_error"] = "Result validation failed (not list of lists)"
        return None, verification_log
    except Exception as e:
        print(f"DEBUG: Solver CRASHED on Test Input: {e}", file=sys.stderr)
        verification_log["test_run_error"] = f"Test execution failed: {str(e)}"
        return None, verification_log
