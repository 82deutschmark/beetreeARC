def print_codegen_analysis(task_data):
    print("\n" + "-" * 80)
    print("Codegen Analysis")
    print("-" * 80)
    
    print(f"{ 'Task:Test':<15} {'Attempt':<10} {'Codegen Used':<14} {'Is Correct':<12} {'Train Correct':<14} {'Crashes'}")
    
    sorted_keys = sorted(task_data.keys(), key=lambda x: (x[0], x[1]))
    
    failed_codegen_info = []

    for task, test in sorted_keys:
        entry = task_data[(task, test)]
        finish_data = entry.get("finish_data")
        
        # 1. Gather all codegen calls from the steps
        # Key: run_id (name), Value: call_data
        codegen_calls = {}
        # Also keep a general lookup for all calls
        all_call_lookup = {}
        
        for step_name, calls in entry.get("steps", {}).items():
            for call in calls:
                all_call_lookup[call["run_id"]] = call
                all_call_lookup[call["name"]] = call
                if "codegen" in call.get("run_id", ""):
                    codegen_calls[call["run_id"]] = call
        
        # Helper to get stats from a call
        def get_call_stats(call_data):
            details = call_data.get("verification_details")
            train_correct = "-"
            crashes = 0
            if details:
                train_results = details.get("train_results", [])
                if train_results:
                    pass_count = sum(1 for r in train_results if r.get("status") == "PASS")
                    total_count = len(train_results)
                    train_correct = f"{pass_count}/{total_count}"
                    crashes = sum(1 for r in train_results if r.get("status") == "CRASH")
            return train_correct, crashes, call_data.get("llm_response", ""), call_data.get("extracted_grid")

        picked_solutions = []
        if finish_data and isinstance(finish_data, dict):
            picked_solutions = finish_data.get("picked_solutions", [])
        
        # Track which codegen calls have been reported via picked solutions
        reported_codegen_calls = set()

        if picked_solutions:
            for idx, solution in enumerate(picked_solutions):
                attempt_num = str(idx + 1)
                models = solution.get("models", [])
                is_correct_val = solution.get("is_correct", False)
                is_correct_str = "YES" if is_correct_val else "NO"
                
                codegen_used = "NO"
                train_correct_str = "-"
                crashes_count = 0
                model_response = ""
                grid_data = None
                
                # Check models associated with this solution
                for m in models:
                    if "codegen" in m:
                        codegen_used = "YES"
                        # m is the full run_id
                        lookup_key = m
                        
                        if lookup_key in codegen_calls:
                            reported_codegen_calls.add(lookup_key)
                            tc, cr, resp, gd = get_call_stats(codegen_calls[lookup_key])
                            train_correct_str = tc
                            crashes_count = cr
                            if not model_response:
                                model_response = resp
                                grid_data = gd
                        else:
                            # Fallback to stripped name lookup if full ID not found
                            stripped_key = m.split("_step_")[0]
                            # Look through codegen_calls values for a matching name
                            for rid, cdata in codegen_calls.items():
                                if cdata["name"] == stripped_key:
                                    reported_codegen_calls.add(rid)
                                    tc, cr, resp, gd = get_call_stats(cdata)
                                    train_correct_str = tc
                                    crashes_count = cr
                                    if not model_response:
                                        model_response = resp
                                        grid_data = gd
                                    break
                
                print(f"{f'{task}:{test}':<15} {attempt_num:<10} {codegen_used:<14} {is_correct_str:<12} {train_correct_str:<14} {crashes_count}")
                
                # Check if all training examples passed
                train_all_passed = False
                if train_correct_str != "-" and "/" in train_correct_str:
                    parts = train_correct_str.split("/")
                    if len(parts) == 2 and parts[0] == parts[1] and parts[0] != "0":
                        train_all_passed = True

                if codegen_used == "YES" and is_correct_str == "NO" and model_response and train_all_passed:
                    failed_codegen_info.append({
                        "title": f"{task}:{test} Attempt {attempt_num}",
                        "code": model_response,
                        "grid": grid_data,
                        "correct_grid": finish_data.get("correct_solution") if finish_data else None
                    })
        
        # If no picked solutions, OR if there are unreported codegen calls, list them
        # This handles the "no attempt in _step_finish.json" case
        remaining_calls = [k for k in codegen_calls if k not in reported_codegen_calls]
        
        if not picked_solutions and not remaining_calls:
            # No picked solutions AND no codegen calls found at all -> Print default "NO" row
            print(f"{f'{task}:{test}':<15} {'-':<10} {'NO':<14} {'NO':<12} {'-':<14} -")
        
        elif remaining_calls:
            # We have codegen calls that weren't in picked solutions
            # (Either because picked_solutions was empty, or these specific models weren't picked)
            for i, call_key in enumerate(remaining_calls):
                attempt_label = "-"
                
                tc, cr, resp, gd = get_call_stats(codegen_calls[call_key])
                
                # Check correctness? If it wasn't picked, we might not know if it's "correct" 
                # unless we check the 'status' field of the call itself.
                # parsing.py sets call['status'] = "PASS"/"FAIL" based on ground truth check
                is_correct_str = "YES" if codegen_calls[call_key].get("status") == "PASS" else "NO"
                
                print(f"{f'{task}:{test}':<15} {attempt_label:<10} {'YES':<14} {is_correct_str:<12} {tc:<14} {cr}")
                
                # Check if all training examples passed
                train_all_passed = False
                if tc != "-" and "/" in tc:
                    parts = tc.split("/")
                    if len(parts) == 2 and parts[0] == parts[1] and parts[0] != "0":
                        train_all_passed = True

                if is_correct_str == "NO" and resp and train_all_passed:
                     failed_codegen_info.append({
                        "title": f"{task}:{test} Unpicked Codegen",
                        "code": resp,
                        "grid": gd,
                        "correct_grid": finish_data.get("correct_solution") if finish_data else None
                    })

    if failed_codegen_info:

        print("\n" + "=" * 80)
        print("Failed Codegen Dump")
        print("=" * 80)
        for info in failed_codegen_info:
            print(f"\n--- {info['title']} ---")
            raw_resp = info["code"]
            # Extract python block
            if "```python" in raw_resp:
                try:
                    start = raw_resp.index("```python") + 9
                    end = raw_resp.index("```", start)
                    print(raw_resp[start:end].strip())
                except ValueError:
                    print(raw_resp.strip())
            else:
                 print(raw_resp.strip())
            
            grid = info.get("grid")
            if grid is not None:
                print("\nGenerated Grid:")
                # Pretty print the grid
                if isinstance(grid, list):
                    print("[")
                    for row in grid:
                        print(f"  {row},")
                    print("]")
                else:
                    print(grid)

            correct_grid = info.get("correct_grid")
            if correct_grid is not None:
                print("\nCorrect Grid:")
                if isinstance(correct_grid, list):
                    print("[")
                    for row in correct_grid:
                        print(f"  {row},")
                    print("]")
                else:
                    print(correct_grid)
