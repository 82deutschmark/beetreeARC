try:
    from stats import determine_strategies_status
    from utils import normalize_model_name
except ImportError:
    from logs_parser.stats import determine_strategies_status
    from logs_parser.utils import normalize_model_name

def print_failed_task_model_stats(task_data):
    print("\n" + "-" * 80)
    print("Pass Frequency by Model Type (Failed Tasks Only)")
    print("-" * 80)
    
    model_counts = {}

    for task_key, entry in task_data.items():
        # Check if failed
        strategies = determine_strategies_status(entry)
        outcome = "SOLVED" if strategies["standard"] else "FAILED"
        if outcome != "FAILED":
            continue
            
        # Get passing unique run IDs
        passing_run_ids = set()
        for step_name, calls in entry["steps"].items():
            for call in calls:
                if call["status"] == "PASS":
                    passing_run_ids.add(call["name"])
        
        # Tally by model type
        for run_id in passing_run_ids:
            model_type = normalize_model_name(run_id)
            model_counts[model_type] = model_counts.get(model_type, 0) + 1

    print(f"{ 'Model Type':<40} {'Frequency'}")
    
    sorted_counts = sorted(model_counts.items(), key=lambda x: x[1], reverse=True)
    for model, count in sorted_counts:
        print(f"{model:<40} {count}")