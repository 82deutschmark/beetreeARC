import argparse
import sys
import time
import warnings
import queue
import multiprocessing
from pathlib import Path
from datetime import datetime
import concurrent.futures

from src.tasks import load_task
from src.run_utils import find_task_path
from src.dashboard import render_table, update_task_states
from src.execution import execute_task
from src.submission import generate_submission

# Conditional import for rich
try:
    from rich.live import Live
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

def main():
    run_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    parser = argparse.ArgumentParser(description="Run ARC task test cases with multiple models in parallel.")
    
    # Task selection group
    task_group = parser.add_mutually_exclusive_group(required=True)
    task_group.add_argument("--task", help="Task ID (e.g., 38007db0) or path to JSON file")
    task_group.add_argument("--task-directory", help="Directory containing task JSON files to run in batch")
    
    parser.add_argument("--test", type=int, default=1, help="Test case index (1-based, default: 1). Ignored if --task-directory is used.")
    parser.add_argument("--workers", type=int, default=10, help="Number of parallel workers (default: 10) PER TASK.")
    parser.add_argument("--task-workers", type=int, default=1, help="Number of tasks to run in parallel (default: 1). CAUTION: Divides global rate limits by this factor.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--models", type=str, help="Comma-separated list of models to run")
    parser.add_argument("--hint", type=str, default=None, help="Optional hint to provide to the model")
    parser.add_argument("--image", action="store_true", help="Generate an image for the task and include it in the prompt.")
    parser.add_argument("--trigger-deep-thinking", action="store_true", help="Append a deep thinking procedure to the prompt.")
    parser.add_argument("--generate-hint", action="store_true", help="Generate a hint for the task using a separate model call.")
    parser.add_argument("--generate-hint-model", type=str, default="gpt-5.1-high", help="Model to use for generating hints.")
    parser.add_argument("--no-dashboard", action="store_true", help="Disable the rich dashboard in batch mode.")
    parser.add_argument("--submissions-directory", type=str, default="submissions/", help="Directory to save submission files (default: submissions/).")

    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--solver", action="store_true", help="Enable solver mode.")
    mode_group.add_argument("--solver-testing", action="store_true", help="Enable solver testing mode with a smaller set of models.")
    
    args = parser.parse_args()
    
    warnings.filterwarnings("ignore", message=r"Pydantic serializer warnings:", category=UserWarning)

    if args.task_directory:
        if args.test != 1:
             print(f"Warning: --test argument ({args.test}) is ignored when using --task-directory. Running all tests for all tasks.", file=sys.stderr)

        directory = Path(args.task_directory)
        if not directory.exists() or not directory.is_dir():
            print(f"Error: Directory '{directory}' does not exist.", file=sys.stderr)
            sys.exit(1)
            
        task_files = list(directory.glob("*.json"))
        if not task_files:
            print(f"No JSON files found in '{directory}'.", file=sys.stderr)
            sys.exit(0)
            
        # Prepare tasks
        tasks_to_run = []
        for task_file in sorted(task_files):
            try:
                task = load_task(task_file)
                num_tests = len(task.test)
                for i in range(num_tests):
                    tasks_to_run.append((task_file, i + 1))
            except Exception as e:
                print(f"Error loading task {task_file}: {e}", file=sys.stderr)

        total_tasks = len(tasks_to_run)
        print(f"Found {len(task_files)} task files. Total test cases: {total_tasks}")
        print(f"Starting batch execution with {args.task_workers} parallel task workers...")
        
        rate_limit_scale = 1.0 / max(1, args.task_workers)
        
        # Determine if we should use the dashboard
        use_dashboard = RICH_AVAILABLE and sys.stdout.isatty() and not args.no_dashboard and (args.solver or args.solver_testing)

        final_results = []

        if use_dashboard:
            manager = multiprocessing.Manager()
            progress_queue = manager.Queue()
            task_states = {}
            
            # Start ProcessPoolExecutor
            with concurrent.futures.ProcessPoolExecutor(max_workers=args.task_workers) as executor:
                future_to_task = {
                    executor.submit(execute_task, args, task_path, test_idx, run_timestamp, rate_limit_scale, progress_queue, True): (task_path, test_idx)
                    for task_path, test_idx in tasks_to_run
                }
                
                remaining_futures = set(future_to_task.keys())
                shutdown_start_time = None
                
                with Live(render_table(task_states), refresh_per_second=4) as live:
                    while remaining_futures or not progress_queue.empty():
                        # 1. Drain queue non-blocking
                        while True:
                            try:
                                msg = progress_queue.get_nowait()
                                update_task_states(task_states, msg)
                            except queue.Empty:
                                break
                        
                        # 2. Check futures with small timeout
                        if remaining_futures:
                            done, not_done = concurrent.futures.wait(
                                remaining_futures, timeout=0.1, return_when=concurrent.futures.FIRST_COMPLETED
                            )
                            
                            for fut in done:
                                remaining_futures.remove(fut)
                                task_path, test_idx = future_to_task[fut]
                                try:
                                    res = fut.result()
                                    final_results.append(res)
                                except SystemExit as se:
                                    # Worker exited with sys.exit()
                                    key = f"{task_path.stem}:{test_idx}"
                                    if key not in task_states or task_states[key].get("status") != "COMPLETED":
                                        status = "COMPLETED" if se.code == 0 else "ERROR"
                                        outcome = "?" if se.code == 0 else "FAIL"
                                        step = "Finished (Silent)" if se.code == 0 else f"Exit Code {se.code}"
                                            
                                        update_task_states(task_states, {
                                            "task_id": task_path.stem,
                                            "test_index": test_idx,
                                            "status": status,
                                            "step": step,
                                            "outcome": outcome,
                                            "event": "FINISH",
                                            "timestamp": time.time()
                                        })
                                except Exception as e:
                                    key = f"{task_path.stem}:{test_idx}"
                                    if key not in task_states or task_states[key].get("status") != "COMPLETED":
                                         update_task_states(task_states, {
                                            "task_id": task_path.stem,
                                            "test_index": test_idx,
                                            "status": "ERROR",
                                            "step": f"Error: {str(e)}",
                                            "outcome": "FAIL",
                                            "event": "FINISH",
                                            "timestamp": time.time()
                                         })

                            # 3. Re-render
                            live.update(render_table(task_states))
                        else:
                            # All futures done. Now ensure we have received final messages for all tasks.
                            if shutdown_start_time is None:
                                shutdown_start_time = time.time()

                            # If queue has data, keep processing
                            if not progress_queue.empty():
                                time.sleep(0.1)
                                live.update(render_table(task_states))
                                continue
                                
                            # Check if any tasks are still technically "running" according to our state
                            all_terminal = True
                            for state in task_states.values():
                                if state.get("status") not in ("COMPLETED", "ERROR"):
                                    all_terminal = False
                                    break
                            
                            if all_terminal:
                                live.update(render_table(task_states))
                                break
                                
                            # Timeout safety: Don't wait forever if a message was dropped
                            if time.time() - shutdown_start_time > 5.0:
                                break
                                
                            time.sleep(0.1)
                            live.update(render_table(task_states))

        else:
            # Fallback to plain logging (interleaved)
            final_results = []
            with concurrent.futures.ProcessPoolExecutor(max_workers=args.task_workers) as executor:
                future_to_task = {
                    executor.submit(execute_task, args, task_path, test_idx, run_timestamp, rate_limit_scale, None, False): (task_path, test_idx)
                    for task_path, test_idx in tasks_to_run
                }
                
                for future in concurrent.futures.as_completed(future_to_task):
                    try:
                        res = future.result()
                        final_results.append(res)
                    except Exception as e:
                        print(f"Task failed: {e}", file=sys.stderr)
                        
        # Generate Submission File
        generate_submission(final_results, args.submissions_directory, run_timestamp)

    else:
        # Single task mode
        try:
            task_path = find_task_path(args.task)
            execute_task(args, task_path, args.test, run_timestamp)
        except FileNotFoundError as e:
             print(f"Error: {e}", file=sys.stderr)
             sys.exit(1)
        except SystemExit as se:
            sys.exit(se.code)

if __name__ == "__main__":
    main()
