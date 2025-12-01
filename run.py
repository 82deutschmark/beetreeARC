import argparse
import sys
import warnings

from src.solver_engine import run_solver_mode
from src.default_engine import run_default_mode

def main():
    parser = argparse.ArgumentParser(description="Run a single ARC task test case with multiple models in parallel.")
    parser.add_argument("--task", required=True, help="Task ID (e.g., 38007db0)")
    parser.add_argument("--test", type=int, default=1, help="Test case index (1-based, default: 1)")
    parser.add_argument("--workers", type=int, default=10, help="Number of parallel workers (default: 10)")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--models", type=str, help="Comma-separated list of models to run")
    parser.add_argument("--hint", type=str, default=None, help="Optional hint to provide to the model")
    parser.add_argument("--image", action="store_true", help="Generate an image for the task and include it in the prompt.")
    parser.add_argument("--trigger-deep-thinking", action="store_true", help="Append a deep thinking procedure to the prompt.")
    parser.add_argument("--generate-hint", action="store_true", help="Generate a hint for the task using a separate model call.")
    parser.add_argument("--generate-hint-model", type=str, default="gpt-5.1-high", help="Model to use for generating hints.")
    parser.add_argument("--solver", action="store_true", help="Enable solver mode.")
    
    args = parser.parse_args()

    # Suppress Pydantic warnings from the Anthropic library
    warnings.filterwarnings("ignore", message=r"Pydantic serializer warnings:", category=UserWarning)

    if args.solver:
        run_solver_mode(args.task, args.test, args.verbose)
    else:
        run_default_mode(args)

if __name__ == "__main__":
    main()