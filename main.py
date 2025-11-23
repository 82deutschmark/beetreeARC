from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List

from openai import OpenAI
from anthropic import Anthropic
from google import genai

from src.models import (
    call_model,
    parse_model_arg,
    SUPPORTED_MODELS,
    PRICING_PER_1M_TOKENS,
    ResultRecord,
    ORDERED_MODELS,
)
from src.tasks import load_task, load_task_paths, build_prompt
from src.utils import parse_grid_from_text, verify_prediction

def get_column_name(model_arg: str) -> str:
    parts = model_arg.split("-")
    formatted_parts = [p.title() if not p[0].isdigit() else p for p in parts]
    name = "-".join(formatted_parts)
    name = name.replace("Gpt", "GPT")
    return name

TABLE_COLUMNS = [get_column_name(m) for m in ORDERED_MODELS]

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Send ARC-AGI tasks to OpenAI.")
    parser.add_argument(
        "task_list",
        type=Path,
        help="JSON file containing a list of task paths (e.g. data/first_100.json).",
    )
    parser.add_argument(
        "--model",
        choices=sorted(SUPPORTED_MODELS),
        default="gpt-5.1-none",
        help="Model to use (e.g. gpt-5.1-none, gpt-5.1-low).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=20,
        help="Number of parallel worker threads (default: 20).",
    )
    return parser.parse_args()

def solve_task(
    openai_client: OpenAI,
    anthropic_client: Anthropic,
    google_client: genai.Client,
    task_path: Path,
    model_arg: str,
) -> List[ResultRecord]:
    task = load_task(task_path)
    outcomes: List[ResultRecord] = []
    for idx, test_example in enumerate(task.test, start=1):
        prompt = build_prompt(task.train, test_example)
        success = False
        duration = 0.0
        cost = 0.0
        try:
            start_time = time.perf_counter()
            model_response = call_model(
                openai_client, anthropic_client, google_client, prompt, model_arg
            )
            duration = time.perf_counter() - start_time

            _, base_model, _ = parse_model_arg(model_arg)
            pricing = PRICING_PER_1M_TOKENS.get(
                base_model, {"input": 0, "cached_input": 0, "output": 0}
            )

            if (
                base_model == "gemini-3-pro-preview"
                and model_response.prompt_tokens > 200000
            ):
                pricing = {"input": 4.00, "cached_input": 0.0, "output": 18.00}

            non_cached_input = max(
                0, model_response.prompt_tokens - model_response.cached_tokens
            )

            cost = (
                (non_cached_input / 1_000_000 * pricing["input"])
                + (
                    model_response.cached_tokens
                    / 1_000_000
                    * pricing.get("cached_input", 0)
                )
                + (model_response.completion_tokens / 1_000_000 * pricing["output"])
            )

            predicted_grid = parse_grid_from_text(model_response.text)
            success = verify_prediction(predicted_grid, test_example.output)
        except Exception as exc:
            print(f"Task {task_path} test {idx} failed: {exc}", file=sys.stderr)
        outcomes.append((task_path, idx, success, model_arg, duration, cost))
    return outcomes

def print_table_header() -> None:
    columns = ["#", "Task", "Test"] + TABLE_COLUMNS
    print("| " + " | ".join(columns) + " |")
    print("| " + " | ".join(["---"] * len(columns)) + " |")

def print_result_row(
    row_idx: int,
    task_path: Path,
    test_idx: int,
    success: bool,
    model_arg: str,
    duration: float,
    cost: float,
) -> None:
    column_key = get_column_name(model_arg)

    if column_key not in TABLE_COLUMNS:
        print(
            f"Unsupported reasoning column {column_key}. Update TABLE_COLUMNS to include it.",
            file=sys.stderr,
        )
        return
    values = {column: "-" for column in TABLE_COLUMNS}
    values[column_key] = "PASS" if success else "FAIL"
    row = (
        [str(row_idx), str(task_path), str(test_idx)]
        + [values[col] for col in TABLE_COLUMNS]
        + [f"{duration:.2f}", f"{cost:.2f}"]
    )
    print("| " + " | ".join(row) + " |")

def main() -> None:
    args = parse_args()

    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        raise RuntimeError("OPENAI_API_KEY environment variable is not set.")
    openai_client = OpenAI(api_key=openai_key)

    claude_key = os.getenv("ANTHROPIC_API_KEY") or os.getenv("CLAUDE_API_KEY")
    anthropic_client = None
    if claude_key:
        anthropic_client = Anthropic(api_key=claude_key)

    google_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    google_client = None
    if google_key:
        google_client = genai.Client(api_key=google_key)

    try:
        task_paths = load_task_paths(args.task_list)
    except ValueError as exc:
        raise RuntimeError(f"Failed to read task list {args.task_list}: {exc}") from exc

    print_table_header()
    row_counter = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_path = {
            executor.submit(
                solve_task,
                openai_client,
                anthropic_client,
                google_client,
                path,
                args.model,
            ): path
            for path in task_paths
        }

        for future in as_completed(future_to_path):
            try:
                task_results = future.result()
                for record in task_results:
                    row_counter += 1
                    print_result_row(row_counter, *record)
            except Exception as exc:
                path = future_to_path[future]
                print(
                    f"Task execution for {path} generated an exception: {exc}",
                    file=sys.stderr,
                )

if __name__ == "__main__":
    main()