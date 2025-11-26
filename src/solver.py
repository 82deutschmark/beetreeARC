import sys
import time
from pathlib import Path
from typing import List, Optional

import httpx
from openai import OpenAI
from anthropic import Anthropic
from google import genai

from src.models import (
    call_model,
    parse_model_arg,
    calculate_cost,
)
from src.types import TaskResult
from src.tasks import load_task, build_prompt
from src.utils import (
    parse_grid_from_text,
    verify_prediction,
)
from src.logging import get_logger

logger = get_logger("solver")

def solve_task(
    openai_key: Optional[str],
    claude_key: Optional[str],
    google_key: Optional[str],
    task_path: Path,
    model_arg: str,
    strategy: str = None,
    verbose: bool = False,
    return_strategy: bool = False,
    verify: bool = False,
) -> List[TaskResult]:
    # Create a thread-local HTTP client with insecure SSL and long timeouts
    # to prevent connection errors and timeouts in threaded environments.
    http_client = httpx.Client(
        timeout=3600.0,
        transport=httpx.HTTPTransport(retries=3, verify=False),
        limits=httpx.Limits(keepalive_expiry=3600),
        verify=False
    )
    
    openai_client = OpenAI(api_key=openai_key, http_client=http_client) if openai_key else None
    
    anthropic_client = None
    if claude_key:
        anthropic_client = Anthropic(api_key=claude_key, http_client=http_client)
        
    google_client = None
    if google_key:
        google_client = genai.Client(api_key=google_key) # Gemini uses its own transport

    try:
        task = load_task(task_path)
        outcomes: List[TaskResult] = []
        for idx, test_example in enumerate(task.test, start=1):
            prompt = build_prompt(
                task.train,
                test_example,
                strategy=strategy,
            )
            
            success = False
            duration = 0.0
            cost = 0.0
            strategy_text = None
            verified = None
            
            # If verification is requested, we must extract the strategy to verify it
            should_extract = return_strategy or verify

            try:
                start_time = time.perf_counter()
                
                # Ensure we have the client for the requested model
                if model_arg.startswith("gpt") and not openai_client:
                     raise RuntimeError("OpenAI API key missing.")
                if "claude" in model_arg and not anthropic_client:
                     raise RuntimeError("Anthropic API key missing.")
                if "gemini" in model_arg and not google_client:
                     raise RuntimeError("Google API key missing.")

                model_response = call_model(
                    openai_client,
                    anthropic_client,
                    google_client,
                    prompt,
                    model_arg,
                    return_strategy=should_extract,
                    verbose=verbose,
                )
                if verbose:
                    if model_response.strategy:
                        logger.debug(f"--- STRATEGY ---\n{model_response.strategy}\n--- END STRATEGY ---")
                    logger.debug(f"--- OUTPUT ---\n{model_response.text}\n--- END OUTPUT ---")

                duration = time.perf_counter() - start_time

                model_config = parse_model_arg(model_arg)
                cost = calculate_cost(model_config, model_response)

                grid_text = model_response.text
                strategy_text = model_response.strategy
                
                predicted_grid = parse_grid_from_text(grid_text)
                success = verify_prediction(predicted_grid, test_example.output)

                # Verification Logic (LOOCV)
                if verify and strategy_text:
                    verified = True
                    
                    def _attempt_verification(subset_train, target_ex) -> bool:
                        """Helper to run a single verification pass."""
                        nonlocal duration, cost
                        _v_prompt = build_prompt(
                            subset_train,
                            target_ex,
                            strategy=strategy_text
                        )
                        _v_start = time.perf_counter()
                        try:
                            _v_resp = call_model(
                                openai_client,
                                anthropic_client,
                                google_client,
                                _v_prompt,
                                model_arg,
                                return_strategy=False,
                                verbose=verbose
                            )
                            duration += (time.perf_counter() - _v_start)
                            cost += calculate_cost(model_config, _v_resp)
                            _v_grid = parse_grid_from_text(_v_resp.text)
                            return verify_prediction(_v_grid, target_ex.output)
                        except Exception as _e:
                            logger.error(f"Verification error: {_e}")
                            return False

                    if len(task.train) <= 2:
                        # Small Dataset Mode: Require 2 successes for each example (allow up to 2 failures)
                        for i, train_ex in enumerate(task.train):
                            temp_train = task.train[:i] + task.train[i+1:]
                            
                            successes = 0
                            failures = 0
                            
                            # Loop until we get 2 successes or bust (fail > 2)
                            while successes < 2 and failures <= 2:
                                if _attempt_verification(temp_train, train_ex):
                                    successes += 1
                                else:
                                    failures += 1
                            
                            if successes < 2:
                                verified = False
                                if verbose:
                                    logger.info(f"Verification failed on training example {i+1} (Small Data Mode: {successes} pass, {failures} fail)")
                                break
                    else:
                        # Standard Mode: All must pass, allow 1 retry per example
                        for i, train_ex in enumerate(task.train):
                            temp_train = task.train[:i] + task.train[i+1:]
                            
                            # Attempt 1
                            passed = _attempt_verification(temp_train, train_ex)
                            
                            # Attempt 2 (Retry) if failed
                            if not passed:
                                if verbose:
                                    logger.info(f"Verification retry on training example {i+1}...")
                                passed = _attempt_verification(temp_train, train_ex)
                            
                            if not passed:
                                verified = False
                                if verbose:
                                    logger.info(f"Verification failed on training example {i+1}")
                                break

            except Exception as exc:
                logger.error(f"Task {task_path} test {idx} failed: {type(exc)} {exc}")
            
            outcomes.append(TaskResult(
                task_path=task_path,
                test_index=idx,
                success=success,
                model_arg=model_arg,
                duration=duration,
                cost=cost,
                strategy=strategy_text,
                verified=verified
            ))
        return outcomes
    finally:
        # Clean up the thread-local http client
        http_client.close()