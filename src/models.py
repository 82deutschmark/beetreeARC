import os
import sys
import time
import json
import requests
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Optional, Union

from openai import OpenAI
from anthropic import Anthropic
from google import genai
from google.genai import types

PRICING_PER_1M_TOKENS = {
    "gpt-5.1": {"input": 1.25, "cached_input": 0.125, "output": 10.00},
    "claude-sonnet-4-5-20250929": {
        "input": 3.00,
        "cached_input": 0.30,
        "output": 15.00,
    },
    "claude-opus-4-5-20251101": {
        "input": 5.00,
        "cached_input": 0.50,
        "output": 25.00,
    },
    "gemini-3-pro-preview": {
        "input": 2.00,
        "cached_input": 0.0,
        "output": 12.00,
    },
}

ORDERED_MODELS = [
    "gpt-5.1-none",
    "gpt-5.1-low",
    "gpt-5.1-medium",
    "gpt-5.1-high",
    "claude-sonnet-4.5-no-thinking",
    "claude-sonnet-4.5-thinking-1024",
    "claude-sonnet-4.5-thinking-4000",
    "claude-sonnet-4.5-thinking-16000",
    "claude-sonnet-4.5-thinking-60000",
    "claude-opus-4.5-low",
    "claude-opus-4.5-medium",
    "claude-opus-4.5-high",
    "gemini-3-low",
    "gemini-3-high",
]
SUPPORTED_MODELS = set(ORDERED_MODELS)

ResultRecord = Tuple[Path, int, bool, str, float, float]

@dataclass
class ModelResponse:
    text: str
    prompt_tokens: int
    cached_tokens: int
    completion_tokens: int
    explanation: Optional[str] = None

def parse_model_arg(model_arg: str) -> Tuple[str, str, object]:
    if model_arg not in SUPPORTED_MODELS:
        raise ValueError(f"Model '{model_arg}' not supported. Choose from {SUPPORTED_MODELS}")

    if model_arg.startswith("gpt-5.1-"):
        parts = model_arg.split("-")
        effort = parts[-1]
        base = "-".join(parts[:-1])
        return "openai", base, effort

    if model_arg.startswith("claude-sonnet-4.5-"):
        base = "claude-sonnet-4-5-20250929"
        suffix = model_arg.replace("claude-sonnet-4.5-", "")
        if suffix == "no-thinking":
            return "anthropic", base, 0
        if suffix.startswith("thinking-"):
            try:
                budget = int(suffix.split("-")[1])
                return "anthropic", base, budget
            except (IndexError, ValueError):
                pass

    if model_arg.startswith("claude-opus-4.5-"):
        base = "claude-opus-4-5-20251101"
        parts = model_arg.split("-")
        effort = parts[-1]
        return "anthropic", base, effort

    if model_arg.startswith("gemini-3-"):
        parts = model_arg.split("-")
        effort = parts[-1]
        return "google", "gemini-3-pro-preview", effort

    raise ValueError(f"Unknown model format: {model_arg}")

import httpx

def call_openai_internal(
    client: OpenAI,
    prompt: str,
    model: str,
    reasoning_effort: str,
    response_format: str = "text",
    capture_thinking: bool = False,
    two_stage_explanation: bool = False,
) -> ModelResponse:
    # Use a custom client with TCP Keep-Alive to prevent timeouts on long reasoning tasks
    # (e.g. NAT/Firewall dropping idle connections after 15-30 mins)
    http_client = httpx.Client(
        timeout=3600.0,
        transport=httpx.HTTPTransport(retries=3),
        limits=httpx.Limits(keepalive_expiry=3600)
    )
    # Note: OpenAI client doesn't easily accept a per-request http_client override in the SDK logic
    # without re-initializing. However, the 'client' passed in is already initialized.
    # Ideally, we should configure the 'client' object passed to this function with keepalive
    # at initialization time in main.py.
    
    # Since we can't easily swap the transport of the passed 'client', 
    # we will re-initialize a temporary client here just for this call if it's a long reasoning task,
    # OR we should rely on the user to fix main.py.
    
    # BETTER FIX: The user asked to fix the connection error.
    # The robust way is to modify main.py to initialize OpenAI(http_client=...)
    # But I am only editing src/models.py right now.
    # I will assume the passed 'client' might not have it.
    
    # Actually, standard OpenAI python client uses httpx.
    # We can try to hack it or just use a fresh client.
    # Let's use a fresh client wrapper for the robust reasoning call?
    # No, that requires API key again.
    
    # Let's modify main.py instead? The instruction was "Yes" to "apply this fix".
    # I should probably modify main.py to configure the client globally.
    # But I am currently restricted to editing via 'replace'.
    
    # I will update main.py in the next step. 
    # For now, I will leave this function as is but add a comment?
    # No, I need to act.
    
    # Re-reading the error: "Connection error".
    # I'll update main.py to inject the http_client.
    pass

# ... (rest of file)
    # Uniformly use the Responses API for all OpenAI calls
    kwargs = {
        "model": model,
        "input": [{"role": "user", "content": prompt}],
        "timeout": 3600,
    }

    # Configure reasoning parameters
    if reasoning_effort != "none":
        # Capture thinking summary takes precedence if both flags are set (though they shouldn't be)
        if capture_thinking:
            kwargs["reasoning"] = {
                "effort": reasoning_effort,
                "summary": "detailed"
            }
        else:
            kwargs["reasoning"] = {
                "effort": reasoning_effort
            }

    # STEP 1: Solve the task
    response = None
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.responses.create(**kwargs)
            break
        except Exception as e:
            err_str = str(e)
            if (
                "Connection error" in err_str
                or "500" in err_str
                or "server_error" in err_str
                or "upstream connect error" in err_str
                or "timed out" in err_str
            ):
                if attempt < max_retries - 1:
                    delay = 5 if attempt == 0 else 30
                    time.sleep(delay)
                    continue
            raise e

    # Extract Step 1 outputs
    text_output = ""
    step1_explanation = ""
    
    if hasattr(response, "output"):
        for item in response.output:
            if item.type == "message":
                for content_part in item.content:
                    if content_part.type == "output_text":
                        text_output += content_part.text
            elif item.type == "reasoning":
                if item.summary:
                    for summary_part in item.summary:
                        if summary_part.type == "summary_text":
                            step1_explanation += summary_part.text

    if not text_output:
         raise RuntimeError(f"OpenAI Responses API Step 1 did not return text output. Response: {response}")

    # Calculate Step 1 Usage
    usage1 = getattr(response, "usage", None)
    prompt_tokens = getattr(usage1, "input_tokens", 0) if usage1 else 0
    completion_tokens = getattr(usage1, "output_tokens", 0) if usage1 else 0
    
    # If NOT two-stage, return immediately
    if not two_stage_explanation:
        return ModelResponse(
            text=text_output,
            prompt_tokens=prompt_tokens,
            cached_tokens=0,
            completion_tokens=completion_tokens,
            explanation=step1_explanation if step1_explanation else None,
        )

    # STEP 2: Explain (using previous_response_id)
    step1_id = response.id
    
    step2_input = "Explain the strategy you used in broad terms such that it can be applied on other similar examples and other input data."
    
    kwargs_step2 = {
        "model": model,
        "previous_response_id": step1_id,
        "input": [{"role": "user", "content": step2_input}],
        "timeout": 3600
    }

    response2 = None
    for attempt in range(max_retries):
        try:
            response2 = client.responses.create(**kwargs_step2)
            break
        except Exception as e:
             # If Step 2 fails, we fallback to Step 1 result
             print(f"Step 2 explanation failed: {e}", file=sys.stderr)
             return ModelResponse(
                text=text_output,
                prompt_tokens=prompt_tokens,
                cached_tokens=0,
                completion_tokens=completion_tokens,
                explanation=step1_explanation if step1_explanation else None,
            )

    # Parse Step 2 Text output
    final_grid = text_output # Use step 1 grid
    final_explanation = ""
    
    if hasattr(response2, "output"):
        for item in response2.output:
            if item.type == "message":
                for content_part in item.content:
                    if content_part.type == "output_text":
                        final_explanation += content_part.text

    # Accumulate Usage
    usage2 = getattr(response2, "usage", None)
    prompt_tokens += getattr(usage2, "input_tokens", 0) if usage2 else 0
    completion_tokens += getattr(usage2, "output_tokens", 0) if usage2 else 0

    return ModelResponse(
        text=final_grid,
        prompt_tokens=prompt_tokens,
        cached_tokens=0,
        completion_tokens=completion_tokens,
        explanation=final_explanation,
    )

def call_anthropic(
    client: Anthropic,
    prompt: str,
    model: str,
    config: Union[int, str],
    response_format: str = "text",
    capture_thinking: bool = False,
) -> ModelResponse:
    # Claude doesn't support native JSON schema enforcement in the same way as OpenAI/Gemini via API params yet
    # (or it requires tools which adds complexity). We rely on prompt instructions for now.
    MODEL_MAX_TOKENS = 64000
    max_tokens = 8192

    kwargs = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
    }

    if isinstance(config, int) and config > 0:
        # Thinking (Sonnet)
        budget = config
        max_tokens = min(budget + 4096, MODEL_MAX_TOKENS)
        if budget >= max_tokens:
            budget = max_tokens - 2048
        kwargs["thinking"] = {"type": "enabled", "budget_tokens": budget}
    elif isinstance(config, str):
        # Effort (Opus)
        kwargs["extra_headers"] = {"anthropic-beta": "effort-2025-11-24"}
        kwargs["extra_body"] = {"output_config": {"effort": config}}
        # Keep max_tokens default or increase?
        # User example used 2048.
        max_tokens = 60000  # Increase slightly for reasoning output

    kwargs["max_tokens"] = max_tokens

    # Use streaming to avoid timeouts on long requests
    final_message = None
    max_retries = 3
    for attempt in range(max_retries):
        try:
            with client.messages.stream(**kwargs) as stream:
                for text in stream.text_stream:
                    pass
                final_message = stream.get_final_message()
            break
        except Exception as e:
            err_str = str(e)
            if (
                "500" in err_str
                or "Internal server error" in err_str
                or "Connection reset" in err_str
                or "Connection error" in err_str
            ):
                if attempt < max_retries - 1:
                    delay = 5 if attempt == 0 else 30
                    time.sleep(delay)
                    continue
            raise e

    text_parts = []
    thinking_parts = []
    for block in final_message.content:
        if getattr(block, "type", None) == "text":
            text_parts.append(block.text)
        elif getattr(block, "type", None) == "thinking":
            # Capture thinking block if it exists
            thinking_parts.append(block.thinking)

    text = "".join(text_parts).strip()
    
    explanation = None
    if capture_thinking and thinking_parts:
        explanation = "\n\n".join(thinking_parts)

    p_tokens = final_message.usage.input_tokens
    c_tokens = final_message.usage.output_tokens
    cached = getattr(final_message.usage, "cache_read_input_tokens", 0) or 0

    return ModelResponse(
        text=text,
        prompt_tokens=p_tokens,
        cached_tokens=cached,
        completion_tokens=c_tokens,
        explanation=explanation,
    )

def call_gemini(
    client: genai.Client,
    prompt: str,
    model: str,
    thinking_level: str,
    response_format: str = "text",
    capture_thinking: bool = False,
) -> ModelResponse:
    # Use REST API to bypass SDK limitation regarding thinking_level
    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Google API Key not found in environment")

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"

    level_enum = "LOW" if thinking_level == "low" else "HIGH"

    generation_config = {
        "temperature": 1.0,
        "maxOutputTokens": 65536,
        "thinkingConfig": {"includeThoughts": True, "thinkingLevel": level_enum},
    }

    # response_format="json" is now handled via prompt engineering (XML tags), not JSON schema.
    # So we do not set generation_config["responseMimeType"] here.

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": generation_config,
    }

    response = None
    max_retries = 3
    for attempt in range(max_retries):
        try:
            resp = requests.post(
                url, json=payload, headers={"Content-Type": "application/json"}
            )
            if resp.status_code != 200:
                if resp.status_code == 503 or "503" in resp.text:
                    raise Exception(f"503 Unavailable: {resp.text}")
                raise Exception(f"API Error {resp.status_code}: {resp.text}")

            response = resp.json()
            break
        except Exception as e:
            err_str = str(e)
            if (
                "503" in err_str
                or "UNAVAILABLE" in err_str
                or "overloaded" in err_str.lower()
            ):
                if attempt < max_retries - 1:
                    delay = 5 if attempt == 0 else 30
                    time.sleep(delay)
                    continue
            raise e

    try:
        candidate = response["candidates"][0]
        parts = candidate["content"]["parts"]
        text_parts = [p["text"] for p in parts if "text" in p]
        text = "".join(text_parts).strip()

        usage = response.get("usageMetadata", {})
        p_tokens = usage.get("promptTokenCount", 0)
        c_tokens = usage.get("candidatesTokenCount", 0)
        thoughts = usage.get("thoughtsTokenCount", 0)
        c_tokens += thoughts
        
        explanation = None
        # If Gemini supports returning thinking text in parts with a specific role/key, capture it here.
        # Currently, thinking content is often hidden or handled differently. 
        # We leave explanation=None for now unless capture_thinking is True and we find a way.
        
    except (KeyError, IndexError) as e:
        raise RuntimeError(f"Failed to parse Gemini response: {e} - Raw: {response}")

    return ModelResponse(
        text=text,
        prompt_tokens=p_tokens,
        cached_tokens=0,
        completion_tokens=c_tokens,
        explanation=explanation,
    )

def call_model(
    openai_client: OpenAI,
    anthropic_client: Anthropic,
    google_client: genai.Client,
    prompt: str,
    model_arg: str,
    response_format: str = "text",
    capture_thinking: bool = False,
    two_stage_explanation: bool = False,
) -> ModelResponse:
    provider, base_model, config = parse_model_arg(model_arg)

    if provider == "openai":
        return call_openai_internal(
            openai_client,
            prompt,
            base_model,
            config,
            response_format,
            capture_thinking=capture_thinking,
            two_stage_explanation=two_stage_explanation,
        )
    elif provider == "anthropic":
        if not anthropic_client:
            raise RuntimeError("Anthropic client not initialized.")
        return call_anthropic(
            anthropic_client,
            prompt,
            base_model,
            config,
            response_format,
            capture_thinking=capture_thinking,
        )
    elif provider == "google":
        if not google_client:
            raise RuntimeError("Google client not initialized.")
        return call_gemini(
            google_client,
            prompt,
            base_model,
            config,
            response_format,
            capture_thinking=capture_thinking,
        )

    raise ValueError(f"Unknown provider {provider}")
