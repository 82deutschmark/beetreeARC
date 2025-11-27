import sys
from typing import Union, Optional

from anthropic import Anthropic

from src.types import ModelConfig, ModelResponse
from src.llm_utils import run_with_retry, orchestrate_two_stage
from src.logging import get_logger

logger = get_logger("providers.anthropic")

def call_anthropic(
    client: Anthropic,
    prompt: str,
    config: ModelConfig,
    return_strategy: bool = False,
    verbose: bool = False,
) -> ModelResponse:
    MODEL_MAX_TOKENS = 64000
    
    model = config.base_model
    cfg_val = config.config

    def _should_retry(e: Exception) -> bool:
        err_str = str(e)
        return (
            "500" in err_str
            or "Internal server error" in err_str
            or "Connection reset" in err_str
            or "Connection error" in err_str
            or "Server disconnected" in err_str
            or "RemoteProtocolError" in err_str
            or "connection closed" in err_str.lower()
            or "peer closed connection" in err_str.lower()
            or "incomplete chunked read" in err_str.lower()
        )

    kwargs = {
        "model": model,
        "max_tokens": 8192
    }
    if isinstance(cfg_val, int) and cfg_val > 0:
        budget = cfg_val
        max_tokens = min(budget + 4096, MODEL_MAX_TOKENS)
        if budget >= max_tokens: budget = max_tokens - 2048
        kwargs["thinking"] = {"type": "enabled", "budget_tokens": budget}
        kwargs["max_tokens"] = max_tokens

    def _solve(p: str) -> ModelResponse:
        kw = kwargs.copy()
        kw["messages"] = [{"role": "user", "content": p}]
        
        def _run():
            with client.messages.stream(**kw) as stream:
                for _ in stream.text_stream: pass
                return stream.get_final_message()
        
        final = run_with_retry(_run, _should_retry)
        
        text_parts = []
        for block in final.content:
            if getattr(block, "type", None) == "text":
                text_parts.append(block.text)
        
        resp = ModelResponse(
            text="".join(text_parts).strip(),
            prompt_tokens=final.usage.input_tokens,
            cached_tokens=getattr(final.usage, "cache_read_input_tokens", 0) or 0,
            completion_tokens=final.usage.output_tokens,
        )
        resp._raw_content = final.content # Store for context
        return resp

    def _explain(p: str, prev_resp: ModelResponse) -> Optional[ModelResponse]:
        try:
            kw = kwargs.copy()
            kw["messages"] = [
                {"role": "user", "content": prompt}, # Original Prompt
                {"role": "assistant", "content": prev_resp._raw_content},
                {"role": "user", "content": p}
            ]
            
            def _run():
                with client.messages.stream(**kw) as stream:
                    for _ in stream.text_stream: pass
                    return stream.get_final_message()

            final = run_with_retry(_run, _should_retry)
            
            text_parts = []
            for block in final.content:
                if getattr(block, "type", None) == "text":
                    text_parts.append(block.text)

            return ModelResponse(
                text="".join(text_parts).strip(),
                prompt_tokens=final.usage.input_tokens,
                cached_tokens=getattr(final.usage, "cache_read_input_tokens", 0) or 0,
                completion_tokens=final.usage.output_tokens,
            )
        except Exception as e:
            logger.error(f"Step 2 strategy extraction failed: {e}")
            return None

    return orchestrate_two_stage(_solve, _explain, prompt, return_strategy, verbose)