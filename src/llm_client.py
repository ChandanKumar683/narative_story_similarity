"""
LLM client using OpenRouter (OpenAI-compatible API).
All models are accessed through a single OpenRouter API key.
"""

import os
import time
from openai import OpenAI, RateLimitError

from src.config import OPENROUTER_BASE_URL, OPENROUTER_API_KEY

MAX_RETRIES = 5
RETRY_BASE_DELAY = 10  # seconds; free-tier rate limit is ~8 req/min


def get_client() -> OpenAI:
    """Create an OpenRouter client."""
    api_key = os.environ.get("OPENROUTER_API_KEY", OPENROUTER_API_KEY)
    return OpenAI(
        base_url=OPENROUTER_BASE_URL,
        api_key=api_key,
    )


def call_llm(
    model_id: str,
    prompt: str,
    system_prompt: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = 64,
    **_kwargs,
) -> str:
    """Send a prompt to an LLM via OpenRouter and return the text response.

    Automatically retries on 429 rate-limit errors with exponential backoff.

    Args:
        model_id: OpenRouter model identifier (e.g. 'openai/gpt-4o').
        prompt: the user message.
        system_prompt: optional system-level instruction.
        temperature: sampling temperature (0.0 = deterministic).
        max_tokens: max tokens in the response.

    Returns:
        The model's text response (stripped).
    """
    client = get_client()

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    # Qwen 3 models use thinking mode by default, which consumes tokens.
    # Increase max_tokens to accommodate thinking + answer.
    is_thinking_model = "qwen3" in model_id.lower() or "qwen/qwen3" in model_id.lower()
    effective_max_tokens = max(max_tokens, 4096) if is_thinking_model else max_tokens

    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=model_id,
                messages=messages,
                temperature=temperature,
                max_tokens=effective_max_tokens,
            )
            content = response.choices[0].message.content.strip()

            # Strip <think>...</think> blocks from thinking models
            if is_thinking_model and "<think>" in content:
                import re
                content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()

            return content
        except RateLimitError as e:
            if attempt == MAX_RETRIES - 1:
                raise
            delay = RETRY_BASE_DELAY * (2 ** attempt)
            print(f"    Rate limited (attempt {attempt+1}/{MAX_RETRIES}), waiting {delay}s...")
            time.sleep(delay)
        except Exception as e:
            # Re-raise non-rate-limit errors (like 402 spend limit)
            raise
