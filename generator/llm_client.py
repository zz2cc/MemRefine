"""LLM API client supporting OpenAI-compatible endpoints (GPT-4, DeepSeek, etc.)."""

import time
import openai
from typing import List, Dict, Optional
from config import config


class LLMClient:
    """Thin wrapper around OpenAI-compatible chat completion API.

    Supports any provider with a compatible endpoint (OpenAI, DeepSeek, local vLLM, etc.).
    """

    def __init__(
        self,
        api_key: str = None,
        base_url: str = None,
        model: str = None,
        max_retries: int = 3,
    ):
        api_key = api_key or config.llm_api_key
        base_url = base_url or config.llm_base_url
        self.model = model or config.generator_model
        self.max_retries = max_retries

        self.client = openai.OpenAI(api_key=api_key, base_url=base_url)

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = None,
        max_tokens: int = None,
        model: str = None,
    ) -> str:
        """Send a chat completion request and return the response text."""
        temperature = temperature if temperature is not None else config.llm_temperature
        max_tokens = max_tokens or config.llm_max_tokens
        model = model or self.model

        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                return response.choices[0].message.content.strip()
            except openai.RateLimitError:
                wait = 2 ** attempt * 5
                print(f"    [LLM] Rate limited, retrying in {wait}s ...")
                time.sleep(wait)
            except openai.APITimeoutError:
                wait = 2 ** attempt * 3
                print(f"    [LLM] Timeout, retrying in {wait}s ...")
                time.sleep(wait)
            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise
                print(f"    [LLM] Error: {e}, retrying ...")
                time.sleep(2)

        raise RuntimeError("LLM call failed after all retries")

    def generate_batch(
        self,
        messages_list: List[List[Dict[str, str]]],
        temperature: float = None,
        model: str = None,
    ) -> List[str]:
        """Generate responses for a batch of message sets (sequential)."""
        return [self.chat(msgs, temperature=temperature, model=model) for msgs in messages_list]

    def generate_with_temperatures(
        self,
        messages: List[Dict[str, str]],
        temperatures: List[float],
        model: str = None,
    ) -> List[str]:
        """Generate multiple candidates at different temperatures — PARALLEL."""
        from concurrent.futures import ThreadPoolExecutor, as_completed
        results = {}
        with ThreadPoolExecutor(max_workers=len(temperatures)) as ex:
            futures = {
                ex.submit(self.chat, messages, t, None, model): i
                for i, t in enumerate(temperatures)
            }
            for f in as_completed(futures):
                idx = futures[f]
                results[idx] = f.result()
        return [results[i] for i in sorted(results)]
