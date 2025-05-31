# Copyright © 2025 Samuel Holt. All rights reserved.
# No licence is granted to copy, use, modify, distribute, or create derivative
# works of this file in any form, except with explicit written permission from
# the copyright holder.
from __future__ import annotations
# llm_utils.py  (top-level imports)
import argparse, time, random, statistics, textwrap, json, re
from typing import List, Tuple, Any, Dict, Optional
from dataclasses import dataclass
import functools, time, random
from openai import (
    RateLimitError,
    APIError,
    APIConnectionError,
    APITimeoutError,
)
import httpx

# ---------------------------------------------------------------------
# Decorator – wraps any function with robust retry-on-fail behaviour
# ---------------------------------------------------------------------
def with_retries(max_retries: int = 50,
                 base_delay: float = 0.5,     # seconds
                 max_delay: float = 30.0,     # clamp sleep
                retry_exceptions: tuple[type[Exception], ...] = (
                    RateLimitError,
                    APIError,
                    APIConnectionError,
                    APITimeoutError,       
                    httpx.ConnectTimeout,   
                ),
                 ):

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_err = None
            for attempt in range(max_retries):
                valid_excs = tuple(
                    exc for exc in retry_exceptions
                    if isinstance(exc, type) and issubclass(exc, BaseException)
                )
                try:
                    return func(*args, **kwargs)
                except valid_excs as err:
                    last_err = err
                    # exp. back-off with jitter: 0.5, 1, 2, 4… capped
                    delay = min(max_delay, base_delay * 2 ** attempt)
                    delay += random.uniform(0, 0.2 * delay)
                    print(f"[retry] {err.__class__.__name__} – "
                          f"attempt {attempt + 1}/{max_retries}, "
                          f"sleeping {delay:.2f}s")
                    time.sleep(delay)
            # all retries exhausted
            raise RuntimeError(f"chat_completion failed after "
                               f"{max_retries} retries") from last_err
        return wrapper
    return decorator


# ─────────────────────────────────────────────────────────────────────────────
# 0.  LLM helper – *fill in your endpoint / key here*
# ─────────────────────────────────────────────────────────────────────────────
from openai import AzureOpenAI  # pip install openai>=1.14.0

@dataclass
class AzureLLMConfig:
    deployment: str
    endpoint: str
    api_key: str
    api_version: str = "2025-03-01-preview"


LLM_REGISTRY: dict[str, AzureLLMConfig] = {
    # to fill in.
}

DEFAULT_LLM = "gpt-4o"
if DEFAULT_LLM is None:
    print("[WARN] LLM_REGISTRY is empty – please edit react_baseline_alfmini.py and add your Azure endpoint+key.")


def _get_client(alias: str = DEFAULT_LLM) -> tuple[AzureOpenAI, str]:
    cfg = LLM_REGISTRY[alias]
    return AzureOpenAI(azure_endpoint=cfg.endpoint, 
                       api_key=cfg.api_key,
                       api_version=cfg.api_version), cfg.deployment

@with_retries(max_retries=50)
def chat_completion(messages: list[dict[str, Any]], *, model: str = DEFAULT_LLM,
                    temperature: float | None = 0.3, max_tokens: int | None = None, tools=None, tool_choice=None, verbose=False):
    client, model = _get_client(model)
    kwargs: dict[str, Any] = dict(model=model, messages=messages)
    if tools is not None:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = tool_choice if tool_choice else "auto"

    is_reasoning = model.startswith(("o1", "o3", "o4"))
    if is_reasoning:
        kwargs["max_completion_tokens"] = max_tokens
    else:
        if temperature is not None:
            kwargs.update(temperature=temperature)
        if max_tokens is not None:
            kwargs.update(max_tokens=max_tokens)

    t0 = time.time()
    resp = client.chat.completions.create(**kwargs)
    latency = (time.time() - t0) * 1000
    if verbose:
        print(f"LLM call ({model}) latency: {latency:5.1f} ms")
    # return resp.choices[0].message.content.strip()
    msg = resp.choices[0].message        # ← keep the full message
    if tool_choice is not None:
        # If a tool was called, return the function call
        if hasattr(msg, "tool_calls") and len(msg.tool_calls) > 0:
            return msg
    else:
        # If no tool was called, return the content
        return msg.content

if __name__ == "__main__":
    import unittest
    from unittest import mock

    # stub error we can freely raise & catch
    class TransientError(Exception):
        """Simulated connectivity / rate-limit problem"""

    # patch time.sleep so tests are instant
    _sleep_patch = mock.patch("time.sleep", return_value=None)
    _sleep_patch.start()

    class RetryWrapperTests(unittest.TestCase):
        """Verify exponential-backoff retry logic with a controllable error."""

        def test_success_first_try(self):
            """No retries when call succeeds immediately."""
            calls = []

            @with_retries(max_retries=5,
                          base_delay=0.0,
                          retry_exceptions=(TransientError,))
            def good():
                calls.append(1)
                return "ok"

            self.assertEqual(good(), "ok")
            self.assertEqual(len(calls), 1)

        def test_retry_then_success(self):
            """Function fails twice, then succeeds."""
            calls = {"n": 0}

            @with_retries(max_retries=5,
                          base_delay=0.0,
                          retry_exceptions=(TransientError,))
            def flaky():
                calls["n"] += 1
                if calls["n"] < 3:
                    raise TransientError("simulated glitch")
                return "done"

            self.assertEqual(flaky(), "done")
            self.assertEqual(calls["n"], 3,
                             "expected 2 retries before success")

        def test_retry_exhausts_and_raises(self):
            """Give up after the configured limit."""
            calls = {"n": 0}

            @with_retries(max_retries=4,
                          base_delay=0.0,
                          retry_exceptions=(TransientError,))
            def always_bad():
                calls["n"] += 1
                raise TransientError("persistent failure")

            with self.assertRaises(RuntimeError):
                always_bad()
            self.assertEqual(calls["n"], 4,
                             "should try exactly max_retries times")

    try:
        unittest.main(exit=False, verbosity=2)
    finally:
        _sleep_patch.stop()