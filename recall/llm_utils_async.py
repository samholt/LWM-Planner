# Copyright © 2025 Samuel Holt. All rights reserved.
# No licence is granted to copy, use, modify, distribute, or create derivative
# works of this file in any form, except with explicit written permission from
# the copyright holder.
"""Async counterparts to *llm_utils.py* with explicit client‑reuse hooks.

Changes in this revision
------------------------
* **Hybrid back‑off** – retry delay grows *exponentially* until it would exceed
  10 s, then switches to *linear* (+2 s per attempt) up to `max_delay`.  Jitter
  is still applied (±20 %).
* All other public APIs are unchanged (`get_async_client`,
  `async_chat_completion`, `async_chat_completion_batch`).

Example
^^^^^^^
```python
client, dep = get_async_client()
async with client:
    # transparently uses the new back‑off schedule
    txt = await async_chat_completion(msgs, client=client, deployment=dep)
```
"""
from __future__ import annotations

import asyncio, time, random, functools, math
from typing import Any, Tuple

import httpx
from openai import (
    RateLimitError,
    APIError,
    APIConnectionError,
    APITimeoutError,
)
from openai import AsyncAzureOpenAI  # pip install openai>=1.14.0

# ── import shared registry ────────────────────────────────────────────────────
from recall.llm_utils import LLM_REGISTRY, AzureLLMConfig, DEFAULT_LLM  # type: ignore

################################################################################
# 1.  Async retry decorator                                                      #
################################################################################

_T_RETRY_EXC = (
    RateLimitError,
    APIError,
    APIConnectionError,
    APITimeoutError,
    httpx.ConnectTimeout,
)


def _hybrid_delay(attempt: int, base: float, max_delay: float) -> float:
    """Return delay (seconds) following *exp → linear* hybrid schedule.

    * Exponential:  ``base * 2**attempt``  **until** that value exceeds 10 s.
    * Linear after threshold: 10 s + *(attempt – N)* × 2 s.
    * Always clamped to *max_delay*.
    """

    exp = base * 2 ** attempt
    if exp <= 10:
        return min(max_delay, exp)

    # first attempt where exp > 10 determines the pivot
    pivot = math.ceil(math.log2(10 / base))  # N
    linear = 10 + (attempt - pivot) * 2.0     # +2 s each further retry
    return min(max_delay, linear)


def with_async_retries(
    max_retries: int = 50,
    *,
    base_delay: float = 0.5,
    max_delay: float = 30.0,
    retry_exceptions: tuple[type[BaseException], ...] = _T_RETRY_EXC,
):
    """Decorate **async** functions with robust retry logic.

    Delay schedule: exponential back‑off up to 10 s, then linear (+2 s) capped
    at *max_delay*. A ±20 % jitter is applied to each wait.
    """

    def decorator(func):  # type: ignore[missing-return-type]
        if not asyncio.iscoroutinefunction(func):
            raise TypeError("@with_async_retries can only wrap async coroutines")

        @functools.wraps(func)
        async def wrapper(*args, **kwargs):  # type: ignore[no-any-return]
            last_err: BaseException | None = None
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except retry_exceptions as err:  # type: ignore[arg-type]
                    last_err = err
                    delay = _hybrid_delay(attempt, base_delay, max_delay)
                    # ±20 % jitter
                    jitter = random.uniform(-0.2 * delay, 0.2 * delay)
                    delay = max(0, delay + jitter)
                    print(
                        f"[async‑retry] {err.__class__.__name__} – attempt {attempt + 1}/{max_retries}, sleeping {delay:.2f}s"
                    )
                    await asyncio.sleep(delay)
            raise RuntimeError(
                f"chat_completion failed after {max_retries} retries"
            ) from last_err

        return wrapper

    return decorator

################################################################################
# 2.  Async client factory                                                       #
################################################################################


def get_async_client(alias: str | None = None) -> Tuple[AsyncAzureOpenAI, str]:
    if alias is None:
        alias = DEFAULT_LLM
    if alias is None:
        raise RuntimeError("DEFAULT_LLM is not set – edit llm_utils.py")

    cfg = LLM_REGISTRY[alias]
    return (
        AsyncAzureOpenAI(
            azure_endpoint=cfg.endpoint,
            api_key=cfg.api_key,
            api_version=cfg.api_version,
        ),
        cfg.deployment,
    )

_get_async_client = get_async_client  # alias for back‑compat

################################################################################
# 3.  Single completion helper                                                   #
################################################################################


@with_async_retries()
async def async_chat_completion(
    messages: list[dict[str, Any]],
    *,
    client: AsyncAzureOpenAI | None = None,
    deployment: str | None = None,
    model: str | None = None,
    temperature: float | None = 0.3,
    max_tokens: int | None = None,
    tools: Any = None,
    tool_choice: Any = None,
    verbose: bool = False,
):
    owns_client = client is None
    if owns_client:
        client, deployment = get_async_client(model)
    elif deployment is None:
        raise ValueError("When reusing a client you must also pass `deployment`.")

    kwargs: dict[str, Any] = dict(model=deployment, messages=messages)
    if tools is not None:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = tool_choice if tool_choice else "auto"

    is_reasoning = deployment.startswith(("o1", "o3", "o4"))
    if is_reasoning:
        kwargs["max_completion_tokens"] = max_tokens
    else:
        if temperature is not None:
            kwargs["temperature"] = temperature
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens

    t0 = time.time()
    if owns_client:
        async with client:
            resp = await client.chat.completions.create(**kwargs)  # type: ignore[arg-type]
    else:
        resp = await client.chat.completions.create(**kwargs)  # type: ignore[arg-type]
    latency = (time.time() - t0) * 1000
    if verbose:
        print(f"[async] LLM call ({deployment}) latency: {latency:5.1f} ms")

    msg = resp.choices[0].message
    if tool_choice is not None and getattr(msg, "tool_calls", None):
        return msg  # type: ignore[return-value]
    return msg.content  # type: ignore[return-value]

################################################################################
# 4.  Batch helper                                                               #
################################################################################


@with_async_retries()
async def async_chat_completion_batch(
    messages: list[dict[str, Any]],
    *,
    n: int,
    client: AsyncAzureOpenAI | None = None,
    deployment: str | None = None,
    model: str | None = None,
    temperature: float | None = 0.3,
    max_tokens: int | None = None,
    verbose: bool = False,
) -> list[str]:
    owns_client = client is None
    if owns_client:
        client, deployment = get_async_client(model)
    elif deployment is None:
        raise ValueError("`deployment` is required when supplying a client")

    cache: dict[str, bool] = getattr(async_chat_completion_batch, "_support", {})
    if deployment not in cache:
        cache[deployment] = not deployment.startswith(("o1", "o3", "o4"))
        setattr(async_chat_completion_batch, "_support", cache)

    async def _one() -> str:
        return await async_chat_completion(
            messages,
            client=client,
            deployment=deployment,
            temperature=temperature,
            max_tokens=max_tokens,
            verbose=verbose,
        )

    if not cache[deployment]:  # → parallel fan‑out
        return await asyncio.gather(*(_one() for _ in range(n)))

    kwargs: dict[str, Any] = dict(model=deployment, messages=messages, n=n)
    if temperature is not None:
        kwargs["temperature"] = temperature
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens

    t0 = time.time()
    if owns_client:
        async with client:
            resp = await client.chat.completions.create(**kwargs)  # type: ignore[arg-type]
    else:
        resp = await client.chat.completions.create(**kwargs)  # type: ignore[arg-type]
    latency = (time.time() - t0) * 1000
    if verbose:
        print(f"[async] LLM *batch* call ({deployment}) n={n} latency: {latency:5.1f} ms")

    return [c.message.content for c in resp.choices]

################################################################################
# 5.  Self‑test                                                                  #
################################################################################

if __name__ == "__main__":
    import pytest

    class _Err(Exception):
        pass

    async def _flaky(counter: list[int]):
        counter[0] += 1
        if counter[0] < 4:
            raise _Err
        return "done"

    async def _noop(_: float):
        return None

    monkey = pytest.MonkeyPatch()
    monkey.setattr(asyncio, "sleep", _noop, raising=True)

    @with_async_retries(max_retries=6, retry_exceptions=(_Err,))
    async def wrapped(counter):
        return await _flaky(counter)

    async def _run():
        c = [0]
        assert await wrapped(c) == "done" and c[0] == 4
        print("Hybrid back‑off test ✓")

    asyncio.run(_run())
    monkey.undo()
