# Copyright © 2025 Samuel Holt. All rights reserved.
# No licence is granted to copy, use, modify, distribute, or create derivative
# works of this file in any form, except with explicit written permission from
# the copyright holder.
from __future__ import annotations
"""
Pure‑random agent (baseline)
============================

A **drop‑in replacement** for `LLMReactAgent` when you need the simplest
possible baseline: each call to `act()` samples uniformly from the environment
action space.  It keeps the same public interface (`reset`, `act`) so the
existing evaluator can treat it identically to the LLM‑based agents.

Example
-------
```python
from text_frozen_lake_env import TextFrozenLakeEnv
from random_agent import RandomAgent

env = TextFrozenLakeEnv(size=4, hole_density=0.3, seed=0)
agent = RandomAgent()
metrics = evaluate_agent(env, agent, max_steps=1_000)
print(metrics)
```
"""

import random
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Tuple

__all__ = ["RandomAgent"]


@dataclass
class RandomAgent:
    """Agent that selects an action uniformly at random each step."""

    allowed_actions: Tuple[str, ...] = ("up", "down", "left", "right")
    history_len: int = 20

    _history: Deque[str] = field(init=False)

    # ------------------------------------------------------------------
    # Lifecycle helpers
    # ------------------------------------------------------------------
    def __post_init__(self) -> None:
        self._history = deque(maxlen=self.history_len)

    def reset(self, observation: str) -> None:  # noqa: D401 (simple reset)
        """Clear internal history at the start of a new episode."""
        self._history.clear()

    # ------------------------------------------------------------------
    # Core policy
    # ------------------------------------------------------------------
    def act(self, observation: str) -> str:  # noqa: D401 – same signature
        """Return a uniformly‑sampled legal action."""
        action = random.choice(self.allowed_actions)
        self._history.append(f"Obs: {observation} → Act: {action}")
        return action
