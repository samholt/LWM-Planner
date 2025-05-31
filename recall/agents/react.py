# Copyright © 2025 Samuel Holt. All rights reserved.
# No licence is granted to copy, use, modify, distribute, or create derivative
# works of this file in any form, except with explicit written permission from
# the copyright holder.
from __future__ import annotations
"""
LLM‑powered ReAct agent **separated** from evaluation logic
==========================================================

This module now cleanly separates the **agent** (decision policy) from the
**evaluation loop** so you can reuse the same evaluator across many agents or
environments.

Components
----------
* **`LLMReactAgent`** – maintains a history buffer and returns an action for an
  observation via a ReAct‑style LLM prompt.  It exposes `reset()` and
  `act(observation: str) -> str`.
* **`evaluate_agent(env, agent, max_steps=10_000)`** – standard RL loop that
  handles env resets *and* calls `agent.reset()` at episode boundaries.  It
  accumulates metrics and returns a dictionary.
* **`plot_metrics(metrics)`** – quick visualisation of *return vs. steps* and
  *return per episode*.

The agent remains prompt‑template driven and supports plug‑and‑play
environments – just craft a new template describing the observation / action
format.
"""

from collections import deque
from dataclasses import dataclass, field
from typing import List, Dict, Any, Deque, Tuple, Optional
import random
import matplotlib.pyplot as plt
import re
import json

# ---------------------------------------------------------------------------
# External dependency supplied by the user runtime.  We fall back to a stub
# that chooses a random legal action so the module remains import‑safe.
# ---------------------------------------------------------------------------
# try:
from recall.llm_utils import chat_completion  # noqa: F401 (provided at runtime)
# except ImportError:  # Fallback stub
#     def chat_completion(messages: List[Dict[str, Any]], *, model: str = "gpt-4o", temperature: float = 0.3, max_tokens: int = 128) -> str:  # type: ignore
#         legal = {"up", "down", "left", "right"}
#         return random.choice(tuple(legal))


def _react_function_schema(allowed_actions: Tuple[str, ...]) -> List[Dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": "react_step",
                "description": "Return the agent's internal thought and chosen action for the current observation.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "thought": {"type": "string", "description": "Agent's private reasoning."},
                        "action": {"type": "string", "description": f"Action: <one of {allowed_actions}>."},
                    },
                    "required": ["thought", "action"],
                    "additionalProperties": False,
                },
                "strict": True,
            },
        },
        {
        "type": "function",
        "function": {
            "name": "react_generate",
            "description": "Return the agent's internal reasoning trace and the final answer for the current prompt.",
            "parameters": {
            "type": "object",
            "properties": {
                "thought": {
                "type": "string",
                "description": "Agent's private reasoning trace (not shown to the user)."
                },
                "answer": {
                "type": "string",
                "description": "The answer that will be returned to the user after reasoning."
                }
            },
            "required": ["thought", "answer"],
            "additionalProperties": False
            },
            "strict": True
        }
        }
    ]


# ---------------------------------------------------------------------------
# Agent definition (no env‑specific code, no evaluation loop inside!)
# ---------------------------------------------------------------------------
@dataclass
class LLMReactAgent:
    """ReAct agent that decides on an action given an observation string."""

    history_len: int = 20
    model: str = "gpt-4o"
    temperature: float = 0.3
    max_tokens: int = 16
    env_description: str | None = None  # can be overridden per environment
    allowed_actions: Tuple[str, ...] = ("up", "down", "left", "right")

    # Internal mutable state
    _history: Deque[str] = field(init=False)
    last_thought: str | None = field(init=False, default=None)

    # ------------------------------------------------------------------
    # Lifecycle helpers
    # ------------------------------------------------------------------
    def __post_init__(self) -> None:
        self._history = deque(maxlen=self.history_len)
        self._tools = _react_function_schema(self.allowed_actions)
        # Generic template for grid worlds like Frozen Lake.
        self.prompt_template = (
            "{env_description}\n\n"
            "You must use the ReAct pattern:\n"
            "Thought: <internal reasoning>\n"
            "Action: <one of {actions}>\n\n"
            "Respond with **exactly** that format—first Thought, then Action.\n\n"
            "Observation: {observation}\n"
            "History (old→new):\n{history}\n\n"
            "Thought:"
        )

    # def append_transition(self, act: str, next_obs: str) -> None:
    #     """
    #     Add the outcome of the last action to the rolling prompt history.
    #     Called *after* env.step().
    #     """
    #     self._history.append(f"Act: {act}")
    #     self._history.append(f"Obs: {next_obs}")

    def reset(self, observation: str) -> None:
        """Clear internal history at the start of a new episode."""
        self._history.clear()
        self.last_thought = None

    # ------------------------------------------------------------------
    # Core policy
    # ------------------------------------------------------------------
    def act(self, observation: str) -> str:
       
        """Return an action (string) for the given *observation*."""
        history_text = "\n".join(self._history) if self._history else "<empty>"
        user_prompt = self.prompt_template.format(
            env_description=self.env_description,
            observation=observation,
            history=history_text,
            actions=", ".join(self.allowed_actions),
        )
        messages = [
            {"role": "system", "content": "You are a expert agent and must call the function react_step."},
            {"role": "user", "content": user_prompt},
        ]
        reply = chat_completion(
            messages,
            model=self.model,
            tools=self._tools,
            tool_choice={
                "type": "function",
                "function": {"name": "react_step"}
            },
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        # ---------------- Parse function call ----------------
        if len(reply.tool_calls) != 1:
            raise ValueError(f"LLM did not return a function call: {reply}")
        args_str = reply.tool_calls[0].function.arguments
        try:
            args = json.loads(args_str)
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON from LLM: {args_str}")

        thought = args.get("thought", "").strip()
        action = args.get("action", "").lower().strip()
        # if action not in self.allowed_actions:
        #     raise ValueError(f"LLM produced invalid action '{action}'.")

        self.last_thought = thought
        self._history.append(f"Obs: {observation} → Act: {action}")
        return action

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _extract_action(self, text: str) -> str:
        """Parse first 'Action: <token>' occurrence."""
        match = re.search(r"Action:\s*(\w+)", text, re.IGNORECASE)
        if match:
            token = match.group(1).lower()
            if token in self.allowed_actions:
                return token
        raise ValueError(f"Invalid action '{text}' returned by LLM. Expected one of {self.allowed_actions}.")
        # logger.warning("Failed to parse action from LLM output, defaulting random. Output was: %s", text)
        # return random.choice(self.allowed_actions)

