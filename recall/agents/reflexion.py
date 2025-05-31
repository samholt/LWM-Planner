# Copyright © 2025 Samuel Holt. All rights reserved.
# No licence is granted to copy, use, modify, distribute, or create derivative
# works of this file in any form, except with explicit written permission from
# the copyright holder.
from __future__ import annotations
"""
LLM ReAct **and** Reflexion agents (+ verbose evaluator)
=======================================================

This single module now contains **both baselines**:

* `LLMReactAgent` – simple ReAct policy (unchanged in behaviour).
* `LLMReflexionAgent` – adds per‑episode *reflection*: after each episode it
  summarises the trajectory into a concise lesson which is prepended to future
  prompts.  A FIFO buffer keeps the **k = 20** most recent lessons.
* `evaluate_agent()` – unchanged call‑signature but now auto‑detects optional
  `record_transition()` and `reflect()` hooks, so it works seamlessly with
  either agent.

Usage
-----
```python
from text_frozen_lake_env import TextFrozenLakeEnv
from llm_agents import LLMReactAgent, LLMReflexionAgent, evaluate_agent

env = TextFrozenLakeEnv(size=4, hole_density=0.3, seed=0)
agent = LLMReflexionAgent()          # or LLMReactAgent()
metrics = evaluate_agent(env, agent, max_steps=5000, verbose=True)
```
"""

from collections import deque
from dataclasses import dataclass, field
from typing import List, Dict, Any, Deque, Tuple
import random
import logging
import matplotlib.pyplot as plt
from recall.agents.react import LLMReactAgent
import json

# ---------------------------------------------------------------------------
# External dependency (stub fallback for offline testing)
# ---------------------------------------------------------------------------
# try:
from recall.llm_utils import chat_completion  # type: ignore # noqa: F401
# except ImportError:  # pragma: no cover – offline stub
#     def chat_completion(messages: List[Dict[str, Any]], *, model: str = "gpt-4o", temperature: float = 0.3, max_tokens: int = 128) -> str:  # type: ignore
#         legal = {"up", "down", "left", "right"}
#         return random.choice(tuple(legal))

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
from recall.utils import tiny_logger

# ---------------------------------------------------------------------------
# Reflexion Agent – extends ReAct with per‑episode lesson synthesis
# ---------------------------------------------------------------------------
@dataclass
class LLMReflexionAgent(LLMReactAgent):
    """ReAct + Reflexion: learns from past trajectories by synthesising lessons."""

    lesson_buffer_len: int = 20  # keep last k lessons
    success_reward_threshold: float = 0.99  # reward > threshold ⇒ success

    _lessons: Deque[str] = field(init=False)
    _trajectory: List[Tuple[str, str, float, str]] = field(init=False)  # (obs, act, reward, next_obs)
    _episode_reward: float = field(init=False, default=0.0)

    def __post_init__(self) -> None:
        super().__post_init__()
        self._lessons = deque(maxlen=self.lesson_buffer_len)
        self._trajectory = []
        # Extend prompt to include lessons
        self.prompt_template = (
            "{env_description}\n\n"
            "Lessons learned:\n{lessons}\n\n"
            "You must use the ReAct pattern:\nThought: <reasoning>\nAction: <one of {actions}>\n\n"
            "Observation: {observation}\n"
            "Recent history (old→new):\n{history}\n\nThought:"
        )

    # ------------------------------------------------------------------
    # Overrides
    # ------------------------------------------------------------------
    def reset(self, observation: str) -> None:
        super().reset(observation)
        self._trajectory.clear()
        self._episode_reward = 0.0

    def act(self, observation: str) -> str:
        lessons_text = "\n".join(f"- {l}" for l in self._lessons) or "<none yet>"
        user_prompt = self.prompt_template.format(
            env_description=self.env_description,
            lessons=lessons_text,
            observation=observation,
            history="\n".join(self._history) or "<empty>",
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
    # Trajectory handling – evaluator will call these hooks
    # ------------------------------------------------------------------
    def record_transition(self, obs: str, act: str, reward: float, next_obs: str, done: bool) -> None:
        self._trajectory.append((obs, act, reward, next_obs))
        self._episode_reward += reward

    def reflect(self) -> None:
        if not self._trajectory:
            return

        outcome = "SUCCESS" if self._episode_reward >= self.success_reward_threshold else "FAILURE"
        trajectory_lines = [
            f"{i}. Obs: {o} | Act: {a} | R: {r:+.0f} | Next: {n}"
            for i, (o, a, r, n) in enumerate(self._trajectory, 1)
        ]
        reflection_prompt = (
            "You are a self‑reflective agent. Analyse the following trajectory and feedback.\n\n"
            f"Outcome: {outcome} (total reward = {self._episode_reward:+.1f})\n"
            "Trajectory:\n" + "\n".join(trajectory_lines) + "\n\n"
            "Based on the outcome, write ONE actionable lesson (≤ 20 words) starting with 'Lesson:'."
        )
        messages = [
            {"role": "system", "content": "You are a helpful agent generating self‑reflective lessons."},
            {"role": "user", "content": reflection_prompt},
        ]
        lesson = chat_completion(messages, model=self.model, temperature=self.temperature, max_tokens=self.max_tokens).strip()
        # Ensure prefix
        if not lesson.lower().startswith("lesson"):
            lesson = "Lesson: " + lesson
        tiny_logger.log(f"Reflexion lesson captured: {lesson}")
        self._lessons.appendleft(lesson)
        # Clear buffers for next episode
        self._trajectory.clear()
        self._episode_reward = 0.0


# # ---------------------------------------------------------------------------
# # Evaluation loop (works for both agents)
# # ---------------------------------------------------------------------------

# def evaluate_agent(env, agent: LLMReactAgent, *, max_steps: int = 10_000, verbose: bool = False) -> Dict[str, Any]:
#     """Run *agent* in *env* up to *max_steps* steps, with verbose logging.

#     *If* the agent implements optional hooks `record_transition()` and
#     `reflect()`, they are invoked automatically to enable Reflexion‑style
#     learning without altering the call‑site.
#     """
#     episode_returns: List[float] = []
#     steps_at_episode_end: List[int] = []
#     return_vs_steps: List[Tuple[int, float]] = []

#     total_steps = 0
#     cumulative_return = 0.0

#     obs = env.reset()
#     agent.reset()

#     episode_idx = 0
#     episode_return = 0.0
#     episode_step = 0

#     if verbose:
#         logger.info("== Evaluation start ==")
#         logger.info("Episode 0 reset → Observation: %s", obs)

#     while total_steps < max_steps:
#         action = agent.act(str(obs))
#         next_obs, reward, done, info = env.step(action)

#         # Hook: record transition if available
#         if hasattr(agent, "record_transition"):
#             agent.record_transition(str(obs), action, reward, str(next_obs), done)

#         obs = next_obs
#         total_steps += 1
#         episode_step += 1
#         episode_return += reward
#         cumulative_return += reward
#         return_vs_steps.append((total_steps, cumulative_return))

#         if verbose:
#             logger.info("Step %5d | Obs: %-30s | Act: %-5s | R: %+2.0f | Done: %s", total_steps, obs, action, reward, done)

#         if done:
#             episode_returns.append(episode_return)
#             steps_at_episode_end.append(total_steps)

#             # Hook: reflection after episode
#             if hasattr(agent, "reflect"):
#                 agent.reflect()

#             if verbose:
#                 logger.info("-- Episode %d end: return=%+.1f, steps=%d --\n", episode_idx, episode_return, episode_step)

#             # Reset episode
#             episode_idx += 1
#             episode_return = 0.0
#             episode_step = 0
#             obs = env.reset()
#             agent.reset()
#             if verbose:
#                 logger.info("Episode %d reset → Observation: %s", episode_idx, obs)

#     if verbose:
#         logger.info("== Evaluation complete: %d episodes, %d steps ==", episode_idx, total_steps)

#     return {
#         "episode_returns": episode_returns,
#         "steps_at_episode_end": steps_at_episode_end,
#         "return_vs_steps": return_vs_steps,
#         "total_steps": total_steps,
#         "episodes": len(episode_returns),
#     }


# # ---------------------------------------------------------------------------
# # Plot utility (unchanged)
# # ---------------------------------------------------------------------------

# def plot_metrics(metrics: Dict[str, Any]) -> None:
#     if not metrics.get("return_vs_steps"):
#         print("No metrics to plot – run evaluation first.")
#         return

#     steps, cum_returns = zip(*metrics["return_vs_steps"])
#     plt.figure()
#     plt.plot(steps, cum_returns)
#     plt.xlabel("Steps")
#     plt.ylabel("Cumulative return")
#     plt.title("Return vs. Steps")

#     plt.figure()
#     plt.plot(range(1, len(metrics["episode_returns"]) + 1), metrics["episode_returns"])
#     plt.xlabel("Episode")
#     plt.ylabel("Episode return")
#     plt.title("Return per Episode")
#     plt.show()


# # ---------------------------------------------------------------------------
# # Smoke test (uses stub chat_completion if offline)
# # ---------------------------------------------------------------------------
# if __name__ == "__main__":  # pragma: no cover
#     from text_frozen_lake_env import TextFrozenLakeEnv

#     env = TextFrozenLakeEnv(size=4, hole_density=0.3, seed=123)
#     agent = LLMReflexionAgent(history_len=5)
#     metrics = evaluate_agent(env, agent, max_steps=500, verbose=True)
#     print(f"Completed {metrics['episodes']} episodes in {metrics['total_steps']} steps.")
