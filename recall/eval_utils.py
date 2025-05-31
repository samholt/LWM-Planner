from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import List, Dict, Any, Deque, Tuple, Optional
import random
import matplotlib.pyplot as plt
from recall.utils import logger


# ---------------------------------------------------------------------------
# Standard RL evaluation loop (separate from agent)
# ---------------------------------------------------------------------------

def evaluate_agent(env, agent, *, max_steps: int = 10_000, verbose: bool = False) -> Dict[str, Any]:
    """Run *agent* in *env* for up to *max_steps* steps, collecting metrics.

    `verbose=True` prints:
    * Each step: step number, observation, action, reward, done flag.
    * Episode summary when a terminal state is reached.
    """
    episode_returns: List[float] = []
    steps_at_episode_end: List[int] = []
    return_vs_steps: List[Tuple[int, float]] = []

    total_steps = 0
    cumulative_return = 0.0

    obs = env.reset()
    if hasattr(env, "render"):
        env.render()

    if hasattr(agent, "online_url"): #detecting occam agent
        agent.reset(env)
    else:
        agent.reset(obs)


    episode_idx = 0
    episode_return = 0.0
    episode_step = 0

    if verbose:
        logger.log("== Evaluation start ==")
        logger.log(f"Episode 0 reset → Observation: {obs}")

    while total_steps < max_steps:
        action = agent.act(str(obs))
        next_obs, reward, done, info = env.step(action)
        # if hasattr(agent, "append_transition"):
        #     agent.append_transition(action, str(next_obs))
        # Hook: record transition if available
        if hasattr(agent, "record_transition"):
            agent.record_transition(str(obs), action, reward, str(next_obs), done)
            

        obs = next_obs
        total_steps += 1
        episode_step += 1
        episode_return += reward
        cumulative_return += reward
        return_vs_steps.append((total_steps, cumulative_return))

        if verbose:
            logger.log(f"[E:{episode_idx}] Step {total_steps} | Obs: {obs[:100]} | Act: {action} | R: {reward} | Done: {done}")

        if done:
            episode_returns.append(episode_return)
            steps_at_episode_end.append(total_steps)

            # Hook: reflection after episode
            if hasattr(agent, "reflect"):
                agent.reflect()

            if verbose:
                logger.log(f"-- Episode {episode_idx} end: return={episode_return}, steps={episode_step} --\n")

            # Reset for next episode
            episode_idx += 1
            episode_return = 0.0
            episode_step = 0
            obs = env.reset()
            
            
            if hasattr(agent, "online_url"): #detecting occam agent
                agent.reset(env)
            else:
                agent.reset(obs)
                
            if verbose:
                logger.log(f"Episode {episode_idx} reset → Observation: {obs}")

    if verbose:
        logger.log(f"== Evaluation complete: {episode_idx} episodes, {total_steps} steps ==")

    return {
        "episode_returns": episode_returns,
        "steps_at_episode_end": steps_at_episode_end,
        "return_vs_steps": return_vs_steps,
        "total_steps": total_steps,
        "episodes": len(episode_returns),
    }



# ---------------------------------------------------------------------------
# Visualisation helper (optional)
# ---------------------------------------------------------------------------

def plot_metrics(metrics: Dict[str, Any]) -> None:
    """Plot return‑vs‑steps and return‑per‑episode curves."""
    if not metrics.get("return_vs_steps"):
        print("No metrics to plot – run evaluation first.")
        return

    steps, cum_returns = zip(*metrics["return_vs_steps"])
    plt.figure()
    plt.plot(steps, cum_returns)
    plt.xlabel("Steps")
    plt.ylabel("Cumulative return")
    plt.title("Return vs. Steps")

    plt.figure()
    plt.plot(range(1, len(metrics["episode_returns"]) + 1), metrics["episode_returns"])
    plt.xlabel("Episode")
    plt.ylabel("Episode return")
    plt.title("Return per Episode")
    plt.show()