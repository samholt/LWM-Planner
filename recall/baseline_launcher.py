# Copyright © 2025 Samuel Holt. All rights reserved.
# No licence is granted to copy, use, modify, distribute, or create derivative
# works of this file in any form, except with explicit written permission from
# the copyright holder.
"""
baseline_launcher.py

A thin orchestration layer for running *baseline* evaluations in the
RECALL project.  It wraps environment creation, agent instantiation and
metric collection, while deliberately diverging in style from the
original internal implementation.
"""
from __future__ import annotations

from typing import Any, Dict

from recall import agents, envs, eval_utils, types
from recall.utils import tiny_logger

# ──────────────────────────────────────────────────────────────────────────
# Private helpers
# ──────────────────────────────────────────────────────────────────────────
def _make_env(cfg: Dict[str, Any]):
    """Factory wrapper around `recall.envs.get_environment`."""
    return envs.get_environment(env_name=cfg["ENV_NAME"], config=cfg)


def _seeded_stub(cfg: Dict[str, Any], seed: int) -> Dict[str, Any]:
    """Return a pre-filled result dictionary."""
    return {
        "env_name": cfg["ENV_NAME"],
        "method": cfg["METHOD"].name,
        "seed": seed,
        "config": cfg,
    }


def _evaluate(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Spin up env+agent, run roll-outs, collect metrics."""
    env = _make_env(cfg)
    agent = agents.get_agent(cfg["METHOD"], cfg, env=env)
    horizon = cfg["TOTAL_TIMESTEPS"]

    outcome = _seeded_stub(cfg, cfg["SEED"])
    outcome.update(
        eval_utils.evaluate_agent(
            env,
            agent,
            max_steps=horizon,
            verbose=cfg.get("VERBOSE", False),
        )
    )

    print(
        f"[{outcome['env_name']}] completed "
        f"{outcome['episodes']} episodes / {outcome['total_steps']} steps"
    )
    return outcome


# ──────────────────────────────────────────────────────────────────────────
# Public entry-points
# ──────────────────────────────────────────────────────────────────────────
def launch(
    *,
    env_name: str,
    seed: int,
    method: str,
    experiment: str,
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Generic experiment launcher.

    Parameters
    ----------
    env_name : str
        Registered environment identifier.
    seed : int
        Random seed.
    method : str
        Agent family (must map to `types.Method`).
    experiment : str
        Experiment type (must map to `types.Experiment`).
    cfg : dict
        Free-form configuration blob passed through to helpers.

    Returns
    -------
    dict
        Collected metrics plus bookkeeping fields.
    """
    exp_kind = types.Experiment(experiment)

    if exp_kind in {types.Experiment.BASELINE, types.Experiment.BASELINES}:
        results = _run_baseline(env_name, seed, method, cfg)
        results["experiment"] = exp_kind.name
        return results

    raise ValueError(f"Unsupported experiment type: {experiment}")


def _run_baseline(
    env_name: str,
    seed: int,
    method: str,
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """Baseline evaluation: pure inference, no learning loops."""
    cfg = dict(cfg)  # copy to avoid mutating the caller’s dict
    cfg.update({"ENV_NAME": env_name, "METHOD": types.Method(method), "SEED": seed})

    tiny_logger.log(
        f"[{env_name}] baseline run | method={cfg['METHOD'].name} | seed={seed}"
    )

    return _evaluate(cfg)