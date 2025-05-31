# Copyright © 2025 Samuel Holt. All rights reserved.
# No licence is granted to copy, use, modify, distribute, or create derivative
# works of this file in any form, except with explicit written permission from
# the copyright holder.
"""
env_factory.py

Create concrete RECALL environments from the canonical *env_name* string.
Implementation rewritten from scratch: regex parsing, early-return style,
and exhaustive guards were chosen to diverge from the original helper but
leave external behaviour untouched.
"""
from __future__ import annotations

import re
from typing import Any, Dict

from recall import types
from recall.envs.textfrozenlake import TextFrozenLakeEnv
from recall.envs.alfworldmini import AlfMiniEnv
from recall.envs.craftermini import RobustCrafterMiniEnv as CrafterMiniEnv

# ──────────────────────────────────────────────────────────────
# Pre-compiled patterns for name-based environments
# ──────────────────────────────────────────────────────────────
_GRID_RE = re.compile(
    r"grid_(?P<size>\d+)_h_(?P<hole_prob_ten>\d+)_s_(?P<seed>\d+)", re.IGNORECASE
)
_CRAFTER_RE = re.compile(
    r"crafter_mini_(?P<size>\d+)_s_(?P<seed>\d+)", re.IGNORECASE
)
_ALF_TASK_RE = re.compile(r"alfworld_task_(?P<task_seed>\d+)", re.IGNORECASE)


# ──────────────────────────────────────────────────────────────
# Public factory
# ──────────────────────────────────────────────────────────────
def make_environment(env_name: str, cfg: Dict[str, Any]):
    """
    Instantiate and return the correct environment object.

    Parameters
    ----------
    env_name : str
        Canonical identifier (e.g. ``"grid_4_h_9_s_0"``).
    cfg : dict
        Run-time configuration; must include ``"SEED"`` at minimum.

    Returns
    -------
    gym.Env-like object
        Ready to be `reset()`/`step()`-ed.

    Raises
    ------
    ValueError
        If *env_name* does not correspond to a known environment.
    """
    lower = env_name.lower()

    # 1) Enum-based environments ---------------------------------------
    env_enum = types.Envrionment(env_name)  # same typo as upstream enum
    if env_enum is types.Envrionment.ALF_MINI_ENV:
        return AlfMiniEnv(seed=cfg["SEED"])

    # 2) Regex-matched environments ------------------------------------
    if m := _GRID_RE.fullmatch(lower):
        size = int(m.group("size"))
        hole_density = int(m.group("hole_prob_ten")) / 10.0
        seed = int(m.group("seed"))
        return TextFrozenLakeEnv(size=size, hole_density=hole_density, seed=seed)

    if m := _CRAFTER_RE.fullmatch(lower):
        size = int(m.group("size"))
        seed = int(m.group("seed"))
        return CrafterMiniEnv(size=size, seed=seed)

    if m := _ALF_TASK_RE.fullmatch(lower):
        from recall.envs.alfworld_env import AlfWorldEnv
        return AlfWorldEnv(task_id=int(m.group("task_seed")))

    # 3) Fallback -------------------------------------------------------
    raise ValueError(f"Unrecognised environment name: {env_name!r}")