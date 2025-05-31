# Copyright © 2025 Samuel Holt. All rights reserved.
# No licence is granted to copy, use, modify, distribute, or create derivative
# works of this file in any form, except with explicit written permission from
# the copyright holder.
"""
experiment_cfg.py

A lightweight configuration object plus a handful of helpers for
serialising / de-serialising the object to JSON, string, and dict
representations.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional


@dataclass(slots=True)
class ExperimentCfg:  # note: class name shortened
    # ── basic run settings ───────────────────────────────────────────────
    workdir: Optional[str] = None                      # new in v2
    env_name: str = "frozen_lake_synthetic"
    method: str = "random"
    experiment: str = "baselines"
    seed: int = 0
    total_timesteps: int = 150

    # ── verbosity / debug flags ─────────────────────────────────────────
    debug: bool = False
    verbose: bool = True

    # ── agent hyper-parameters ──────────────────────────────────────────
    react_history_len: int = 51
    fact_buffer_len: int = 200
    reflexion_lesson_buffer_len: int = 5        # keep last k lessons
    world_model_search_depth: int = 3
    world_model_branch_factor: int = 4
    world_model_discount: float = 0.99

    # ── LLM backend settings ────────────────────────────────────────────
    llm_model: str = "gpt-4o-sh-1"
    llm_temperature: float = 0.0
    llm_max_tokens: int = 8512

    # -------------------------------------------------------------------
    # Serialisation helpers
    # -------------------------------------------------------------------
    def to_json(self, **dumps_kwargs) -> str:
        """
        Serialise the config to a JSON string.
        Any extra keyword args are forwarded to json.dumps().
        """
        return json.dumps(asdict(self), **dumps_kwargs)

    @classmethod
    def from_json(cls, payload: str | bytes, **loads_kwargs) -> "ExperimentCfg":
        """
        Restore an ExperimentCfg from a JSON string/bytes.
        Extra keyword args are forwarded to json.loads().
        """
        raw: Dict[str, Any] = json.loads(payload, **loads_kwargs)
        return cls(**raw)

    # -------------------------------------------------------------------
    # String representation round-trip
    # -------------------------------------------------------------------
    def __str__(self) -> str:  # voila: replaces to_string()
        # deterministic ordering keeps tests happy without relying on dataclasses'
        # internal field order
        items = (f"{k}={v!r}" for k, v in sorted(asdict(self).items()))
        return ", ".join(items)

    @classmethod
    def parse(cls, s: str) -> "ExperimentCfg":  # replaces from_string()
        """
        Reconstruct an ExperimentCfg from the __str__ representation.
        Very forgiving: ignores whitespace and allows quote style variations.
        """
        kv: Dict[str, Any] = {}
        for pair in filter(None, (seg.strip() for seg in s.split(","))):
            k, v = map(str.strip, pair.split("=", 1))
            # naive literal eval – fine here because we only emit repr-style values
            kv[k] = json.loads(v.replace("'", '"')) if v.startswith(("'", '"')) else eval(v)  # noqa: S307
        return cls(**kv)

    # -------------------------------------------------------------------
    # Convenience views
    # -------------------------------------------------------------------
    def as_upper_dict(self) -> Dict[str, Any]:
        """Return a *new* dict with keys upper-cased for frameworks that expect that."""
        return {k.upper(): v for k, v in asdict(self).items()}


# -----------------------------------------------------------------------
# Stand-alone helper (kept for API parity, delegates to instance method)
# -----------------------------------------------------------------------
def config_to_upper_dict(cfg: ExperimentCfg) -> Dict[str, Any]:
    """Functional wrapper around ExperimentCfg.as_upper_dict()."""
    return cfg.as_upper_dict()