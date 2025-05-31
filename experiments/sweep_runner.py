# Copyright © 2025 Samuel Holt. All rights reserved.
# No licence is granted to copy, use, modify, distribute, or create derivative
# works of this file in any form, except with explicit written permission from
# the copyright holder.
"""
sweep_runner.py

Kick off a set of baseline RECALL experiments over a Cartesian product of
hyper-parameters and pickle intermediate results.  The layout, helpers
and naming diverge intentionally from the internal reference version.
"""
from __future__ import annotations

import copy
import itertools
import pickle
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Sequence

from recall import baseline_launcher
from recall import config as cfg_lib
from recall.utils import tiny_logger as tlog

# ──────────────────────────────────────────────────────────────────────────
# Default template config ─ user can override by supplying kwargs
# ──────────────────────────────────────────────────────────────────────────
BASE_CFG = cfg_lib.ExperimentCfg()  # ← renamed dataclass in earlier rewrite

# ¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤
def launch_sweep(argv: Sequence[str] = ()) -> None:
    """Entry-point mimicking `main()` in the legacy script."""
    del argv  # still unused – placeholder for absl/app parity

    # ------------------------------------------------------------------ #
    # 1. Define the hyper-grid                                             #
    # ------------------------------------------------------------------ #
    grid: Dict[str, list[Any]] = {
        "env_name": ["grid_4_h_9_s_0"],
        "method": ["react", "reflexion", "world_model_facts_parallel_d_1"],
        "experiment": ["baselines"],
        "seed": [0],
    }

    # ------------------------------------------------------------------ #
    # 2. Prepare output directory                                         #
    # ------------------------------------------------------------------ #
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    tag = f"{'-'.join(grid['env_name'])}__{'-'.join(grid['method'])}"
    tag = tag if len(tag) <= 100 else str(hash(tag))[:10]

    out_dir = Path("./logs") / f"{timestamp}_{tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    pickle_path = out_dir / "results.pickle"

    tlog.stamp(f"Logging to directory: {out_dir}")

    # ------------------------------------------------------------------ #
    # 3. Iterate over the Cartesian product                               #
    # ------------------------------------------------------------------ #
    all_runs: list[dict[str, Any]] = []
    for combo in itertools.product(*grid.values()):
        sweep_update = dict(zip(grid.keys(), combo))

        # Deep-copy base template, then update with sweep overrides
        cfg = copy.deepcopy(BASE_CFG)
        for k, v in sweep_update.items():
            setattr(cfg, k, v)

        # Prepare launch bundle
        exp_cfg_dict = cfg_lib.config_to_upper_dict(cfg)
        job = {
            "env_name": cfg.env_name,
            "seed": cfg.seed,
            "method": cfg.method,
            "experiment": cfg.experiment,
            "config": exp_cfg_dict,
        }

        tlog.stamp(f"▶ Running job: {job}")

        try:
            outcome = baseline_launcher.launch(**job)  # renamed entry-point
        except Exception as exc:  # noqa: BLE001
            tlog.stamp(f"✖ Failure: {exc}")
            traceback.print_exc(file=sys.stderr)
            continue  # proceed to next combo

        all_runs.append({"config": job, "result": outcome})

        with pickle_path.open("wb") as fh:
            pickle.dump(all_runs, fh)
        tlog.stamp(f"✓ Saved interim results → {pickle_path}")

    tlog.stamp("🎉 All sweeps finished!")


# Allow `python sweep_runner.py` execution
if __name__ == "__main__":
    launch_sweep()