# Copyright © 2025 Samuel Holt. All rights reserved.
# No licence is granted to copy, use, modify, distribute, or create derivative
# works of this file in any form, except with explicit written permission from
# the copyright holder.
"""
single_sweep.py

CLI helper to launch ONE baseline experiment under the RECALL project and
persist the outcome as a pickle.
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path
from typing import Any, Dict

from recall import baseline_launcher
from recall import config as cfg_lib


# ──────────────────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────────────────
def _build_cfg(ns: argparse.Namespace) -> cfg_lib.ExperimentConfig:
    """Translate argparse.Namespace → ExperimentConfig instance."""
    return cfg_lib.ExperimentConfig(
        env_name=ns.env_name,
        method=ns.method,
        seed=ns.seed,
        experiment=ns.experiment,
        workdir=ns.results_dir,
    )


def _dump(out_dir: Path, fname: str, payload: Dict[str, Any]) -> None:
    """Serialise *payload* (dict) to pickle → out_dir/fname."""
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / fname).open("wb") as fh:
        pickle.dump(payload, fh)


# ──────────────────────────────────────────────────────────────────────────
# Main entry-point
# ──────────────────────────────────────────────────────────────────────────
def run(argv: list[str] | None = None) -> None:
    """Parse CLI flags, launch experiment, pickle results."""
    argv = sys.argv[1:] if argv is None else argv

    p = argparse.ArgumentParser(prog="single_sweep")
    p.add_argument("--env_name", required=True)
    p.add_argument("--method", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--experiment", required=True)
    p.add_argument("--results_dir", default="./logs")

    ns = p.parse_args(argv)

    cfg = _build_cfg(ns)
    exp_cfg_dict = cfg_lib.convert_config_to_dict(cfg)

    outcome = baseline_launcher.run_experiment(
        cfg.env_name, cfg.seed, cfg.method, cfg.experiment, exp_cfg_dict
    )

    fname = f"{cfg.env_name}_{cfg.method}_seed{cfg.seed}.pickle"
    _dump(Path(cfg.workdir), fname, {"config": vars(ns), "result": outcome})

    print(f"✔ Results stored at {Path(cfg.workdir) / fname}")


# Allow `python single_sweep.py ...` execution
if __name__ == "__main__":
    run()