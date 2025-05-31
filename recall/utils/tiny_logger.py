# Copyright © 2025 Samuel Holt. All rights reserved.
# No licence is granted to copy, use, modify, distribute, or create derivative
# works of this file in any form, except with explicit written permission from
# the copyright holder.
"""tiny_logger.py
A micro-utility for console timestamping.
"""
from __future__ import annotations

from datetime import datetime


def stamp(msg: str) -> None:  # renamed from `log`
    """
    Emit *msg* to stdout with an ISO-like timestamp prefix.

    Parameters
    ----------
    msg : str
        Text you want to display.
    """
    ts: str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}")