# Copyright © 2025 Samuel Holt. All rights reserved.
# No licence is granted to copy, use, modify, distribute, or create derivative
# works of this file in any form, except with explicit written permission from
# the copyright holder.
from __future__ import annotations
"""
Text‑based Frozen Lake (descriptive observation)
===============================================

This variant emits a **natural‑language observation** each step, e.g.:

```text
You are at (0, 1) on ice.
```

Only the agent’s **current coordinates** and the *terrain of the square it
stands on* are revealed—no global map information is leaked.  The board itself
is fixed at reset; Start is (0, 0) and Goal is (n − 1, n − 1).

*   **Observation** – plain text description of position + terrain.
*   **Actions** – "up", "down", "left", "right".
*   **Rewards** – +1 reaching the goal, −1 falling in hole, 0 otherwise.
*   **Solvable** – Manhattan corridor ensures at least one path.
*   **Deterministic** with optional *seed*.

Example
-------
```python
from text_frozen_lake_env import TextFrozenLakeEnv

env = TextFrozenLakeEnv(size=4, hole_density=0.25, seed=7)
obs = env.reset()
print(obs)  # "You are at (0, 0) on start."
obs, r, done, _ = env.step("right")
print(obs)  # "You are at (0, 1) on ice." (assuming safe)
```
"""

import random
from typing import List, Tuple, Optional, Dict

Action = str           # "up" | "down" | "left" | "right"
Observation = str      # descriptive text


class TextFrozenLakeEnv:
    """Frozen Lake with text‑only local observations."""

    ACTIONS: Tuple[Action, ...] = ("up", "down", "left", "right")

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def __init__(self, size: int = 4, hole_density: float = 0.2, *, seed: Optional[int] = None) -> None:
        if size < 2:
            raise ValueError("size must be ≥ 2")
        if not (0.0 <= hole_density <= 1.0):
            raise ValueError("hole_density must be in [0, 1]")

        self.size = size
        self.hole_density = hole_density
        self._rng = random.Random(seed)
        self.seed_value = seed

        # Episode length limit: 4× optimal path length (2·(n−1))
        self.max_steps = 8 * (self.size - 1)
        self._step_count = 0

        # Immutable board layout
        self._board: List[List[str]] = self._generate_board()
        self.agent_pos: Tuple[int, int] = (0, 0)

    # ------------------------------------------------------------------
    # Public RL interface
    # ------------------------------------------------------------------
    def reset(self) -> Observation:
        self.agent_pos = (0, 0)
        self._step_count = 0
        return self._observation()

    def step(self, action: Action) -> Tuple[Observation, float, bool, Dict]:
        if action not in self.ACTIONS:
            raise ValueError(f"Invalid action: {action} ∉ {self.ACTIONS}")

        r, c = self.agent_pos
        if action == "up":
            r = max(r - 1, 0)
        elif action == "down":
            r = min(r + 1, self.size - 1)
        elif action == "left":
            c = max(c - 1, 0)
        elif action == "right":
            c = min(c + 1, self.size - 1)
        self.agent_pos = (r, c)

        tile = self._board[r][c]
        done = tile in {"H", "G"}
        reward = 1.0 if tile == "G" else (-1.0 if tile == "H" else 0.0)

        # Time‑limit enforcement
        self._step_count += 1
        if self._step_count >= self.max_steps and not done:
            done = True
        info = {"time_limit_reached": self._step_count >= self.max_steps}
        return self._observation(), reward, done, info

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _observation(self) -> Observation:
        r, c = self.agent_pos
        tile = self._board[r][c]
        terrain = {
            "S": "start",
            "G": "goal",
            "H": "hole",
            ".": "ice",
        }[tile]
        return f"You are at ({r}, {c}) on {terrain}."

    def _generate_board(self) -> List[List[str]]:
        n = self.size
        board = [["." for _ in range(n)] for _ in range(n)]

        # Build a zig‑zag safe path that stays near the diagonal and off the walls.
        path_cells = {(0, 0)}
        r = c = 0
        move_right = True  # alternate moves
        while (r, c) != (n - 1, n - 1):
            if move_right and c < n - 1:
                c += 1
            elif r < n - 1:
                r += 1
            else:  # at last row, must move right until end
                c += 1
            path_cells.add((r, c))
            move_right = not move_right

        # Sample holes elsewhere
        for i in range(n):
            for j in range(n):
                if (i, j) in path_cells or (i, j) in {(0, 0), (n - 1, n - 1)}:
                    continue
                if self._rng.random() < self.hole_density:
                    board[i][j] = "H"

        # Mark tiles
        board[0][0] = "S"
        board[n - 1][n - 1] = "G"
        for (i, j) in path_cells - {(0, 0), (n - 1, n - 1)}:
            board[i][j] = "."  # already default, but explicit for clarity
        return board

    # ------------------------------------------------------------------
    # Debug render (not exposed to agent)
    # ------------------------------------------------------------------
    def render(self) -> None:  # pragma: no cover
        r_a, c_a = self.agent_pos
        for r in range(self.size):
            row = []
            for c in range(self.size):
                row.append("A" if (r, c) == (r_a, c_a) else self._board[r][c])
            print(" ".join(row))
        print()

    # ------------------------------------------------------------------
    # Utilities / compatibility
    # ------------------------------------------------------------------
    def seed(self, seed: Optional[int] = None) -> None:
        self._rng.seed(seed)
        self.seed_value = seed

    @property
    def action_space(self) -> Tuple[Action, ...]:
        return self.ACTIONS

    @property
    def observation_space(self) -> Tuple[int, int]:
        """Return coordinate bounds (rows, cols)."""
        return self.size, self.size
    
    @property
    def env_description(self) -> str:
        return f"""
You are an agent in a deterministic simulator of a {self.size}×{self.size} Frozen-Lake gridworld. Start=(0,0), Goal=({self.size-1},{self.size-1}). Holes are lethal. Your goal is to reach the goal (G) while avoiding holes (H). Each observation is given as a tuple of (row, column) coordinates, and the terrain type at that position. You receive -1 reward at a hole, +1 at the goal, and 0 otherwise. The maximum number of steps is {self.max_steps}. The hole density is {self.hole_density:.2f}. There is gauranteed to be a path to the goal.
"""

    def __str__(self) -> str:  # pragma: no cover
        return f"TextFrozenLakeEnv(size={self.size}, holes={self.hole_density:.2f})"
    


    __repr__ = __str__


# -------------------------------------------------------------------------
# Smoke test
# -------------------------------------------------------------------------
if __name__ == "__main__":  # pragma: no cover
    env = TextFrozenLakeEnv(size=8, hole_density=0.5, seed=123)
    obs = env.reset()
    print(obs)
    for a in ("right", "down", "down", "right", "right", "down"):
        obs, r, done, _ = env.step(a)
        print(f"{a:5s}: {obs}  reward={r:+.0f}  done={done}")
        if done:
            env.render()
            break
