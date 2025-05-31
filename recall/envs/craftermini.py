# Copyright © 2025 Samuel Holt. All rights reserved.
# No licence is granted to copy, use, modify, distribute, or create derivative
# works of this file in any form, except with explicit written permission from
# the copyright holder.
"""Deterministic, autorestoring *text‑only Crafter* environment.

This is a **drop‑in replacement** for the `CrafterTextEnv` class from
`crafter_text_mcts.py`.  The gameplay rules are unchanged – only the *reset*
behaviour was fixed so that every episode starts from a **pristine world**.

Key additions
-------------
* **Blueprint snapshot** – the initial grid is deep‑copied and kept in
  `self._blueprint`.  Subsequent calls to `reset()` restore that blueprint
  (hill‑climbing performance reproducibility).
* **`reset(seed=…)`** – pass a seed to generate a *new* deterministic world and
  update the blueprint to that layout.
* **Inventory/crafted/position** are re‑initialised each reset, of course.

If you already import `CrafterTextEnv` elsewhere, just replace the import:
    from crafter_env import CrafterTextEnv
"""
from __future__ import annotations

import copy, random
from collections import Counter
from typing import Dict, List, Tuple
import json, re
from typing import Any, Dict


__all__ = ["CrafterMiniEnv"]


class CrafterMiniEnv:
    """A tiny textual Crafter world with deterministic resets."""

    # ------------ static constants ---------------------------------------
    NAMES = {
        0: "north",
        1: "south",
        2: "east",
        3: "west",
        4: "collect",
        5: "craft_wood_pickaxe",
        6: "craft_stone_pickaxe",
        7: "craft_iron_pickaxe",
    }
    DIRS: Dict[int, Tuple[int, int]] = {0: (-1, 0), 1: (1, 0), 2: (0, 1), 3: (0, -1)}
    RES_TILES = {"tree": "wood", "stone": "stone", "iron": "iron"}

    RECIPES: Dict[str, Counter] = {
        "wood_pickaxe": Counter({"wood": 3}),
        "stone_pickaxe": Counter({"wood": 1, "stone": 3}),
        "iron_pickaxe": Counter({"stone_pickaxe": 1, "iron": 3}),
    }
    CRAFT_IDS = {5: "wood_pickaxe", 6: "stone_pickaxe", 7: "iron_pickaxe"}
    CRAFT_REWARD = {"wood_pickaxe": 10, "stone_pickaxe": 20, "iron_pickaxe": 50}

    # ------------------------------------------------------------------
    def __init__(self, *, size: int = 5, max_steps: int | None = None, seed: int | None = None):
        self.size = size
        self.max_steps = max_steps if max_steps is not None else 4 * size ** 2
        self._seed = seed
        self._rng = random.Random(seed)
        self._generate_world()                # sets self.grid
        self._blueprint = copy.deepcopy(self.grid)  # immutable reference layout
        self.reset()  # initialise episode‑state (pos, inventory, …)

    # ------------------------------------------------------------------
    # World generation & helpers
    # ------------------------------------------------------------------
    def _generate_world(self) -> None:
        tiles = ["grass", "tree", "stone", "iron", "water"]
        while True:
            self.grid = [[self._rng.choice(tiles) for _ in range(self.size)] for _ in range(self.size)]
            flat = sum(self.grid, [])
            if all(r in flat for r in ("tree", "stone", "iron")):
                break

    def _observe(self) -> str:
        x, y = self.pos
        tile = self.grid[x][y]
        neigh = {
            self.NAMES[a]: self.grid[(x + dx) % self.size][(y + dy) % self.size]
            for a, (dx, dy) in self.DIRS.items()
        }
        inv = ", ".join(f"{k}={v}" for k, v in self.inventory.items()) or "empty"
        crafted = [k for k, v in self.crafted.items() if v]
        crafted_str = "Crafted: " + ", ".join(crafted) if crafted else "Nothing crafted yet."
        return (
            f"You stand on {tile} at ({x},{y}). "
            + " | ".join(f"{d.upper()}: {t}" for d, t in neigh.items())
            + f". Inventory: {inv}. "
            + crafted_str
        )

    def _can_craft(self, item: str) -> bool:
        need = self.RECIPES[item]
        have = self.inventory + Counter({k: 1 for k, v in self.crafted.items() if v})
        return all(have[r] >= c for r, c in need.items())

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def reset(self, *, seed: int | None = None) -> str:
        """Start a fresh episode.
        • pass `seed` to *rebuild* a brand‑new room deterministically
          (and update the blueprint).
        • omit `seed` to restore the original blueprint exactly.
        """
        if seed is not None:
            self._seed = seed
            self._rng = random.Random(seed)
            self._generate_world()
            self._blueprint = copy.deepcopy(self.grid)
        else:
            self.grid = copy.deepcopy(self._blueprint)

        # episode‑local state ------------------------------------------
        self.pos = (self._rng.randrange(self.size), self._rng.randrange(self.size))
        self.inventory: Counter[str] = Counter()
        self.crafted: Dict[str, bool] = {k: False for k in self.RECIPES}
        self.steps = 0
        return self._observe()

    def available_actions(self) -> List[int]:
        acts: List[int] = list(range(4))
        if self.grid[self.pos[0]][self.pos[1]] in self.RES_TILES:
            acts.append(4)                           # collect only on resources
        for aid, name in self.CRAFT_IDS.items():
            if not self.crafted[name] and self._can_craft(name):
                acts.append(aid)
        return acts

    def step(self, action: int):
        action = int(action)
        legal = action in self.available_actions()


        self.steps += 1
        reward, done = -1, False  # step cost

        # --------------------------------------------------------------
        # Handle illegal actions – do nothing except maybe terminate
        # --------------------------------------------------------------
        if not legal:
            done = self.crafted["iron_pickaxe"] or self.steps >= self.max_steps
            return self._observe(), reward, done, {"illegal": True}

        if action in self.DIRS:  # movement
            dx, dy = self.DIRS[action]
            x, y = self.pos
            self.pos = ((x + dx) % self.size, (y + dy) % self.size)

        elif action == 4:  # collect
            x, y = self.pos
            tile = self.grid[x][y]
            if tile in self.RES_TILES:
                res = self.RES_TILES[tile]
                self.inventory[res] += 1
                self.grid[x][y] = "grass"

        else:  # crafting
            item = self.CRAFT_IDS[action]
            if self._can_craft(item):
                for res, cnt in self.RECIPES[item].items():
                    if res.endswith("_pickaxe"):
                        self.crafted[res] = False  # consume tool
                    else:
                        self.inventory[res] -= cnt
                self.crafted[item] = True
                reward += self.CRAFT_REWARD[item]

        done = self.crafted["iron_pickaxe"] or self.steps >= self.max_steps
        return self._observe(), reward, done, {}

    # -------- utilities -------------------------------------------------
    def clone(self) -> "CrafterMiniEnv":
        return copy.deepcopy(self)

    def render(self) -> None:
        glyph = {"grass": ".", "tree": "T", "stone": "S", "iron": "I", "water": "~"}
        for i in range(self.size):
            row = ["@" if (i, j) == self.pos else glyph[self.grid[i][j]] for j in range(self.size)]
            print(" ".join(row))
        print(self._observe(), "\n")

    def __repr__(self) -> str:  # handy for hashing / caching
        return str(self.pos) + str(self.inventory) + str(self.crafted) + "|".join(sum(self.grid, []))
    
    @property
    def action_space(self) -> Tuple[str, ...]:
        return ("0", "1", "2", "3", "4", "5", "6", "7")

    @property
    def env_description(self) -> str:
        """Return a description of the environment."""
        return f"""You are an agent in a deterministic, text-only {self.size}x{self.size} *Crafter* world laid out on a toroidal (wrap-around) grid.  
Each tile is one of: grass (.), tree (T), stone (S), iron (I) or water (~); the generator guarantees at least one tree, stone and iron so the game is always solvable.

**Observation format** - a single sentence that tells you  
* the terrain under your feet and your (row, col) coordinates,  
* the terrain in the four cardinal directions,  
* your inventory contents, and  
* which tools have been crafted.

**Actions** (integers):  
* 0 north
* 1 south
* 2 east
* 3 west
* 4 collect resource on current tile
* 5 craft *wood pickaxe* (needs 3 wood → +10 reward)
* 6 craft *stone pickaxe* (needs 1 wood + 3 stone → +20 reward)
* 7 craft *iron pickaxe* (needs 1 stone pickaxe + 3 iron → +50 reward)

Craft actions are only listed when their recipe can be satisfied.
Collecting turns the resource tile to grass and adds the material to your inventory.

**Rewards & termination** - every step costs -1 reward; crafting grants the bonus above.  
The episode ends immediately after crafting an *iron pickaxe* **or** after {self.max_steps} steps, whichever comes first.
"""

class RobustCrafterMiniEnv(CrafterMiniEnv):
    """CrafterMiniEnv that survives the occasional nonsense string from an LLM."""

    #: loose mapping from (lower-case) synonyms → *canonical* integer actions
    _SYNONYMS: Dict[str, int] = {
        # movement ------------------------------------------------------
        "north": 0,
        "up": 0,
        "n": 0,
        "south": 1,
        "down": 1,
        "s": 1,
        "east": 2,
        "right": 2,
        "e": 2,
        "west": 3,
        "left": 3,
        "w": 3,
        # collect -------------------------------------------------------
        "collect": 4,
        "gather": 4,
        "mine": 4,
        # crafting ------------------------------------------------------
        "craft wood pickaxe": 5,
        "craft_wood_pickaxe": 5,
        "wood pickaxe": 5,
        "craft wood": 5,
        "craft stone pickaxe": 6,
        "craft_stone_pickaxe": 6,
        "stone pickaxe": 6,
        "craft iron pickaxe": 7,
        "craft_iron_pickaxe": 7,
        "iron pickaxe": 7,
    }

    _DIGIT_RE = re.compile(r"\d+")

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _sanitize_action(self, action: Any) -> int | None:
        """Attempt to coerce *action* to a legal integer (0–7).

        Returns the integer on success or **None** on failure.
        """
        # --- already an int/float ------------------------------------
        if isinstance(action, int):
            return action if 0 <= action <= 7 else None
        if isinstance(action, float):
            ai = int(action)
            return ai if 0 <= ai <= 7 else None

        # --- textual input -------------------------------------------
        if isinstance(action, str):
            s = action.strip().lower()

            # try JSON: {"action": 2} or {"action": "north"}
            try:
                obj = json.loads(s)
                if isinstance(obj, dict) and "action" in obj:
                    s = str(obj["action"]).lower().strip()
            except Exception:
                pass  # not JSON – carry on

            # explicit integer inside the string (e.g. "move 3 now!")
            if (m := self._DIGIT_RE.search(s)):
                ai = int(m.group(0))
                if 0 <= ai <= 7:
                    return ai

            # synonyms / substrings
            for key, aid in self._SYNONYMS.items():
                if key in s:
                    return aid

        # --- give up --------------------------------------------------
        return None

    # ------------------------------------------------------------------
    # Public API overrides
    # ------------------------------------------------------------------
    def step(self, action):  # type: ignore[override]
        """Identical to *CrafterMiniEnv.step* except for robust parsing."""
        parsed = self._sanitize_action(action)
        if parsed is None:  # unrecognised → illegal
            self.steps += 1
            reward = -1
            done = self.crafted["iron_pickaxe"] or self.steps >= self.max_steps
            return self._observe(), reward, done, {
                "illegal": True,
                "parse_error": True,
                "raw_action": action,
            }
        # delegate to parent implementation
        return super().step(parsed)

    # we intentionally inherit everything else without change
