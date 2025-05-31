# Copyright © 2025 Samuel Holt. All rights reserved.
# No licence is granted to copy, use, modify, distribute, or create derivative
# works of this file in any form, except with explicit written permission from
# the copyright holder.
"""Minimal ALFWorld-style symbolic environment for text-based agents.

Revision 2025-05-13
───────────────────
* **Deterministic resets** – `reset(seed=…)` now (re)generates the *exact same*
  room layout whenever the same `seed` is supplied.  If no seed is given, the
  room is restored to the **original blueprint** created at construction time
  (all drawers closed, objects back to their starting receptacles).
* **Clean state guarantee** – every call to `reset()` closes all openables and
  clears inventory, ensuring a fresh episode.
* **Never-Solved Guarantee** – unchanged: reset still makes sure the initial
  configuration does not already satisfy the goal.

Usage
─────
```python
env = AlfMiniEnv(seed=42)   # build a deterministic room once
env.reset()                 # ⇒ fresh copy of that room every time

env.reset(seed=123)         # build an entirely new deterministic room
```
"""
from __future__ import annotations
import random, re, textwrap, dataclasses as _dc, copy as _copy
from typing import List, Tuple

###############################################################################
# Basic data containers                                                        #
###############################################################################

_EXAMPLES_REACT = "Example 1 :\n\n You are in the middle of a room. Looking quickly around you, you see a cabinet 4, a cabinet 3, a cabinet 2, a cabinet 1, a countertop 1, a garbagecan 1, a handtowelholder 2, a handtowelholder 1, a sinkbasin 2, a sinkbasin 1, a toilet 1, a toiletpaperhanger 1, and a towelholder 1.\nYour task is to: put some spraybottle on toilet.\n> think: To solve the task, I need to find and take a sparybottle, then put it on toilet.\nOK.\n> think: First I need to find a spraybottle. A spraybottle is more likely to appear in cabinet (1-4), countertop (1), toilet (1), sinkbasin (1-2), garbagecan (1). I can check one by one, starting with cabinet 1.\nOK.\n> go to cabinet 1\nOn the cabinet 1, you see a cloth 1, a soapbar 1, a soapbottle 1.\n> go to cabinet 2\nThe cabinet 2 is closed.\n> open cabinet 2\nYou open the cabinet 2. The cabinet 2 is open. In it, you see a candle 1, and a spraybottle 2.\n> think: Now I find a spraybottle (2). Next, I need to take it.\n> take spraybottle 2 from cabinet 2\nYou pick up the spraybottle 2 from the cabinet 2.\n> think: Now I take a spraybottle (2). Next, I need to put it in/on toilet 1.\nOK.\n> go to toilet 1\nOn the toilet 1, you see a soapbottle 2.\n> put spraybottle 2 in/on toilet 1\nYou put the spraybottle 2 in/on the toilet 1.\n\nExample 2:\n\nYou are in the middle of a room. Looking quickly around you, you see a bed 1, a desk 1, a drawer 3, a drawer 2, a drawer 1, a garbagecan 1, a safe 1, a shelf 5, a shelf 4, a shelf 3, a shelf 2, a shelf 1, a sidetable 2, and a sidetable 1.\nYour task is to: look at bowl under the desklamp.\n> think: To solve the task, I need to find and take a bowl, then find and use a desklamp. \nOK. \n> think: First I need to find a bowl. A bowl is more likely to appear in drawer (1-3), desk (1), sidetable (1-2), shelf (1-5), garbagecan (1). I can check one by one, starting with drawer 1.\nOK.\n> go to drawer 1\nOn the drawer 1, you see nothing.\n> go to drawer 2\nThe drawer 2 is closed.\n> open drawer 2\nYou open the drawer 2. The drawer 2 is open. In it, you see nothing.\n> go to drawer 3\nThe drawer 3 is closed.\n> open drawer 3\nYou open the drawer 3. The drawer 3 is open. In it, you see nothing.\n> go to desk 1\nOn the desk 1, you see a alarmclock 2, a book 1, a cellphone 1, a keychain 1, a laptop 2, a laptop 1, and a pen 3.\n> go to sidetable 1\nOn the sidetable 1, you see a cd 1, a pen 1, and a pencil 1.\n> go to shelf 1\nOn the shelf 1, you see nothing.\n> go to shelf 2\nOn the shelf 2, you see a bowl 1.\n> think: Now I find a bowl (1). Next, I need to take it.\nOK.\n> take bowl 1 from shelf 2\nYou pick up the bowl 1 from the shelf 2.\n> think: Now I take a bowl (1). Next, I need to find a desklamp. A desklamp is more likely to appear in desk (1), sidetable (1-2), shelf (1-5), bed (1), drawer (1-3). I can check one by one, starting with desk 1.\nOK.\n> go to desk 1\nOn the desk 1, you see a alarmclock 2, a book 1, a cellphone 1, a keychain 1, a laptop 2, a laptop 1, and a pen 3.\n> go to sidetable 1\nOn the sidetable 1, you see a cd 1, a pen 1, and a pencil 1.\n> go to sidetable 2\nOn the sidetable 2, you see a alarmclock 1, a desklamp 1, and a pen 2.\n> think: Now I find a desklamp (1). Next, I need to use it.\nOK.\n> use desklamp 1\nYou turn on the desklamp 1.\n"

Action = str          

@_dc.dataclass
class Object:
    name: str
    movable: bool = True


@_dc.dataclass
class Receptacle:
    name: str
    openable: bool
    is_open: bool
    loc: int
    objects: List[Object]

    def brief(self) -> str:
        if self.openable:
            state = "open" if self.is_open else "closed"
            return f"The {self.name} is {state}."
        return f"You are at the {self.name}."


@_dc.dataclass
class Goal:
    instruction: str
    target_obj_prefix: str
    target_receptacle: str


###############################################################################
# The environment                                                              #
###############################################################################


class AlfMiniEnv:
    """A tiny subset of ALFWorld geared toward language-only agents."""

    _HELP = textwrap.dedent(
        """Available commands:
  look:                             look around your current location
  inventory:                        check your current inventory
  go to (receptacle):               move to a receptacle
  open (receptacle):                open a receptacle
  close (receptacle):               close a receptacle
  take (object) from (receptacle):  take an object from a receptacle
  move (object) to (receptacle):    place an object in or on a receptacle
  examine (something):              examine a receptacle or an object
  use (object):                     use an object
  heat (object) with (receptacle):  heat an object using a receptacle
  clean (object) with (receptacle): clean an object using a receptacle
  cool (object) with (receptacle):  cool an object using a receptacle
  slice (object) with (object):     slice an object using a sharp object
  help:                             show this list of commands""")

    _RECEPTACLE_TYPES = [
        "drawer", "shelf", "sidetable", "cabinet", "sofa", "armchair",
        "dresser", "garbagecan", "safe",
    ]
    _OPENABLE_SET = {"drawer", "cabinet", "garbagecan", "safe"}

    _OBJECT_TYPES = ["vase", "keychain", "book", "apple", "knife"]

    # ------------------------------------------------------------------

    def __init__(self, *, seed: int = 0, max_steps: int = 18) -> None:
        """Create a world layout determined by *seed* (deterministic)."""
        self._seed = seed
        self._rng = random.Random(seed)
        self._max_steps = max_steps
        self._build_static_world()              # initial room layout
        # keep a pristine copy – used for cheap deterministic restores
        self._blueprint: List[Receptacle] = _copy.deepcopy(self._receptacles)
        self.reset()  # initialise episode-state (inventory, step-counter, …)
        self.goal = "task: put some vase in safe"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self, *, seed: int | None = None) -> str:
        """Start a **fresh** episode.

        • `seed` provided → re-generate a brand-new *deterministic* room.
        • `seed` omitted → restore the **original room blueprint** created at
          construction time (closes drawers, moves objects back, etc.).
        """
        if seed is not None:
            # brand-new deterministic world ---------------------------------
            self._seed = seed
            self._rng = random.Random(seed)
            self._build_static_world()
            self._blueprint = _copy.deepcopy(self._receptacles)
        else:
            # cheap restore from pristine blueprint -------------------------
            self._receptacles = _copy.deepcopy(self._blueprint)
            self._rec_by_name = {r.name: r for r in self._receptacles}

        # episode-level state ----------------------------------------------
        self._steps = 0
        self._done = False
        self._inventory: List[Object] = []
        self._at: str | None = None  # None → middle of the room

        # fixed canonical goal – update/extend as desired
        self._goal = Goal(
            instruction="put some vase in safe",
            target_obj_prefix="vase",
            target_receptacle="safe 1",
        )

        self._ensure_unsolved()  # may shuffle vases if needed
        return self._look()

    # ------------------------------------------------------------------
    # Step – execute a command                                            
    # ------------------------------------------------------------------

    def step(self, command: str) -> Tuple[str, float, bool, dict]:
        """Executes *command* and returns `(observation, reward, done)`."""
        if self._done:
            return "Episode already finished.", 0.0, True, {}

        self._steps += 1
        command = command.strip().lower()
        obs = self._dispatch(command)

        reward = 0.0
        if not self._done and self._check_goal():
            reward = 1.0
            self._done = True
            obs = "You won!"

        if self._steps >= self._max_steps:
            self._done = True

        return obs, reward, self._done, {}

    # ------------------------------------------------------------------
    # Command dispatchers                                                
    # ------------------------------------------------------------------

    def _dispatch(self, cmd: str) -> str:
        if cmd == "look":
            return self._look()
        if cmd == "inventory":
            return self._inv_desc()
        if cmd == "help":
            return self._HELP

        patterns = [
            (r"go to (.+)", self._go_to),
            (r"open (.+)", self._open),
            (r"close (.+)", self._close),
            (r"take (.+) from (.+)", self._take),
            (r"move (.+) to (.+)", self._move),
            (r"put (.+) in/on (.+)", self._move),  # alias
            (r"examine (.+)", self._examine),
        ]
        for pat, fn in patterns:
            m = re.fullmatch(pat, cmd)
            if m:
                return fn(*m.groups())

        if re.match(r"(use|heat|clean|cool|slice) ", cmd):
            return "Nothing happens."

        return "Nothing happens."

    # ---- concrete handlers ------------------------------------------

    def _go_to(self, rec_name: str) -> str:
        rec = self._rec(rec_name)
        if not rec:
            return "Nothing happens."
        self._at = rec.name
        return f"You arrive at {rec.name}. {rec.brief()}" + self._list_contents(rec)

    def _open(self, rec_name: str) -> str:
        rec = self._rec(rec_name)
        if not rec or not rec.openable or rec.is_open:
            return "Nothing happens."
        rec.is_open = True
        if len(self._list_contents(rec)) == 0:
            return f"You open the {rec.name}. {rec.brief()} In it, you see nothing."
        else:
            return f"You open the {rec.name}. {rec.brief()}" + self._list_contents(rec)

    def _close(self, rec_name: str) -> str:
        rec = self._rec(rec_name)
        if not rec or not rec.openable or not rec.is_open:
            return "Nothing happens."
        rec.is_open = False
        return f"You close the {rec.name}. The {rec.name} is closed."

    def _take(self, obj_name: str, rec_name: str) -> str:
        rec = self._rec(rec_name)
        if not rec or (rec.openable and not rec.is_open):
            return "Nothing happens."
        for i, obj in enumerate(rec.objects):
            if obj.name == obj_name:
                self._inventory.append(rec.objects.pop(i))
                return f"You pick up the {obj_name} from the {rec_name}."
        return "Nothing happens."

    def _move(self, obj_name: str, rec_name: str) -> str:
        rec = self._rec(rec_name)
        if not rec or (rec.openable and not rec.is_open):
            return "Nothing happens."
        for i, obj in enumerate(self._inventory):
            if obj.name == obj_name:
                rec.objects.append(self._inventory.pop(i))
                return f"You move the {obj_name} to the {rec_name}."
        return "Nothing happens."

    def _examine(self, target_name: str) -> str:
        rec = self._rec(target_name)
        if rec:
            return rec.brief() + self._list_contents(rec)
        for obj in self._inventory:
            if obj.name == target_name:
                return f"It's a {obj.name}."
        for rec in self._receptacles:
            for obj in rec.objects:
                if obj.name == target_name:
                    return f"It's a {obj.name}."
        return "Nothing happens."

    # ------------------------------------------------------------------
    # Helper methods                                                    
    # ------------------------------------------------------------------

    def _look(self) -> str:
        names = [r.name for r in self._receptacles]
        around = ", ".join(names[:-1]) + ", and " + names[-1] if len(names) > 1 else names[0]
        return (
            "You are in the middle of a room. Looking quickly around you, you see a "
            f"{around}.\n\nYour task is to: {self._goal.instruction}."
        )

    def _inv_desc(self) -> str:
        return (
            "You are carrying: " + ", ".join(o.name for o in self._inventory) + "."
            if self._inventory else "You aren't carrying anything."
        )

    def _list_contents(self, rec: Receptacle) -> str:
        if rec.openable and rec.is_open and rec.objects:
            return " In it, you see a " + ", ".join(o.name for o in rec.objects) + "."
        if not rec.openable and rec.objects:
            return " On it, you see a " + ", ".join(o.name for o in rec.objects) + "."
        return ""

    def _rec(self, name: str) -> Receptacle | None:
        return self._rec_by_name.get(name)

    def _check_goal(self) -> bool:
        rec = self._rec(self._goal.target_receptacle)
        return bool(rec and any(o.name.startswith(self._goal.target_obj_prefix) for o in rec.objects))

    @property
    def action_space(self) -> Tuple[str, ...]:
        return ("look", "inventory", "go to (receptacle)", "open (receptacle)",
                "close (receptacle)", "take (object) from (receptacle)",
                "move (object) to (receptacle)", "examine (something)", "help")

    @property
    def env_description(self) -> str:  # noqa: D401
        return f"""You are an expert household‑navigation agent operating in the ALFWorld text environment. You have a discrete list of admissible actions and must decide the single next action to take.
The room contains receptacles - e.g., drawers, shelves, a cabinet, a sofa, an arm-chair, a dresser, a garbage-can - each of which may hold everyday objects (e.g., vases, key-chains, books, apples, knives).
All openables start **closed** and your inventory is **empty**.

The environment returns your location description and any objects you see, as well as feedback from previous actions.

**Task/Goal** - {self.goal}
When you complete the task, you will receive a reward of +1.0. Every other action will receive a reward of 0.0.
The episode terminates on success or after a maximum of {self._max_steps} steps.


List of admissible actions:
look: look around your current location
inventory: check your current inventory
go to (receptacle): move to a receptacle
open (receptacle): open a receptacle
close (receptacle): close a receptacle
take (object) from (receptacle): take an object from a receptacle
move (object) to (receptacle): place an object in or on a receptacle
examine (something): examine a receptacle or an object
use (object): use an object
heat (object) with (receptacle): heat an object using a receptacle
clean (object) with (receptacle): clean an object using a receptacle
cool (object) with (receptacle): cool an object using a receptacle
slice (object) with (object): slice an object using a sharp object
help: repeat the available actions

Here are two examples of how to use the environment:


{_EXAMPLES_REACT}
"""

#     @property
#     def env_description(self) -> str:
#         return f"""You are an agent in a **deterministic single-room household simulator** (AlfMini).  
# The room contains receptacles - drawers, shelves, a cabinet, a sofa, an arm-chair, a dresser, a garbage-can, and exactly one safe - each of which may hold everyday objects (vases, key-chains, books, apples, knives).  
# All openables start **closed** and your inventory is **empty**.

# **Goal** — {self._goal.instruction}.  
# You succeed (and earn **+1 reward**) the first time *any* object whose name starts with “{self._goal.target_obj_prefix}” is inside “{self._goal.target_receptacle}”.  
# Every other action yields **0 reward**.  
# The episode terminates on success or after **{self._max_steps} steps**.

# **Observation** after every step is a concise text description of what you now see, what you are carrying, or the result of your action.

# **Available commands** (one per step - see `help` in-game for exact grammar):

# * look:                             look around your current location
# * inventory:                        check your current inventory
# * go to (receptacle):               move to a receptacle
# * open (receptacle):                open a receptacle
# * close (receptacle):               close a receptacle
# * take (object) from (receptacle):  take an object from a receptacle
# * move (object) to (receptacle):    place an object in or on a receptacle
# * examine (something):              examine a receptacle or an object
# * help:                             show this list of commands


# You have {self._max_steps} chances to transform the room so that the goal condition is met.
# """


    # ------------------------------------------------------------------
    # Make sure the episode isn't solved at start                        
    # ------------------------------------------------------------------

    def _ensure_unsolved(self):
        if not self._check_goal():
            return  # already unsolved
        goal_rec = self._rec(self._goal.target_receptacle)
        if not goal_rec:
            return
        # Move all matching objects out of the goal receptacle
        movers = [o for o in goal_rec.objects if o.name.startswith(self._goal.target_obj_prefix)]
        for obj in movers:
            goal_rec.objects.remove(obj)
            dest = self._rng.choice([r for r in self._receptacles if r.name != goal_rec.name])
            dest.objects.append(obj)

    # ------------------------------------------------------------------
    # Static world generation                                            
    # ------------------------------------------------------------------

    def _build_static_world(self):
        """Populate `self._receptacles` deterministically from `self._rng`."""
        self._receptacles: List[Receptacle] = []  # clear previous world
        loc = 1
        for r_type in self._RECEPTACLE_TYPES:
            if r_type == "safe":
                # Only one safe
                self._receptacles.append(
                    Receptacle(
                        name=f"{r_type} 1",
                        openable=True,
                        is_open=False,
                        loc=loc,
                        objects=[],
                    )
                )
                loc += 1
                continue
            for idx in range(1, self._rng.randint(1, 3) + 1):
                self._receptacles.append(
                    Receptacle(
                        name=f"{r_type} {idx}",
                        openable=r_type in self._OPENABLE_SET,
                        is_open=False,
                        loc=loc,
                        objects=[],
                    )
                )
                loc += 1

        for o_type in self._OBJECT_TYPES:
            for idx in range(1, self._rng.randint(1, 3) + 1):
                rec = self._rng.choice(self._receptacles)
                if rec.name == "safe 1":
                    # Don't put objects in the safe at start
                    continue
                rec.objects.append(Object(name=f"{o_type} {idx}"))

        self._rec_by_name = {r.name: r for r in self._receptacles}


###############################################################################
# Simple CLI demo                                                              #
###############################################################################


if __name__ == "__main__":  # pragma: no cover
    env = AlfMiniEnv(seed=0)
    print(env.reset())
    done = False
    while not done:
        cmd = input("\n> ")
        obs, reward, done, info = env.step(cmd)
        print(obs)
        if reward:
            print("Reward:", reward)
