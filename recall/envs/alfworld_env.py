# Copyright © 2025 Samuel Holt. All rights reserved.
# No licence is granted to copy, use, modify, distribute, or create derivative
# works of this file in any form, except with explicit written permission from
# the copyright holder.
from __future__ import annotations
"""
ALFWorld Text Environment Wrapper
================================

This module provides a **thin adapter** around the official *ALFWorld* text
environment so that it follows the exact same RL interface contract as the
other environments in our code‑base (see `TextFrozenLakeEnv` for reference).

* **Observation** – raw textual description emitted by ALFWorld (batch size 1).
* **Actions** – the fixed set of language templates in :data:`ACTIONS`.
* **Rewards** – passed through directly from ALFWorld with no modification.
* **Episode length** – capped at ``max_steps`` taken from the YAML config.

Example
-------
```python
from alfworld_env import AlfWorldEnv

env = AlfWorldEnv(seed=7)
obs = env.reset()
print(obs)
obs, r, done, _ = env.step("look")
```
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml
from alfworld.agents.environment import get_environment
import yaml

Action = str
Observation = str


_EXAMPLES_REACT = "Example 1 :\n\n You are in the middle of a room. Looking quickly around you, you see a cabinet 4, a cabinet 3, a cabinet 2, a cabinet 1, a countertop 1, a garbagecan 1, a handtowelholder 2, a handtowelholder 1, a sinkbasin 2, a sinkbasin 1, a toilet 1, a toiletpaperhanger 1, and a towelholder 1.\nYour task is to: put some spraybottle on toilet.\n> think: To solve the task, I need to find and take a sparybottle, then put it on toilet.\nOK.\n> think: First I need to find a spraybottle. A spraybottle is more likely to appear in cabinet (1-4), countertop (1), toilet (1), sinkbasin (1-2), garbagecan (1). I can check one by one, starting with cabinet 1.\nOK.\n> go to cabinet 1\nOn the cabinet 1, you see a cloth 1, a soapbar 1, a soapbottle 1.\n> go to cabinet 2\nThe cabinet 2 is closed.\n> open cabinet 2\nYou open the cabinet 2. The cabinet 2 is open. In it, you see a candle 1, and a spraybottle 2.\n> think: Now I find a spraybottle (2). Next, I need to take it.\n> take spraybottle 2 from cabinet 2\nYou pick up the spraybottle 2 from the cabinet 2.\n> think: Now I take a spraybottle (2). Next, I need to put it in/on toilet 1.\nOK.\n> go to toilet 1\nOn the toilet 1, you see a soapbottle 2.\n> put spraybottle 2 in/on toilet 1\nYou put the spraybottle 2 in/on the toilet 1.\n\nExample 2:\n\nYou are in the middle of a room. Looking quickly around you, you see a bed 1, a desk 1, a drawer 3, a drawer 2, a drawer 1, a garbagecan 1, a safe 1, a shelf 5, a shelf 4, a shelf 3, a shelf 2, a shelf 1, a sidetable 2, and a sidetable 1.\nYour task is to: look at bowl under the desklamp.\n> think: To solve the task, I need to find and take a bowl, then find and use a desklamp. \nOK. \n> think: First I need to find a bowl. A bowl is more likely to appear in drawer (1-3), desk (1), sidetable (1-2), shelf (1-5), garbagecan (1). I can check one by one, starting with drawer 1.\nOK.\n> go to drawer 1\nOn the drawer 1, you see nothing.\n> go to drawer 2\nThe drawer 2 is closed.\n> open drawer 2\nYou open the drawer 2. The drawer 2 is open. In it, you see nothing.\n> go to drawer 3\nThe drawer 3 is closed.\n> open drawer 3\nYou open the drawer 3. The drawer 3 is open. In it, you see nothing.\n> go to desk 1\nOn the desk 1, you see a alarmclock 2, a book 1, a cellphone 1, a keychain 1, a laptop 2, a laptop 1, and a pen 3.\n> go to sidetable 1\nOn the sidetable 1, you see a cd 1, a pen 1, and a pencil 1.\n> go to shelf 1\nOn the shelf 1, you see nothing.\n> go to shelf 2\nOn the shelf 2, you see a bowl 1.\n> think: Now I find a bowl (1). Next, I need to take it.\nOK.\n> take bowl 1 from shelf 2\nYou pick up the bowl 1 from the shelf 2.\n> think: Now I take a bowl (1). Next, I need to find a desklamp. A desklamp is more likely to appear in desk (1), sidetable (1-2), shelf (1-5), bed (1), drawer (1-3). I can check one by one, starting with desk 1.\nOK.\n> go to desk 1\nOn the desk 1, you see a alarmclock 2, a book 1, a cellphone 1, a keychain 1, a laptop 2, a laptop 1, and a pen 3.\n> go to sidetable 1\nOn the sidetable 1, you see a cd 1, a pen 1, and a pencil 1.\n> go to sidetable 2\nOn the sidetable 2, you see a alarmclock 1, a desklamp 1, and a pen 2.\n> think: Now I find a desklamp (1). Next, I need to use it.\nOK.\n> use desklamp 1\nYou turn on the desklamp 1.\n"

class AlfWorldEnv:  # pylint: disable=too-few-public-methods
    """Text‑only ALFWorld environment with a clean, single‑agent API."""

    #: Canonical list of admissible *action templates*.
    ACTIONS: Tuple[Action, ...] = (
        "look",
        "inventory",
        "go to (receptacle)",
        "open (receptacle)",
        "close (receptacle)",
        "take (object) from (receptacle)",
        "move (object) to (receptacle)",
        "examine (something)",
        "use (object)",
        "heat (object) with (receptacle)",
        "clean (object) with (receptacle)",
        "cool (object) with (receptacle)",
        "slice (object) with (object)",
    )




    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def __init__(
        self,
        task_id: int = 0,
        *,
        cfg_path: str | Path = "recall/envs/base_config.yaml",
        train_eval: str = "eval_out_of_distribution",
    ) -> None:
        """Create a new ALFWorld text environment.

        Parameters
        ----------
        task_id
            Index into the ALFWorld *evaluation* split.
        cfg_path
            Path to a YAML file with ALFWorld hyper‑parameters.
        train_eval
            Which subset of tasks to load ("train", "eval_in_distribution", …).
        seed
            RNG seed forwarded to the underlying ALFWorld simulator.
        """
        cfg_path = Path(cfg_path)
        if not cfg_path.is_file():
            raise FileNotFoundError(cfg_path)

        self.cfg: Dict = yaml.safe_load(cfg_path.read_text())

        # Instantiate underlying env (batch_size=1 ⇒ lists of length 1).
        base_cls = get_environment("AlfredTWEnv")
        self._env = base_cls(self.cfg, train_eval=train_eval).init_env(batch_size=1)

        self.max_steps: int = self.cfg["rl"]["training"]["max_nb_steps_per_episode"]
        self._step_count: int = 0
        self.seed(task_id)
        self.goal = self._extract_goal()

    def _extract_goal(self) -> str:
        """Extract the goal from the environment config."""
        # The goal is a string that describes the task.
        # It is stored in the environment config under the key "goal".
        # The goal is a list of strings, so we join them into a single string.
        _obs  = self.reset()
        goal = _obs.split('\n')[-1]
        return goal

    # ------------------------------------------------------------------
    # Public RL interface
    # ------------------------------------------------------------------
    def reset(self) -> Observation:
        """Start a fresh episode and return the first observation."""
        self._step_count = 0
        self._env.seed(self.seed_value)
        obs_list, _info = self._env.reset()
        # ALFWorld returns *batched* outputs ⇒ unwrap the singleton.
        return obs_list[0]

    def step(self, action: Action) -> Tuple[Observation, float, bool, Dict]:
        """Execute *one* action.

        Notes
        -----
        * The underlying ALFWorld `.step()` expects a *list* of actions because
          it supports batching.  We therefore wrap the single action string in
          a list and unwrap the outputs accordingly.
        * A hard time‑limit (`max_steps`) identical to the training config is
          enforced; it sets the ``"time_limit_reached"`` flag in *info*.
        """
        if not isinstance(action, str):
            raise TypeError("Action must be a string – got {type(action).__name__}.")

        obs_list, reward_list, done_list, info = self._env.step([action])
        obs = obs_list[0] if isinstance(obs_list, (list, tuple)) else obs_list 
        # obs: Observation = obs_list[0]
        # if info["won"][0]:
        #     done = bool(info.get("won", False))
        #     reward = 1.0
        # else:
        #     reward = 0.0
        #     done = False
        reward: float = float(reward_list[0])
        done: bool = bool(done_list[0])

        # Time‑limit enforcement -------------------------------------------------
        self._step_count += 1
        if self._step_count >= self.max_steps and not done:
            done = True
            info = {**info, "time_limit_reached": True}

        return obs, reward, done, info

    # ------------------------------------------------------------------
    # Utilities / compatibility
    # ------------------------------------------------------------------
    def seed(self, seed: Optional[int] = None) -> None:
        """(Re‑)seed the environment’s RNGs."""
        self._env.seed(seed)
        self.seed_value = seed

    # -----------------------
    # Read‑only API properties
    # -----------------------
    @property
    def action_space(self) -> Tuple[Action, ...]:  # noqa: D401 (property returns value)
        """Return the tuple of *admissible* action templates."""
        return self.ACTIONS

    @property
    def observation_space(self):  # noqa: D401
        """Observation space is textual – no fixed bounds (returns *None*)."""
        return None

    @property
    def env_description(self) -> str:  # noqa: D401
        return f"""You are an expert household‑navigation agent operating in the ALFWorld text environment. You have a discrete list of admissible actions and must decide the single next action to take.
The room contains receptacles - e.g., drawers, shelves, a cabinet, a sofa, an arm-chair, a dresser, a garbage-can - each of which may hold everyday objects (e.g., vases, key-chains, books, apples, knives).
All openables start **closed** and your inventory is **empty**.

The environment returns your location description and any objects you see, as well as feedback from previous actions.

**Task/Goal** - {self.goal}
When you complete the task, you will receive a reward of +1.0. Every other action will receive a reward of 0.0.
The episode terminates on success or after a maximum of {self.max_steps} steps.


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
``
{_EXAMPLES_REACT}
``

Advice:
* You have to go to a receptacle to be infront of it.
* All openable receptacles are closed at the beginning of the episode, showing nothing. You have to open them to see what is inside.
* Other non-openable receptacles are always open and show their contents only when you are infront of them (by going to them).
* You have to be infront of a receptacle before you can open or close it.
* You must be infront of the receptacle that holds the object that you want to interact with.
* You can only transfer objects between receptacles infront of you and your inventory vice versa through take or move.
* If an action is not admissible, the action is ignored and the environment will return the observation of "Nothing happens."
* You start in the middle room far away from the receptacles. You must go to one of the receptacles first to be infront of it to interact with it. Otherwise, doing other actions in the middle of the room will not do anything, returning "Nothing happens.".
* If you try to interact with an receptacle and nothing happens, it means that you are not infront of it. You go to that receptacle first.
* To take an object, you must be infront of the receptacle that holds the object, then use take (object) from (receptacle).
* You cannot move an object from one receptacle to another direclty. You must first go to the receptacle with the object, take the object from the receptacle, go to the destination receptacle, and then then move the object to the destination receptacle.
* Key information to help you solve the task that you should keep note of is the location of the desired goal object where it starts location wise, to make it easier to find it in subsequent episodes.
"""

    # ------------------------------------------------------------------
    # Python niceties
    # ------------------------------------------------------------------
    def __str__(self) -> str:  # pragma: no cover
        return f"AlfWorldEnv(max_steps={self.max_steps}, seed={self.seed_value})"

    __repr__ = __str__


# [90, ]

# -------------------------------------------------------------------------
# Smoke test 3, 5, 10, 15, 18
# -------------------------------------------------------------------------
if __name__ == "__main__":  # pragma: no cover
    for i in range(500):
        env = AlfWorldEnv(task_id=i)
        obs = env.reset()
        # print(f'Initial: {obs}')
        goal = obs.split("\n")[-1]
        print(f'i {i} - {goal}')
    # obs, r, done, info = env.step("go to drawer 1")
    # print(f"obs: {obs} reward: {r} done: {done} info: {info}")
    # obs, r, done, info = env.step("open drawer 1")
    # print(f"obs: {obs} reward: {r} done: {done} info: {info}")
    # obs, r, done, info = env.step("take watch 1 from drawer 1")
    # print(f"obs: {obs} reward: {r} done: {done} info: {info}")
    # obs, r, done, info = env.step("go to safe 1")
    # print(f"obs: {obs} reward: {r} done: {done} info: {info}")
    # obs, r, done, info = env.step("open safe 1")
    # print(f"obs: {obs} reward: {r} done: {done} info: {info}")
    # obs, r, done, info = env.step("move watch 1 to safe 1")
    # print(f"obs: {obs} reward: {r} done: {done} info: {info}")
    # for _ in range(3):
    #     print(r, done, info)
    #     if done:
    #         break
    # print("Final observation:", obs)
    # # test reset
    # obs = env.reset()
    # print(f'Initial: {obs}')
    # for _ in range(3):
    #     obs, r, done, info = env.step("look")
    #     print(r, done, info)
    #     if done:
    #         break
    # # test reset
    # obs = env.reset()
    # print(f'Initial: {obs}')
    # for _ in range(3):
    #     obs, r, done, info = env.step("look")
    #     print(r, done, info)
    #     if done:
    #         break