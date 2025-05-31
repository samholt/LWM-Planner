# Copyright © 2025 Samuel Holt. All rights reserved.
# No licence is granted to copy, use, modify, distribute, or create derivative
# works of this file in any form, except with explicit written permission from
# the copyright holder.
from __future__ import annotations
"""Async forward‑simulation planner with per‑path history prompts
-----------------------------------------------------------------
The agent performs limited‑depth look‑ahead using an LLM world model and value
function.  Every prompt now contains the *full trajectory so far*.

**2025‑05‑15 additions**
* `verbose` flag (0‑3) with helper `_v()` for easy conditional logging.
* Deterministic world‑model/value calls (`temperature=0.0`) to stabilise search.
* Root Q‑value table printed when `verbose≥1`; prompts and truncated replies
  when `verbose≥2`; raw JSON tool calls when `verbose≥3`.
* Internal fix already applied: `Act:` line is included in history passed to
  `_simulate_step_async`.
"""
import asyncio, random, json, textwrap
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple, Sequence
from dataclasses import dataclass, field
from typing import Any, Dict, Set
from collections import deque

from recall.utils import tiny_logger
from recall.agents.react_fact_extraction import LLMReactFactExtractionAgent

from recall.llm_utils_async import (
    get_async_client,
    async_chat_completion,
)

# ── helper -------------------------------------------------------------------
import asyncio, concurrent.futures          # add concurrent.futures at the top


def _run_coro_in_new_loop(coro):
    """Helper: run *coro* inside its own loop in a worker-thread."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()



def _planning_function_schema(allowed_actions: Tuple[str, ...], branch_factor: int) -> List[Dict[str, Any]]:
    """OpenAI function‑call schema for propose / simulate / value."""
    return [
        {
            "type": "function",
            "function": {
                "name": "propose_actions",
                "description": f"Given the current observation, propose up to {branch_factor} most likely next best unique actions to try next that make the agent solve the environment task optimally.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "thought": {"type": "string", "description": "Agent's private reasoning."},
                        "actions": {
                            "type": "array",
                            "items": {"type": "string"},
                             "description": f"List up to {branch_factor} most likely next best unique actions to try next that make the agent solve the environment task optimally, each from {allowed_actions}."
                        },
                    },
                    "required": ["thought", "actions"],
                    "additionalProperties": False,
                },
                "strict": True,
            },
        },
        {
            "type": "function",
            "function": {
                "name": "simulate_step",
                "description": "You are a latent world model. Given the current observation and an action, predict: the next (perhaps latent) observation, the immediate reward, and the done flag (whether the resulting state ends the episode).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "thought": {"type": "string", "description": "Private reasoning."},
                        "next_observation": {"type": "string", "description": "Predicted (perhaps latent) observation after the action."},
                        "reward": {"type": "number", "description": "Predicted immediate reward (float) after the action."},
                        "done": {
                            "type": "boolean",
                            "description": "True if the resulting state ends the episode (terminal), false otherwise.",
                        },
                    },
                    "required": ["thought", "next_observation", "reward", "done"],
                    "additionalProperties": False,
                },
                "strict": True,
            },
        },
        {
            "type": "function",
            "function": {
                "name": "estimate_value",
                "description": "You are a state value function estimator for the given environment. You estimate the cumulative future reward from the current (perhaps latent) observation.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "thought": {"type": "string", "description": "Private reasoning."},
                        "value": {"type": "number", "description": "Estimated state value (float). The cumulative future reward from the current (perhaps latent) observation."},
                    },
                    "required": ["thought", "value"],
                    "additionalProperties": False,
                },
                "strict": True,
            },
        },
    ]

# ── agent --------------------------------------------------------------------

@dataclass
class LLMWorldModelPlanningAgentAsync(LLMReactFactExtractionAgent):
    search_depth: int = 2
    branch_factor: int = 4
    discount: float = 1.0
    verbose: int = 0               # 0: silent … 3: very noisy
    step_penalty: float = -0.02      # small, so reaching the goal still dominates

    _value_cache: Dict[Any, Any] = field(init=False, repr=False, default_factory=dict)
    _terminal_obs: Set[str] = field(init=False, repr=False, default_factory=set)

    # ------------------------------------------------------------------
    def __post_init__(self):
        super().__post_init__()
        self._planning_tools = _planning_function_schema(self.allowed_actions, self.branch_factor)
        self._tools += self._planning_tools
        self._value_cache.clear()
        self._terminal_obs.clear()

    # ------------------------------------------------------------------
    # helper for conditional logging
    # ------------------------------------------------------------------
    def _v(self, lvl: int, msg: str):
        if self.verbose >= lvl:
            tiny_logger.log(msg)

    # ------------------------------------------------------------------
    # Public sync wrapper
    # ------------------------------------------------------------------
    # def act(self, observation: str) -> str:
        # return asyncio.run(self.act_async(observation))
    # ----------------------------------------------------------------------
    def act(self, observation: str) -> str:                 # ← replace old body
        """Synchronous wrapper that is *event-loop aware*."""
        coro = self.act_async(observation)

        try:
            # Is there already a loop *running in this thread*?
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():                       # ← we are inside a loop
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                return ex.submit(_run_coro_in_new_loop, coro).result()
        else:                                                # ← no loop → normal path
            return asyncio.run(coro)


    # ------------------------------------------------------------------
    async def act_async(self, observation: str) -> str:
        self._value_cache.clear()
        root_hist: List[str] = list(self._history)
        root_hist.append(f"Obs: `{observation}`")
        # root_hist: List[str] = [f"Obs: {observation}"]

        client, dep = get_async_client(self.model)
        async with client:
            acts = await self._propose_actions_async(observation, root_hist, client, dep)
            # if not acts:
                # raise ValueError("propose_actions returned empty list")
            if not acts:
                return ""

            tasks = [
                self._q_value_async(observation, a, self.search_depth - 1, root_hist, client, dep)
                for a in acts
            ]
            q = list(await asyncio.gather(*tasks))

        if self.verbose:
            self._v(1, f"Root Q-values: {[(a, round(v,3)) for a,v in q]}")

        best = max(v for _, v in q)
        best_acts = [a for a, v in q if abs(v - best) <= 1e-6]
        # choice = random.choice(best_acts)
        choice = best_acts[0]
        self.last_thought = f"chosen {choice} (Q={best:+.3f})"
        # self._history.append(f"Obs: {observation} → Act: {choice}")
        self._history.append(f"Obs: `{observation}`")
        self._history.append(f"Act: `{choice}`")
        return choice

    # ------------------------------------------------------------------
    async def _q_value_async(
        self,
        observation: str,
        action: str,
        depth: int,
        hist: Sequence[str],
        client,
        dep,
    ) -> Tuple[str, float]:
        hist_a = [*hist, f"Act: `{action}`"]
        nxt, rew, done = await self._simulate_step_async(
            observation, action, hist_a, client, dep
        )
        hist_n = [*hist_a, f"Obs: {nxt}"]

        # ── if the transition ends the episode there is no future value ──
        if done:
            tail = 0.0
        elif depth <= 0:
            tail = await self._estimate_value_async(nxt, hist_n, client, dep)
        else:
            tail = await self._estimate_recursive_async(
                nxt, depth - 1, hist_n, client, dep
            )
        q = rew + self.discount * tail + self.step_penalty
        self._v(2, f"Q({action})={q:+.3f} | hist size={len(hist)}")
        return action, q

    # ------------------------------------------------------------------
    # LLM helpers (history-aware)
    # ------------------------------------------------------------------
    async def _propose_actions_async(self, obs, hist, client, dep):
        key = ("prop", obs, tuple(hist))
        if obs in self._terminal_obs:
            self._value_cache[key] = []
            return []
        if key in self._value_cache:
            return self._value_cache[key]
        
        # env_examples = _REACT_PROMPT_PREAMBLE + _EXAMPLES_REACT + '\nHere is the task.\n' #+ self.env_description
        history_lines = "\n".join(hist) if hist else "<empty>"

        prompt = f"""
You are an next best action proposing agent, task with solving the given environment defined below optimally. Your task is to propose up to {self.branch_factor} most likely next best unique actions to try next that make the agent solve the environment task optimally.

Environment description:
```
{self.env_description}
```

Atomic facts that help to predict next state value / next reward accurately (at beginning of episode):
```
{list(self._facts)}
```

Current Observation:
```
{obs}
```

Recent history (old→new):
```
{history_lines}
```

You now see Observation: `{obs}`. Now reason through (using the atomic facts, and recent obervation and action history), then give propose up to {self.branch_factor} most likely next best unique actions to try next that make the agent solve the environment task optimally, each from {self.allowed_actions}. You will call the function `propose_actions` to do this.
        """
        self._v(2, textwrap.indent(prompt, "PROP> ")[:400])
        msg = [
            {"role": "system", "content": "You must call propose_actions."},
            {"role": "user", "content": prompt},
        ]
        if self.verbose >= 3:
            print(f'PROMPT PROPOSE ACTIONS\n\n{prompt}\n\n')
        rep = await async_chat_completion(
            msg,
            client=client,
            deployment=dep,
            tools=self._planning_tools,
            tool_choice={"type": "function", "function": {"name": "propose_actions"}},
            temperature=self.temperature,
        )
        if self.verbose >= 3:
            self._v(3, f"RAW: {rep}")
        args = json.loads(rep.tool_calls[0].function.arguments)
        acts = [a.strip().lower() for a in args.get("actions", [])]
        acts = list(dict.fromkeys(acts))[: self.branch_factor]
        if self.verbose >= 3:
            print(acts)
        self._value_cache[key] = acts
        return acts

    async def _simulate_step_async(self, obs, act, hist, client, dep):
        key = ("sim", obs, act, tuple(hist))
        if key in self._value_cache:
            return self._value_cache[key]
        
        history_lines = "\n".join(hist) if hist else "<empty>"

        prompt = f"""
You are a latent world model for the given environment defined below. Given the current observation and an action, predict: the next (perhaps latent) observation, immediate reward and done flag (whether the resulting state ends the episode). You must be as accurate as possible, as your output is used as a planner to solve the given environment optimally.

Environment description:
```
{self.env_description}
```

Atomic facts that help to predict next state value / next reward accurately (at beginning of episode):
```
{list(self._facts)}
```

Current Observation:
```
{obs}
```

Recent history (old→new):
```
{history_lines}
```

Given action to simulate the next observation and reward for:
```
{act}
```

You now see Observation: `{obs}`. Now reason through (using the atomic facts, and recent obervation and action history, and recent obervation and action history), and predict the next (perhaps latent) observation, immediate reward, and done flag (whether the resulting state ends the episode) after taking the given action of `{act}`. You must be as accurate as possible (for the predicted reward, and ensure your predicted next observation has enough observation information to predict future rewards for the given task in the given environment), as your output is used as a planner to solve the given environment optimally. You will call the function `simulate_step` to do this.
"""
        self._v(2, textwrap.indent(prompt, "SIM> ")[:400])
        msg = [
            {"role": "system", "content": "You must call simulate_step."},
            {"role": "user", "content": prompt},
        ]
        if self.verbose >= 3:
            print(f'PROMPT LATENT WORLD MODEL SIMULATE\n\n{prompt}\n\n')
        rep = await async_chat_completion(
            msg,
            client=client,
            deployment=dep,
            tools=self._planning_tools,
            tool_choice={"type": "function", "function": {"name": "simulate_step"}},
            temperature=0.0,  # deterministic
        )
        if self.verbose >= 3:
            self._v(3, f"RAW: {rep}")
        a = json.loads(rep.tool_calls[0].function.arguments)
        nxt, rew, done = a.get("next_observation", "<unk>"), float(a.get("reward", 0.0)), bool(a.get("done", False))
        if done:
            self._terminal_obs.add(nxt)
            if self.verbose >= 3:
                print(f"SIMULATE DONE: {obs} → {nxt} | {rew} | {done}")
        self._value_cache[key] = (nxt, rew, done)
        return nxt, rew, done

    async def _estimate_value_async(self, obs, hist, client, dep):
        key = ("val", obs, tuple(hist))
        # terminal state ⇒ by definition no future reward
        if obs in self._terminal_obs:
            self._value_cache[key] = 0.0
            return 0.0

        if key in self._value_cache:
            return self._value_cache[key]

        history_lines = "\n".join(hist) if hist else "<empty>"

        prompt = f"""
You are a state value function estimator for the given environment defined below. You must predict the current cumulative future reward from the current (perhaps latent) observation. You must be as accurate as possible, as your output is used as a planner to solve the given environment optimally. The environment's discount factor is {self.discount}.

Environment description:
```
{self.env_description}
```

Atomic facts that help to predict next state value / next reward accurately (at beginning of episode):
```
{list(self._facts)}
```

Current Observation (to predict the current cumulative future reward for):
```
{obs}
```

Recent history (old→new):
```
{history_lines}
```

You now see Observation: `{obs}`. Now reason through (using the atomic facts, and recent obervation and action history, and recent obervation and action history), and predict the current cumulative future reward from the current (perhaps latent) observation. You must be as accurate as possible, as your output is used as a planner to solve the given environment optimally. The environment's discount factor is {self.discount}. You will call the function `estimate_value` to do this.
"""
        self._v(2, textwrap.indent(prompt, "VAL> ")[:400])
        msg = [
            {"role": "system", "content": "You must call estimate_value."},
            {"role": "user", "content": prompt},
        ]
        if self.verbose >= 3:
            print(f'PROMPT ESTIMATE VALUE\n\n{prompt}\n\n')
        rep = await async_chat_completion(
            msg,
            client=client,
            deployment=dep,
            tools=self._planning_tools,
            tool_choice={"type": "function", "function": {"name": "estimate_value"}},
            temperature=0.0,
        )
        if self.verbose >= 3:
            self._v(3, f"RAW: {rep}")
        val = float(json.loads(rep.tool_calls[0].function.arguments).get("value", 0.0))
        self._value_cache[key] = val
        return val

    async def _estimate_recursive_async(self, obs, depth, hist, client, dep):
        # known terminal ⇒ value 0
        if obs in self._terminal_obs:
            return 0.0
        if depth <= 0:
            return await self._estimate_value_async(obs, hist, client, dep)
        acts = await self._propose_actions_async(obs, hist, client, dep)
        if not acts:                     # nothing to expand
            return 0.0
        best = -float("inf")
        for a in acts:
            nxt, rew, done = await self._simulate_step_async(
                obs, a, hist + [f"Act: {a}"], client, dep
            )

            if done:
                # nothing beyond a terminal node
                val = rew
            else:
                val = rew + self.discount * await self._estimate_recursive_async(
                    nxt,
                    depth - 1,
                    hist + [f"Act: {a}", f"Obs: {nxt}"],
                    client,
                    dep,
                )
            best = max(best, val)
        return best

import unittest
from unittest.mock import patch, AsyncMock, MagicMock
import asyncio
import json
import math
from recall.envs.oracle.frozenlake import FrozenLakeEnv
import numpy as np

# Import your agent class
# from your_module import LLMWorldModelPlanningAgentAsync # Assuming your code is in your_module.py

# For standalone testing, copy the agent class here or import correctly
# ... (paste your LLMWorldModelPlanningAgentAsync class definition here) ...


class TestLLMPlannerFrozenLake(unittest.TestCase):

    def setUp(self):
        self.env_description = """
Frozen Lake Environment (4x4 grid):
S F F F
F H F H
F F F H
H F F G
S: Start, F: Frozen, H: Hole (penalty), G: Goal (reward).
Actions: "left", "down", "right", "up".
Movement is deterministic. Going off-grid results in staying in the same state.
Goal reward: +1. Hole penalty: -1. Every other step there is no reward (r=0.0).

Observation format: S{row}{col} (e.g., S00 for row 0, col 0).
"""
        self.allowed_actions = ("left", "down", "right", "up")
        self.agent_step_penalty = -0.02
        self.agent_discount = 0.99 # Example, match your agent's discount

        self.agent = LLMWorldModelPlanningAgentAsync(
            model="gpt-4o", # Mocked, so doesn't matter much
            env_description=self.env_description,
            allowed_actions=self.allowed_actions,
            search_depth=2, # Or whatever default you test
            branch_factor=4,
            discount=self.agent_discount,
            step_penalty=self.agent_step_penalty,
            verbose=0 # Keep tests quiet unless debugging a specific test
        )
        
        # Initialize a simple FrozenLakeEnv to get state details and true Q-values
        self.fl_env = FrozenLakeEnv(step_penalty=self.agent_step_penalty, hole_penalty=-1.0, goal_reward=1.0)
        self.true_q_values = self.fl_env.get_true_q_values(gamma=self.agent_discount)

        # Mock client and deployment for async_chat_completion
        self.mock_client = AsyncMock()
        self.mock_deployment = "test_deployment"
        
        # Ensure get_async_client is patched if it's used directly by methods under test
        # If it's only used in act_async, and we test _q_value_async directly, it might not be needed here
        # For simplicity, assuming _q_value_async takes client and dep as args

    def _create_llm_response(self, tool_name, arguments):
        # Helper to create a mock LLM response structure
        response_mock = MagicMock()
        response_mock.tool_calls = [MagicMock()]
        response_mock.tool_calls[0].function.name = tool_name
        response_mock.tool_calls[0].function.arguments = json.dumps(arguments)
        return response_mock

    # Test 1: _simulate_step_async logic with mocked LLM
    @patch('__main__.async_chat_completion', new_callable=AsyncMock) # Or your_module.async_chat_completion
    async def test_simulate_step_parsing(self, mock_chat_completion):
        obs = "S00"
        act = "right"
        hist = ["Obs: S00"]
        
        # Expected LLM output for simulate_step
        sim_args = {"thought": "Moving right from S00", "next_observation": "S01", "reward": 0.0, "done": False}
        mock_chat_completion.return_value = self._create_llm_response("simulate_step", sim_args)

        nxt, rew, done = await self.agent._simulate_step_async(obs, act, hist, self.mock_client, self.mock_deployment)

        self.assertEqual(nxt, "S01")
        self.assertEqual(rew, 0.0)
        self.assertFalse(done)
        # Check that the cache was populated
        self.assertIn(("sim", obs, act, tuple(hist)), self.agent._value_cache)

    # Test 2: _estimate_value_async logic with mocked LLM
    @patch('__main__.async_chat_completion', new_callable=AsyncMock)
    async def test_estimate_value_parsing(self, mock_chat_completion):
        obs = "S32" # One step from goal
        hist = ["Obs: S00", "Act: right", "Obs: S01", "..."] # Dummy history
        
        # Expected LLM output for estimate_value
        val_args = {"thought": "S32 is one step from goal S33. Goal gives +1.", "value": 0.99} # V(S32) approx gamma * R_goal
        mock_chat_completion.return_value = self._create_llm_response("estimate_value", val_args)

        value = await self.agent._estimate_value_async(obs, hist, self.mock_client, self.mock_deployment)
        
        self.assertAlmostEqual(value, 0.99, places=2)
        self.assertIn(("val", obs, tuple(hist)), self.agent._value_cache)

    # Test 3: _propose_actions_async logic with mocked LLM
    @patch('__main__.async_chat_completion', new_callable=AsyncMock)
    async def test_propose_actions_parsing(self, mock_chat_completion):
        obs = "S00"
        hist = []
        prop_args = {"thought": "From S00, can go right or down.", "actions": ["right", "down"]}
        mock_chat_completion.return_value = self._create_llm_response("propose_actions", prop_args)

        actions = await self.agent._propose_actions_async(obs, hist, self.mock_client, self.mock_deployment)
        self.assertEqual(actions, ["right", "down"])

    # Test 4: _q_value_async calculation - Terminal state
    async def test_q_value_terminal_state(self):
        # Mock _simulate_step_async to return a terminal state
        # The agent.step_penalty is ALREADY INCLUDED in the reward by _simulate_step_async
        # if the LLM is prompted to predict total immediate reward including penalties.
        # However, the current _q_value_async adds step_penalty explicitly.
        # For this test, assume LLM returns environment reward, and _q_value_async adds penalty.

        mock_simulate = AsyncMock(return_value=("S33", 1.0, True)) # (nxt, env_reward, done)
        self.agent._simulate_step_async = mock_simulate

        obs = "S32" # State before goal
        act = "down" # Action leading to goal
        depth = 1 
        hist = ["Obs: S32"]

        # expected_q = env_reward + agent_step_penalty
        # Here, env_reward = 1.0, self.agent.step_penalty = -0.02
        # Q = 1.0 + (-0.02) = 0.98
        expected_q = 1.0 + self.agent.step_penalty 

        _, q_val = await self.agent._q_value_async(obs, act, depth, hist, self.mock_client, self.mock_deployment)

        self.assertAlmostEqual(q_val, expected_q, places=3)
        mock_simulate.assert_called_once()

    # Test 5: _q_value_async calculation - Non-terminal, depth 0 (uses _estimate_value_async)
    async def test_q_value_non_terminal_depth_0(self):
        # Mock _simulate_step_async
        # Assume moving from S00 right to S01. Env reward = 0.
        mock_simulate = AsyncMock(return_value=("S01", 0.0, False)) # (nxt, env_reward, done)
        self.agent._simulate_step_async = mock_simulate

        # Mock _estimate_value_async for V(S01)
        # Let's say the true V(S01) from our calculation is ~0.6 (example)
        # If we want to test the formula, we provide a V_s_prime
        v_s_prime_estimate = 0.65 
        mock_estimate_value = AsyncMock(return_value=v_s_prime_estimate)
        self.agent._estimate_value_async = mock_estimate_value
        
        obs = "S00"
        act = "right"
        depth = 0 # This will trigger _estimate_value_async
        hist = ["Obs: S00"]
        
        # Q = env_rew + discount * V(s') + step_penalty
        # Q = 0.0 + 0.99 * 0.65 + (-0.02)
        expected_q = 0.0 + self.agent.discount * v_s_prime_estimate + self.agent.step_penalty

        _, q_val = await self.agent._q_value_async(obs, act, depth, hist, self.mock_client, self.mock_deployment)
        
        self.assertAlmostEqual(q_val, expected_q, places=3)
        mock_simulate.assert_called_once_with(obs, act, [*hist, f"Act: `{act}`"], self.mock_client, self.mock_deployment)
        mock_estimate_value.assert_called_once() # Check it was called for S01

    # Test 6: _q_value_async vs True Q-value for a specific (s,a) with "perfect" LLM mocks
    # This is a more integrated test where mocks emulate perfect FrozenLake knowledge
    async def test_q_value_s00_right_vs_true_q(self):
        self.agent.search_depth = 1 # For a simple 1-step lookahead Q value

        s00_int = self.fl_env._to_s(0,0)
        s01_int = self.fl_env._to_s(0,1)
        
        # "Perfect" LLM Mocks for S00 -> right -> S01
        # _simulate_step_async for (S00, right)
        # Reward from fl_env.P already includes step_penalty based on our setup
        # So, LLM should predict this "effective" reward
        # If LLM predicts pure env reward (0 for F->F), then agent adds step_penalty
        
        # Let's assume LLM predict pure environment reward:
        # S00 -> right -> S01 (Frozen). Env reward = 0. Done = False.
        sim_next_obs, sim_env_reward, sim_done = "S01", 0.0, False
        self.agent._simulate_step_async = AsyncMock(return_value=(sim_next_obs, sim_env_reward, sim_done))

        # _estimate_value_async for V(S01) (as depth is search_depth - 1 = 0)
        # V(s') is Sum_a' [pi(a'|s') * Q(s',a')] which means max_a' Q_true(s',a')
        # The true V(s') = max_a Q_true(s', a'). Note: Q_true incorporates step_penalty AND discount.
        # So, _estimate_value_async should predict V(s') that will be discounted again by agent.
        # This is tricky. The LLM's V(s) should be E[ sum_{k=0 to inf} gamma^k * (r_{t+k+1} + step_penalty) ]
        # Our true_q_values are R_immediate + gamma * V_true(s'), where R_immediate has step_penalty.
        # V_true(s') = max_a Q_true(s',a').

        # The value returned by _estimate_value_async is V(s') for the Bellman eq:
        # Q(s,a) = (env_r + step_penalty) + discount * V(s')
        # So, V(s') should be the expected sum of future discounted (env_rewards + step_penalties) from s'.
        # This is exactly what value iteration computes for V values if rewards include step_penalty.
        
        V_s_prime_true = np.max(self.true_q_values[s01_int, :]) / self.agent.discount # True V(s') if Q=R+gammaV
                                                                                  # where R includes step penalty.
                                                                                  # If the _estimate_value_async is supposed to return
                                                                                  # the value that is directly plugged into
                                                                                  # Q = r_sim + discount * V_est + penalty_step,
                                                                                  # then V_est = V_true(s').
        
        # Let's assume _estimate_value_async is asked to predict the sum of discounted future rewards *excluding* the immediate step_penalty
        # (as step_penalty is added explicitly in _q_value_async).
        # And it should be the sum of discounted *environmental* rewards.
        # This means the LLM's value function is conceptually V_env(s) = E[sum gamma^k r_env_{t+k+1}].
        # Q(s,a) = r_env + step_penalty + gamma * V_env(s')
        # The true Q values we calculated have R = r_env + step_penalty.
        # Q_true(s,a) = (r_env + step_penalty) + gamma * max_{a'} Q_true(s', a')
        # Let V_true(s') = max_{a'} Q_true(s',a').
        # Then V_est (from LLM) should be V_true(s') / (something related to step penalty and discount)? This gets complex.

        # Simpler: if _estimate_value_async is a perfect value function for the MDP where
        # rewards are (env_reward + step_penalty), then it should return V_true(s').
        v_s_prime_from_llm_estimate = np.max(self.true_q_values[s01_int, :]) # This is V*(S01) where rewards included step_penalty
        self.agent._estimate_value_async = AsyncMock(return_value=v_s_prime_from_llm_estimate)
        
        obs, act = "S00", "right"
        depth = self.agent.search_depth -1 # Will be 0 for this test, calling _estimate_value_async

        # Agent calculates: sim_env_reward + self.agent.discount * v_s_prime_from_llm_estimate + self.agent.step_penalty
        # q_val = 0.0 + self.agent.discount * V*(S01) + self.agent.step_penalty
        
        true_q_s00_right = self.true_q_values[s00_int, self.fl_env.rev_action_map[act]]
        
        _, q_val_agent = await self.agent._q_value_async(obs, act, depth, ["Obs: S00"], self.mock_client, self.mock_deployment)

        # The agent's Q formula is: rew_sim + self.discount * tail + self.step_penalty
        # If rew_sim is env_reward, and tail is V*(s'), then:
        # q_val_agent = env_reward(S00,right -> S01) + discount * V*(S01) + step_penalty
        # V*(S01) itself = max_a' [ (env_reward(S01,a' -> S'') + step_penalty) + discount * V*(S'') ]
        # The true_q_s00_right = (env_reward(S00,right -> S01) + step_penalty) + discount * V*(S01)
        # So, if LLM returns env_reward for sim, and V*(s') for value estimate, the agent Q matches true Q.
        self.assertAlmostEqual(q_val_agent, true_q_s00_right, places=3, 
                             msg=f"Agent Q: {q_val_agent}, True Q: {true_q_s00_right}")


    # Helper to run async tests
    def run_async_test(self, coro):
        loop = asyncio.get_event_loop_policy().new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()

    # Override run method to handle async tests if not using a special test runner
    def __getattribute__(self, name):
        attr = object.__getattribute__(self, name)
        if name.startswith('test_') and asyncio.iscoroutinefunction(attr):
            return lambda: self.run_async_test(attr())
        return attr


# In your TestLLMPlannerFrozenLake class:

# ... (existing setUp and mocked tests from your provided code) ...

    # --- LIVE LLM TESTS ---
    # These tests make actual calls to the LLM.
    # Ensure your API keys are configured in the environment.
    # Run with: RUN_LIVE_LLM_TESTS=true python your_test_file.py

    # @unittest.skipUnless(RUN_LIVE_LLM_TESTS and API_CONFIGURED, 
                        #  "Skipping live LLM tests. Set RUN_LIVE_LLM_TESTS=true and configure API keys.")
    async def test_llm_live_propose_actions_from_s00(self):
        """
        Tests if the LLM correctly proposes actions from the start state S00.
        It should propose 'right' and 'down', and NOT 'up' or 'left'.
        """
        # self.agent.verbose = 3 # Uncomment for detailed logs during this test
        obs = "S00"
        hist = [] # No history at the start

        # The agent's methods will internally get a client and deployment
        client, dep = get_async_client(self.agent.model) 
        async with client: # We manage client lifecycle here for clarity
            proposed_actions = await self.agent._propose_actions_async(obs, hist, client, dep)

        print(f"\n[Live Test] Proposed actions from S00: {proposed_actions}")
        
        self.assertIn("down", proposed_actions, "LLM should propose 'down' from S00.")
        self.assertIn("right", proposed_actions, "LLM should propose 'right' from S00.")
        self.assertNotIn("up", proposed_actions, "LLM should NOT propose 'up' from S00.")
        self.assertNotIn("left", proposed_actions, "LLM should NOT propose 'left' from S00.")
        
        # Additional checks
        self.assertTrue(all(a in self.agent.allowed_actions for a in proposed_actions),
                        "All proposed actions should be in allowed_actions.")
        self.assertEqual(len(proposed_actions), len(set(proposed_actions)), 
                         "Proposed actions should be unique.")
        if self.agent.branch_factor > 0: # only if we expect actions
             self.assertGreater(len(proposed_actions), 0, "Should propose at least one action if possible.")


    # @unittest.skipUnless(RUN_LIVE_LLM_TESTS and API_CONFIGURED, "Skipping live LLM tests.")
    async def test_llm_live_simulate_step_s00_invalid_action_up(self):
        """
        Tests LLM's simulation of an invalid action ('up') from S00.
        Expected: Stay in S00, 0 environment reward, not done.
        """
        # self.agent.verbose = 3
        obs = "S00"
        act = "up" 
        # History includes the current observation as per agent's logic before calling _simulate_step_async
        hist = [f"Obs: `{obs}`"] 

        expected_next_obs = "S00"
        expected_env_reward = 0.0 
        expected_done = False

        client, dep = get_async_client(self.agent.model)
        async with client:
            next_obs, reward, done = await self.agent._simulate_step_async(obs, act, hist, client, dep)
        
        print(f"\n[Live Test] Simulate (S00, up): next_obs='{next_obs}', reward={reward}, done={done}")
        self.assertEqual(next_obs, expected_next_obs, "LLM simulation for (S00, up) next_obs is incorrect.")
        self.assertAlmostEqual(reward, expected_env_reward, places=2, 
                               msg="LLM simulation for (S00, up) reward is incorrect.")
        self.assertEqual(done, expected_done, "LLM simulation for (S00, up) done flag is incorrect.")

    # @unittest.skipUnless(RUN_LIVE_LLM_TESTS and API_CONFIGURED, "Skipping live LLM tests.")
    async def test_llm_live_simulate_step_s00_valid_action_right(self):
        """
        Tests LLM's simulation of a valid action ('right') from S00.
        Expected: Move to S01, 0 environment reward, not done.
        """
        # self.agent.verbose = 3
        obs = "S00"
        act = "right"
        hist = [f"Obs: `{obs}`"]

        expected_next_obs = "S01"
        expected_env_reward = 0.0
        expected_done = False

        client, dep = get_async_client(self.agent.model)
        async with client:
            next_obs, reward, done = await self.agent._simulate_step_async(obs, act, hist, client, dep)

        print(f"\n[Live Test] Simulate (S00, right): next_obs='{next_obs}', reward={reward}, done={done}")
        self.assertEqual(next_obs, expected_next_obs, "LLM simulation for (S00, right) next_obs is incorrect.")
        self.assertAlmostEqual(reward, expected_env_reward, places=2,
                               msg="LLM simulation for (S00, right) reward is incorrect.")
        self.assertEqual(done, expected_done, "LLM simulation for (S00, right) done flag is incorrect.")

    # @unittest.skipUnless(RUN_LIVE_LLM_TESTS and API_CONFIGURED, "Skipping live LLM tests.")
    async def test_llm_live_simulate_step_to_goal(self):
        """
        Tests LLM's simulation of an action leading to the Goal.
        Expected: Goal state, +1.0 environment reward, done=True.
        """
        # self.agent.verbose = 3
        obs = "S32" # State (3,2), to the left of Goal at (3,3)
        act = "right"
        hist = [f"Obs: `{obs}`"]
        
        # The LLM might return "S33", "S33G", "S33 (Goal State)", etc.
        # We'll check if "S33" is part of the returned next_observation.
        expected_next_obs_substring = "S33" 
        expected_env_reward = 1.0 
        expected_done = True
        
        self.agent._terminal_obs.clear() # Ensure clean slate for terminal observation tracking

        client, dep = get_async_client(self.agent.model)
        async with client:
            next_obs, reward, done = await self.agent._simulate_step_async(obs, act, hist, client, dep)

        print(f"\n[Live Test] Simulate (S32, right) to Goal: next_obs='{next_obs}', reward={reward}, done={done}")
        self.assertIn(expected_next_obs_substring, next_obs, 
                      f"LLM simulation to goal next_obs '{next_obs}' should contain '{expected_next_obs_substring}'.")
        self.assertAlmostEqual(reward, expected_env_reward, places=2,
                               msg="LLM simulation to goal reward is incorrect.")
        self.assertEqual(done, expected_done, "LLM simulation to goal done flag is incorrect.")
        self.assertIn(next_obs, self.agent._terminal_obs, 
                      "Goal state should be added to agent's terminal observation set.")


    # @unittest.skipUnless(RUN_LIVE_LLM_TESTS and API_CONFIGURED, "Skipping live LLM tests.")
    async def test_llm_live_simulate_step_to_hole(self):
        """
        Tests LLM's simulation of an action leading to a Hole.
        Expected: Hole state, -1.0 environment reward, done=True.
        """
        # self.agent.verbose = 3
        obs = "S01" # State (0,1), above Hole at (1,1)
        act = "down"
        hist = [f"Obs: `{obs}`"]
        self.agent._facts = deque(['Hole at S11)'], maxlen=self.agent.fact_buffer_len)

        expected_next_obs_substring = "S11" 
        expected_env_reward = -1.0 
        expected_done = True

        self.agent._terminal_obs.clear()

        client, dep = get_async_client(self.agent.model)
        async with client:
            next_obs, reward, done = await self.agent._simulate_step_async(obs, act, hist, client, dep)

        print(f"\n[Live Test] Simulate (S01, down) to Hole: next_obs='{next_obs}', reward={reward}, done={done}")
        self.assertIn(expected_next_obs_substring, next_obs,
                      f"LLM simulation to hole next_obs '{next_obs}' should contain '{expected_next_obs_substring}'.")
        self.assertAlmostEqual(reward, expected_env_reward, places=2,
                               msg="LLM simulation to hole reward is incorrect.")
        self.assertEqual(done, expected_done, "LLM simulation to hole done flag is incorrect.")
        self.assertIn(next_obs, self.agent._terminal_obs,
                      "Hole state should be added to agent's terminal observation set.")

    # @unittest.skipUnless(RUN_LIVE_LLM_TESTS and API_CONFIGURED, "Skipping live LLM tests.")
    async def test_llm_live_estimate_value_near_goal(self):
        """
        Tests LLM's value estimation for a state one step from the Goal (S32).
        V*(S32) = max_a Q*(S32,a).
        Q*(S32, 'right') = (env_reward_goal + step_penalty) + discount * V*(S33_Goal=0)
                         = (1.0 + (-0.02)) + 0.99 * 0 = 0.98.
        So, V*(S32) should be approximately 0.98.
        """
        # self.agent.verbose = 3
        obs = "S32"
        hist = [f"Obs: `{obs}`"] # Simplified history for direct value estimation

        s32_int = self.fl_env._to_s(3,2)
        expected_v_star_s32 = np.max(self.true_q_values[s32_int, :]) # Should be ~0.98

        self.agent._terminal_obs.clear() # Ensure S32 isn't accidentally marked terminal

        client, dep = get_async_client(self.agent.model)
        async with client:
            estimated_value = await self.agent._estimate_value_async(obs, hist, client, dep)
        
        print(f"\n[Live Test] Estimate value V(S32): Got {estimated_value}, True V*(S32) is {expected_v_star_s32:.3f}")
        self.assertAlmostEqual(estimated_value, expected_v_star_s32, delta=0.15, # Allow some LLM estimation error
                               msg="LLM value estimate for state near goal is significantly off.")

    # @unittest.skipUnless(RUN_LIVE_LLM_TESTS and API_CONFIGURED, "Skipping live LLM tests.")
    async def test_llm_live_estimate_value_goal_itself(self):
        """
        Tests LLM's value estimation for the Goal state S33.
        Expected: Value should be 0.0 as it's terminal.
        """
        # self.agent.verbose = 3
        obs = "S33" # A goal state representation
        hist = [f"Obs: `{obs}`"]

        expected_value = 1.0
        
        # Agent's _estimate_value_async has a fast path: if obs in self._terminal_obs, returns 0.
        # To test the LLM's direct response, ensure obs is not in _terminal_obs cache.
        self.agent._terminal_obs.clear() 

        client, dep = get_async_client(self.agent.model)
        async with client:
            estimated_value = await self.agent._estimate_value_async(obs, hist, client, dep)
        
        print(f"\n[Live Test] Estimate value V(S33 - Goal): Got {estimated_value}")
        self.assertAlmostEqual(estimated_value, expected_value, delta=0.1, # LLM might not be perfectly zero
                               msg="LLM value estimate for a goal state should be close to 0.")

    # @unittest.skipUnless(RUN_LIVE_LLM_TESTS and API_CONFIGURED, "Skipping live LLM tests.")
    async def test_llm_live_full_act_from_s00(self):
        """
        Tests the agent's act_async method from S00 for one step.
        It should choose a reasonable action (right or down).
        """
        self.agent.search_depth = 1 # Simplify for easier debugging: 1-step lookahead
        self.agent.verbose = 1      # See root Q-values
        obs = "S00"
        self.agent._history.clear() # Reset history for a fresh start

        s00_int = self.fl_env._to_s(0,0)
        true_q_s00 = self.true_q_values[s00_int, :]
        best_true_action_indices = np.where(true_q_s00 == np.max(true_q_s00))[0]
        expected_optimal_actions = [self.fl_env.action_map[i] for i in best_true_action_indices]
        
        print(f"\n[Live Test] True optimal actions from S00: {expected_optimal_actions} (Q-values: {true_q_s00})")

        # act_async handles its own client management
        chosen_action = await self.agent.act_async(obs)

        print(f"[Live Test] act_async from S00 chose: {chosen_action}. Agent thought: {self.agent.last_thought}")
        
        self.assertIn(chosen_action, expected_optimal_actions,
                      f"Agent chose '{chosen_action}' from S00, but expected one of {expected_optimal_actions}.")
        # Specifically address the user's observed issue:
        self.assertNotIn(chosen_action, ["up", "left"],
                         "Agent should NOT choose 'up' or 'left' from S00 as the best action.")

if __name__ == '__main__':
    # To run these tests, you'd need to place your LLMWorldModelPlanningAgentAsync class
    # definition in this file or import it correctly. For demonstration, I've used
    # @patch('__main__.async_chat_completion', ...) assuming it's in the same file.
    # If it's in `your_module.py`, it would be @patch('your_module.async_chat_completion', ...).
    
    # You would also need the LLMWorldModelPlanningAgentAsync class definition available.
    # For this example to be self-contained, I'll stub it.
    
    # # --- STUB Agent Class (replace with your actual class) ---
    # @dataclass
    # class LLMReactFactExtractionAgent: # Minimal stub for parent
    #     model: str = "test"
    #     allowed_actions: Tuple[str, ...] = tuple()
    #     env_description: str = ""
    #     temperature: float = 0.0
    #     _history: List[str] = field(default_factory=list)
    #     _facts: Set[str] = field(default_factory=set)
    #     _tools: List[Any] = field(default_factory=list)
    #     def __post_init__(self): pass
    
    # # Paste your LLMWorldModelPlanningAgentAsync class here
    # # For brevity, I'm not pasting the whole class again. Assume it's defined above.
    # # Ensure to adjust @patch decorators if class is in a different module. e.g. 'your_agent_module.async_chat_completion'

    # --- End STUB ---

    unittest.main(argv=['first-arg-is-ignored'], exit=False)