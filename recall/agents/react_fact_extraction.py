# Copyright © 2025 Samuel Holt. All rights reserved.
# No licence is granted to copy, use, modify, distribute, or create derivative
# works of this file in any form, except with explicit written permission from
# the copyright holder.
from __future__ import annotations
"""
LLM ReAct **and** Reflexion agents (+ verbose evaluator)
=======================================================

This single module now contains **both baselines**:

* `LLMReactAgent` – simple ReAct policy (unchanged in behaviour).
* `LLMReflexionAgent` – adds per‑episode *reflection*: after each episode it
  summarises the trajectory into a concise lesson which is prepended to future
  prompts.  A FIFO buffer keeps the **k = 20** most recent lessons.
* `evaluate_agent()` – unchanged call‑signature but now auto‑detects optional
  `record_transition()` and `reflect()` hooks, so it works seamlessly with
  either agent.

Usage
-----
```python
from text_frozen_lake_env import TextFrozenLakeEnv
from llm_agents import LLMReactAgent, LLMReflexionAgent, evaluate_agent

env = TextFrozenLakeEnv(size=4, hole_density=0.3, seed=0)
agent = LLMReflexionAgent()          # or LLMReactAgent()
metrics = evaluate_agent(env, agent, max_steps=5000, verbose=True)
```
"""

from collections import deque
from dataclasses import dataclass, field
from typing import List, Dict, Any, Deque, Tuple
import random
import logging
import matplotlib.pyplot as plt
from recall.agents.react import LLMReactAgent
import json

# ---------------------------------------------------------------------------
# External dependency (stub fallback for offline testing)
# ---------------------------------------------------------------------------
# try:
from recall.llm_utils import chat_completion  # type: ignore # noqa: F401
# except ImportError:  # pragma: no cover – offline stub
#     def chat_completion(messages: List[Dict[str, Any]], *, model: str = "gpt-4o", temperature: float = 0.3, max_tokens: int = 128) -> str:  # type: ignore
#         legal = {"up", "down", "left", "right"}
#         return random.choice(tuple(legal))

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
from recall.utils import tiny_logger

# ---------------------------------------------------------------------------
# Reflexion Agent – extends ReAct with per‑episode lesson synthesis
# ---------------------------------------------------------------------------

def _fact_extraction_tools() -> List[Dict[str, Any]]:
    """OpenAI function‑call schema for fact extraction."""
    return [
        {
            "type": "function",
            "function": {
                "name": "fact_extraction",
                "description": "Given the environment description, current sucessful/unsucessful trajectory and existing facts derived, extract atomic facts that you did not know already to help with predicting the next state value / next reward, such that if you had this fact you would have improved your prediction for the next state value, when being a world model (that is be able to complete the task optimally in the minimum number of steps, therefore extract key information that helps you).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "thought": {"type": "string", "description": "Agent's private reasoning."},
                        "new_facts": {
                            "type": "array",
                            "items": {"type": "string"},
                             "description": "Extract new atomic facts that you did not know already to help with predicting the next state value / next reward, such that if you had this fact you would have improved your prediction for the next state value, when being a world model (i.e. given a state, action an LLM can use these to predict the next state, next reward and terminal state). Make facts as concise as posssible. Optimize them for other agents reading and decision making given a current state. Never duplicate the facts if they already exist within our following fact set. Do not include any other text or reasoning, just the facts. If no new facts just return empty list."
                        },
                    },
                    "required": ["thought", "new_facts"],
                    "additionalProperties": False,
                },
                "strict": True,
            },
        },
        {
            "type": "function",
            "function": {
                "name": "fact_redundancy_remover",
                "description": "Remove any redundant facts that are already included in the list of all facts given to you. You will also always be given the environment description, therefore you can use that to help you remove any redundant facts. Always keep all exhaustive factual knowledge, just remove any duplicate facts, or redundant information already contained within the environment description, otherwise copy over the existing facts verbatim. You optimize the facts so they can be read by another LLM agent using them for being a world model of the environment (where the agent has to simulate given a state,action to predict the next state, next reward and terminal state).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "thought": {"type": "string", "description": "Private reasoning."},
                        "all_facts": {
                            "type": "array",
                            "items": {"type": "string"},
                             "description": "List of all facts exhuastively that you did not know already (not contained within the environment description) to help with predicting the next state value / next reward, such that if you had this fact you would have improved your prediction for the next state value, when being a world model. Remove any redudancy, otherwise copy over the existing facts verbatim. Do not include any other formatting text such as bullets or numbering, just the facts. If no new facts just return empty list."
                        },
                    },
                    "required": ["thought", "all_facts"],
                    "additionalProperties": False,
                },
                "strict": True,
            },
        },
    ]


@dataclass
class LLMReactFactExtractionAgent(LLMReactAgent):
    """ReAct + Reflexion: learns from past trajectories by synthesising lessons."""

    fact_buffer_len: int = 200  # keep last k lessons
    success_reward_threshold: float = 0.99  # reward > threshold ⇒ success

    _facts: Deque[str] = field(init=False)
    _trajectory: List[Tuple[str, str, float, str]] = field(init=False)  # (obs, act, reward, next_obs)
    _episode_reward: float = field(init=False, default=0.0)
    compress: bool = field(default=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        self._facts = deque(maxlen=self.fact_buffer_len)
        self._fact_extraction_tools = _fact_extraction_tools()
        self._tools += self._fact_extraction_tools
        self._trajectory = []
        # Extend prompt to include facts
        self.prompt_template = (
            "{env_description}\n\n"
            "Atomic facts that help to predict next state value / next reward accurately:\n{facts}\n\n"
            "You must use the ReAct pattern:\nThought: <reasoning>\nAction: <one of {actions}>\n\n"
            "Current Observation: {observation}\n"
            "Recent history (old→new):\n```\n{history}\n```\n\n"
            "You are now see Observation: {observation}\nNow reason through the next action you should take, think carefully and long, particularly looking over the atomic facts that you have previously learned, and then give one of the following actions of [{actions}].\n"
        )

    # ------------------------------------------------------------------
    # Overrides
    # ------------------------------------------------------------------
    def reset(self, observation: str) -> None:
        super().reset(observation)
        self._trajectory.clear()
        self._episode_reward = 0.0

    def act(self, observation: str) -> str:
        facts_text = "\n".join(f"- {l}" for l in self._facts) or "<none yet>"
        user_prompt = self.prompt_template.format(
            env_description=self.env_description,
            facts=facts_text,
            observation=observation,
            history="\n".join(self._history) or "<empty>",
            actions=", ".join(self.allowed_actions),
        )
        messages = [
            {"role": "system", "content": "You are a expert agent and must call the function react_step."},
            {"role": "user", "content": user_prompt},
        ]
        reply = chat_completion(
            messages,
            model=self.model,
            tools=self._tools,
            tool_choice={
                "type": "function",
                "function": {"name": "react_step"}
            },
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        # ---------------- Parse function call ----------------
        if len(reply.tool_calls) != 1:
            raise ValueError(f"LLM did not return a function call: {reply}")
        args_str = reply.tool_calls[0].function.arguments
        try:
            args = json.loads(args_str)
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON from LLM: {args_str}")

        thought = args.get("thought", "").strip()
        action = args.get("action", "").lower().strip()
        # if action not in self.allowed_actions:
        #     raise ValueError(f"LLM produced invalid action '{action}'.")

        self.last_thought = thought
        self._history.append(f"Obs: {observation} → Act: {action}")
        return action

    # ------------------------------------------------------------------
    # Trajectory handling – evaluator will call these hooks
    # ------------------------------------------------------------------
    def record_transition(self, obs: str, act: str, reward: float, next_obs: str, done: bool) -> None:
        self._trajectory.append((obs, act, reward, next_obs))
        self._episode_reward += reward


    def _call_llm_for_fact_extraction(self, trajectory_summary: str, facts: List[str]) -> List[str]:
        """Calls LLM to extract new facts from a trajectory."""
        fact_extraction_prompt = f"""
You are a LLM fact extraction agent. Operating in the following environment defined below. Your task is to extract atomic facts that you did not know already to help with predicting the next state value / next reward, such that if you had this fact you would have improved your prediction for the next state value, when being a world model (that is be able to complete the task optimally in the minimum number of steps, therefore extract key information that helps you).

Environment Description:
```
{self.env_description}
```

{trajectory_summary}

We already know and have the following facts (ensure you do not duplicate them) (at beginning of episode):
```
{facts}
```

Now respond with minimal new atomic facts (at beginning of episode) that you did not already know, for the rest of the states assume you areadly know them. Make facts as concise as posssible. Optimize them for other agents reading and decision making given a current state. Never duplicate the facts if they already exist within our following fact set. Do not include any other text or reasoning, just the facts. If no new facts just return empty string. Use function "fact_extraction" to do this now.
"""
        messages = [
            {"role": "system", "content": "You are a expert agent."},
            {"role": "user", "content": fact_extraction_prompt},
        ]
        reply = chat_completion(
            messages,
            model=self.model,
            tools=self._tools,
            tool_choice={
                "type": "function",
                "function": {"name": "fact_extraction"}
            },
            temperature=0.0,
            # max_tokens=self.max_tokens,
        )

        # ---------------- Parse function call ----------------
        if len(reply.tool_calls) != 1:
            raise ValueError(f"LLM did not return a function call: {reply}")
        args = json.loads(reply.tool_calls[0].function.arguments)
        new_facts = [a.strip().lower() for a in args.get("new_facts", [])]
        tiny_logger.log(f"LLM extracted new facts: {new_facts}")
        return new_facts

    def _call_llm_for_fact_compression(self, current_facts_summary: List[str]) -> List[str]:
        """Calls LLM to compress/remove redundancy from existing facts."""
        fact_extraction_prompt = f"""
Remove any redundant facts that are already included in the list of all facts given to you. You will also always be given the environment description, therefore you can use that to help you remove any redundant facts. Always keep all exhaustive factual knowledge, just remove any duplicate facts, or redundant information already contained within the environment description. You optimize the facts so they can be read by another LLM agent using them for being a world model of the environment (where the agent has to simulate given a state,action to predict the next state, next reward and terminal state). Remove any redudancy, otherwise copy over the existing facts verbatim.

Environment Description:
```
{self.env_description}
```

Facts (at beginning of episode):
```
{current_facts_summary}
```

List of all facts (at beginning of episode) that you did not know already (not contained within the environment description) to help with predicting the next state value / next reward, such that if you had this fact you would have improved your prediction for the next state value, when being a world model. Optimize them for other agents reading and decision making given a current state. Use function "fact_redundancy_remover" to do this now.
"""
        messages = [
            {"role": "system", "content": "You are a expert agent."},
            {"role": "user", "content": fact_extraction_prompt},
        ]
        reply = chat_completion(
            messages,
            model=self.model,
            tools=self._tools,
            tool_choice={
                "type": "function",
                "function": {"name": "fact_redundancy_remover"}
            },
            temperature=0.0,
            # max_tokens=self.max_tokens,
        )

        # ---------------- Parse function call ----------------
        if len(reply.tool_calls) != 1:
            raise ValueError(f"LLM did not return a function call: {reply}")
        args = json.loads(reply.tool_calls[0].function.arguments)
        all_facts = [a.strip().lower() for a in args.get("all_facts", [])]
        tiny_logger.log(f"LLM compressed facts to: {all_facts}")
        return all_facts

    def reflect(self) -> None:
        if not self._trajectory:
            return

        outcome = "SUCCESS" if self._episode_reward >= self.success_reward_threshold else "FAILURE"
        trajectory_lines = [
            f"{i}. Obs: {o} | Act: {a} | R: {r:+.0f} | Next: {n}"
            for i, (o, a, r, n) in enumerate(self._trajectory, 1)
        ]
        trajectory_str = "\n".join(trajectory_lines)
        trajectory_summary = (
            f"It is now the end of an episode, here is the episode trajectory:\n"
            f"```\nOutcome: {outcome} (total reward = {self._episode_reward:+.1f})\n"
            f"Trajectory:\n{trajectory_str}\n```"
        )
        tiny_logger.log(f"Reflecting on episode:\n\n{trajectory_summary}\n")
        # --- 1. Fact Extraction ---
        newly_extracted_facts = self._call_llm_for_fact_extraction(trajectory_summary, list(self._facts))
        if len(newly_extracted_facts) > 0:
            self._facts.appendleft(newly_extracted_facts)
        tiny_logger.log(f"All facts before compression: {list(self._facts)}")
        # Clear buffers for next episode
        if self.compress:
            if not self._facts: # No facts to compress
                tiny_logger.log("Compression skipped: No facts to compress.")
                # current_facts_summary for compression prompt
            else:
                raw_current_facts = list(self._facts)
                updated_facts = self._call_llm_for_fact_compression(raw_current_facts)
                tiny_logger.log(f"Updated Facts: {list(updated_facts)} | Previous Facts: {list(self._facts)}")
                
                self._facts.clear()
                # Repopulate with compressed facts, respecting maxlen of deque
                for fact in updated_facts: # Add to the right by default
                    if len(self._facts) < self.fact_buffer_len:
                        self._facts.append(fact)
                    else: # If buffer is full, it means older facts are pushed out by new ones
                        break 

            # if len(all_facts) > 0:
            #     self._facts = deque(all_facts, maxlen=self.fact_buffer_len)
        # Clear buffers for next episode
        self._trajectory.clear()
        self._episode_reward = 0.0

# S F F G
# F H . .
# . . . .
# . . . .

TEST_ENV_DESCRIPTION = """
Frozen Lake Environment (4x4 grid)
S: Start, F: Frozen, H: Hole (penalty), G: Goal (reward).
Actions: "left", "down", "right", "up".
Movement is deterministic. Going off-grid results in staying in the same state.
Goal reward: +1. Any other step incurs a small penalty implicitly or has 0 direct env reward. Falling into a Hole ends the episode with a large penalty.
"""


import unittest
from unittest.mock import patch, MagicMock, AsyncMock # If any part still uses async
from collections import deque
import os
import json
import asyncio


# Helper to create LLM response for function call (if needed for any mixed tests)
def _create_llm_tool_call_response(tool_name, arguments_dict):
    response_mock = MagicMock() # or an appropriate mock for your chat_completion's return type
    # Assuming reply structure like OpenAI's client:
    tool_call_mock = MagicMock()
    tool_call_mock.function.name = tool_name
    tool_call_mock.function.arguments = json.dumps(arguments_dict)
    response_mock.tool_calls = [tool_call_mock]
    return response_mock

ALFWORLD_ENV_DESCRIPTION_FOR_TEST = """
You are an agent in a household environment. You can navigate between rooms and interact with objects.
Common objects include: armchair, cabinet, drawer, dresser, garbagecan, safe, shelf, sidetable, sofa, table, bed, desk, microwave, fridge, sink, toaster, coffeemachine, etc.
Common items include: cellphone, newspaper, remotecontrol, statue, keychain, creditcard, tissuebox, box, vase, book, apple, bread, cup, plate, knife, fork, spoon, soapbottle, toiletpaper, lightswitch, watch, etc.
Your goal is to follow instructions like "put X on Y" or "clean X with Y".
Observations describe your current location and what you see.
Actions can be: 'go to <receptacle>', 'open <receptacle>', 'close <receptacle>', 'take <object> from <receptacle>', 'put <object> in/on <receptacle>', 'use <object> on <receptacle>', 'examine <object/receptacle>', 'inventory', 'look'.
Receptacles can be open or closed. You need to open them to see or interact with their contents.
"""

# Parsed trajectory leading to watch discovery
# We'll focus on the segment around finding the watch for the fact extraction test.
# A longer trajectory can be used, but the key is the observation revealing the watch.

alfworld_trajectory_for_watch_test = [
    # ... (earlier steps can be summarized or omitted if too long for a focused test,
    # but some history helps the LLM understand the exploration)
    # Let's pick up from a few steps before finding the watch
    ("You arrive at drawer 3. The drawer 3 is closed.", "go to drawer 3", 0.0, "You open the drawer 3. The drawer 3 is open. In it, you see nothing."), # Step 17-18
    ("You open the drawer 3. The drawer 3 is open. In it, you see nothing.", "open drawer 3", 0.0, "You arrive at drawer 4. The drawer 4 is closed."), # Step 18 (next_obs from trace is a bit off here, but we'll use the pattern)
    # For simplicity, let's jump closer to the discovery for the direct test of fact extraction
    # Ideally, the trajectory would show opening drawer 2, 3, 4 and finding nothing.
    # The key observation before the action that reveals the watch:
    ("You arrive at drawer 5. The drawer 5 is closed.", "go to drawer 5", 0.0, "You arrive at drawer 5. The drawer 5 is closed."), # Simulating arrival at step 22
    ("You arrive at drawer 5. The drawer 5 is closed.", "open drawer 5", 0.0, "You open the drawer 5. The drawer 5 is open. In it, you see a watch 1."), # Step 23
    # The next few steps are taking the watch, which are consequences, not part of discovering *where* it was.
]

# For the test, the trajectory summary should include the "Outcome"
alfworld_task_goal = "put some watch on safe."
alfworld_outcome_watch_found = "PARTIAL_PROGRESS" # Or "FAILURE" if it ended without completing task
alfworld_episode_reward_watch_found = 0.0 # Assuming reward only comes at the very end

class TestLLMReflexionComponents(unittest.TestCase):
    def setUp(self):
        self.agent = LLMReactFactExtractionAgent(
            model="gpt-4o", # Use a capable model, e.g., gpt-4o or gpt-4-turbo-preview
            env_description=TEST_ENV_DESCRIPTION,
            allowed_actions=("left", "down", "right", "up"),
            fact_buffer_len=10,
            compress=False # Default, will be overridden in compression tests
        )
        # Ensure tools are correctly initialized for the agent
        # The __post_init__ should handle this now by adding _fact_extraction_tools to self._tools

    # --- Tests for _call_llm_for_fact_extraction ---

    # @unittest.skipUnless(RUN_LIVE_LLM_TESTS and API_CONFIGURED, "Skipping live LLM tests.")
    def test_live_fact_extraction_hole_discovery(self):
        """Tests _call_llm_for_fact_extraction for discovering a new hole."""
        self.agent._facts.clear()
        trajectory_str = "1. Obs: S00 | Act: down | R: +0 | Next: S10\n2. Obs: S10 | Act: right | R: -1 | Next: S11"
        trajectory_summary = (
            f"It is now the end of an episode, here is the episode trajectory:\n"
            f"```\nOutcome: FAILURE (total reward = -1.0)\n"
            f"Trajectory:\n{trajectory_str}\n```"
        )
        existing_facts_summary = "<none>"

        # self.agent.verbose = 3 # Set on agent if it has such a flag for LLM call logging
        print(f"\n[Live Test Extraction] Prompting for hole discovery. Existing facts: {existing_facts_summary}")
        
        new_facts = self.agent._call_llm_for_fact_extraction(trajectory_summary, existing_facts_summary)
        
        print(f"[Live Test Extraction] Extracted new facts: {new_facts}")
        self.assertGreater(len(new_facts), 0, "Should have extracted at least one fact about the hole.")
        found_hole_fact = any(
            "s11" in fact and ("-1" in fact)
            for fact in new_facts
        )
        self.assertTrue(found_hole_fact, "LLM did not extract a clear fact about S11 being a hole/terminal state.")

    # @unittest.skipUnless(RUN_LIVE_LLM_TESTS and API_CONFIGURED, "Skipping live LLM tests.")
    def test_live_fact_extraction_goal_discovery(self):
        """Tests _call_llm_for_fact_extraction for discovering a new goal."""
        self.agent._facts.clear()
        trajectory_str = ("1. Obs: S00 | Act: right | R: +0 | Next: S01\n"
                          "2. Obs: S01 | Act: right | R: +0 | Next: S02\n"
                          "3. Obs: S02 | Act: right | R: +1 | Next: S03")
        trajectory_summary = (
            f"It is now the end of an episode, here is the episode trajectory:\n"
            f"```\nOutcome: SUCCESS (total reward = +1.0)\n"
            f"Trajectory:\n{trajectory_str}\n```"
        )
        existing_facts_summary = "<none>"
        
        print(f"\n[Live Test Extraction] Prompting for goal discovery. Existing facts: {existing_facts_summary}")
        new_facts = self.agent._call_llm_for_fact_extraction(trajectory_summary, existing_facts_summary)

        print(f"[Live Test Extraction] Extracted new facts: {new_facts}")
        self.assertGreater(len(new_facts), 0, "Should have extracted at least one fact about the goal.")
        found_goal_fact = any(
            "s03" in fact #and ("goal" in fact or "terminal" in fact or "reward" in fact or "success" in fact)
            for fact in new_facts
        )
        self.assertTrue(found_goal_fact, "LLM did not extract a clear fact about S03 being the goal.")

    # @unittest.skipUnless(RUN_LIVE_LLM_TESTS and API_CONFIGURED, "Skipping live LLM tests.")
    def test_live_fact_extraction_avoids_redundancy(self):
        """Tests _call_llm_for_fact_extraction avoids extracting known facts."""
        self.agent._facts.clear()
        initial_fact_str = "s11 is a hole state"
        self.agent._facts.append(initial_fact_str) # Agent already knows this
        
        trajectory_str = "1. Obs: S00 | Act: down | R: +0 | Next: S10\n2. Obs: S10 | Act: right | R: -1 | Next: S11" # Rediscover S11
        trajectory_summary = (
            f"It is now the end of an episode, here is the episode trajectory:\n"
            f"```\nOutcome: FAILURE (total reward = -1.0)\n"
            f"Trajectory:\n{trajectory_str}\n```"
        )
        existing_facts_summary = f"- {initial_fact_str}"

        print(f"\n[Live Test Extraction] Prompting for redundant fact. Existing facts: {existing_facts_summary}")
        new_facts = self.agent._call_llm_for_fact_extraction(trajectory_summary, existing_facts_summary)
        
        print(f"[Live Test Extraction] Extracted new facts (should be few/none related to S11): {new_facts}")
        is_new_fact_about_s11_hole = any("s11" in fact and "hole" in fact for fact in new_facts)
        self.assertFalse(is_new_fact_about_s11_hole,
                         f"LLM re-extracted a fact about S11 ({new_facts}) when it was already known via exact phrasing.")
        # Note: LLM might extract a *differently phrased* but semantically similar fact. This test checks for exact re-extraction based on prompt.

    # --- Tests for _call_llm_for_fact_compression ---

    # @unittest.skipUnless(RUN_LIVE_LLM_TESTS and API_CONFIGURED, "Skipping live LLM tests.")
    def test_live_fact_compression_maintains_critical_info(self):
        """Tests _call_llm_for_fact_compression maintains critical info and removes clear redundancy."""
        initial_uncompressed_facts = [
            "s11 is a dangerous hole location.", # Critical
            "state s11 is where you fall in a hole.", # Redundant with first
            "s22 is also a hole.", # Critical
            "the goal is at s03 and gives positive reward.", # Critical
            "moving from s00 to s01 is a valid action on frozen surface." # Less critical / env detail
        ]
        current_facts_summary = "\n".join(f"- {f}" for f in initial_uncompressed_facts)

        print(f"\n[Live Test Compression] Compressing facts. Initial: {initial_uncompressed_facts}")
        compressed_facts = self.agent._call_llm_for_fact_compression(current_facts_summary)
        
        print(f"[Live Test Compression] Compressed facts: {compressed_facts}")
        self.assertTrue(any("s11" in fact and "hole" in fact for fact in compressed_facts), "S11 hole info lost.")
        self.assertTrue(any("s22" in fact and "hole" in fact for fact in compressed_facts), "S22 hole info lost.")
        self.assertTrue(any("s03" in fact and "goal" in fact for fact in compressed_facts), "S03 goal info lost.")
        
        # # Check for reduction in count if significant redundancy existed
        # self.assertLess(len(compressed_facts), len(initial_uncompressed_facts),
        #                 "Compression should have reduced the number of facts due to clear redundancy.")
        
        # # Check that the specific redundant phrase "state s11 is where you fall in a hole" is gone
        # # (or a very similar one if the LLM rephrased the first)
        # self.assertFalse(any("state s11 is where you fall in a hole" in fact for fact in compressed_facts),
        #                  "Clearly redundant fact about S11 was not removed.")


    # @unittest.skipUnless(RUN_LIVE_LLM_TESTS and API_CONFIGURED, "Skipping live LLM tests.")
    def test_live_fact_compression_no_change_if_all_critical_unique(self):
        """Tests _call_llm_for_fact_compression when facts are already concise and critical."""
        initial_critical_facts = [
            "s11 is a hole.",
            "s03 is the goal."
        ]
        current_facts_summary = "\n".join(f"- {f}" for f in initial_critical_facts)

        print(f"\n[Live Test Compression] Compressing already concise facts. Initial: {initial_critical_facts}")
        compressed_facts = self.agent._call_llm_for_fact_compression(current_facts_summary)

        print(f"[Live Test Compression] Compressed facts: {compressed_facts}")
        self.assertEqual(len(compressed_facts), len(initial_critical_facts),
                         "Number of facts changed when they were already concise and critical.")
        for fact in initial_critical_facts:
            self.assertIn(fact, compressed_facts, f"Critical fact '{fact}' lost or altered unnecessarily.")

    # @unittest.skipUnless(RUN_LIVE_LLM_TESTS and API_CONFIGURED, "Skipping live LLM tests.")
    def test_live_fact_compression_empty_input(self):
        """Tests _call_llm_for_fact_compression with no initial facts."""
        current_facts_summary = "<none>"

        print(f"\n[Live Test Compression] Compressing empty facts.")
        compressed_facts = self.agent._call_llm_for_fact_compression(current_facts_summary)

        print(f"[Live Test Compression] Compressed facts (should be empty): {compressed_facts}")
        self.assertEqual(len(compressed_facts), 0, "Compression with empty input should result in empty facts list.")

# In TestLLMReflexionComponents class:

    # ... (existing setUp and FrozenLake tests) ...

    # @unittest.skipUnless(RUN_LIVE_LLM_TESTS and API_CONFIGURED, "Skipping live LLM tests.")
    def test_live_fact_extraction_alfworld_real_watch_location(self):
        """
        Tests _call_llm_for_fact_extraction for discovering the location of a watch in ALFWorld.
        """
        # Override agent's env_description for this ALFWorld-specific test
        from recall.envs.alfworld_env import AlfWorldEnv
        env = AlfWorldEnv()
        original_env_description = self.agent.env_description
        self.agent.env_description = env.env_description
        self.agent._facts.clear()

        trajectory_summary = """
It is now the end of an episode, here is the episode trajectory:
```
Outcome: SUCCESS (total reward = +1.0)
Trajectory:
1. Obs: -= Welcome to TextWorld, ALFRED! =-

You are in the middle of a room. Looking quickly around you, you see a armchair 1, a cabinet 4, a cabinet 3, a cabinet 2, a cabinet 1, a drawer 5, a drawer 4, a drawer 3, a drawer 2, a drawer 1, a dresser 1, a garbagecan 1, a safe 1, a shelf 12, a shelf 11, a shelf 10, a shelf 9, a shelf 8, a shelf 7, a shelf 6, a shelf 5, a shelf 4, a shelf 3, a shelf 2, a shelf 1, a sidetable 1, and a sofa 1.

Your task is to: put some watch on safe. | Act: go to dresser 1 | R: +0 | Next: You arrive at dresser 1. On the dresser 1, you see a cellphone 1, a newspaper 1, a remotecontrol 1, and a statue 1.
2. Obs: You arrive at dresser 1. On the dresser 1, you see a cellphone 1, a newspaper 1, a remotecontrol 1, and a statue 1. | Act: go to cabinet 1 | R: +0 | Next: You arrive at cabinet 1. The cabinet 1 is closed.
3. Obs: You arrive at cabinet 1. The cabinet 1 is closed. | Act: open cabinet 1 | R: +0 | Next: You open the cabinet 1. The cabinet 1 is open. In it, you see a tissuebox 1.
4. Obs: You open the cabinet 1. The cabinet 1 is open. In it, you see a tissuebox 1. | Act: go to sidetable 1 | R: +0 | Next: You arrive at sidetable 1. On the sidetable 1, you see a keychain 3, a keychain 2, and a keychain 1.
5. Obs: You arrive at sidetable 1. On the sidetable 1, you see a keychain 3, a keychain 2, and a keychain 1. | Act: go to shelf 1 | R: +0 | Next: You arrive at shelf 1. On the shelf 1, you see nothing.
6. Obs: You arrive at shelf 1. On the shelf 1, you see nothing. | Act: go to drawer 2 | R: +0 | Next: You arrive at drawer 2. The drawer 2 is closed.
7. Obs: You arrive at drawer 2. The drawer 2 is closed. | Act: open drawer 2 | R: +0 | Next: You open the drawer 2. The drawer 2 is open. In it, you see nothing.
8. Obs: You open the drawer 2. The drawer 2 is open. In it, you see nothing. | Act: go to drawer 5 | R: +0 | Next: You arrive at drawer 5. The drawer 5 is closed.
9. Obs: You arrive at drawer 5. The drawer 5 is closed. | Act: open drawer 5 | R: +0 | Next: You open the drawer 5. The drawer 5 is open. In it, you see a watch 1.
10. Obs: You open the drawer 5. The drawer 5 is open. In it, you see a watch 1. | Act: move watch 1 to safe 1 | R: +0 | Next: Nothing happens.
11. Obs: Nothing happens. | Act: go to safe 1 | R: +0 | Next: You arrive at safe 1. The safe 1 is closed.
12. Obs: You arrive at safe 1. The safe 1 is closed. | Act: look | R: +0 | Next: You are facing the safe 1. Next to it, you see nothing.
13. Obs: You are facing the safe 1. Next to it, you see nothing. | Act: open safe 1 | R: +0 | Next: You open the safe 1. The safe 1 is open. In it, you see a vase 2.
14. Obs: You open the safe 1. The safe 1 is open. In it, you see a vase 2. | Act: move watch 1 to safe 1 | R: +0 | Next: Nothing happens.
15. Obs: Nothing happens. | Act: go to drawer 5 | R: +0 | Next: You arrive at drawer 5. The drawer 5 is open. In it, you see a watch 1.
16. Obs: You arrive at drawer 5. The drawer 5 is open. In it, you see a watch 1. | Act: take watch 1 from drawer 5 | R: +0 | Next: You pick up the watch 1 from the drawer 5.
17. Obs: You pick up the watch 1 from the drawer 5. | Act: move watch 1 to safe 1 | R: +0 | Next: Nothing happens.
18. Obs: Nothing happens. | Act: look | R: +0 | Next: You are facing the drawer 5. Next to it, you see nothing.
19. Obs: You are facing the drawer 5. Next to it, you see nothing. | Act: move watch 1 to safe 1 | R: +0 | Next: Nothing happens.
20. Obs: Nothing happens. | Act: go to safe 1 | R: +0 | Next: You arrive at safe 1. The safe 1 is open. In it, you see a vase 2.
21. Obs: You arrive at safe 1. The safe 1 is open. In it, you see a vase 2. | Act: move watch 1 to safe 1 | R: +1 | Next: You move the watch 1 to the safe 1.
```
"""
        print(f"\n[Live Test ALFWorld Extraction] Prompting for watch location. Task: {alfworld_task_goal}")
        # print(f"Trajectory Summary:\n{trajectory_summary}")
        # print(f"Existing Facts: {existing_facts_summary}")
        existing_facts_summary = []

        new_facts = self.agent._call_llm_for_fact_extraction(trajectory_summary, existing_facts_summary)
        
        print(f"[Live Test ALFWorld Extraction] Extracted new facts: {new_facts}")
        self.assertGreater(len(new_facts), 0, "Should have extracted at least one fact about the watch.")
        
        # The LLM should extract something like:
        # - "a watch 1 is in drawer 5"
        # - "drawer 5 contains a watch 1"
        # - "found watch 1 in open drawer 5"
        found_watch_location_fact = any(
            "watch" in fact and ("1" in fact or True) and # "watch 1" or just "watch"
            "drawer 5" in fact and
            ("in" in fact or "contains" in fact or "inside" in fact or "located" in fact)
            for fact in new_facts
        )
        self.assertTrue(found_watch_location_fact, 
                        "LLM did not extract a clear fact stating 'watch 1 is in drawer 5' or similar.")

        # Restore original env_description if other tests rely on it
        self.agent.env_description = original_env_description




    # @unittest.skipUnless(RUN_LIVE_LLM_TESTS and API_CONFIGURED, "Skipping live LLM tests.")
    def test_live_fact_extraction_alfworld_watch_location(self):
        """
        Tests _call_llm_for_fact_extraction for discovering the location of a watch in ALFWorld.
        """
        # Override agent's env_description for this ALFWorld-specific test
        original_env_description = self.agent.env_description
        self.agent.env_description = ALFWORLD_ENV_DESCRIPTION_FOR_TEST
        self.agent._facts.clear()

        # Construct trajectory leading to the watch discovery
        # We want the LLM to extract "watch 1 is in drawer 5" or similar
        obs_before_open_drawer5 = "You arrive at drawer 5. The drawer 5 is closed."
        act_open_drawer5 = "open drawer 5"
        # This is the CRITICAL observation after the action:
        next_obs_watch_found = "You open the drawer 5. The drawer 5 is open. In it, you see a watch 1."
        
        # For a focused test, the trajectory might only need the critical step.
        # However, providing a bit more context can be helpful for the LLM.
        # Let's use a slightly more complete trajectory based on the trace.
        
        # We'll simulate a simplified trajectory leading to this discovery.
        # The observation just BEFORE opening drawer 5 is what's important.
        # The action is "open drawer 5".
        # The observation AFTER opening drawer 5 reveals the watch.
        
        trajectory_list_for_prompt = [
            # (Optional previous steps to show exploration)
            # ("You are in the middle of a room...", "go to drawer 4", 0.0, "You arrive at drawer 4... closed."),
            # ("You arrive at drawer 4... closed.", "open drawer 4", 0.0, "You open drawer 4... In it, you see nothing."),
            (obs_before_open_drawer5, act_open_drawer5, 0.0, next_obs_watch_found),
        ]

        trajectory_lines = [
            f"{i+1}. Obs: {o} | Act: {a} | R: {r:+.0f} | Next: {n}"
            for i, (o, a, r, n) in enumerate(trajectory_list_for_prompt)
        ]
        trajectory_str = "\n".join(trajectory_lines)
        
        # Assume the task was "put some watch on safe" and it's not yet complete.
        outcome = "FAILURE" # or "IN_PROGRESS" if that's a category
        current_episode_reward = 0.0 # No task completion reward yet

        trajectory_summary = (
            f"It is now the end of an episode (or a significant discovery point). Task: {alfworld_task_goal}\n"
            f"```\nOutcome: {outcome} (total reward = {current_episode_reward:+.1f})\n"
            f"Trajectory:\n{trajectory_str}\n```"
        )
        existing_facts_summary = "<none>" # No prior knowledge of the watch

        print(f"\n[Live Test ALFWorld Extraction] Prompting for watch location. Task: {alfworld_task_goal}")
        # print(f"Trajectory Summary:\n{trajectory_summary}")
        # print(f"Existing Facts: {existing_facts_summary}")

        new_facts = self.agent._call_llm_for_fact_extraction(trajectory_summary, existing_facts_summary)
        
        print(f"[Live Test ALFWorld Extraction] Extracted new facts: {new_facts}")
        self.assertGreater(len(new_facts), 0, "Should have extracted at least one fact about the watch.")
        
        # The LLM should extract something like:
        # - "a watch 1 is in drawer 5"
        # - "drawer 5 contains a watch 1"
        # - "found watch 1 in open drawer 5"
        found_watch_location_fact = any(
            "watch" in fact and ("1" in fact or True) and # "watch 1" or just "watch"
            "drawer 5" in fact and
            ("in" in fact or "contains" in fact or "inside" in fact or "located" in fact)
            for fact in new_facts
        )
        self.assertTrue(found_watch_location_fact, 
                        "LLM did not extract a clear fact stating 'watch 1 is in drawer 5' or similar.")

        # Restore original env_description if other tests rely on it
        self.agent.env_description = original_env_description


    # @unittest.skipUnless(RUN_LIVE_LLM_TESTS and API_CONFIGURED, "Skipping live LLM tests.")
    def test_live_fact_extraction_alfworld_object_on_receptacle(self):
        """
        Tests _call_llm_for_fact_extraction for discovering an object on an open receptacle.
        Example: "On the sidetable 1, you see a keychain 3, a keychain 2, and a keychain 1."
        """
        original_env_description = self.agent.env_description
        self.agent.env_description = ALFWORLD_ENV_DESCRIPTION_FOR_TEST
        self.agent._facts.clear()

        obs_at_sidetable = "You are in the middle of a room. Looking quickly around you, you see a sidetable 1..." # Initial state
        act_go_sidetable = "go to sidetable 1"
        # CRITICAL observation after arriving:
        next_obs_at_sidetable_items = "You arrive at sidetable 1. On the sidetable 1, you see a keychain 3, a keychain 2, and a keychain 1."
        
        trajectory_list_for_prompt = [
            (obs_at_sidetable, act_go_sidetable, 0.0, next_obs_at_sidetable_items),
        ]
        trajectory_lines = [f"{i+1}. Obs: {o} | Act: {a} | R: {r:+.0f} | Next: {n}" for i, (o,a,r,n) in enumerate(trajectory_list_for_prompt)]
        trajectory_str = "\n".join(trajectory_lines)
        
        outcome = "IN_PROGRESS" 
        current_episode_reward = 0.0

        trajectory_summary = (
            f"It is now the end of an episode (or a significant discovery point). Task: {alfworld_task_goal}\n"
            f"```\nOutcome: {outcome} (total reward = {current_episode_reward:+.1f})\n"
            f"Trajectory:\n{trajectory_str}\n```"
        )
        existing_facts_summary = "<none>"

        print(f"\n[Live Test ALFWorld Extraction] Prompting for items on sidetable 1. Task: {alfworld_task_goal}")
        new_facts = self.agent._call_llm_for_fact_extraction(trajectory_summary, existing_facts_summary)
        
        print(f"[Live Test ALFWorld Extraction] Extracted new facts: {new_facts}")
        self.assertGreaterEqual(len(new_facts), 1, "Should have extracted facts about items on sidetable 1.") # Expecting at least 1, ideally more
        
        # Check for specific items. LLM might consolidate.
        found_keychain1_fact = any("keychain 1" in fact and "sidetable 1" in fact and ("on" in fact or "is on" in fact) for fact in new_facts)
        found_keychain2_fact = any("keychain 2" in fact and "sidetable 1" in fact and ("on" in fact or "is on" in fact) for fact in new_facts)
        found_keychain3_fact = any("keychain 3" in fact and "sidetable 1" in fact and ("on" in fact or "is on" in fact) for fact in new_facts)
        
        # A more robust check might be if it mentions "keychains" (plural) on sidetable 1
        found_any_keychain_fact = any("keychain" in fact and "sidetable 1" in fact and ("on" in fact or "is on" in fact) for fact in new_facts)

        self.assertTrue(found_any_keychain_fact, "LLM did not extract a fact about keychains on sidetable 1.")
        # If you want to be stricter:
        # self.assertTrue(found_keychain1_fact and found_keychain2_fact and found_keychain3_fact,
        #                 "LLM did not extract facts for all keychains on sidetable 1.")

        self.agent.env_description = original_env_description

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)