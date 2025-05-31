# Copyright © 2025 Samuel Holt. All rights reserved.
# No licence is granted to copy, use, modify, distribute, or create derivative
# works of this file in any form, except with explicit written permission from
# the copyright holder.
from recall import types
from recall.agents.react import LLMReactAgent
from recall.agents.reflexion import LLMReflexionAgent
from recall.agents.react_fact_extraction import LLMReactFactExtractionAgent
from recall.agents.world_model_planning_parallel import LLMWorldModelPlanningAgentAsync
from recall.agents.random import RandomAgent

def get_agent(method: str, config, env):
    """Get the agent based on the configuration and environment."""

    method_type = types.Method(config["METHOD"])
    if method_type == types.Method.REACT:
        return LLMReactAgent(
            history_len=config["REACT_HISTORY_LEN"],
            model=config["LLM_MODEL"],
            temperature=config["LLM_TEMPERATURE"],
            max_tokens=config["LLM_MAX_TOKENS"],
            env_description=env.env_description,
            allowed_actions=env.action_space,
        )
    elif method_type == types.Method.REFLEXION:
        return LLMReflexionAgent(
            history_len=config["REACT_HISTORY_LEN"],
            model=config["LLM_MODEL"],
            temperature=config["LLM_TEMPERATURE"],
            max_tokens=config["LLM_MAX_TOKENS"],
            env_description=env.env_description,
            allowed_actions=env.action_space,
            lesson_buffer_len=config["REFLEXION_LESSON_BUFFER_LEN"],
        )
    elif method_type == types.Method.REACT_FACT_EXTRACTION_COMPRESS:
        return LLMReactFactExtractionAgent(
            history_len=config["REACT_HISTORY_LEN"],
            model=config["LLM_MODEL"],
            temperature=config["LLM_TEMPERATURE"],
            max_tokens=config["LLM_MAX_TOKENS"],
            env_description=env.env_description,
            allowed_actions=env.action_space,
            fact_buffer_len=config["FACT_BUFFER_LEN"],
            compress=True,
        )
    elif "world_model_facts_parallel_d_" in method_type.name.lower():
        # Extract the depth from the method name
        method_name = method_type.name.lower()
        if '_b_' in method_name:
            depth = int(method_name.split("_d_")[-1].split("_b_")[0])
            branch_factor = int(method_name.split("_b_")[-1])
        else:
            depth = int(method_name.split("_d_")[-1])
            branch_factor = config["WORLD_MODEL_BRANCH_FACTOR"]
        # depth = 3
        return LLMWorldModelPlanningAgentAsync(
            history_len=config["REACT_HISTORY_LEN"],
            model=config["LLM_MODEL"],
            temperature=config["LLM_TEMPERATURE"],
            max_tokens=config["LLM_MAX_TOKENS"],
            env_description=env.env_description,
            allowed_actions=env.action_space,
            fact_buffer_len=config["FACT_BUFFER_LEN"],
            compress=True,
            search_depth=depth,
            branch_factor=branch_factor,  # how many children to expand (≥ 1)
            discount=config["WORLD_MODEL_DISCOUNT"],        # γ for value backup (default 1 – undiscounted)
            verbose=1,
        )
    elif method_type == types.Method.RANDOM:
        return RandomAgent(
            allowed_actions=env.action_space,
            history_len=config["REACT_HISTORY_LEN"],
        )
    else:
        raise ValueError(f"Unknown agent name: {method_type}")