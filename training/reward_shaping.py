from __future__ import annotations
from typing import List, Dict, Any
from training.rollout_collector import is_valid_action_json


def sepsis_reward_fn(completions: List[str], prompts: List[str], **kwargs):
    rewards: List[float] = []
    metadata = kwargs.get("metadata", [{} for _ in completions])
    for comp, meta in zip(completions, metadata):
        env_r = float(meta.get("env_reward_placeholder", 0.0))
        format_bonus = 0.2 if is_valid_action_json(comp) else -0.3
        rewards.append(env_r + format_bonus)
    return rewards
