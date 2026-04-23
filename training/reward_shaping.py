from __future__ import annotations
import json
import os
import re
import uuid
from typing import List, Dict, Any, Callable

import requests

from agents.nurse import HeuristicNurse
from agents.lab import HeuristicLab
from agents.pharmacist import HeuristicPharmacist
from agents.physician import HeuristicPhysician


def is_valid_action_json(text: str) -> bool:
    try:
        parsed = json.loads(text.strip())
        return isinstance(parsed, dict) and "operation" in parsed
    except Exception:
        return False


def _parse_action(text: str, role: str) -> Dict[str, Any]:
    stripped = text.strip()
    try:
        parsed = json.loads(stripped)
        if isinstance(parsed, dict) and "operation" in parsed:
            return parsed
    except Exception:
        pass
    m = re.search(r"\{[^{}]*\"operation\"[^{}]*\}", stripped)
    if m:
        try:
            parsed = json.loads(m.group(0))
            if isinstance(parsed, dict) and "operation" in parsed:
                return parsed
        except Exception:
            pass
    return {"operation": "noop" if role != "physician" else "do_nothing"}


def _role_from_prompt(prompt: str) -> str:
    m = re.search(r"Role:\s*(nurse|lab|pharmacist|physician)", prompt, flags=re.IGNORECASE)
    if m:
        return m.group(1).lower()
    lowered = prompt.lower()
    for role in ("nurse", "lab", "pharmacist", "physician"):
        if role in lowered:
            return role
    return "physician"


class OnlineSepsisReward:
    """Live environment reward function for GRPO training.

    For each completion, this function:
    1) Infers role from prompt.
    2) Builds a full 4-role action bundle using heuristics for non-target roles.
    3) Injects the generated action for the target role at EVERY step.
    4) Calls /reset then /step on the live environment using a dedicated session.
    5) Returns the role-specific reward averaged over all steps.
    """

    def __init__(
        self, env_url: str, task_name: str = "task1_textbook",
        seed: int = 42, max_steps_per_eval: int = 5,
    ) -> None:
        self.env_url = env_url.rstrip("/")
        self.task_name = task_name
        self.seed = seed
        self.max_steps_per_eval = max_steps_per_eval
        self._nurse = HeuristicNurse()
        self._lab = HeuristicLab()
        self._pharmacist = HeuristicPharmacist()
        self._physician = HeuristicPhysician()

    def _headers(self, session_id: str | None = None) -> Dict[str, str]:
        if session_id:
            return {"X-Session-Id": session_id}
        return {}

    def _reset(self, seed: int, session_id: str | None = None) -> Dict[str, Any]:
        r = requests.post(
            f"{self.env_url}/reset",
            json={"task_name": self.task_name, "seed": seed},
            headers=self._headers(session_id),
            timeout=30,
        )
        r.raise_for_status()
        return r.json()

    def _step(self, actions: Dict[str, Dict[str, Any]], session_id: str | None = None) -> Dict[str, Any]:
        r = requests.post(
            f"{self.env_url}/step",
            json={"actions": actions},
            headers=self._headers(session_id),
            timeout=30,
        )
        r.raise_for_status()
        return r.json()

    def _baseline_actions(self, observations: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        return {
            "nurse": self._nurse.decide(observations["nurse"]),
            "lab": self._lab.decide(observations["lab"]),
            "pharmacist": self._pharmacist.decide(observations["pharmacist"]),
            "physician": self._physician.decide(observations["physician"]),
        }

    def __call__(self, completions: List[str], prompts: List[str], **kwargs: Any) -> List[float]:
        rewards: List[float] = []
        for i, (completion, prompt) in enumerate(zip(completions, prompts)):
            role = _role_from_prompt(prompt)
            session_id = uuid.uuid4().hex[:12]
            try:
                llm_action = _parse_action(completion, role)
                bundle = self._reset(self.seed + i, session_id)
                observations = bundle["observations"]
                actions = self._baseline_actions(observations)
                actions[role] = llm_action
                result = self._step(actions, session_id)
                cumulative_r = float(result["rewards"].get(role, 0.0))
                done = result.get("done", False)
                steps = 1
                while not done and steps < self.max_steps_per_eval:
                    obs = result["observations"]
                    actions = self._baseline_actions(obs)
                    actions[role] = llm_action
                    result = self._step(actions, session_id)
                    cumulative_r += float(result["rewards"].get(role, 0.0))
                    done = result.get("done", False)
                    steps += 1
                env_r = cumulative_r / max(1, steps)
            except Exception:
                env_r = -0.5
            finally:
                try:
                    requests.delete(f"{self.env_url}/session/{session_id}", timeout=5)
                except Exception:
                    pass
            parsed = _parse_action(completion, role)
            if parsed.get("operation") in ("noop", "do_nothing"):
                env_r -= 0.1
            rewards.append(env_r)
        return rewards


def format_reward_fn(completions: List[str], prompts: List[str], **kwargs: Any) -> List[float]:
    """Independent reward for valid JSON action format."""
    return [0.3 if is_valid_action_json(c) else -0.5 for c in completions]


def make_online_sepsis_reward_fn(
    env_url: str,
    task_name: str = "task1_textbook",
    seed: int = 42,
    max_steps_per_eval: int = 5,
) -> Callable[[List[str], List[str]], List[float]]:
    """Factory for online, live environment reward shaping used by GRPO."""
    return OnlineSepsisReward(
        env_url=env_url, task_name=task_name, seed=seed,
        max_steps_per_eval=max_steps_per_eval,
    )


def sepsis_reward_fn(completions: List[str], prompts: List[str], **kwargs: Any) -> List[float]:
    """Offline fallback reward function (metadata-linked, not environment-linked).

    Kept for compatibility; prefer make_online_sepsis_reward_fn for real training.
    """
    if os.getenv("ALLOW_OFFLINE_REWARD_FALLBACK", "0") != "1":
        raise RuntimeError(
            "sepsis_reward_fn is an offline compatibility fallback and is disabled by default. "
            "Use make_online_sepsis_reward_fn(...) for live environment rewards, or set "
            "ALLOW_OFFLINE_REWARD_FALLBACK=1 to force offline mode."
        )
    rewards: List[float] = []
    metadata = kwargs.get("metadata", [{} for _ in completions])
    for comp, meta in zip(completions, metadata):
        env_r = float(meta.get("env_reward_placeholder", 0.0))
        format_bonus = 0.2 if is_valid_action_json(comp) else -0.3
        rewards.append(env_r + format_bonus)
    return rewards
