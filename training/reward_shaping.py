from __future__ import annotations
import hashlib
import json
import os
import re
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Callable, Tuple

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


_ROLE_RE = re.compile(r"Role:\s*(nurse|lab|pharmacist|physician)", re.IGNORECASE)


def _role_from_prompt(prompt: str) -> str:
    m = _ROLE_RE.search(prompt)
    if m:
        return m.group(1).lower()
    lowered = prompt.lower()
    for role in ("nurse", "lab", "pharmacist", "physician"):
        if role in lowered:
            return role
    return "physician"


class OnlineSepsisReward:
    """Live environment reward for GRPO training.

    Runs a short evaluation window per completion:
      1) Warmup: heuristic-only ticks to build realistic patient state
      2) Inject: LLM action replaces the target role, accumulate per-step rewards
      3) Compare against cached baseline (all-heuristic) for the same window
      4) Return scaled advantage

    Optimizations vs naive full-episode eval:
      - 12 ticks per eval instead of 48 (warmup 4 + inject 8)
      - Per-step reward accumulation, no /grader call needed
      - Baseline cached by (seed, role) — never recomputed
      - Thread-parallel evaluation of independent completions
      - HTTP connection pooling via requests.Session
      - Quick-reject for noop / missing patient_id actions
    """

    def __init__(
        self, env_url: str, task_name: str = "task1_textbook",
        seed: int = 42, warmup_ticks: int = 4, inject_ticks: int = 4,
        max_workers: int = 4,
    ) -> None:
        self.env_url = env_url.rstrip("/")
        self.task_name = task_name
        self.seed = seed
        self.warmup_ticks = warmup_ticks
        self.inject_ticks = inject_ticks
        self._nurse = HeuristicNurse()
        self._lab = HeuristicLab()
        self._pharmacist = HeuristicPharmacist()
        self._physician = HeuristicPhysician()
        self._baseline_cache: Dict[Tuple[int, str], float] = {}
        self._local = threading.local()
        self._executor = ThreadPoolExecutor(max_workers=max_workers)

    def _get_http(self) -> requests.Session:
        if not hasattr(self._local, 'http'):
            s = requests.Session()
            s.headers.update({"Content-Type": "application/json"})
            self._local.http = s
        return self._local.http

    def _reset(self, seed: int, session_id: str) -> Dict[str, Any]:
        r = self._get_http().post(
            f"{self.env_url}/reset",
            json={"task_name": self.task_name, "seed": seed},
            headers={"X-Session-Id": session_id},
            timeout=30,
        )
        r.raise_for_status()
        return r.json()

    def _step(self, actions: Dict[str, Dict[str, Any]], session_id: str) -> Dict[str, Any]:
        r = self._get_http().post(
            f"{self.env_url}/step",
            json={"actions": actions},
            headers={"X-Session-Id": session_id},
            timeout=30,
        )
        r.raise_for_status()
        return r.json()

    def _delete_session(self, session_id: str) -> None:
        try:
            self._get_http().delete(
                f"{self.env_url}/session/{session_id}", timeout=5
            )
        except Exception:
            pass

    def _baseline_actions(self, observations: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        return {
            "nurse": self._nurse.decide(observations["nurse"]),
            "lab": self._lab.decide(observations["lab"]),
            "pharmacist": self._pharmacist.decide(observations["pharmacist"]),
            "physician": self._physician.decide(observations["physician"]),
        }

    def _run_window(
        self, seed: int, role: str, llm_action: Dict[str, Any] | None,
    ) -> float:
        """Run warmup + injection window, return accumulated role reward."""
        session_id = uuid.uuid4().hex[:12]
        try:
            bundle = self._reset(seed, session_id)
            done = False

            for _ in range(self.warmup_ticks):
                if done:
                    break
                actions = self._baseline_actions(bundle["observations"])
                bundle = self._step(actions, session_id)
                done = bundle.get("done", False)

            accumulated = 0.0
            for _ in range(self.inject_ticks):
                if done:
                    break
                actions = self._baseline_actions(bundle["observations"])
                if llm_action is not None:
                    actions[role] = llm_action
                bundle = self._step(actions, session_id)
                accumulated += float(bundle.get("rewards", {}).get(role, 0.0))
                done = bundle.get("done", False)

            return accumulated
        except Exception:
            return 0.0
        finally:
            self._delete_session(session_id)

    def _get_baseline(self, seed: int, role: str) -> float:
        key = (seed, role)
        if key in self._baseline_cache:
            return self._baseline_cache[key]
        score = self._run_window(seed, role, llm_action=None)
        self._baseline_cache[key] = score
        return score

    def _eval_single(
        self, idx: int, completion: str, prompt: str,
    ) -> Tuple[int, float]:
        """Evaluate one completion, return (index, reward)."""
        role = _role_from_prompt(prompt)
        llm_action = _parse_action(completion, role)
        has_patient = bool(llm_action.get("patient_id"))
        has_rationale = bool(
            llm_action.get("rationale", "").strip().replace("...", "")
        )

        seed = int(hashlib.md5(prompt.encode()).hexdigest()[:8], 16) % 100000
        baseline_r = self._get_baseline(seed, role)
        llm_r = self._run_window(seed, role, llm_action)
        advantage = llm_r - baseline_r

        env_r = advantage * 2.0

        if not has_rationale and has_patient:
            env_r -= 0.1

        return idx, max(-1.0, min(1.0, env_r))

    def __call__(
        self, completions: List[str], prompts: List[str], **kwargs: Any,
    ) -> List[float]:
        rewards = [0.0] * len(completions)

        futures = [
            self._executor.submit(self._eval_single, i, c, p)
            for i, (c, p) in enumerate(zip(completions, prompts))
        ]
        for future in as_completed(futures):
            idx, reward = future.result()
            rewards[idx] = reward

        return rewards


def format_reward_fn(
    completions: List[str], prompts: List[str], **kwargs: Any,
) -> List[float]:
    """Reward for valid JSON action format — scaled to not dominate env reward."""
    rewards = []
    for c in completions:
        stripped = c.strip()
        parsed = None
        if is_valid_action_json(stripped):
            parsed = json.loads(stripped)
        else:
            m = re.search(r"\{[^{}]*\"operation\"[^{}]*\}", stripped)
            if m:
                try:
                    candidate = json.loads(m.group(0))
                    if isinstance(candidate, dict) and "operation" in candidate:
                        parsed = candidate
                except Exception:
                    pass
        if parsed is None:
            rewards.append(-0.3)
            continue
        score = 0.1
        if parsed.get("patient_id"):
            score += 0.05
        rationale = parsed.get("rationale", parsed.get("reason", ""))
        if rationale and len(rationale) > 10 and "..." not in rationale:
            score += 0.05
        elif rationale and ("..." in rationale or len(rationale) <= 3):
            score -= 0.1
        rewards.append(score)
    return rewards


def make_online_sepsis_reward_fn(
    env_url: str,
    task_name: str = "task1_textbook",
    seed: int = 42,
    warmup_ticks: int = 4,
    inject_ticks: int = 4,
    max_workers: int = 4,
) -> Callable[[List[str], List[str]], List[float]]:
    """Factory for online, live environment reward shaping used by GRPO."""
    return OnlineSepsisReward(
        env_url=env_url, task_name=task_name, seed=seed,
        warmup_ticks=warmup_ticks, inject_ticks=inject_ticks,
        max_workers=max_workers,
    )


def sepsis_reward_fn(
    completions: List[str], prompts: List[str], **kwargs: Any,
) -> List[float]:
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
