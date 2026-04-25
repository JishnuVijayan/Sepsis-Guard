from __future__ import annotations
import hashlib
import json
import logging
import os
import re
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Callable, Tuple, Optional

import requests

logger = logging.getLogger(__name__)

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


_VALID_OPS: Dict[str, set] = {
    "nurse": {"escalate_to_physician", "request_lab_test", "administer_medication", "flag_concern", "noop"},
    "lab": {"release_result", "flag_critical", "recommend_followup_test", "noop"},
    "pharmacist": {"flag_interaction", "flag_immunosuppression", "recommend_antibiotic", "check_dosing", "noop"},
    "physician": {"order_antibiotics", "order_lab_test", "admit_to_icu", "request_consult", "do_nothing"},
}

_VALID_KEYS: Dict[str, set] = {
    "nurse": {"operation", "patient_id", "urgency", "test_type", "rationale"},
    "lab": {"operation", "patient_id", "test", "reason"},
    "pharmacist": {"operation", "patient_id", "drug", "rationale"},
    "physician": {"operation", "patient_id", "drug", "test", "specialty"},
}

_OPS_REQUIRING_PATIENT_ID: Dict[str, set] = {
    "nurse": {"escalate_to_physician", "request_lab_test", "administer_medication", "flag_concern"},
    "lab": {"release_result", "flag_critical", "recommend_followup_test"},
    "pharmacist": {"flag_interaction", "flag_immunosuppression", "recommend_antibiotic", "check_dosing"},
    "physician": {"order_antibiotics", "order_lab_test", "admit_to_icu", "request_consult"},
}

def _default_op(role: str) -> str:
    return "do_nothing" if role == "physician" else "noop"


def _sanitize_action(parsed: Dict[str, Any], role: str) -> Dict[str, Any]:
    """Ensure the action is valid for the role. Fix common LLM mistakes."""
    op = parsed.get("operation", "")
    # Fix cross-role operation mistakes
    if role == "physician" and op == "noop":
        op = "do_nothing"
    elif role != "physician" and op == "do_nothing":
        op = "noop"
    # If operation still invalid for this role, fall back
    if op not in _VALID_OPS.get(role, set()):
        return {"operation": _default_op(role)}
    parsed["operation"] = op
    # Strip keys that Pydantic would reject for this role
    valid_keys = _VALID_KEYS.get(role, set())
    return {k: v for k, v in parsed.items() if k in valid_keys}


def _parse_action(text: str, role: str) -> Dict[str, Any]:
    stripped = text.strip()
    try:
        parsed = json.loads(stripped)
        if isinstance(parsed, dict) and "operation" in parsed:
            return _sanitize_action(parsed, role)
    except Exception:
        pass
    m = re.search(r"\{[^{}]*\"operation\"[^{}]*\}", stripped)
    if m:
        try:
            parsed = json.loads(m.group(0))
            if isinstance(parsed, dict) and "operation" in parsed:
                return _sanitize_action(parsed, role)
        except Exception:
            pass
    return {"operation": _default_op(role)}


def _extract_observation_from_prompt(prompt: str) -> Dict[str, Any]:
    marker = "Observation:\n"
    end_marker = "\nAction (JSON):"
    if marker not in prompt or end_marker not in prompt:
        return {}
    start = prompt.index(marker) + len(marker)
    end = prompt.rfind(end_marker)
    raw = prompt[start:end].strip()
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def _observation_has_actionable_signal(role: str, obs: Dict[str, Any]) -> bool:
    if role == "nurse":
        for p in obs.get("patient_vitals", []):
            if (
                p.get("qsofa_score", 0) >= 2
                or p.get("mean_arterial_pressure", 999) < 65
                or p.get("systolic_bp", 999) <= 100 and p.get("respiratory_rate", 0) >= 22
                or p.get("oxygen_saturation", 100) < 93
                or p.get("mental_status") in ("confused", "unresponsive")
            ):
                return True
        return False
    if role == "lab":
        for r in obs.get("lab_results", []):
            if (
                (r.get("lactate") is not None and r["lactate"] > 2.2)
                or (r.get("wbc") is not None and (r["wbc"] > 12 or r["wbc"] < 4))
                or (r.get("procalcitonin") is not None and r["procalcitonin"] > 0.5)
                or (r.get("creatinine") is not None and r["creatinine"] > 1.5)
                or (r.get("platelets") is not None and r["platelets"] < 150)
                or (r.get("bilirubin_total") is not None and r["bilirubin_total"] > 1.2)
            ):
                return True
        return False
    if role == "pharmacist":
        meds = obs.get("patient_medications", [])
        if any(m.get("immunocompromised") for m in meds):
            return True
        if any(f.get("flag_type") == "critical_lab" for f in obs.get("lab_flags_this_tick", [])):
            return True
        return False
    if role == "physician":
        known = obs.get("known_patient_summaries", [])
        if any(
            s.get("qsofa_score", 0) >= 2
            or s.get("organ_dysfunction_score", 0.0) >= 2.0
            or len(s.get("flags_raised", [])) >= 2
            for s in known
        ):
            return True
        return False
    return False


def _noop_reward(role: str, obs: Dict[str, Any]) -> float:
    if _observation_has_actionable_signal(role, obs):
        return -0.8
    return 0.05


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
        seed: int = 42, warmup_ticks: int = 8, inject_ticks: int = 1,
        max_workers: int = 4, verbose: bool = False,
    ) -> None:
        self.env_url = env_url.rstrip("/")
        self.task_name = task_name
        self.seed = seed
        self.warmup_ticks = warmup_ticks
        self.inject_ticks = inject_ticks
        self.verbose = verbose
        self._baseline_cache: Dict[Tuple[int, str], float] = {}
        self._local = threading.local()
        self._executor = ThreadPoolExecutor(max_workers=max_workers)

    def _log(self, message: str) -> None:
        if self.verbose:
            print(message)

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

    def _baseline_actions(
        self,
        observations: Dict[str, Any],
        nurse: HeuristicNurse,
        lab: HeuristicLab,
        pharmacist: HeuristicPharmacist,
        physician: HeuristicPhysician,
    ) -> Dict[str, Dict[str, Any]]:
        return {
            "nurse": nurse.decide(observations["nurse"]),
            "lab": lab.decide(observations["lab"]),
            "pharmacist": pharmacist.decide(observations["pharmacist"]),
            "physician": physician.decide(observations["physician"]),
        }

    # Role-specific warmup depths.
    # Physician needs deeper warmup so nurse/lab flags accumulate before it acts.
    # During physician warmup the physician holds back (do_nothing) so the
    # escalation queue is still live when the LLM inject tick fires.
    _ROLE_WARMUP: Dict[str, int] = {
        "nurse": 2,
        "lab": 3,
        "pharmacist": 3,
        "physician": 5,
    }

    def _run_window(
        self, seed: int, role: str, llm_action: Dict[str, Any] | None,
        _diag: Dict[str, Any] | None = None,
    ) -> float:
        """Run warmup + injection window, return accumulated role reward."""
        import traceback
        session_id = uuid.uuid4().hex[:12]
        nurse = HeuristicNurse()
        lab = HeuristicLab()
        pharmacist = HeuristicPharmacist()
        physician = HeuristicPhysician()

        role_warmup = self._ROLE_WARMUP.get(role, self.warmup_ticks)

        try:
            bundle = self._reset(seed, session_id)
            done = False

            warmup_done_early = 0
            for w in range(role_warmup):
                if done:
                    warmup_done_early = w
                    break
                actions = self._baseline_actions(
                    bundle["observations"],
                    nurse,
                    lab,
                    pharmacist,
                    physician,
                )
                # Physician holds back during its warmup so escalation flags
                # are still live (untreated) when the LLM inject tick fires.
                if role == "physician":
                    actions["physician"] = {"operation": "do_nothing"}
                bundle = self._step(actions, session_id)
                done = bundle.get("done", False)

            if _diag is not None:
                _diag["warmup_done"] = done
                _diag["warmup_done_early"] = warmup_done_early
                _diag["tick_after_warmup"] = bundle.get("info", {}).get("tick", "?")

            accumulated = 0.0
            tick_rewards = []
            for t in range(self.inject_ticks):
                if done:
                    break
                actions = self._baseline_actions(
                    bundle["observations"],
                    nurse,
                    lab,
                    pharmacist,
                    physician,
                )
                # Inject LLM action on tick 0 only — repeating it on subsequent ticks
                # causes repeat-escalation penalties that collapse the reward signal.
                if llm_action is not None and t == 0:
                    actions[role] = llm_action
                bundle = self._step(actions, session_id)
                step_r = float(bundle.get("rewards", {}).get(role, 0.0))
                tick_rewards.append(step_r)
                accumulated += step_r
                done = bundle.get("done", False)

            if _diag is not None:
                _diag["inject_tick_rewards"] = tick_rewards
                _diag["accumulated"] = accumulated
                _diag["inject_done_early"] = done and len(tick_rewards) < self.inject_ticks

            return accumulated
        except Exception as exc:
            tb = traceback.format_exc()
            logger.warning("[OnlineSepsisReward] _run_window failed (seed=%d, role=%s): %s", seed, role, exc)
            self._log(f"[REWARD ERR] _run_window EXCEPTION seed={seed} role={role} action={llm_action}")
            self._log(f"  Error: {exc}")
            self._log(f"  Traceback:\n{tb}")
            if _diag is not None:
                _diag["exception"] = str(exc)
                _diag["traceback"] = tb
            return 0.0
        finally:
            self._delete_session(session_id)

    def _preview_role_state(
        self, seed: int, role: str,
    ) -> Tuple[Dict[str, Any] | None, Dict[str, Any] | None]:
        """Warm up deterministically and return (observation, heuristic_action)."""
        session_id = uuid.uuid4().hex[:12]
        nurse = HeuristicNurse()
        lab = HeuristicLab()
        pharmacist = HeuristicPharmacist()
        physician = HeuristicPhysician()
        role_warmup = self._ROLE_WARMUP.get(role, self.warmup_ticks)
        try:
            bundle = self._reset(seed, session_id)
            done = False
            for _ in range(role_warmup):
                if done:
                    return None, None
                actions = self._baseline_actions(
                    bundle["observations"],
                    nurse,
                    lab,
                    pharmacist,
                    physician,
                )
                if role == "physician":
                    actions["physician"] = {"operation": "do_nothing"}
                bundle = self._step(actions, session_id)
                done = bundle.get("done", False)
            if done:
                return None, None
            obs = bundle.get("observations", {}).get(role)
            if not obs:
                return None, None
            heuristic_action = self._baseline_actions(
                bundle["observations"],
                nurse,
                lab,
                pharmacist,
                physician,
            ).get(role)
            return obs, heuristic_action
        except Exception:
            return None, None
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
    ) -> Tuple[int, float, Dict[str, Any]]:
        """Evaluate one completion, return (index, reward, diag)."""
        role = _role_from_prompt(prompt)
        obs = _extract_observation_from_prompt(prompt)
        llm_action = _parse_action(completion, role)
        is_noop = llm_action.get("operation") in (_default_op(role), "noop")
        has_patient = bool(llm_action.get("patient_id"))
        needs_patient = llm_action.get("operation") in _OPS_REQUIRING_PATIENT_ID.get(role, set())
        rationale_val = llm_action.get("rationale") or ""
        has_rationale = bool(str(rationale_val).strip().replace("...", ""))

        seed = int(hashlib.md5(prompt.encode()).hexdigest()[:8], 16) % 100000

        diag: Dict[str, Any] = {
            "idx": idx, "role": role, "seed": seed,
            "action": llm_action, "is_noop": is_noop,
            "completion_len": len(completion.split()),
        }

        if is_noop:
            noop_r = _noop_reward(role, obs)
            diag["path"] = "noop_eval"
            diag["actionable_signal"] = _observation_has_actionable_signal(role, obs)
            diag["final_r"] = noop_r
            return idx, noop_r, diag

        baseline_r = self._get_baseline(seed, role)
        diag_llm: Dict[str, Any] = {}
        llm_r = self._run_window(seed, role, llm_action, _diag=diag_llm)
        advantage = llm_r - baseline_r
        env_r = advantage * 2.0

        if not has_rationale and has_patient:
            env_r -= 0.1
        if needs_patient and not has_patient:
            env_r -= 0.1

        final_r = max(-1.0, min(1.0, env_r))

        diag.update({
            "path": "env_eval",
            "baseline_r": round(baseline_r, 4),
            "llm_r": round(llm_r, 4),
            "advantage": round(advantage, 4),
            "final_r": round(final_r, 4),
            "inject_tick_rewards": diag_llm.get("inject_tick_rewards", []),
            "warmup_done_early": diag_llm.get("warmup_done_early", 0),
            "inject_done_early": diag_llm.get("inject_done_early", False),
            "tick_after_warmup": diag_llm.get("tick_after_warmup", "?"),
        })

        return idx, final_r, diag

    def __call__(
        self, completions: List[str], prompts: List[str], **kwargs: Any,
    ) -> List[float]:
        rewards = [0.0] * len(completions)
        diags: List[Dict[str, Any]] = [{}] * len(completions)

        futures = [
            self._executor.submit(self._eval_single, i, c, p)
            for i, (c, p) in enumerate(zip(completions, prompts))
        ]
        errors: List[str] = []
        for future in as_completed(futures):
            try:
                idx, reward, diag = future.result()
                rewards[idx] = reward
                diags[idx] = diag
            except Exception as exc:
                errors.append(str(exc))

        if errors:
            logger.warning("[OnlineSepsisReward] %d futures raised: %s", len(errors), errors[:3])

        if self.verbose:
            zero_count = sum(1 for r in rewards if r == 0.0)
            noop_count = sum(1 for d in diags if d.get("is_noop"))
            env_eval_count = sum(1 for d in diags if d.get("path") == "env_eval")
            mean_r = sum(rewards) / len(rewards) if rewards else 0.0
            min_r = min(rewards) if rewards else 0.0
            max_r = max(rewards) if rewards else 0.0
            roles = [d.get("role", "?") for d in diags]
            role_summary = {r: roles.count(r) for r in set(roles)}
            self._log(
                f"[REWARD] mean={mean_r:.3f} min={min_r:.3f} max={max_r:.3f} "
                f"zeros={zero_count}/{len(rewards)} noops={noop_count} "
                f"env_evals={env_eval_count} roles={role_summary}"
            )

        return rewards

    def preflight_check(self) -> bool:
        """Verify the reward surface is not flat before training.

        The old preflight assumed a fixed seed/patient action, which is brittle
        after environment changes. The new check searches a handful of seeds and
        roles, finds a warmed-up state where the heuristic agent wants to act,
        and verifies that action scores better than a noop in that same setting.
        """
        candidate_seeds = [1, 2, 3, 7, 11, 42, 99]
        role_noops = {
            "nurse": {"operation": "noop"},
            "lab": {"operation": "noop"},
            "pharmacist": {"operation": "noop"},
            "physician": {"operation": "do_nothing"},
        }

        for role in ("nurse", "lab", "pharmacist", "physician"):
            for seed in candidate_seeds:
                obs, heuristic_action = self._preview_role_state(seed, role)
                if obs is None or heuristic_action is None:
                    continue
                if heuristic_action.get("operation") == role_noops[role]["operation"]:
                    continue
                noop_r = self._run_window(seed, role, role_noops[role])
                action_r = self._run_window(seed, role, heuristic_action)
                ok = action_r > noop_r
                if self.verbose:
                    self._log(
                        f"[PREFLIGHT] role={role} seed={seed} "
                        f"action={heuristic_action} noop={noop_r:.4f} act={action_r:.4f} ok={ok}"
                    )
                if ok:
                    return True
        return False


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
        op = parsed.get("operation")
        if op in ("noop", "do_nothing"):
            score -= 0.02
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
    warmup_ticks: int = 8,
    inject_ticks: int = 1,
    max_workers: int = 4,
    verbose: bool = False,
) -> Callable[[List[str], List[str]], List[float]]:
    """Factory for online, live environment reward shaping used by GRPO."""
    return OnlineSepsisReward(
        env_url=env_url, task_name=task_name, seed=seed,
        warmup_ticks=warmup_ticks, inject_ticks=inject_ticks,
        max_workers=max_workers, verbose=verbose,
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
