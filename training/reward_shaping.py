from __future__ import annotations
import hashlib
import json
import logging
import os
import re
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Callable, Tuple

import requests

logger = logging.getLogger(__name__)

from agents.nurse import HeuristicNurse
from agents.lab import HeuristicLab
from agents.pharmacist import HeuristicPharmacist
from agents.physician import HeuristicPhysician

FORMAT_REWARD_SCALE = 0.25

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
        max_workers: int = 4,
    ) -> None:
        self.env_url = env_url.rstrip("/")
        self.task_name = task_name
        self.seed = seed
        self.warmup_ticks = warmup_ticks
        self.inject_ticks = inject_ticks
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
            print(f"[REWARD ERR] _run_window EXCEPTION seed={seed} role={role} action={llm_action}")
            print(f"  Error: {exc}")
            print(f"  Traceback:\n{tb}")
            if _diag is not None:
                _diag["exception"] = str(exc)
                _diag["traceback"] = tb
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
    ) -> Tuple[int, float, Dict[str, Any]]:
        """Evaluate one completion, return (index, reward, diag)."""
        role = _role_from_prompt(prompt)
        llm_action = _parse_action(completion, role)
        is_noop = llm_action.get("operation") in (_default_op(role), "noop")
        has_patient = bool(llm_action.get("patient_id"))
        rationale_val = llm_action.get("rationale") or ""
        has_rationale = bool(str(rationale_val).strip().replace("...", ""))

        seed = int(hashlib.md5(prompt.encode()).hexdigest()[:8], 16) % 100000

        diag: Dict[str, Any] = {
            "idx": idx, "role": role, "seed": seed,
            "action": llm_action, "is_noop": is_noop,
            "completion_len": len(completion.split()),
        }

        # Noop penalty: advantage of noop vs baseline is always 0 (both do nothing),
        # so GRPO gets no gradient. Explicit -0.3 gives a push away from collapse.
        if is_noop:
            diag["path"] = "noop_penalty"
            return idx, -1.5, diag

        baseline_r = self._get_baseline(seed, role)
        diag_llm: Dict[str, Any] = {}
        llm_r = self._run_window(seed, role, llm_action, _diag=diag_llm)
        advantage = llm_r - baseline_r
        env_r = advantage * 2.0

        if not has_rationale and has_patient:
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
        self._call_count = getattr(self, "_call_count", 0) + 1
        call_id = self._call_count

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
            print(f"[REWARD ERR call={call_id}] {len(errors)} futures raised: {errors[:3]}")

        # --- per-call summary ---
        zero_count = sum(1 for r in rewards if r == 0.0)
        noop_count = sum(1 for d in diags if d.get("is_noop"))
        noop_penalty_count = sum(1 for d in diags if d.get("path") == "noop_penalty")
        env_eval_count = sum(1 for d in diags if d.get("path") == "env_eval")
        non_zero = [r for r in rewards if r != 0.0]
        mean_r = sum(rewards) / len(rewards) if rewards else 0.0
        mean_nz = sum(non_zero) / len(non_zero) if non_zero else 0.0
        min_r = min(rewards) if rewards else 0.0
        max_r = max(rewards) if rewards else 0.0

        roles = [d.get("role", "?") for d in diags]
        role_summary = {r: roles.count(r) for r in set(roles)}

        print(
            f"[REWARD call={call_id}] "
            f"mean={mean_r:.3f} min={min_r:.3f} max={max_r:.3f} "
            f"zeros={zero_count}/{len(rewards)} noops={noop_count} "
            f"env_evals={env_eval_count} roles={role_summary}"
        )

        # --- per-completion detail for env_eval path ---
        for d in diags:
            if d.get("path") == "env_eval":
                status = "ZERO" if abs(d.get("final_r", 0)) < 0.001 else "OK"
                print(
                    f"  [{status} idx={d['idx']}] role={d['role']} seed={d['seed']} "
                    f"action={d['action'].get('operation')} pid={d['action'].get('patient_id')} "
                    f"baseline={d.get('baseline_r', '?'):.3f} llm={d.get('llm_r', '?'):.3f} "
                    f"adv={d.get('advantage', '?'):.3f} final={d.get('final_r', '?'):.3f} "
                    f"tick_rewards={d.get('inject_tick_rewards')} "
                    f"warmup_early={d.get('warmup_done_early')} inject_early={d.get('inject_done_early')}"
                )
            elif d.get("path") == "noop_penalty":
                print(
                    f"  [NOOP idx={d['idx']}] role={d['role']} "
                    f"op={d['action'].get('operation')} len={d.get('completion_len')} → -1.5"
                )

        # --- collapse warning ---
        if noop_penalty_count >= len(completions) * 0.5:
            print(
                f"[COLLAPSE WARN call={call_id}] "
                f"{noop_penalty_count}/{len(completions)} completions are noops — "
                f"model may be collapsing. Check step sample output."
            )
        elif zero_count > len(rewards) * 0.7:
            print(
                f"[REWARD WARN call={call_id}] {zero_count}/{len(rewards)} are 0.0 — "
                f"baseline≈llm or episode ended early. "
                f"warmup_ticks={self.warmup_ticks} inject_ticks={self.inject_ticks}"
            )

        return rewards

    def preflight_check(self) -> bool:
        """Verify the reward function is discriminative before training.

        Tests nurse role on seed=1 (known true-sepsis seed) directly via
        _run_window — bypasses prompt-hash so the seed is predictable.
        Returns True if reward is discriminative (escalate > noop).
        """
        seed = 1
        role = "nurse"
        noop_action = {"operation": "noop"}
        escalate_action = {
            "operation": "escalate_to_physician",
            "patient_id": "P01",
            "urgency": "critical",
            "rationale": "HR 130 BP 80 temp 39.1",
        }
        print(f"[PREFLIGHT] Testing seed={seed} role={role} directly via _run_window ...")
        baseline = self._get_baseline(seed, role)
        noop_r = self._run_window(seed, role, noop_action)
        escalate_r = self._run_window(seed, role, escalate_action)
        noop_adv = round((noop_r - baseline) * 2.0, 4)
        esc_adv = round((escalate_r - baseline) * 2.0, 4)
        print(f"[PREFLIGHT] baseline={baseline:.4f}  noop_r={noop_r:.4f}→adv={noop_adv}  escalate_r={escalate_r:.4f}→adv={esc_adv}")
        ok = esc_adv > noop_adv
        if ok:
            print(f"[PREFLIGHT] ✅ Discriminative (escalate {esc_adv} > noop {noop_adv})")
        else:
            print(f"[PREFLIGHT] 🔴 NOT discriminative — DO NOT TRAIN")
            print(f"[PREFLIGHT]    Cause: check environment.py flag_counts fix is deployed and server restarted")
        return ok


def format_reward_fn(
    completions: List[str], prompts: List[str], **kwargs: Any,
) -> List[float]:
    """Reward for valid JSON action format.

    Intentionally down-weighted to avoid overpowering environment reward.
    """
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
        op = str(parsed.get("operation", "")).strip().lower()
        if op in {"noop", "do_nothing"}:
            # Prevent convergence to always-noop policies that exploit format reward.
            score -= 0.08
        rationale = parsed.get("rationale", parsed.get("reason", ""))
        if rationale and len(rationale) > 10 and "..." not in rationale:
            score += 0.05
        elif rationale and ("..." in rationale or len(rationale) <= 3):
            score -= 0.1
        rewards.append(score * FORMAT_REWARD_SCALE)
    return rewards


def make_online_sepsis_reward_fn(
    env_url: str,
    task_name: str = "task1_textbook",
    seed: int = 42,
    warmup_ticks: int = 8,
    inject_ticks: int = 1,
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
