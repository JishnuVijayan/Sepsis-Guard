from __future__ import annotations

import json
import re
from collections import defaultdict
from typing import List, Dict, Any, Iterable, Optional

from training.prompts import build_role_prompt, safe_truncate_obs

ROLES = ("nurse", "lab", "pharmacist", "physician")
_NOISE_FIELDS = {
    "done",
    "reward",
    "cumulative_reward",
    "normalized_score",
    "last_action_result",
}


def generate_completion(model, tokenizer, prompt: str, max_new_tokens: int = 96) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
    )
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def parse_action(text: str, role: str) -> Dict[str, Any]:
    text = text.strip()
    try:
        parsed = json.loads(text)
        if "operation" in parsed:
            return parsed
    except Exception:
        pass
    m = re.search(r"\{[^{}]*\"operation\"[^{}]*\}", text)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass
    return {"operation": "noop" if role != "physician" else "do_nothing"}


def is_valid_action_json(text: str) -> bool:
    try:
        parsed = json.loads(text.strip())
        return isinstance(parsed, dict) and "operation" in parsed
    except Exception:
        return False


def observation_has_actionable_signal(role: str, obs: Dict[str, Any]) -> bool:
    if role == "nurse":
        return any(
            p.get("qsofa_score", 0) >= 2
            or p.get("mean_arterial_pressure", 999) < 65
            or p.get("oxygen_saturation", 100) < 93
            or p.get("mental_status") in ("confused", "unresponsive")
            or (
                p.get("systolic_bp", 999) <= 100
                and p.get("respiratory_rate", 0) >= 22
            )
            for p in obs.get("patient_vitals", [])
        )
    if role == "lab":
        return any(
            (r.get("lactate") is not None and r["lactate"] > 2.2)
            or (r.get("wbc") is not None and (r["wbc"] > 12 or r["wbc"] < 4))
            or (r.get("procalcitonin") is not None and r["procalcitonin"] > 0.5)
            or (r.get("creatinine") is not None and r["creatinine"] > 1.5)
            or (r.get("platelets") is not None and r["platelets"] < 150)
            or (r.get("bilirubin_total") is not None and r["bilirubin_total"] > 1.2)
            for r in obs.get("lab_results", [])
        )
    if role == "pharmacist":
        return (
            any(m.get("immunocompromised") for m in obs.get("patient_medications", []))
            or any(f.get("flag_type") == "critical_lab" for f in obs.get("lab_flags_this_tick", []))
        )
    if role == "physician":
        return any(
            s.get("qsofa_score", 0) >= 2
            or s.get("organ_dysfunction_score", 0.0) >= 2.0
            or len(s.get("flags_raised", [])) >= 2
            for s in obs.get("known_patient_summaries", [])
        )
    return False


def normalize_obs_for_dedup(obs: Dict[str, Any]) -> Dict[str, Any]:
    """Remove non-clinical noise while preserving staleness and temporal context."""
    compact = safe_truncate_obs(obs, max_patients=8)
    compact = {k: v for k, v in compact.items() if k not in _NOISE_FIELDS}
    compact.pop("tick", None)
    compact.pop("max_ticks", None)
    return compact


def prompt_clinical_key(prompt: str) -> str:
    try:
        obs_start = prompt.find("Observation:\n")
        obs_end = prompt.rfind("\nAction (JSON):")
        if obs_start == -1 or obs_end == -1:
            return prompt
        obs_json = prompt[obs_start + len("Observation:\n"):obs_end]
        obs_dict = json.loads(obs_json)
        cleaned = normalize_obs_for_dedup(obs_dict)
        role_prefix = prompt[:obs_start]
        return role_prefix + json.dumps(cleaned, sort_keys=True)
    except Exception:
        return prompt


def dedupe_rollouts(
    rollouts: Iterable[Dict[str, Any]],
    keep_actionable_duplicates: int = 2,
) -> List[Dict[str, Any]]:
    """Deduplicate by clinical state, while allowing a few actionable repeats."""
    seen_counts: Dict[str, int] = defaultdict(int)
    deduped: List[Dict[str, Any]] = []
    for row in rollouts:
        key = prompt_clinical_key(row["prompt"])
        limit = keep_actionable_duplicates if row.get("actionable_signal") else 1
        if seen_counts[key] >= limit:
            continue
        seen_counts[key] += 1
        deduped.append(row)
    return deduped


def rebalance_rollouts(
    rollouts: Iterable[Dict[str, Any]],
    quiet_keep_ratio: float = 0.35,
) -> List[Dict[str, Any]]:
    """Keep all actionable states and a controlled share of quiet states per role."""
    by_role_actionable: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    by_role_quiet: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rollouts:
        bucket = by_role_actionable if row.get("actionable_signal") else by_role_quiet
        bucket[row["role"]].append(row)

    rebalanced: List[Dict[str, Any]] = []
    for role in ROLES:
        actionable = by_role_actionable.get(role, [])
        quiet = by_role_quiet.get(role, [])
        rebalanced.extend(actionable)
        quiet_cap = int(max(8, len(actionable) * quiet_keep_ratio)) if actionable else min(len(quiet), 32)
        rebalanced.extend(quiet[:quiet_cap])
    return rebalanced


def build_prompt_dataset(
    rollouts: Iterable[Dict[str, Any]],
    dedupe: bool = True,
    quiet_keep_ratio: float = 0.35,
):
    from datasets import Dataset

    rows = list(rollouts)
    if dedupe:
        rows = dedupe_rollouts(rows)
    rows = rebalance_rollouts(rows, quiet_keep_ratio=quiet_keep_ratio)
    return Dataset.from_list([{"prompt": row["prompt"]} for row in rows])


def _maybe_create_session(env_client) -> Optional[str]:
    create_session = getattr(env_client, "create_session", None)
    if callable(create_session):
        return create_session()
    return None


def _maybe_delete_session(env_client, session_id: Optional[str]) -> None:
    if not session_id:
        return
    delete_session = getattr(env_client, "delete_session", None)
    if callable(delete_session):
        delete_session(session_id)


def _env_reset(env_client, task_name: str, seed: int, session_id: Optional[str]):
    kwargs = {"task_name": task_name, "seed": seed}
    if session_id is not None:
        kwargs["session_id"] = session_id
    return env_client.reset(**kwargs)


def _env_step(env_client, actions: Dict[str, Dict[str, Any]], session_id: Optional[str]):
    if session_id is not None:
        return env_client.step(actions, session_id=session_id)
    return env_client.step(actions)


def collect_prompt_rollouts(
    env_client,
    heuristic_agents: Dict[str, Any],
    task_name: str = "task1_textbook",
    seeds: Optional[Iterable[int]] = None,
    n_episodes: int = 16,
    max_ticks_per_episode: int = 64,
    include_quiet: bool = True,
) -> List[Dict[str, Any]]:
    """
    Generate prompt-only rollout rows from heuristic play.

    Each row includes the prompt plus metadata that helps filter and rebalance
    the Colab training set without changing the GRPO prompt format.
    """
    if seeds is None:
        seeds = range(42, 42 + n_episodes)

    rows: List[Dict[str, Any]] = []
    for seed in seeds:
        episode_agents = {
            role: type(agent)()
            for role, agent in heuristic_agents.items()
        }
        session_id = _maybe_create_session(env_client)
        try:
            bundle = _env_reset(env_client, task_name=task_name, seed=seed, session_id=session_id)
            done = False
            tick = 0
            while not done and tick < max_ticks_per_episode:
                obs = bundle["observations"]
                actions: Dict[str, Dict[str, Any]] = {}
                for role in ROLES:
                    role_obs = obs[role]
                    actionable = observation_has_actionable_signal(role, role_obs)
                    prompt = build_role_prompt(role_obs, role)
                    action = episode_agents[role].decide(role_obs)
                    actions[role] = action
                    if include_quiet or actionable:
                        rows.append({
                            "prompt": prompt,
                            "role": role,
                            "task_name": task_name,
                            "seed": seed,
                            "tick": tick + 1,
                            "actionable_signal": actionable,
                            "heuristic_operation": action.get("operation"),
                        })
                bundle = _env_step(env_client, actions, session_id=session_id)
                done = bool(bundle.get("done", False))
                tick += 1
        finally:
            _maybe_delete_session(env_client, session_id)
    return rows


def collect_rollouts(model, tokenizer, env_client, n_episodes: int = 4, task="task1_textbook"):
    """Returns list of {prompt, completion, env_reward, role} dicts."""
    rollouts: List[Dict[str, Any]] = []
    for ep in range(n_episodes):
        bundle = env_client.reset(task_name=task, seed=42 + ep)
        done = False
        while not done:
            obs = bundle["observations"]
            actions: Dict[str, Dict[str, Any]] = {}
            batch_this_tick: List[Dict[str, Any]] = []
            for role in ROLES:
                prompt = build_role_prompt(obs[role], role)
                completion = generate_completion(model, tokenizer, prompt)
                actions[role] = parse_action(completion, role)
                batch_this_tick.append({
                    "prompt": prompt,
                    "completion": completion,
                    "role": role,
                    "env_reward_placeholder": None,
                    "actionable_signal": observation_has_actionable_signal(role, obs[role]),
                })
            bundle = env_client.step(actions)
            for i, role in enumerate(ROLES):
                batch_this_tick[i]["env_reward_placeholder"] = float(bundle["rewards"][role])
            rollouts.extend(batch_this_tick)
            done = bundle["done"]
    return rollouts
