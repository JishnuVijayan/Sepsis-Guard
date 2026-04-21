from __future__ import annotations
import numpy as np
from typing import Any, Dict, Optional, List

try:
    from openenv.core.env_server.interfaces import Environment
    from openenv.core.env_server.types import EnvironmentMetadata
except ImportError:
    class Environment:
        pass
    class EnvironmentMetadata:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

from models import (
    PatientState, StepRequest, AgentFlag, SepsisState,
    NurseAction, LabAction, PharmacistAction, PhysicianAction,
)
from server.config import DEFAULT_TASK, TASK_CONFIGS, Outcome
from server.physiology import (
    generate_patients, advance_physiology, mature_pending_labs,
)
from server.observations import build_observations
from server.resolution import resolve_step
from server.rewards import (
    compute_nurse_reward, compute_lab_reward, compute_pharmacist_reward,
    compute_physician_reward, compute_team_reward_delta,
    compute_terminal_team_score,
)


class SepsisEnvironment(Environment):
    SUPPORTS_CONCURRENT_SESSIONS: bool = True
    _last_grader_data: Dict[str, Any] = {}

    def __init__(self) -> None:
        super().__init__()
        self._task_name: str = DEFAULT_TASK
        self._task_cfg: Dict[str, Any] = TASK_CONFIGS[DEFAULT_TASK]
        self._rng: np.random.Generator = np.random.default_rng(42)
        self._seed: int = 42
        self._tick: int = 0
        self._done: bool = True
        self._patients: List[PatientState] = []
        self._nurse_assignment: Dict[str, List[str]] = {}
        self._active_flags: List[AgentFlag] = []
        self._physician_trust: float = 1.0
        self._cumulative_rewards: Dict[str, float] = {}
        self._total_team_reward: float = 0.0
        self._last_results: Dict[str, Optional[str]] = {}
        self._normalized_score: Optional[float] = None
        self._total_escalations: int = 0
        self._total_false_escalations: int = 0
        self._coord_events: Dict[str, int] = {"total": 0, "max_possible": 1}
        self._flag_counts: Dict[str, int] = {}
        self._prev_lives_metrics: Dict[str, int] = {"lives_saved": 0, "lives_lost": 0}

    def reset(
        self, seed: Optional[int] = None, episode_id: Optional[str] = None,
        task_name: Optional[str] = None, **kwargs: Any,
    ) -> Dict[str, Any]:
        resolved = task_name or episode_id or DEFAULT_TASK
        if resolved not in TASK_CONFIGS:
            resolved = DEFAULT_TASK
        self._task_name = resolved
        self._task_cfg = TASK_CONFIGS[resolved]
        self._seed = int(seed) if seed is not None else 42
        self._rng = np.random.default_rng(self._seed)
        self._tick = 1
        self._done = False
        self._patients = generate_patients(
            self._rng,
            n_patients=self._task_cfg["n_patients"],
            n_sepsis_cases=self._task_cfg["n_sepsis_cases"],
            n_false_alarms=self._task_cfg["n_false_alarms"],
            max_steps=self._task_cfg["max_steps"],
        )
        n_assigned = min(5, len(self._patients))
        self._nurse_assignment = {
            "nurse": [p.patient_id for p in self._patients[:n_assigned]]
        }
        self._active_flags = []
        self._physician_trust = 1.0
        self._cumulative_rewards = {
            "nurse": 0.0, "lab": 0.0, "pharmacist": 0.0, "physician": 0.0, "team": 0.0,
        }
        self._total_team_reward = 0.0
        self._last_results = {}
        self._normalized_score = None
        self._total_escalations = 0
        self._total_false_escalations = 0
        self._coord_events = {
            "total": 0, "max_possible": max(2, 2 * self._task_cfg["n_sepsis_cases"]),
        }
        self._flag_counts = {}
        self._prev_lives_metrics = {"lives_saved": 0, "lives_lost": 0}

        for p in self._patients:
            mature_pending_labs(p, self._tick, self._rng)

        return self._build_obs_bundle()

    def step(self, action: Any, **kwargs: Any) -> Dict[str, Any]:
        if self._done:
            return self._build_obs_bundle(done=True)

        if isinstance(action, dict):
            if "actions" in action:
                action = action["actions"]
            request = StepRequest(
                nurse=NurseAction(**action["nurse"]),
                lab=LabAction(**action["lab"]),
                pharmacist=PharmacistAction(**action["pharmacist"]),
                physician=PhysicianAction(**action["physician"]),
            )
        elif isinstance(action, StepRequest):
            request = action
        else:
            request = StepRequest.model_validate(action)

        new_flags, results, phys_meta = resolve_step(
            request, self._patients, self._tick,
            self._task_cfg["lab_result_delay"], self._physician_trust,
        )
        self._active_flags = new_flags
        self._last_results = results

        for f in new_flags:
            self._flag_counts[f.patient_id] = self._flag_counts.get(f.patient_id, 0) + 1

        flagged_by_source: Dict[str, set] = {}
        for f in new_flags:
            flagged_by_source.setdefault(f.patient_id, set()).add(f.source_role)
        for pid, sources in flagged_by_source.items():
            if len(sources) >= 2:
                self._coord_events["total"] = min(
                    self._coord_events["max_possible"],
                    self._coord_events["total"] + 1,
                )

        r_nurse = compute_nurse_reward(request.nurse, self._patients, new_flags, self._flag_counts)
        r_lab = compute_lab_reward(request.lab, self._patients, new_flags)
        r_pharm = compute_pharmacist_reward(request.pharmacist, self._patients)
        r_phys = compute_physician_reward(
            request.physician, self._patients, self._tick, phys_meta, new_flags,
        )

        for p in self._patients:
            advance_physiology(p, self._tick, self._rng)
            mature_pending_labs(p, self._tick, self._rng)

        team_delta, self._prev_lives_metrics = compute_team_reward_delta(
            self._patients, self._prev_lives_metrics,
        )

        rewards_raw = {"nurse": r_nurse, "lab": r_lab, "pharmacist": r_pharm, "physician": r_phys}
        per_agent_rewards = {k: float(v + team_delta) for k, v in rewards_raw.items()}
        for k, v in per_agent_rewards.items():
            self._cumulative_rewards[k] += v
        self._cumulative_rewards["team"] += team_delta
        self._total_team_reward += team_delta

        for f in new_flags:
            if f.source_role == "nurse" and f.flag_type == "escalation":
                self._total_escalations += 1
                p = next((p for p in self._patients if p.patient_id == f.patient_id), None)
                if p is not None and not p.infection_present:
                    self._total_false_escalations += 1

        if self._total_escalations > 0:
            false_rate = self._total_false_escalations / self._total_escalations
            self._physician_trust = max(0.2, min(1.0, 1.0 - 0.8 * false_rate))

        self._tick += 1
        self._done = self._tick > self._task_cfg["max_steps"]

        terminal_bonus = 0.0
        if self._done:
            score, metrics = compute_terminal_team_score(
                self._patients, self._total_escalations, self._total_false_escalations,
                self._coord_events, self._task_cfg["max_steps"],
            )
            self._normalized_score = score
            terminal_bonus = score
            for k in per_agent_rewards:
                per_agent_rewards[k] += terminal_bonus
                self._cumulative_rewards[k] += terminal_bonus
            SepsisEnvironment._last_grader_data = {
                "task": self._task_name,
                "score": score,
                "agent_rewards": dict(self._cumulative_rewards),
                "ticks_completed": self._tick - 1,
                "metrics": {
                    **metrics,
                    "total_escalations": self._total_escalations,
                    "total_false_escalations": self._total_false_escalations,
                    "success_threshold": self._task_cfg["success_threshold"],
                    "passed": score >= self._task_cfg["success_threshold"],
                },
            }

        return self._build_obs_bundle(
            per_agent_rewards=per_agent_rewards, team_reward=team_delta, done=self._done,
        )

    @property
    def state(self) -> SepsisState:
        return SepsisState(
            episode_id=None, step_count=self._tick,
            task_name=self._task_name,
            task_description=self._task_cfg["description"],
            tick=self._tick, max_ticks=self._task_cfg["max_steps"],
            patients=[
                {
                    "patient_id": p.patient_id,
                    "bed_number": p.bed_number,
                    "outcome": p.outcome.value,
                    "infection_severity": round(p.infection_severity, 3),
                    "antibiotics_administered": p.antibiotics_administered,
                    "icu_admitted": p.icu_admitted,
                }
                for p in self._patients
            ],
            active_flags=self._active_flags,
            physician_trust=round(self._physician_trust, 3),
            cumulative_rewards={k: round(v, 4) for k, v in self._cumulative_rewards.items()},
            total_team_reward=round(self._total_team_reward, 4),
            normalized_score=self._normalized_score,
            patients_with_sepsis=sum(1 for p in self._patients if p.infection_present),
            total_escalations=self._total_escalations,
        )

    def get_metadata(self) -> EnvironmentMetadata:
        return EnvironmentMetadata(
            name="sepsisguard",
            description=(
                "Multi-agent sepsis coordination environment — 4 agents with "
                "asymmetric information must coordinate to detect and treat sepsis "
                "under time pressure and alarm fatigue constraints."
            ),
            version="0.1.0",
            author="SepsisGuard team",
        )

    def close(self) -> None:
        pass

    def _build_obs_bundle(
        self, per_agent_rewards: Optional[Dict[str, float]] = None,
        team_reward: float = 0.0, done: bool = False,
    ) -> Dict[str, Any]:
        pending = [
            {"patient_id": p.patient_id, "test": t, "due_tick": dt}
            for p in self._patients for t, dt in p.pending_labs.items()
        ]
        obs = build_observations(
            patients=self._patients,
            nurse_assignment=self._nurse_assignment,
            active_flags_this_tick=self._active_flags,
            tick=self._tick, max_ticks=self._task_cfg["max_steps"],
            task_name=self._task_name,
            physician_trust=self._physician_trust,
            cumulative_rewards=self._cumulative_rewards,
            last_results=self._last_results,
            pending_labs_summary=pending,
        )
        rewards = per_agent_rewards or {"nurse": 0.0, "lab": 0.0, "pharmacist": 0.0, "physician": 0.0}
        for role, o in obs.items():
            o.done = done
            o.reward = round(rewards.get(role, 0.0), 4)
            if done:
                o.normalized_score = self._normalized_score

        return {
            "observations": {k: v.model_dump() for k, v in obs.items()},
            "rewards": {k: round(rewards.get(k, 0.0), 4) for k in ("nurse", "lab", "pharmacist", "physician")},
            "team_reward": round(team_reward, 4),
            "done": done,
            "info": {
                "tick": self._tick,
                "physician_trust": round(self._physician_trust, 3),
                "active_flag_count": len(self._active_flags),
            },
        }
