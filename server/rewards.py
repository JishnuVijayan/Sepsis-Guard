from __future__ import annotations
from typing import List, Dict, Any, Tuple
from models import (
    PatientState, AgentFlag, StepRequest,
    NurseAction, LabAction, PharmacistAction, PhysicianAction,
)
from server.config import Outcome, ANTIBIOGRAM


def _patient_has_active_sepsis(p: PatientState) -> bool:
    return p.infection_present and p.infection_severity > 0.5 and p.outcome not in (
        Outcome.RECOVERED, Outcome.DIED,
    )


def _patient_truly_sepsis(p: PatientState) -> bool:
    return p.infection_present


def compute_nurse_reward(
    action: NurseAction, patients: List[PatientState], flags_this_tick: List[AgentFlag],
    prior_flag_counts: Dict[str, int],
) -> float:
    r = 0.0
    if action.operation == "escalate_to_physician" and action.patient_id:
        p = next((p for p in patients if p.patient_id == action.patient_id), None)
        if p is None: return -0.2
        if _patient_truly_sepsis(p) and p.infection_severity > 0.4:
            r += 1.5
        elif p.is_false_alarm_patient or not p.infection_present:
            r -= 1.0
        else:
            r += 0.3
    elif action.operation == "flag_concern" and action.patient_id:
        p = next((p for p in patients if p.patient_id == action.patient_id), None)
        if p and _patient_truly_sepsis(p):
            r += 0.5
        if prior_flag_counts.get(action.patient_id, 0) >= 5:
            r -= 0.2
    nurse_patients_escalated = {
        f.patient_id for f in flags_this_tick
        if f.source_role == "nurse" and f.flag_type == "escalation"
    }
    for p in patients:
        if (_patient_has_active_sepsis(p) and p.infection_severity > 0.8
                and p.patient_id not in nurse_patients_escalated
                and p.antibiotics_administered is None):
            r -= 2.0
    return r


def compute_lab_reward(
    action: LabAction, patients: List[PatientState], flags_this_tick: List[AgentFlag],
    prior_flag_counts: Dict[str, int] | None = None,
) -> float:
    r = 0.0
    if action.operation == "flag_critical" and action.patient_id:
        p = next((p for p in patients if p.patient_id == action.patient_id), None)
        if p is None: return -0.3
        abnormal_lab = any([
            p.lactate is not None and p.lactate > 2.0,
            p.wbc is not None and (p.wbc > 12 or p.wbc < 4),
            p.procalcitonin is not None and p.procalcitonin > 0.5,
        ])
        if abnormal_lab and _patient_truly_sepsis(p):
            prior = (prior_flag_counts or {}).get(action.patient_id, 0)
            if prior == 0:
                r += 1.2
            elif prior < 3:
                r += 0.3
            else:
                r -= 0.1
        elif not abnormal_lab:
            r -= 0.8
    elif action.operation == "recommend_followup_test" and action.patient_id:
        p = next((p for p in patients if p.patient_id == action.patient_id), None)
        if p is None:
            r -= 0.3
        elif _patient_truly_sepsis(p):
            r += 0.4
        else:
            r -= 0.2
    return r


def compute_pharmacist_reward(
    action: PharmacistAction, patients: List[PatientState],
    active_flags: List[AgentFlag] | None = None,
    prior_flag_counts: Dict[str, int] | None = None,
) -> float:
    r = 0.0
    prior = (prior_flag_counts or {}).get(action.patient_id or "", 0) if action.patient_id else 0
    if action.operation == "flag_immunosuppression" and action.patient_id:
        p = next((p for p in patients if p.patient_id == action.patient_id), None)
        if p and p.immunocompromised:
            r += 1.0 if prior == 0 else (0.2 if prior < 3 else -0.1)
        else:
            r -= 0.3
    elif action.operation == "recommend_antibiotic" and action.patient_id:
        p = next((p for p in patients if p.patient_id == action.patient_id), None)
        if p is None:
            r -= 0.3
        elif action.drug and action.drug in ANTIBIOGRAM:
            resistance = ANTIBIOGRAM[action.drug]
            has_signal = False
            if active_flags:
                has_signal = any(
                    f.patient_id == action.patient_id and f.flag_type == "critical_lab"
                    for f in active_flags
                )
            if not has_signal:
                has_signal = p.infection_present and p.infection_severity > 0.3
            if resistance > 0.40:
                r -= 0.6
            elif has_signal:
                r += 0.8 if prior < 2 else 0.1
            else:
                r += 0.1 if prior == 0 else 0.0
    elif action.operation == "flag_interaction" and action.patient_id:
        p = next((p for p in patients if p.patient_id == action.patient_id), None)
        if p and len(p.current_medications) >= 2:
            r += 0.3
        else:
            r -= 0.1
    return r


def compute_physician_reward(
    action: PhysicianAction, patients: List[PatientState], tick: int,
    phys_meta: Dict[str, Any], active_flags: List[AgentFlag],
) -> float:
    r = 0.0
    if phys_meta.get("antibiotics_ordered"):
        p = next((p for p in patients if p.patient_id == action.patient_id), None)
        if p is None: return 0.0
        if phys_meta.get("on_false_alarm") or not p.infection_present:
            r -= 1.5
        elif phys_meta.get("on_valid_escalation"):
            onset = p.sepsis_onset_tick
            if onset is not None:
                ticks_since_onset = tick - onset
                hours_since = ticks_since_onset / 2.0
                if hours_since <= 1.0:
                    r += 2.0
                elif hours_since <= 3.0:
                    r += 1.0
            else:
                r += 0.5
        else:
            r += 0.2
    if phys_meta.get("icu_ordered"):
        p = next((p for p in patients if p.patient_id == action.patient_id), None)
        if p and p.outcome == Outcome.SEPTIC_SHOCK:
            r += 1.5
        else:
            r -= 0.5
    if action.operation == "do_nothing":
        valid_escalation_patients = set()
        for f in active_flags:
            if f.flag_type == "escalation" and f.urgency in ("urgent", "critical"):
                valid_escalation_patients.add(f.patient_id)
        for pid in valid_escalation_patients:
            p = next((p for p in patients if p.patient_id == pid), None)
            if p and _patient_truly_sepsis(p) and p.antibiotics_administered is None:
                sources = {f.source_role for f in active_flags if f.patient_id == pid}
                if len(sources) >= 2:
                    r -= 3.0
                    break
    return r


def compute_team_reward_delta(
    patients: List[PatientState], prev_metrics: Dict[str, int],
) -> Tuple[float, Dict[str, int]]:
    lives_saved_now = sum(1 for p in patients if p.outcome == Outcome.RECOVERED)
    lives_lost_now = sum(1 for p in patients if p.outcome == Outcome.DIED)
    delta_saved = lives_saved_now - prev_metrics.get("lives_saved", 0)
    delta_lost = lives_lost_now - prev_metrics.get("lives_lost", 0)
    team_delta = 0.1 * delta_saved - 0.05 * delta_lost
    new_metrics = {"lives_saved": lives_saved_now, "lives_lost": lives_lost_now}
    return team_delta, new_metrics


def compute_terminal_team_score(
    patients: List[PatientState], total_escalations: int, total_false_escalations: int,
    coordination_events: Dict[str, int], total_ticks: int,
) -> Tuple[float, Dict[str, Any]]:
    sepsis_patients = [p for p in patients if p.infection_present]
    n_sepsis = max(1, len(sepsis_patients))
    treated_in_time = sum(
        1 for p in sepsis_patients
        if p.antibiotics_administered is not None
        and p.sepsis_onset_tick is not None
        and p.antibiotic_tick is not None
        and (p.antibiotic_tick - p.sepsis_onset_tick) / 2.0 <= 4.0
    )
    false_alarm_rate = (total_false_escalations / max(1, total_escalations))
    coord_events = coordination_events.get("total", 0)
    coord_max = max(1, coordination_events.get("max_possible", 1))
    coord_score = min(1.0, coord_events / coord_max)
    avg_time_to_abx = 0.0
    treated = [p for p in sepsis_patients if p.antibiotics_administered is not None]
    if treated:
        avg_time_to_abx = sum(
            (p.antibiotic_tick - p.sepsis_onset_tick) / 2.0 for p in treated
        ) / len(treated)
    time_efficiency = max(0.0, 1.0 - avg_time_to_abx / 6.0)

    score = (
        0.40 * (treated_in_time / n_sepsis)
        + 0.25 * (1.0 - min(1.0, false_alarm_rate))
        + 0.20 * coord_score
        + 0.15 * time_efficiency
    )
    metrics = {
        "patients_with_sepsis": len(sepsis_patients),
        "patients_treated_in_time": treated_in_time,
        "false_alarm_rate": round(false_alarm_rate, 3),
        "coordination_score": round(coord_score, 3),
        "time_efficiency": round(time_efficiency, 3),
        "avg_time_to_antibiotics_hours": round(avg_time_to_abx, 2),
    }
    return round(max(0.0, min(1.0, score)), 4), metrics
