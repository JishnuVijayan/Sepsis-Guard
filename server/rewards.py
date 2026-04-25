from __future__ import annotations

from typing import List, Dict, Any, Tuple, Optional

from models import (
    PatientState,
    AgentFlag,
    NurseAction,
    LabAction,
    PharmacistAction,
    PhysicianAction,
)
from server.config import Outcome, ANTIBIOGRAM

EARLY_DETECTION_WINDOW_TICKS = 24   # 12 hours in 30-minute ticks
OPTIMAL_DETECTION_TICKS = 12        # 6 hours early
LATE_DETECTION_WINDOW_TICKS = 6     # 3 hours late


def _find_patient(patients: List[PatientState], patient_id: Optional[str]) -> Optional[PatientState]:
    if not patient_id:
        return None
    return next((p for p in patients if p.patient_id == patient_id), None)


def _detection_utility_from_tick(patient: PatientState, detection_tick: Optional[int]) -> float:
    if not patient.infection_present:
        return -0.05 if detection_tick is not None else 0.0
    if patient.sepsis_onset_tick is None:
        return 0.0 if detection_tick is None else 0.15
    if detection_tick is None:
        return -2.0

    delta = detection_tick - patient.sepsis_onset_tick
    if delta < -EARLY_DETECTION_WINDOW_TICKS:
        return -0.05
    if delta <= -OPTIMAL_DETECTION_TICKS:
        ratio = (delta + EARLY_DETECTION_WINDOW_TICKS) / (EARLY_DETECTION_WINDOW_TICKS - OPTIMAL_DETECTION_TICKS)
        return -0.05 + ratio * 1.05
    if delta <= LATE_DETECTION_WINDOW_TICKS:
        ratio = (delta + OPTIMAL_DETECTION_TICKS) / (OPTIMAL_DETECTION_TICKS + LATE_DETECTION_WINDOW_TICKS)
        return 1.0 - ratio * 3.0
    return -2.0


def _treatment_utility_from_tick(patient: PatientState, antibiotic_tick: Optional[int]) -> float:
    if not patient.infection_present:
        return -0.4 if antibiotic_tick is not None else 0.0
    if patient.sepsis_onset_tick is None:
        return 0.0 if antibiotic_tick is None else 0.2
    if antibiotic_tick is None:
        return -1.5 if patient.outcome in (Outcome.SEPTIC_SHOCK, Outcome.DIED) else -0.6

    delay = antibiotic_tick - patient.sepsis_onset_tick
    if delay <= 2:
        return 1.0
    if delay <= 6:
        ratio = (delay - 2) / 4.0
        return 1.0 - ratio * 0.8
    if delay <= 12:
        ratio = (delay - 6) / 6.0
        return 0.2 - ratio * 1.2
    return -1.0


def _outcome_utility(patient: PatientState) -> float:
    if patient.outcome == Outcome.RECOVERED:
        return 1.0
    if patient.outcome == Outcome.DIED:
        return -1.5
    if patient.outcome == Outcome.SEPTIC_SHOCK:
        return -0.6
    if patient.outcome == Outcome.DETERIORATING:
        return -0.2
    return 0.0


def _patient_progress_utility(patient: PatientState, tick: int) -> float:
    detection = 0.0
    if patient.first_detection_tick is not None:
        detection = _detection_utility_from_tick(patient, patient.first_detection_tick)
    elif patient.infection_present and patient.sepsis_onset_tick is not None and tick > patient.sepsis_onset_tick:
        hours_late = (tick - patient.sepsis_onset_tick) / 2.0
        detection = -min(2.0, 0.15 * hours_late)

    treatment = 0.0
    if patient.first_antibiotic_tick is not None:
        treatment = _treatment_utility_from_tick(patient, patient.first_antibiotic_tick)
    elif patient.infection_present and patient.sepsis_onset_tick is not None and tick >= patient.sepsis_onset_tick:
        delay_hours = (tick - patient.sepsis_onset_tick) / 2.0
        treatment = -min(1.2, 0.12 * delay_hours)
        if patient.best_antibiotic_recommendation_score > 0:
            treatment += min(0.15, 0.15 * patient.best_antibiotic_recommendation_score)

    recommendation = 0.0
    if patient.antibiotics_administered is None and patient.best_antibiotic_recommendation_score > 0:
        recommendation = 0.12 * patient.best_antibiotic_recommendation_score

    return round(detection + treatment + recommendation + _outcome_utility(patient), 4)


def _snapshot_delta(
    patient: PatientState,
    prior_snapshot: Optional[Dict[str, Any]],
    tick: int,
    field: str,
    utility_fn,
) -> float:
    if prior_snapshot is None:
        prior_value = None
    else:
        prior_value = prior_snapshot.get(field)
    current_value = getattr(patient, field)
    if current_value is None or current_value == prior_value:
        return 0.0
    before = utility_fn(patient, prior_value)
    after = utility_fn(patient, current_value)
    return round(after - before, 4)


def compute_nurse_reward(
    action: NurseAction,
    patients: List[PatientState],
    tick: int,
    prior_patient_snapshots: Dict[str, Dict[str, Any]] | None = None,
) -> float:
    p = _find_patient(patients, action.patient_id)
    if action.operation == "noop":
        return 0.0
    if p is None:
        return -0.1
    prior = (prior_patient_snapshots or {}).get(p.patient_id)
    if action.operation == "escalate_to_physician":
        reward = _snapshot_delta(p, prior, tick, "first_detection_tick", _detection_utility_from_tick)
        if prior and prior.get("first_escalation_tick") is None and p.first_escalation_tick == tick:
            reward += 0.12
        if action.urgency in ("urgent", "critical") and p.qsofa_score >= 2:
            reward += 0.05
        if not p.infection_present:
            reward -= 0.25
        return round(reward, 4)
    if action.operation == "request_lab_test":
        had_pending = action.test_type in (prior or {}).get("pending_labs", set()) if prior else False
        if action.test_type and not had_pending and action.test_type in p.pending_labs:
            return 0.05 if p.first_abnormal_vitals_tick is not None else 0.02
        return 0.0
    if action.operation == "flag_concern":
        return 0.04 if p.first_abnormal_vitals_tick is not None and p.first_detection_tick is None else 0.0
    return 0.0


def compute_lab_reward(
    action: LabAction,
    patients: List[PatientState],
    tick: int,
    prior_patient_snapshots: Dict[str, Dict[str, Any]] | None = None,
) -> float:
    p = _find_patient(patients, action.patient_id)
    if action.operation == "noop":
        return 0.0
    if p is None:
        return -0.1
    prior = (prior_patient_snapshots or {}).get(p.patient_id)
    if action.operation == "flag_critical":
        reward = _snapshot_delta(p, prior, tick, "first_detection_tick", _detection_utility_from_tick) * 0.85
        if prior and prior.get("first_critical_lab_tick") is None and p.first_critical_lab_tick == tick:
            reward += 0.10
        if not p.infection_present:
            reward -= 0.20
        return round(reward, 4)
    if action.operation == "recommend_followup_test":
        if action.test and action.test in p.pending_labs:
            return 0.05 if p.first_detection_tick is None else 0.02
        return 0.0
    return 0.0


def compute_pharmacist_reward(
    action: PharmacistAction,
    patients: List[PatientState],
    tick: int,
    prior_patient_snapshots: Dict[str, Dict[str, Any]] | None = None,
    active_flags: List[AgentFlag] | None = None,
) -> float:
    p = _find_patient(patients, action.patient_id)
    if action.operation == "noop":
        return 0.0
    if p is None:
        return -0.1
    prior = (prior_patient_snapshots or {}).get(p.patient_id)
    if action.operation == "flag_immunosuppression":
        if p.immunocompromised and prior and prior.get("first_immunosuppression_flag_tick") is None:
            return 0.14
        return -0.05 if not p.immunocompromised else 0.02
    if action.operation == "recommend_antibiotic" and action.drug:
        prev_score = float((prior or {}).get("best_antibiotic_recommendation_score", 0.0))
        score_delta = p.best_antibiotic_recommendation_score - prev_score
        has_signal = any(
            f.patient_id == p.patient_id and f.flag_type == "critical_lab"
            for f in (active_flags or [])
        ) or p.first_abnormal_vitals_tick is not None
        resistance = ANTIBIOGRAM.get(action.drug, 0.5)
        if has_signal:
            penalty = max(0.0, resistance - 0.25) * 0.8
            return round(max(0.0, score_delta) - penalty + (0.04 if p.infection_present else -0.03), 4)
        return round(-0.02 - resistance * 0.1, 4)
    if action.operation == "flag_interaction":
        return 0.05 if len(p.current_medications) >= 2 else -0.02
    return 0.0


def compute_physician_reward(
    action: PhysicianAction,
    patients: List[PatientState],
    tick: int,
    phys_meta: Dict[str, Any],
    active_flags: List[AgentFlag],
    prior_patient_snapshots: Dict[str, Dict[str, Any]] | None = None,
) -> float:
    p = _find_patient(patients, action.patient_id)
    if action.operation == "do_nothing":
        untreated_flagged = 0
        for flag in active_flags:
            candidate = _find_patient(patients, flag.patient_id)
            if candidate and candidate.infection_present and candidate.antibiotics_administered is None:
                if flag.flag_type in ("escalation", "critical_lab"):
                    untreated_flagged += 1
        return round(-min(0.4, 0.1 * untreated_flagged), 4)
    if p is None:
        return -0.1
    prior = (prior_patient_snapshots or {}).get(p.patient_id)
    if action.operation == "order_antibiotics" and phys_meta.get("antibiotics_ordered"):
        if not p.infection_present:
            return -0.25
        reward = _snapshot_delta(p, prior, tick, "first_antibiotic_tick", _treatment_utility_from_tick)
        if reward == 0.0 and phys_meta.get("trust_penalised"):
            reward = max(0.0, _treatment_utility_from_tick(p, tick)) * 0.5
        reward += _snapshot_delta(p, prior, tick, "first_detection_tick", _detection_utility_from_tick) * 0.25
        return round(reward, 4)
    if action.operation == "admit_to_icu":
        return 0.22 if p.outcome == Outcome.SEPTIC_SHOCK else -0.08
    if action.operation == "order_lab_test":
        return 0.03 if p.first_detection_tick is None else 0.01
    return 0.0


def compute_team_reward_delta(
    patients: List[PatientState],
    prev_metrics: Dict[str, Any],
    tick: Optional[int] = None,
) -> Tuple[float, Dict[str, Any]]:
    if tick is None:
        lives_saved_now = sum(1 for p in patients if p.outcome == Outcome.RECOVERED)
        lives_lost_now = sum(1 for p in patients if p.outcome == Outcome.DIED)
        delta_saved = lives_saved_now - prev_metrics.get("lives_saved", 0)
        delta_lost = lives_lost_now - prev_metrics.get("lives_lost", 0)
        team_delta = 1.0 * delta_saved - 0.5 * delta_lost
        new_metrics = {"lives_saved": lives_saved_now, "lives_lost": lives_lost_now}
        return team_delta, new_metrics

    prev_utilities = prev_metrics.get("patient_utilities", {})
    current_utilities = {
        p.patient_id: _patient_progress_utility(p, tick)
        for p in patients
    }
    team_delta = round(
        sum(current_utilities.values()) - sum(prev_utilities.get(pid, 0.0) for pid in current_utilities),
        4,
    )
    lives_saved_now = sum(1 for p in patients if p.outcome == Outcome.RECOVERED)
    lives_lost_now = sum(1 for p in patients if p.outcome == Outcome.DIED)
    return team_delta, {
        "lives_saved": lives_saved_now,
        "lives_lost": lives_lost_now,
        "patient_utilities": current_utilities,
    }


def compute_terminal_team_score(
    patients: List[PatientState],
    total_alerts: int,
    total_false_alerts: int,
    coordination_events: Dict[str, int],
    total_ticks: int,
) -> Tuple[float, Dict[str, Any]]:
    del total_ticks
    sepsis_patients = [p for p in patients if p.infection_present]
    n_sepsis = max(1, len(sepsis_patients))

    raw_detection = sum(_detection_utility_from_tick(p, p.first_detection_tick) for p in patients)
    inactive_detection = -2.0 * len(sepsis_patients)
    optimal_detection = 1.0 * len(sepsis_patients)
    denom = max(1e-6, optimal_detection - inactive_detection)
    detection_utility = max(0.0, min(1.0, (raw_detection - inactive_detection) / denom))

    raw_treatment = sum(_treatment_utility_from_tick(p, p.first_antibiotic_tick) for p in sepsis_patients)
    treatment_utility = max(0.0, min(1.0, (raw_treatment + 1.5 * n_sepsis) / (2.5 * n_sepsis)))

    recovered = sum(1 for p in sepsis_patients if p.outcome == Outcome.RECOVERED)
    died = sum(1 for p in sepsis_patients if p.outcome == Outcome.DIED)
    outcome_score = max(0.0, min(1.0, (recovered - 0.5 * died) / n_sepsis))

    false_alarm_rate = total_false_alerts / max(1, total_alerts)
    precision_score = max(0.0, 1.0 - min(1.0, false_alarm_rate))

    coord_events = coordination_events.get("total", 0)
    coord_max = max(1, coordination_events.get("max_possible", 1))
    coord_score = min(1.0, coord_events / coord_max)

    score = (
        0.45 * detection_utility
        + 0.25 * treatment_utility
        + 0.15 * outcome_score
        + 0.10 * precision_score
        + 0.05 * coord_score
    )

    treated_in_time = sum(
        1 for p in sepsis_patients
        if p.first_antibiotic_tick is not None
        and p.sepsis_onset_tick is not None
        and (p.first_antibiotic_tick - p.sepsis_onset_tick) / 2.0 <= 4.0
    )
    timely_detections = sum(
        1 for p in sepsis_patients
        if p.first_detection_tick is not None
        and p.sepsis_onset_tick is not None
        and (p.first_detection_tick - p.sepsis_onset_tick) <= LATE_DETECTION_WINDOW_TICKS
    )
    metrics = {
        "patients_with_sepsis": len(sepsis_patients),
        "patients_treated_in_time": treated_in_time,
        "patients_detected_in_window": timely_detections,
        "false_alarm_rate": round(false_alarm_rate, 3),
        "coordination_score": round(coord_score, 3),
        "detection_utility": round(detection_utility, 3),
        "treatment_utility": round(treatment_utility, 3),
        "outcome_score": round(outcome_score, 3),
        "precision_score": round(precision_score, 3),
    }
    return round(max(0.0, min(1.0, score)), 4), metrics
