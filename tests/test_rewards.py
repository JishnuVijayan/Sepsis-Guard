import numpy as np
from models import NurseAction, LabAction, PharmacistAction, PhysicianAction, AgentFlag
from server.physiology import generate_patients
from server.config import Outcome
from server.rewards import (
    compute_nurse_reward, compute_lab_reward, compute_pharmacist_reward,
    compute_physician_reward, compute_team_reward_delta, compute_terminal_team_score,
)


def _snapshot(*patients):
    return {
        p.patient_id: {
            "first_detection_tick": p.first_detection_tick,
            "first_escalation_tick": p.first_escalation_tick,
            "first_critical_lab_tick": p.first_critical_lab_tick,
            "first_immunosuppression_flag_tick": p.first_immunosuppression_flag_tick,
            "first_antibiotic_tick": p.first_antibiotic_tick,
            "first_antibiotic_recommendation_tick": p.first_antibiotic_recommendation_tick,
            "best_antibiotic_recommendation_score": p.best_antibiotic_recommendation_score,
            "pending_labs": set(p.pending_labs.keys()),
        }
        for p in patients
    }


def test_nurse_correct_escalation_positive():
    rng = np.random.default_rng(0)
    patients = generate_patients(rng, 5, 1, 0, 48)
    septic = next(p for p in patients if p.infection_present)
    septic.sepsis_onset_tick = 10
    snap = _snapshot(septic)
    septic.first_escalation_tick = 4
    septic.first_detection_tick = 4
    action = NurseAction(
        operation="escalate_to_physician", patient_id=septic.patient_id,
        urgency="critical", rationale="HR 130",
    )
    r = compute_nurse_reward(action, patients, tick=4, prior_patient_snapshots=snap)
    assert r > 0.1


def test_nurse_false_escalation_negative():
    rng = np.random.default_rng(0)
    patients = generate_patients(rng, 5, 0, 1, 48)
    false_alarm = next((p for p in patients if p.is_false_alarm_patient), patients[0])
    snap = _snapshot(false_alarm)
    false_alarm.first_escalation_tick = 3
    false_alarm.first_detection_tick = 3
    action = NurseAction(
        operation="escalate_to_physician", patient_id=false_alarm.patient_id,
        urgency="critical", rationale="HR 108",
    )
    r = compute_nurse_reward(action, patients, tick=3, prior_patient_snapshots=snap)
    assert r < 0


def test_pharmacist_resistant_antibiotic_penalty():
    rng = np.random.default_rng(0)
    patients = generate_patients(rng, 5, 1, 0, 48)
    septic = next(p for p in patients if p.infection_present)
    septic.first_abnormal_vitals_tick = 4
    snap = _snapshot(septic)
    action = PharmacistAction(
        operation="recommend_antibiotic", patient_id=septic.patient_id,
        drug="ciprofloxacin",
    )
    septic.best_antibiotic_recommendation_score = 0.65
    r_ok = compute_pharmacist_reward(action, patients, tick=4, prior_patient_snapshots=snap, active_flags=[])
    assert r_ok > 0
    from server import config as cfg
    cfg.ANTIBIOGRAM["test_bad"] = 0.5
    try:
        action2 = PharmacistAction(
            operation="recommend_antibiotic", patient_id=septic.patient_id, drug="test_bad",
        )
        r_bad = compute_pharmacist_reward(action2, patients, tick=4, prior_patient_snapshots=_snapshot(septic), active_flags=[])
        assert r_bad < 0
    finally:
        cfg.ANTIBIOGRAM.pop("test_bad", None)


def test_terminal_score_perfect_response():
    rng = np.random.default_rng(0)
    patients = generate_patients(rng, 5, 1, 0, 48)
    septic = next(p for p in patients if p.infection_present)
    septic.sepsis_onset_tick = 20
    septic.first_detection_tick = 8
    septic.antibiotics_administered = "meropenem"
    septic.antibiotic_tick = 21
    septic.first_antibiotic_tick = 21
    septic.outcome = Outcome.RECOVERED
    score, metrics = compute_terminal_team_score(
        patients, total_alerts=1, total_false_alerts=0,
        coordination_events={"total": 2, "max_possible": 2}, total_ticks=48,
    )
    assert score >= 0.70
    assert metrics["patients_treated_in_time"] == 1


def test_terminal_score_missed_all():
    rng = np.random.default_rng(0)
    patients = generate_patients(rng, 5, 1, 0, 48)
    septic = next(p for p in patients if p.infection_present)
    septic.sepsis_onset_tick = 10
    score, metrics = compute_terminal_team_score(
        patients, total_alerts=10, total_false_alerts=8,
        coordination_events={"total": 0, "max_possible": 2}, total_ticks=48,
    )
    assert score < 0.40


def test_physician_do_nothing_penalty_on_multi_source():
    rng = np.random.default_rng(0)
    patients = generate_patients(rng, 5, 1, 0, 48)
    septic = next(p for p in patients if p.infection_present)
    septic.sepsis_onset_tick = 4
    flags = [
        AgentFlag(source_role="nurse", patient_id=septic.patient_id,
                  flag_type="escalation", urgency="critical", tick=5),
        AgentFlag(source_role="lab", patient_id=septic.patient_id,
                  flag_type="critical_lab", urgency="urgent", tick=5),
    ]
    action = PhysicianAction(operation="do_nothing")
    r = compute_physician_reward(action, patients, tick=5, phys_meta={}, active_flags=flags)
    assert r < 0


def test_physician_antibiotics_on_non_septic_penalized():
    rng = np.random.default_rng(0)
    patients = generate_patients(rng, 5, 1, 0, 48)
    healthy = next(p for p in patients if not p.infection_present)
    action = PhysicianAction(
        operation="order_antibiotics", patient_id=healthy.patient_id,
        drug="meropenem",
    )
    meta = {"antibiotics_ordered": True, "on_false_alarm": True, "on_valid_escalation": False}
    r = compute_physician_reward(action, patients, tick=5, phys_meta=meta, active_flags=[], prior_patient_snapshots=_snapshot(healthy))
    assert r < 0


def test_team_reward_delta_on_recovery():
    rng = np.random.default_rng(0)
    patients = generate_patients(rng, 5, 1, 0, 48)
    prev = {"lives_saved": 0, "lives_lost": 0}
    patients[0].outcome = Outcome.RECOVERED
    delta, new_metrics = compute_team_reward_delta(patients, prev)
    assert delta > 0
    assert new_metrics["lives_saved"] == 1


def test_team_reward_delta_on_death():
    rng = np.random.default_rng(0)
    patients = generate_patients(rng, 5, 1, 0, 48)
    prev = {"lives_saved": 0, "lives_lost": 0}
    patients[0].outcome = Outcome.DIED
    delta, new_metrics = compute_team_reward_delta(patients, prev)
    assert delta < 0
    assert new_metrics["lives_lost"] == 1


def test_nurse_penalty_for_multiple_unescalated_critical():
    rng = np.random.default_rng(0)
    patients = generate_patients(rng, 10, 3, 0, 96)
    septic_patients = [p for p in patients if p.infection_present]
    for p in septic_patients:
        p.sepsis_onset_tick = 4
    action = NurseAction(operation="noop")
    r = compute_nurse_reward(action, patients, tick=5, prior_patient_snapshots={})
    assert r == 0.0


def test_lab_followup_on_non_septic_penalized():
    rng = np.random.default_rng(0)
    patients = generate_patients(rng, 5, 1, 0, 48)
    healthy = next(p for p in patients if not p.infection_present)
    healthy.pending_labs["blood_culture"] = 9
    action = LabAction(
        operation="recommend_followup_test", patient_id=healthy.patient_id,
        test="blood_culture", reason="routine",
    )
    r = compute_lab_reward(action, patients, tick=5, prior_patient_snapshots=_snapshot(healthy))
    assert r >= 0


def test_pharmacist_recommend_without_signal_reduced():
    rng = np.random.default_rng(0)
    patients = generate_patients(rng, 5, 1, 0, 48)
    healthy = next(p for p in patients if not p.infection_present)
    action = PharmacistAction(
        operation="recommend_antibiotic", patient_id=healthy.patient_id,
        drug="meropenem",
    )
    r = compute_pharmacist_reward(action, patients, tick=4, prior_patient_snapshots=_snapshot(healthy), active_flags=[])
    assert r < 0
