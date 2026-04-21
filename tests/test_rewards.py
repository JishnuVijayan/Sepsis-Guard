import numpy as np
from models import NurseAction, LabAction, PharmacistAction, PhysicianAction, AgentFlag
from server.physiology import generate_patients
from server.rewards import (
    compute_nurse_reward, compute_lab_reward, compute_pharmacist_reward,
    compute_physician_reward, compute_terminal_team_score,
)


def test_nurse_correct_escalation_positive():
    rng = np.random.default_rng(0)
    patients = generate_patients(rng, 5, 1, 0, 48)
    septic = next(p for p in patients if p.infection_present)
    septic.infection_severity = 0.7
    action = NurseAction(
        operation="escalate_to_physician", patient_id=septic.patient_id,
        urgency="critical", rationale="HR 130",
    )
    r = compute_nurse_reward(action, patients, flags_this_tick=[AgentFlag(
        source_role="nurse", patient_id=septic.patient_id, flag_type="escalation",
        urgency="critical", rationale="", tick=5)], prior_flag_counts={})
    assert r >= 1.4


def test_nurse_false_escalation_negative():
    rng = np.random.default_rng(0)
    patients = generate_patients(rng, 5, 0, 1, 48)
    false_alarm = next((p for p in patients if p.is_false_alarm_patient), patients[0])
    action = NurseAction(
        operation="escalate_to_physician", patient_id=false_alarm.patient_id,
        urgency="critical", rationale="HR 108",
    )
    r = compute_nurse_reward(action, patients, flags_this_tick=[], prior_flag_counts={})
    assert r < 0


def test_pharmacist_resistant_antibiotic_penalty():
    rng = np.random.default_rng(0)
    patients = generate_patients(rng, 5, 1, 0, 48)
    action = PharmacistAction(
        operation="recommend_antibiotic", patient_id=patients[0].patient_id,
        drug="ciprofloxacin",
    )
    r_ok = compute_pharmacist_reward(action, patients)
    assert r_ok > 0
    from server import config as cfg
    cfg.ANTIBIOGRAM["test_bad"] = 0.5
    action2 = PharmacistAction(
        operation="recommend_antibiotic", patient_id=patients[0].patient_id, drug="test_bad",
    )
    r_bad = compute_pharmacist_reward(action2, patients)
    assert r_bad < 0


def test_terminal_score_perfect_response():
    rng = np.random.default_rng(0)
    patients = generate_patients(rng, 5, 1, 0, 48)
    septic = next(p for p in patients if p.infection_present)
    septic.sepsis_onset_tick = 5
    septic.antibiotics_administered = "meropenem"
    septic.antibiotic_tick = 6
    score, metrics = compute_terminal_team_score(
        patients, total_escalations=1, total_false_escalations=0,
        coordination_events={"total": 2, "max_possible": 2}, total_ticks=48,
    )
    assert score >= 0.80
    assert metrics["patients_treated_in_time"] == 1


def test_terminal_score_missed_all():
    rng = np.random.default_rng(0)
    patients = generate_patients(rng, 5, 1, 0, 48)
    score, metrics = compute_terminal_team_score(
        patients, total_escalations=10, total_false_escalations=8,
        coordination_events={"total": 0, "max_possible": 2}, total_ticks=48,
    )
    assert score < 0.40
