import numpy as np
from server.physiology import generate_patients
from server.observations import build_observations


def test_no_is_real_leakage():
    rng = np.random.default_rng(42)
    patients = generate_patients(rng, 10, 3, 0, 96)
    obs = build_observations(
        patients=patients,
        nurse_assignment={"nurse": [p.patient_id for p in patients[:5]]},
        active_flags_this_tick=[],
        tick=1, max_ticks=96, task_name="task2_atypical",
        physician_trust=1.0,
        cumulative_rewards={"nurse": 0, "lab": 0, "pharmacist": 0, "physician": 0},
        last_results={},
        pending_labs_summary=[],
    )
    for role, o in obs.items():
        s = o.model_dump_json()
        assert "infection_present" not in s, f"infection_present leaked in {role}"
        assert "infection_severity" not in s, f"infection_severity leaked in {role}"
        assert "is_false_alarm_patient" not in s, f"is_false_alarm_patient leaked in {role}"
        assert "sepsis_onset_tick" not in s, f"sepsis_onset_tick leaked in {role}"


def test_nurse_sees_only_assigned_patients():
    rng = np.random.default_rng(0)
    patients = generate_patients(rng, 10, 3, 0, 96)
    assigned = [p.patient_id for p in patients[:5]]
    obs = build_observations(
        patients=patients,
        nurse_assignment={"nurse": assigned},
        active_flags_this_tick=[],
        tick=1, max_ticks=96, task_name="task2_atypical",
        physician_trust=1.0,
        cumulative_rewards={"nurse": 0, "lab": 0, "pharmacist": 0, "physician": 0},
        last_results={},
        pending_labs_summary=[],
    )
    nurse_patient_ids = {v["patient_id"] for v in obs["nurse"].patient_vitals}
    assert nurse_patient_ids == set(assigned)


def test_lab_does_not_see_vitals():
    rng = np.random.default_rng(0)
    patients = generate_patients(rng, 10, 3, 0, 96)
    obs = build_observations(
        patients=patients,
        nurse_assignment={"nurse": [p.patient_id for p in patients[:5]]},
        active_flags_this_tick=[],
        tick=1, max_ticks=96, task_name="task2_atypical",
        physician_trust=1.0,
        cumulative_rewards={"nurse": 0, "lab": 0, "pharmacist": 0, "physician": 0},
        last_results={},
        pending_labs_summary=[],
    )
    for lab_view in obs["lab"].lab_results:
        assert "heart_rate" not in lab_view
        assert "systolic_bp" not in lab_view


def test_physician_sees_escalated_only():
    from models import AgentFlag
    rng = np.random.default_rng(0)
    patients = generate_patients(rng, 10, 3, 0, 96)
    flag = AgentFlag(
        source_role="nurse", patient_id=patients[0].patient_id,
        flag_type="escalation", urgency="critical",
        rationale="HR 130, confused", tick=5,
    )
    obs = build_observations(
        patients=patients,
        nurse_assignment={"nurse": [p.patient_id for p in patients[:5]]},
        active_flags_this_tick=[flag],
        tick=5, max_ticks=96, task_name="task2_atypical",
        physician_trust=1.0,
        cumulative_rewards={"nurse": 0, "lab": 0, "pharmacist": 0, "physician": 0},
        last_results={},
        pending_labs_summary=[],
    )
    physician_ids = {s["patient_id"] for s in obs["physician"].known_patient_summaries}
    assert physician_ids == {patients[0].patient_id}
