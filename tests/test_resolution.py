import numpy as np
from models import StepRequest, NurseAction, LabAction, PharmacistAction, PhysicianAction
from server.physiology import generate_patients
from server.resolution import resolve_step


def _bundle(**overrides):
    base = dict(
        nurse=NurseAction(operation="noop"),
        lab=LabAction(operation="noop"),
        pharmacist=PharmacistAction(operation="noop"),
        physician=PhysicianAction(operation="do_nothing"),
    )
    base.update(overrides)
    return StepRequest(**base)


def test_nurse_escalation_produces_flag():
    rng = np.random.default_rng(0)
    patients = generate_patients(rng, 5, 1, 0, 48)
    pid = patients[0].patient_id
    req = _bundle(nurse=NurseAction(
        operation="escalate_to_physician", patient_id=pid,
        urgency="critical", rationale="HR 130",
    ))
    flags, results, meta = resolve_step(req, patients, tick=1, lab_delay=1, physician_trust=1.0)
    assert len(flags) == 1
    assert flags[0].source_role == "nurse"
    assert flags[0].urgency == "critical"


def test_physician_order_records_antibiotic():
    rng = np.random.default_rng(0)
    patients = generate_patients(rng, 5, 1, 0, 48)
    septic = next(p for p in patients if p.infection_present)
    req = _bundle(
        nurse=NurseAction(
            operation="escalate_to_physician", patient_id=septic.patient_id,
            urgency="critical", rationale="HR 130",
        ),
        physician=PhysicianAction(
            operation="order_antibiotics", patient_id=septic.patient_id,
            drug="piperacillin_tazobactam",
        ),
    )
    flags, results, meta = resolve_step(req, patients, tick=1, lab_delay=1, physician_trust=1.0)
    assert meta["antibiotics_ordered"] is True
    assert septic.antibiotics_administered == "piperacillin_tazobactam"
    assert septic.antibiotic_tick == 1
    assert meta["on_valid_escalation"] is True


def test_physician_order_without_escalation_not_counted_valid():
    rng = np.random.default_rng(0)
    patients = generate_patients(rng, 5, 1, 0, 48)
    septic = next(p for p in patients if p.infection_present)
    req = _bundle(physician=PhysicianAction(
        operation="order_antibiotics", patient_id=septic.patient_id,
        drug="piperacillin_tazobactam",
    ))
    flags, results, meta = resolve_step(req, patients, tick=1, lab_delay=1, physician_trust=1.0)
    assert meta["antibiotics_ordered"] is True
    assert meta["on_valid_escalation"] is False


def test_low_trust_delays_physician():
    """Low trust: decision is still evaluated for reward (antibiotics_ordered=True,
    trust_penalised=True) but the antibiotic is NOT actually administered to the
    patient, so GRPO gets a gradient signal without bypassing the trust mechanic."""
    rng = np.random.default_rng(0)
    patients = generate_patients(rng, 5, 1, 0, 48)
    patient_id = patients[0].patient_id
    req = _bundle(physician=PhysicianAction(
        operation="order_antibiotics", patient_id=patient_id,
        drug="meropenem",
    ))
    flags, results, meta = resolve_step(req, patients, tick=1, lab_delay=1, physician_trust=0.3)
    # Decision quality is scored so GRPO gets a signal even in low-trust states.
    assert meta["antibiotics_ordered"] is True
    assert meta["trust_penalised"] is True
    # But the antibiotic must NOT actually be administered to the patient.
    p = next(p for p in patients if p.patient_id == patient_id)
    assert p.antibiotics_administered is None
    assert "delayed" in results["physician"].lower()
