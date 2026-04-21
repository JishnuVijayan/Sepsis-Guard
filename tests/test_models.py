from models import NurseAction, LabAction, PharmacistAction, PhysicianAction, StepRequest, PatientState
from server.config import MentalStatus, Outcome


def test_nurse_action_valid():
    a = NurseAction(operation="escalate_to_physician", patient_id="P01",
                    urgency="critical", rationale="HR 130, BP falling")
    assert a.operation == "escalate_to_physician"


def test_step_request_bundle():
    req = StepRequest(
        nurse=NurseAction(operation="noop"),
        lab=LabAction(operation="noop"),
        pharmacist=PharmacistAction(operation="noop"),
        physician=PhysicianAction(operation="do_nothing"),
    )
    assert req.nurse.operation == "noop"


def test_patient_state_defaults():
    p = PatientState(patient_id="P01", bed_number=1, age=65,
                     admission_reason="pneumonia",
                     heart_rate=88, systolic_bp=120, respiratory_rate=16,
                     temperature=37.0, oxygen_saturation=97)
    assert p.outcome == Outcome.STABLE
    assert p.mental_status == MentalStatus.ALERT
    assert p.lactate is None
