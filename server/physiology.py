from __future__ import annotations
import numpy as np
from typing import List, Dict, Any
from models import PatientState
from server.config import (
    Outcome, MentalStatus, VITAL_NORMAL_RANGES, LAB_NORMAL_RANGES,
)

ADMISSION_REASONS = [
    "pneumonia", "post-surgical", "UTI", "cellulitis",
    "abdominal pain", "chest pain", "altered mental status",
    "fever of unknown origin", "diabetic ketoacidosis", "dehydration",
]

COMMON_MEDICATIONS = [
    "metformin", "lisinopril", "aspirin", "atorvastatin",
    "omeprazole", "albuterol",
]

IMMUNOSUPPRESSANTS = ["prednisone", "methotrexate", "tacrolimus", "rituximab"]


def generate_patients(
    rng: np.random.Generator,
    n_patients: int,
    n_sepsis_cases: int,
    n_false_alarms: int,
    max_steps: int,
) -> List[PatientState]:
    """Create initial patient roster. Fully deterministic given rng."""
    sepsis_indices = set(rng.choice(n_patients, size=n_sepsis_cases, replace=False).tolist())
    remaining = [i for i in range(n_patients) if i not in sepsis_indices]
    false_alarm_indices = set(
        rng.choice(remaining, size=min(n_false_alarms, len(remaining)), replace=False).tolist()
    ) if n_false_alarms > 0 else set()

    patients: List[PatientState] = []
    for i in range(n_patients):
        is_sepsis = i in sepsis_indices
        is_false_alarm = i in false_alarm_indices
        age = int(rng.integers(25, 85))

        onset_tick = int(rng.integers(2, max(3, int(max_steps * 0.6)))) if is_sepsis else None

        meds = rng.choice(COMMON_MEDICATIONS,
                          size=int(rng.integers(0, 3)), replace=False).tolist()
        immunocompromised = bool(rng.random() < 0.25)
        if immunocompromised:
            meds.append(str(rng.choice(IMMUNOSUPPRESSANTS)))

        p = PatientState(
            patient_id=f"P{i+1:02d}",
            bed_number=i + 1,
            age=age,
            admission_reason=str(rng.choice(ADMISSION_REASONS)),
            heart_rate=float(rng.uniform(68, 92)),
            systolic_bp=float(rng.uniform(105, 135)),
            respiratory_rate=float(rng.uniform(13, 19)),
            temperature=float(rng.uniform(36.3, 37.1)),
            oxygen_saturation=float(rng.uniform(96, 99)),
            current_medications=meds,
            immunocompromised=immunocompromised,
            infection_present=is_sepsis,
            sepsis_onset_tick=onset_tick,
            is_false_alarm_patient=is_false_alarm,
        )
        patients.append(p)
    return patients


def advance_physiology(
    patient: PatientState,
    tick: int,
    rng: np.random.Generator,
) -> None:
    """Mutate patient state one tick forward. Deterministic given rng stream."""
    if patient.outcome in (Outcome.DIED, Outcome.RECOVERED):
        return

    patient.heart_rate += float(rng.normal(0, 1.2))
    patient.systolic_bp += float(rng.normal(0, 1.5))
    patient.respiratory_rate += float(rng.normal(0, 0.4))
    patient.temperature += float(rng.normal(0, 0.06))
    patient.oxygen_saturation = max(85.0,
        patient.oxygen_saturation + float(rng.normal(0, 0.3)))

    if patient.is_false_alarm_patient and tick > 4 and tick < 20:
        patient.heart_rate += float(rng.uniform(0.5, 1.5))
        patient.temperature += float(rng.uniform(0.02, 0.08))

    if patient.infection_present and patient.antibiotics_administered is None:
        if patient.sepsis_onset_tick is not None and tick >= patient.sepsis_onset_tick:
            patient.infection_severity = min(1.0,
                patient.infection_severity + float(rng.uniform(0.05, 0.15)))

            sev = patient.infection_severity
            if sev > 0.6:
                patient.outcome = Outcome.DETERIORATING
                temp_bump = 0.08 if patient.immunocompromised else 0.15
                patient.heart_rate += float(rng.uniform(5, 10))
                patient.systolic_bp -= float(rng.uniform(3, 8))
                patient.respiratory_rate += float(rng.uniform(1, 3))
                patient.temperature += float(rng.uniform(0.0, temp_bump))
            if sev > 0.85:
                patient.outcome = Outcome.SEPTIC_SHOCK
                patient.mental_status = MentalStatus.CONFUSED
            if sev > 0.95:
                patient.critical_ticks += 1
                if patient.critical_ticks >= 4:
                    patient.outcome = Outcome.DIED
                    patient.mental_status = MentalStatus.UNRESPONSIVE

    if patient.infection_present and patient.antibiotics_administered is not None:
        patient.infection_severity = max(0.0,
            patient.infection_severity - float(rng.uniform(0.10, 0.25)))
        if patient.infection_severity < 0.1 and patient.outcome != Outcome.DIED:
            patient.outcome = Outcome.RECOVERED

    patient.heart_rate = max(30.0, min(220.0, patient.heart_rate))
    patient.systolic_bp = max(50.0, min(220.0, patient.systolic_bp))
    patient.respiratory_rate = max(6.0, min(50.0, patient.respiratory_rate))
    patient.temperature = max(34.0, min(42.0, patient.temperature))
    patient.oxygen_saturation = max(60.0, min(100.0, patient.oxygen_saturation))


def mature_pending_labs(patient: PatientState, tick: int, rng: np.random.Generator) -> List[str]:
    """Fill in lab values whose delay has elapsed. Return list of newly-available test names."""
    ready = [test for test, due_tick in patient.pending_labs.items() if due_tick <= tick]
    if not ready:
        return []

    sev = patient.infection_severity if patient.infection_present else 0.0
    for test in ready:
        if test == "lactate":
            base = 1.2 + 3.0 * sev
            patient.lactate = round(base + float(rng.normal(0, 0.25)), 2)
        elif test == "wbc":
            base = 7.0 + 10.0 * sev
            if patient.immunocompromised:
                base = 7.0 + 3.0 * sev
            patient.wbc = round(base + float(rng.normal(0, 1.2)), 2)
        elif test == "procalcitonin":
            base = 0.2 + 3.5 * sev
            patient.procalcitonin = round(base + float(rng.normal(0, 0.15)), 3)
        elif test == "creatinine":
            base = 0.9 + 1.5 * sev
            patient.creatinine = round(base + float(rng.normal(0, 0.12)), 2)
        elif test == "blood_culture":
            patient.blood_culture_result = (
                "gram_positive" if (patient.infection_present and rng.random() < 0.5)
                else "gram_negative" if patient.infection_present
                else "no_growth"
            )
        patient.pending_labs.pop(test, None)
    return ready


def order_lab_test(patient: PatientState, test: str, tick: int, delay: int) -> bool:
    """Queue a lab test. Returns True if newly ordered, False if already pending or done."""
    if test in patient.pending_labs:
        return False
    patient.pending_labs[test] = tick + delay
    return True
