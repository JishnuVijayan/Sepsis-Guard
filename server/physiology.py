from __future__ import annotations

import numpy as np
from typing import List

from models import PatientState
from server.config import (
    LAB_NORMAL_RANGES,
    Outcome,
    MentalStatus,
    VITAL_NORMAL_RANGES,
)

ADMISSION_REASONS = [
    "pneumonia",
    "post-surgical",
    "UTI",
    "cellulitis",
    "abdominal pain",
    "chest pain",
    "altered mental status",
    "fever of unknown origin",
    "diabetic ketoacidosis",
    "dehydration",
]

COMMON_MEDICATIONS = [
    "metformin",
    "lisinopril",
    "aspirin",
    "atorvastatin",
    "omeprazole",
    "albuterol",
]

IMMUNOSUPPRESSANTS = ["prednisone", "methotrexate", "tacrolimus", "rituximab"]


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _map_from_pressures(sbp: float, dbp: float) -> float:
    return dbp + (sbp - dbp) / 3.0


def _mark_first_abnormal_vitals(patient: PatientState, tick: int) -> None:
    abnormal = any([
        patient.heart_rate > 110,
        patient.systolic_bp < 100,
        patient.mean_arterial_pressure < 65,
        patient.respiratory_rate >= 22,
        patient.temperature >= 38.0,
        patient.oxygen_saturation < 94,
        patient.mental_status != MentalStatus.ALERT,
    ])
    if abnormal and patient.first_abnormal_vitals_tick is None:
        patient.first_abnormal_vitals_tick = tick


def _update_clinical_scores(patient: PatientState, tick: int) -> None:
    qsofa = 0
    qsofa += int(patient.respiratory_rate >= 22)
    qsofa += int(patient.systolic_bp <= 100)
    qsofa += int(patient.mental_status != MentalStatus.ALERT)
    patient.qsofa_score = qsofa

    organ = 0.0
    organ += 1.0 if patient.mean_arterial_pressure < 65 else 0.0
    organ += 1.0 if patient.oxygen_saturation < 92 else 0.0
    organ += 1.0 if patient.mental_status != MentalStatus.ALERT else 0.0
    if patient.lactate is not None:
        organ += 1.0 if patient.lactate > 4.0 else 0.5 if patient.lactate > 2.2 else 0.0
    if patient.creatinine is not None:
        organ += 1.0 if patient.creatinine > 2.0 else 0.5 if patient.creatinine > 1.3 else 0.0
    if patient.bilirubin_total is not None:
        organ += 1.0 if patient.bilirubin_total > 2.0 else 0.5 if patient.bilirubin_total > 1.2 else 0.0
    if patient.platelets is not None:
        organ += 1.0 if patient.platelets < 100 else 0.5 if patient.platelets < 150 else 0.0
    patient.organ_dysfunction_score = round(organ, 3)

    infection_active = (
        patient.infection_present
        and patient.infection_start_tick is not None
        and tick >= patient.infection_start_tick
    )
    if infection_active and patient.organ_dysfunction_score >= 2.0 and patient.sepsis_onset_tick is None:
        patient.sepsis_onset_tick = tick

    _mark_first_abnormal_vitals(patient, tick)


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
        age = int(rng.integers(25, 91))
        immunocompromised = bool(rng.random() < 0.25)
        meds = rng.choice(
            COMMON_MEDICATIONS,
            size=int(rng.integers(0, 3)),
            replace=False,
        ).tolist()
        if immunocompromised:
            meds.append(str(rng.choice(IMMUNOSUPPRESSANTS)))

        infection_start_tick = (
            int(rng.integers(2, max(4, int(max_steps * 0.55))))
            if is_sepsis else None
        )
        dbp = float(rng.uniform(60, 82))
        sbp = float(rng.uniform(105, 138))
        p = PatientState(
            patient_id=f"P{i+1:02d}",
            bed_number=i + 1,
            age=age,
            gender=int(rng.integers(0, 2)),
            unit1=int(rng.integers(0, 2)),
            unit2=int(rng.integers(0, 2)),
            hosp_adm_time=-float(rng.integers(2, 72)),
            iculos_hours=0.0,
            admission_reason=str(rng.choice(ADMISSION_REASONS)),
            heart_rate=float(rng.uniform(68, 96)),
            systolic_bp=sbp,
            mean_arterial_pressure=_map_from_pressures(sbp, dbp),
            diastolic_bp=dbp,
            respiratory_rate=float(rng.uniform(12, 20)),
            temperature=float(rng.uniform(36.3, 37.3)),
            oxygen_saturation=float(rng.uniform(95, 99)),
            current_medications=meds,
            immunocompromised=immunocompromised,
            infection_present=is_sepsis,
            infection_start_tick=infection_start_tick,
            is_false_alarm_patient=is_false_alarm,
            fio2=21.0,
            sao2=float(rng.uniform(95, 100)),
            glucose=float(rng.uniform(82, 125)),
            hemoglobin=float(rng.uniform(11.5, 15.2)),
            bicarbonate=float(rng.uniform(22.0, 27.5)),
            ph=float(rng.uniform(7.36, 7.43)),
            paco2=float(rng.uniform(36.0, 44.0)),
            base_excess=float(rng.uniform(-1.5, 1.5)),
        )
        _update_clinical_scores(p, tick=1)
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

    patient.iculos_hours = round(tick / 2.0, 2)

    patient.heart_rate += float(rng.normal(0, 1.0))
    patient.diastolic_bp += float(rng.normal(0, 1.1))
    patient.systolic_bp += float(rng.normal(0, 1.6))
    patient.respiratory_rate += float(rng.normal(0, 0.45))
    patient.temperature += float(rng.normal(0, 0.05))
    patient.oxygen_saturation += float(rng.normal(0, 0.25))
    if patient.glucose is not None:
        patient.glucose += float(rng.normal(0, 2.0))

    if patient.is_false_alarm_patient and 4 <= tick <= 20:
        patient.heart_rate += float(rng.uniform(0.8, 2.0))
        patient.respiratory_rate += float(rng.uniform(0.1, 0.8))
        patient.temperature += float(rng.uniform(0.02, 0.10))
        patient.oxygen_saturation -= float(rng.uniform(0.0, 0.5))

    infection_active = (
        patient.infection_present
        and patient.infection_start_tick is not None
        and tick >= patient.infection_start_tick
    )
    if infection_active and patient.antibiotics_administered is None:
        patient.infection_severity = min(
            1.0,
            patient.infection_severity + float(rng.uniform(0.04, 0.10)),
        )
        sev = patient.infection_severity
        temp_bump = float(rng.uniform(0.0, 0.08 if patient.immunocompromised else 0.18))
        patient.heart_rate += float(rng.uniform(2.0, 5.0) + 5.5 * sev)
        patient.respiratory_rate += float(rng.uniform(0.5, 1.5) + 2.5 * sev)
        patient.systolic_bp -= float(rng.uniform(1.0, 3.0) + 5.0 * sev)
        patient.diastolic_bp -= float(rng.uniform(0.5, 1.5) + 2.5 * sev)
        patient.temperature += temp_bump + float(sev * 0.15)
        patient.oxygen_saturation -= float(rng.uniform(0.2, 0.8) + 1.2 * sev)
        if patient.glucose is not None:
            patient.glucose += float(rng.uniform(1.0, 4.0) + 8.0 * sev)
        if sev > 0.70 or patient.organ_dysfunction_score >= 2.0:
            patient.outcome = Outcome.DETERIORATING
        if sev > 0.82 or patient.organ_dysfunction_score >= 3.5:
            patient.outcome = Outcome.SEPTIC_SHOCK
            patient.mental_status = MentalStatus.CONFUSED
        if sev > 0.94 or patient.organ_dysfunction_score >= 5.0:
            patient.critical_ticks += 1
            if patient.critical_ticks >= 4:
                patient.outcome = Outcome.DIED
                patient.mental_status = MentalStatus.UNRESPONSIVE

    if patient.infection_present and patient.antibiotics_administered is not None:
        patient.infection_severity = max(
            0.0,
            patient.infection_severity - float(rng.uniform(0.07, 0.16)),
        )
        patient.heart_rate -= float(rng.uniform(0.6, 2.0))
        patient.respiratory_rate -= float(rng.uniform(0.3, 1.2))
        patient.systolic_bp += float(rng.uniform(0.5, 2.0))
        patient.diastolic_bp += float(rng.uniform(0.2, 1.2))
        patient.temperature -= float(rng.uniform(0.01, 0.08))
        patient.oxygen_saturation += float(rng.uniform(0.2, 0.8))
        if patient.glucose is not None:
            patient.glucose -= float(rng.uniform(0.5, 2.5))
        if patient.infection_severity < 0.45 and patient.mental_status != MentalStatus.ALERT:
            patient.mental_status = MentalStatus.ALERT
        if patient.infection_severity < 0.15 and patient.organ_dysfunction_score < 1.5:
            patient.outcome = Outcome.RECOVERED

    patient.systolic_bp = _clamp(patient.systolic_bp, 50.0, 220.0)
    patient.diastolic_bp = _clamp(patient.diastolic_bp, 30.0, 130.0)
    patient.mean_arterial_pressure = _clamp(
        _map_from_pressures(patient.systolic_bp, patient.diastolic_bp),
        35.0,
        160.0,
    )
    patient.heart_rate = _clamp(patient.heart_rate, 30.0, 220.0)
    patient.respiratory_rate = _clamp(patient.respiratory_rate, 6.0, 50.0)
    patient.temperature = _clamp(patient.temperature, 34.0, 42.0)
    patient.oxygen_saturation = _clamp(patient.oxygen_saturation, 60.0, 100.0)
    if patient.sao2 is not None:
        patient.sao2 = _clamp(patient.oxygen_saturation + float(rng.normal(0, 0.6)), 60.0, 100.0)
    if patient.ph is not None:
        patient.ph = _clamp(
            patient.ph - float(max(0.0, patient.infection_severity - 0.4) * 0.01) + float(rng.normal(0, 0.004)),
            7.05,
            7.55,
        )
    if patient.paco2 is not None:
        patient.paco2 = _clamp(patient.paco2 + float(rng.normal(0, 0.6)), 20.0, 70.0)
    if patient.bicarbonate is not None:
        patient.bicarbonate = _clamp(
            patient.bicarbonate - float(max(0.0, patient.infection_severity - 0.4) * 0.8) + float(rng.normal(0, 0.3)),
            8.0,
            36.0,
        )
    if patient.base_excess is not None:
        patient.base_excess = _clamp(
            patient.base_excess - float(max(0.0, patient.infection_severity - 0.35) * 0.6) + float(rng.normal(0, 0.25)),
            -15.0,
            10.0,
        )

    if patient.mental_status == MentalStatus.UNRESPONSIVE and patient.outcome != Outcome.DIED:
        patient.mental_status = MentalStatus.CONFUSED
    if patient.outcome == Outcome.STABLE and patient.qsofa_score >= 2:
        patient.outcome = Outcome.DETERIORATING

    _update_clinical_scores(patient, tick)


def mature_pending_labs(patient: PatientState, tick: int, rng: np.random.Generator) -> List[str]:
    """Fill in lab values whose delay has elapsed. Return list of newly-available test names."""
    ready = [test for test, due_tick in patient.pending_labs.items() if due_tick <= tick]
    if not ready:
        return []

    sev = patient.infection_severity if patient.infection_present else 0.0
    dysfunction = patient.organ_dysfunction_score

    for test in ready:
        if test == "lactate":
            base = 1.1 + 3.3 * sev + 0.2 * dysfunction
            patient.lactate = round(base + float(rng.normal(0, 0.25)), 2)
        elif test == "wbc":
            base = 7.0 + 9.5 * sev
            if patient.immunocompromised:
                base = 6.2 + 3.2 * sev
            patient.wbc = round(base + float(rng.normal(0, 1.2)), 2)
        elif test == "procalcitonin":
            base = 0.15 + 3.8 * sev
            patient.procalcitonin = round(base + float(rng.normal(0, 0.18)), 3)
        elif test == "creatinine":
            base = 0.85 + 1.4 * sev + 0.18 * dysfunction
            patient.creatinine = round(base + float(rng.normal(0, 0.12)), 2)
        elif test == "bun":
            base = 12.0 + 16.0 * sev + 2.0 * dysfunction
            patient.bun = round(base + float(rng.normal(0, 1.8)), 2)
        elif test == "bilirubin_total":
            base = 0.5 + 1.6 * sev + 0.3 * dysfunction
            patient.bilirubin_total = round(base + float(rng.normal(0, 0.1)), 2)
        elif test == "platelets":
            base = 260.0 - 95.0 * sev - 14.0 * dysfunction
            patient.platelets = round(base + float(rng.normal(0, 8.0)), 1)
        elif test == "glucose":
            base = 100.0 + 30.0 * sev
            patient.glucose = round(base + float(rng.normal(0, 6.0)), 1)
        elif test == "hemoglobin":
            base = 13.2 - 0.8 * sev
            patient.hemoglobin = round(base + float(rng.normal(0, 0.25)), 2)
        elif test == "ptt":
            base = 29.0 + 12.0 * sev + 1.5 * dysfunction
            patient.ptt = round(base + float(rng.normal(0, 1.0)), 2)
        elif test == "fibrinogen":
            base = 300.0 + 140.0 * sev
            patient.fibrinogen = round(base + float(rng.normal(0, 12.0)), 1)
        elif test == "bicarbonate":
            base = 25.0 - 5.5 * sev - 0.7 * dysfunction
            patient.bicarbonate = round(base + float(rng.normal(0, 0.6)), 2)
        elif test == "ph":
            base = 7.40 - 0.08 * sev - 0.015 * dysfunction
            patient.ph = round(base + float(rng.normal(0, 0.008)), 3)
        elif test == "paco2":
            base = 39.0 + 4.0 * max(0.0, sev - 0.2)
            patient.paco2 = round(base + float(rng.normal(0, 0.9)), 2)
        elif test == "sao2":
            base = patient.oxygen_saturation - 0.5 * sev
            patient.sao2 = round(base + float(rng.normal(0, 0.8)), 2)
        elif test == "base_excess":
            base = -0.4 - 5.0 * sev - 0.6 * dysfunction
            patient.base_excess = round(base + float(rng.normal(0, 0.5)), 2)
        elif test == "blood_culture":
            patient.blood_culture_result = (
                "gram_positive" if (patient.infection_present and rng.random() < 0.5)
                else "gram_negative" if patient.infection_present
                else "no_growth"
            )
        patient.last_measured_tick[test] = tick
        patient.pending_labs.pop(test, None)

    _update_clinical_scores(patient, tick)
    return ready


def order_lab_test(patient: PatientState, test: str, tick: int, delay: int) -> bool:
    """Queue a lab test. Returns True if newly ordered, False if already pending or just measured."""
    if test in patient.pending_labs:
        return False
    if patient.last_measured_tick.get(test) == tick:
        return False
    patient.pending_labs[test] = tick + delay
    return True
