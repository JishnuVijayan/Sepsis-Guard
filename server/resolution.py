from __future__ import annotations
from typing import List, Tuple, Dict, Any
import numpy as np
from models import (
    PatientState, AgentFlag, StepRequest,
    NurseAction, LabAction, PharmacistAction, PhysicianAction,
)
from server.physiology import order_lab_test
from server.config import VALID_ANTIBIOTICS, ANTIBIOGRAM


def _find_patient(patients: List[PatientState], patient_id: str) -> PatientState | None:
    for p in patients:
        if p.patient_id == patient_id:
            return p
    return None


def apply_lab_action(
    action: LabAction, patients: List[PatientState], tick: int,
) -> Tuple[List[AgentFlag], str]:
    flags: List[AgentFlag] = []
    if action.operation == "release_result":
        return flags, "Lab released results (deterministic auto-release in physiology)"
    if action.operation == "flag_critical":
        p = _find_patient(patients, action.patient_id)
        if p is None:
            return flags, f"Lab: unknown patient {action.patient_id}"
        flags.append(AgentFlag(
            source_role="lab", patient_id=p.patient_id, flag_type="critical_lab",
            urgency="urgent", rationale=action.reason or "critical lab value", tick=tick,
        ))
        return flags, f"Lab flagged critical for {p.patient_id}"
    if action.operation == "recommend_followup_test":
        p = _find_patient(patients, action.patient_id)
        if p is None:
            return flags, "Lab: unknown patient"
        flags.append(AgentFlag(
            source_role="lab", patient_id=p.patient_id, flag_type="followup_recommended",
            urgency="routine", rationale=action.reason, tick=tick,
        ))
        return flags, f"Lab recommended followup {action.test}"
    return flags, "Lab noop"


def apply_pharmacist_action(
    action: PharmacistAction, patients: List[PatientState], tick: int,
) -> Tuple[List[AgentFlag], str]:
    flags: List[AgentFlag] = []
    if action.operation == "flag_interaction":
        p = _find_patient(patients, action.patient_id)
        if p is None:
            return flags, "Pharmacist: unknown patient"
        flags.append(AgentFlag(
            source_role="pharmacist", patient_id=p.patient_id,
            flag_type="drug_interaction", urgency="urgent",
            rationale=action.rationale, tick=tick,
        ))
        return flags, f"Pharmacist flagged interaction for {p.patient_id}"
    if action.operation == "flag_immunosuppression":
        p = _find_patient(patients, action.patient_id)
        if p is None:
            return flags, "Pharmacist: unknown patient"
        flags.append(AgentFlag(
            source_role="pharmacist", patient_id=p.patient_id,
            flag_type="immunosuppression", urgency="urgent",
            rationale=action.rationale or "patient on immunosuppressants", tick=tick,
        ))
        return flags, f"Pharmacist flagged immunosuppression for {p.patient_id}"
    if action.operation == "recommend_antibiotic":
        p = _find_patient(patients, action.patient_id)
        if p is None:
            return flags, "Pharmacist: unknown patient"
        flags.append(AgentFlag(
            source_role="pharmacist", patient_id=p.patient_id,
            flag_type="antibiotic_recommendation", urgency="routine",
            rationale=f"recommend {action.drug}: {action.rationale}", tick=tick,
        ))
        return flags, f"Pharmacist recommended {action.drug}"
    return flags, "Pharmacist noop"


def apply_nurse_action(
    action: NurseAction, patients: List[PatientState], tick: int, lab_delay: int,
) -> Tuple[List[AgentFlag], str]:
    flags: List[AgentFlag] = []
    if action.operation == "escalate_to_physician":
        p = _find_patient(patients, action.patient_id)
        if p is None:
            return flags, "Nurse: unknown patient"
        urgency = action.urgency or "routine"
        flags.append(AgentFlag(
            source_role="nurse", patient_id=p.patient_id, flag_type="escalation",
            urgency=urgency, rationale=action.rationale, tick=tick,
        ))
        return flags, f"Nurse escalated {p.patient_id} ({urgency})"
    if action.operation == "request_lab_test":
        p = _find_patient(patients, action.patient_id)
        if p is None or action.test_type is None:
            return flags, "Nurse: bad lab request"
        ordered = order_lab_test(p, action.test_type, tick, lab_delay)
        return flags, f"Nurse ordered {action.test_type} for {p.patient_id} (ordered={ordered})"
    if action.operation == "flag_concern":
        p = _find_patient(patients, action.patient_id)
        if p is None:
            return flags, "Nurse: unknown patient"
        flags.append(AgentFlag(
            source_role="nurse", patient_id=p.patient_id, flag_type="concern",
            urgency="routine", rationale=action.rationale, tick=tick,
        ))
        return flags, f"Nurse flagged concern for {p.patient_id}"
    return flags, "Nurse noop"


def apply_physician_action(
    action: PhysicianAction, patients: List[PatientState], tick: int,
    active_flags: List[AgentFlag], physician_trust: float, lab_delay: int,
) -> Tuple[str, Dict[str, Any]]:
    meta: Dict[str, Any] = {
        "antibiotics_ordered": False,
        "icu_ordered": False,
        "on_valid_escalation": False,
        "on_false_alarm": False,
    }
    if action.operation == "order_antibiotics":
        p = _find_patient(patients, action.patient_id)
        if p is None or action.drug not in VALID_ANTIBIOTICS:
            return "Physician: invalid antibiotic order", meta
        if physician_trust < 0.4:
            return "Physician order delayed (low trust)", meta
        if p.antibiotics_administered is None:
            p.antibiotics_administered = action.drug
            p.antibiotic_tick = tick
            meta["antibiotics_ordered"] = True
            for f in active_flags:
                if f.patient_id == p.patient_id and f.flag_type in ("escalation", "critical_lab"):
                    meta["on_valid_escalation"] = True
            if p.is_false_alarm_patient:
                meta["on_false_alarm"] = True
        return f"Physician ordered {action.drug} for {p.patient_id}", meta
    if action.operation == "admit_to_icu":
        p = _find_patient(patients, action.patient_id)
        if p is None:
            return "Physician: unknown patient", meta
        p.icu_admitted = True
        meta["icu_ordered"] = True
        return f"Physician admitted {p.patient_id} to ICU", meta
    if action.operation == "order_lab_test":
        p = _find_patient(patients, action.patient_id)
        if p is None or action.test is None:
            return "Physician: bad lab request", meta
        order_lab_test(p, action.test, tick, lab_delay)
        return f"Physician ordered {action.test} for {p.patient_id}", meta
    return "Physician did nothing", meta


def resolve_step(
    request: StepRequest, patients: List[PatientState], tick: int,
    lab_delay: int, physician_trust: float,
) -> Tuple[List[AgentFlag], Dict[str, str], Dict[str, Any]]:
    all_flags: List[AgentFlag] = []
    results: Dict[str, str] = {}

    lab_flags, lab_res = apply_lab_action(request.lab, patients, tick)
    all_flags.extend(lab_flags); results["lab"] = lab_res

    pharm_flags, pharm_res = apply_pharmacist_action(request.pharmacist, patients, tick)
    all_flags.extend(pharm_flags); results["pharmacist"] = pharm_res

    nurse_flags, nurse_res = apply_nurse_action(request.nurse, patients, tick, lab_delay)
    all_flags.extend(nurse_flags); results["nurse"] = nurse_res

    phys_res, phys_meta = apply_physician_action(
        request.physician, patients, tick, all_flags, physician_trust, lab_delay,
    )
    results["physician"] = phys_res
    return all_flags, results, phys_meta
