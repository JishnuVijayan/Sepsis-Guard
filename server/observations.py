from __future__ import annotations
from typing import List, Dict, Any, Optional
from models import (
    PatientState, AgentFlag,
    NurseObservation, LabObservation,
    PharmacistObservation, PhysicianObservation,
)
from server.config import ANTIBIOGRAM, CORE_VITAL_FIELDS, LAB_FIELDS


def _measurement_age(patient: PatientState, field: str, tick: int) -> Optional[float]:
    last_tick = patient.last_measured_tick.get(field)
    if last_tick is None:
        return None
    return round((tick - last_tick) / 2.0, 2)


def _nurse_patient_view(p: PatientState) -> Dict[str, Any]:
    return {
        "patient_id": p.patient_id,
        "bed_number": p.bed_number,
        "age": p.age,
        "admission_reason": p.admission_reason,
        "heart_rate": round(p.heart_rate, 1),
        "systolic_bp": round(p.systolic_bp, 1),
        "mean_arterial_pressure": round(p.mean_arterial_pressure, 1),
        "diastolic_bp": round(p.diastolic_bp, 1),
        "respiratory_rate": round(p.respiratory_rate, 1),
        "temperature": round(p.temperature, 2),
        "oxygen_saturation": round(p.oxygen_saturation, 1),
        "mental_status": p.mental_status.value,
        "qsofa_score": p.qsofa_score,
        "iculos_hours": p.iculos_hours,
        # Fix 3: boolean so nurse knows the physician already acted — she can
        # stop escalating and avoid unnecessary repeat-flag penalties.
        "antibiotics_administered": p.antibiotics_administered is not None,
    }


def _lab_patient_view(p: PatientState) -> Dict[str, Any]:
    ages = {
        field: p.last_measured_tick.get(field)
        for field in LAB_FIELDS
        if p.last_measured_tick.get(field) is not None
    }
    return {
        "patient_id": p.patient_id,
        "lactate": p.lactate,
        "wbc": p.wbc,
        "procalcitonin": p.procalcitonin,
        "creatinine": p.creatinine,
        "bun": p.bun,
        "bilirubin_total": p.bilirubin_total,
        "platelets": p.platelets,
        "glucose": p.glucose,
        "hemoglobin": p.hemoglobin,
        "ptt": p.ptt,
        "fibrinogen": p.fibrinogen,
        "bicarbonate": p.bicarbonate,
        "ph": p.ph,
        "paco2": p.paco2,
        "sao2": p.sao2,
        "base_excess": p.base_excess,
        "blood_culture_result": p.blood_culture_result,
        "last_measured_tick": ages,
    }


def _pharmacist_patient_view(p: PatientState, tick: int) -> Dict[str, Any]:
    return {
        "patient_id": p.patient_id,
        "age": p.age,
        "current_medications": list(p.current_medications),
        "immunocompromised": p.immunocompromised,
        "antibiotics_administered": p.antibiotics_administered,
        "iculos_hours": p.iculos_hours,
        "renal_risk": (p.creatinine or 0.0) > 1.8,
        "creatinine_age_hours": _measurement_age(p, "creatinine", tick),
    }


def _physician_known_patient(
    p: PatientState, flags_for_patient: List[AgentFlag], tick: int,
    memory_entry: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    view: Dict[str, Any] = {
        "patient_id": p.patient_id,
        "age": p.age,
        "admission_reason": p.admission_reason,
        "mental_status": p.mental_status.value,
        "antibiotics_administered": p.antibiotics_administered,
        "icu_admitted": p.icu_admitted,
        "iculos_hours": p.iculos_hours,
        "qsofa_score": p.qsofa_score,
        "organ_dysfunction_score": round(p.organ_dysfunction_score, 2),
        "first_seen_tick": memory_entry.get("first_seen_tick") if memory_entry else tick,
        "last_seen_tick": tick,
        "flags_raised": [
            {"source": f.source_role, "type": f.flag_type,
             "urgency": f.urgency, "rationale": f.rationale}
            for f in flags_for_patient
        ],
    }
    for f in flags_for_patient:
        if f.source_role == "lab" and f.flag_type == "critical_lab":
            view["lab_snapshot"] = {
                "lactate": p.lactate,
                "wbc": p.wbc,
                "procalcitonin": p.procalcitonin,
                "creatinine": p.creatinine,
                "bun": p.bun,
                "platelets": p.platelets,
                "bilirubin_total": p.bilirubin_total,
                "age_hours": {
                    field: _measurement_age(p, field, tick)
                    for field in ("lactate", "wbc", "procalcitonin", "creatinine", "platelets", "bilirubin_total")
                },
            }
    nurse_escalated = any(
        f.source_role == "nurse" and f.flag_type == "escalation"
        for f in flags_for_patient
    )
    if nurse_escalated:
        view["vitals_at_escalation"] = {
            "heart_rate": round(p.heart_rate, 1),
            "systolic_bp": round(p.systolic_bp, 1),
            "mean_arterial_pressure": round(p.mean_arterial_pressure, 1),
            "respiratory_rate": round(p.respiratory_rate, 1),
            "temperature": round(p.temperature, 2),
            "oxygen_saturation": round(p.oxygen_saturation, 1),
        }
    if memory_entry and memory_entry.get("flag_history"):
        view["flag_history"] = list(memory_entry["flag_history"])
    return view


def build_observations(
    patients: List[PatientState],
    nurse_assignment: Dict[str, List[str]],
    active_flags_this_tick: List[AgentFlag],
    tick: int,
    max_ticks: int,
    task_name: str,
    physician_trust: float,
    cumulative_rewards: Dict[str, float],
    last_results: Dict[str, Optional[str]],
    pending_labs_summary: List[Dict[str, Any]],
    full_info_sharing: bool = False,
    physician_memory: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    by_patient: Dict[str, List[AgentFlag]] = {}
    for f in active_flags_this_tick:
        by_patient.setdefault(f.patient_id, []).append(f)

    nurse_ids = set(nurse_assignment.get("nurse", []))
    nurse_patient_rows = [
        _nurse_patient_view(p)
        for p in patients if full_info_sharing or p.patient_id in nurse_ids
    ]
    nurse_obs = NurseObservation(
        done=False, reward=0.0,
        tick=tick, max_ticks=max_ticks, task_name=task_name,
        assigned_patient_ids=sorted(nurse_ids),
        patient_vitals=nurse_patient_rows,
        pharmacist_flags_this_tick=[
            f for f in active_flags_this_tick if f.source_role == "pharmacist"
        ],
        lab_flags_this_tick=[
            f for f in active_flags_this_tick if f.source_role == "lab"
        ],
        physician_trust=physician_trust,
        last_action_result=last_results.get("nurse"),
        cumulative_reward=cumulative_rewards.get("nurse", 0.0),
    )

    lab_obs = LabObservation(
        done=False, reward=0.0,
        tick=tick, max_ticks=max_ticks, task_name=task_name,
        lab_results=[_lab_patient_view(p) for p in patients],
        pending_labs=pending_labs_summary,
        last_action_result=last_results.get("lab"),
        cumulative_reward=cumulative_rewards.get("lab", 0.0),
    )

    pharmacist_obs = PharmacistObservation(
        done=False, reward=0.0,
        tick=tick, max_ticks=max_ticks, task_name=task_name,
        patient_medications=[_pharmacist_patient_view(p, tick) for p in patients],
        antibiogram=dict(ANTIBIOGRAM),
        lab_flags_this_tick=[
            f for f in active_flags_this_tick if f.source_role == "lab"
        ],
        last_action_result=last_results.get("pharmacist"),
        cumulative_reward=cumulative_rewards.get("pharmacist", 0.0),
    )

    physician_memory = physician_memory or {}
    escalated_patient_ids = {f.patient_id for f in active_flags_this_tick}
    known_patient_ids = escalated_patient_ids | set(physician_memory.keys())
    known_summaries = [
        _physician_known_patient(
            p,
            by_patient.get(p.patient_id, []),
            tick,
            physician_memory.get(p.patient_id),
        )
        for p in patients if full_info_sharing or p.patient_id in known_patient_ids
    ]
    if full_info_sharing:
        for p, summary in zip(patients, known_summaries):
            summary.setdefault("vitals_snapshot", {
                "heart_rate": round(p.heart_rate, 1),
                "systolic_bp": round(p.systolic_bp, 1),
                "mean_arterial_pressure": round(p.mean_arterial_pressure, 1),
                "respiratory_rate": round(p.respiratory_rate, 1),
                "temperature": round(p.temperature, 2),
                "oxygen_saturation": round(p.oxygen_saturation, 1),
            })
            summary.setdefault("lab_snapshot", {
                "lactate": p.lactate,
                "wbc": p.wbc,
                "procalcitonin": p.procalcitonin,
                "creatinine": p.creatinine,
                "blood_culture_result": p.blood_culture_result,
            })
    physician_obs = PhysicianObservation(
        done=False, reward=0.0,
        tick=tick, max_ticks=max_ticks, task_name=task_name,
        nurse_escalations_this_tick=[
            f for f in active_flags_this_tick if f.source_role == "nurse"
        ],
        lab_flags_this_tick=[
            f for f in active_flags_this_tick if f.source_role == "lab"
        ],
        pharmacist_flags_this_tick=[
            f for f in active_flags_this_tick if f.source_role == "pharmacist"
        ],
        known_patient_summaries=known_summaries,
        physician_trust=physician_trust,
        last_action_result=last_results.get("physician"),
        cumulative_reward=cumulative_rewards.get("physician", 0.0),
    )

    return {
        "nurse": nurse_obs,
        "lab": lab_obs,
        "pharmacist": pharmacist_obs,
        "physician": physician_obs,
    }
