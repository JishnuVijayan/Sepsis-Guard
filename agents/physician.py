from __future__ import annotations
from typing import Dict, Any
from agents.base import RoleAgent


class HeuristicPhysician(RoleAgent):
    def decide(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        escalations = obs.get("nurse_escalations_this_tick", [])
        lab_flags = obs.get("lab_flags_this_tick", [])
        pharm_flags = obs.get("pharmacist_flags_this_tick", [])
        known = obs.get("known_patient_summaries", [])
        already_treated = {
            s["patient_id"] for s in known
            if s.get("antibiotics_administered") is not None
        }

        escalated_pids = {
            f["patient_id"] for f in escalations
            if f.get("urgency") in ("urgent", "critical")
        } - already_treated
        lab_pids = {f["patient_id"] for f in lab_flags} - already_treated

        multi_source = escalated_pids & lab_pids
        if multi_source:
            pid = next(iter(multi_source))
            drug = self._pick_drug(pid, pharm_flags)
            return {"operation": "order_antibiotics", "patient_id": pid, "drug": drug}

        for f in escalations:
            if f.get("urgency") == "critical" and f["patient_id"] not in already_treated:
                return {
                    "operation": "order_antibiotics", "patient_id": f["patient_id"],
                    "drug": "piperacillin_tazobactam",
                }

        for pid in lab_pids:
            return {
                "operation": "order_antibiotics", "patient_id": pid,
                "drug": "piperacillin_tazobactam",
            }

        flagged_pids = escalated_pids | lab_pids
        for summary in known:
            pid = summary.get("patient_id")
            if not pid or pid in already_treated:
                continue
            has_flags = pid in flagged_pids or len(summary.get("flags_raised", [])) > 0
            vitals = summary.get("vitals_snapshot") or summary.get("vitals_at_escalation") or {}
            labs = summary.get("lab_snapshot") or {}
            severe_vitals = (
                vitals.get("heart_rate", 0) > 115
                or vitals.get("systolic_bp", 999) < 95
                or vitals.get("respiratory_rate", 0) > 24
                or vitals.get("temperature", 0.0) >= 38.4
                or vitals.get("oxygen_saturation", 100) < 93
            )
            concerning_labs = (
                (labs.get("lactate") is not None and labs.get("lactate") > 2.2)
                or (labs.get("procalcitonin") is not None and labs.get("procalcitonin") > 1.5)
                or (labs.get("wbc") is not None and (labs.get("wbc") > 14 or labs.get("wbc") < 3.5))
            )
            if (severe_vitals and concerning_labs) or (has_flags and (severe_vitals or concerning_labs)):
                return {
                    "operation": "order_antibiotics", "patient_id": pid,
                    "drug": "piperacillin_tazobactam",
                }
        return {"operation": "do_nothing"}

    @staticmethod
    def _pick_drug(pid: str, pharm_flags: list) -> str:
        abx_rec = [f for f in pharm_flags
                    if f["patient_id"] == pid and f["flag_type"] == "antibiotic_recommendation"]
        if abx_rec:
            rationale = abx_rec[0].get("rationale", "")
            for candidate in ("piperacillin_tazobactam", "meropenem", "ceftriaxone", "vancomycin"):
                if candidate in rationale:
                    return candidate
        return "piperacillin_tazobactam"
