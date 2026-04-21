from __future__ import annotations
from typing import Dict, Any
from agents.base import RoleAgent


class HeuristicPhysician(RoleAgent):
    def decide(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        escalations = obs.get("nurse_escalations_this_tick", [])
        lab_flags = obs.get("lab_flags_this_tick", [])
        pharm_flags = obs.get("pharmacist_flags_this_tick", [])
        escalated_pids = {f["patient_id"] for f in escalations if f.get("urgency") in ("urgent", "critical")}
        lab_pids = {f["patient_id"] for f in lab_flags}
        multi_source = escalated_pids & lab_pids
        if multi_source:
            pid = next(iter(multi_source))
            abx_rec = [f for f in pharm_flags if f["patient_id"] == pid and f["flag_type"] == "antibiotic_recommendation"]
            drug = "piperacillin_tazobactam"
            if abx_rec:
                rationale = abx_rec[0].get("rationale", "")
                for candidate in ("piperacillin_tazobactam", "meropenem", "ceftriaxone", "vancomycin"):
                    if candidate in rationale:
                        drug = candidate
                        break
            return {
                "operation": "order_antibiotics", "patient_id": pid, "drug": drug,
            }
        for f in escalations:
            if f.get("urgency") == "critical":
                return {
                    "operation": "order_antibiotics", "patient_id": f["patient_id"],
                    "drug": "piperacillin_tazobactam",
                }
        return {"operation": "do_nothing"}
