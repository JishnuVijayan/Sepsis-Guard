from __future__ import annotations
from typing import Dict, Any, Set
from agents.base import RoleAgent


class HeuristicPharmacist(RoleAgent):
    def __init__(self):
        self._already_flagged: Set[str] = set()

    def decide(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        immuno = [m for m in obs.get("patient_medications", []) if m.get("immunocompromised")]
        for m in immuno:
            if m["patient_id"] not in self._already_flagged:
                self._already_flagged.add(m["patient_id"])
                return {
                    "operation": "flag_immunosuppression",
                    "patient_id": m["patient_id"],
                    "rationale": f"Patient on {', '.join(m['current_medications'])}",
                }
        lab_flags = obs.get("lab_flags_this_tick", [])
        for f in lab_flags:
            if f["flag_type"] == "critical_lab":
                return {
                    "operation": "recommend_antibiotic",
                    "patient_id": f["patient_id"],
                    "drug": "piperacillin_tazobactam",
                    "rationale": "broad-spectrum empirical, antibiogram-favored",
                }
        return {"operation": "noop"}
