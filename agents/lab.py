from __future__ import annotations
from typing import Dict, Any
from agents.base import RoleAgent


class HeuristicLab(RoleAgent):
    def decide(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        for r in obs.get("lab_results", []):
            pid = r["patient_id"]
            if r.get("lactate") is not None and r["lactate"] > 2.5:
                return {
                    "operation": "flag_critical", "patient_id": pid,
                    "reason": f"lactate={r['lactate']:.2f} (>2.5)",
                }
            if r.get("procalcitonin") is not None and r["procalcitonin"] > 2.0:
                return {
                    "operation": "flag_critical", "patient_id": pid,
                    "reason": f"procalcitonin={r['procalcitonin']:.2f} (>2.0)",
                }
            if r.get("wbc") is not None and (r["wbc"] > 15 or r["wbc"] < 3):
                return {
                    "operation": "flag_critical", "patient_id": pid,
                    "reason": f"WBC={r['wbc']:.2f}",
                }
        return {"operation": "noop"}
