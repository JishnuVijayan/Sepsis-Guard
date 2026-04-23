from __future__ import annotations
from typing import Dict, Any, Set
from agents.base import RoleAgent


class HeuristicLab(RoleAgent):
    def __init__(self):
        self._recently_flagged: Set[str] = set()

    def decide(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        critical = []
        for r in obs.get("lab_results", []):
            pid = r["patient_id"]
            abnormal = (
                (r.get("lactate") is not None and r["lactate"] > 2.0)
                or (r.get("procalcitonin") is not None and r["procalcitonin"] > 0.5)
                or (r.get("wbc") is not None and (r["wbc"] > 12 or r["wbc"] < 4))
            )
            if abnormal:
                reason_parts = []
                if r.get("lactate") is not None and r["lactate"] > 2.0:
                    reason_parts.append(f"lactate={r['lactate']:.2f}")
                if r.get("procalcitonin") is not None and r["procalcitonin"] > 0.5:
                    reason_parts.append(f"procalcitonin={r['procalcitonin']:.2f}")
                if r.get("wbc") is not None and (r["wbc"] > 12 or r["wbc"] < 4):
                    reason_parts.append(f"WBC={r['wbc']:.2f}")
                critical.append((pid, ", ".join(reason_parts)))

        if critical:
            not_recent = [(pid, reason) for pid, reason in critical
                          if pid not in self._recently_flagged]
            pick = not_recent[0] if not_recent else critical[0]
            self._recently_flagged.add(pick[0])
            return {
                "operation": "flag_critical", "patient_id": pick[0],
                "reason": pick[1],
            }
        return {"operation": "noop"}
