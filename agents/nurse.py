from __future__ import annotations
from typing import Dict, Any, Set
from agents.base import RoleAgent


class HeuristicNurse(RoleAgent):
    def __init__(self):
        self._recently_escalated: Set[str] = set()

    def decide(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        patients = obs.get("patient_vitals", [])
        pharm_flags = obs.get("pharmacist_flags_this_tick", [])
        lab_flags = obs.get("lab_flags_this_tick", [])
        physician_trust = float(obs.get("physician_trust", 1.0))
        task_name = str(obs.get("task_name", ""))

        for f in lab_flags:
            if f["flag_type"] == "critical_lab":
                pid = f["patient_id"]
                if pid not in self._recently_escalated and any(p["patient_id"] == pid for p in patients):
                    self._recently_escalated.add(pid)
                    return {
                        "operation": "escalate_to_physician", "patient_id": pid,
                        "urgency": "critical",
                        "rationale": f"Lab flagged critical: {f.get('rationale')}",
                    }

        immuno_patients = {f["patient_id"] for f in pharm_flags
                           if f["flag_type"] == "immunosuppression"}
        for p in patients:
            if p["patient_id"] in immuno_patients and p["patient_id"] not in self._recently_escalated:
                if p["heart_rate"] > 100 or p["respiratory_rate"] > 20 or p["systolic_bp"] < 100:
                    self._recently_escalated.add(p["patient_id"])
                    return {
                        "operation": "escalate_to_physician", "patient_id": p["patient_id"],
                        "urgency": "urgent",
                        "rationale": "Immunocompromised + abnormal vitals",
                    }

        if physician_trust >= 0.6 and task_name == "task1_textbook":
            for p in patients:
                if p["patient_id"] in self._recently_escalated:
                    continue
                signs = sum([
                    p["heart_rate"] > 105,
                    p["systolic_bp"] < 100,
                    p["respiratory_rate"] > 22,
                    p["temperature"] >= 38.0,
                    p["oxygen_saturation"] < 94,
                ])
                if signs >= 2:
                    self._recently_escalated.add(p["patient_id"])
                    return {
                        "operation": "escalate_to_physician", "patient_id": p["patient_id"],
                        "urgency": "urgent",
                        "rationale": (
                            f"Early warning: HR={p['heart_rate']:.0f} BP={p['systolic_bp']:.0f} "
                            f"RR={p['respiratory_rate']:.0f} T={p['temperature']:.1f} "
                            f"SpO2={p['oxygen_saturation']:.0f}"
                        ),
                    }

        candidates = []
        for p in patients:
            if p["patient_id"] in self._recently_escalated:
                continue
            severe = (
                p["heart_rate"] > 120 or p["systolic_bp"] < 90
                or p["respiratory_rate"] > 24 or p["oxygen_saturation"] < 90
                or p["temperature"] > 38.5
                or p["mental_status"] != "alert"
            )
            if severe:
                candidates.append(p)

        if candidates:
            pick = candidates[0]
            self._recently_escalated.add(pick["patient_id"])
            return {
                "operation": "escalate_to_physician", "patient_id": pick["patient_id"],
                "urgency": "critical",
                "rationale": (
                    f"HR={pick['heart_rate']:.0f} BP={pick['systolic_bp']:.0f} "
                    f"RR={pick['respiratory_rate']:.0f} T={pick['temperature']:.1f} "
                    f"SpO2={pick['oxygen_saturation']:.0f}"
                ),
            }

        for p in patients:
            moderate = p["heart_rate"] > 100 or p["temperature"] > 37.8
            if moderate:
                return {
                    "operation": "request_lab_test", "patient_id": p["patient_id"],
                    "test_type": "lactate",
                }
        return {"operation": "noop"}
