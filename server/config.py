from __future__ import annotations
from enum import Enum
from typing import Any, Dict

class AgentRole(str, Enum):
    NURSE = "nurse"
    LAB = "lab"
    PHARMACIST = "pharmacist"
    PHYSICIAN = "physician"

class Outcome(str, Enum):
    STABLE = "stable"
    DETERIORATING = "deteriorating"
    SEPTIC_SHOCK = "septic_shock"
    RECOVERED = "recovered"
    DIED = "died"

class MentalStatus(str, Enum):
    ALERT = "alert"
    CONFUSED = "confused"
    UNRESPONSIVE = "unresponsive"

TASK_CONFIGS: Dict[str, Dict[str, Any]] = {
    "task1_textbook": {
        "description": "Single textbook sepsis case — warm-up",
        "difficulty": "easy",
        "n_patients": 5,
        "n_sepsis_cases": 1,
        "n_false_alarms": 0,
        "max_steps": 48,
        "success_threshold": 0.70,
        "lab_result_delay": 1,
        "full_info_sharing": True,
    },
    "task2_atypical": {
        "description": "Atypical presentations, full asymmetric observability",
        "difficulty": "medium",
        "n_patients": 10,
        "n_sepsis_cases": 3,
        "n_false_alarms": 0,
        "max_steps": 96,
        "success_threshold": 0.55,
        "lab_result_delay": 2,
        "full_info_sharing": False,
    },
    "task3_high_acuity": {
        "description": "Mixed sepsis + false alarms + lab delays",
        "difficulty": "hard",
        "n_patients": 10,
        "n_sepsis_cases": 4,
        "n_false_alarms": 2,
        "max_steps": 144,
        "success_threshold": 0.40,
        "lab_result_delay": 3,
        "full_info_sharing": False,
    },
}

DEFAULT_TASK = "task1_textbook"

VITAL_NORMAL_RANGES = {
    "heart_rate": (60, 100),
    "systolic_bp": (90, 140),
    "respiratory_rate": (12, 20),
    "temperature": (36.1, 37.2),
    "oxygen_saturation": (95, 100),
}

LAB_NORMAL_RANGES = {
    "lactate": (0.5, 2.0),
    "wbc": (4.0, 11.0),
    "procalcitonin": (0.0, 0.5),
    "creatinine": (0.6, 1.3),
}

ANTIBIOGRAM = {
    "piperacillin_tazobactam": 0.08,
    "meropenem": 0.04,
    "ceftriaxone": 0.22,
    "vancomycin": 0.12,
    "ciprofloxacin": 0.35,
}

VALID_ANTIBIOTICS = list(ANTIBIOGRAM.keys())
