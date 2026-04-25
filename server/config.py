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
    "mean_arterial_pressure": (70, 100),
    "diastolic_bp": (55, 90),
    "respiratory_rate": (12, 20),
    "temperature": (36.1, 37.2),
    "oxygen_saturation": (95, 100),
}

LAB_NORMAL_RANGES = {
    "lactate": (0.5, 2.0),
    "wbc": (4.0, 11.0),
    "procalcitonin": (0.0, 0.5),
    "creatinine": (0.6, 1.3),
    "bun": (7.0, 20.0),
    "bilirubin_total": (0.2, 1.2),
    "platelets": (150.0, 400.0),
    "glucose": (70.0, 140.0),
    "hemoglobin": (12.0, 17.0),
    "ptt": (25.0, 35.0),
    "fibrinogen": (200.0, 400.0),
    "bicarbonate": (22.0, 28.0),
    "ph": (7.35, 7.45),
    "paco2": (35.0, 45.0),
    "sao2": (94.0, 100.0),
    "base_excess": (-2.0, 2.0),
}

ANTIBIOGRAM = {
    "piperacillin_tazobactam": 0.08,
    "meropenem": 0.04,
    "ceftriaxone": 0.22,
    "vancomycin": 0.12,
    "ciprofloxacin": 0.35,
}

VALID_ANTIBIOTICS = list(ANTIBIOGRAM.keys())

CORE_VITAL_FIELDS = [
    "heart_rate",
    "systolic_bp",
    "mean_arterial_pressure",
    "diastolic_bp",
    "respiratory_rate",
    "temperature",
    "oxygen_saturation",
]

LAB_FIELDS = [
    "lactate",
    "wbc",
    "procalcitonin",
    "creatinine",
    "bun",
    "bilirubin_total",
    "platelets",
    "glucose",
    "hemoglobin",
    "ptt",
    "fibrinogen",
    "bicarbonate",
    "ph",
    "paco2",
    "sao2",
    "base_excess",
    "blood_culture",
]

DEFAULT_TEST_DELAYS = {
    "lactate": 1,
    "wbc": 1,
    "procalcitonin": 2,
    "creatinine": 1,
    "bun": 1,
    "bilirubin_total": 2,
    "platelets": 1,
    "glucose": 0,
    "hemoglobin": 1,
    "ptt": 2,
    "fibrinogen": 2,
    "bicarbonate": 1,
    "ph": 1,
    "paco2": 1,
    "sao2": 0,
    "base_excess": 1,
    "blood_culture": 4,
}
