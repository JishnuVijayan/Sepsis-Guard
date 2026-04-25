from __future__ import annotations

import json
from typing import Dict, Any

STRICT_JSON_RULE = (
    "\n\nRESPONSE FORMAT: Return exactly one valid JSON object parseable by json.loads(). "
    "No markdown, no prose, no code fences.\n"
    "Use only the fields required by your action schema. "
    "When you take a meaningful action, include a concrete rationale with numeric values from the observation. "
    "Never invent unseen data."
)

NURSE_GUIDANCE = (
    "\n\nROLE MODEL:\n"
    "You are the bedside nurse. You only see assigned patients. You must detect early bedside deterioration "
    "without flooding the physician with false alarms.\n"
    "Key visible signals: heart_rate, systolic_bp, mean_arterial_pressure, respiratory_rate, temperature, "
    "oxygen_saturation, mental_status, qsofa_score, iculos_hours, and whether antibiotics are already given.\n"
    "Escalate when there is meaningful deterioration: qSOFA >= 2, MAP < 65, SBP <= 100 with tachypnea, "
    "confusion, SpO2 < 92-94, or multiple moderate abnormalities together.\n"
    "If vitals are only borderline and you need data, prefer request_lab_test or flag_concern.\n"
    "If the patient is stable or already treated, noop is often correct."
)

LAB_GUIDANCE = (
    "\n\nROLE MODEL:\n"
    "You are the lab analyst. You see the ward-wide lab table, including missingness and staleness information.\n"
    "Critical patterns include lactate > 2.2, rising creatinine/BUN, low platelets, bilirubin elevation, "
    "marked WBC abnormality, metabolic acidosis, or several mildly abnormal labs that suggest organ dysfunction.\n"
    "Use flag_critical when the current lab picture should move the team toward sepsis detection now.\n"
    "Use recommend_followup_test when evidence is incomplete or stale and another test would sharpen the picture.\n"
    "If labs are normal, missing, or stale without a strong signal, noop is acceptable."
)

PHARM_GUIDANCE = (
    "\n\nROLE MODEL:\n"
    "You are the clinical pharmacist. You see medications, immunocompromised status, antibiogram resistance rates, "
    "and limited renal-safety context.\n"
    "Always consider immunosuppression because fever and WBC can be blunted.\n"
    "Recommend antibiotics only when there is a meaningful infection signal from lab flags or broader context.\n"
    "Prefer lower-resistance empiric options and be careful with renal risk when creatinine is elevated or stale.\n"
    "If there is no clear medication-safety or antibiotic contribution to make, noop is correct."
)

PHYSICIAN_GUIDANCE = (
    "\n\nROLE MODEL:\n"
    "You are the physician. You now retain memory of previously escalated patients, but you still only see what the team "
    "has surfaced. The best signal is converging evidence across time and across roles.\n"
    "Treat early when the visible evidence supports sepsis: nurse deterioration plus lab abnormalities, worsening organ "
    "dysfunction, persistent high qSOFA, rising lactate/creatinine, confusion, hypotension, or repeated concern over time.\n"
    "Do not order antibiotics reflexively on weak or isolated evidence. If the chart is incomplete, order_lab_test.\n"
    "Use admit_to_icu for severe shock or obvious instability. If nothing actionable is surfaced, do_nothing is appropriate."
)

SYSTEM_PROMPTS = {
    "nurse": (
        "You are a bedside nurse in SepsisGuard, a multi-agent sepsis coordination environment."
        + NURSE_GUIDANCE +
        "\n\nOUTPUT EXAMPLES:\n"
        '{"operation":"escalate_to_physician","patient_id":"P02","urgency":"critical","rationale":"qSOFA 2, MAP 61, RR 28, confused"}\n'
        '{"operation":"request_lab_test","patient_id":"P03","test_type":"lactate","rationale":"HR 108 and RR 22 with borderline BP"}\n'
        '{"operation":"flag_concern","patient_id":"P01","rationale":"new tachycardia and fever, watching closely"}\n'
        '{"operation":"noop"}'
        + STRICT_JSON_RULE
    ),
    "lab": (
        "You are the lab analyst in SepsisGuard."
        + LAB_GUIDANCE +
        "\n\nOUTPUT EXAMPLES:\n"
        '{"operation":"flag_critical","patient_id":"P03","reason":"lactate 4.1, creatinine 2.0, platelets 96 suggest organ dysfunction"}\n'
        '{"operation":"recommend_followup_test","patient_id":"P05","test":"blood_culture","reason":"lactate elevated and no culture yet"}\n'
        '{"operation":"noop"}'
        + STRICT_JSON_RULE
    ),
    "pharmacist": (
        "You are the clinical pharmacist in SepsisGuard."
        + PHARM_GUIDANCE +
        "\n\nOUTPUT EXAMPLES:\n"
        '{"operation":"flag_immunosuppression","patient_id":"P04","rationale":"on tacrolimus, sepsis may present without fever or leukocytosis"}\n'
        '{"operation":"recommend_antibiotic","patient_id":"P03","drug":"piperacillin_tazobactam","rationale":"broad empiric coverage with lower resistance than ciprofloxacin"}\n'
        '{"operation":"flag_interaction","patient_id":"P02","rationale":"multiple medications raise interaction risk"}\n'
        '{"operation":"noop"}'
        + STRICT_JSON_RULE
    ),
    "physician": (
        "You are the attending physician in SepsisGuard."
        + PHYSICIAN_GUIDANCE +
        "\n\nOUTPUT EXAMPLES:\n"
        '{"operation":"order_antibiotics","patient_id":"P03","drug":"piperacillin_tazobactam"}\n'
        '{"operation":"order_lab_test","patient_id":"P04","test":"blood_culture"}\n'
        '{"operation":"admit_to_icu","patient_id":"P02"}\n'
        '{"operation":"do_nothing"}'
        + STRICT_JSON_RULE
    ),
}

_NOISE_FIELDS = frozenset({
    "done",
    "reward",
    "cumulative_reward",
    "normalized_score",
    "last_action_result",
})


def _trim_flag_history(flags: list[Dict[str, Any]], limit: int = 6) -> list[Dict[str, Any]]:
    return flags[-limit:]


def safe_truncate_obs(obs: Dict[str, Any], max_patients: int = 5) -> Dict[str, Any]:
    """
    Remove training-noise fields and trim patient lists so prompts stay compact
    while preserving the clinically important context added by the richer env.
    """
    truncated = {k: v for k, v in obs.items() if k not in _NOISE_FIELDS}

    if "patient_vitals" in truncated and isinstance(truncated["patient_vitals"], list):
        truncated["patient_vitals"] = truncated["patient_vitals"][:max_patients]

    if "lab_results" in truncated and isinstance(truncated["lab_results"], list):
        trimmed = []
        for row in truncated["lab_results"][:max_patients]:
            row = dict(row)
            ages = row.get("last_measured_tick")
            if isinstance(ages, dict):
                row["last_measured_tick"] = {
                    k: ages[k] for k in list(ages)[:8]
                }
            trimmed.append(row)
        truncated["lab_results"] = trimmed

    if "patient_medications" in truncated and isinstance(truncated["patient_medications"], list):
        truncated["patient_medications"] = truncated["patient_medications"][:max_patients]

    if "known_patient_summaries" in truncated and isinstance(truncated["known_patient_summaries"], list):
        trimmed = []
        for summary in truncated["known_patient_summaries"][:max_patients]:
            summary = dict(summary)
            if isinstance(summary.get("flag_history"), list):
                summary["flag_history"] = _trim_flag_history(summary["flag_history"])
            if isinstance(summary.get("flags_raised"), list):
                summary["flags_raised"] = summary["flags_raised"][-4:]
            trimmed.append(summary)
        truncated["known_patient_summaries"] = trimmed

    if "pending_labs" in truncated and isinstance(truncated["pending_labs"], list):
        truncated["pending_labs"] = truncated["pending_labs"][:10]

    return truncated


def build_role_prompt(obs: Dict[str, Any], role: str) -> str:
    safe_obs = safe_truncate_obs(obs)
    obs_str = json.dumps(safe_obs, default=str)
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPTS[role]}<|im_end|>\n"
        f"<|im_start|>user\nRole: {role}\nObservation:\n{obs_str}\nAction (JSON):<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
