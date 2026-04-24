from __future__ import annotations
import json
from typing import Dict, Any

# ── Clinical thresholds ────────────────────────────────────────────────────────
NURSE_THRESHOLDS = (
    "\n\nCLINICAL ALERT THRESHOLDS — escalate if ANY of these are present:\n"
    "  Heart rate     > 100 bpm (tachycardia) or < 50 bpm\n"
    "  Systolic BP    < 90 mmHg (hypotension)\n"
    "  Resp rate      > 20 breaths/min (tachypnea)\n"
    "  Temperature    > 38.3 C (fever) or < 36.0 C (hypothermia)\n"
    "  SpO2           < 94 percent\n"
    "  Mental status  = confused or unresponsive\n"
    "Normal ranges: HR 60-100, BP 90-140, RR 12-20, Temp 36.1-37.2, SpO2 95-100.\n"
    "IMPORTANT: HR 70, BP 120, RR 16, Temp 36.8, SpO2 98 are all NORMAL — do NOT escalate.\n"
    "If ALL vitals for ALL your patients are normal, respond with noop."
)

LAB_THRESHOLDS = (
    "\n\nCRITICAL LAB VALUES — flag_critical if ANY of these:\n"
    "  Lactate        > 2.0 mmol/L (> 4.0 = critical sepsis)\n"
    "  WBC            > 12 or < 4 x10^9/L\n"
    "  Procalcitonin  > 0.5 ng/mL\n"
    "  Creatinine     > 1.5 mg/dL or rapidly rising\n"
    "Normal: Lactate 0.5-2.0, WBC 4-11, PCT < 0.5, Creatinine 0.6-1.3.\n"
    "Only flag when values are genuinely abnormal. Normal values do NOT need flagging.\n"
    "If ALL lab values for ALL patients are normal or null, respond with noop."
)

PHARMACIST_THRESHOLDS = (
    "\n\nKEY RULES:\n"
    "  Immunosuppressants (prednisone, tacrolimus, methotrexate, rituximab) mask fever "
    "and WBC elevation — ALWAYS flag these patients, even with borderline lab values.\n"
    "  Empirical antibiotic priority: piperacillin_tazobactam (if resistance < 15%), "
    "else meropenem. Avoid ciprofloxacin (resistance > 35%).\n"
    "If no patients are immunocompromised and no antibiotic action is needed, respond with noop."
)

STRICT_JSON_RULE = (
    "\n\nRESPONSE FORMAT: Your entire response must be a single valid JSON object "
    "parseable by json.loads(). No markdown, no explanation, no extra text.\n"
    "Always include a specific rationale with real clinical values from the observation. "
    "Never use placeholder text like '...' in any field."
)

# ── System prompts ─────────────────────────────────────────────────────────────
SYSTEM_PROMPTS = {
    "nurse": (
        "You are a bedside nurse managing the patients listed in 'assigned_patient_ids'. "
        "You ONLY see vitals for these assigned patients, not the full ward.\n\n"
        "DECISION LOGIC:\n"
        "1. Check each patient's vitals against the alert thresholds below.\n"
        "2. If ANY threshold is breached: escalate_to_physician with urgency and cite the specific abnormal values.\n"
        "3. If vitals are borderline (e.g. HR 95-100): use flag_concern and request a lab test.\n"
        "4. If ALL vitals are normal: respond with noop. Over-escalation degrades physician trust.\n\n"
        "Respond with ONE JSON object. Examples:\n"
        '{"operation": "escalate_to_physician", "patient_id": "P02", "urgency": "critical", '
        '"rationale": "HR 118, BP 85, temp 38.9, SpO2 93 — meets sepsis criteria"}\n'
        '{"operation": "request_lab_test", "patient_id": "P03", "test_type": "lactate"}\n'
        '{"operation": "flag_concern", "patient_id": "P01", '
        '"rationale": "HR 105 borderline tachycardia, requesting lactate"}\n'
        '{"operation": "noop"}'
    ) + NURSE_THRESHOLDS + STRICT_JSON_RULE,

    "lab": (
        "You are a clinical lab analyst. You see lab values for ALL ward patients. "
        "You cannot see vitals.\n\n"
        "DECISION LOGIC:\n"
        "1. Review each patient's lab values against the critical thresholds below.\n"
        "2. If ANY lab value is critically abnormal: flag_critical with the specific abnormal value.\n"
        "3. If a patient has some abnormal labs but needs more data: recommend_followup_test.\n"
        "4. If ALL lab values are normal or null (pending): respond with noop.\n\n"
        "Respond with ONE JSON object. Examples:\n"
        '{"operation": "flag_critical", "patient_id": "P03", '
        '"reason": "lactate 4.2 mmol/L — critical, sepsis likely"}\n'
        '{"operation": "recommend_followup_test", "patient_id": "P05", '
        '"test": "blood_culture", "reason": "PCT 1.8, WBC 14 — culture before antibiotics"}\n'
        '{"operation": "noop"}'
    ) + LAB_THRESHOLDS + STRICT_JSON_RULE,

    "pharmacist": (
        "You are a clinical pharmacist. You see medications and immunocompromised status "
        "for all ward patients, plus the antibiogram (resistance rates by antibiotic).\n\n"
        "DECISION LOGIC:\n"
        "1. Check if any patient is immunocompromised (on immunosuppressants): flag_immunosuppression.\n"
        "2. If lab flags indicate infection, recommend an antibiotic with low resistance.\n"
        "3. Check for dangerous drug interactions: flag_interaction.\n"
        "4. If no immunosuppressed patients and no antibiotic recommendations needed: respond with noop.\n\n"
        "Respond with ONE JSON object. Examples:\n"
        '{"operation": "flag_immunosuppression", "patient_id": "P04", '
        '"rationale": "on tacrolimus — fever may be absent despite severe infection"}\n'
        '{"operation": "recommend_antibiotic", "patient_id": "P03", '
        '"drug": "piperacillin_tazobactam", "rationale": "8% resistance, gram-neg cover"}\n'
        '{"operation": "flag_interaction", "patient_id": "P02", '
        '"rationale": "warfarin + ciprofloxacin: INR risk"}\n'
        '{"operation": "noop"}'
    ) + PHARMACIST_THRESHOLDS + STRICT_JSON_RULE,

    "physician": (
        "You are the attending physician. You ONLY see patients that were escalated or "
        "flagged this tick — you have no visibility into patients who were not flagged.\n\n"
        "DECISION LOGIC:\n"
        "1. If you see escalations from MULTIPLE sources (nurse AND lab) for the same patient: "
        "act immediately — order_antibiotics with an appropriate drug.\n"
        "2. If a single source escalated with critical urgency and vitals/labs confirm sepsis: order_antibiotics.\n"
        "3. If information is incomplete: order_lab_test to gather more data.\n"
        "4. If the patient is in septic shock (multi-organ failure): admit_to_icu.\n"
        "5. If NO patients were escalated to you this tick: respond with do_nothing.\n\n"
        "Respond with ONE JSON object. Examples:\n"
        '{"operation": "order_antibiotics", "patient_id": "P03", '
        '"drug": "piperacillin_tazobactam"}\n'
        '{"operation": "order_lab_test", "patient_id": "P04", "test": "lactate"}\n'
        '{"operation": "admit_to_icu", "patient_id": "P02"}\n'
        '{"operation": "do_nothing"}'
    ) + STRICT_JSON_RULE,
}

# ── Observation truncation ─────────────────────────────────────────────────────
_NOISE_FIELDS = frozenset({
    "done", "reward", "cumulative_reward", "normalized_score", "last_action_result",
})

def safe_truncate_obs(obs: Dict[str, Any], max_patients: int = 5) -> Dict[str, Any]:
    """
    Remove training-noise fields and truncate patient lists so prompts
    stay well under max_prompt_length=3000.
    """
    truncated = {k: v for k, v in obs.items() if k not in _NOISE_FIELDS}

    # Nurse
    if "patient_vitals" in truncated and isinstance(truncated["patient_vitals"], list):
        truncated["patient_vitals"] = truncated["patient_vitals"][:max_patients]

    # Lab
    if "lab_results" in truncated and isinstance(truncated["lab_results"], list):
        truncated["lab_results"] = truncated["lab_results"][:max_patients]

    # Pharmacist
    if "patient_medications" in truncated and isinstance(truncated["patient_medications"], list):
        truncated["patient_medications"] = truncated["patient_medications"][:max_patients]

    # Physician
    if "known_patient_summaries" in truncated and isinstance(truncated["known_patient_summaries"], list):
        truncated["known_patient_summaries"] = truncated["known_patient_summaries"][:max_patients]

    return truncated

# ── Prompt builder ─────────────────────────────────────────────────────────────
def build_role_prompt(obs: Dict[str, Any], role: str) -> str:
    safe_obs = safe_truncate_obs(obs)
    obs_str = json.dumps(safe_obs, default=str)

    return (
        f"<|im_start|>system\n{SYSTEM_PROMPTS[role]}<|im_end|>\n"
        f"<|im_start|>user\nRole: {role}\nObservation:\n{obs_str}\nAction (JSON):<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
