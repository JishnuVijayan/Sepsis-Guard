from __future__ import annotations
import json
from typing import Dict, Any

SYSTEM_PROMPTS = {
    "nurse": "You are a bedside nurse. Respond with ONE JSON action.",
    "lab": "You are a clinical lab analyst. Respond with ONE JSON action.",
    "pharmacist": "You are a clinical pharmacist. Respond with ONE JSON action.",
    "physician": "You are the attending physician. Respond with ONE JSON action.",
}


def build_role_prompt(obs: Dict[str, Any], role: str) -> str:
    return (
        f"<SYS>{SYSTEM_PROMPTS[role]}</SYS>\n"
        f"Role: {role}\n"
        f"Observation:\n{json.dumps(obs, default=str)[:3000]}\n"
        f"Action (JSON):"
    )
