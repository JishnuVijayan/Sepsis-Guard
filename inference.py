"""
SepsisGuard -- LLM Inference Script

Required env vars:
  API_BASE_URL   e.g. https://router.huggingface.co/v1
  MODEL_NAME     e.g. meta-llama/Llama-3.1-8B-Instruct
    OPENAI_API_KEY  Preferred API token
    HF_TOKEN       Fallback HuggingFace API token
  ENV_BASE_URL   URL of the SepsisGuard Space (default: http://localhost:7860)

STDOUT lines:
  [START] task=<name> env=sepsisguard model=<name>
    [STEP] step=<n> action=<json> reward=<r> done=<bool> error=<msg|null>
    [END] success=<bool> steps=<n> score=<s> rewards=<r1,r2,...>
"""
from __future__ import annotations
import os, json, sys, re, textwrap
from typing import Dict, Any, List, Optional
import requests
from openai import OpenAI

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("HF_TOKEN") or os.getenv("API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:7860").rstrip("/")
TASK = os.getenv("SEPSIS_TASK")
ALL_TASKS = ["task1_textbook", "task2_atypical", "task3_high_acuity"]

SYSTEM_PROMPTS = {
    "nurse": textwrap.dedent("""
        You are a bedside nurse in a hospital ward. You see vitals for your 5 assigned patients.
        You cannot see lab values unless the lab flags them. You cannot see medications
        unless the pharmacist flags them. Your job is to escalate to the physician when
        a patient is deteriorating. Over-escalation causes alarm fatigue (the physician stops
        listening). Under-escalation means patients die.
        Respond with ONE JSON object matching NurseAction schema. No explanation.
        Available ops: escalate_to_physician, request_lab_test, administer_medication, flag_concern, noop.
        Always include rationale for escalations -- the physician reads it.
    """).strip(),
    "lab": textwrap.dedent("""
        You are a clinical lab analyst. You see lab values for all 10 patients.
        You cannot see vitals. You flag critical values to alert the team. Correctly flagging
        abnormal labs on truly septic patients is rewarded; flagging normal labs is penalised.
        Respond with ONE JSON object matching LabAction schema.
        Available ops: release_result, flag_critical, recommend_followup_test, noop.
    """).strip(),
    "pharmacist": textwrap.dedent("""
        You are a clinical pharmacist. You see medications and the antibiogram for all 10 patients.
        You must flag immunosuppression (masks sepsis signs), recommend empirical antibiotics,
        and flag interactions. Avoid recommending antibiotics with high resistance rates.
        Respond with ONE JSON object matching PharmacistAction schema.
        Available ops: flag_interaction, flag_immunosuppression, recommend_antibiotic, check_dosing, noop.
    """).strip(),
    "physician": textwrap.dedent("""
        You are the attending physician. You see ONLY patients that the nurse, lab, or pharmacist
        escalated or flagged this tick. Multi-source flags (nurse + lab on same patient) are the strongest
        signal. Order antibiotics within 1 hour of a valid escalation. Do not order antibiotics on
        false-alarm patients. Dismissing a multi-source valid escalation is the worst outcome.
        Respond with ONE JSON object matching PhysicianAction schema.
        Available ops: order_antibiotics, order_lab_test, admit_to_icu, request_consult, do_nothing.
    """).strip(),
}


def log(prefix: str, msg: str): print(f"[{prefix}] {msg}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]):
    err = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={str(done).lower()} error={err}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={float(score):.4f} rewards={rewards_str}",
        flush=True,
    )

def env_reset(task: str, seed: int) -> Dict[str, Any]:
    r = requests.post(f"{ENV_BASE_URL}/reset", json={"task_name": task, "seed": seed}, timeout=30)
    r.raise_for_status()
    return r.json()

def env_step(actions: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    r = requests.post(f"{ENV_BASE_URL}/step", json={"actions": actions}, timeout=30)
    r.raise_for_status()
    return r.json()


def heuristic_fallback(role: str, obs: Dict[str, Any]) -> Dict[str, Any]:
    from agents.nurse import HeuristicNurse
    from agents.lab import HeuristicLab
    from agents.pharmacist import HeuristicPharmacist
    from agents.physician import HeuristicPhysician
    agents = {
        "nurse": HeuristicNurse(), "lab": HeuristicLab(),
        "pharmacist": HeuristicPharmacist(), "physician": HeuristicPhysician(),
    }
    return agents[role].decide(obs)


def build_user_prompt(role: str, obs: Dict[str, Any]) -> str:
    return f"Current observation (JSON):\n{json.dumps(obs, default=str)[:4000]}\n\nRespond with ONE JSON action."


def parse_action(text: str, role: str, obs: Dict[str, Any]) -> Dict[str, Any]:
    text = text.strip()
    try:
        if text.startswith("```"):
            text = text.strip("`").strip()
            if text.startswith("json"):
                text = text[4:].strip()
        parsed = json.loads(text)
        if isinstance(parsed, dict) and "operation" in parsed:
            return parsed
    except Exception:
        pass
    m = re.search(r"\{[^{}]*\"operation\"[^{}]*\}", text)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass
    return heuristic_fallback(role, obs)


def run_task(task_name: str, seed: int, client: Optional[OpenAI]):
    log("START", f"task={task_name} env=sepsisguard model={MODEL_NAME}")
    bundle = env_reset(task_name, seed)
    done = False
    steps = 0
    last_result: Dict[str, Any] = {}
    using_fallback = client is None
    if using_fallback:
        log("WARN", "No OPENAI_API_KEY / HF_TOKEN / API_KEY -- using heuristic fallback for all episodes")
    while not done:
        obs = bundle["observations"]
        actions: Dict[str, Dict[str, Any]] = {}
        for role in ("nurse", "lab", "pharmacist", "physician"):
            if using_fallback:
                actions[role] = heuristic_fallback(role, obs[role])
                continue
            try:
                resp = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPTS[role]},
                        {"role": "user", "content": build_user_prompt(role, obs[role])},
                    ],
                    temperature=0.2,
                    max_tokens=160,
                )
                text = resp.choices[0].message.content or ""
                actions[role] = parse_action(text, role, obs[role])
            except Exception as exc:
                log("WARN", f"LLM failure for {role}: {exc} -- fallback")
                using_fallback = True
                actions[role] = heuristic_fallback(role, obs[role])
        step_error: Optional[str] = None
        bundle = env_step(actions)
        steps = int(bundle["info"]["tick"])
        done = bundle["done"]
        action_str = json.dumps(actions, separators=(",", ":"))
        log_step(
            step=steps,
            action=action_str,
            reward=float(sum(bundle.get("rewards", {}).values())),
            done=done,
            error=step_error,
        )
        last_result = bundle

    grader = requests.get(f"{ENV_BASE_URL}/grader").json()
    score = grader.get("score")
    passed = (grader.get("metrics") or {}).get("passed", False)
    agent_rewards = (grader.get("agent_rewards") or {})
    rewards = [float(agent_rewards.get(role, 0.0)) for role in ("nurse", "lab", "pharmacist", "physician")]
    log_end(success=bool(passed), steps=steps, score=float(score or 0.0), rewards=rewards)


def main():
    client: Optional[OpenAI] = None
    if not API_KEY:
        log("WARN", "No OPENAI_API_KEY / HF_TOKEN / API_KEY -- using heuristic baseline for all episodes")
    else:
        try:
            client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
        except Exception as exc:
            log("WARN", f"OpenAI client init failed: {exc} -- heuristic fallback")
    tasks = [TASK] if TASK else ALL_TASKS
    for i, t in enumerate(tasks):
        run_task(t, seed=42 + i, client=client)


if __name__ == "__main__":
    main()
