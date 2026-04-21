from __future__ import annotations
from typing import List, Dict, Any
import json, re
from training.prompts import build_role_prompt

ROLES = ("nurse", "lab", "pharmacist", "physician")


def generate_completion(model, tokenizer, prompt: str, max_new_tokens: int = 96) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    out = model.generate(
        **inputs, max_new_tokens=max_new_tokens,
        do_sample=True, temperature=0.7, top_p=0.9,
    )
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def parse_action(text: str, role: str) -> Dict[str, Any]:
    text = text.strip()
    try:
        parsed = json.loads(text)
        if "operation" in parsed:
            return parsed
    except Exception:
        pass
    m = re.search(r"\{[^{}]*\"operation\"[^{}]*\}", text)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass
    return {"operation": "noop" if role != "physician" else "do_nothing"}


def is_valid_action_json(text: str) -> bool:
    try:
        parsed = json.loads(text.strip())
        return isinstance(parsed, dict) and "operation" in parsed
    except Exception:
        return False


def collect_rollouts(model, tokenizer, env_client, n_episodes: int = 4, task="task1_textbook"):
    """Returns list of {prompt, completion, env_reward, role} dicts."""
    rollouts: List[Dict[str, Any]] = []
    for ep in range(n_episodes):
        bundle = env_client.reset(task_name=task, seed=42 + ep)
        done = False
        while not done:
            obs = bundle["observations"]
            actions: Dict[str, Dict[str, Any]] = {}
            batch_this_tick: List[Dict[str, Any]] = []
            for role in ROLES:
                prompt = build_role_prompt(obs[role], role)
                completion = generate_completion(model, tokenizer, prompt)
                actions[role] = parse_action(completion, role)
                batch_this_tick.append({
                    "prompt": prompt, "completion": completion, "role": role,
                    "env_reward_placeholder": None,
                })
            bundle = env_client.step(actions)
            for i, role in enumerate(ROLES):
                batch_this_tick[i]["env_reward_placeholder"] = float(bundle["rewards"][role])
            rollouts.extend(batch_this_tick)
            done = bundle["done"]
    return rollouts
