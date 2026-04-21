from __future__ import annotations
from typing import Any, Dict
import gradio as gr
from server.environment import SepsisEnvironment
from agents.nurse import HeuristicNurse
from agents.lab import HeuristicLab
from agents.pharmacist import HeuristicPharmacist
from agents.physician import HeuristicPhysician


def build_dashboard(env: SepsisEnvironment):
    nurse = HeuristicNurse()
    lab = HeuristicLab()
    pharm = HeuristicPharmacist()
    phys = HeuristicPhysician()

    def start_episode(task_name: str, seed: int):
        bundle = env.reset(seed=int(seed), task_name=task_name)
        return _render(env, bundle, "Episode started")

    def step_episode():
        bundle_pre = env._build_obs_bundle()
        obs = bundle_pre["observations"]
        actions = {
            "nurse": nurse.decide(obs["nurse"]),
            "lab": lab.decide(obs["lab"]),
            "pharmacist": pharm.decide(obs["pharmacist"]),
            "physician": phys.decide(obs["physician"]),
        }
        result = env.step({"actions": actions})
        return _render(env, result, f"Step {env._tick - 1} complete")

    def _render(e: SepsisEnvironment, bundle: Dict[str, Any], status: str):
        state = e.state
        patient_lines = []
        for p in state.patients:
            patient_lines.append(
                f"Bed {p['bed_number']} ({p['patient_id']}): {p['outcome']} "
                f"sev={p['infection_severity']:.2f} abx={p['antibiotics_administered']}"
            )
        flag_lines = [
            f"[{f.source_role}] {f.patient_id}: {f.flag_type} ({f.urgency}) -- {f.rationale}"
            for f in state.active_flags
        ]
        score = state.normalized_score if state.normalized_score is not None else "pending"
        summary = (
            f"Task: {state.task_name} | Tick: {state.tick}/{state.max_ticks}\n"
            f"Physician trust: {state.physician_trust}\n"
            f"Escalations: {state.total_escalations}\n"
            f"Team reward: {state.total_team_reward:.3f} | Score: {score}\n\n"
            f"Status: {status}"
        )
        return (
            summary,
            "\n".join(patient_lines) or "(no patients)",
            "\n".join(flag_lines) or "(no active flags this tick)",
        )

    with gr.Blocks(title="SepsisGuard Dashboard") as demo:
        gr.Markdown("# SepsisGuard -- Multi-Agent Sepsis Coordination")
        with gr.Row():
            task = gr.Dropdown(
                choices=["task1_textbook", "task2_atypical", "task3_high_acuity"],
                value="task1_textbook", label="Task",
            )
            seed = gr.Number(value=42, label="Seed", precision=0)
            start_btn = gr.Button("New episode", variant="primary")
            step_btn = gr.Button("Step (heuristic agents)")

        summary_out = gr.Textbox(label="Summary", lines=6)
        patients_out = gr.Textbox(label="Patients", lines=12)
        flags_out = gr.Textbox(label="Active flags this tick", lines=6)

        start_btn.click(start_episode, inputs=[task, seed],
                        outputs=[summary_out, patients_out, flags_out])
        step_btn.click(step_episode, inputs=None,
                       outputs=[summary_out, patients_out, flags_out])
    return demo
