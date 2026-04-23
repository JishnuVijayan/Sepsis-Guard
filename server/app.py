from __future__ import annotations
import os
from typing import Any, Dict
from fastapi import FastAPI, Body, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

from server.environment import SepsisEnvironment
from server.config import TASK_CONFIGS
from models import StepRequest

app = FastAPI(title="SepsisGuard OpenEnv", version="0.1.0")
_env = SepsisEnvironment()


class ResetBody(BaseModel):
    seed: int = 42
    task_name: str = "task1_textbook"


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/reset")
def reset(body: ResetBody = Body(default_factory=ResetBody)):
    return _env.reset(seed=body.seed, task_name=body.task_name)


class StepBody(BaseModel):
    actions: Dict[str, Dict[str, Any]]


@app.post("/step")
def step(body: StepBody):
    try:
        return _env.step({"actions": body.actions})
    except (ValueError, KeyError, TypeError) as e:
        raise HTTPException(422, str(e))


@app.get("/state")
def get_state():
    return _env.state.model_dump()


@app.post("/observations")
def get_observations(payload: Dict[str, str] = Body(...)):
    role = payload.get("agent_role") or payload.get("role")
    if role not in ("nurse", "lab", "pharmacist", "physician"):
        raise HTTPException(400, f"Invalid role: {role}")
    bundle = _env._build_obs_bundle()
    return bundle["observations"][role]


@app.get("/tasks")
def tasks():
    return {
        "tasks": [
            {"task_name": name, **cfg} for name, cfg in TASK_CONFIGS.items()
        ],
        "action_schemas": {
            "nurse": StepRequest.model_json_schema()["$defs"].get("NurseAction", {}),
            "lab": StepRequest.model_json_schema()["$defs"].get("LabAction", {}),
            "pharmacist": StepRequest.model_json_schema()["$defs"].get("PharmacistAction", {}),
            "physician": StepRequest.model_json_schema()["$defs"].get("PhysicianAction", {}),
        },
    }


@app.get("/grader")
def grader():
    data = _env.last_grader_data
    if not data:
        return {"status": "no_episode_completed"}
    return {"status": "ok", **data}


@app.get("/baseline")
def baseline():
    from agents.nurse import HeuristicNurse
    from agents.lab import HeuristicLab
    from agents.pharmacist import HeuristicPharmacist
    from agents.physician import HeuristicPhysician

    nurse = HeuristicNurse()
    lab = HeuristicLab()
    pharm = HeuristicPharmacist()
    phys = HeuristicPhysician()
    results = []
    for task_name in TASK_CONFIGS:
        env = SepsisEnvironment()
        bundle = env.reset(seed=42, task_name=task_name)
        done = False
        while not done:
            obs = bundle["observations"]
            actions = {
                "nurse": nurse.decide(obs["nurse"]),
                "lab": lab.decide(obs["lab"]),
                "pharmacist": pharm.decide(obs["pharmacist"]),
                "physician": phys.decide(obs["physician"]),
            }
            bundle = env.step({"actions": actions})
            done = bundle["done"]
        grader_data = env.last_grader_data
        results.append({
            "task": task_name,
            "score": grader_data.get("score"),
            "metrics": grader_data.get("metrics", {}),
        })
    return {"results": results}


@app.get("/", response_class=HTMLResponse)
def root():
    return """
    <html><head><title>SepsisGuard</title></head>
    <body style="font-family: sans-serif; max-width: 800px; margin: 2rem auto">
    <h1>SepsisGuard &mdash; Multi-Agent Sepsis Coordination Environment</h1>
    <p>OpenEnv-compatible environment for training clinical coordination.</p>
    <ul>
      <li><a href="/docs">Swagger UI</a></li>
      <li><a href="/health">Health</a></li>
      <li><a href="/tasks">Tasks</a></li>
      <li><a href="/dashboard">Dashboard</a></li>
    </ul>
    </body></html>
    """


try:
    from server.dashboard import build_dashboard
    import gradio as gr
    demo = build_dashboard(_env)
    app = gr.mount_gradio_app(app, demo, path="/dashboard")
except Exception as e:
    print(f"[WARN] Dashboard not mounted: {e}")


def main():
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
