from __future__ import annotations
import os
import uuid
import threading
from typing import Any, Dict, Optional
from fastapi import FastAPI, Body, HTTPException, Header
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

from server.environment import SepsisEnvironment
from server.config import TASK_CONFIGS
from models import StepRequest

app = FastAPI(title="SepsisGuard OpenEnv", version="0.1.0")

_lock = threading.Lock()
_sessions: Dict[str, SepsisEnvironment] = {}
_default_env = SepsisEnvironment()

MAX_SESSIONS = 64


def _get_env(session_id: Optional[str]) -> SepsisEnvironment:
    if not session_id:
        return _default_env
    with _lock:
        env = _sessions.get(session_id)
    if env is None:
        raise HTTPException(404, f"Session {session_id} not found. Call /reset first.")
    return env


class ResetBody(BaseModel):
    seed: int = 42
    task_name: str = "task1_textbook"


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/reset")
def reset(
    body: ResetBody = Body(default_factory=ResetBody),
    x_session_id: Optional[str] = Header(None),
):
    if x_session_id:
        session_id = x_session_id
    else:
        session_id = None

    if session_id:
        with _lock:
            if session_id not in _sessions:
                if len(_sessions) >= MAX_SESSIONS:
                    oldest = next(iter(_sessions))
                    del _sessions[oldest]
                _sessions[session_id] = SepsisEnvironment()
            env = _sessions[session_id]
    else:
        env = _default_env

    result = env.reset(seed=body.seed, task_name=body.task_name)
    if session_id:
        result["session_id"] = session_id
    return result


class StepBody(BaseModel):
    actions: Dict[str, Dict[str, Any]]


@app.post("/step")
def step(
    body: StepBody,
    x_session_id: Optional[str] = Header(None),
):
    env = _get_env(x_session_id)
    try:
        return env.step({"actions": body.actions})
    except (ValueError, KeyError, TypeError) as e:
        raise HTTPException(422, str(e))


@app.get("/state")
def get_state(x_session_id: Optional[str] = Header(None)):
    env = _get_env(x_session_id)
    return env.state.model_dump()


@app.post("/observations")
def get_observations(
    payload: Dict[str, str] = Body(...),
    x_session_id: Optional[str] = Header(None),
):
    role = payload.get("agent_role") or payload.get("role")
    if role not in ("nurse", "lab", "pharmacist", "physician"):
        raise HTTPException(400, f"Invalid role: {role}")
    env = _get_env(x_session_id)
    bundle = env._build_obs_bundle()
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
def grader(x_session_id: Optional[str] = Header(None)):
    env = _get_env(x_session_id)
    data = env.last_grader_data
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


@app.post("/session")
def create_session():
    session_id = uuid.uuid4().hex[:12]
    with _lock:
        if len(_sessions) >= MAX_SESSIONS:
            oldest = next(iter(_sessions))
            del _sessions[oldest]
        _sessions[session_id] = SepsisEnvironment()
    return {"session_id": session_id}


@app.delete("/session/{session_id}")
def delete_session(session_id: str):
    with _lock:
        if session_id in _sessions:
            del _sessions[session_id]
            return {"status": "deleted"}
    raise HTTPException(404, f"Session {session_id} not found")


@app.get("/", response_class=HTMLResponse)
def root():
    return """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>SepsisGuard — Multi-Agent Sepsis Coordination</title>
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

    :root {
      --red:    #e63946;
      --navy:   #0d1b2a;
      --blue:   #1b3a5c;
      --teal:   #1d8a8a;
      --light:  #f0f4f8;
      --muted:  #8090a0;
      --white:  #ffffff;
      --green:  #2a9d5c;
      --orange: #e07b39;
    }

    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      background: var(--light);
      color: var(--navy);
      line-height: 1.6;
    }

    /* ── HERO ── */
    .hero {
      background: linear-gradient(135deg, var(--navy) 0%, var(--blue) 100%);
      color: var(--white);
      padding: 3.5rem 2rem 3rem;
      text-align: center;
      position: relative;
      overflow: hidden;
    }
    .hero::before {
      content: "";
      position: absolute; inset: 0;
      background: radial-gradient(ellipse at 70% 40%, rgba(230,57,70,.18) 0%, transparent 60%);
      pointer-events: none;
    }
    .hero-badge {
      display: inline-block;
      background: rgba(230,57,70,.18);
      border: 1px solid rgba(230,57,70,.5);
      color: #ff8a94;
      font-size: .75rem;
      font-weight: 600;
      letter-spacing: .08em;
      text-transform: uppercase;
      padding: .3rem .9rem;
      border-radius: 999px;
      margin-bottom: 1.2rem;
    }
    .hero h1 {
      font-size: clamp(1.8rem, 5vw, 2.8rem);
      font-weight: 800;
      letter-spacing: -.02em;
      margin-bottom: .6rem;
    }
    .hero h1 span { color: var(--red); }
    .hero p.sub {
      font-size: 1.05rem;
      color: #a8c4df;
      max-width: 560px;
      margin: 0 auto 2rem;
    }
    .hero-stats {
      display: flex;
      justify-content: center;
      gap: 2.5rem;
      flex-wrap: wrap;
      margin-top: 1.5rem;
    }
    .stat { text-align: center; }
    .stat-num {
      display: block;
      font-size: 2rem;
      font-weight: 800;
      color: var(--white);
      line-height: 1;
    }
    .stat-num.red { color: var(--red); }
    .stat-label {
      font-size: .78rem;
      color: #a8c4df;
      text-transform: uppercase;
      letter-spacing: .06em;
      margin-top: .25rem;
    }

    /* ── SECTION WRAPPER ── */
    .section { padding: 2.8rem 1.5rem; max-width: 960px; margin: 0 auto; }
    .section-title {
      font-size: 1.25rem;
      font-weight: 700;
      margin-bottom: 1.2rem;
      color: var(--navy);
      display: flex;
      align-items: center;
      gap: .5rem;
    }
    .section-title::after {
      content: "";
      flex: 1;
      height: 1px;
      background: #d0dce8;
      margin-left: .5rem;
    }

    /* ── ALERT BANNER ── */
    .alert {
      background: #fff8e6;
      border-left: 4px solid var(--orange);
      border-radius: 6px;
      padding: .8rem 1.2rem;
      font-size: .88rem;
      color: #6b4c0f;
      max-width: 960px;
      margin: 1.2rem auto 0;
    }

    /* ── AGENT CARDS ── */
    .cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; }
    .card {
      background: var(--white);
      border-radius: 12px;
      padding: 1.4rem 1.2rem;
      box-shadow: 0 2px 8px rgba(0,0,0,.07);
      border-top: 4px solid transparent;
      transition: transform .15s, box-shadow .15s;
    }
    .card:hover { transform: translateY(-3px); box-shadow: 0 6px 18px rgba(0,0,0,.11); }
    .card.nurse   { border-color: var(--teal); }
    .card.lab     { border-color: #7c5cbf; }
    .card.pharma  { border-color: var(--orange); }
    .card.phys    { border-color: var(--red); }
    .card-icon { font-size: 1.8rem; margin-bottom: .5rem; }
    .card h3 { font-size: 1rem; font-weight: 700; margin-bottom: .3rem; }
    .card .sees { font-size: .82rem; color: var(--muted); }
    .card .sees strong { color: var(--navy); }

    /* ── TASK TABLE ── */
    .tbl-wrap { overflow-x: auto; }
    table { width: 100%; border-collapse: collapse; font-size: .9rem; background: var(--white); border-radius: 10px; overflow: hidden; box-shadow: 0 2px 8px rgba(0,0,0,.06); }
    thead { background: var(--navy); color: var(--white); }
    th { padding: .75rem 1rem; text-align: left; font-weight: 600; font-size: .82rem; text-transform: uppercase; letter-spacing: .05em; }
    td { padding: .7rem 1rem; border-bottom: 1px solid #e8eef4; }
    tr:last-child td { border-bottom: none; }
    tr:hover td { background: #f5f8fc; }
    .badge {
      display: inline-block;
      padding: .18rem .6rem;
      border-radius: 999px;
      font-size: .75rem;
      font-weight: 700;
      letter-spacing: .03em;
    }
    .badge.easy   { background: #d4f4e4; color: #1a6e3f; }
    .badge.medium { background: #fff0d4; color: #7a4a00; }
    .badge.hard   { background: #ffe0e3; color: #8a1020; }

    /* ── RESULTS ── */
    .results-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 1rem; }
    .result-card {
      background: var(--white);
      border-radius: 10px;
      padding: 1.2rem;
      box-shadow: 0 2px 8px rgba(0,0,0,.06);
      text-align: center;
    }
    .result-card .role { font-size: .78rem; text-transform: uppercase; letter-spacing: .07em; color: var(--muted); margin-bottom: .3rem; }
    .result-card .delta {
      font-size: 1.8rem;
      font-weight: 800;
      line-height: 1;
    }
    .delta.up   { color: var(--green); }
    .delta.down { color: var(--red); }
    .result-card .caption { font-size: .8rem; color: var(--muted); margin-top: .3rem; }

    /* ── API LINKS ── */
    .links { display: flex; flex-wrap: wrap; gap: .75rem; }
    .link-btn {
      display: inline-flex;
      align-items: center;
      gap: .4rem;
      padding: .55rem 1.1rem;
      border-radius: 8px;
      font-size: .88rem;
      font-weight: 600;
      text-decoration: none;
      transition: opacity .15s;
    }
    .link-btn:hover { opacity: .82; }
    .link-btn.primary  { background: var(--navy); color: var(--white); }
    .link-btn.secondary{ background: var(--white); color: var(--navy); border: 1.5px solid #c8d8e8; }
    .link-btn.teal     { background: var(--teal); color: var(--white); }
    .link-btn.red      { background: var(--red); color: var(--white); }

    /* ── QUICKSTART ── */
    .qs-box {
      background: var(--white);
      border-radius: 14px;
      box-shadow: 0 2px 12px rgba(0,0,0,.08);
      padding: 1.6rem 1.8rem;
      border: 1px solid #dce8f0;
      overflow: hidden;
    }
    .qs-step {
      display: flex;
      gap: 1.2rem;
      align-items: flex-start;
      min-width: 0;
    }
    .qs-step > div:last-child {
      min-width: 0;
      flex: 1;
    }
    .qs-num {
      flex-shrink: 0;
      width: 2rem; height: 2rem;
      background: var(--navy);
      color: var(--white);
      border-radius: 50%;
      font-size: .85rem;
      font-weight: 800;
      display: flex; align-items: center; justify-content: center;
      margin-top: .15rem;
    }
    .qs-label {
      font-weight: 700;
      font-size: 1rem;
      margin-bottom: .2rem;
      color: var(--navy);
    }
    .qs-desc {
      font-size: .85rem;
      color: var(--muted);
      margin-bottom: .65rem;
    }
    .qs-code {
      background: #0d1b2a;
      color: #c9e0f5;
      font-family: "SF Mono", "Fira Code", Consolas, monospace;
      font-size: .78rem;
      line-height: 1.65;
      padding: .9rem 1.1rem;
      border-radius: 8px;
      overflow-x: auto;
      white-space: pre;
      margin-bottom: .55rem;
      max-width: 100%;
      word-break: normal;
      overflow-wrap: normal;
    }
    .qs-code .kw  { color: #79c0ff; font-weight: 700; }
    .qs-code .cm  { color: #6e8ca0; }
    .qs-code .key { color: #a5d6ff; }
    .qs-code .str { color: #a8d8a8; }
    .qs-code .num { color: #f2cc60; }
    .qs-code code { background: none; color: inherit; font-size: inherit; }
    .qs-copy-row {
      display: flex;
      gap: .6rem;
      align-items: center;
      margin-bottom: .3rem;
    }
    .qs-copy {
      background: #e8f0f8;
      border: none;
      border-radius: 6px;
      padding: .28rem .75rem;
      font-size: .78rem;
      font-weight: 600;
      color: var(--navy);
      cursor: pointer;
      transition: background .15s;
    }
    .qs-copy:hover { background: #d0e4f4; }
    .qs-copy.copied { background: #d4f4e4; color: #1a6e3f; }
    .qs-try {
      font-size: .78rem;
      color: var(--teal);
      text-decoration: none;
      font-weight: 600;
    }
    .qs-try:hover { text-decoration: underline; }
    .qs-divider {
      height: 1px;
      background: #e8eef4;
      margin: 1.2rem 0;
    }
    .qs-tip {
      display: flex;
      gap: .7rem;
      align-items: flex-start;
      background: #f0f8ff;
      border-left: 3px solid var(--teal);
      border-radius: 6px;
      padding: .75rem 1rem;
      font-size: .84rem;
      color: #2a4a5e;
      margin-top: 1.2rem;
    }
    .qs-tip-icon { font-size: 1.1rem; flex-shrink: 0; }
    .qs-tip code {
      background: #d8edf8;
      border-radius: 3px;
      padding: .05rem .35rem;
      font-size: .78rem;
    }

    /* ── FOOTER ── */
    footer {
      background: var(--navy);
      color: #6a88a8;
      text-align: center;
      font-size: .8rem;
      padding: 1.4rem 1rem;
    }
    footer a { color: #a8c4df; text-decoration: none; }
    footer a:hover { text-decoration: underline; }
  </style>
</head>
<body>

<!-- HERO -->
<div class="hero">
  <div class="hero-badge">OpenEnv &mdash; Meta &times; HuggingFace Hackathon 2026</div>
  <h1>Sepsis<span>Guard</span></h1>
  <p class="sub">A multi-agent RL environment where four AI agents must coordinate to detect sepsis before it kills.</p>
  <div class="hero-stats">
    <div class="stat">
      <span class="stat-num red">11M</span>
      <span class="stat-label">Deaths / year</span>
    </div>
    <div class="stat">
      <span class="stat-num">20%</span>
      <span class="stat-label">Of all global deaths</span>
    </div>
    <div class="stat">
      <span class="stat-num red">+7%</span>
      <span class="stat-label">Mortality / hour delay</span>
    </div>
    <div class="stat">
      <span class="stat-num">4</span>
      <span class="stat-label">Coordinating agents</span>
    </div>
  </div>
</div>

<div style="max-width:960px;margin:0 auto;padding:0 1.5rem">
  <div class="alert">
    &#9888;&nbsp; <strong>Research prototype only.</strong> Not intended as medical advice or for clinical decision-making.
  </div>
</div>

<!-- AGENTS -->
<div class="section">
  <div class="section-title">The Four Agents</div>
  <p style="margin-bottom:1.2rem;color:#4a5f72;font-size:.95rem">Each agent holds a different, partial view of the same patients. No single agent can diagnose sepsis alone — they must learn to coordinate.</p>
  <div class="cards">
    <div class="card nurse">
      <div class="card-icon">&#129654;</div>
      <h3>Nurse</h3>
      <p class="sees"><strong>Sees:</strong> Vitals for 5 assigned patients<br><strong>Actions:</strong> Escalate, request labs, flag concern<br><strong>Challenge:</strong> Over-escalating burns physician trust</p>
    </div>
    <div class="card lab">
      <div class="card-icon">&#129514;</div>
      <h3>Lab Analyst</h3>
      <p class="sees"><strong>Sees:</strong> Lab values for all patients<br><strong>Actions:</strong> Flag critical results, recommend tests<br><strong>Challenge:</strong> No vitals context — labs alone are ambiguous</p>
    </div>
    <div class="card pharma">
      <div class="card-icon">&#128138;</div>
      <h3>Pharmacist</h3>
      <p class="sees"><strong>Sees:</strong> Medication lists, resistance rates<br><strong>Actions:</strong> Flag immunosuppression, recommend antibiotics<br><strong>Challenge:</strong> Immunosuppressants mask fever &amp; WBC signals</p>
    </div>
    <div class="card phys">
      <div class="card-icon">&#129658;</div>
      <h3>Physician</h3>
      <p class="sees"><strong>Sees:</strong> Only what others formally escalate<br><strong>Actions:</strong> Order antibiotics, admit to ICU<br><strong>Challenge:</strong> Must act on secondhand, incomplete information</p>
    </div>
  </div>
</div>

<!-- TASKS -->
<div class="section" style="padding-top:0">
  <div class="section-title">Task Difficulty Levels</div>
  <div class="tbl-wrap">
    <table>
      <thead>
        <tr>
          <th>Task</th><th>Difficulty</th><th>Patients</th><th>Sepsis Cases</th><th>False Alarms</th><th>Duration</th><th>Pass Threshold</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td><strong>task1_textbook</strong></td>
          <td><span class="badge easy">Easy</span></td>
          <td>5</td><td>1</td><td>0</td><td>24h (48 ticks)</td><td>0.70</td>
        </tr>
        <tr>
          <td><strong>task2_atypical</strong></td>
          <td><span class="badge medium">Medium</span></td>
          <td>10</td><td>3</td><td>0</td><td>48h (96 ticks)</td><td>0.55</td>
        </tr>
        <tr>
          <td><strong>task3_high_acuity</strong></td>
          <td><span class="badge hard">Hard</span></td>
          <td>10</td><td>4</td><td>2</td><td>72h (144 ticks)</td><td>0.40</td>
        </tr>
      </tbody>
    </table>
  </div>
</div>

<!-- RESULTS -->
<div class="section" style="padding-top:0">
  <div class="section-title">Training Results — Qwen2.5-3B-Instruct + TRL GRPO</div>
  <p style="margin-bottom:1.2rem;color:#4a5f72;font-size:.95rem">Trained on Task 1 for ~450 steps on a HuggingFace A10G GPU. Rewards connected to the live environment — not a static dataset.</p>
  <div class="results-grid">
    <div class="result-card">
      <div class="role">Nurse</div>
      <div class="delta up">+0.090</div>
      <div class="caption">0.753 &rarr; 0.843</div>
    </div>
    <div class="result-card">
      <div class="role">Lab Analyst</div>
      <div class="delta up">+0.250</div>
      <div class="caption">0.400 &rarr; 0.650</div>
    </div>
    <div class="result-card">
      <div class="role">Pharmacist</div>
      <div class="delta up">+0.093</div>
      <div class="caption">0.670 &rarr; 0.763</div>
    </div>
    <div class="result-card">
      <div class="role">Physician</div>
      <div class="delta down">&minus;0.305</div>
      <div class="caption">Needs more training — hardest role</div>
    </div>
  </div>
  <p style="margin-top:1rem;font-size:.84rem;color:var(--muted)">Heuristic baseline score: 0.8176 for all roles. Physician regression is expected — the role requires synthesizing incomplete multi-agent signals and needs longer training with a more varied task curriculum.</p>
</div>

<!-- LINKS -->
<div class="section" style="padding-top:0">
  <div class="section-title">Explore</div>
  <div class="links">
    <a class="link-btn primary" href="/docs">&#128196; API Docs (Swagger)</a>
    <a class="link-btn teal"    href="/dashboard">&#128200; Live Dashboard</a>
    <a class="link-btn secondary" href="/tasks">&#128203; Task Configs</a>
    <a class="link-btn secondary" href="/baseline">&#9654; Run Baseline</a>
    <a class="link-btn secondary" href="/health">&#10003; Health</a>
    <a class="link-btn red" href="https://github.com/JishnuVijayan/Sepsis-Guard" target="_blank">&#128279; GitHub</a>
  </div>
</div>

<!-- QUICKSTART -->
<div class="section" style="padding-top:0">
  <div class="section-title">Quick Start</div>
  <div class="qs-box">

    <div class="qs-step">
      <div class="qs-num">1</div>
      <div>
        <div class="qs-label">Start a new episode</div>
        <div class="qs-desc">Reset the environment with a task and seed. Returns initial observations for all 4 agents.</div>
        <pre class="qs-code"><span class="kw">POST</span> https://jishnu-vijayan-03-sepsis-guard.hf.space/reset
<span class="cm">Content-Type: application/json</span>

{
  <span class="key">"task_name"</span>: <span class="str">"task1_textbook"</span>,
  <span class="key">"seed"</span>: <span class="num">42</span>
}</pre>
        <div class="qs-copy-row">
          <button class="qs-copy" onclick="copyCode(this, 'POST https://jishnu-vijayan-03-sepsis-guard.hf.space/reset\nContent-Type: application/json\n\n{\"task_name\": \"task1_textbook\", \"seed\": 42}')">Copy</button>
          <a class="qs-try" href="/docs#/default/reset_reset_post" target="_blank">Try in Swagger ↗</a>
        </div>
      </div>
    </div>

    <div class="qs-divider"></div>

    <div class="qs-step">
      <div class="qs-num">2</div>
      <div>
        <div class="qs-label">Submit agent actions</div>
        <div class="qs-desc">Send one action per agent each tick. The environment resolves them, advances patient physiology, and returns updated observations + rewards.</div>
        <pre class="qs-code"><span class="kw">POST</span> https://jishnu-vijayan-03-sepsis-guard.hf.space/step
<span class="cm">Content-Type: application/json</span>

{
  <span class="key">"actions"</span>: {
    <span class="key">"nurse"</span>: {
      <span class="key">"operation"</span>: <span class="str">"escalate_to_physician"</span>,
      <span class="key">"patient_id"</span>: <span class="str">"P1"</span>,
      <span class="key">"urgency"</span>: <span class="str">"urgent"</span>,
      <span class="key">"rationale"</span>: <span class="str">"HR 118, BP 88"</span>
    },
    <span class="key">"lab"</span>: {
      <span class="key">"operation"</span>: <span class="str">"flag_critical"</span>,
      <span class="key">"patient_id"</span>: <span class="str">"P1"</span>,
      <span class="key">"test"</span>: <span class="str">"lactate"</span>,
      <span class="key">"reason"</span>: <span class="str">"lactate 3.8"</span>
    },
    <span class="key">"pharmacist"</span>: {
      <span class="key">"operation"</span>: <span class="str">"recommend_antibiotic"</span>,
      <span class="key">"patient_id"</span>: <span class="str">"P1"</span>,
      <span class="key">"drug"</span>: <span class="str">"piperacillin_tazobactam"</span>,
      <span class="key">"rationale"</span>: <span class="str">"sepsis suspected"</span>
    },
    <span class="key">"physician"</span>: {
      <span class="key">"operation"</span>: <span class="str">"order_antibiotics"</span>,
      <span class="key">"patient_id"</span>: <span class="str">"P1"</span>,
      <span class="key">"drug"</span>: <span class="str">"piperacillin_tazobactam"</span>
    }
  }
}</pre>
        <div class="qs-copy-row">
          <button class="qs-copy" onclick="copyCode(this, 'POST https://jishnu-vijayan-03-sepsis-guard.hf.space/step')">Copy URL</button>
          <a class="qs-try" href="/docs#/default/step_step_post" target="_blank">Try in Swagger ↗</a>
        </div>
      </div>
    </div>

    <div class="qs-divider"></div>

    <div class="qs-step">
      <div class="qs-num">3</div>
      <div>
        <div class="qs-label">Get the episode score</div>
        <div class="qs-desc">After the episode ends (<code>done: true</code>), fetch the grader for team score and per-role metrics.</div>
        <pre class="qs-code"><span class="kw">GET</span>  https://jishnu-vijayan-03-sepsis-guard.hf.space/grader</pre>
        <div class="qs-copy-row">
          <button class="qs-copy" onclick="copyCode(this, 'GET https://jishnu-vijayan-03-sepsis-guard.hf.space/grader')">Copy</button>
          <a class="qs-try" href="/grader" target="_blank">Open ↗</a>
        </div>
      </div>
    </div>

    <div class="qs-divider"></div>

    <div class="qs-step">
      <div class="qs-num">4</div>
      <div>
        <div class="qs-label">Run heuristic baseline</div>
        <div class="qs-desc">Runs rule-based agents across all three tasks. Use this score as the bar your trained model needs to beat.</div>
        <pre class="qs-code"><span class="kw">GET</span>  https://jishnu-vijayan-03-sepsis-guard.hf.space/baseline</pre>
        <div class="qs-copy-row">
          <button class="qs-copy" onclick="copyCode(this, 'GET https://jishnu-vijayan-03-sepsis-guard.hf.space/baseline')">Copy</button>
          <a class="qs-try" href="/baseline" target="_blank">Run now ↗</a>
        </div>
      </div>
    </div>

    <div class="qs-tip">
      <span class="qs-tip-icon">&#128161;</span>
      <span>Each tick = 30 simulated minutes. Repeat <strong>Step 2</strong> until <code>done: true</code>. For multi-session parallel training, create an isolated session first via <code>POST /session</code> and pass the returned <code>session_id</code> as the <code>X-Session-Id</code> header.</span>
    </div>

  </div>
</div>

<!-- FOOTER -->
<footer>
  <p>SepsisGuard &mdash; <a href="https://www.who.int/news-room/fact-sheets/detail/sepsis" target="_blank">11M deaths/year from sepsis (WHO)</a> &mdash; Built at Meta &times; HuggingFace OpenEnv Hackathon, Bangalore, April 2026</p>
  <p style="margin-top:.4rem">Not medical advice. Research prototype only.</p>
</footer>

<script>
function copyCode(btn, text) {
  navigator.clipboard.writeText(text).then(function() {
    btn.textContent = 'Copied!';
    btn.classList.add('copied');
    setTimeout(function() {
      btn.textContent = 'Copy';
      btn.classList.remove('copied');
    }, 1800);
  });
}
</script>
</body>
</html>"""


try:
    from server.dashboard import build_dashboard
    import gradio as gr
    demo = build_dashboard(_default_env)
    app = gr.mount_gradio_app(app, demo, path="/dashboard")
except Exception as e:
    print(f"[WARN] Dashboard not mounted: {e}")


def main():
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
