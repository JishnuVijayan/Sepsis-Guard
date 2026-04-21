---
title: SepsisGuard
emoji: "\U0001F3E5"
colorFrom: red
colorTo: blue
sdk: docker
pinned: false
tags:
  - openenv
  - multi-agent
  - healthcare
---

# SepsisGuard

**A Multi-Agent Clinical Coordination Environment for OpenEnv**

> 11 million people die from sepsis every year. The treatment is simple: fluids and antibiotics every hospital stocks. The diagnosis is not simple — it requires four different people, each holding one piece of evidence, to coordinate in under three hours. SepsisGuard is the first RL environment that trains that coordination.

## The Problem

Sepsis kills more people than all cancers combined, yet ~80% of deaths are preventable with timely treatment. The failure isn't knowledge — it's **coordination**. Four hospital roles each hold partial information, and by the time it reaches the physician, hours have passed.

## How It Works

SepsisGuard simulates a hospital ward with 5-10 patients over 24-72 simulated hours (30-minute ticks). Four AI agents with **asymmetric information** must coordinate to detect and treat sepsis:

| Agent | Sees | Cannot See |
|---|---|---|
| **Nurse** | Vitals for 5 assigned patients, bedside behavior | Lab results, medication context |
| **Lab Analyst** | Lab values for all 10 patients | Vitals, medications |
| **Pharmacist** | Medication lists, antibiogram, drug interactions | Vitals, lab results |
| **Physician** | Only what is formally escalated by others | Real-time vitals, routine labs |

No single agent has enough information to diagnose sepsis alone. They must learn to communicate the right information at the right time.

## Key Mechanics

- **Alarm Fatigue**: Over-escalation degrades physician trust — future valid escalations get delayed or ignored
- **Asymmetric Observability**: Each agent sees a structurally different view of the same patients
- **Time Pressure**: Each hour of delayed antibiotics increases mortality by ~7%
- **False Alarms**: Some patients mimic sepsis signs without infection — agents must learn precision
- **Immunosuppression Masking**: Some patients on immunosuppressants have blunted fever/WBC responses

## Three Difficulty Levels

| Task | Patients | Sepsis Cases | False Alarms | Duration | Threshold |
|---|---|---|---|---|---|
| **Task 1 — Textbook** | 5 | 1 | 0 | 24h (48 ticks) | 0.70 |
| **Task 2 — Atypical** | 10 | 3 | 0 | 48h (96 ticks) | 0.55 |
| **Task 3 — High Acuity** | 10 | 4 | 2 | 72h (144 ticks) | 0.40 |

## Scoring

```
team_score = 0.40 * (treated_in_time / sepsis_patients)
           + 0.25 * (1.0 - false_alarm_rate)
           + 0.20 * coordination_score
           + 0.15 * time_efficiency
```

## API Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Liveness check |
| POST | `/reset` | Start new episode: `{"seed": 42, "task_name": "task1_textbook"}` |
| POST | `/step` | Submit 4 agent actions, get observations + rewards |
| GET | `/state` | Full environment state (debug) |
| GET | `/tasks` | Task list + action schemas |
| GET | `/grader` | Team score + metrics after episode |
| GET | `/baseline` | Run heuristic agents on all tasks |
| GET | `/dashboard` | Interactive Gradio UI |

## Quick Start

```bash
# Install
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Start server
uvicorn server.app:app --port 7860

# Or with Docker
docker build -t sepsisguard .
docker run -p 7860:7860 sepsisguard
```

## Training

Training uses TRL GRPO with a single model serving all 4 roles via role-conditioned prompts. See `training/colab_training.ipynb` for the full pipeline.

```
# Rollout collection → Reward scoring → GRPO update
# 4 agents x 200 steps x 8 completions = 6,400 LLM calls per run
# Target model: Qwen2.5-3B-Instruct (4-bit) on Colab T4
```

## Project Structure

```
sepsisguard/
├── models.py              # Pydantic models (patients, observations, actions)
├── inference.py           # LLM inference script with heuristic fallback
├── server/
│   ├── app.py             # FastAPI server (all endpoints)
│   ├── environment.py     # SepsisEnvironment (reset/step/state)
│   ├── physiology.py      # Patient deterioration/recovery simulation
│   ├── observations.py    # Asymmetric info filters per agent
│   ├── rewards.py         # 3-layer reward (per-agent + team + terminal)
│   ├── resolution.py      # Action precedence and resolution
│   ├── config.py          # Enums, task configs, constants
│   └── dashboard.py       # Gradio interactive UI
├── agents/
│   ├── nurse.py           # Heuristic baseline nurse
│   ├── lab.py             # Heuristic baseline lab analyst
│   ├── pharmacist.py      # Heuristic baseline pharmacist
│   └── physician.py       # Heuristic baseline physician
├── training/
│   ├── colab_training.ipynb   # GRPO training notebook
│   ├── rollout_collector.py   # Environment rollout collection
│   ├── prompts.py             # Role-conditioned prompt templates
│   └── reward_shaping.py      # TRL-compatible reward function
└── tests/                 # 26 tests across all modules
```

## Themes

- **Primary**: Multi-Agent Interactions — four agents with structurally different information must coordinate
- **Secondary**: World Modeling (Professional Tasks) — each agent maintains an internal model of patients that updates as asynchronous information arrives

## Built For

Meta x Hugging Face OpenEnv Hackathon, Bangalore (April 2026)
