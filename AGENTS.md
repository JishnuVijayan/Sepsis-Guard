# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## Project Overview

SepsisGuard is a multi-agent clinical coordination environment built for the OpenEnv framework (Meta x Hugging Face Hackathon). It simulates a hospital ward where 4 AI agents (Nurse, Lab Analyst, Pharmacist, Physician) must coordinate sepsis diagnosis and treatment under time pressure with **asymmetric information** — no single agent sees the full patient picture.

## Commands

```bash
# Install (editable with dev deps)
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Run a single test file
pytest tests/test_rewards.py -v

# Start the FastAPI server
uvicorn server.app:app --port 7860

# Docker build & run
docker build -t sepsisguard .
docker run -p 7860:7860 sepsisguard
```

## Environment Variables

Copy `.env.example` to `.env`. Required variables:
- `API_BASE_URL` — LLM endpoint (default: HuggingFace router)
- `MODEL_NAME` — model ID (default: `meta-llama/Llama-3.1-8B-Instruct`)
- `HF_TOKEN` — HuggingFace API token
- `ENV_BASE_URL` — environment server URL (default: `http://localhost:7860`)

Inference (`inference.py`) tries `OPENAI_API_KEY`, then `HF_TOKEN`, then `API_KEY`, falling back to heuristic agents if none are set.

## Architecture

### Core Loop

The environment runs in **ticks** (30 simulated minutes each). Each tick: agents receive role-filtered observations → submit actions → environment resolves actions, advances physiology, computes rewards.

### Key Modules

- **`models.py`** — All Pydantic models: `PatientState`, per-role observations (`NurseObservation`, `LabObservation`, `PharmacistObservation`, `PhysicianObservation`), per-role actions, `StepRequest`, `SepsisState`
- **`server/app.py`** — FastAPI endpoints: `/reset`, `/step`, `/state`, `/observations`, `/tasks`, `/grader`, `/baseline`, `/dashboard`
- **`server/environment.py`** — `SepsisEnvironment` class managing episode lifecycle, agent tracking, physician trust
- **`server/physiology.py`** — Patient deterioration/recovery simulation, lab result delays, patient generation
- **`server/observations.py`** — Asymmetric information filtering per role
- **`server/rewards.py`** — 3-layer reward: per-agent, team delta, and terminal score (40% treatment timeliness, 25% false alarm avoidance, 20% coordination, 15% efficiency)
- **`server/resolution.py`** — Action precedence and resolution logic
- **`server/config.py`** — Task configs, vital/lab normal ranges, antibiogram (drug resistance rates)
- **`inference.py`** — LLM inference orchestration with heuristic fallback

### Agent System

- **`agents/`** — Heuristic baseline agents (`HeuristicNurse`, `HeuristicLab`, `HeuristicPharmacist`, `HeuristicPhysician`) inheriting from `RoleAgent` base class
- Agents communicate via `AgentFlag` objects (escalations, lab flags, immunosuppression warnings)
- The Physician only sees patients that other agents have escalated/flagged

### Training Pipeline

- **`training/`** — RL training using TRL GRPO (Grouped Relative Policy Optimization)
- `training/prompts.py` — Role-conditioned system prompts for LLM agents
- `training/reward_shaping.py` — `OnlineSepsisReward` evaluates completions against live environment
- `training/colab_training.ipynb` — Notebook for Colab/HF Spaces training with Unsloth 4-bit quantization
- Training is excluded from the Docker container; it runs in Colab/notebooks

### Three Task Difficulties

| Task | Patients | Sepsis | False Alarms | Ticks | Threshold |
|------|----------|--------|--------------|-------|-----------|
| `task1_textbook` | 5 | 1 | 0 | 48 | 0.70 |
| `task2_atypical` | 10 | 3 | 0 | 96 | 0.55 |
| `task3_high_acuity` | 10 | 4 | 2 | 144 | 0.40 |

## Key Design Constraints

- **Asymmetric information**: Nurse sees vitals only, Lab sees labs only, Pharmacist sees medications/resistance, Physician sees only escalated patients. Actions and observations are structurally different per role.
- **Alarm fatigue**: Physician trust degrades with false escalations, affecting how they respond.
- **Lab delays**: Lab results take 1-3 ticks to return depending on task difficulty.
- **Clinical realism**: Immunosuppression masks inflammatory response; every hour of antibiotic delay increases mortality ~7%.
