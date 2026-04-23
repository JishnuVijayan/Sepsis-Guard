import json
from server.environment import SepsisEnvironment


def test_reset_and_step_once():
    env = SepsisEnvironment()
    bundle = env.reset(seed=42, task_name="task1_textbook")
    assert "observations" in bundle
    assert set(bundle["observations"]) == {"nurse", "lab", "pharmacist", "physician"}
    action = {
        "actions": {
            "nurse": {"operation": "noop"},
            "lab": {"operation": "noop"},
            "pharmacist": {"operation": "noop"},
            "physician": {"operation": "do_nothing"},
        }
    }
    result = env.step(action)
    assert "rewards" in result
    assert "done" in result


def test_full_episode_runs():
    env = SepsisEnvironment()
    env.reset(seed=42, task_name="task1_textbook")
    action = {
        "actions": {
            "nurse": {"operation": "noop"},
            "lab": {"operation": "noop"},
            "pharmacist": {"operation": "noop"},
            "physician": {"operation": "do_nothing"},
        }
    }
    for _ in range(60):
        result = env.step(action)
        if result["done"]:
            break
    assert result["done"] is True


def test_deterministic_episodes():
    """Same seed produces identical episode traces."""
    def run_episode(seed):
        env = SepsisEnvironment()
        env.reset(seed=seed, task_name="task1_textbook")
        noop = {
            "actions": {
                "nurse": {"operation": "noop"},
                "lab": {"operation": "noop"},
                "pharmacist": {"operation": "noop"},
                "physician": {"operation": "do_nothing"},
            }
        }
        rewards = []
        for _ in range(50):
            r = env.step(noop)
            rewards.append(r["rewards"])
            if r["done"]:
                break
        return rewards

    run1 = run_episode(seed=99)
    run2 = run_episode(seed=99)
    assert len(run1) == len(run2)
    for r1, r2 in zip(run1, run2):
        assert r1 == r2


def test_coordination_events_tracked():
    """Multi-source flags on same patient increment coordination counter."""
    env = SepsisEnvironment()
    bundle = env.reset(seed=42, task_name="task1_textbook")
    obs = bundle["observations"]
    patients = obs["nurse"]["patient_vitals"]
    pid = patients[0]["patient_id"]
    action = {
        "actions": {
            "nurse": {"operation": "escalate_to_physician", "patient_id": pid,
                       "urgency": "critical", "rationale": "test"},
            "lab": {"operation": "flag_critical", "patient_id": pid, "reason": "test"},
            "pharmacist": {"operation": "noop"},
            "physician": {"operation": "do_nothing"},
        }
    }
    env.step(action)
    assert env._coord_events["total"] >= 1


def test_reset_clears_state():
    """Reset produces clean state regardless of prior episode state."""
    env = SepsisEnvironment()
    env.reset(seed=42, task_name="task1_textbook")
    noop = {
        "actions": {
            "nurse": {"operation": "noop"},
            "lab": {"operation": "noop"},
            "pharmacist": {"operation": "noop"},
            "physician": {"operation": "do_nothing"},
        }
    }
    for _ in range(10):
        env.step(noop)
    env.reset(seed=42, task_name="task1_textbook")
    assert env._tick == 1
    assert env._done is False
    assert env._physician_trust == 1.0
    assert env._total_escalations == 0
    assert all(v == 0.0 for v in env._cumulative_rewards.values())


def test_grader_data_populated_after_episode():
    env = SepsisEnvironment()
    env.reset(seed=42, task_name="task1_textbook")
    noop = {
        "actions": {
            "nurse": {"operation": "noop"},
            "lab": {"operation": "noop"},
            "pharmacist": {"operation": "noop"},
            "physician": {"operation": "do_nothing"},
        }
    }
    for _ in range(60):
        r = env.step(noop)
        if r["done"]:
            break
    data = env.last_grader_data
    assert "score" in data
    assert "metrics" in data
    assert data["metrics"]["success_threshold"] == 0.70


def test_observations_serializable():
    """All observations must be JSON-serializable for the API."""
    env = SepsisEnvironment()
    bundle = env.reset(seed=42, task_name="task1_textbook")
    json_str = json.dumps(bundle, default=str)
    assert len(json_str) > 100
