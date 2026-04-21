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
