import pytest
from fastapi.testclient import TestClient
from server.app import app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "healthy"


def test_reset_and_step():
    r = client.post("/reset", json={"seed": 42, "task_name": "task1_textbook"})
    assert r.status_code == 200
    assert "observations" in r.json()

    r = client.post("/step", json={"actions": {
        "nurse": {"operation": "noop"},
        "lab": {"operation": "noop"},
        "pharmacist": {"operation": "noop"},
        "physician": {"operation": "do_nothing"},
    }})
    assert r.status_code == 200
    assert "rewards" in r.json()


def test_tasks_endpoint():
    r = client.get("/tasks")
    assert r.status_code == 200
    tasks = r.json()["tasks"]
    names = [t["task_name"] for t in tasks]
    assert "task1_textbook" in names
    assert "task2_atypical" in names
    assert "task3_high_acuity" in names


def test_grader_before_episode():
    r = client.get("/grader")
    assert r.status_code == 200


def test_state_endpoint():
    client.post("/reset", json={"seed": 42, "task_name": "task1_textbook"})
    r = client.get("/state")
    assert r.status_code == 200
    data = r.json()
    assert "tick" in data
    assert "patients" in data


def test_observations_endpoint():
    client.post("/reset", json={"seed": 42, "task_name": "task1_textbook"})
    r = client.post("/observations", json={"agent_role": "nurse"})
    assert r.status_code == 200
    assert "patient_vitals" in r.json()


def test_observations_invalid_role():
    client.post("/reset", json={"seed": 42, "task_name": "task1_textbook"})
    r = client.post("/observations", json={"agent_role": "hacker"})
    assert r.status_code == 400


def test_full_episode_with_grader():
    client.post("/reset", json={"seed": 42, "task_name": "task1_textbook"})
    noop = {"actions": {
        "nurse": {"operation": "noop"},
        "lab": {"operation": "noop"},
        "pharmacist": {"operation": "noop"},
        "physician": {"operation": "do_nothing"},
    }}
    for _ in range(60):
        r = client.post("/step", json=noop)
        if r.json()["done"]:
            break
    grader = client.get("/grader").json()
    assert grader["status"] == "ok"
    assert "score" in grader
    assert "metrics" in grader


def test_step_with_invalid_action_structure():
    client.post("/reset", json={"seed": 42, "task_name": "task1_textbook"})
    r = client.post("/step", json={"actions": {"nurse": {"operation": "noop"}}})
    assert r.status_code in (400, 422)
