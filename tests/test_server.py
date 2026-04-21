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
