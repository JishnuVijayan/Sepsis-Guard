import numpy as np
import pytest
from server.physiology import (
    generate_patients, advance_physiology, mature_pending_labs, order_lab_test,
)
from server.config import Outcome


def test_determinism_generate_patients():
    rng_a = np.random.default_rng(42)
    rng_b = np.random.default_rng(42)
    pa = generate_patients(rng_a, 10, 3, 2, 96)
    pb = generate_patients(rng_b, 10, 3, 2, 96)
    for a, b in zip(pa, pb):
        assert a.patient_id == b.patient_id
        assert a.age == b.age
        assert a.infection_present == b.infection_present
        assert a.sepsis_onset_tick == b.sepsis_onset_tick


def test_sepsis_progression():
    rng = np.random.default_rng(0)
    patients = generate_patients(rng, 5, 1, 0, 96)
    septic = next(p for p in patients if p.infection_present)
    septic.infection_start_tick = 2
    septic.sepsis_onset_tick = 2
    for tick in range(1, 50):
        advance_physiology(septic, tick, rng)
        if septic.outcome == Outcome.DIED:
            break
    assert septic.infection_severity > 0.5, \
        f"Untreated sepsis did not progress: severity={septic.infection_severity}"


def test_antibiotics_stop_progression():
    rng = np.random.default_rng(0)
    patients = generate_patients(rng, 5, 1, 0, 96)
    septic = next(p for p in patients if p.infection_present)
    septic.infection_start_tick = 2
    septic.sepsis_onset_tick = 2
    for tick in range(1, 10):
        advance_physiology(septic, tick, rng)
    pre_sev = septic.infection_severity
    septic.antibiotics_administered = "piperacillin_tazobactam"
    septic.antibiotic_tick = 10
    for tick in range(10, 30):
        advance_physiology(septic, tick, rng)
    assert septic.infection_severity < pre_sev, "Antibiotics did not reduce severity"


def test_lab_maturation_delay():
    rng = np.random.default_rng(0)
    patients = generate_patients(rng, 5, 1, 0, 96)
    p = patients[0]
    assert order_lab_test(p, "lactate", tick=5, delay=2)
    ready = mature_pending_labs(p, tick=5, rng=rng)
    assert ready == []
    ready = mature_pending_labs(p, tick=7, rng=rng)
    assert "lactate" in ready
    assert p.lactate is not None


def test_immunosuppressed_has_blunted_wbc():
    rng = np.random.default_rng(1)
    patients = generate_patients(rng, 20, 10, 0, 96)
    septic_immuno = [p for p in patients if p.infection_present and p.immunocompromised]
    septic_normal = [p for p in patients if p.infection_present and not p.immunocompromised]
    if not septic_immuno or not septic_normal:
        pytest.skip("No matched pair in this seed")
    for p in septic_immuno + septic_normal:
        p.infection_severity = 0.8
        order_lab_test(p, "wbc", tick=1, delay=0)
        mature_pending_labs(p, tick=1, rng=rng)
    avg_immuno = np.mean([p.wbc for p in septic_immuno])
    avg_normal = np.mean([p.wbc for p in septic_normal])
    assert avg_immuno < avg_normal, \
        f"Immunosuppressed WBC ({avg_immuno}) should be lower than normal ({avg_normal})"
