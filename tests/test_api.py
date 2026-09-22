"""HTTP API tests: parameter rejection, response consistency, case archive,
the seeded demo case, and isolation of concurrently submitted cases."""

import math
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.cases import DEMO_CASE_NAME
from app.main import app

client = TestClient(app)


# --- invalid input rejection -------------------------------------------------

@pytest.mark.parametrize(
    "payload,parameter",
    [
        ({"omega_p": 0.0, "gamma": 3.3, "alpha": 0.01}, "omega_p"),
        ({"omega_p": -0.5, "gamma": 3.3, "alpha": 0.01}, "omega_p"),
        ({"omega_p": 0.5, "gamma": 0.5, "alpha": 0.01}, "gamma"),
        ({"omega_p": 0.5, "gamma": 3.3, "alpha": 0.01, "wind_speed": -10.0},
         "wind_speed"),
        ({"omega_p": 0.5, "gamma": 3.3, "alpha": 0.01, "wind_speed": 0.0},
         "wind_speed"),
        ({"omega_p": 0.5, "gamma": 3.3, "hs_target": 2.0, "fetch": -1.0}, "fetch"),
        ({"omega_p": 0.5, "gamma": 3.3, "alpha": -0.01}, "alpha"),
        ({"omega_p": 0.5, "gamma": 3.3, "hs_target": 0.0}, "hs_target"),
        ({"omega_p": 0.5, "gamma": 3.3}, "alpha/hs_target"),
        ({"omega_p": 0.5, "gamma": 3.3, "alpha": 0.01, "hs_target": 2.0},
         "alpha/hs_target"),
    ],
)
def test_invalid_parameters_rejected_with_reason(payload, parameter):
    resp = client.post("/spectrum", json=payload)
    assert resp.status_code == 400
    body = resp.json()
    assert body["error"] == "invalid_parameter"
    assert body["parameter"] == parameter
    assert body["reason"]

    resp = client.post("/moments", json=payload)
    assert resp.status_code == 400
    assert resp.json()["error"] == "invalid_parameter"


# --- forward computation -----------------------------------------------------

def test_spectrum_endpoint_returns_consistent_result():
    resp = client.post(
        "/spectrum",
        json={"omega_p": 0.5, "gamma": 3.3, "alpha": 0.01, "wind_speed": 12.0},
    )
    assert resp.status_code == 200
    body = resp.json()

    assert len(body["omega"]) == body["n_points"] == len(body["s"])
    assert body["m0"] > 0 and body["m1"] > 0 and body["m2"] > 0
    assert body["hs"] == pytest.approx(4.0 * math.sqrt(body["m0"]), rel=1e-12)
    assert body["tp"] == pytest.approx(2.0 * math.pi / 0.5, rel=1e-12)
    assert body["inversion"] is None

    # Peak of the returned curve sits at omega_p.
    omega = np.array(body["omega"])
    s = np.array(body["s"])
    step = omega[1] - omega[0]
    assert abs(omega[np.argmax(s)] - 0.5) <= step


def test_moments_endpoint_matches_spectrum_endpoint():
    payload = {"omega_p": 0.6, "gamma": 2.0, "hs_target": 3.0}
    full = client.post("/spectrum", json=payload).json()
    mom = client.post("/moments", json=payload).json()

    assert mom["m0"] == pytest.approx(full["m0"])
    assert mom["hs"] == pytest.approx(full["hs"])
    assert mom["tz"] == pytest.approx(full["tz"])
    assert mom["inversion"]["converged"] is True
    assert mom["hs"] == pytest.approx(3.0, rel=1e-9)


# --- case archive ------------------------------------------------------------

def test_case_roundtrip_and_recompute():
    case = {
        "name": "pytest_case",
        "description": "roundtrip",
        "params": {"omega_p": 0.7, "gamma": 2.0, "hs_target": 2.5},
    }
    assert client.post("/cases", json=case).status_code == 201
    assert "pytest_case" in client.get("/cases").json()["cases"]

    stored = client.get("/cases/pytest_case").json()
    assert stored["params"]["gamma"] == 2.0

    recomputed = client.get("/cases/pytest_case/spectrum").json()
    assert recomputed["hs"] == pytest.approx(2.5, rel=1e-9)

    assert client.delete("/cases/pytest_case").status_code == 204
    assert client.get("/cases/pytest_case").status_code == 404


def test_invalid_case_is_not_archived():
    bad = {"name": "bad_case", "params": {"omega_p": 0.0, "gamma": 3.3,
                                          "alpha": 0.01}}
    assert client.post("/cases", json=bad).status_code == 400
    assert client.get("/cases/bad_case").status_code == 404


def test_seeded_demo_case_is_self_consistent():
    assert DEMO_CASE_NAME in client.get("/cases").json()["cases"]

    body = client.get(f"/cases/{DEMO_CASE_NAME}/spectrum").json()
    assert body["gamma"] > 1.0

    omega = np.array(body["omega"])
    s = np.array(body["s"])
    step = omega[1] - omega[0]
    assert abs(omega[np.argmax(s)] - body["omega_p"]) <= step
    assert body["hs"] == pytest.approx(4.0 * math.sqrt(body["m0"]), rel=1e-12)
    assert body["inversion"]["converged"] is True


# --- concurrency --------------------------------------------------------------

def test_parallel_cases_do_not_cross_contaminate():
    targets = [1.5, 2.0, 3.0, 4.5, 5.5, 2.5, 3.5, 6.0]

    def run(hs_target):
        resp = client.post(
            "/spectrum",
            json={"omega_p": 0.5, "gamma": 3.3, "hs_target": hs_target},
        )
        assert resp.status_code == 200
        return hs_target, resp.json()

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(run, targets))

    for hs_target, body in results:
        # Each response closes to its own constraint, not a neighbour's.
        assert body["hs"] == pytest.approx(hs_target, rel=1e-9)
        assert body["m0"] == pytest.approx((hs_target / 4.0) ** 2, rel=1e-6)

    # Distinct constraints must yield distinct alphas.
    alphas = [body["alpha"] for _, body in results]
    assert len(set(alphas)) == len(set(targets))
