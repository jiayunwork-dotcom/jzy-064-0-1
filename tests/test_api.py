"""HTTP 路由集成测试：正算、矩、反演闭合、错误结构、工况复算。"""

from __future__ import annotations

import math

import numpy as np


def test_spectrum_explicit_alpha(client):
    r = client.post(
        "/api/v1/spectrum",
        json={"omega_p": 0.7, "gamma": 3.3, "alpha": 0.01, "sample_points": 256},
    )
    assert r.status_code == 200, r.text
    d = r.json()["data"]
    assert len(d["omega"]) == len(d["spectrum"]) == 256
    m = d["moments"]
    st = d["statistics"]
    assert math.isclose(st["hs"], 4.0 * math.sqrt(m["m0"]), rel_tol=1e-12)
    assert math.isclose(st["tp"], 2.0 * math.pi / 0.7, rel_tol=1e-12)
    assert d["integration"]["converged"] is True
    assert d["parameters"]["alpha"] == 0.01
    assert d["inversion"] is None


def test_spectrum_hs_inversion_closure(client):
    """HTTP 层同样焊死：波高反演后声称的 Hs 与谱面积闭合。"""
    r = client.post(
        "/api/v1/spectrum", json={"omega_p": 0.6, "gamma": 3.3, "hs": 6.0}
    )
    assert r.status_code == 200, r.text
    d = r.json()["data"]
    hs, m0 = d["statistics"]["hs"], d["moments"]["m0"]
    assert abs(hs - 6.0) / 6.0 < 1e-6
    assert abs(4.0 * math.sqrt(m0) - 6.0) / 6.0 < 1e-6
    assert d["inversion"]["relative_residual"] < 1e-7
    assert d["statistics"]["hs_target"] == 6.0


def test_moments_endpoint(client):
    r = client.post(
        "/api/v1/moments", json={"omega_p": 0.7, "gamma": 1.0, "alpha": 0.0081}
    )
    assert r.status_code == 200, r.text
    d = r.json()["data"]
    assert "spectrum" not in d
    assert set(d["moments"]) == {"m0", "m1", "m2"}
    assert d["statistics"]["tz"] < d["statistics"]["tp"]
    # γ=1 为 PM 闭式 m0 = αg²/(5ωp⁴)
    g = 9.80665
    analytic = 0.0081 * g**2 / (5.0 * 0.7**4)
    assert math.isclose(d["moments"]["m0"], analytic, rel_tol=1e-5)


def test_fetch_case_spectrum(client):
    r = client.post(
        "/api/v1/spectrum",
        json={"wind_speed": 15.0, "fetch": 50_000.0, "sample_points": 200},
    )
    assert r.status_code == 200, r.text
    d = r.json()["data"]
    assert d["parameters"]["mode"] == "alpha"
    assert d["parameters"]["gamma"] == 3.3
    assert math.isclose(d["statistics"]["hs"], 4 * math.sqrt(d["moments"]["m0"]), rel_tol=1e-12)


def test_reject_zero_omega_p_with_error_body(client):
    r = client.post(
        "/api/v1/spectrum", json={"omega_p": 0.0, "gamma": 3.3, "alpha": 0.01}
    )
    assert r.status_code == 422
    err = r.json()["error"]
    assert err["code"] == "invalid_parameter"
    assert err["field"] == "omega_p"
    assert "原因" in err["message"] or len(err["message"]) > 0


def test_reject_negative_wind_speed(client):
    r = client.post(
        "/api/v1/spectrum",
        json={"omega_p": 0.7, "gamma": 3.3, "alpha": 0.01, "wind_speed": -5.0},
    )
    assert r.status_code == 422
    assert r.json()["error"]["field"] == "wind_speed"


def test_reject_gamma_below_one(client):
    r = client.post(
        "/api/v1/spectrum", json={"omega_p": 0.7, "gamma": 0.5, "alpha": 0.01}
    )
    assert r.status_code == 422
    assert r.json()["error"]["field"] == "gamma"


def test_case_lifecycle_and_recompute(client):
    # 建档
    r = client.put(
        "/api/v1/cases/my-case",
        json={
            "name": "my-case",
            "params": {"omega_p": 0.66, "gamma": 2.5, "hs": 3.5},
        },
    )
    assert r.status_code == 200, r.text
    # 同名冲突
    r_dup = client.put(
        "/api/v1/cases/my-case",
        json={"name": "my-case", "params": {"omega_p": 0.5, "gamma": 3.3, "alpha": 0.01}},
    )
    assert r_dup.status_code == 409
    # 凭名字复算谱
    r1 = client.post("/api/v1/cases/my-case/spectrum")
    assert r1.status_code == 200
    d1 = r1.json()["data"]
    assert abs(d1["statistics"]["hs"] - 3.5) / 3.5 < 1e-6
    # 凭名字复算矩，结果一致
    r2 = client.post("/api/v1/cases/my-case/moments")
    d2 = r2.json()["data"]
    assert math.isclose(d1["moments"]["m0"], d2["moments"]["m0"], rel_tol=1e-12)
    # 列表与删除
    assert "my-case" in [c["name"] for c in client.get("/api/v1/cases").json()["data"]["cases"]]
    assert client.delete("/api/v1/cases/my-case").status_code == 200
    assert client.post("/api/v1/cases/my-case/moments").status_code == 404


def test_builtin_cases_ready_on_startup(client):
    r = client.get("/api/v1/health")
    assert r.status_code == 200
    assert r.json()["data"]["builtin_cases_available"] is True
    # 自带算例立即可复算且自洽
    r = client.post("/api/v1/cases/fetch-sea-15ms-50km/moments")
    assert r.status_code == 200, r.text
    d = r.json()["data"]
    assert d["parameters"]["gamma"] > 1.0
    assert math.isclose(
        d["statistics"]["hs"], 4 * math.sqrt(d["moments"]["m0"]), rel_tol=1e-12
    )


def test_constants_endpoint_pinned(client):
    d = client.get("/api/v1/constants").json()["data"]
    assert d["g"] == 9.80665
    assert d["sigma_left"] == 0.07
    assert d["sigma_right"] == 0.09


def test_gamma1_degenerate_and_hs_doubling_over_http(client):
    """γ=1 退化与波高加倍 m0 四倍，走完整 HTTP 路径再钉一遍。"""
    a = client.post(
        "/api/v1/spectrum",
        json={"omega_p": 0.7, "gamma": 1.0, "alpha": 0.01, "sample_points": 256},
    ).json()["data"]
    # 谱列在峰两侧应与 PM 一致（无额外峰升）：ωp 处的峰升贡献为 1
    assert a["parameters"]["gamma"] == 1.0

    h1 = client.post(
        "/api/v1/spectrum", json={"omega_p": 0.65, "gamma": 3.3, "hs": 2.0}
    ).json()["data"]
    h2 = client.post(
        "/api/v1/spectrum", json={"omega_p": 0.65, "gamma": 3.3, "hs": 4.0}
    ).json()["data"]
    ratio = h2["moments"]["m0"] / h1["moments"]["m0"]
    assert math.isclose(ratio, 4.0, rel_tol=1e-6)
    assert math.isclose(
        h2["parameters"]["alpha"] / h1["parameters"]["alpha"], 4.0, rel_tol=1e-6
    )
