"""编排层测试：正算/反演结果自洽、风区经验式、并行隔离。"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np

from app import constants as C
from app.service import (
    calculate_moments,
    calculate_spectrum,
    resolve_input,
)


def test_spectrum_end_to_end_with_explicit_alpha():
    inp = resolve_input(omega_p=0.7, gamma=3.3, alpha=0.01)
    res = calculate_spectrum(inp, sample_points=256)
    assert len(res.omega) == len(res.spectrum) == 256
    # 谱列单调覆盖 ωp/3 ~ 8ωp
    assert np.isclose(res.omega[0], 0.7 / 3.0, rtol=1e-12)
    assert np.isclose(res.omega[-1], 0.7 * 8.0, rtol=1e-12)
    # 每个采样点非负有限；对数网格上 ωp 位于网格区间
    arr = np.array(res.spectrum)
    assert np.all(np.isfinite(arr)) and np.all(arr >= 0)
    mid = len(arr) // 2
    assert res.omega[mid - 1] < 0.7 <= res.omega[mid + 1]
    # 峰点（最大采样值）就在 ωp 附近的 1 个网格内
    assert abs(res.omega[int(np.argmax(arr))] - 0.7) / 0.7 < 0.02
    # Hs 与矩闭合
    assert np.isclose(res.hs, 4.0 * np.sqrt(res.m0), rtol=1e-15)
    assert np.isclose(res.tp, 2.0 * np.pi / 0.7, rtol=1e-15)
    assert res.tz < res.tp
    assert res.integration_converged


def test_spectrum_with_hs_constraint_is_self_consistent():
    """Hs 约束路径：返回的 Hs 与谱面积闭合，反演残差在容差内。"""
    inp = resolve_input(omega_p=0.62, gamma=2.8, hs=5.0)
    res = calculate_spectrum(inp, sample_points=300)
    assert res.mode == "hs_inversion"
    assert res.inversion_residual is not None
    assert res.inversion_residual < C.INVERSION_REL_TOL
    assert abs(res.hs - 5.0) / 5.0 < C.INVERSION_REL_TOL
    assert np.isclose(res.m0, (5.0 / 4.0) ** 2, rtol=1e-6)
    assert res.hs_target == 5.0


def test_fetch_empirical_case():
    """有限风区经验式：风区加倍 → ωp 下降、Hs 上升。"""
    inp1 = resolve_input(wind_speed=15.0, fetch=25_000.0)
    inp2 = resolve_input(wind_speed=15.0, fetch=50_000.0)
    r1 = calculate_moments(inp1)
    r2 = calculate_moments(inp2)
    assert inp1.gamma > 1.0 and inp2.gamma > 1.0
    assert r2.omega_p < r1.omega_p  # 风区更长，峰频更低
    assert r2.hs > r1.hs
    # 经验比例：ωp ∝ F^-1/3
    ratio = (r2.omega_p / r1.omega_p) ** 3
    assert np.isclose(ratio, 25_000.0 / 50_000.0, rtol=1e-9)


def test_moments_only_endpoint_has_no_samples():
    inp = resolve_input(omega_p=0.7, gamma=3.3, alpha=0.01)
    m = calculate_moments(inp)
    d = m.to_dict()
    assert "spectrum" not in d and "omega" not in d
    assert set(d["moments"]) == {"m0", "m1", "m2"}
    assert {"hs", "tp", "tz", "t1"} <= set(d["statistics"])
    assert np.isclose(d["statistics"]["hs"], 4.0 * np.sqrt(m.m0))


def test_parallel_cases_do_not_cross_contaminate():
    """多个不同工况并行提交，中间量绝不串档。"""
    specs = [
        dict(omega_p=0.4, gamma=3.3, hs=2.0),
        dict(omega_p=0.8, gamma=1.0, alpha=0.005),
        dict(omega_p=0.6, gamma=7.0, hs=8.0),
        dict(omega_p=0.9, gamma=2.0, alpha=0.02),
        dict(omega_p=0.3, gamma=3.3, hs=4.0),
        dict(wind_speed=15.0, fetch=50_000.0),
    ]

    def run(spec):
        inp = resolve_input(**spec)
        return calculate_spectrum(inp, sample_points=200)

    with ThreadPoolExecutor(max_workers=6) as pool:
        results = list(pool.map(run, specs))

    # 每条结果只与自己的入参一致
    for spec, res in zip(specs, results):
        if "omega_p" in spec:
            assert res.omega_p == spec["omega_p"]
        if "gamma" in spec:
            assert res.gamma == spec["gamma"]
        assert abs(4.0 * np.sqrt(res.m0) - res.hs) / res.hs < 1e-9
        if res.hs_target is not None:
            assert abs(res.hs - res.hs_target) / res.hs_target < 1e-6

    # 两个 Hs 约束工况的 m0 必须精确对应各自波高，互不串
    by_m0 = {round(r.m0, 8) for r in results}
    assert (2.0 / 4.0) ** 2 in by_m0 or any(
        abs(r.hs - 2.0) < 1e-6 for r in results
    )


def test_alpha_takes_precedence_over_hs():
    inp = resolve_input(omega_p=0.7, gamma=3.3, alpha=0.01, hs=99.0)
    res = calculate_moments(inp)
    assert res.alpha == 0.01
    assert res.hs != 99.0  # 给了 α 就不反演
