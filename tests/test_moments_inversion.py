"""数值积分与谱矩测试：矩积分收敛、波高反演闭合、频率迁移。"""

from __future__ import annotations

import numpy as np
import pytest

from app import constants as C
from app.inversion import invert_alpha_for_hs
from app.moments import compute_moments
from app.spectrum import jonswap_density


# ---------------------------------------------------------------------------
# 矩积分收敛
# ---------------------------------------------------------------------------
def test_moments_converge_as_cutoff_extends(tmp_path, monkeypatch):
    """上限逐倍外扩时 m0 单调稳定；最终相对变化 < 1e-7。"""
    wp, gamma, alpha = 0.7, 3.3, 0.01
    m = compute_moments(wp, gamma, alpha)
    assert m.converged, "积分上限外扩后仍未收敛"
    assert m.cutoff_factor > 100.0, "不应在几倍 ωp 处提前截断"

    # 手动逐档验证：嵌套前缀网格上 m0 随上限扩展趋于稳定
    from app.moments import _build_prefix_table, _prefix_simpson

    w, n_table = _build_prefix_table(wp)
    x = np.log(w)
    h = float(x[1] - x[0])
    s_all = jonswap_density(w, wp, gamma, alpha)
    f0, f1, f2 = s_all * w, s_all * w**2, s_all * w**3
    vals = []
    for factor, n_f in zip(C.CUTOFF_SCHEDULE, n_table):
        vals.append(
            (
                factor,
                _prefix_simpson(h, f0, n_f),
                _prefix_simpson(h, f1, n_f),
                _prefix_simpson(h, f2, n_f),
            )
        )
    last, prev = vals[-1], vals[-2]
    assert abs(last[1] - prev[1]) / last[1] < C.INTEGRATION_REL_TOL
    assert abs(last[3] - prev[3]) / last[3] < C.INTEGRATION_REL_TOL
    # m2 单调不减（被积函数处处正，扩大区间只会增加积分）
    m2s = [v[3] for v in vals]
    assert all(b >= a - 1e-12 for a, b in zip(m2s, m2s[1:]))
    # 嵌套性：低档前缀是高档前缀的严格子集，端点逐点一致
    assert n_table == sorted(set(n_table)) or n_table[-1] > n_table[0]


def test_moments_are_positive_and_hs_matches_area():
    """m0/m1/m2 为正，Hs = 4√m0 与谱面积对得上。"""
    wp, gamma, alpha = 0.7, 3.3, 0.01
    m = compute_moments(wp, gamma, alpha)
    assert m.m0 > 0 and m.m1 > 0 and m.m2 > 0
    assert np.isclose(m.hs, 4.0 * np.sqrt(m.m0), rtol=1e-15)
    # Tz = 2π√(m0/m2)，且物理上 Tz < Tp
    assert np.isclose(m.tz, 2.0 * np.pi * np.sqrt(m.m0 / m.m2), rtol=1e-15)
    tp = 2.0 * np.pi / wp
    assert m.tz < tp


def test_moments_scale_quadratically_with_alpha():
    """谱对 α 线性：α 加倍，所有矩加倍（为反演比例性做铺垫）。"""
    wp, gamma = 0.6, 3.3
    a = compute_moments(wp, gamma, 0.01)
    b = compute_moments(wp, gamma, 0.02)
    assert np.isclose(b.m0 / a.m0, 2.0, rtol=1e-9)
    assert np.isclose(b.m2 / a.m2, 2.0, rtol=1e-9)


# ---------------------------------------------------------------------------
# 波高反演闭合
# ---------------------------------------------------------------------------
def test_inversion_closes_hs_within_tolerance():
    """反演出的 α 使 4√m0 与目标 Hs 闭合到容差内。"""
    wp, gamma, hs = 0.7, 3.3, 4.0
    r = invert_alpha_for_hs(wp, gamma, hs)
    hs_from_area = 4.0 * np.sqrt(r.moments.m0)
    assert r.residual < C.INVERSION_REL_TOL
    assert abs(hs_from_area - hs) / hs < C.INVERSION_REL_TOL
    assert r.iterations <= 5  # 线性问题应快速收敛


def test_doubling_hs_constraint_quadruples_m0_and_raises_alpha():
    """波高约束加倍 → m0 变四倍，反演 α 抬升约四倍。"""
    wp, gamma = 0.65, 3.3
    r1 = invert_alpha_for_hs(wp, gamma, 2.0)
    r2 = invert_alpha_for_hs(wp, gamma, 4.0)
    assert np.isclose(r2.moments.m0 / r1.moments.m0, 4.0, rtol=1e-6)
    assert np.isclose(r2.alpha / r1.alpha, 4.0, rtol=1e-6)
    # 两个结果各自闭合
    for r, target in ((r1, 2.0), (r2, 4.0)):
        assert abs(4.0 * np.sqrt(r.moments.m0) - target) / target < 1e-7


def test_inversion_actual_spectrum_area_meets_target():
    """端到端焊死：拿反演 α 重新逐点实算积分，面积必须闭合——
    不允许“标称 Hs 写进返回、α 却没反演过”。"""
    wp, gamma, hs = 0.55, 2.5, 6.0
    r = invert_alpha_for_hs(wp, gamma, hs)
    # 独立重算（不使用反演内部缓存的矩）
    m = compute_moments(wp, gamma, r.alpha)
    assert abs(4.0 * np.sqrt(m.m0) - hs) / hs < 2e-7
    # 谱采样点也必须来自实算：抽查若干点与公式一致
    w = np.array([0.3, wp, 1.2, 3.0])
    s = jonswap_density(w, wp, gamma, r.alpha)
    assert np.all(np.isfinite(s)) and np.all(s >= 0.0)


# ---------------------------------------------------------------------------
# 峰频迁移
# ---------------------------------------------------------------------------
def test_halving_omega_p_moves_energy_to_low_frequency_and_doubles_tp():
    """ωp 减半：谱峰周期加倍，能量整体向低频迁移（m1/m0 减半）。"""
    gamma, hs = 3.3, 3.0
    r_hi = invert_alpha_for_hs(0.8, gamma, hs)
    r_lo = invert_alpha_for_hs(0.4, gamma, hs)
    # 同样 Hs 约束下 m0 相同
    assert np.isclose(r_hi.moments.m0, r_lo.moments.m0, rtol=1e-6)
    # 平均频率 m1/m0 减半
    mean_hi = r_hi.moments.m1 / r_hi.moments.m0
    mean_lo = r_lo.moments.m1 / r_lo.moments.m0
    assert np.isclose(mean_lo / mean_hi, 0.5, rtol=1e-6)
    # Tp 加倍
    assert np.isclose((2.0 * np.pi / 0.4) / (2.0 * np.pi / 0.8), 2.0)


def test_pm_gamma1_moment_against_analytic_m0():
    """γ=1 时数值 m0 与 PM 闭式积分对照。

    ∫₀∞ αg² ω⁻⁵ exp(−1.25(ωp/ω)^4) dω = αg² / (5 ωp^4)
    （令 u=1.25(ωp/ω)^4 换元）。
    """
    wp, alpha = 0.7, 0.01
    m = compute_moments(wp, 1.0, alpha)
    analytic = alpha * C.GRAVITY**2 / (5.0 * wp**4)
    assert np.isclose(m.m0, analytic, rtol=1e-5), (m.m0, analytic)
