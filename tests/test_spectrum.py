"""谱形正算的因果关系测试。

钉住：
- 逐点实算，峰确实落在 ωp；
- γ 增大（>1）同一 α 下峰更尖、峰值更高；
- γ=1 严格退化到 Pierson–Moskowitz 谱形；
- 高频幂次：ω⁻⁵ 前置、指数里 (ωp/ω)⁴，谱尾必须单调衰减（不翘尾）。
"""

from __future__ import annotations

import numpy as np

from app import constants as C
from app.spectrum import (
    jonswap_density,
    peak_density,
    peak_enhancement,
    pierson_moskowitz_density,
)


def test_peak_located_at_omega_p():
    """峰升因子的峰严格在 ωp；JONSWAP 全谱峰随 γ 增大钉在 ωp 附近。"""
    wp, alpha = 0.7, 0.01
    w = np.linspace(0.2, 3.0, 200001)

    # 峰升因子 γ^r 的最大值恰在 ωp
    enh = peak_enhancement(w, wp, 3.3)
    assert abs(w[int(np.argmax(enh))] - wp) < 2.0e-5

    # γ=3.3 时全谱峰相对 ωp 的偏移 <1%（PM 基底在峰处仍缓慢上升）
    s33 = jonswap_density(w, wp, 3.3, alpha)
    assert abs(w[int(np.argmax(s33))] - wp) / wp < 1.0e-2
    # γ 更大时峰被进一步钉回 ωp
    s7 = jonswap_density(w, wp, 7.0, alpha)
    assert abs(w[int(np.argmax(s7))] - wp) / wp < 5.0e-3
    # 峰点闭式值（S(ωp) = αg²ωp⁻⁵e^(−1.25)·γ）
    assert np.isclose(
        float(jonswap_density(np.array(wp), wp, 3.3, alpha)),
        peak_density(wp, 3.3, alpha),
        rtol=1e-12,
    )


def test_gamma_raises_sharper_and_higher_peak_same_alpha():
    """同一 α 下 γ 从 1 加大，峰值线性抬高、峰两侧更尖（远离峰处趋于一致）。"""
    wp, alpha = 0.8, 0.012
    w = np.linspace(0.25, 2.5, 9001)
    s_g1 = jonswap_density(w, wp, 1.0, alpha)
    s_g33 = jonswap_density(w, wp, 3.3, alpha)
    s_g7 = jonswap_density(w, wp, 7.0, alpha)

    pk1 = float(jonswap_density(np.array(wp), wp, 1.0, alpha))
    pk33 = float(jonswap_density(np.array(wp), wp, 3.3, alpha))
    pk7 = float(jonswap_density(np.array(wp), wp, 7.0, alpha))
    # 在 ωp 处 γ^r=γ，谱值严格按 γ 倍数抬高（同一 PM 基底）
    assert np.isclose(pk33 / pk1, 3.3, rtol=1e-12)
    assert np.isclose(pk7 / pk1, 7.0, rtol=1e-12)
    assert pk7 > pk33 > pk1

    # 峰升因子处处 >= 1，且在 ωp 处恰为 γ
    assert np.all(s_g33 >= s_g1 * (1.0 - 1e-14))
    r = peak_enhancement(np.array([wp]), wp, 3.3)
    assert np.isclose(float(np.asarray(r)[0]), 3.3)

    # “更尖”：高 γ 时能量更向峰附近集中 —— 半峰宽更窄
    def half_width(s):
        above = w[s >= float(np.max(s)) / 2.0]
        return above[-1] - above[0]

    assert half_width(s_g7) < half_width(s_g33) < half_width(s_g1)


def test_gamma_one_equals_pierson_moskowitz():
    """γ=1 时逐点等于 PM 谱（整条曲线，不只是峰附近）。"""
    wp, alpha = 0.65, 0.0081
    w = np.linspace(0.2, 5.0, 5001)
    s_j = jonswap_density(w, wp, 1.0, alpha)
    s_pm = pierson_moskowitz_density(w, wp, alpha)
    assert np.allclose(s_j, s_pm, rtol=1e-14, atol=0.0)
    # 峰升因子恒为 1
    assert np.allclose(peak_enhancement(w, wp, 1.0), 1.0)


def test_high_frequency_tail_decays_no_upturn():
    """高频尾必须单调衰减——专门钉 ⁻⁵ / ⁻⁴ 写反导致的翘尾。"""
    wp, gamma, alpha = 0.7, 3.3, 0.01
    w = np.geomspace(wp * 2.0, wp * 200.0, 400)
    s = jonswap_density(w, wp, gamma, alpha)
    assert np.all(np.diff(s) < 0.0), "高频尾出现抬升，检查 ω 的幂次"

    # 进一步钉死幂次：ω≫ωp 时 exp 项≈1，S ≈ αg²ω⁻⁵，
    # 故 S(2ω)/S(ω) → 2⁻⁵ = 1/32
    far = np.array([wp * 500.0, wp * 1000.0])
    s_far = jonswap_density(far, wp, gamma, alpha)
    ratio = s_far[1] / s_far[0]  # 频率比 2
    assert np.isclose(ratio, 0.5**5, rtol=2e-3), f"高频尾幂次不对：ratio={ratio}"


def test_low_frequency_side_exponential_cutoff():
    """低频侧 ω≪ωp 被 exp(−1.25(ωp/ω)^4) 压到接近零。"""
    wp, gamma, alpha = 0.7, 3.3, 0.01
    s_low = jonswap_density(np.array([wp * 0.05]), wp, gamma, alpha)
    s_peak = peak_density(wp, gamma, alpha)
    assert s_low < 1e-20 * s_peak


def test_constants_are_pinned():
    """σ 左右值与 g 在服务里钉死。"""
    assert C.GRAVITY == 9.80665
    assert C.SIGMA_LEFT == 0.07
    assert C.SIGMA_RIGHT == 0.09
    assert C.EXP_FETCH_COEFF == 1.25
