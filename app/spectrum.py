"""JONSWAP 谱形核算（Hasselmann 标准式）。

    S(ω) = α·g²·ω⁻⁵·exp(−1.25·(ωp/ω)⁴)·γ^r
    r    = exp(−(ω−ωp)²/(2σ²ωp²))

注意高频幂次：前置因子是 ω**−5，指数里是 (ωp/ω)**4（即 ω**−4）。
两者写反会让谱在高频翘尾，测试专门钉这一点。

当 γ=1 时 γ^r ≡ 1，谱形严格退化为 Pierson–Moskowitz 谱：
S_PM(ω) = α·g²·ω⁻⁵·exp(−1.25(ωp/ω)⁴)。
"""

from __future__ import annotations

import numpy as np

from . import constants as C


def peak_enhancement(omega: np.ndarray | float, omega_p: float, gamma: float) -> np.ndarray | float:
    """峰升因子 γ^r；左右两侧 σ 取不同定值。"""
    w = np.asarray(omega, dtype=np.float64)
    if gamma == 1.0:
        return np.ones_like(w) if w.ndim else 1.0

    sigma = np.where(w <= omega_p, C.SIGMA_LEFT, C.SIGMA_RIGHT)
    r = np.exp(-((w - omega_p) ** 2) / (2.0 * sigma**2 * omega_p**2))
    return float(np.power(gamma, r)) if not w.ndim else np.power(gamma, r)


def jonswap_density(
    omega: np.ndarray | float,
    omega_p: float,
    gamma: float,
    alpha: float,
) -> np.ndarray | float:
    """逐点实算 JONSWAP 谱密度 S(ω)。

    不做任何尖峰形状的近似/写死；调用方可以喂标量也可以喂数组。
    入参合法性由 validation 层在进入本函数前保证。
    """
    w = np.asarray(omega, dtype=np.float64)
    ratio = omega_p / w

    pm_base = (
        alpha
        * C.GRAVITY**2
        * np.power(w, -5.0)  # ω⁻⁵（不是 ⁻⁴）
        * np.exp(-C.EXP_FETCH_COEFF * np.power(ratio, 4.0))  # (ωp/ω)⁴
    )
    enhancement = peak_enhancement(w, omega_p, gamma)
    if np.ndim(omega) == 0:
        return float(pm_base * enhancement)
    return pm_base * enhancement


def pierson_moskowitz_density(
    omega: np.ndarray | float, omega_p: float, alpha: float
) -> np.ndarray | float:
    """PM 谱；保留独立入口以便测试直接比对 γ=1 退化。"""
    w = np.asarray(omega, dtype=np.float64)
    return (
        alpha
        * C.GRAVITY**2
        * np.power(w, -5.0)
        * np.exp(-C.EXP_FETCH_COEFF * np.power(omega_p / w, 4.0))
    )


def peak_density(omega_p: float, gamma: float, alpha: float) -> float:
    """谱峰位置 ω=ωp 处的谱值（闭式，测试用它核对峰值）。

    在 ω=ωp 处：exp(−1.25)·γ¹，σ 取左侧常数（不影响峰点，r=1）。
    """
    return float(alpha * C.GRAVITY**2 * omega_p**-5 * np.exp(-C.EXP_FETCH_COEFF) * gamma)
