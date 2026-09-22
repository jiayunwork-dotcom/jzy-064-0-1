"""有限风区 JONSWAP 经验式（Hasselmann et al., 1973）。

给定 10 m 高度风速 U 与风区长度 F：
    无量纲风区      x̃ = g·F / U²
    α = 0.076·x̃⁻⁰·²²
    峰频（无量纲）  ω̃p = 22·x̃⁻¹/³，即 ωp = 22·(g/U)·x̃⁻¹/³
    γ = 3.3（平均取值，γ>1）

仅当调用方提供了 wind_speed + fetch、且未直接给 α / Hs 时使用。
"""

from __future__ import annotations

from . import constants as C

ALPHA_FETCH_COEFF = 0.076
ALPHA_FETCH_EXP = -0.22
GAMMA_FETCH_DEFAULT = 3.3
DIMENSIONLESS_PEAK_COEFF = 22.0  # 2π·3.5 ≈ 21.99


def fetch_params(wind_speed: float, fetch: float) -> dict[str, float]:
    """由风速 [m/s] 与风区 [m] 估算 α、ωp、γ。"""
    x_tilde = C.GRAVITY * fetch / wind_speed**2
    alpha = ALPHA_FETCH_COEFF * x_tilde**ALPHA_FETCH_EXP
    omega_p = (
        DIMENSIONLESS_PEAK_COEFF * (C.GRAVITY / wind_speed) * x_tilde ** (-1.0 / 3.0)
    )
    return {
        "alpha": float(alpha),
        "omega_p": float(omega_p),
        "gamma": GAMMA_FETCH_DEFAULT,
    }
