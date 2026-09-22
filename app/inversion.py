"""有效波高约束下的尺度因子 α 反演（与积分模块独立）。

谱对 α 是线性的：S = α·F(ω)，故 m0(α) = α·M0，
目标 Hs 满足 Hs² = 16·m0 = 16·α·M0。

这里仍按要求“迭代调整 α、由积分算出的 m0 闭合到容差”来实现：
每轮在固定收敛网格上重算 m0 并做定点修正
    α_{n+1} = α_n · Hs_target² / (16·m0(α_n))
直到 |4√m0 − Hs| / Hs < INVERSION_REL_TOL。
（线性问题通常两步即收敛，但闭合判定由真实积分给出，不做闭式代换。）
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import constants as C
from .errors import InversionError
from .moments import GridMoments, Moments, compute_moments, moments_on_fixed_grid
from .spectrum import jonswap_density


@dataclass(frozen=True)
class InversionResult:
    alpha: float
    moments: Moments
    iterations: int
    residual: float  # |4√m0 − Hs| / Hs
    hs_target: float


def invert_alpha_for_hs(
    omega_p: float,
    gamma: float,
    hs_target: float,
    *,
    alpha0: float = 1.0e-2,
    tol: float = C.INVERSION_REL_TOL,
    max_iter: int = C.INVERSION_MAX_ITER,
) -> InversionResult:
    """反演 α 使 4√m0 闭合到给定有效波高。"""
    # 先用初始 α 选好收敛网格（积分上限与网格只与形状、不随 α 改变）
    grid: GridMoments = compute_moments(
        omega_p, gamma, alpha0, return_grid=True
    )  # type: ignore[assignment]
    if not grid.moments.converged:
        raise InversionError(
            f"矩积分在最大截止倍率 {C.CUTOFF_SCHEDULE[-1]:g}·ωp 处仍未收敛，"
            "拒绝在被截断的面积上反演 α"
        )
    w = grid.omega

    alpha = float(alpha0)
    moments: Moments = grid.moments
    residual = float("inf")
    iterations = 0

    for iterations in range(1, max_iter + 1):
        s = jonswap_density(w, omega_p, gamma, alpha)
        m0, m1, m2 = moments_on_fixed_grid(w, s)
        if m0 <= 0.0 or not np.isfinite(m0):
            raise InversionError(
                f"反演过程中 m0 非正或非有限（α={alpha:g}），无法闭合波高约束"
            )
        hs_actual = 4.0 * np.sqrt(m0)
        residual = abs(hs_actual - hs_target) / hs_target
        moments = Moments(
            m0=m0,
            m1=m1,
            m2=m2,
            omega_cutoff=grid.moments.omega_cutoff,
            cutoff_factor=grid.moments.cutoff_factor,
            converged=grid.moments.converged,
            iterations=grid.moments.iterations,
        )
        if residual < tol:
            break
        # 定点修正：谱对 α 线性，直接按比例校正
        alpha = alpha * hs_target**2 / (16.0 * m0)
    else:
        raise InversionError(
            f"α 反演在 {max_iter} 次迭代后仍未闭合（残差 {residual:.3e} > {tol:.0e}）"
        )

    return InversionResult(
        alpha=alpha,
        moments=moments,
        iterations=iterations,
        residual=residual,
        hs_target=hs_target,
    )
