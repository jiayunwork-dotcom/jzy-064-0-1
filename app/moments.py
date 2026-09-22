"""数值积分与谱矩 m0/m1/m2（与波高反演严格分离）。

积分策略：在 log(ω) 均匀网格上用复合 Simpson 公式积分
    ∫ S(ω) dω = ∫ S(e^x)·e^x dx,  x = ln ω

实现要点——**嵌套前缀网格**：
- 为截止倍率表中的每一档 f 预算偶数区间数
  n_f = ceil_even( ln(f/ω_lo_factor) / h )；
- 在统一的 log 网格（到最细档）上，每一档就是前 n_f 个区间构成的前缀；
- 公共区间在各档之间逐点完全相同，相邻档之差纯粹是新增尾流的贡献，
  收敛判定不被网格抖动污染。

上限逐倍外扩直到 m0/m1/m2 的相对增量全部 < INTEGRATION_REL_TOL，
杜绝“积到一半截断导致 m0 偏小”。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import constants as C
from .spectrum import jonswap_density


@dataclass(frozen=True)
class Moments:
    """谱矩与派生统计量（纯核算结果，不含入参回显）。"""

    m0: float
    m1: float
    m2: float
    omega_cutoff: float  # 最终采用的积分上限
    cutoff_factor: float  # omega_cutoff / omega_p
    converged: bool
    iterations: int  # 用了倍率表中的第几档（1 基）

    @property
    def hs(self) -> float:
        """有效波高 Hs = 4√m0。"""
        return 4.0 * np.sqrt(max(self.m0, 0.0))

    @property
    def tz(self) -> float:
        """跨零周期 Tz = 2π·√(m0/m2)。"""
        if self.m2 <= 0.0:
            return float("inf")
        return float(2.0 * np.pi * np.sqrt(self.m0 / self.m2))

    @property
    def t1(self) -> float:
        """能量加权平均周期 T1 = 2π·m0/m1（附加统计量）。"""
        if self.m1 <= 0.0:
            return float("inf")
        return float(2.0 * np.pi * self.m0 / self.m1)


@dataclass(frozen=True)
class GridMoments:
    """谱矩 + 最终收敛所用的网格（反演复用，避免重复选档）。"""

    moments: Moments
    omega: np.ndarray
    s: np.ndarray


def _ceil_even(n: int) -> int:
    return n if n % 2 == 0 else n + 1


def _build_prefix_table(omega_p: float) -> tuple[np.ndarray, list[int]]:
    """返回共享 log 网格 ω_i 与每档对应的偶数前缀区间数表。"""
    x_lo = float(np.log(omega_p * C.OMEGA_LOW_FACTOR))
    h = C.LOG_STEP

    n_per_factor: list[int] = []
    for factor in C.CUTOFF_SCHEDULE:
        x_f = float(np.log(omega_p * factor))
        n_per_factor.append(max(_ceil_even(int(np.ceil((x_f - x_lo) / h))), 2))

    # 单调保护（倍率递增，区间数必递增）
    for i in range(1, len(n_per_factor)):
        if n_per_factor[i] <= n_per_factor[i - 1]:
            n_per_factor[i] = n_per_factor[i - 1] + 2

    n_last = n_per_factor[-1]
    x = x_lo + h * np.arange(n_last + 1)
    return np.exp(x), n_per_factor


def _simpson(h: float, f: np.ndarray) -> float:
    """复合 Simpson 积分；f 的区间数（len−1）必须为偶数。"""
    if (len(f) - 1) % 2 != 0:
        raise ValueError("Simpson 积分要求区间数为偶数")
    return float(h / 3.0 * (f[0] + f[-1] + 4.0 * f[1:-1:2].sum() + 2.0 * f[2:-1:2].sum()))


def _prefix_simpson(h: float, f: np.ndarray, n: int) -> float:
    """对数组前 n 个区间（n+1 个点）做 Simpson。"""
    return _simpson(h, f[: n + 1])


def _rel_change(prev: tuple[float, float, float], cur: tuple[float, float, float]) -> float:
    return max(abs(c - p) / max(abs(c), C.EPS_SMALL) for p, c in zip(prev, cur))


def compute_moments(
    omega_p: float,
    gamma: float,
    alpha: float,
    *,
    tol: float = C.INTEGRATION_REL_TOL,
    return_grid: bool = False,
) -> Moments | GridMoments:
    """自适应外扩上限计算谱矩。

    谱密度只在共享网格上实算一次，每一档对同一数组的前缀积分，
    公共区间结果逐位一致。return_grid=True 时额外返回收敛档前缀网格
    与 S(ω)，供反演循环复用。
    """
    w, n_table = _build_prefix_table(omega_p)
    x = np.log(w)
    h = float(x[1] - x[0])
    s = jonswap_density(w, omega_p, gamma, alpha)

    # log 轴被积函数：dω = ω dx
    f0 = s * w
    f1 = f0 * w
    f2 = f1 * w

    prev: tuple[float, float, float] | None = None
    cur: tuple[float, float, float] = (0.0, 0.0, 0.0)
    n_used = 0
    converged = False
    used = 0

    for used, n_f in enumerate(n_table, start=1):
        cur = (
            _prefix_simpson(h, f0, n_f),
            _prefix_simpson(h, f1, n_f),
            _prefix_simpson(h, f2, n_f),
        )
        if prev is not None and _rel_change(prev, cur) < tol:
            converged = True
            n_used = n_f
            break
        prev = cur
        n_used = n_f

    m0, m1, m2 = cur
    omega_cutoff = float(w[n_used])
    moments = Moments(
        m0=float(m0),
        m1=float(m1),
        m2=float(m2),
        omega_cutoff=omega_cutoff,
        cutoff_factor=omega_cutoff / omega_p,
        converged=converged,
        iterations=used,
    )
    if return_grid:
        return GridMoments(moments=moments, omega=w[: n_used + 1], s=s[: n_used + 1])
    return moments


def moments_on_fixed_grid(omega: np.ndarray, s: np.ndarray) -> tuple[float, float, float]:
    """在给定（log 均匀）网格上重算 m0/m1/m2，反演循环内复用。"""
    x = np.log(omega)
    h = float(x[1] - x[0])
    m0 = _simpson(h, s * omega)
    m1 = _simpson(h, s * omega**2)
    m2 = _simpson(h, s * omega**3)
    return m0, m1, m2
