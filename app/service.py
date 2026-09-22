"""谱正算 / 反演 / 矩统计的编排层（无 HTTP 细节，纯领域服务）。

并发安全：本层所有计算都是纯函数式的——每次请求的网格、谱数组、
反演中间量都是调用内的局部对象，不同工况之间不存在任何共享可变状态，
并行提交各算各的。唯一的共享状态是工况档存储（cases.py，自带锁）。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import constants as C
from .fetch import GAMMA_FETCH_DEFAULT, fetch_params
from .inversion import InversionResult, invert_alpha_for_hs
from .moments import Moments, compute_moments
from .spectrum import jonswap_density
from .validation import SpectrumInput, validate_spectrum_input


@dataclass(frozen=True)
class SpectrumResult:
    """一次完整谱计算的自洽结果。"""

    omega: list[float]
    spectrum: list[float]
    m0: float
    m1: float
    m2: float
    hs: float  # 始终由 4√m0 实算，保证与谱面积闭合
    hs_target: float | None
    tp: float  # 2π/ωp
    tz: float  # 跨零周期 2π√(m0/m2)
    t1: float
    omega_p: float
    gamma: float
    alpha: float
    mode: str
    omega_cutoff: float
    cutoff_factor: float
    integration_converged: bool
    inversion_iterations: int | None = None
    inversion_residual: float | None = None
    sample_points: int = 0
    wind_speed: float | None = None
    fetch: float | None = None
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "omega": self.omega,
            "spectrum": self.spectrum,
            "moments": {"m0": self.m0, "m1": self.m1, "m2": self.m2},
            "statistics": {
                "hs": self.hs,
                "hs_target": self.hs_target,
                "tp": self.tp,
                "tz": self.tz,
                "t1": self.t1,
            },
            "parameters": {
                "omega_p": self.omega_p,
                "gamma": self.gamma,
                "alpha": self.alpha,
                "mode": self.mode,
                "wind_speed": self.wind_speed,
                "fetch": self.fetch,
            },
            "integration": {
                "omega_cutoff": self.omega_cutoff,
                "cutoff_factor": self.cutoff_factor,
                "converged": self.integration_converged,
            },
            "inversion": None
            if self.inversion_iterations is None
            else {
                "iterations": self.inversion_iterations,
                "relative_residual": self.inversion_residual,
            },
            "sample_points": self.sample_points,
            "meta": self.meta,
        }


@dataclass(frozen=True)
class MomentsResult:
    """矩与由矩导出的统计周期（不含谱采样列）。"""

    m0: float
    m1: float
    m2: float
    hs: float
    tp: float
    tz: float
    t1: float
    omega_p: float
    gamma: float
    alpha: float
    mode: str
    omega_cutoff: float
    cutoff_factor: float
    converged: bool

    def to_dict(self) -> dict:
        return {
            "moments": {"m0": self.m0, "m1": self.m1, "m2": self.m2},
            "statistics": {
                "hs": self.hs,
                "tp": self.tp,
                "tz": self.tz,
                "t1": self.t1,
            },
            "parameters": {
                "omega_p": self.omega_p,
                "gamma": self.gamma,
                "alpha": self.alpha,
                "mode": self.mode,
            },
            "integration": {
                "omega_cutoff": self.omega_cutoff,
                "cutoff_factor": self.cutoff_factor,
                "converged": self.converged,
            },
        }


def resolve_input(
    *,
    omega_p: float | None,
    gamma: float | None,
    alpha: float | None = None,
    hs: float | None = None,
    wind_speed: float | None = None,
    fetch: float | None = None,
) -> SpectrumInput:
    """补齐经验式默认值后做统一校验。

    - 未给 ωp 但给了 wind_speed+fetch：按有限风区经验式推出 ωp、α、γ。
    - γ 缺省取 3.3（JONSWAP 平均峰升因子，>1）。
    - α 优先于 Hs 约束，Hs 优先于风区经验 α。
    """
    gv = GAMMA_FETCH_DEFAULT if gamma is None else gamma

    if (alpha is None and hs is None) and wind_speed is not None and fetch is not None:
        emp = fetch_params(wind_speed, fetch)
        if omega_p is None:
            omega_p = emp["omega_p"]
        if gamma is None:
            gv = emp["gamma"]
        alpha = emp["alpha"]

    if omega_p is None:
        # 校验层会给出明确报错，这里不做静默猜测
        from .errors import ValidationError

        raise ValidationError(
            "必须提供 omega_p，或同时提供 wind_speed 与 fetch 以按风区经验式推算",
            field="omega_p",
        )

    validated = validate_spectrum_input(
        omega_p=omega_p,
        gamma=gv,
        alpha=alpha,
        hs=hs,
        wind_speed=wind_speed,
        fetch=fetch,
    )
    return validated


def _resolve_alpha(inp: SpectrumInput) -> tuple[float, Moments, dict]:
    """按输入模式确定 α 及对应矩：直接给定 / Hs 反演 / 风区经验。

    返回 (alpha, moments, inversion_info)。
    """
    inv_info: dict = {}
    if inp.alpha is not None:
        moments = compute_moments(inp.omega_p, inp.gamma, inp.alpha)
        if not moments.converged:
            from .errors import ServiceError

            raise ServiceError(
                f"矩积分在最大截止倍率 {C.CUTOFF_SCHEDULE[-1]:g}·ωp 处仍未收敛，"
                "拒绝返回被截断的谱面积",
                code="integration_not_converged",
                status_code=422,
            )
        return inp.alpha, moments, inv_info

    # alpha 缺省且 hs 缺省的情况理论上被 validation 挡下（fetch 分支已在
    # resolve_input 转成 alpha），这里防御性处理。
    if inp.hs is None:
        from .errors import ValidationError

        raise ValidationError("缺少 α 或有效波高 hs，无法确定谱尺度", field="alpha")

    result: InversionResult = invert_alpha_for_hs(inp.omega_p, inp.gamma, inp.hs)
    inv_info = {
        "iterations": result.iterations,
        "relative_residual": result.residual,
        "hs_target": result.hs_target,
    }
    return result.alpha, result.moments, inv_info


def sample_grid(omega_p: float, points: int) -> np.ndarray:
    """对外谱采样网格（log 均匀，覆盖 ωp/3 ~ 8ωp）。"""
    points = max(C.SAMPLE_POINTS_MIN, min(points, C.SAMPLE_POINTS_MAX))
    return np.exp(
        np.linspace(
            np.log(omega_p * C.SAMPLE_LOW_FACTOR),
            np.log(omega_p * C.SAMPLE_HIGH_FACTOR),
            points,
        )
    )


def calculate_spectrum(
    inp: SpectrumInput, *, sample_points: int = C.SAMPLE_POINTS_DEFAULT
) -> SpectrumResult:
    """正算入口：确定 α（可能反演）→ 积分得矩 → 逐点实算谱采样。"""
    alpha, moments, inv_info = _resolve_alpha(inp)

    omega = sample_grid(inp.omega_p, sample_points)
    # 每个采样点都由标准公式实算，禁止写死尖峰形状
    s_values = jonswap_density(omega, inp.omega_p, inp.gamma, alpha)

    hs = moments.hs
    tp = 2.0 * np.pi / inp.omega_p

    return SpectrumResult(
        omega=[float(v) for v in omega],
        spectrum=[float(v) for v in s_values],
        m0=float(moments.m0),
        m1=float(moments.m1),
        m2=float(moments.m2),
        hs=float(hs),
        hs_target=inv_info.get("hs_target"),
        tp=float(tp),
        tz=float(moments.tz),
        t1=float(moments.t1),
        omega_p=inp.omega_p,
        gamma=inp.gamma,
        alpha=float(alpha),
        mode=inp.mode,
        omega_cutoff=float(moments.omega_cutoff),
        cutoff_factor=float(moments.cutoff_factor),
        integration_converged=bool(moments.converged),
        inversion_iterations=inv_info.get("iterations"),
        inversion_residual=inv_info.get("relative_residual"),
        sample_points=len(omega),
        wind_speed=inp.wind_speed,
        fetch=inp.fetch,
    )


def calculate_moments(inp: SpectrumInput) -> MomentsResult:
    """矩专用入口：只回谱矩与统计周期，不生成采样列。"""
    alpha, moments, _ = _resolve_alpha(inp)
    tp = 2.0 * np.pi / inp.omega_p
    return MomentsResult(
        m0=float(moments.m0),
        m1=float(moments.m1),
        m2=float(moments.m2),
        hs=float(moments.hs),
        tp=float(tp),
        tz=float(moments.tz),
        t1=float(moments.t1),
        omega_p=inp.omega_p,
        gamma=inp.gamma,
        alpha=float(alpha),
        mode=inp.mode,
        omega_cutoff=float(moments.omega_cutoff),
        cutoff_factor=float(moments.cutoff_factor),
        converged=bool(moments.converged),
    )
