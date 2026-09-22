"""谱参数入参校验。

峰频 ωp、峰升因子 γ、风速等量任一非正，或 γ<1，都在进入谱计算之前
挡下，抛出带原因的 ValidationError。校验独立成模块，不做任何谱计算。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from . import constants as C
from .errors import ValidationError


@dataclass(frozen=True)
class SpectrumInput:
    """归一化后的谱输入（校验通过后才能构造）。"""

    omega_p: float
    gamma: float
    alpha: float | None = None
    hs: float | None = None
    wind_speed: float | None = None
    fetch: float | None = None

    @property
    def mode(self) -> str:
        """alpha 来源：直接给定 / 波高约束反演 / 有限风区经验式。"""
        if self.alpha is not None:
            return "alpha"
        if self.hs is not None:
            return "hs_inversion"
        return "fetch_empirical"


def _finite_positive(value: float, field: str, *, allow_none: bool = False,
                     upper: float | None = None) -> None:
    if value is None:
        if allow_none:
            return
        raise ValidationError(f"{field} 为必填项", field=field)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValidationError(f"{field} 必须是数值", field=field)
    v = float(value)
    if not math.isfinite(v):
        raise ValidationError(f"{field} 必须是有限数值，收到 {value!r}", field=field)
    if v <= 0.0:
        raise ValidationError(
            f"{field} 必须为正数（非正入参不得进入谱计算），收到 {v:g}",
            field=field,
        )
    if upper is not None and v > upper:
        raise ValidationError(f"{field} 超出物理合理范围（>{upper:g}）：{v:g}", field=field)


def validate_spectrum_input(
    *,
    omega_p: float,
    gamma: float,
    alpha: float | None = None,
    hs: float | None = None,
    wind_speed: float | None = None,
    fetch: float | None = None,
) -> SpectrumInput:
    """校验并归一化谱输入。

    规则：
    - ωp、γ 必填且为正；γ<1 拒绝；γ==1 合法（退化为 PM）。
    - 风速 wind_speed 若给必须为正（为负/零拒绝）。
    - 风区 fetch 若给必须为正。
    - α 若直接给必须为正且在合理范围；α 优先于波高约束。
    - Hs 约束若给必须为正。
    - α、Hs、风区三者都没给时拒绝（没有任何尺度来源）。
    """

    _finite_positive(omega_p, "omega_p", upper=C.OMEGA_MAX)
    if float(omega_p) < C.OMEGA_MIN:
        raise ValidationError(
            f"omega_p 过小（<{C.OMEGA_MIN:g}），数值上无法稳定积分", field="omega_p"
        )

    # gamma 单独处理：允许恰好等于 1，但拒绝小于 1
    if isinstance(gamma, bool) or not isinstance(gamma, (int, float)):
        raise ValidationError("gamma 必须是数值", field="gamma")
    gv = float(gamma)
    if not math.isfinite(gv):
        raise ValidationError(f"gamma 必须是有限数值，收到 {gamma!r}", field="gamma")
    if gv <= 0.0:
        raise ValidationError(
            f"gamma 必须为正数，收到 {gv:g}（非正入参不得进入谱计算）", field="gamma"
        )
    if gv < C.GAMMA_MIN:
        raise ValidationError(
            f"gamma 不得小于 1（γ=1 退化为 PM 谱），收到 {gv:g}", field="gamma"
        )
    if gv > C.GAMMA_MAX:
        raise ValidationError(f"gamma 超出合理范围（>{C.GAMMA_MAX:g}）：{gv:g}", field="gamma")

    _finite_positive(alpha, "alpha", allow_none=True, upper=C.ALPHA_MAX)
    _finite_positive(hs, "hs", allow_none=True)
    _finite_positive(wind_speed, "wind_speed", allow_none=True, upper=C.WIND_SPEED_MAX)
    _finite_positive(fetch, "fetch", allow_none=True, upper=C.FETCH_MAX)

    if alpha is None and hs is None and fetch is None:
        raise ValidationError(
            "必须提供 alpha、有效波高 hs 或风区 fetch 三者之一，"
            "否则谱没有尺度来源",
            field="alpha",
        )

    return SpectrumInput(
        omega_p=float(omega_p),
        gamma=gv,
        alpha=None if alpha is None else float(alpha),
        hs=None if hs is None else float(hs),
        wind_speed=None if wind_speed is None else float(wind_speed),
        fetch=None if fetch is None else float(fetch),
    )


def validate_sample_points(points: int) -> int:
    if isinstance(points, bool) or not isinstance(points, int):
        raise ValidationError("sample_points 必须是整数", field="sample_points")
    if not (C.SAMPLE_POINTS_MIN <= points <= C.SAMPLE_POINTS_MAX):
        raise ValidationError(
            f"sample_points 必须落在 [{C.SAMPLE_POINTS_MIN}, {C.SAMPLE_POINTS_MAX}]，"
            f"收到 {points}",
            field="sample_points",
        )
    return points
