"""非法参数拦截测试：非正 ωp、负风速、γ<1 等必须在计算前挡下。"""

from __future__ import annotations

import pytest

from app.errors import ValidationError
from app.validation import validate_spectrum_input


def test_zero_omega_p_rejected():
    with pytest.raises(ValidationError) as ei:
        validate_spectrum_input(omega_p=0.0, gamma=3.3, alpha=0.01)
    assert ei.value.field == "omega_p"
    assert "omega_p" in ei.value.message


def test_negative_omega_p_rejected():
    with pytest.raises(ValidationError) as ei:
        validate_spectrum_input(omega_p=-0.7, gamma=3.3, alpha=0.01)
    assert ei.value.field == "omega_p"


def test_negative_wind_speed_rejected():
    with pytest.raises(ValidationError) as ei:
        validate_spectrum_input(
            omega_p=0.7, gamma=3.3, alpha=0.01, wind_speed=-12.0
        )
    assert ei.value.field == "wind_speed"


def test_zero_wind_speed_rejected():
    with pytest.raises(ValidationError):
        validate_spectrum_input(
            omega_p=0.7, gamma=3.3, alpha=0.01, wind_speed=0.0
        )


def test_gamma_below_one_rejected():
    with pytest.raises(ValidationError) as ei:
        validate_spectrum_input(omega_p=0.7, gamma=0.9, alpha=0.01)
    assert ei.value.field == "gamma"
    assert "1" in ei.value.message


def test_nonpositive_gamma_rejected():
    for bad in (0.0, -2.0):
        with pytest.raises(ValidationError):
            validate_spectrum_input(omega_p=0.7, gamma=bad, alpha=0.01)


def test_gamma_equal_one_accepted():
    inp = validate_spectrum_input(omega_p=0.7, gamma=1.0, alpha=0.01)
    assert inp.gamma == 1.0


def test_nonpositive_alpha_and_hs_rejected():
    with pytest.raises(ValidationError):
        validate_spectrum_input(omega_p=0.7, gamma=3.3, alpha=-0.01)
    with pytest.raises(ValidationError):
        validate_spectrum_input(omega_p=0.7, gamma=3.3, hs=0.0)


def test_missing_scale_source_rejected():
    """ωp、γ 给了但 α/hs/fetch 全缺：没有尺度来源，拒绝。"""
    with pytest.raises(ValidationError) as ei:
        validate_spectrum_input(omega_p=0.7, gamma=3.3)
    assert "alpha" in ei.value.message or "hs" in ei.value.message


def test_nan_inf_rejected():
    with pytest.raises(ValidationError):
        validate_spectrum_input(omega_p=float("nan"), gamma=3.3, alpha=0.01)
    with pytest.raises(ValidationError):
        validate_spectrum_input(
            omega_p=float("inf"), gamma=3.3, alpha=0.01
        )


def test_negative_fetch_rejected():
    with pytest.raises(ValidationError):
        validate_spectrum_input(
            omega_p=0.7, gamma=3.3, wind_speed=15.0, fetch=-1000.0
        )
