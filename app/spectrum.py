"""JONSWAP spectrum shape (forward evaluation) and input validation.

Spectrum definition (Hasselmann et al.):

    S(omega) = alpha * g^2 * omega^-5
               * exp(-1.25 * (omega_p / omega)^4)
               * gamma^r
    r = exp(-(omega - omega_p)^2 / (2 * sigma^2 * omega_p^2))

with sigma = SIGMA_LEFT for omega <= omega_p and SIGMA_RIGHT above it.
Every sample point is computed from this closed-form expression; nothing
about the curve shape is hard-coded.
"""

from __future__ import annotations

import math

import numpy as np

from .constants import GRAVITY, MIN_GAMMA, SIGMA_LEFT, SIGMA_RIGHT


class ParameterError(ValueError):
    """Raised when a spectrum parameter fails physical validation.

    Carries the offending parameter name, its value and a human-readable
    reason so the HTTP layer can return a structured error body.
    """

    def __init__(self, parameter: str, value, reason: str):
        self.parameter = parameter
        self.value = value
        self.reason = reason
        super().__init__(f"invalid parameter {parameter!r} = {value!r}: {reason}")


def _require_finite(name: str, value: float) -> None:
    if not math.isfinite(value):
        raise ParameterError(name, value, "value must be a finite number")


def _require_positive(name: str, value: float) -> None:
    _require_finite(name, value)
    if value <= 0.0:
        raise ParameterError(name, value, "value must be strictly positive")


def validate_spectrum_inputs(
    omega_p: float,
    gamma: float,
    alpha: float | None = None,
    hs_target: float | None = None,
    wind_speed: float | None = None,
    fetch: float | None = None,
    n_points: int | None = None,
) -> None:
    """Reject physically invalid inputs before any spectrum computation.

    omega_p, gamma, wind_speed (and alpha / hs_target / fetch when given)
    must be positive; gamma must be >= 1; exactly one of alpha / hs_target
    selects the scaling mode.
    """
    _require_positive("omega_p", omega_p)
    _require_finite("gamma", gamma)
    if gamma < MIN_GAMMA:
        raise ParameterError(
            "gamma", gamma, f"peak enhancement factor must be >= {MIN_GAMMA}"
        )
    if alpha is not None:
        _require_positive("alpha", alpha)
    if hs_target is not None:
        _require_positive("hs_target", hs_target)
    if wind_speed is not None:
        _require_positive("wind_speed", wind_speed)
    if fetch is not None:
        _require_positive("fetch", fetch)
    if (alpha is None) == (hs_target is None):
        raise ParameterError(
            "alpha/hs_target",
            {"alpha": alpha, "hs_target": hs_target},
            "exactly one of 'alpha' or 'hs_target' must be provided",
        )
    if n_points is not None:
        from .constants import MAX_N_POINTS, MIN_N_POINTS

        if not (MIN_N_POINTS <= n_points <= MAX_N_POINTS):
            raise ParameterError(
                "n_points",
                n_points,
                f"grid size must be within [{MIN_N_POINTS}, {MAX_N_POINTS}]",
            )


def peak_enhancement(omega, omega_p: float, gamma: float) -> np.ndarray:
    """Return gamma^r with r = exp(-(omega-omega_p)^2 / (2 sigma^2 omega_p^2)).

    sigma switches between SIGMA_LEFT (omega <= omega_p) and
    SIGMA_RIGHT (omega > omega_p).
    """
    omega = np.asarray(omega, dtype=float)
    sigma = np.where(omega <= omega_p, SIGMA_LEFT, SIGMA_RIGHT)
    r = np.exp(-((omega - omega_p) ** 2) / (2.0 * (sigma * omega_p) ** 2))
    return np.power(float(gamma), r)


def pierson_moskowitz(omega, omega_p: float, alpha: float) -> np.ndarray:
    """Pierson-Moskowitz spectrum (the gamma == 1 degenerate case)."""
    omega = np.asarray(omega, dtype=float)
    return alpha * GRAVITY**2 * omega**-5 * np.exp(-1.25 * (omega_p / omega) ** 4)


def jonswap(omega, omega_p: float, alpha: float, gamma: float) -> np.ndarray:
    """Evaluate the JONSWAP spectrum at the given angular frequencies."""
    omega = np.asarray(omega, dtype=float)
    if omega.size == 0 or np.any(omega <= 0.0):
        raise ParameterError(
            "omega", None, "spectrum samples require strictly positive frequencies"
        )
    return pierson_moskowitz(omega, omega_p, alpha) * peak_enhancement(
        omega, omega_p, gamma
    )
