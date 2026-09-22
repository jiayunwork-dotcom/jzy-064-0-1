"""Frequency grid, numerical integration and spectral moments.

Moments are obtained by trapezoidal integration of the sampled spectrum:

    m0 = ∫ S(ω) dω,   m1 = ∫ ω S(ω) dω,   m2 = ∫ ω² S(ω) dω

The grid reaches OMEGA_MAX_FACTOR * omega_p so the omega**-5 tail is
captured; refining the grid must leave the moments stable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .constants import DEFAULT_N_POINTS, OMEGA_MAX_FACTOR, OMEGA_MIN_FACTOR


def omega_grid(omega_p: float, n_points: int = DEFAULT_N_POINTS) -> np.ndarray:
    """Linear frequency grid from OMEGA_MIN_FACTOR*omega_p to
    OMEGA_MAX_FACTOR*omega_p (both as multiples of the peak frequency)."""
    return np.linspace(
        OMEGA_MIN_FACTOR * omega_p, OMEGA_MAX_FACTOR * omega_p, int(n_points)
    )


@dataclass(frozen=True)
class SpectralMoments:
    m0: float
    m1: float
    m2: float


def spectral_moments(omega: np.ndarray, s: np.ndarray) -> SpectralMoments:
    """Trapezoidal integration of the 0th, 1st and 2nd spectral moments."""
    omega = np.asarray(omega, dtype=float)
    s = np.asarray(s, dtype=float)
    return SpectralMoments(
        m0=float(np.trapezoid(s, omega)),
        m1=float(np.trapezoid(omega * s, omega)),
        m2=float(np.trapezoid(omega**2 * s, omega)),
    )


@dataclass(frozen=True)
class DerivedStatistics:
    hs: float    # significant wave height, 4*sqrt(m0)
    tp: float    # peak period, 2*pi/omega_p
    tz: float    # mean zero-crossing period, 2*pi*sqrt(m0/m2)
    t01: float   # mean wave period, 2*pi*m0/m1


def derived_statistics(
    moments: SpectralMoments, omega_p: float
) -> DerivedStatistics:
    return DerivedStatistics(
        hs=4.0 * math.sqrt(moments.m0),
        tp=2.0 * math.pi / omega_p,
        tz=2.0 * math.pi * math.sqrt(moments.m0 / moments.m2),
        t01=2.0 * math.pi * moments.m0 / moments.m1,
    )
