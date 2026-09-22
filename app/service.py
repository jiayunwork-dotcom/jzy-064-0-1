"""Orchestration glue: turn validated inputs into a full spectrum result.

This module only wires the independent pieces together (validation ->
inversion if needed -> spectrum sampling -> moment integration -> derived
statistics). It holds no physics of its own and keeps no state, so
concurrently submitted cases can never leak intermediate quantities into
each other.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .constants import DEFAULT_N_POINTS
from .inversion import InversionResult, solve_alpha_for_hs
from .moments import (
    DerivedStatistics,
    SpectralMoments,
    derived_statistics,
    omega_grid,
    spectral_moments,
)
from .spectrum import jonswap, validate_spectrum_inputs


@dataclass(frozen=True)
class SpectrumResult:
    omega: np.ndarray
    s: np.ndarray
    omega_p: float
    gamma: float
    alpha: float
    moments: SpectralMoments
    stats: DerivedStatistics
    inversion: InversionResult | None


def compute_spectrum(
    omega_p: float,
    gamma: float,
    alpha: float | None = None,
    hs_target: float | None = None,
    wind_speed: float | None = None,
    fetch: float | None = None,
    n_points: int | None = None,
) -> SpectrumResult:
    """Compute a self-consistent JONSWAP spectrum and its statistics.

    Exactly one of ``alpha`` / ``hs_target`` must be given. With
    ``hs_target`` the scale factor is inverted so that the integrated
    spectrum satisfies Hs = 4*sqrt(m0).
    """
    n = n_points if n_points is not None else DEFAULT_N_POINTS
    validate_spectrum_inputs(
        omega_p=omega_p,
        gamma=gamma,
        alpha=alpha,
        hs_target=hs_target,
        wind_speed=wind_speed,
        fetch=fetch,
        n_points=n,
    )

    inversion: InversionResult | None = None
    if hs_target is not None:
        inversion = solve_alpha_for_hs(hs_target, omega_p, gamma, n_points=n)
        alpha = inversion.alpha
    assert alpha is not None  # guaranteed by validate_spectrum_inputs

    omega = omega_grid(omega_p, n)
    s = jonswap(omega, omega_p, alpha, gamma)
    moments = spectral_moments(omega, s)
    stats = derived_statistics(moments, omega_p)
    return SpectrumResult(
        omega=omega,
        s=s,
        omega_p=omega_p,
        gamma=gamma,
        alpha=alpha,
        moments=moments,
        stats=stats,
        inversion=inversion,
    )
