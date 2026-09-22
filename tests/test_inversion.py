"""Wave-height inversion tests: closure of Hs = 4*sqrt(m0), scaling of
alpha and m0 with the constraint, and agreement between the reported Hs
and the area under the returned spectrum."""

import math

import numpy as np
import pytest

from app.constants import INVERSION_REL_TOL
from app.inversion import solve_alpha_for_hs
from app.moments import spectral_moments
from app.service import compute_spectrum

OMEGA_P = 0.5
GAMMA = 3.3


def test_inversion_closes_within_tolerance():
    hs_target = 3.5
    result = compute_spectrum(omega_p=OMEGA_P, gamma=GAMMA, hs_target=hs_target)

    assert result.inversion is not None and result.inversion.converged
    assert abs(result.stats.hs - hs_target) <= 1e-6 * hs_target
    # The returned moments are the ones that close the constraint.
    assert 4.0 * math.sqrt(result.moments.m0) == pytest.approx(hs_target, rel=1e-9)


def test_doubling_hs_quadruples_m0_and_alpha():
    r1 = compute_spectrum(omega_p=OMEGA_P, gamma=GAMMA, hs_target=2.0)
    r2 = compute_spectrum(omega_p=OMEGA_P, gamma=GAMMA, hs_target=4.0)

    assert r2.moments.m0 == pytest.approx(4.0 * r1.moments.m0, rel=1e-9)
    assert r2.alpha == pytest.approx(4.0 * r1.alpha, rel=1e-9)


def test_reported_spectrum_integrates_back_to_target_hs():
    """Re-integrate the returned sample points independently: the area must
    reproduce the claimed Hs (no nominal Hs with an un-inverted alpha)."""
    hs_target = 5.0
    result = compute_spectrum(omega_p=0.4, gamma=2.5, hs_target=hs_target)

    moments = spectral_moments(result.omega, result.s)
    hs_from_area = 4.0 * math.sqrt(moments.m0)
    assert hs_from_area == pytest.approx(hs_target, rel=1e-6)
    assert result.stats.hs == pytest.approx(hs_from_area, rel=1e-12)


def test_inversion_iteration_count_and_residual():
    res = solve_alpha_for_hs(4.0, OMEGA_P, GAMMA)
    assert res.converged
    assert res.iterations <= 5  # linear in alpha: converges almost immediately
    assert abs(res.residual) <= INVERSION_REL_TOL * 4.0


def test_inversion_respects_gamma_and_omega_p():
    """Different (omega_p, gamma) pairs need different alphas for the same
    Hs, and each must close individually."""
    for omega_p, gamma in [(0.4, 1.0), (0.6, 3.3), (0.8, 6.0)]:
        result = compute_spectrum(omega_p=omega_p, gamma=gamma, hs_target=3.0)
        assert result.stats.hs == pytest.approx(3.0, rel=1e-9)
