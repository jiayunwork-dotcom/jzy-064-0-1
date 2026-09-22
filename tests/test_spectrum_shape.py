"""Forward spectrum shape tests: gamma effect, PM degeneracy, peak shift,
and the high-frequency omega**-5 tail (guarding against a -5/-4 exponent
swap that would make the spectrum blow up at high frequency)."""

import numpy as np
import pytest

from app.constants import GRAVITY, SIGMA_LEFT, SIGMA_RIGHT
from app.moments import omega_grid
from app.spectrum import jonswap, pierson_moskowitz

ALPHA = 0.0081
OMEGA_P = 0.6


def test_gamma_raises_and_sharpens_peak():
    omega = omega_grid(OMEGA_P)
    s1 = jonswap(omega, OMEGA_P, ALPHA, gamma=1.0)
    s5 = jonswap(omega, OMEGA_P, ALPHA, gamma=5.0)

    # Same alpha: larger gamma -> higher peak.
    assert s5.max() > s1.max()

    # Larger gamma -> sharper peak (narrower width at half maximum).
    def half_max_width(s):
        above = omega[s > 0.5 * s.max()]
        return above[-1] - above[0]

    assert half_max_width(s5) < half_max_width(s1)

    # Away from the peak the gamma^r factor tends to 1: tails coincide.
    tail = omega > 4 * OMEGA_P
    np.testing.assert_allclose(s5[tail], s1[tail], rtol=1e-6, atol=1e-30)


def test_gamma_one_degenerates_to_pierson_moskowitz():
    omega = omega_grid(OMEGA_P)
    s = jonswap(omega, OMEGA_P, ALPHA, gamma=1.0)
    pm = pierson_moskowitz(omega, OMEGA_P, ALPHA)
    np.testing.assert_allclose(s, pm, rtol=1e-14, atol=0.0)
    # And the PM reference itself is the analytic expression.
    analytic = ALPHA * GRAVITY**2 * omega**-5 * np.exp(-1.25 * (OMEGA_P / omega) ** 4)
    np.testing.assert_allclose(s, analytic, rtol=1e-14, atol=0.0)


def test_halving_omega_p_shifts_energy_and_doubles_tp():
    omega1 = omega_grid(OMEGA_P)
    omega2 = omega_grid(OMEGA_P / 2)
    s1 = jonswap(omega1, OMEGA_P, ALPHA, gamma=3.3)
    s2 = jonswap(omega2, OMEGA_P / 2, ALPHA, gamma=3.3)

    peak1 = omega1[np.argmax(s1)]
    peak2 = omega2[np.argmax(s2)]
    assert peak2 == pytest.approx(peak1 / 2, rel=1e-2)

    tp1 = 2 * np.pi / OMEGA_P
    tp2 = 2 * np.pi / (OMEGA_P / 2)
    assert tp2 == pytest.approx(2 * tp1)


def test_peak_sits_at_omega_p():
    omega = omega_grid(OMEGA_P)
    s = jonswap(omega, OMEGA_P, ALPHA, gamma=3.3)
    step = omega[1] - omega[0]
    assert abs(omega[np.argmax(s)] - OMEGA_P) <= step


def test_high_frequency_tail_follows_omega_minus_5():
    """At omega >> omega_p the exponential and gamma^r factors tend to 1,
    so S ~ alpha*g^2*omega^-5. A -5/-4 exponent swap would show up here
    as a clearly wrong log-log slope (and an upturned tail)."""
    omega = np.array([6.0, 8.0, 10.0]) * OMEGA_P
    s = jonswap(omega, OMEGA_P, ALPHA, gamma=3.3)

    slopes = np.diff(np.log(s)) / np.diff(np.log(omega))
    np.testing.assert_allclose(slopes, -5.0, atol=0.02)

    # Monotonically decaying tail: no high-frequency upturn.
    dense = omega_grid(OMEGA_P)
    s_dense = jonswap(dense, OMEGA_P, ALPHA, gamma=3.3)
    tail = dense >= 2 * OMEGA_P
    assert np.all(np.diff(s_dense[tail]) < 0)


def test_every_sample_comes_from_the_formula():
    """Spot-check samples against an independent re-implementation of the
    Hasselmann form, including the left/right sigma switch."""
    rng = np.random.default_rng(7)
    omega = rng.uniform(0.05, 10.0, 200) * OMEGA_P
    gamma = 4.2
    s = jonswap(omega, OMEGA_P, ALPHA, gamma)

    sigma = np.where(omega <= OMEGA_P, SIGMA_LEFT, SIGMA_RIGHT)
    r = np.exp(-((omega - OMEGA_P) ** 2) / (2 * (sigma * OMEGA_P) ** 2))
    expected = (
        ALPHA
        * GRAVITY**2
        * omega**-5
        * np.exp(-1.25 * (OMEGA_P / omega) ** 4)
        * gamma**r
    )
    np.testing.assert_allclose(s, expected, rtol=1e-14)
