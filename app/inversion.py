"""Wave-height constraint inversion: solve alpha so that 4*sqrt(m0) == Hs.

When the caller specifies a target significant wave height instead of the
Phillips scale factor alpha, the service iteratively adjusts alpha until
the integrated m0 satisfies Hs = 4*sqrt(m0) within the set tolerance.
Because S(omega) is linear in alpha, m0 scales linearly and the update
alpha <- alpha * (Hs_target / Hs)^2 converges very quickly; the loop still
verifies the residual every iteration and refuses to return an alpha whose
integrated Hs does not close to the target.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .constants import (
    ALPHA_INITIAL,
    DEFAULT_N_POINTS,
    INVERSION_MAX_ITER,
    INVERSION_REL_TOL,
)
from .moments import omega_grid, spectral_moments
from .spectrum import jonswap


class InversionError(RuntimeError):
    """Raised when the alpha iteration fails to converge."""

    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(reason)


@dataclass(frozen=True)
class InversionResult:
    alpha: float
    hs_achieved: float
    residual: float      # hs_achieved - hs_target
    iterations: int
    converged: bool


def solve_alpha_for_hs(
    hs_target: float,
    omega_p: float,
    gamma: float,
    n_points: int = DEFAULT_N_POINTS,
    rel_tol: float = INVERSION_REL_TOL,
    max_iter: int = INVERSION_MAX_ITER,
) -> InversionResult:
    """Iteratively adjust alpha until 4*sqrt(m0(alpha)) closes to hs_target."""
    omega = omega_grid(omega_p, n_points)

    def hs_of(alpha: float) -> float:
        moments = spectral_moments(omega, jonswap(omega, omega_p, alpha, gamma))
        return 4.0 * math.sqrt(moments.m0)

    alpha = ALPHA_INITIAL
    for iteration in range(1, max_iter + 1):
        hs = hs_of(alpha)
        residual = hs - hs_target
        if abs(residual) <= rel_tol * hs_target:
            return InversionResult(
                alpha=alpha,
                hs_achieved=hs,
                residual=residual,
                iterations=iteration,
                converged=True,
            )
        # m0 is linear in alpha, so Hs scales with sqrt(alpha).
        alpha *= (hs_target / hs) ** 2
    raise InversionError(
        f"alpha inversion did not converge within {max_iter} iterations "
        f"(hs_target={hs_target}, omega_p={omega_p}, gamma={gamma})"
    )
