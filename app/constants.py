"""Physical constants and numerical defaults for the whole service.

Every module (spectrum shape, moment integration, inversion, HTTP layer)
must import these values from here instead of defining its own copies, so
that one single consistent set of constants is used everywhere.
"""

# Gravity acceleration [m/s^2].
GRAVITY = 9.81

# JONSWAP peak-shape widths (Hasselmann et al., 1973):
# sigma takes different constant values on the two sides of the peak.
SIGMA_LEFT = 0.07   # for omega <= omega_p
SIGMA_RIGHT = 0.09  # for omega > omega_p

# The peak enhancement factor gamma must be >= 1 (gamma == 1 degenerates
# to the Pierson-Moskowitz spectrum).
MIN_GAMMA = 1.0

# Frequency grid defaults. The grid starts slightly above zero (the spectrum
# vanishes at omega = 0 anyway) and extends to a multiple of the peak
# frequency so the omega**-5 tail is captured and the spectral moments
# converge under grid refinement instead of being truncated halfway.
OMEGA_MIN_FACTOR = 0.05   # grid start, as a fraction of omega_p
OMEGA_MAX_FACTOR = 10.0   # integration cut-off, as a multiple of omega_p
DEFAULT_N_POINTS = 4096
MIN_N_POINTS = 64
MAX_N_POINTS = 200_000

# Wave-height inversion defaults.
INVERSION_REL_TOL = 1e-10
INVERSION_MAX_ITER = 100
ALPHA_INITIAL = 0.0081  # classical fetch-limited value, used as first guess
