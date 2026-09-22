"""全服务唯一一份物理常数与数值参数（single source of truth）。

Hasselmann 标准 JONSWAP 谱：

    S(ω) = α·g²·ω⁻⁵·exp(−1.25·(ωp/ω)⁴)·γ^r
    r    = exp(−(ω−ωp)² / (2·σ²·ωp²))

谱峰左侧 σ=SIGMA_LEFT，右侧 σ=SIGMA_RIGHT；这两个常数与重力加速度 g
在整个服务里钉死在此处，所有模块都从这里引用，禁止任何接口改写。
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# 钉死的物理常数
# ---------------------------------------------------------------------------
GRAVITY: float = 9.80665  # 重力加速度 g [m/s²]
SIGMA_LEFT: float = 0.07  # ω ≤ ωp 一侧的谱形宽度常数 σ_a
SIGMA_RIGHT: float = 0.09  # ω > ωp 一侧的谱形宽度常数 σ_b

# 谱形指数常数（同样钉死，避免 ⁻⁵ / ⁻⁴ 被改反）
EXP_FETCH_COEFF: float = 1.25  # exp(−1.25·(ωp/ω)^4) 里的系数

# ---------------------------------------------------------------------------
# 校验边界
# ---------------------------------------------------------------------------
GAMMA_MIN: float = 1.0  # γ 不得小于 1；γ == 1 退化为 PM 谱
GAMMA_MAX: float = 100.0  # 拒绝数值上无意义的超大 γ

SAMPLE_POINTS_MIN: int = 64
SAMPLE_POINTS_MAX: int = 8192

# 风速上限仅用于拦截非物理入参（单位 m/s，~250 kn）
WIND_SPEED_MAX: float = 200.0
# 峰频单位 rad/s，覆盖极端长涌到极小尺度
OMEGA_MIN: float = 1.0e-4
OMEGA_MAX: float = 1.0e3
FETCH_MAX: float = 1.0e7  # m，~10000 km，超此视为误填
ALPHA_MAX: float = 1.0

# ---------------------------------------------------------------------------
# 数值积分参数：在 log(ω) 均匀网格上用 Simpson 公式积分
#
# 上限随网格加密向外扩展，直到 m0/m1/m2 相对变化全部落在
# INTEGRATION_REL_TOL 以内，杜绝“积到一半截断”导致 m0 偏小。
# m2 的高频尾为 O(ω⁻⁵)（log 积分尺度下），要到 ω/ωp≈4096 才衰减到
# 1e-7 量级，因此倍率表必须足够长。
# ---------------------------------------------------------------------------
OMEGA_LOW_FACTOR: float = 0.25  # 积分下限 = ωp · 0.25
CUTOFF_SCHEDULE: tuple[float, ...] = (
    4.0,
    8.0,
    16.0,
    64.0,
    256.0,
    1024.0,
    4096.0,
    16384.0,
    65536.0,
)
LOG_STEP: float = 2.0e-3  # log(ω) 网格步长（h⁴ 精度，兼顾速度与 1e-7 容差）
INTEGRATION_REL_TOL: float = 1.0e-7  # 矩积分相对收敛容差

# 返回给调用方的谱采样（不参与积分，仅用于展示/复核）
SAMPLE_LOW_FACTOR: float = 1.0 / 3.0
SAMPLE_HIGH_FACTOR: float = 8.0
SAMPLE_POINTS_DEFAULT: int = 1024

# ---------------------------------------------------------------------------
# 波高反演参数：定点迭代 α_{n+1} = α_n·Hs_target²/(16·m0(α_n))
# ---------------------------------------------------------------------------
INVERSION_REL_TOL: float = 1.0e-7  # |4√m0 − Hs|/Hs 闭合容差
INVERSION_MAX_ITER: int = 50

# ---------------------------------------------------------------------------
# 派生统计量
# ---------------------------------------------------------------------------
EPS_SMALL: float = 1.0e-12


def constants_snapshot() -> dict[str, float]:
    """对外暴露当前生效的常数，便于调用方核对“各处引用同一套”。"""
    return {
        "g": GRAVITY,
        "sigma_left": SIGMA_LEFT,
        "sigma_right": SIGMA_RIGHT,
        "exp_coeff": EXP_FETCH_COEFF,
        "integration_rel_tol": INTEGRATION_REL_TOL,
        "inversion_rel_tol": INVERSION_REL_TOL,
    }
