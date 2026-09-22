# JONSWAP Spectrum Service

JONSWAP 频谱的正算与波高约束反演后端。只做谱核算，仅经 HTTP 对外服务，无网页界面。

## 谱形定义（Hasselmann 标准式）

```
S(ω) = α·g²·ω⁻⁵·exp(−1.25·(ωp/ω)⁴)·γ^r
r = exp(−(ω−ωp)² / (2σ²ωp²))
σ = 0.07 (ω ≤ ωp),  σ = 0.09 (ω > ωp),  g = 9.81 m/s²
```

常数集中在 `app/constants.py`，全服务引用同一份。谱矩由梯形数值积分得到
（m0/m1/m2），积分上限取 10·ωp 并随网格加密收敛；派生量 Hs = 4√m0、
Tp = 2π/ωp、Tz = 2π√(m0/m2)、T01 = 2π·m0/m1。

## 模块划分

| 模块 | 职责 |
|---|---|
| `app/constants.py` | 物理常数与数值默认值（唯一来源） |
| `app/spectrum.py` | 谱形正算与入参物理校验 |
| `app/moments.py` | 频率网格、数值积分、谱矩与派生统计量 |
| `app/inversion.py` | 波高约束反演：迭代调 α 直至 4√m0 闭合到 Hs |
| `app/cases.py` | 工况档管理，JSON 持久化到容器内（`CASES_PATH`） |
| `app/service.py` | 无状态编排（校验→反演→采样→积分） |
| `app/schemas.py` / `app/main.py` | 载荷模型与 HTTP 路由 |

## 接口

- `POST /spectrum` — 交入 `{omega_p, gamma, alpha}` 或 `{omega_p, gamma, hs_target}`
  （可选 `wind_speed`、`fetch`、`n_points`），返回谱采样点列、m0–m2、Hs、Tp、Tz、T01。
- `POST /moments` — 同样入参，只回谱矩与统计周期。
- `POST /cases` / `GET /cases` / `GET /cases/{name}` / `GET /cases/{name}/spectrum`
  / `DELETE /cases/{name}` — 工况建档、查询与复算。
- `GET /cases/demo_fetch_limited/spectrum` — 内置有限风区算例（γ=3.3，Hs=4.0 m），
  服务启动即可用于核对。
- 非法输入（ωp≤0、γ<1、风速非正、α 与 hs_target 同时给或都不给等）返回
  `400 {"error","parameter","value","reason"}`。

## 构建与运行

```bash
docker build -t jonswap-service .   # 构建期间自动执行 pytest，测试不过则构建失败
docker run -p 8000:8000 jonswap-service
```

本地开发：`pip install -r requirements.txt && python -m pytest tests -q`，
然后 `uvicorn app.main:app`。

## 测试覆盖（`tests/`，31 项）

- 谱形正算：γ 增大峰更高更尖；γ=1 退化为 Pierson–Moskowitz；ωp 减半能量迁移、
  Tp 加倍；高频尾巴斜率钉在 ω⁻⁵（防 ⁻⁵/⁻⁴ 写反翘尾）；采样点逐点对照公式。
- 矩积分：网格加密收敛（1024→16384）；截断检验（延长积分上限 m0 变化 <1‰）。
- 波高反演：4√m0 与目标 Hs 闭合到容差内；Hs 加倍 → α 与 m0 变四倍；
  返回谱线独立重积分复现声称 Hs。
- API：非法参数拦截、工况档往返、预置算例自洽、并行工况互不串扰。
