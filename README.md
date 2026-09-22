# JONSWAP 频谱核算与反演后端

常驻 HTTP 计算后端：把**风区 / 峰频 / 有效波高约束**翻译成一条自洽的
JONSWAP 频谱曲线及其谱矩统计，供波浪载荷预报工具调用。仅提供 HTTP JSON
接口，无网页界面，与气象预报、航线规划等应用无关。

## 谱形（Hasselmann 标准式，常数全服务钉死一份）

```
S(ω) = α·g²·ω⁻⁵·exp(−1.25·(ωp/ω)⁴)·γ^r
r    = exp(−(ω−ωp)²/(2σ²ωp²))
```

- ω ≤ ωp：σ = 0.07；ω > ωp：σ = 0.09；g = 9.80665 —— 统一在
  `app/constants.py`，任何接口不得改写。
- γ = 1 时 γ^r ≡ 1，谱形严格退化为 Pierson–Moskowitz 谱。
- ωp、γ、风速等任一非正、或 γ < 1，在进入谱计算前即被拒绝，返回
  `{"error": {"code", "message", "field"}}`。

## 反演路径（波高约束）

不给 α、改为声明 `hs` 时，服务迭代调整 α，在收敛网格上反复积分 m0，
直到 `|4√m0 − hs| / hs < 1e-7`；返回的 Hs **始终**由 `4√m0` 实算，
与谱面积严格闭合（测试焊死，不允许只写标称波高）。

谱矩 m0/m1/m2 在 log(ω) 均匀网格上用 Simpson 公式积分，上限按
4→8→…→65536 倍 ωp 逐档外扩，三矩相对变化均 < 1e-7 才停止，
不会提前截断导致 m0 偏小。派生量：Hs = 4√m0、Tp = 2π/ωp、
Tz = 2π√(m0/m2)（另附 T1 = 2πm0/m1）。

## 模块划分

| 模块 | 职责 |
|---|---|
| `app/constants.py` | 钉死的物理常数与积分/反演容差（单一事实源） |
| `app/spectrum.py` | JONSWAP / PM 谱形逐点核算 |
| `app/moments.py` | 数值积分与谱矩（独立于反演） |
| `app/inversion.py` | Hs 约束下 α 反演 |
| `app/fetch.py` | 有限风区经验式（Hasselmann 1973） |
| `app/validation.py` | 入参校验（计算前拦截） |
| `app/service.py` | 正算/反演/矩编排，全部纯函数式、并行互不串档 |
| `app/cases.py` | 工况档 JSON 持久化（线程锁 + 原子写） |
| `app/api/__init__.py` | HTTP 路由 |
| `app/main.py` | FastAPI 应用与统一错误结构 |

## 接口

- `POST /api/v1/spectrum` — 入谱参数/波高约束/风区，返回谱采样列、
  m0~m2、Hs/Tp/Tz/T1、积分与反演诊断。
- `POST /api/v1/moments` — 只返回谱矩与统计周期。
- `GET|PUT|DELETE /api/v1/cases[/{name}]`、
  `POST /api/v1/cases/{name}/spectrum|moments` — 工况建档与凭名字复算。
- `GET /api/v1/constants`、`GET /api/v1/health`。

请求体示例：

```json
{"omega_p": 0.62, "gamma": 3.3, "hs": 5.0}
{"wind_speed": 15.0, "fetch": 50000}
```

服务自带三个有限风区算例（`fetch-coastal-15ms-25km` 等，γ=3.3>1，
Hs 与 4√m0 对得上），拉起即可核对。

## 构建与运行（测试随构建执行）

```bash
docker build -t jonswap-backend .        # 构建时自动跑全部 pytest
docker run -p 8000:8000 jonswap-backend  # 一启动即对外服务
```

本地（Python 3.12）：

```bash
pip install -r requirements.txt
pytest            # 正算 / 积分收敛 / 反演闭合 / γ=1 退化 / 非法拦截
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

工况档默认持久化在容器内 `/srv/data/cases.json`（环境变量
`JONSWAP_DATA_DIR` 可覆盖目录）。
