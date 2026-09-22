"""JONSWAP 频谱计算后端。

模块划分：
- constants: 全服务唯一一份物理常数与收敛参数（单一事实源）
- spectrum:  JONSWAP / PM 谱形核算
- moments:   数值积分与谱矩 m0/m1/m2（独立于反演）
- inversion: 有效波高约束下的 alpha 反演
- fetch:     有限风区经验式（Hasselmann 1973）
- validation / errors: 入参校验与统一错误结构
- service:   谱正算、矩与统计周期的编排层
- cases:     工况档持久化管理
- api/routes: HTTP 路由
"""
