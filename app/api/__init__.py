"""HTTP 路由层。

对外只提供两件事：
1) POST /api/v1/spectrum          交入谱参数（或波高约束/风区），回谱采样+矩+统计
2) POST /api/v1/moments           只回谱矩与由矩导出的统计周期

另有工况档管理（/api/v1/cases...）与健康检查 / constants 查询。
"""

from __future__ import annotations

from fastapi import APIRouter, Query

from .. import constants as C
from ..cases import store
from ..errors import NotFoundError, ServiceError
from ..schemas import CaseSaveRequest, MomentsRequest, SpectrumRequest
from ..service import (
    calculate_moments,
    calculate_spectrum,
    resolve_input,
)

router = APIRouter(prefix="/api/v1")


@router.post("/spectrum", tags=["spectrum"])
def spectrum_endpoint(req: SpectrumRequest) -> dict:
    inp = resolve_input(
        omega_p=req.omega_p,
        gamma=req.gamma,
        alpha=req.alpha,
        hs=req.hs,
        wind_speed=req.wind_speed,
        fetch=req.fetch,
    )
    points = req.sample_points or C.SAMPLE_POINTS_DEFAULT
    result = calculate_spectrum(inp, sample_points=points)
    return {"data": result.to_dict()}


@router.post("/moments", tags=["spectrum"])
def moments_endpoint(req: MomentsRequest) -> dict:
    inp = resolve_input(
        omega_p=req.omega_p,
        gamma=req.gamma,
        alpha=req.alpha,
        hs=req.hs,
        wind_speed=req.wind_speed,
        fetch=req.fetch,
    )
    result = calculate_moments(inp)
    return {"data": result.to_dict()}


# --------------------------------------------------------------------- cases
@router.get("/cases", tags=["cases"])
def list_cases() -> dict:
    return {"data": {"cases": store.list_cases()}}


@router.get("/cases/{name}", tags=["cases"])
def get_case(name: str) -> dict:
    return {"data": store.get(name)}


@router.put("/cases/{name}", tags=["cases"])
def save_case(name: str, req: CaseSaveRequest) -> dict:
    if req.name != name:
        raise ServiceError("路径中的工况名与请求体不一致", field="name")
    params = req.params.model_dump(exclude_none=True)
    params.pop("sample_points", None)
    saved = store.save(name, params, description=req.description, overwrite=req.overwrite)
    return {"data": saved}


@router.delete("/cases/{name}", tags=["cases"])
def delete_case(name: str) -> dict:
    store.delete(name)
    return {"data": {"deleted": name}}


@router.post("/cases/{name}/spectrum", tags=["cases"])
def case_spectrum(
    name: str,
    sample_points: int | None = Query(default=None, ge=64, le=8192),
) -> dict:
    """凭名字取回复算：谱采样 + 矩。"""
    spec = store.params_of(name)
    inp = resolve_input(**spec)
    result = calculate_spectrum(inp, sample_points=sample_points or C.SAMPLE_POINTS_DEFAULT)
    body = result.to_dict()
    body["case"] = name
    return {"data": body}


@router.post("/cases/{name}/moments", tags=["cases"])
def case_moments(name: str) -> dict:
    """凭名字取回复算：仅矩与统计周期。"""
    spec = store.params_of(name)
    inp = resolve_input(**spec)
    result = calculate_moments(inp)
    body = result.to_dict()
    body["case"] = name
    return {"data": body}


# --------------------------------------------------------------------- infra
@router.get("/constants", tags=["infra"])
def get_constants() -> dict:
    return {
        "data": {
            **C.constants_snapshot(),
            "gamma_min": C.GAMMA_MIN,
            "omega_low_factor": C.OMEGA_LOW_FACTOR,
            "cutoff_schedule": list(C.CUTOFF_SCHEDULE),
            "log_step": C.LOG_STEP,
            "sample_low_factor": C.SAMPLE_LOW_FACTOR,
            "sample_high_factor": C.SAMPLE_HIGH_FACTOR,
        }
    }


@router.get("/health", tags=["infra"])
def health() -> dict:
    # 健康检查顺手确认自带算例可复算（只取参数，不触发计算，保持轻量）
    try:
        store.get("fetch-sea-15ms-50km")
        builtin_ok = True
    except NotFoundError:
        builtin_ok = False
    return {"data": {"status": "ok", "builtin_cases_available": builtin_ok}}
