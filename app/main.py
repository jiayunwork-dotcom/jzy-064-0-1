"""FastAPI 应用入口：仅 HTTP 服务，不带任何网页界面。"""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from .api import router
from .cases import store
from .errors import ServiceError

app = FastAPI(
    title="JONSWAP 频谱核算与反演后端",
    version="1.0.0",
    description=(
        "只做 JONSWAP 频谱的正算、谱矩数值积分与有效波高约束反演，"
        "经 HTTP 提供 JSON 接口；无网页界面，与气象/航线等应用无关。"
    ),
    docs_url="/docs",
    redoc_url=None,
    openapi_url="/openapi.json",
)
app.include_router(router)


@app.exception_handler(ServiceError)
async def service_error_handler(request: Request, exc: ServiceError) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code, content=exc.to_payload())


@app.exception_handler(RequestValidationError)
async def request_validation_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    # 统一成与领域错误一致的结构
    first = exc.errors()[0] if exc.errors() else {}
    loc = ".".join(str(p) for p in first.get("loc", []) if p not in ("body", "query"))
    return JSONResponse(
        status_code=422,
        content={
            "error": {
                "code": "invalid_parameter",
                "message": first.get("msg", "请求参数不合法"),
                "field": loc or None,
                "detail": exc.errors(),
            }
        },
    )


@app.on_event("startup")
def _ensure_builtin_cases() -> None:
    # 任何方式拉起服务（包括持久化文件丢失）都保证自带算例就位
    store.reset_builtins()
