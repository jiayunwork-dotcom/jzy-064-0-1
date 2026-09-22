"""统一的领域错误与错误结构。

所有被拦截的入参 / 未找到的工况等，都抛出 ``ServiceError``，
由路由层翻译成统一 JSON：

    {"error": {"code": "...", "message": "...", "field": "..."}}
"""

from __future__ import annotations

from typing import Any


class ServiceError(Exception):
    """业务领域错误基类，携带机器可读 code 与原因 message。"""

    code: str = "invalid_request"
    status_code: int = 400

    def __init__(
        self,
        message: str,
        *,
        code: str | None = None,
        status_code: int | None = None,
        field: str | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        if code is not None:
            self.code = code
        if status_code is not None:
            self.status_code = status_code
        self.field = field

    def to_payload(self) -> dict[str, Any]:
        err: dict[str, Any] = {"code": self.code, "message": self.message}
        if self.field is not None:
            err["field"] = self.field
        return {"error": err}


class ValidationError(ServiceError):
    """谱参数在进入计算前被挡下。"""

    code = "invalid_parameter"
    status_code = 422


class NotFoundError(ServiceError):
    code = "case_not_found"
    status_code = 404


class ConflictError(ServiceError):
    code = "case_exists"
    status_code = 409


class InversionError(ServiceError):
    code = "inversion_failed"
    status_code = 422
