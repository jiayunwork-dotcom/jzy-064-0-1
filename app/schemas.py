"""HTTP 请求 / 响应的 pydantic 模型（边界层，不含核算逻辑）。"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class SpectrumRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    omega_p: float | None = Field(default=None, description="谱峰角频率 [rad/s]")
    gamma: float | None = Field(default=None, description="峰升因子，>=1；=1 退化为 PM")
    alpha: float | None = Field(default=None, description="尺度因子（直接给定时优先）")
    hs: float | None = Field(default=None, description="有效波高约束 [m]，触发 α 反演")
    wind_speed: float | None = Field(default=None, description="10 m 高度风速 [m/s]")
    fetch: float | None = Field(default=None, description="风区长度 [m]")
    sample_points: int | None = Field(default=None, ge=64, le=8192)


class MomentsRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    omega_p: float | None = None
    gamma: float | None = None
    alpha: float | None = None
    hs: float | None = None
    wind_speed: float | None = None
    fetch: float | None = None


class CaseSaveRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=128)
    description: str | None = Field(default=None, max_length=512)
    overwrite: bool = False
    params: SpectrumRequest


class ErrorBody(BaseModel):
    code: str
    message: str
    field: str | None = None


class ErrorResponse(BaseModel):
    error: ErrorBody


class OkResponse(BaseModel):
    data: dict[str, Any]
