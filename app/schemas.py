"""Pydantic request/response schemas for the HTTP layer.

Physical validation (positivity, gamma >= 1, exactly one of alpha /
hs_target) lives in app.spectrum.validate_spectrum_inputs so that the
service and the API share one single validation path; these models only
carry and type-check the payloads.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class SpectrumParams(BaseModel):
    omega_p: float = Field(..., description="peak angular frequency [rad/s]")
    gamma: float = Field(..., description="peak enhancement factor, >= 1")
    alpha: float | None = Field(
        None, description="Phillips scale factor (mutually exclusive with hs_target)"
    )
    hs_target: float | None = Field(
        None, description="target significant wave height [m] (inverts alpha)"
    )
    wind_speed: float | None = Field(None, description="wind speed [m/s]")
    fetch: float | None = Field(None, description="fetch length [m]")
    n_points: int | None = Field(None, description="frequency grid size override")


class InversionInfo(BaseModel):
    converged: bool
    iterations: int
    residual: float
    hs_achieved: float


class MomentsBody(BaseModel):
    m0: float
    m1: float
    m2: float
    hs: float
    tp: float
    tz: float
    t01: float


class SpectrumResponse(MomentsBody):
    omega_p: float
    gamma: float
    alpha: float
    n_points: int
    omega: list[float]
    s: list[float]
    inversion: InversionInfo | None


class MomentsResponse(MomentsBody):
    omega_p: float
    gamma: float
    alpha: float
    inversion: InversionInfo | None


class CaseSaveRequest(BaseModel):
    name: str
    description: str = ""
    params: SpectrumParams


class CaseBody(BaseModel):
    name: str
    description: str
    params: SpectrumParams


class CaseListResponse(BaseModel):
    cases: list[str]
