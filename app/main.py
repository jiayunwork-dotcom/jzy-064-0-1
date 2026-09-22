"""HTTP routes for the JONSWAP spectrum service.

Two computation endpoints (full spectrum, moments only) plus a named-case
archive. All invalid-parameter rejections return a structured error body
{"error", "parameter", "value", "reason"}.
"""

from __future__ import annotations

import os

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from .cases import CaseNotFoundError, CaseStore
from .inversion import InversionError
from .schemas import (
    CaseBody,
    CaseListResponse,
    CaseSaveRequest,
    MomentsResponse,
    SpectrumParams,
    SpectrumResponse,
)
from .service import SpectrumResult, compute_spectrum
from .spectrum import ParameterError

CASES_PATH = os.environ.get("CASES_PATH", "cases.json")

app = FastAPI(
    title="JONSWAP Spectrum Service",
    description=(
        "Forward evaluation and wave-height-constraint inversion of the "
        "JONSWAP frequency spectrum, with spectral moments and derived "
        "wave statistics. Computation only; no web UI."
    ),
    version="1.0.0",
)

store = CaseStore(CASES_PATH)


@app.exception_handler(ParameterError)
async def parameter_error_handler(_request: Request, exc: ParameterError):
    return JSONResponse(
        status_code=400,
        content={
            "error": "invalid_parameter",
            "parameter": exc.parameter,
            "value": exc.value,
            "reason": exc.reason,
        },
    )


@app.exception_handler(InversionError)
async def inversion_error_handler(_request: Request, exc: InversionError):
    return JSONResponse(
        status_code=500,
        content={"error": "inversion_failed", "reason": exc.reason},
    )


@app.exception_handler(CaseNotFoundError)
async def case_not_found_handler(_request: Request, exc: CaseNotFoundError):
    return JSONResponse(
        status_code=404,
        content={
            "error": "case_not_found",
            "reason": f"no case stored under name {exc.name!r}",
        },
    )


def _run(params: SpectrumParams) -> SpectrumResult:
    return compute_spectrum(
        omega_p=params.omega_p,
        gamma=params.gamma,
        alpha=params.alpha,
        hs_target=params.hs_target,
        wind_speed=params.wind_speed,
        fetch=params.fetch,
        n_points=params.n_points,
    )


def _moments_body(result: SpectrumResult) -> dict:
    return {
        "m0": result.moments.m0,
        "m1": result.moments.m1,
        "m2": result.moments.m2,
        "hs": result.stats.hs,
        "tp": result.stats.tp,
        "tz": result.stats.tz,
        "t01": result.stats.t01,
    }


def _inversion_body(result: SpectrumResult) -> dict | None:
    inv = result.inversion
    if inv is None:
        return None
    return {
        "converged": inv.converged,
        "iterations": inv.iterations,
        "residual": inv.residual,
        "hs_achieved": inv.hs_achieved,
    }


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/spectrum", response_model=SpectrumResponse)
def spectrum(params: SpectrumParams) -> dict:
    """Full spectrum: sample points, moments m0..m2, Hs, Tp, Tz, T01."""
    result = _run(params)
    return {
        "omega_p": result.omega_p,
        "gamma": result.gamma,
        "alpha": result.alpha,
        "n_points": int(result.omega.size),
        "omega": result.omega.tolist(),
        "s": result.s.tolist(),
        **_moments_body(result),
        "inversion": _inversion_body(result),
    }


@app.post("/moments", response_model=MomentsResponse)
def moments(params: SpectrumParams) -> dict:
    """Spectral moments and the statistical periods derived from them."""
    result = _run(params)
    return {
        "omega_p": result.omega_p,
        "gamma": result.gamma,
        "alpha": result.alpha,
        **_moments_body(result),
        "inversion": _inversion_body(result),
    }


@app.post("/cases", response_model=CaseBody, status_code=201)
def save_case(request: CaseSaveRequest) -> dict:
    """Archive a parameter set under a case name for later recomputation."""
    # Validate before archiving so no broken case can be stored.
    _run(request.params)
    return store.save_case(
        request.name,
        request.params.model_dump(exclude_none=True),
        request.description,
    )


@app.get("/cases", response_model=CaseListResponse)
def list_cases() -> dict:
    return {"cases": store.list_cases()}


@app.get("/cases/{name}", response_model=CaseBody)
def get_case(name: str) -> dict:
    return store.get_case(name)


@app.get("/cases/{name}/spectrum", response_model=SpectrumResponse)
def case_spectrum(name: str) -> dict:
    """Recompute the spectrum for a previously archived case."""
    case = store.get_case(name)
    return spectrum(SpectrumParams(**case["params"]))


@app.delete("/cases/{name}", status_code=204)
def delete_case(name: str) -> None:
    store.delete_case(name)
