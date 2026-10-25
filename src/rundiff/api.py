"""Small FastAPI surface restricted to bundled synthetic examples."""

from __future__ import annotations

from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from .agent import LiveModelUnavailable
from .loader import RunInputError
from .service import ApplicationService, available_examples

app = FastAPI(
    title="RunDiff Agent API",
    description="Evidence-grounded comparison of synthetic ML experiment runs.",
    version="0.1.0",
)
service = ApplicationService()


class AnalyzeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    example_id: str = Field(min_length=1, max_length=100)
    mode: Literal["mock", "live"] = "mock"


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/examples")
def examples() -> dict[str, list[dict[str, str]]]:
    return {"examples": available_examples()}


@app.post("/api/analyze")
def analyze(request: AnalyzeRequest) -> dict:
    try:
        return service.analyze_example(request.example_id, mode=request.mode)
    except LiveModelUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except RunInputError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
