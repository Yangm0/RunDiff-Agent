"""Validated input and report models for RunDiff Agent."""

from __future__ import annotations

import math
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, RootModel, field_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class MetricDirection(StrEnum):
    MAXIMIZE = "maximize"
    MINIMIZE = "minimize"
    NEUTRAL = "neutral"


class RunManifest(StrictModel):
    run_id: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_.:-]+$")
    git_commit: str | None = Field(default=None, max_length=128)
    dataset_name: str | None = Field(default=None, max_length=256)
    dataset_version: str | None = Field(default=None, max_length=256)
    python_version: str | None = Field(default=None, max_length=128)
    framework: str | None = Field(default=None, max_length=128)
    framework_version: str | None = Field(default=None, max_length=128)
    dependencies: dict[str, str] = Field(default_factory=dict, max_length=500)
    platform: str | None = Field(default=None, max_length=256)


class RunConfig(RootModel[dict[str, Any]]):
    """A configuration is intentionally open-ended and represented as a JSON object."""

    @field_validator("root", mode="before")
    @classmethod
    def require_object(cls, value: Any) -> Any:
        if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
            raise ValueError("config.json must contain a JSON object with string keys")
        return value


class MetricResult(StrictModel):
    value: float
    direction: MetricDirection
    unit: str = Field(min_length=1, max_length=64)
    history: list[float] | None = Field(default=None, max_length=100_000)

    @field_validator("value", mode="before")
    @classmethod
    def numeric_value_only(cls, value: Any) -> Any:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("metric value must be an integer or finite number")
        return value

    @field_validator("value")
    @classmethod
    def finite_value(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("metric value must be finite; NaN and Infinity are not allowed")
        return value

    @field_validator("history", mode="before")
    @classmethod
    def numeric_history_only(cls, values: Any) -> Any:
        if values is not None and (
            not isinstance(values, list)
            or any(
                isinstance(value, bool) or not isinstance(value, (int, float)) for value in values
            )
        ):
            raise ValueError("metric history must be a list of numbers")
        return values

    @field_validator("history")
    @classmethod
    def finite_history(cls, values: list[float] | None) -> list[float] | None:
        if values is not None and any(not math.isfinite(value) for value in values):
            raise ValueError("metric history must contain only finite numbers")
        return values


class MetricsFile(StrictModel):
    metrics: dict[str, MetricResult] = Field(min_length=1, max_length=500)


class EvidenceKind(StrEnum):
    CONFIG = "config"
    ENVIRONMENT = "environment"
    METRIC = "metric"
    LOG = "log"


class Evidence(StrictModel):
    evidence_id: str
    kind: EvidenceKind
    run_id: str
    file: str
    json_path: str | None = None
    line_number: int | None = Field(default=None, ge=1)
    value: Any | None = None
    unit: str | None = None
    excerpt: str | None = Field(default=None, max_length=2000)


class ClaimType(StrEnum):
    OBSERVED = "OBSERVED"
    SUPPORTED_HYPOTHESIS = "SUPPORTED_HYPOTHESIS"
    UNVERIFIED = "UNVERIFIED"
    REJECTED = "REJECTED"


class Finding(StrictModel):
    claim_id: str = Field(min_length=1, max_length=100)
    classification: ClaimType
    text: str = Field(min_length=1, max_length=1200)
    evidence_ids: list[str] = Field(default_factory=list, max_length=30)

    @field_validator("evidence_ids")
    @classmethod
    def bounded_evidence_ids(cls, values: list[str]) -> list[str]:
        if any(not value or len(value) > 64 for value in values):
            raise ValueError("each evidence ID must contain 1 to 64 characters")
        return values


class AgentDraft(StrictModel):
    summary: str = Field(min_length=1, max_length=2000)
    hypotheses: list[Finding] = Field(default_factory=list, max_length=20)
    unresolved_questions: list[str] = Field(default_factory=list, max_length=20)
    recommended_checks: list[str] = Field(default_factory=list, max_length=20)

    @field_validator("unresolved_questions", "recommended_checks")
    @classmethod
    def bounded_text_lists(cls, values: list[str]) -> list[str]:
        if any(len(value) > 1000 for value in values):
            raise ValueError("each list item must be at most 1000 characters")
        return values


class AnalysisReport(StrictModel):
    task_id: str
    mode: str
    status: str
    summary: str
    findings: list[Finding]
    hypotheses: list[Finding]
    unresolved_questions: list[str]
    recommended_checks: list[str]
    missing_information: list[str]
    evidence: list[Evidence]
    validation: dict[str, Any]
