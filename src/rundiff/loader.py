"""Safe loading and validation of a pair of local run directories."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from .models import MetricsFile, RunConfig, RunManifest

MAX_JSON_BYTES = 2_000_000
MAX_LOG_BYTES = 10_000_000


class RunInputError(ValueError):
    """A run directory is missing data or contains invalid input."""


@dataclass(frozen=True)
class RunBundle:
    label: str
    root: Path
    manifest: RunManifest
    config: RunConfig
    metrics: MetricsFile
    log_text: str | None

    @property
    def run_id(self) -> str:
        return self.manifest.run_id


@dataclass(frozen=True)
class RunPair:
    run_a: RunBundle
    run_b: RunBundle

    def by_id(self, run_id: str) -> RunBundle:
        matches = [run for run in (self.run_a, self.run_b) if run.run_id == run_id]
        if len(matches) != 1:
            raise ValueError(f"run_id must match exactly one input run; received {run_id!r}")
        return matches[0]


def _read_json(path: Path) -> Any:
    if path.is_symlink():
        raise RunInputError(f"Refusing symlink input file: {path.name}")
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise RunInputError(f"Unable to inspect {path.name}: {exc}") from exc
    if size > MAX_JSON_BYTES:
        raise RunInputError(f"{path.name} exceeds {MAX_JSON_BYTES} bytes")

    def finite_float(token: str) -> float:
        value = float(token)
        if not math.isfinite(value):
            raise ValueError(f"non-finite numeric value {token}")
        return value

    def reject_constant(token: str) -> None:
        raise ValueError(f"non-finite numeric constant {token}")

    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            parse_float=finite_float,
            parse_constant=reject_constant,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise RunInputError(f"Invalid {path.name}: {exc}") from exc


def _validate(model: type, value: Any, filename: str) -> Any:
    try:
        return model.model_validate(value)
    except ValidationError as exc:
        details = "; ".join(
            f"{'.'.join(map(str, error['loc'])) or '<root>'}: {error['msg']}"
            for error in exc.errors(include_url=False)
        )
        raise RunInputError(f"Invalid {filename}: {details}") from exc


def load_run(path: str | Path, label: str) -> RunBundle:
    try:
        root = Path(path).expanduser().resolve(strict=True)
    except OSError as exc:
        raise RunInputError(f"Run {label} path does not exist or is inaccessible") from exc
    if not root.is_dir():
        raise RunInputError(f"Run {label} must be a directory")
    values: dict[str, Any] = {}
    for filename in ("manifest.json", "config.json", "metrics.json"):
        candidate = root / filename
        if not candidate.is_file():
            raise RunInputError(f"Run {label} is missing required file {filename}")
        values[filename] = _read_json(candidate)
    manifest = _validate(RunManifest, values["manifest.json"], "manifest.json")
    config = _validate(RunConfig, values["config.json"], "config.json")
    metrics = _validate(MetricsFile, values["metrics.json"], "metrics.json")

    log_path = root / "train.log"
    log_text: str | None = None
    if log_path.exists():
        if log_path.is_symlink() or not log_path.is_file():
            raise RunInputError("train.log must be a regular file, not a symlink")
        if log_path.stat().st_size > MAX_LOG_BYTES:
            raise RunInputError(f"train.log exceeds {MAX_LOG_BYTES} bytes")
        try:
            log_text = log_path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            raise RunInputError(f"Unable to read train.log: {exc}") from exc
    return RunBundle(label, root, manifest, config, metrics, log_text)


def load_pair(run_a_path: str | Path, run_b_path: str | Path) -> RunPair:
    run_a = load_run(run_a_path, "A")
    run_b = load_run(run_b_path, "B")
    if run_a.root == run_b.root:
        raise RunInputError("Run A and Run B must be different directories")
    if run_a.run_id == run_b.run_id:
        raise RunInputError("manifest.run_id must be unique between Run A and Run B")
    return RunPair(run_a, run_b)
