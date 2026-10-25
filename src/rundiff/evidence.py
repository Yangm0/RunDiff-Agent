"""Stable, path-safe evidence creation and post-generation reference validation."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Iterable
from typing import Any

from .loader import RunPair
from .models import ClaimType, Evidence, EvidenceKind, Finding

SECRET_PATTERNS = (
    re.compile(r"(?i)(api[_-]?key|token|secret|password)\s*[:=]\s*[^\s,;]+"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{12,}\b"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"),
    re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]+"),
)
NUMBER_PATTERN = re.compile(r"(?<![\w.])[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?(?![\w.])")


def encode_path_part(value: str) -> str:
    return value.replace("~", "~0").replace(".", "~2")


def decode_path_part(value: str) -> str:
    return value.replace("~2", ".").replace("~0", "~")


def redact(text: str) -> str:
    safe = text
    for index, pattern in enumerate(SECRET_PATTERNS):
        if index == 0:
            safe = pattern.sub(lambda match: match.group(1) + "=[REDACTED]", safe)
        else:
            safe = pattern.sub("[REDACTED]", safe)
    return safe


def _stable_id(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
    return f"E_{payload['kind'].upper()}_{digest}"


def _numeric_values(value: Any) -> list[float]:
    if isinstance(value, bool):
        return []
    if isinstance(value, (int, float)):
        return [float(value)]
    if isinstance(value, list):
        return [number for child in value for number in _numeric_values(child)]
    if isinstance(value, dict):
        return [number for child in value.values() for number in _numeric_values(child)]
    return []


def make_evidence(
    *,
    kind: EvidenceKind,
    run_id: str,
    file: str,
    json_path: str | None = None,
    line_number: int | None = None,
    value: Any = None,
    unit: str | None = None,
    excerpt: str | None = None,
) -> Evidence:
    if file not in {"manifest.json", "config.json", "metrics.json", "train.log"}:
        raise ValueError("Evidence file must be one of the run protocol files")
    safe_excerpt = redact(excerpt)[:2000] if excerpt is not None else None
    safe_value = redact(value) if isinstance(value, str) else value
    identity = {
        "kind": kind.value,
        "run_id": run_id,
        "file": file,
        "json_path": json_path,
        "line_number": line_number,
        "value": safe_value,
        "unit": unit,
        "excerpt": safe_excerpt,
    }
    return Evidence(evidence_id=_stable_id(identity), **identity)


def validate_evidence(evidence: Iterable[Evidence], runs: RunPair) -> dict[str, Any]:
    items = list(evidence)
    errors: list[str] = []
    ids: set[str] = set()
    for item in items:
        if item.evidence_id in ids:
            errors.append(f"duplicate evidence id {item.evidence_id}")
        ids.add(item.evidence_id)
        try:
            run = runs.by_id(item.run_id)
        except ValueError:
            errors.append(f"evidence {item.evidence_id} names a run outside this task")
            continue
        if item.file not in {"manifest.json", "config.json", "metrics.json", "train.log"}:
            errors.append(f"evidence {item.evidence_id} names an unsupported file")
            continue
        expected_file = {
            EvidenceKind.CONFIG: "config.json",
            EvidenceKind.ENVIRONMENT: "manifest.json",
            EvidenceKind.METRIC: "metrics.json",
            EvidenceKind.LOG: "train.log",
        }[item.kind]
        if item.file != expected_file:
            errors.append(f"evidence {item.evidence_id} has a kind/file mismatch")
            continue
        identity = {
            "kind": item.kind.value,
            "run_id": item.run_id,
            "file": item.file,
            "json_path": item.json_path,
            "line_number": item.line_number,
            "value": item.value,
            "unit": item.unit,
            "excerpt": item.excerpt,
        }
        if _stable_id(identity) != item.evidence_id:
            errors.append(f"evidence {item.evidence_id} does not match its stable source ID")
            continue
        if item.file == "train.log":
            if run.log_text is None:
                errors.append(f"evidence {item.evidence_id} references a missing log")
            elif item.line_number is None or item.line_number > len(run.log_text.splitlines()):
                errors.append(f"evidence {item.evidence_id} has an invalid log line")
            elif redact(run.log_text.splitlines()[item.line_number - 1]) != item.excerpt:
                errors.append(f"evidence {item.evidence_id} log excerpt does not match its line")
        elif not item.json_path:
            errors.append(f"evidence {item.evidence_id} is missing a JSON field path")
        else:
            source = {
                "manifest.json": run.manifest.model_dump(mode="json"),
                "config.json": run.config.root,
                "metrics.json": run.metrics.model_dump(mode="json"),
            }[item.file]
            try:
                actual = source
                for part in item.json_path.split("."):
                    part = decode_path_part(part)
                    actual = actual[int(part)] if isinstance(actual, list) else actual[part]
            except (KeyError, IndexError, ValueError, TypeError):
                errors.append(f"evidence {item.evidence_id} has an invalid JSON path")
                continue
            comparable = redact(actual) if isinstance(actual, str) else actual
            if comparable != item.value:
                errors.append(f"evidence {item.evidence_id} value does not match {item.file}")
            if item.kind == EvidenceKind.METRIC:
                unit_path = ".".join([*item.json_path.split(".")[:-1], "unit"])
                unit_value: Any = source
                try:
                    for part in unit_path.split("."):
                        part = decode_path_part(part)
                        unit_value = (
                            unit_value[int(part)]
                            if isinstance(unit_value, list)
                            else unit_value[part]
                        )
                except (KeyError, IndexError, ValueError, TypeError):
                    errors.append(f"evidence {item.evidence_id} metric unit path is invalid")
                else:
                    if unit_value != item.unit:
                        errors.append(
                            f"evidence {item.evidence_id} metric unit does not match its source"
                        )
    return {"valid": not errors, "evidence_count": len(items), "errors": errors}


def validate_findings(
    findings: list[Finding], evidence: list[Evidence], runs: RunPair
) -> tuple[list[Finding], list[str]]:
    valid = {item.evidence_id for item in evidence}
    evidence_by_id = {item.evidence_id: item for item in evidence}
    checked: list[Finding] = []
    warnings: list[str] = []
    for finding in findings:
        safe_text = redact(finding.text)
        if safe_text != finding.text:
            finding = Finding.model_validate({**finding.model_dump(mode="json"), "text": safe_text})
        trusted_ids = [item for item in finding.evidence_ids if item in valid]
        invalid_ids = [item for item in finding.evidence_ids if item not in valid]
        if invalid_ids:
            warnings.append(f"{finding.claim_id}: rejected unknown evidence IDs {invalid_ids}")
        if finding.evidence_ids and not trusted_ids:
            finding = Finding.model_validate(
                {
                    **finding.model_dump(mode="json"),
                    "classification": "UNVERIFIED",
                    "evidence_ids": [],
                }
            )
        elif (
            finding.classification
            in {
                ClaimType.OBSERVED,
                ClaimType.SUPPORTED_HYPOTHESIS,
            }
            and not trusted_ids
        ):
            finding = Finding.model_validate(
                {
                    **finding.model_dump(mode="json"),
                    "classification": "UNVERIFIED",
                    "evidence_ids": [],
                }
            )
        else:
            finding = finding.model_copy(update={"evidence_ids": trusted_ids})
        if finding.classification in {ClaimType.OBSERVED, ClaimType.SUPPORTED_HYPOTHESIS}:
            cited_numbers: list[float] = []
            for evidence_id in trusted_ids:
                cited_numbers.extend(_numeric_values(evidence_by_id[evidence_id].value))
            claim_numbers = [
                float(match.group()) for match in NUMBER_PATTERN.finditer(finding.text)
            ]
            unsupported_numbers = [
                number
                for number in claim_numbers
                if not any(
                    math.isclose(number, cited, rel_tol=1e-9, abs_tol=1e-12)
                    for cited in cited_numbers
                )
            ]
            if unsupported_numbers:
                warnings.append(
                    f"{finding.claim_id}: rejected unsupported numeric claims {unsupported_numbers}"
                )
                finding = Finding.model_validate(
                    {
                        **finding.model_dump(mode="json"),
                        "classification": ClaimType.UNVERIFIED.value,
                        "evidence_ids": [],
                    }
                )
        checked.append(finding)
    return checked, warnings
