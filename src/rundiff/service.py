"""Shared application service used by the CLI, API, and evaluation runner."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .agent import run_agent
from .evidence import redact
from .loader import RunInputError, load_pair
from .tools import analyze_metrics, compare_configs, compare_environments

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES_ROOT = PROJECT_ROOT / "examples" / "cases"


def available_examples() -> list[dict[str, str]]:
    index = EXAMPLES_ROOT / "index.json"
    if not index.is_file():
        return []
    try:
        return json.loads(index.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Example catalog could not be read: {exc}") from exc


def get_example(example_id: str) -> tuple[Path, Path]:
    allowed = "abcdefghijklmnopqrstuvwxyz0123456789-"
    if not example_id or any(character not in allowed for character in example_id):
        raise RunInputError("Unknown example case ID")
    catalog = {item["id"] for item in available_examples()}
    if example_id not in catalog:
        raise RunInputError(f"Unknown example case ID: {example_id}")
    root = EXAMPLES_ROOT.resolve(strict=True)
    folder = (root / example_id).resolve(strict=True)
    if folder.parent != root:
        raise RunInputError("Example path is outside the approved examples directory")
    return folder / "run_a", folder / "run_b"


class ApplicationService:
    """Request-scoped orchestration with no shared mutable run state."""

    def compare(self, run_a: str | Path, run_b: str | Path) -> dict[str, Any]:
        pair = load_pair(run_a, run_b)
        result = {
            "runs": {"run_a": pair.run_a.run_id, "run_b": pair.run_b.run_id},
            "compare_configs": compare_configs(pair),
            "compare_environments": compare_environments(pair),
            "analyze_metrics": analyze_metrics(pair),
        }
        for tool_result in result.values():
            if isinstance(tool_result, dict) and "evidence" in tool_result:
                tool_result["evidence"] = [
                    item.model_dump(mode="json") for item in tool_result.get("evidence", [])
                ]
        return result

    def analyze(
        self,
        run_a: str | Path,
        run_b: str | Path,
        *,
        mode: str = "mock",
        output_dir: str | Path | None = None,
        model: Any | None = None,
    ) -> dict[str, Any]:
        pair = load_pair(run_a, run_b)
        report = run_agent(pair, mode, model=model)
        if output_dir is not None:
            self.write_outputs(report, output_dir)
        return report

    def analyze_example(
        self,
        example_id: str,
        *,
        mode: str = "mock",
        output_dir: str | Path | None = None,
        model: Any | None = None,
    ) -> dict[str, Any]:
        run_a, run_b = get_example(example_id)
        report = self.analyze(run_a, run_b, mode=mode, output_dir=output_dir, model=model)
        report["example_id"] = example_id
        if output_dir is not None:
            self.write_outputs(report, output_dir)
        return report

    @staticmethod
    def write_outputs(report: dict[str, Any], output_dir: str | Path) -> None:
        destination = Path(output_dir).expanduser().resolve()
        destination.mkdir(parents=True, exist_ok=True)
        safe_report = {key: value for key, value in report.items() if key != "trace"}
        (destination / "report.json").write_text(
            json.dumps(safe_report, ensure_ascii=False, indent=2, default=str) + "\n",
            encoding="utf-8",
        )
        (destination / "report.md").write_text(report_to_markdown(report), encoding="utf-8")
        trace_lines = [
            json.dumps(redact_trace(event), ensure_ascii=False, default=str)
            for event in report.get("trace", [])
        ]
        (destination / "trace.jsonl").write_text(
            "\n".join(trace_lines) + ("\n" if trace_lines else ""), encoding="utf-8"
        )


def redact_trace(event: dict[str, Any]) -> dict[str, Any]:
    return {key: redact(value) if isinstance(value, str) else value for key, value in event.items()}


def report_to_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# RunDiff Agent report",
        "",
        f"- **Task:** {report['task_id']}",
        f"- **Mode:** {report['mode']}",
        f"- **Status:** {report['status']}",
        "",
        "## Summary",
        "",
        report["summary"],
        "",
        "## Observed findings",
        "",
    ]
    findings = report.get("findings", [])
    if findings:
        for finding in findings:
            citations = ", ".join(finding.get("evidence_ids", [])) or "no verified evidence"
            lines.append(
                f"- **{finding['classification']}** {finding['text']} (Evidence: {citations})"
            )
    else:
        lines.append("- No deterministic differences were found.")
    lines += ["", "## Supported hypotheses", ""]
    hypotheses = report.get("hypotheses", [])
    if hypotheses:
        for item in hypotheses:
            citations = ", ".join(item.get("evidence_ids", [])) or "no verified evidence"
            lines.append(f"- **{item['classification']}** {item['text']} (Evidence: {citations})")
    else:
        lines.append("- No supported hypotheses were returned.")
    lines += ["", "## Missing information", ""]
    missing = report.get("missing_information", [])
    lines.extend([f"- {item}" for item in missing] or ["- No required files were missing."])
    lines += ["", "## Unresolved questions", ""]
    questions = report.get("unresolved_questions", [])
    lines.extend([f"- {item}" for item in questions] or ["- None recorded."])
    lines += ["", "## Recommended checks", ""]
    checks = report.get("recommended_checks", [])
    lines.extend([f"- {item}" for item in checks] or ["- None recorded."])
    lines += [
        "",
        "## Evidence validation",
        "",
        f"- Valid: {report.get('validation', {}).get('valid', False)}",
        f"- Evidence records: {report.get('validation', {}).get('evidence_count', 0)}",
        "",
        "Static comparisons describe co-occurring differences; they do not prove a cause.",
        "",
    ]
    return "\n".join(lines)
