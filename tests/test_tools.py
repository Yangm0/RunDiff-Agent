from __future__ import annotations

import pytest

from rundiff.evidence import validate_evidence
from rundiff.loader import load_pair
from rundiff.service import EXAMPLES_ROOT
from rundiff.tools import (
    InspectLogsArgs,
    analyze_metrics,
    compare_configs,
    compare_environments,
    dispatch_tool,
)


def test_nested_config_add_remove_change_and_evidence(run_pair_factory):
    run_a, run_b = run_pair_factory(
        config_a={"model": {"width": 32, "dropout": 0.1}, "removed": True},
        config_b={"model": {"width": 64, "dropout": 0.1}, "added": "yes"},
    )
    pair = load_pair(run_a, run_b)
    result = compare_configs(pair)
    changes = {item["path"]: item["change"] for item in result["changes"]}
    assert changes == {"added": "added", "model.width": "changed", "removed": "removed"}
    assert validate_evidence(result["evidence"], pair)["valid"]


def test_config_evidence_ids_are_stable(run_pair_factory):
    run_a, run_b = run_pair_factory()
    pair = load_pair(run_a, run_b)
    first = compare_configs(pair)["evidence"]
    second = compare_configs(pair)["evidence"]
    assert [item.evidence_id for item in first] == [item.evidence_id for item in second]


def test_config_keys_with_dots_remain_verifiable(run_pair_factory):
    run_a, run_b = run_pair_factory(config_a={"field.with.dot": 1}, config_b={"field.with.dot": 2})
    pair = load_pair(run_a, run_b)
    result = compare_configs(pair)
    assert result["changes"][0]["path"] == "field~2with~2dot"
    assert validate_evidence(result["evidence"], pair)["valid"]


def test_environment_comparison_includes_commit_dataset_and_dependencies(run_pair_factory):
    run_a, run_b = run_pair_factory(
        manifest_b={
            "git_commit": "1234567",
            "dataset_version": "v2",
            "dependencies": {"numpy": "2.1.0", "pandas": "2.2.0"},
        }
    )
    pair = load_pair(run_a, run_b)
    changes = {item["path"] for item in compare_environments(pair)["changes"]}
    assert {"git_commit", "dataset_version", "dependencies.numpy", "dependencies.pandas"} <= changes


def test_metric_direction_unit_delta_and_history_signal(run_pair_factory):
    metrics_a = {"score": {"value": 0.8, "direction": "maximize", "unit": "fraction"}}
    metrics_b = {
        "score": {
            "value": 0.7,
            "direction": "maximize",
            "unit": "fraction",
            "history": [0.7, 0.71, 0.7, 0.72, 0.71, 0.9],
        }
    }
    run_a, run_b = run_pair_factory(metrics_a=metrics_a, metrics_b=metrics_b)
    result = analyze_metrics(load_pair(run_a, run_b))["comparisons"][0]
    assert result["absolute_change"] == pytest.approx(-0.1)
    assert result["relative_change"] == pytest.approx(-0.125)
    assert result["assessment"] == "run_a_better"
    assert result["history_run_b"]["flagged"] is True


def test_metric_missing_in_run_a_uses_run_b_evidence(run_pair_factory):
    metric = {"value": 0.42, "direction": "maximize", "unit": "fraction"}
    run_a, run_b = run_pair_factory(
        metrics_a={"only_a": {"value": 0.1, "direction": "neutral", "unit": "score"}},
        metrics_b={"missing_from_a": metric},
    )
    pair = load_pair(run_a, run_b)
    result = analyze_metrics(pair)
    comparison = next(item for item in result["comparisons"] if item["name"] == "missing_from_a")
    evidence = next(
        item for item in result["evidence"] if item.evidence_id == comparison["evidence_ids"][0]
    )

    assert comparison["status"] == "missing_in_run_a"
    assert evidence.run_id == pair.run_b.run_id
    assert evidence.value == metric["value"]
    assert validate_evidence([evidence], pair)["valid"] is True


def test_metric_missing_in_run_b_uses_run_a_evidence(run_pair_factory):
    metric = {"value": 0.73, "direction": "minimize", "unit": "loss"}
    run_a, run_b = run_pair_factory(
        metrics_a={"missing_from_b": metric},
        metrics_b={"only_b": {"value": 0.2, "direction": "neutral", "unit": "score"}},
    )
    pair = load_pair(run_a, run_b)
    result = analyze_metrics(pair)
    comparison = next(item for item in result["comparisons"] if item["name"] == "missing_from_b")
    evidence = next(
        item for item in result["evidence"] if item.evidence_id == comparison["evidence_ids"][0]
    )

    assert comparison["status"] == "missing_in_run_b"
    assert evidence.run_id == pair.run_a.run_id
    assert evidence.value == metric["value"]
    assert validate_evidence([evidence], pair)["valid"] is True


def test_unknown_or_incompatible_metric_definitions_are_not_ranked(run_pair_factory):
    metrics_a = {
        "value": {"value": 2.0, "direction": "neutral", "unit": "score"},
        "changed_unit": {"value": 3.0, "direction": "minimize", "unit": "seconds"},
    }
    metrics_b = {
        "value": {"value": 1.0, "direction": "neutral", "unit": "score"},
        "changed_unit": {"value": 4.0, "direction": "minimize", "unit": "ms"},
    }
    run_a, run_b = run_pair_factory(metrics_a=metrics_a, metrics_b=metrics_b)
    comparisons = {
        item["name"]: item for item in analyze_metrics(load_pair(run_a, run_b))["comparisons"]
    }
    assert comparisons["value"]["assessment"] == "direction_neutral"
    assert comparisons["changed_unit"]["status"] == "incomparable_definition"
    assert comparisons["changed_unit"]["absolute_change"] is None


def test_log_range_returns_exact_line_evidence(run_pair_factory):
    run_a, run_b = run_pair_factory(log_a="first\nRuntimeError: bad batch\nthird\n")
    pair = load_pair(run_a, run_b)
    result = dispatch_tool(
        "inspect_logs",
        {"run_id": pair.run_a.run_id, "start_line": 2, "end_line": 3},
        pair,
    )
    assert [item["line_number"] for item in result["matches"]] == [2, 3]
    assert validate_evidence(result["evidence"], pair)["valid"]


def test_log_keyword_and_secrets_are_filtered(run_pair_factory):
    run_a, run_b = run_pair_factory(log_a="INFO ready\napi_key=synthetic-secret-value failed\n")
    pair = load_pair(run_a, run_b)
    result = dispatch_tool(
        "inspect_logs",
        {"run_id": pair.run_a.run_id, "keywords": ["failed"]},
        pair,
    )
    assert len(result["matches"]) == 1
    assert "synthetic-secret-value" not in result["matches"][0]["text"]
    assert result["matches"][0]["line_number"] == 2


def test_prompt_like_log_text_is_data_and_credentials_are_redacted():
    pair = load_pair(
        EXAMPLES_ROOT / "noisy-untrusted-log" / "run_a",
        EXAMPLES_ROOT / "noisy-untrusted-log" / "run_b",
    )
    result = dispatch_tool(
        "inspect_logs",
        {"run_id": pair.run_b.run_id, "keywords": ["NOTICE"]},
        pair,
    )
    assert len(result["matches"]) == 1
    assert "ignore previous instructions" in result["matches"][0]["text"]
    assert "synthetic-secret-value" not in result["matches"][0]["text"]
    assert "API_KEY=[REDACTED]" in result["matches"][0]["text"]
    assert validate_evidence(result["evidence"], pair)["valid"]


@pytest.mark.parametrize(
    "args",
    [
        {"run_id": "x", "start_line": 1},
        {"run_id": "x", "start_line": 4, "end_line": 3},
        {"run_id": "x", "start_line": 1, "end_line": 202},
        {"run_id": "x", "max_results": 21},
        {"run_id": "x", "not_allowed": True},
    ],
)
def test_log_tool_schema_rejects_bad_parameters(args):
    with pytest.raises(ValueError):
        InspectLogsArgs.model_validate(args)


def test_unknown_tool_and_bad_arguments_fail_closed(run_pair_factory):
    pair = load_pair(*run_pair_factory())
    with pytest.raises(ValueError, match="Unknown tool"):
        dispatch_tool("shell", {}, pair)
    with pytest.raises(ValueError):
        dispatch_tool("inspect_logs", {"run_id": "wrong"}, pair)
