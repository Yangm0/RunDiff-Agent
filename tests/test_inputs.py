from __future__ import annotations

import json

import pytest

from rundiff.loader import RunInputError, load_pair


def test_valid_pair_has_distinct_run_ids(example_pair):
    pair = load_pair(*example_pair)
    assert pair.run_a.run_id != pair.run_b.run_id
    assert pair.by_id(pair.run_b.run_id).label == "B"


def test_invalid_json_has_a_clear_error(run_pair_factory):
    run_a, run_b = run_pair_factory()
    (run_a / "config.json").write_text("{bad json", encoding="utf-8")
    with pytest.raises(RunInputError, match="Invalid config.json"):
        load_pair(run_a, run_b)


def test_missing_required_file_is_reported(run_pair_factory):
    run_a, run_b = run_pair_factory()
    (run_a / "metrics.json").unlink()
    with pytest.raises(RunInputError, match="metrics.json"):
        load_pair(run_a, run_b)


def test_nonexistent_path_has_a_clear_input_error(tmp_path):
    with pytest.raises(RunInputError, match="does not exist"):
        load_pair(tmp_path / "missing-a", tmp_path / "missing-b")


def test_unknown_manifest_fields_are_rejected(run_pair_factory):
    run_a, run_b = run_pair_factory()
    manifest = json.loads((run_a / "manifest.json").read_text(encoding="utf-8"))
    manifest["unreviewed_field"] = True
    (run_a / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(RunInputError, match="unreviewed_field"):
        load_pair(run_a, run_b)


@pytest.mark.parametrize("value", ["NaN", "Infinity", "-Infinity"])
def test_non_finite_metric_values_are_rejected(run_pair_factory, value):
    run_a, run_b = run_pair_factory()
    (run_a / "metrics.json").write_text(
        '{"metrics":{"score":{"value":' + value + ',"direction":"maximize","unit":"fraction"}}}',
        encoding="utf-8",
    )
    with pytest.raises(RunInputError, match="finite"):
        load_pair(run_a, run_b)


@pytest.mark.parametrize("value", ["NaN", "Infinity", "1e999"])
def test_non_finite_json_values_are_rejected_even_in_config(run_pair_factory, value):
    run_a, run_b = run_pair_factory()
    (run_a / "config.json").write_text('{"threshold":' + value + "}", encoding="utf-8")
    with pytest.raises(RunInputError, match="Invalid config.json"):
        load_pair(run_a, run_b)


def test_metric_requires_explicit_direction_and_unit(run_pair_factory):
    run_a, run_b = run_pair_factory()
    (run_a / "metrics.json").write_text('{"metrics":{"score":{"value":1}}}', encoding="utf-8")
    with pytest.raises(RunInputError, match="direction|unit"):
        load_pair(run_a, run_b)


@pytest.mark.parametrize("value", ['"0.8"', "true"])
def test_metric_rejects_coercible_but_invalid_types(run_pair_factory, value):
    run_a, run_b = run_pair_factory()
    (run_a / "metrics.json").write_text(
        '{"metrics":{"score":{"value":' + value + ',"direction":"maximize","unit":"fraction"}}}',
        encoding="utf-8",
    )
    with pytest.raises(RunInputError, match="integer or finite"):
        load_pair(run_a, run_b)


def test_duplicate_ids_and_same_directory_are_rejected(run_pair_factory):
    run_a, run_b = run_pair_factory()
    manifest = json.loads((run_b / "manifest.json").read_text(encoding="utf-8"))
    manifest["run_id"] = "test-run-a"
    (run_b / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(RunInputError, match="unique"):
        load_pair(run_a, run_b)
    with pytest.raises(RunInputError, match="different directories"):
        load_pair(run_a, run_a)


def test_missing_log_is_allowed(run_pair_factory):
    run_a, run_b = run_pair_factory(log_b=None)
    pair = load_pair(run_a, run_b)
    assert pair.run_b.log_text is None
