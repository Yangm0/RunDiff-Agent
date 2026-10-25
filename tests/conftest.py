from __future__ import annotations

import json
from pathlib import Path

import pytest

from rundiff.service import EXAMPLES_ROOT


@pytest.fixture
def example_pair() -> tuple[Path, Path]:
    return (
        EXAMPLES_ROOT / "learning-rate-wave" / "run_a",
        EXAMPLES_ROOT / "learning-rate-wave" / "run_b",
    )


@pytest.fixture
def run_pair_factory(tmp_path: Path):
    def create(
        *,
        config_a: dict | None = None,
        config_b: dict | None = None,
        manifest_a: dict | None = None,
        manifest_b: dict | None = None,
        metrics_a: dict | None = None,
        metrics_b: dict | None = None,
        log_a: str | None = "epoch=1\n",
        log_b: str | None = "epoch=2\n",
    ) -> tuple[Path, Path]:
        roots = (tmp_path / "run-a", tmp_path / "run-b")
        for label, root, config, manifest_patch, metrics, log in (
            ("a", roots[0], config_a or {"model": {"lr": 0.1}}, manifest_a or {}, metrics_a, log_a),
            ("b", roots[1], config_b or {"model": {"lr": 0.2}}, manifest_b or {}, metrics_b, log_b),
        ):
            root.mkdir(parents=True, exist_ok=True)
            manifest = {
                "run_id": f"test-run-{label}",
                "git_commit": "abcdef0",
                "dataset_name": "sample",
                "dataset_version": "v1",
                "python_version": "3.13.2",
                "framework": "pytorch",
                "framework_version": "2.5.0",
                "dependencies": {"numpy": "2.0.0"},
            }
            manifest.update(manifest_patch)
            metric_values = metrics or {
                "score": {
                    "value": 0.8 if label == "a" else 0.7,
                    "direction": "maximize",
                    "unit": "fraction",
                }
            }
            for filename, content in (
                ("manifest.json", manifest),
                ("config.json", config),
                ("metrics.json", {"metrics": metric_values}),
            ):
                (root / filename).write_text(json.dumps(content), encoding="utf-8")
            if log is not None:
                (root / "train.log").write_text(log, encoding="utf-8")
        return roots

    return create
