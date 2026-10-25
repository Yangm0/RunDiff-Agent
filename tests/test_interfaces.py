from __future__ import annotations

from fastapi.testclient import TestClient
from typer.testing import CliRunner

from rundiff.api import app
from rundiff.cli import app as cli
from rundiff.evaluation import run_evaluation


def test_fastapi_health_examples_docs_and_analysis():
    client = TestClient(app)
    assert client.get("/health").json() == {"status": "ok"}
    examples = client.get("/api/examples").json()["examples"]
    assert len(examples) == 8
    response = client.post(
        "/api/analyze", json={"example_id": "learning-rate-wave", "mode": "mock"}
    )
    assert response.status_code == 200
    assert response.json()["mode"] == "mock"
    assert response.json()["validation"]["valid"] is True
    assert client.get("/docs").status_code == 200


def test_api_rejects_unknown_cases_and_arbitrary_path_fields():
    client = TestClient(app)
    assert client.post("/api/analyze", json={"example_id": ".."}).status_code == 422
    assert (
        client.post(
            "/api/analyze",
            json={"example_id": "learning-rate-wave", "run_a": "C:/private"},
        ).status_code
        == 422
    )


def test_api_live_mode_explains_missing_credentials(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    response = TestClient(app).post(
        "/api/analyze",
        json={"example_id": "learning-rate-wave", "mode": "live"},
    )
    assert response.status_code == 503
    assert "OPENAI_API_KEY" in response.json()["detail"]


def test_cli_analyze_writes_three_outputs(tmp_path):
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "analyze",
            "--example",
            "learning-rate-wave",
            "--mode",
            "mock",
            "--output-dir",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 0, result.output
    assert (tmp_path / "report.json").is_file()
    assert (tmp_path / "report.md").is_file()
    assert (tmp_path / "trace.jsonl").is_file()


def test_evaluation_uses_eight_cases_and_reports_mock_validation(tmp_path):
    result = run_evaluation("mock", tmp_path)
    assert result["case_count"] == 8
    assert result["evaluation_mode"] == "mock_flow_validation"
    assert result["answer_keys_loaded_after_analysis"] is True
    assert all(item["multi_step"]["evidence_validity"] is True for item in result["cases"])
    assert all(item["multi_step"]["tool_call_success_rate"] == 1 for item in result["cases"])
    assert (tmp_path / "evaluation.json").is_file()
