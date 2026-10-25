from __future__ import annotations

import json

from langchain_core.messages import AIMessage

from rundiff.agent import LiveModelUnavailable, run_agent
from rundiff.service import report_to_markdown

VALID_DRAFT = json.dumps(
    {
        "summary": "A structured test draft.",
        "hypotheses": [],
        "unresolved_questions": [],
        "recommended_checks": [],
    }
)


class ScriptedModel:
    def __init__(self, responses):
        self.responses = list(responses)

    def bind_tools(self, tools, **kwargs):
        return self

    def invoke(self, messages):
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def tool_call(name="compare_configs", args=None, call_id="c1"):
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": args or {}, "id": call_id, "type": "tool_call"}],
    )


def test_mock_agent_runs_langgraph_tool_loop_and_validates_evidence(example_pair):
    report = run_agent_from_paths(example_pair)
    assert report["mode"] == "mock"
    assert report["status"] == "complete"
    assert report["tool_calls"] == 2
    assert report["decision_rounds"] == 3
    assert report["validation"]["valid"] is True
    assert report["investigation_summary"]["source"] == "mock_model_simulation"
    assert report["investigation_summary"]["is_simulated"] is True
    assert report["investigation_summary"]["verified"] is False
    assert report["findings"]
    assert any(item["step"] == "tool" and item["success"] for item in report["trace"])


def run_agent_from_paths(pair_paths):
    from rundiff.loader import load_pair

    return run_agent(load_pair(*pair_paths), mode="mock")


def test_model_exception_produces_partial_status_and_trace(run_pair_factory):
    from rundiff.loader import load_pair

    pair = load_pair(*run_pair_factory())
    model = ScriptedModel([TimeoutError("model timeout")])
    report = run_agent(pair, mode="live", model=model)
    assert report["status"] == "partial"
    assert "model timeout" in report["errors"][0]
    assert any(
        event["success"] is False and event["step"] == "agent_decision" for event in report["trace"]
    )


def test_invalid_structured_output_does_not_claim_completion(run_pair_factory):
    from rundiff.loader import load_pair

    pair = load_pair(*run_pair_factory())
    report = run_agent(pair, mode="live", model=ScriptedModel([AIMessage(content="not JSON")]))
    assert report["status"] == "partial"
    assert "Structured AgentDraft parsing failed" in report["errors"][0]


def test_structured_model_summary_is_redacted_bounded_and_reported(run_pair_factory):
    from rundiff.loader import load_pair

    pair = load_pair(*run_pair_factory())
    summary = "API_KEY=synthetic-secret-value " + "x" * 1100
    draft = json.dumps(
        {
            "summary": summary,
            "hypotheses": [],
            "unresolved_questions": [],
            "recommended_checks": [],
        }
    )
    report = run_agent(pair, mode="live", model=ScriptedModel([AIMessage(content=draft)]))

    investigation = report["investigation_summary"]
    assert investigation["source"] == "live_model"
    assert investigation["is_simulated"] is False
    assert investigation["verified"] is False
    assert len(investigation["text"]) <= 1000
    assert "API_KEY=[REDACTED]" in investigation["text"]
    assert "synthetic-secret-value" not in investigation["text"]
    markdown = report_to_markdown(report)
    assert "## Investigation summary" in markdown
    assert "Model-generated summary; not deterministically verified." in markdown
    assert investigation["text"] in markdown


def test_unknown_evidence_in_model_draft_is_unverified(run_pair_factory):
    from rundiff.loader import load_pair

    pair = load_pair(*run_pair_factory())
    draft = json.dumps(
        {
            "summary": "Test.",
            "hypotheses": [
                {
                    "claim_id": "invented",
                    "classification": "SUPPORTED_HYPOTHESIS",
                    "text": "Unsupported reasoning.",
                    "evidence_ids": ["E_LOG_forged"],
                }
            ],
            "unresolved_questions": [],
            "recommended_checks": [],
        }
    )
    report = run_agent(pair, mode="live", model=ScriptedModel([AIMessage(content=draft)]))
    assert report["status"] == "complete"
    assert report["hypotheses"][0]["classification"] == "UNVERIFIED"
    assert report["validation"]["claim_warnings"]


def test_duplicate_tool_call_is_rejected_and_traced(run_pair_factory):
    from rundiff.loader import load_pair

    pair = load_pair(*run_pair_factory())
    duplicated = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "inspect_logs",
                "args": {"run_id": pair.run_a.run_id},
                "id": "c1",
                "type": "tool_call",
            },
            {
                "name": "inspect_logs",
                "args": {"run_id": pair.run_a.run_id},
                "id": "c2",
                "type": "tool_call",
            },
        ],
    )
    report = run_agent(
        pair, mode="live", model=ScriptedModel([duplicated, AIMessage(content=VALID_DRAFT)])
    )
    failures = [
        event for event in report["trace"] if event["step"] == "tool" and not event["success"]
    ]
    assert report["tool_calls"] == 1
    assert len(failures) == 1
    assert "Repeated tool call" in failures[0]["error"]


def test_tool_budget_exhaustion_returns_tool_observation(run_pair_factory):
    from rundiff.loader import load_pair

    pair = load_pair(*run_pair_factory())
    report = run_agent(
        pair,
        mode="live",
        model=ScriptedModel([tool_call(), AIMessage(content=VALID_DRAFT)]),
        max_tool_calls=0,
    )
    assert report["status"] == "complete"
    assert report["tool_calls"] == 0
    assert any("budget" in (event.get("error") or "") for event in report["trace"])


def test_decision_round_budget_yields_partial_report(run_pair_factory):
    from rundiff.loader import load_pair

    pair = load_pair(*run_pair_factory())
    report = run_agent(
        pair,
        mode="live",
        model=ScriptedModel([tool_call("inspect_logs", {"run_id": pair.run_a.run_id})]),
        max_decision_rounds=1,
    )
    assert report["status"] == "partial"
    assert report["tool_calls"] == 1
    assert any("decision rounds" in error for error in report["errors"])


def test_invalid_tool_arguments_are_returned_as_observations(run_pair_factory):
    from rundiff.loader import load_pair

    pair = load_pair(*run_pair_factory())
    report = run_agent(
        pair,
        mode="live",
        model=ScriptedModel(
            [
                tool_call("inspect_logs", {"run_id": pair.run_a.run_id, "start_line": 3}),
                AIMessage(content=VALID_DRAFT),
            ]
        ),
    )
    assert report["status"] == "complete"
    assert any(
        not event["success"] and "start_line and end_line" in event["error"]
        for event in report["trace"]
    )


def test_live_mode_requires_a_real_api_key(monkeypatch):
    from rundiff.agent import _live_model

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    try:
        _live_model()
    except LiveModelUnavailable as exc:
        assert "OPENAI_API_KEY" in str(exc)
    else:
        raise AssertionError("Live model unexpectedly initialized without credentials")
