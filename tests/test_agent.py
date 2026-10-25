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
        self.calls = []
        self.tool_bindings = []
        self.bindings = []

    def bind_tools(self, tools, **kwargs):
        self.tool_bindings.append({"tool_count": len(tools), **kwargs})
        return self

    def bind(self, **kwargs):
        self.bindings.append(kwargs)
        return self

    def invoke(self, messages):
        self.calls.append(messages)
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
    assert "TimeoutError" in report["errors"][0]
    assert any(
        event["success"] is False and event["step"] == "agent_decision" for event in report["trace"]
    )


def test_invalid_structured_output_retries_once_then_stays_partial(run_pair_factory):
    from rundiff.loader import load_pair

    pair = load_pair(*run_pair_factory())
    model = ScriptedModel(
        [
            AIMessage(content="not JSON"),
            AIMessage(content="still not JSON"),
            AIMessage(content="still invalid"),
        ]
    )
    report = run_agent(pair, mode="live", model=model)

    assert report["status"] == "partial"
    assert "json_invalid" in report["errors"][0]
    assert "not JSON" not in " ".join(report["errors"])
    finalizations = [event for event in report["trace"] if event["step"] == "finalization"]
    invalid_parses = [
        event for event in report["trace"] if event.get("structured_parse_status") == "invalid"
    ]
    assert len(finalizations) == 2
    assert len(invalid_parses) == 3
    assert len(model.bindings) == 2


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
    decision = next(event for event in report["trace"] if event["step"] == "agent_decision")
    parse = next(event for event in report["trace"] if event["step"] == "structured_output")
    assert decision["finish_reason"] == "unavailable"
    assert decision["response_content_length"] == len(draft)
    assert decision["tool_calls_count"] == 0
    assert decision["output_tokens"] == "unavailable"
    assert decision["reasoning_tokens"] == "unavailable"
    assert decision["request_max_tokens"] == 3072
    assert parse["structured_parse_status"] == "valid"


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


def test_live_agent_finishes_after_multiple_log_tool_calls(run_pair_factory):
    from rundiff.loader import load_pair

    pair = load_pair(*run_pair_factory())
    model = ScriptedModel(
        [
            tool_call("inspect_logs", {"run_id": pair.run_a.run_id}, "log-a"),
            tool_call("inspect_logs", {"run_id": pair.run_b.run_id}, "log-b"),
            AIMessage(content=VALID_DRAFT),
        ]
    )

    report = run_agent(pair, mode="live", model=model)

    assert report["status"] == "complete"
    assert report["tool_calls"] == 2
    assert report["decision_rounds"] == 3
    assert report["validation"]["valid"] is True
    assert report["investigation_summary"]["text"] == "A structured test draft."
    assert sum(event["step"] == "tool" and event["success"] for event in report["trace"]) == 2
    system_prompt = model.calls[0][0].content
    assert "have already been executed once" in system_prompt
    assert "Do not call them again" in system_prompt
    assert "return only one compact JSON object" in system_prompt
    assert "Do not include analysis, markdown, code fences" in system_prompt


def test_repeated_tools_finalize_without_tools_and_validate_final_claim(run_pair_factory):
    from rundiff.loader import load_pair

    pair = load_pair(*run_pair_factory())
    final_draft = json.dumps(
        {
            "summary": "Logs were inspected for both runs.",
            "hypotheses": [
                {
                    "claim_id": "unsupported-final-claim",
                    "classification": "SUPPORTED_HYPOTHESIS",
                    "text": "This deliberately cites unknown evidence.",
                    "evidence_ids": ["E_LOG_forged"],
                }
            ],
            "unresolved_questions": [],
            "recommended_checks": [],
        }
    )
    model = ScriptedModel(
        [
            tool_call("inspect_logs", {"run_id": pair.run_a.run_id}, "log-a"),
            tool_call("inspect_logs", {"run_id": pair.run_b.run_id}, "log-b"),
            tool_call("compare_configs", {}, "duplicate-compare"),
            AIMessage(content=final_draft),
        ]
    )

    report = run_agent(pair, mode="live", model=model)

    assert report["status"] == "complete"
    assert report["tool_calls"] == 2
    assert report["decision_rounds"] == 3
    assert sum(event["step"] == "tool" and event["success"] for event in report["trace"]) == 2
    assert any(
        event["step"] == "tool" and not event["success"] and "Repeated tool call" in event["error"]
        for event in report["trace"]
    )
    finalization = next(event for event in report["trace"] if event["step"] == "finalization")
    assert finalization["tool_calls_count"] == 0
    assert finalization["structured_parse_status"] == "pending"
    assert finalization["request_max_tokens"] == 3072
    assert finalization["finalization_reason"] == "repeated_tool_calls"
    assert len(model.tool_bindings) == 1
    assert len(model.bindings) == 1
    assert model.bindings[0]["response_format"] == {"type": "json_object"}
    assert model.bindings[0]["max_tokens"] == 3072
    assert "tools" not in model.bindings[0]
    assert report["hypotheses"][0]["classification"] == "UNVERIFIED"
    assert report["validation"]["claim_warnings"]


def test_output_token_limit_diagnostics_are_recorded_without_content(run_pair_factory):
    from rundiff.loader import load_pair

    pair = load_pair(*run_pair_factory())
    limited_response = AIMessage(
        content="{",
        response_metadata={
            "finish_reason": "length",
            "token_usage": {
                "prompt_tokens": 100,
                "completion_tokens": 3072,
                "total_tokens": 3172,
                "completion_tokens_details": {"reasoning_tokens": 1300},
            },
        },
        usage_metadata={
            "input_tokens": 100,
            "output_tokens": 3072,
            "total_tokens": 3172,
            "output_token_details": {"reasoning": 1300},
        },
    )
    report = run_agent(
        pair,
        mode="live",
        model=ScriptedModel([limited_response, AIMessage(content=VALID_DRAFT)]),
    )

    decision = next(event for event in report["trace"] if event["step"] == "agent_decision")
    parse_failure = next(
        event
        for event in report["trace"]
        if event["step"] == "structured_output" and not event["success"]
    )
    assert decision["finish_reason"] == "length"
    assert decision["response_content_length"] == 1
    assert decision["tool_calls_count"] == 0
    assert decision["output_tokens"] == 3072
    assert decision["reasoning_tokens"] == 1300
    assert decision["structured_parse_status"] == "pending"
    assert decision["request_max_tokens"] == 3072
    assert parse_failure["structured_parse_status"] == "invalid"
    assert parse_failure["failure_type"] == "json_invalid"
    assert "content" not in decision
    assert "raw_response" not in decision


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
