from __future__ import annotations

import hashlib
import json
import stat
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from openai import OpenAI
from openai.types.chat import ChatCompletionMessage

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lgagent.config import LGAgentConfig, LGAgentPlusConfig, LegalMCQSettings
from lgagent.legal_mcq import LegalAgentError, SolverDecision, parse_question_text
from lgagent.legal_mcq.observability import CallJournal, redact_telemetry
from lgagent.model import (
    BudgetExceededError,
    BudgetedSeededChatModel,
    ChatMessage,
    ExecutionBudget,
    ModelCallError,
    ModelRequest,
    ModelResponse,
    OpenAIChatModel,
    TokenUsage,
)
from lgagent.runner import LGAgentPlusRunner
from lgagent.trace import RunTrace
from test_task22_legal_mcq import (
    CONFIG,
    QUESTION,
    FakeRoleModel,
    controller_payload,
    solver_payload,
    solver_v2_payload,
    verifier_payload,
)

SECRETS = ("fixture-controller-key", "fixture-solver-key", "fixture-verifier-key")


class ResponseModel:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.requests = []

    def complete(self, request):
        self.requests.append(request)
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response


def make_runner(
    *,
    controller=None,
    solver=None,
    solver_fallback=None,
    verifier=None,
    role_reasoning_effort=None,
    **settings,
):
    config = LGAgentConfig(
        generation=CONFIG,
        lgagent_plus=LGAgentPlusConfig(enabled=False),
        legal_mcq=LegalMCQSettings(
            enabled=True,
            controller_model=replace(
                CONFIG,
                api_key=SECRETS[0],
                max_tokens=1200,
                reasoning_effort=role_reasoning_effort,
            ),
            solver_model=replace(
                CONFIG,
                api_key=SECRETS[1],
                max_tokens=6144,
                reasoning_effort=role_reasoning_effort,
            ),
            solver_fallback_model=(
                replace(
                    CONFIG,
                    api_key="fixture-solver-fallback-key",
                    model="solver-fallback",
                    max_tokens=4096,
                    reasoning_effort=role_reasoning_effort,
                )
                if solver_fallback is not None
                else None
            ),
            verifier_model=replace(
                CONFIG,
                api_key=SECRETS[2],
                max_tokens=768,
                reasoning_effort=role_reasoning_effort,
            ),
            **settings,
        ),
    )
    controller = controller or FakeRoleModel([controller_payload()])
    solver = solver or FakeRoleModel([solver_payload()])
    verifier = verifier or FakeRoleModel([verifier_payload()])
    return LGAgentPlusRunner(
        config,
        reasoning_model=controller,
        evaluation_model=solver,
        legal_controller_model=controller,
        legal_solver_model=solver,
        legal_solver_fallback_model=solver_fallback,
        legal_verifier_model=verifier,
    )


def sdk_response(content="visible", *, finish_reason="stop"):
    return SimpleNamespace(
        id="provider-response-id",
        model="actual-model-version",
        choices=[
            SimpleNamespace(
                message=ChatCompletionMessage(
                    role="assistant",
                    content=content,
                    reasoning_content="private-reasoning-do-not-log",
                ),
                finish_reason=finish_reason,
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=70,
            completion_tokens=30,
            total_tokens=100,
            completion_tokens_details=SimpleNamespace(reasoning_tokens=25),
            prompt_tokens_details=SimpleNamespace(cached_tokens=40),
        ),
    )


def sdk_model(response, captured):
    def create(**kwargs):
        captured.append(kwargs)
        return response

    return OpenAIChatModel(
        SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    )


@pytest.mark.parametrize(
    ("content", "state"),
    [(json.dumps(solver_payload()), "text")],
)
def test_sdk_metadata_survives_budget_wrapper_without_changing_request(content, state):
    captured = []
    adapter = sdk_model(sdk_response(content, finish_reason="length"), captured)
    budget = ExecutionBudget(max_calls=5, max_tokens=1000, max_seconds=180)
    wrapped = BudgetedSeededChatModel(adapter, budget, base_seed=42)
    response = wrapped.complete(
        ModelRequest(
            model="requested-model",
            messages=(ChatMessage("user", "fixture"),),
            max_tokens=100,
        )
    )
    assert response.finish_reason == "length"
    assert response.response_model == "actual-model-version"
    assert response.content_state == state
    assert response.visible_content == (content or "")
    assert response.reasoning_tokens == 25
    assert response.cached_prompt_tokens == 40
    assert response.reasoning_present is True
    assert response.usage_reported is True
    assert response.seed_requested == captured[0]["seed"]
    assert response.usage == TokenUsage(70, 30, 100)
    assert set(captured[0]) == {
        "model", "messages", "max_tokens", "temperature", "top_p", "seed",
        "timeout",
    }
    assert 0 < captured[0]["timeout"] <= 180
    SolverDecision.from_text(response.content)


def test_openai_adapter_forwards_solver_json_schema_unchanged():
    captured = []
    response_format = SolverDecision.response_format()
    observed = sdk_model(
        sdk_response(json.dumps(solver_v2_payload())),
        captured,
    ).complete(
        ModelRequest(
            model="solver",
            messages=(ChatMessage("user", "fixture"),),
            response_format=response_format,
        )
    )

    assert observed.finish_reason == "stop"
    assert captured[0]["response_format"] == response_format
    assert captured[0]["response_format"]["json_schema"]["strict"] is True


def test_openai_adapter_forwards_reasoning_effort_unchanged():
    captured = []
    sdk_model(sdk_response(), captured).complete(
        ModelRequest(
            model="solver",
            messages=(ChatMessage("user", "fixture"),),
            reasoning_effort="low",
        )
    )

    assert captured[0]["reasoning_effort"] == "low"


def test_role_reasoning_effort_reaches_requests_and_trace():
    controller = FakeRoleModel([controller_payload()])
    solver = FakeRoleModel([solver_v2_payload()])
    verifier = FakeRoleModel([verifier_payload()])
    result = make_runner(
        controller=controller,
        solver=solver,
        verifier=verifier,
        role_reasoning_effort="low",
    ).run_legal_request(parse_question_text(QUESTION))

    assert controller.requests[0].reasoning_effort == "low"
    assert solver.requests[0].reasoning_effort == "low"
    assert verifier.requests[0].reasoning_effort == "low"
    assert {
        call.reasoning_effort_requested for call in result.trace.model_calls
    } == {"low"}


@pytest.mark.parametrize(
    ("content", "state"),
    [("", "empty"), ("  \n", "empty"), (None, "null")],
)
def test_empty_visible_content_is_an_explicit_provider_error(content, state):
    captured = []
    budget = ExecutionBudget(max_calls=5, max_tokens=1000, max_seconds=180)
    wrapped = BudgetedSeededChatModel(
        sdk_model(sdk_response(content, finish_reason="length"), captured),
        budget,
        base_seed=42,
    )
    with pytest.raises(ModelCallError) as caught:
        wrapped.complete(
            ModelRequest(
                model="requested-model",
                messages=(ChatMessage("user", "fixture"),),
                max_tokens=100,
            )
        )
    response = caught.value.response
    assert response.content_state == state
    assert response.visible_content == (content or "")
    assert response.finish_reason == "length"
    assert response.reasoning_tokens == 25
    assert response.tool_calls_present is False
    assert caught.value.diagnostics["provider_error_code"] == "empty_assistant_content"
    assert caught.value.diagnostics["seed_requested"] == captured[0]["seed"]
    assert budget.snapshot()["tokens_used"] == 100


def test_missing_usage_is_unknown_not_reported_zero():
    response = sdk_response()
    response.usage = None
    observed = sdk_model(response, []).complete(ModelRequest("fixture", ()))
    assert observed.usage_reported is False
    assert observed.usage.total_tokens == 0
    assert observed.reasoning_tokens is None
    assert observed.cached_prompt_tokens is None


def test_multipart_telemetry_keeps_visible_text_only():
    response = sdk_response()
    response.choices[0].message = SimpleNamespace(
        content=[
            "first",
            {"text": "untyped-text"},
            {"type": "text", "text": "typed-text"},
            {"type": "output_text", "text": "output-text"},
            {"type": "reasoning", "text": "private-reasoning-do-not-log"},
            {"type": "image_url", "image_url": "https://offline.invalid/image"},
        ]
    )
    observed = sdk_model(response, []).complete(ModelRequest("fixture", ()))
    assert observed.content_state == "parts"
    assert observed.visible_content == "first\nuntyped-text\ntyped-text\noutput-text"


def test_network_timeout_has_no_invented_response_metadata(tmp_path):
    def handler(request):
        raise httpx.ReadTimeout("fixture timeout", request=request)

    with httpx.Client(transport=httpx.MockTransport(handler)) as http:
        sdk = OpenAI(
            api_key=SECRETS[0],
            base_url="https://offline.invalid/v1",
            http_client=http,
            max_retries=0,
        )
        path = tmp_path / "calls.jsonl"
        with CallJournal(path, secrets=SECRETS) as journal:
            trace = RunTrace(_event_sink=journal.write)
            with pytest.raises(LegalAgentError):
                make_runner(
                    controller=OpenAIChatModel(sdk), max_attempts=1
                ).run_legal_request(parse_question_text(QUESTION), trace=trace)
    call = trace.model_calls[0]
    assert call.outcome == "provider_error"
    assert "APITimeoutError" in call.error_message
    assert call.model_invoked is True
    assert call.finish_reason is None
    assert call.content_state is None
    assert call.usage_reported is None
    assert call.request_id is None
    assert call.schema_valid is None
    assert call.budget_after["calls_used"] == 1


def test_runner_enforces_hard_deadline_during_provider_call():
    release = threading.Event()

    class BlockingModel:
        def __init__(self):
            self.requests = []

        def complete(self, request):
            self.requests.append(request)
            release.wait(1.0)
            return ModelResponse(json.dumps(controller_payload()))

    controller = BlockingModel()
    trace = RunTrace()
    started = time.perf_counter()
    try:
        with pytest.raises(BudgetExceededError) as caught:
            make_runner(
                controller=controller,
                max_wall_time_seconds=0.05,
                model_call_timeout_seconds=0.5,
            ).run_legal_request(parse_question_text(QUESTION), trace=trace)
    finally:
        release.set()
    assert time.perf_counter() - started < 0.5
    assert caught.value.reason == "max_seconds"
    assert len(controller.requests) == 1
    assert 0 < controller.requests[0].timeout_seconds <= 0.05
    call = trace.model_calls[0]
    assert call.outcome == "deadline_exceeded"
    assert call.model_invoked is True
    assert call.deadline_exceeded_during_call is True
    assert call.budget_after["tokens_reserved"] == 0


def test_no_choices_keeps_usage_and_response_identity():
    response = sdk_response()
    response.choices = []
    with pytest.raises(ModelCallError) as caught:
        sdk_model(response, []).complete(ModelRequest("fixture", ()))
    assert caught.value.response.content_state == "missing"
    assert caught.value.response.request_id == "provider-response-id"
    assert caught.value.usage.total_tokens == 100


def test_schema_retry_has_correlated_start_finish_events_and_safe_optional_output(tmp_path):
    invalid = '{"note":"' + " ".join(SECRETS) + '"'
    truncated = ModelResponse(
        content=invalid,
        usage=TokenUsage(30, 20, 50),
        request_id="truncated-id",
        finish_reason="length",
        response_model="reported-cheap-model",
        content_state="text",
        visible_content=invalid,
        usage_reported=True,
        reasoning_tokens=5,
    )
    valid = ModelResponse(json.dumps(controller_payload()), usage=TokenUsage(10, 5, 15))
    controller = ResponseModel(truncated, valid)
    path = tmp_path / "calls.jsonl"
    with CallJournal(path, secrets=SECRETS) as journal:
        trace = RunTrace(_event_sink=journal.write)
        result = make_runner(controller=controller).run_legal_request(
            parse_question_text(QUESTION), trace=trace, log_model_output=True
        )
    events = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(events) == 8
    for start, end in zip(events[::2], events[1::2]):
        assert start["event"] == "model_call_started"
        assert end["event"] == "model_call_finished"
        assert start["call_id"] == end["call_id"]
        assert start["run_id"] == end["run_id"] == trace.run_id
        assert start["sequence"] < end["sequence"]
    first = events[1]
    assert first["outcome"] == "schema_error"
    assert first["schema_valid"] is False
    assert first["finish_reason"] == "length"
    assert first["content_chars"] == len(invalid)
    assert "[REDACTED]" in first["output_text"]
    assert first["content_sha256"] == hashlib.sha256(
        first["output_text"].encode()
    ).hexdigest()
    assert first["budget_before"]["calls_used"] == 0
    assert first["budget_after"]["calls_used"] == 1
    assert result.execution_budget["calls_used"] == 4
    assert result.diagnostics()["trace"]["total_tokens"] == 95
    assert result.answer.selected_options == ("C",)
    serialized = path.read_text() + json.dumps(result.diagnostics())
    assert not any(secret in serialized for secret in SECRETS)
    assert "golden_answers" not in path.read_text()
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_schema_retry_preserves_complete_revision_and_final_verifier_budget():
    controller = FakeRoleModel(["{", controller_payload()])
    solver = FakeRoleModel([solver_payload("D"), solver_payload("C")])
    verifier = FakeRoleModel([
        verifier_payload(False, suggested="C"),
        verifier_payload(True),
    ])

    result = make_runner(
        controller=controller,
        solver=solver,
        verifier=verifier,
        max_model_calls=7,
    ).run_legal_request(parse_question_text(QUESTION))

    assert result.revision_count == 1
    assert result.verification.accepted is True
    assert result.execution_budget["calls_used"] == 6
    assert [call.agent for call in result.trace.model_calls] == [
        "legal_mcq_controller",
        "legal_mcq_controller",
        "legal_mcq_solver",
        "legal_mcq_verifier_1",
        "legal_mcq_solver_revision",
        "legal_mcq_verifier_2",
    ]
    assert all(call.outcome != "budget_blocked" for call in result.trace.model_calls)


def test_auto_mode_records_budgeted_prompt_fallback_when_schema_is_unsupported():
    unsupported = ModelCallError(
        "response_format json_schema is unsupported by this model",
        diagnostics={
            "http_status": 400,
            "provider_error_code": "unsupported_response_format",
        },
    )
    solver = ResponseModel(
        unsupported,
        ModelResponse(json.dumps(solver_v2_payload())),
    )

    result = make_runner(
        solver=solver,
        solver_structured_output_mode="auto",
    ).run_legal_request(parse_question_text(QUESTION))

    assert result.answer.selected_options == ("C",)
    assert len(solver.requests) == 2
    assert solver.requests[0].response_format["type"] == "json_schema"
    assert solver.requests[1].response_format is None
    solver_calls = [
        call
        for call in result.trace.model_calls
        if call.agent == "legal_mcq_solver"
    ]
    assert [call.response_format_type for call in solver_calls] == [
        "json_schema",
        None,
    ]
    assert any(
        route.route == "legal-mcq-structured-output-fallback"
        for route in result.trace.routes
    )


def test_required_json_schema_mode_does_not_silently_fallback():
    unsupported = ModelCallError(
        "response_format json_schema is unsupported by this model",
        diagnostics={"provider_error_code": "unsupported_response_format"},
    )
    solver = ResponseModel(unsupported, ModelResponse(json.dumps(solver_v2_payload())))

    with pytest.raises(LegalAgentError):
        make_runner(
            solver=solver,
            solver_structured_output_mode="json_schema",
        ).run_legal_request(parse_question_text(QUESTION))

    assert len(solver.requests) == 1
    assert solver.requests[0].response_format["type"] == "json_schema"


def test_revision_is_skipped_as_partial_when_full_pair_no_longer_fits_tokens():
    controller = FakeRoleModel([controller_payload()])
    solver = FakeRoleModel([solver_payload("D")])
    verifier = ResponseModel(
        ModelResponse(
            json.dumps(verifier_payload(False, suggested="C")),
            usage=TokenUsage(prompt_tokens=14000, completion_tokens=1000, total_tokens=15000),
            usage_reported=True,
        )
    )

    result = make_runner(
        controller=controller,
        solver=solver,
        verifier=verifier,
        max_total_tokens=20000,
    ).run_legal_request(parse_question_text(QUESTION))

    assert result.revision_count == 0
    assert result.answer.needs_review is True
    assert "REVISION_SKIPPED_MAX_TOKENS" in result.answer.warnings
    assert len(solver.requests) == 1
    assert len(verifier.requests) == 1
    assert any(
        route.route == "legal-mcq-revision-skipped-budget"
        for route in result.trace.routes
    )


def test_metadata_only_is_default_and_does_not_store_hidden_reasoning(tmp_path):
    response = sdk_response(None, finish_reason="length")
    response.choices[0].message.refusal = "private-refusal-do-not-log"
    adapter = sdk_model(response, [])
    path = tmp_path / "calls.jsonl"
    with CallJournal(path) as journal:
        trace = RunTrace(_event_sink=journal.write)
        with pytest.raises(LegalAgentError) as caught:
            make_runner(controller=adapter, max_attempts=1).run_legal_request(
                parse_question_text(QUESTION), trace=trace, log_model_output=True
            )
    call = trace.model_calls[0]
    assert call.content_state == "null"
    assert call.content_chars == 0
    assert call.output_text == ""
    assert call.refusal_present is True
    assert call.reasoning_present is True
    assert call.reasoning_tokens == 25
    assert call.outcome == "provider_error"
    assert call.schema_valid is None
    assert call.provider_error_code == "assistant_refusal"
    assert caught.value.diagnostics["trace"]["total_tokens"] == 100
    assert caught.value.diagnostics["execution_budget"]["tokens_used"] == 100
    assert "private-reasoning-do-not-log" not in path.read_text()
    assert "private-refusal-do-not-log" not in path.read_text()

    plain = make_runner().run_legal_request(parse_question_text(QUESTION))
    assert all(call.output_text is None for call in plain.trace.model_calls)
    assert all(call.content_chars > 0 for call in plain.trace.model_calls)


def test_tool_call_without_visible_content_is_rejected_and_classified():
    response = sdk_response(None, finish_reason="tool_calls")
    response.choices[0].message.tool_calls = [
        SimpleNamespace(model_dump=lambda: {"function": {"name": "unexpected"}})
    ]
    with pytest.raises(ModelCallError) as caught:
        sdk_model(response, []).complete(ModelRequest("fixture", ()))
    assert caught.value.response.tool_calls_present is True
    assert caught.value.response.finish_reason == "tool_calls"
    assert caught.value.diagnostics["provider_error_code"] == "unexpected_tool_calls"


def test_provider_error_preserves_http_details_and_redacts_all_role_credentials(tmp_path):
    def handler(request):
        return httpx.Response(
            401,
            headers={"x-request-id": "provider-error-id"},
            json={
                "error": {
                    "message": "Invalid API key: " + " ".join(SECRETS),
                    "code": "invalid_api_key",
                    "type": "authentication_error",
                }
            },
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as http:
        sdk = OpenAI(
            api_key=SECRETS[0],
            base_url="https://offline.invalid/v1",
            http_client=http,
            max_retries=2,
        )
        path = tmp_path / "calls.jsonl"
        with CallJournal(path, secrets=SECRETS) as journal:
            trace = RunTrace(_event_sink=journal.write)
            with pytest.raises(LegalAgentError) as caught:
                make_runner(controller=OpenAIChatModel(sdk)).run_legal_request(
                    parse_question_text(QUESTION), trace=trace
                )
    assert len(trace.model_calls) == 1
    call = trace.model_calls[0]
    assert call.http_status == 401
    assert call.request_id == "provider-error-id"
    assert call.provider_error_code == "invalid_api_key"
    assert call.outcome == "provider_error"
    assert call.schema_valid is None
    assert call.model_invoked is True
    assert call.finish_reason is None
    assert call.seed_requested is not None
    assert caught.value.diagnostics["failed_agent"] == "legal_mcq_controller"
    assert caught.value.diagnostics["execution_budget"]["calls_used"] == 1
    assert not any(
        key in path.read_text() + json.dumps(caught.value.diagnostics) for key in SECRETS
    )


def test_pre_call_budget_block_is_not_counted_as_a_provider_call():
    solver = ResponseModel()
    trace = RunTrace()
    with pytest.raises(BudgetExceededError) as caught:
        make_runner(solver=solver, max_model_calls=1).run_legal_request(
            parse_question_text(QUESTION), trace=trace
        )
    assert solver.requests == []
    assert len(trace.model_calls) == 1
    call = trace.model_calls[-1]
    assert call.agent == "legal_mcq_controller"
    assert call.outcome == "budget_blocked"
    assert call.model_invoked is False
    assert call.schema_valid is None
    assert call.budget_before["calls_used"] == call.budget_after["calls_used"] == 0
    assert call.reserve_calls_after == 4
    assert caught.value.diagnostics["failed_agent"] == "legal_mcq_controller"


def test_post_response_budget_error_keeps_finish_reason_and_usage():
    response = ModelResponse(
        json.dumps(controller_payload()),
        usage=TokenUsage(100, 24900, 25000),
        request_id="over-budget-id",
        finish_reason="length",
        response_model="actual-model",
        usage_reported=True,
        reasoning_tokens=1700,
    )
    trace = RunTrace()
    with pytest.raises(BudgetExceededError) as caught:
        make_runner(
            controller=ResponseModel(response),
            max_total_tokens=20000,
        ).run_legal_request(parse_question_text(QUESTION), trace=trace)
    call = trace.model_calls[0]
    assert call.outcome == "budget_exceeded_after_response"
    assert call.finish_reason == "length"
    assert call.request_id == "over-budget-id"
    assert call.response_model == "actual-model"
    assert call.usage.total_tokens == trace.total_tokens == 25000
    assert call.reasoning_tokens == 1700
    assert call.schema_valid is None
    assert call.seed_requested is not None
    assert caught.value.diagnostics["execution_budget"]["tokens_used"] == 25000


def test_interrupt_keeps_begin_event_flushed_and_records_cancelled(tmp_path):
    path = tmp_path / "calls.jsonl"

    class InterruptModel:
        def complete(self, request):
            events = [json.loads(line) for line in path.read_text().splitlines()]
            assert events[-1]["event"] == "model_call_started"
            raise KeyboardInterrupt()

    with CallJournal(path) as journal:
        trace = RunTrace(_event_sink=journal.write)
        with pytest.raises(KeyboardInterrupt):
            make_runner(controller=InterruptModel()).run_legal_request(
                parse_question_text(QUESTION), trace=trace
            )
    events = [json.loads(line) for line in path.read_text().splitlines()]
    assert [event["event"] for event in events] == [
        "model_call_started", "model_call_finished"
    ]
    assert events[-1]["outcome"] == "cancelled"
    assert events[-1]["schema_valid"] is None


def test_parallel_questions_have_distinct_correlated_traces(tmp_path):
    path = tmp_path / "calls.jsonl"
    with CallJournal(path, secrets=SECRETS) as journal:
        def run(index):
            trace = RunTrace(
                _event_sink=lambda event: journal.write({**event, "question_id": str(index)})
            )
            result = make_runner().run_legal_request(
                parse_question_text(QUESTION, question_id=str(index)),
                trace=trace,
            )
            return result.diagnostics()["trace"]["run_id"]

        with ThreadPoolExecutor(max_workers=3) as pool:
            run_ids = list(pool.map(run, range(3)))
    assert len(set(run_ids)) == 3
    events = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(events) == 18
    for index, run_id in enumerate(run_ids):
        scoped = [event for event in events if event["run_id"] == run_id]
        assert len(scoped) == 6
        assert {event["question_id"] for event in scoped} == {str(index)}
        assert [event["sequence"] for event in scoped] == list(range(1, 7))
    with pytest.raises(FileExistsError):
        CallJournal(path)


def test_telemetry_sanitizer_handles_nested_and_overlapping_secrets():
    safe = redact_telemetry(
        {"message": "abcd abc", "nested": [{"api_key": "unlisted-key"}]},
        ("abc", "abcd"),
    )
    assert safe == {
        "message": "[REDACTED] [REDACTED]",
        "nested": [{"api_key": "[REDACTED]"}],
    }
