import json
import threading
from dataclasses import replace
from pathlib import Path

import httpx
from openai import OpenAI
import pytest

from lgagent.legal_mcq import (
    ControllerPlan,
    LegalAgentError,
    SolverDecision,
    VerificationResult,
    parse_question_text,
    parse_request,
)
from lgagent.legal_mcq.decision import DecisionValidator
from lgagent.legal_mcq.models import SolveMode, compact_json
from lgagent.model import ModelCallError, ModelResponse, OpenAIChatModel, TokenUsage, estimate_message_tokens
from lgagent.protocol import StructuredOutputError
from lgagent.trace import RunTrace
from test_legal_mcq_observability import ResponseModel, make_runner
from test_task22_legal_mcq import (
    QUESTION, FakeRoleModel, controller_payload, solver_v2_payload,
    solver_v3_payload, verifier_payload,
)


def controller_v2():
    return {
        "protocol_version": "controller-v2",
        "facts": ["Delegation is subject to internal limits."],
        "issues": [{"description": "Effect of internal authority limits?", "options": list("ABCD")}],
        "option_claims": {label: ["Necessary proposition."] for label in "ABCD"},
        "checks": ["Check third-party good faith."],
    }


def controller_v3():
    return {
        "protocol_version": "controller-v3",
        "facts": ["Delegation is subject to internal limits."],
        "issues": [
            {
                "description": "Effect of internal authority limits?",
                "options": list("ABCD"),
            }
        ],
        "checks": ["Check third-party good faith."],
    }


def verifier_v2(*, accepted=True, code=""):
    return {
        "protocol_version": "verifier-v2",
        "accepted": accepted,
        "error_codes": [] if accepted else [code or "LEGAL_ERROR"],
        "challenged_options": [] if accepted else ["C"],
        "suggested_selected_options": [],
        "note": "" if accepted else "决定性规则适用错误。",
    }


def test_controller_v2_is_compact_and_assigns_stable_identifiers():
    data = controller_v2()
    parsed = ControllerPlan.from_text(compact_json(data))
    assert parsed.facts[0].fact_id == "F1"
    assert parsed.issues[0].issue_id == "I1"
    assert parsed.option_claims["A"] == ("A1 Necessary proposition.",)
    assert parsed.raw == data
    assert parsed.prompt_payload()["issues"][0]["issue_id"] == "I1"
    assert parsed.prompt_payload()["option_claims"]["D"][0].startswith("D1 ")
    assert len(compact_json(data)) < len(compact_json(controller_payload()))
    assert ControllerPlan.from_text(compact_json(controller_payload())).option_claims


def test_controller_v3_claims_are_bound_from_exact_question_options():
    parsed = ControllerPlan.from_text(compact_json(controller_v3()))
    request = parse_question_text(QUESTION)
    assert parsed.option_claims == {}

    bound = parsed.bind_option_claims(request.options)

    assert bound.raw == controller_v3()
    assert bound.option_claims == {
        option.label: (f"{option.label}1 {option.text}",)
        for option in request.options
    }
    assert bound.prompt_payload()["claim_binding"] == "deterministic-option-text-v1"


@pytest.mark.parametrize("mutate", [
    lambda p: p.update(answer="C"),
    lambda p: p.update(protocol_version="controller-v3"),
    lambda p: p["issues"][0].update(conclusion="A"),
    lambda p: p["option_claims"].update(A=[]),
    lambda p: p["option_claims"].update(A={"claim_id": "A1"}),
    lambda p: p.update(facts=[]),
    lambda p: p["issues"][0].update(options=[]),
    lambda p: p["issues"][0].update(options=["A", "A"]),
])
def test_controller_v2_rejects_nonconforming_output(mutate):
    data = controller_v2()
    mutate(data)
    with pytest.raises(StructuredOutputError):
        ControllerPlan.from_text(compact_json(data))


def test_controller_v3_schema_omits_model_generated_claims():
    format_ = ControllerPlan.response_format(("A", "B", "C"))
    assert format_["json_schema"]["strict"] is True
    assert format_["json_schema"]["name"] == "legal_mcq_controller_v3"
    schema = format_["json_schema"]["schema"]
    assert set(schema["properties"]) == {
        "protocol_version",
        "facts",
        "issues",
        "checks",
    }
    assert "option_claims" not in schema["properties"]

    def check(node):
        if isinstance(node, dict):
            if node.get("type") == "object":
                assert node["additionalProperties"] is False
                assert set(node["required"]) == set(node["properties"])
            for value in node.values():
                check(value)
        elif isinstance(node, list):
            for value in node:
                check(value)
    check(schema)


def test_generated_claim_ids_cannot_be_replaced_by_same_number_of_wrong_ids():
    plan = ControllerPlan.from_text(compact_json(controller_v2()))
    data = solver_v2_payload()
    data["options"][0]["claims"][0]["claim_id"] = "A99"
    result = DecisionValidator().validate(
        parse_request(parse_question_text(QUESTION)), plan,
        SolverDecision.from_text(compact_json(data)), (), effective_mode=SolveMode.CLOSED_BOOK,
    )
    assert "CONTROLLER_CLAIM_ID_MISMATCH" in result.error_codes


def test_controller_and_solver_schema_survive_real_sdk_wire_and_budget_wrapper():
    bodies = []

    def handle(request):
        body = json.loads(request.content)
        bodies.append(body)
        response = [controller_v3(), solver_v3_payload(), verifier_v2()][len(bodies) - 1]
        return httpx.Response(200, json={
            "id": "offline", "model": body["model"], "object": "chat.completion", "created": 0,
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": compact_json(response)}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        })

    with OpenAI(api_key="fixture", base_url="https://offline.invalid",
                max_retries=0, http_client=httpx.Client(transport=httpx.MockTransport(handle))) as client:
        model = OpenAIChatModel(client)
        result = make_runner(controller=model, solver=model, verifier=model).run_legal_request(
            parse_question_text(QUESTION)
        )
    assert result.answer.status.value == "completed"
    assert bodies[0]["response_format"]["json_schema"]["name"] == "legal_mcq_controller_v3"
    assert bodies[1]["response_format"]["json_schema"]["name"] == "legal_mcq_solver_v3"
    assert bodies[2]["response_format"]["json_schema"]["name"] == "legal_mcq_verifier_v2"
    assert bodies[1]["max_tokens"] == 4096
    assert all("golden_answers" not in json.dumps(body) for body in bodies)


def test_solver_v3_removes_duplicate_issue_and_rationale_sections():
    v2 = solver_v2_payload()
    v3 = solver_v3_payload()
    parsed = SolverDecision.from_text(compact_json(v3))
    assert set(v3) == {
        "protocol_version",
        "selected_options",
        "options",
        "confidence",
    }
    assert parsed.issue_analyses == ()
    assert parsed.rationale
    assert len(compact_json(v3)) < len(compact_json(v2))


def test_solver_prompt_contains_question_only_once():
    solver = FakeRoleModel([solver_v3_payload()])
    make_runner(solver=solver).run_legal_request(parse_question_text(QUESTION))
    payload = solver.requests[0].messages[1].content
    assert '"question_metadata"' not in payload
    assert payload.count("刘某委托关某经营个人独资企业") == 1


def test_verifier_v2_schema_is_strict_and_compact():
    response_format = VerificationResult.response_format(tuple("ABCD"))
    assert response_format["json_schema"]["name"] == "legal_mcq_verifier_v2"
    schema = response_format["json_schema"]["schema"]
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == set(schema["properties"])
    assert VerificationResult.from_text(compact_json(verifier_v2())).accepted
    invalid = verifier_v2()
    invalid["note"] = "x"
    with pytest.raises(StructuredOutputError):
        VerificationResult.from_text(compact_json(invalid))


def test_verifier_length_retry_is_compact_and_bounded():
    verifier = ResponseModel(
        ModelResponse(
            '{"protocol_version":"verifier-v2","accepted":',
            finish_reason="length",
        ),
        ModelResponse(compact_json(verifier_v2())),
    )
    result = make_runner(
        solver=FakeRoleModel([solver_v3_payload()]),
        verifier=verifier,
    ).run_legal_request(parse_question_text(QUESTION))
    assert result.answer.status.value == "completed"
    assert [request.max_tokens for request in verifier.requests] == [384, 512]
    assert all(request.response_format is not None for request in verifier.requests)
    assert all(message.role != "assistant" for message in verifier.requests[1].messages)


def test_verifier_auto_mode_records_explicit_schema_fallback():
    verifier = ResponseModel(
        ModelCallError(
            "response_format json_schema is unsupported",
            diagnostics={"provider_error_code": "unsupported_response_format"},
        ),
        ModelResponse(compact_json(verifier_payload())),
    )
    result = make_runner(
        solver=FakeRoleModel([solver_v3_payload()]),
        verifier=verifier,
    ).run_legal_request(parse_question_text(QUESTION))
    assert result.answer.status.value == "completed"
    assert verifier.requests[0].response_format is not None
    assert verifier.requests[1].response_format is None
    assert any(
        route.route == "legal-mcq-structured-output-fallback"
        for route in result.trace.routes
    )


def timeout_error():
    return ModelCallError("APITimeoutError: Request timed out.")


def test_solver_timeout_switches_once_to_fallback_without_primary_retry():
    primary = ResponseModel(timeout_error())
    fallback = FakeRoleModel([solver_v3_payload()])
    result = make_runner(
        solver=primary,
        solver_fallback=fallback,
        verifier=FakeRoleModel([verifier_v2()]),
    ).run_legal_request(parse_question_text(QUESTION))
    assert result.answer.status.value == "completed"
    assert len(primary.requests) == 1
    assert len(fallback.requests) == 1
    assert primary.requests[0].timeout_seconds == 70
    assert fallback.requests[0].timeout_seconds == 45
    assert result.execution_budget["calls_used"] == 4
    assert any(r.route == "legal-mcq-solver-fallback" for r in result.trace.routes)


def test_mispartitioned_v2_claims_cannot_break_fallback_validation():
    question = """关于法律职业人员职业道德，下列哪一说法是不正确的？

A. 法官职业道德更强调法官独立性、中立地位
B. 检察官职业道德体现职业义务、责任及行为准则
C. 律师职业道德只规范律师执业行为，不规范律师事务所行为
D. 公证员职业道德应受重视，因为公证活动的特点是公信力"""
    controller = controller_v2()
    controller["option_claims"] = {
        "A": ["法官职业道德强调独立与中立"],
        "B": ["检察官职业道德体现职业义务与责任"],
        "C": ["律师职业道德只规范律师执业行为"],
        "D": [
            "律师职业道德不规范律师事务所行为",
            "公证员职业道德应受重视",
        ],
    }
    fallback_decision = solver_v3_payload("C")
    for option in fallback_decision["options"]:
        verdict = (
            "contradicted" if option["label"] == "C" else "supported"
        )
        option["verdict"] = verdict
        option["claims"][0]["verdict"] = verdict

    result = make_runner(
        controller=FakeRoleModel([controller]),
        solver=ResponseModel(timeout_error()),
        solver_fallback=FakeRoleModel([fallback_decision]),
        verifier=FakeRoleModel([verifier_v2()]),
    ).run_legal_request(parse_question_text(question))

    assert result.answer.status.value == "completed"
    assert result.answer.selected_options == ("C",)
    assert not {
        "CONTROLLER_CLAIM_COVERAGE_MISMATCH",
        "CONTROLLER_CLAIM_ID_MISMATCH",
    }.intersection(result.answer.warnings)
    assert result.controller_plan.option_claims == {
        option.label: (f"{option.label}1 {option.text}",)
        for option in result.parsed_question.request.options
    }


def test_fallback_path_does_not_reserve_disabled_revision_calls():
    verifier = ResponseModel(
        ModelResponse(
            '{"protocol_version":"verifier-v2","accepted":',
            finish_reason="length",
        ),
        ModelResponse(compact_json(verifier_v2())),
    )
    result = make_runner(
        solver=ResponseModel(timeout_error()),
        solver_fallback=FakeRoleModel([solver_v3_payload()]),
        verifier=verifier,
    ).run_legal_request(parse_question_text(QUESTION))

    assert result.answer.status.value == "completed"
    assert len(verifier.requests) == 2
    assert [request.max_tokens for request in verifier.requests] == [384, 512]
    assert result.execution_budget["calls_used"] == 5


def test_hard_watchdog_timeout_switches_to_fallback():
    release = threading.Event()

    class BlockingSolver:
        def __init__(self):
            self.requests = []

        def complete(self, request):
            self.requests.append(request)
            release.wait(1)
            return ModelResponse(compact_json(solver_v3_payload()))

    primary = BlockingSolver()
    try:
        result = make_runner(
            solver=primary,
            solver_fallback=FakeRoleModel([solver_v3_payload()]),
            verifier=FakeRoleModel([verifier_v2()]),
            max_wall_time_seconds=1,
            solver_call_timeout_seconds=0.02,
            solver_fallback_timeout_seconds=0.2,
            verifier_call_timeout_seconds=0.1,
        ).run_legal_request(parse_question_text(QUESTION))
    finally:
        release.set()
    assert result.answer.status.value == "completed"
    assert len(primary.requests) == 1
    assert any(
        call.agent == "legal_mcq_solver"
        and call.outcome == "deadline_exceeded"
        for call in result.trace.model_calls
    )
    assert any(
        call.agent == "legal_mcq_solver_fallback"
        and call.outcome == "success"
        for call in result.trace.model_calls
    )


def test_solver_circuit_opens_after_two_timeouts_for_rest_of_runner_batch():
    primary = ResponseModel(timeout_error(), timeout_error())
    fallback = FakeRoleModel([solver_v3_payload(), solver_v3_payload(), solver_v3_payload()])
    verifier = FakeRoleModel([verifier_v2(), verifier_v2(), verifier_v2()])
    runner = make_runner(
        controller=FakeRoleModel(
            [controller_v2(), controller_v2(), controller_v2()]
        ),
        solver=primary,
        solver_fallback=fallback,
        verifier=verifier,
        max_revision_rounds=0,
        max_model_calls=4,
    )
    results = [
        runner.run_legal_request(parse_question_text(QUESTION))
        for _ in range(3)
    ]
    assert all(result.answer.status.value == "completed" for result in results)
    assert len(primary.requests) == 2
    assert len(fallback.requests) == 3
    assert any(
        route.route == "legal-mcq-solver-circuit-open"
        for route in results[-1].trace.routes
    )


def test_fallback_rejection_does_not_start_unbounded_revision():
    result = make_runner(
        solver=ResponseModel(timeout_error()),
        solver_fallback=FakeRoleModel([solver_v3_payload("D")]),
        verifier=FakeRoleModel([verifier_v2(accepted=False)]),
    ).run_legal_request(parse_question_text(QUESTION))
    assert result.answer.status.value == "partial"
    assert result.revision_count == 0
    assert "REVISION_SKIPPED_SOLVER_FALLBACK" in result.answer.warnings


def test_revision_solver_timeout_uses_fallback_and_preserves_final_verifier():
    primary = ResponseModel(
        ModelResponse(compact_json(solver_v3_payload("D"))),
        timeout_error(),
    )
    fallback = FakeRoleModel([solver_v3_payload("C")])
    verifier = FakeRoleModel(
        [verifier_v2(accepted=False), verifier_v2()]
    )
    result = make_runner(
        solver=primary,
        solver_fallback=fallback,
        verifier=verifier,
    ).run_legal_request(parse_question_text(QUESTION))
    assert result.answer.status.value == "completed"
    assert result.answer.selected_options == ("C",)
    assert result.revision_count == 1
    assert result.execution_budget["calls_used"] == 6
    assert result.trace.model_calls[-1].agent == "legal_mcq_verifier_2"
    assert any(
        call.agent == "legal_mcq_solver_revision_fallback"
        for call in result.trace.model_calls
    )


def test_deterministic_error_skips_first_verifier_and_goes_to_revision():
    invalid = solver_v3_payload()
    invalid["options"][0]["claims"][0]["claim_id"] = "A99"
    solver = FakeRoleModel([invalid, solver_v3_payload()])
    verifier = FakeRoleModel([verifier_v2()])
    result = make_runner(
        controller=FakeRoleModel([controller_v2()]),
        solver=solver,
        verifier=verifier,
    ).run_legal_request(parse_question_text(QUESTION))
    assert result.revision_count == 1
    assert len(verifier.requests) == 1
    assert any(
        route.route == "legal-mcq-verifier-skipped-deterministic"
        for route in result.trace.routes
    )


@pytest.mark.parametrize("mode,expect_fallback", [("auto", True), ("json_schema", False)])
def test_controller_schema_fallback_is_explicit_and_bounded(mode, expect_fallback):
    controller = ResponseModel(
        ModelCallError("unsupported response_format json_schema", diagnostics={"http_status": 400}),
        ModelResponse(compact_json(controller_v3())),
    )
    trace = RunTrace()
    runner = make_runner(controller=controller, controller_structured_output_mode=mode)
    if expect_fallback:
        result = runner.run_legal_request(parse_question_text(QUESTION), trace=trace)
        assert result.execution_budget["calls_used"] == 4
        assert controller.requests[1].response_format is None
        assert any(r.route == "legal-mcq-structured-output-fallback" for r in trace.routes)
    else:
        with pytest.raises(LegalAgentError):
            runner.run_legal_request(parse_question_text(QUESTION), trace=trace)
        assert len(controller.requests) == 1
    assert controller.requests[0].response_format["json_schema"]["name"] == "legal_mcq_controller_v3"


def test_prompt_only_controller_keeps_local_protocol_validation():
    controller = FakeRoleModel([controller_v3()])
    make_runner(controller=controller, controller_structured_output_mode="prompt_only").run_legal_request(
        parse_question_text(QUESTION)
    )
    assert controller.requests[0].response_format is None


def length_response(*, empty=False):
    response = ModelResponse(
        "" if empty else '{"protocol_version":"solver-v2","partial_marker":',
        usage=TokenUsage(100, 4096, 4196),
        usage_reported=True, finish_reason="length",
        content_state="null" if empty else "text",
        reasoning_tokens=4095,
    )
    if empty:
        return ModelCallError("empty assistant content", response=response)
    return response


@pytest.mark.parametrize("empty", [False, True])
def test_length_retry_reserves_visible_space_and_discards_partial_json(empty):
    solver = ResponseModel(length_response(empty=empty), ModelResponse(compact_json(solver_v2_payload())))
    result = make_runner(solver=solver).run_legal_request(parse_question_text(QUESTION))
    first, second = solver.requests
    assert first.max_tokens == 4096
    assert second.max_tokens == 6144
    assert "partial_marker" not in " ".join(m.content for m in second.messages)
    assert all(m.role != "assistant" for m in second.messages)
    assert second.response_format == first.response_format
    assert second.budget_tokens >= estimate_message_tokens(second.messages) + second.max_tokens
    assert result.execution_budget["calls_used"] == 4
    retry_call = [c for c in result.trace.model_calls if c.agent == "legal_mcq_solver"][1]
    assert retry_call.visible_output_tokens == 2048
    assert retry_call.reasoning_allowance_tokens == 4096
    assert retry_call.completion_token_cap == 6144
    assert any(r.route == "legal-mcq-length-retry" for r in result.trace.routes)


def test_length_growth_cannot_consume_final_verifier_call_reservation():
    solver = ResponseModel(length_response(), ModelResponse(compact_json(solver_v2_payload())))
    trace = RunTrace()
    with pytest.raises(LegalAgentError):
        make_runner(solver=solver, max_model_calls=5).run_legal_request(
            parse_question_text(QUESTION), trace=trace,
        )
    assert len(solver.requests) == 1
    assert any(r.route == "legal-mcq-retry-skipped-budget" for r in trace.routes)


def test_length_growth_cannot_consume_downstream_token_reservation():
    solver = ResponseModel(length_response())
    trace = RunTrace()
    with pytest.raises(LegalAgentError):
        make_runner(solver=solver, max_total_tokens=18000).run_legal_request(
            parse_question_text(QUESTION), trace=trace,
        )
    assert len(solver.requests) == 1
    skipped = [r for r in trace.routes if r.route == "legal-mcq-retry-skipped-budget"]
    assert skipped[0].details["reason"] == "max_tokens"


def test_controller_recovery_solver_length_recovery_and_revision_fit_seven_calls():
    controller = ResponseModel(
        ModelResponse('{"facts":["truncated', finish_reason="length",
                      usage=TokenUsage(600, 2227, 2827), usage_reported=True),
        ModelResponse(compact_json(controller_v2())),
    )
    solver = ResponseModel(
        length_response(empty=True),
        ModelResponse(compact_json(solver_v2_payload("D"))),
        ModelResponse(compact_json(solver_v2_payload("C"))),
    )
    verifier = FakeRoleModel([verifier_payload(False, suggested="C"), verifier_payload(True)])
    result = make_runner(controller=controller, solver=solver, verifier=verifier).run_legal_request(
        parse_question_text(QUESTION)
    )
    assert result.execution_budget["calls_used"] == 7
    assert result.answer.status.value == "completed"
    assert result.answer.selected_options == ("C",)
    assert result.revision_count == 1
    assert result.trace.model_calls[-1].agent == "legal_mcq_verifier_2"
    assert solver.requests[1].max_tokens > solver.requests[0].max_tokens
    assert solver.requests[2].max_tokens == solver.requests[0].max_tokens
    assert controller.requests[1].response_format["type"] == "json_schema"
    assert all(m.role != "assistant" for m in controller.requests[1].messages)


def test_no_identical_solver_retry_at_the_completion_cap():
    solver = ResponseModel(length_response())
    runner = make_runner(solver=solver)
    runner.config = replace(
        runner.config,
        legal_mcq=replace(runner.config.legal_mcq, solver_model=replace(
            runner.config.legal_mcq.solver_model, max_tokens=4096,
        )),
    )
    trace = RunTrace()
    with pytest.raises(LegalAgentError):
        runner.run_legal_request(parse_question_text(QUESTION), trace=trace)
    assert len(solver.requests) == 1
    assert any(r.route == "legal-mcq-length-retry-skipped" for r in trace.routes)


def test_at_most_one_length_recovery_even_with_three_attempts():
    solver = ResponseModel(length_response(), length_response())
    runner = make_runner(solver=solver, max_attempts=3)
    runner.config = replace(
        runner.config,
        legal_mcq=replace(runner.config.legal_mcq, solver_model=replace(
            runner.config.legal_mcq.solver_model, max_tokens=8192,
        )),
    )
    trace = RunTrace()
    with pytest.raises(LegalAgentError):
        runner.run_legal_request(parse_question_text(QUESTION), trace=trace)
    assert len(solver.requests) == 2
    assert any(r.details.get("reason") == "length_recovery_limit" for r in trace.routes)


def test_real_failed_visible_outputs_are_not_salvaged_as_valid_decisions():
    log = Path(__file__).resolve().parents[1] / "output/legal_mcq/health-dev5-low-20260911/results.calls.jsonl"
    if not log.exists():
        pytest.skip("private provider replay artifact is not present")
    failures = 0
    for line in log.read_text().splitlines():
        call = json.loads(line)
        if call["event"] != "model_call_finished" or call["outcome"] == "success":
            continue
        parser = ControllerPlan if call["agent"] == "legal_mcq_controller" else SolverDecision
        with pytest.raises(StructuredOutputError):
            parser.from_text(call["output_text"])
        failures += 1
    assert failures == 12
