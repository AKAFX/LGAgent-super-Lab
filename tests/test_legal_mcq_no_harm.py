from __future__ import annotations

import json
from dataclasses import replace
from datetime import date

import pytest

from lgagent.config import ConfigurationError, LegalMCQSettings, ModelConfig
from lgagent.legal_mcq import (
    DirectAnchorDecision,
    LegalEvidence,
    LegalMCQAgent,
    LegalMCQPolicy,
    NoHarmGate,
    SolveMode,
    SolverDecision,
    parse_question_text,
)
from lgagent.model import ModelRequest, ModelResponse, TokenUsage


QUESTION = """关于合同效力，下列说法正确的是？

A. 第一项
B. 第二项
C. 第三项
D. 第四项"""

CONFIG = ModelConfig(
    backend="fake",
    base_url="https://offline.invalid/v1",
    api_key="",
    model="fake",
    temperature=0.0,
    top_p=1.0,
    max_tokens=4096,
)


class FakeRoleModel:
    def __init__(self, outputs: list[dict[str, object] | str | Exception]) -> None:
        self.outputs = list(outputs)
        self.requests: list[ModelRequest] = []

    def complete(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        output = self.outputs.pop(0)
        if isinstance(output, Exception):
            raise output
        content = output if isinstance(output, str) else json.dumps(output)
        return ModelResponse(content, usage=TokenUsage(10, 5, 15))


def anchor_payload(answer: str = "C") -> dict[str, object]:
    return {
        "protocol_version": "direct-anchor-v1",
        "selected_options": [answer],
        "confidence": 0.8,
    }


def controller_payload() -> dict[str, object]:
    return {
        "protocol_version": "controller-v3",
        "facts": ["当事人订立合同"],
        "issues": [
            {
                "description": "合同效力判断",
                "options": list("ABCD"),
            }
        ],
        "checks": ["核验题干极性"],
    }


def solver_payload(
    answer: str = "C",
    *,
    evidence_ids: dict[str, tuple[str, ...]] | None = None,
) -> dict[str, object]:
    citations = evidence_ids or {}
    options: list[dict[str, object]] = []
    for label in "ABCD":
        verdict = "supported" if label == answer else "contradicted"
        options.append(
            {
                "label": label,
                "claims": [
                    {
                        "claim_id": f"{label}1",
                        "verdict": verdict,
                        "reason": f"{label}项决定性理由",
                        "evidence_ids": list(citations.get(label, ())),
                    }
                ],
                "verdict": verdict,
            }
        )
    return {
        "protocol_version": "solver-v3",
        "selected_options": [answer],
        "options": options,
        "confidence": 0.9,
    }


def verifier_payload() -> dict[str, object]:
    return {
        "protocol_version": "verifier-v2",
        "accepted": True,
        "error_codes": [],
        "challenged_options": [],
        "suggested_selected_options": [],
        "note": "",
    }


def anchor(answer: str) -> DirectAnchorDecision:
    return DirectAnchorDecision.from_text(json.dumps(anchor_payload(answer)))


def decision(
    answer: str,
    *,
    evidence_ids: dict[str, tuple[str, ...]] | None = None,
) -> SolverDecision:
    return SolverDecision.from_text(
        json.dumps(solver_payload(answer, evidence_ids=evidence_ids))
    )


def evidence(
    evidence_id: str,
    *,
    source_type: str = "statute",
    authority_level: int = 5,
    effective_from: date | None = date(2021, 1, 1),
    effective_until: date | None = None,
) -> LegalEvidence:
    return LegalEvidence(
        evidence_id=evidence_id,
        title="权威法源",
        url=f"https://law.example/{evidence_id}",
        publisher="全国人大",
        source_type=source_type,
        authority_level=authority_level,
        quote=f"{evidence_id} 对应的法律正文。",
        effective_from=effective_from,
        effective_until=effective_until,
    )


def build_agent(
    *,
    solver: FakeRoleModel,
    controller: FakeRoleModel | None = None,
    verifier: FakeRoleModel | None = None,
    gate_enabled: bool,
) -> LegalMCQAgent:
    return LegalMCQAgent(
        controller_model=controller or FakeRoleModel([controller_payload()]),
        solver_model=solver,
        verifier_model=verifier or FakeRoleModel([verifier_payload()]),
        controller_config=replace(CONFIG, model="controller"),
        solver_config=replace(CONFIG, model="solver"),
        verifier_config=replace(CONFIG, model="verifier"),
        policy=LegalMCQPolicy(
            max_revision_rounds=0,
            no_harm_gate_enabled=gate_enabled,
        ),
        runtime_date_provider=lambda: date(2026, 1, 1),
    )


def test_direct_anchor_rejects_non_array_selected_options() -> None:
    payload = anchor_payload()
    payload["selected_options"] = "C"

    with pytest.raises(Exception, match="selected_options must be an array"):
        DirectAnchorDecision.from_text(json.dumps(payload))


def test_direct_anchor_rejects_empty_and_duplicate_selections_locally() -> None:
    for selected in ([], ["C", "C"]):
        payload = anchor_payload()
        payload["selected_options"] = selected
        with pytest.raises(Exception, match="unique and non-empty"):
            DirectAnchorDecision.from_text(json.dumps(payload))

    schema = DirectAnchorDecision.response_format(tuple("ABCD"))
    selected_schema = schema["json_schema"]["schema"]["properties"][
        "selected_options"
    ]
    assert "uniqueItems" not in selected_schema


def test_gate_passes_matching_answer_without_evidence() -> None:
    result = NoHarmGate().evaluate(
        anchor=anchor("C"),
        candidate=decision("C"),
        evidence=(),
        effective_mode=SolveMode.CLOSED_BOOK,
        as_of_date=date(2026, 1, 1),
    )

    assert result.action == "matched"
    assert not result.preserve_anchor


def test_closed_book_disagreement_preserves_anchor() -> None:
    result = NoHarmGate().evaluate(
        anchor=anchor("B"),
        candidate=decision("D"),
        evidence=(),
        effective_mode=SolveMode.CLOSED_BOOK,
        as_of_date=date(2026, 1, 1),
    )

    assert result.action == "preserve_anchor"
    assert result.anchor_selected_options == ("B",)
    assert result.candidate_selected_options == ("D",)


def test_open_book_requires_evidence_for_every_changed_option() -> None:
    law = evidence("law-505")
    result = NoHarmGate().evaluate(
        anchor=anchor("B"),
        candidate=decision("D", evidence_ids={"D": ("law-505",)}),
        evidence=(law,),
        effective_mode=SolveMode.OPEN_BOOK,
        as_of_date=date(2026, 1, 1),
    )

    assert result.action == "preserve_anchor"
    assert "option B" in result.reason


def test_open_book_allows_override_when_every_changed_option_is_evidenced() -> None:
    law_b = evidence("law-b")
    law_d = evidence("law-d")
    result = NoHarmGate().evaluate(
        anchor=anchor("B"),
        candidate=decision(
            "D",
            evidence_ids={
                "B": ("law-b",),
                "D": ("law-d",),
            },
        ),
        evidence=(law_b, law_d),
        effective_mode=SolveMode.OPEN_BOOK,
        as_of_date=date(2026, 1, 1),
    )

    assert result.action == "allow_override"
    assert result.authoritative_evidence_ids == ("law-b", "law-d")


@pytest.mark.parametrize(
    ("candidate_valid", "verifier_accepted", "reason"),
    [
        (False, True, "deterministic validation"),
        (True, False, "independent verification"),
    ],
)
def test_open_book_rejects_candidate_that_failed_chain_checks(
    candidate_valid: bool,
    verifier_accepted: bool,
    reason: str,
) -> None:
    law = evidence("law")
    result = NoHarmGate().evaluate(
        anchor=anchor("B"),
        candidate=decision(
            "D",
            evidence_ids={"B": ("law",), "D": ("law",)},
        ),
        evidence=(law,),
        effective_mode=SolveMode.OPEN_BOOK,
        as_of_date=date(2026, 1, 1),
        candidate_valid=candidate_valid,
        verifier_accepted=verifier_accepted,
    )

    assert result.action == "preserve_anchor"
    assert reason in result.reason


@pytest.mark.parametrize(
    "item",
    [
        evidence("snippet", source_type="search_snippet"),
        evidence("low-authority", authority_level=3),
        evidence("expired", effective_until=date(2025, 12, 31)),
    ],
    ids=["snippet", "low-authority", "expired"],
)
def test_open_book_rejects_non_decisive_evidence(item: LegalEvidence) -> None:
    result = NoHarmGate().evaluate(
        anchor=anchor("B"),
        candidate=decision(
            "D",
            evidence_ids={
                "B": (item.evidence_id,),
                "D": (item.evidence_id,),
            },
        ),
        evidence=(item,),
        effective_mode=SolveMode.OPEN_BOOK,
        as_of_date=date(2026, 1, 1),
    )

    assert result.action == "preserve_anchor"
    assert result.authoritative_evidence_ids == ()


def test_disabled_gate_preserves_existing_call_count_and_result() -> None:
    solver = FakeRoleModel([solver_payload("C")])
    agent = build_agent(solver=solver, gate_enabled=False)

    result = agent.solve(parse_question_text(QUESTION))

    assert result.answer.selected_options == ("C",)
    assert result.answer.status.value == "completed"
    assert len(solver.requests) == 1
    assert result.direct_anchor is None
    assert result.no_harm_gate == {}
    assert [call.agent for call in result.trace.model_calls] == [
        "legal_mcq_controller",
        "legal_mcq_solver",
        "legal_mcq_verifier_1",
    ]


def test_enabled_gate_adds_independent_anchor_call() -> None:
    solver = FakeRoleModel([anchor_payload("C"), solver_payload("C")])
    agent = build_agent(solver=solver, gate_enabled=True)

    result = agent.solve(parse_question_text(QUESTION))

    assert result.answer.selected_options == ("C",)
    assert result.answer.status.value == "completed"
    assert len(solver.requests) == 2
    assert result.direct_anchor is not None
    assert result.no_harm_gate["action"] == "matched"
    assert [call.agent for call in result.trace.model_calls] == [
        "legal_mcq_direct_anchor",
        "legal_mcq_controller",
        "legal_mcq_solver",
        "legal_mcq_verifier_1",
    ]


def test_anchor_failure_fails_open_and_keeps_warning() -> None:
    solver = FakeRoleModel(["not-json", solver_payload("C")])
    agent = build_agent(solver=solver, gate_enabled=True)

    result = agent.solve(parse_question_text(QUESTION))

    assert result.answer.selected_options == ("C",)
    assert result.answer.status.value == "completed"
    assert "NO_HARM_ANCHOR_UNAVAILABLE" in result.answer.warnings
    assert result.direct_anchor is None
    assert result.no_harm_gate == {}
    routes = [item.route for item in result.trace.routes]
    assert "legal-mcq-no-harm-anchor-unavailable" in routes


def test_closed_book_disagreement_returns_partial_anchor_and_audit_data() -> None:
    solver = FakeRoleModel([anchor_payload("B"), solver_payload("D")])
    agent = build_agent(solver=solver, gate_enabled=True)

    result = agent.solve(parse_question_text(QUESTION))

    assert result.answer.selected_options == ("B",)
    assert result.answer.status.value == "partial"
    assert result.answer.needs_review
    assert result.answer.option_assessments == ()
    assert "NO_HARM_ANCHOR_PRESERVED" in result.answer.warnings
    assert result.solver_decision.selected_options == ("D",)
    assert result.no_harm_gate["action"] == "preserve_anchor"
    assert result.no_harm_gate["anchor_selected_options"] == ["B"]
    assert result.no_harm_gate["candidate_selected_options"] == ["D"]
    assert result.no_harm_gate["candidate_validation_valid"] is True
    assert result.no_harm_gate["candidate_verifier_accepted"] is True
    assert result.verification.error_codes == ("NO_HARM_ANCHOR_PRESERVED",)
    routes = [item.route for item in result.trace.routes]
    assert "legal-mcq-no-harm-preserve_anchor" in routes


def test_config_rejects_insufficient_no_harm_call_budget() -> None:
    payload = {
        "enabled": True,
        "max_revision_rounds": 1,
        "max_model_calls": 6,
        "no_harm_gate_enabled": True,
        "solver_fallback_model": {"model_name": "fallback"},
    }

    with pytest.raises(ConfigurationError, match="requires at least 7"):
        LegalMCQSettings.from_mapping(payload, generation=CONFIG, environ={})

    payload["max_model_calls"] = 7
    settings = LegalMCQSettings.from_mapping(
        payload,
        generation=CONFIG,
        environ={},
    )
    assert settings.no_harm_gate_enabled
