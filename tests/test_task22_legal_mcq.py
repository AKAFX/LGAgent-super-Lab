from __future__ import annotations

import json
import sys
import tempfile
import unittest
from dataclasses import asdict, replace
from datetime import date
from pathlib import Path

import yaml

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from lgagent.config import (
    LGAgentConfig,
    LGAgentPlusConfig,
    LegalMCQSettings,
    ModelConfig,
    load_lgagent_config,
)
from lgagent.evidence_audit import (
    AuditLabel,
    AuditedEvidence,
    AuditedEvidenceMatrix,
    OptionEvidenceAudit,
)
from lgagent.legal_mcq import (
    BenchmarkRecordAdapter,
    ControllerPlan,
    LegalAgentError,
    LegalEvidence,
    LegalMCQAgent,
    LegalMCQPolicy,
    LegalSkillRegistry,
    LeakGuard,
    LegalEvidenceAdapterError,
    QuestionType,
    SOLVER_PROTOCOL_VERSION,
    SOLVER_PROTOCOL_V2_VERSION,
    SolveMode,
    SolverDecision,
    adapt_audited_evidence,
    build_openai_role_clients,
    evaluate_case,
    parse_question_text,
    parse_request,
)
from lgagent.model import (
    BudgetExceededError,
    ModelCallError,
    ModelRequest,
    ModelResponse,
    TokenUsage,
)
from lgagent.runner import LGAgentPlusRunner


QUESTION = """刘某委托关某经营个人独资企业。下列说法正确的是？

A. 内部权限限制当然使合同无效
B. 合同一律效力待定
C. 内部限制不得对抗善意第三人
D. 受托人当然对第三人承担连带责任"""


def controller_payload() -> dict[str, object]:
    return {
        "facts": [{"fact_id": "F1", "text": "刘某委托关某经营企业"}],
        "issues": [
            {
                "issue_id": "I1",
                "description": "内部权限限制的对外效力",
                "decisive_for_options": ["A", "B", "C", "D"],
                "candidate_laws": ["个人独资企业法"],
            }
        ],
        "option_claims": {
            label: [f"{label}1 原子命题"] for label in "ABCD"
        },
        "legal_domains": ["商法"],
        "named_laws": ["个人独资企业法"],
        "trap_checks": ["内部关系与外部关系"],
        "verification_checklist": ["题干极性", "责任对象"],
    }


def solver_payload(
    answer: str = "C",
    *,
    evidence_id: str | None = None,
) -> dict[str, object]:
    assessments = []
    for label in "ABCD":
        verdict = "supported" if label == answer else "contradicted"
        assessments.append(
            {
                "label": label,
                "claims": [
                    {
                        "claim_id": f"{label}1",
                        "text": f"{label}项命题",
                        "verdict": verdict,
                        "reason": f"{label}项核验结果",
                        "rule": "内部权限限制不得对抗善意第三人",
                        "evidence_ids": (
                            [evidence_id] if evidence_id and label == answer else []
                        ),
                    }
                ],
                "verdict": verdict,
                "decisive_reason": f"{label}项{verdict}",
                "confidence": 0.9,
            }
        )
    return {
        "selected_options": [answer],
        "issue_analyses": [
            {
                "issue_id": "I1",
                "issue": "内部权限限制的对外效力",
                "application": "乙为善意第三人",
                "conclusion": f"{answer}项成立",
            }
        ],
        "option_assessments": assessments,
        "rationale": f"逐项判断后选择{answer}",
        "confidence": 0.9,
    }


def solver_v2_payload(
    answer: str = "C",
    *,
    evidence_id: str | None = None,
) -> dict[str, object]:
    options = []
    for label in "ABCD":
        verdict = "supported" if label == answer else "contradicted"
        options.append(
            {
                "label": label,
                "claims": [
                    {
                        "claim_id": f"{label}1",
                        "verdict": verdict,
                        "reason": f"{label}项核验结果",
                        "evidence_ids": (
                            [evidence_id]
                            if evidence_id and label == answer
                            else []
                        ),
                    }
                ],
                "verdict": verdict,
            }
        )
    return {
        "protocol_version": SOLVER_PROTOCOL_V2_VERSION,
        "selected_options": [answer],
        "issues": [
            {
                "issue_id": "I1",
                "analysis": "内部权限限制不得对抗善意第三人",
                "conclusion": f"{answer}项成立",
            }
        ],
        "options": options,
        "rationale": f"逐项判断后选择{answer}",
        "confidence": 0.9,
    }


def solver_v3_payload(
    answer: str = "C",
    *,
    evidence_id: str | None = None,
) -> dict[str, object]:
    payload = solver_v2_payload(answer, evidence_id=evidence_id)
    payload["protocol_version"] = SOLVER_PROTOCOL_VERSION
    payload.pop("issues")
    payload.pop("rationale")
    return payload


def solver_payload_for_verdicts(
    selected: list[str],
    verdicts: dict[str, str],
) -> dict[str, object]:
    payload = solver_payload(selected[0])
    payload["selected_options"] = selected
    assessments = payload["option_assessments"]
    assert isinstance(assessments, list)
    for raw in assessments:
        assert isinstance(raw, dict)
        label = str(raw["label"])
        verdict = verdicts[label]
        raw["verdict"] = verdict
        claims = raw["claims"]
        assert isinstance(claims, list)
        assert isinstance(claims[0], dict)
        claims[0]["verdict"] = verdict
    return payload


def verifier_payload(
    accepted: bool = True,
    *,
    suggested: str | None = None,
) -> dict[str, object]:
    return {
        "accepted": accepted,
        "error_codes": [] if accepted else ["RESPONSIBILITY_TARGET_ERROR"],
        "challenged_options": [] if accepted else ["D"],
        "explanation": "" if accepted else "混淆了内部责任与对外责任",
        "suggested_selected_options": [suggested] if suggested else [],
    }


CONFIG = ModelConfig(
    backend="fake",
    base_url="https://offline.invalid/v1",
    api_key="",
    model="fallback",
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
        return ModelResponse(
            (
                output
                if isinstance(output, str)
                else json.dumps(output, ensure_ascii=False)
            ),
            usage=TokenUsage(10, 5, 15),
        )


class ParserAndIsolationTest(unittest.TestCase):
    def test_plain_text_parser_handles_polarity_date_and_multiline_options(self) -> None:
        request = parse_question_text(
            """2020年下列说法不正确的是？
Ａ．第一行
继续描述
Ｂ．第二项
Ｃ．第三项
Ｄ．第四项""",
            mode=SolveMode.CLOSED_BOOK,
        )
        parsed = parse_request(request, runtime_date=date(2026, 1, 1))

        self.assertEqual([item.label for item in request.options], list("ABCD"))
        self.assertIn("继续描述", request.options[0].text)
        self.assertTrue(parsed.asks_for_incorrect_option)
        self.assertEqual(parsed.request.as_of_date, date(2020, 12, 31))
        self.assertEqual(parsed.date_source, "question_explicit")

    def test_polarity_uses_final_question_not_narrative_negative_words(self) -> None:
        positive_stems = (
            "法院认为申请不符合破产条件，裁定不予受理。"
            "在现行法律框架下，下列哪项法理构成主要依据？",
            "原审认定逮捕错误，并判决承担赔偿责任。"
            "若要防止类似权力滥用，哪项制度创新最为适当？",
            "法院指出分别罚款的处理错误。"
            "面对法律与道德冲突，法官应如何裁判？",
            "法院认定逮捕错误但当事人争议较大，关于本案哪项说法正确？",
        )
        for stem in positive_stems:
            with self.subTest(stem=stem):
                request = parse_question_text(f"{stem}\nA.甲\nB.乙")
                parsed = parse_request(request, runtime_date=date(2026, 1, 1))
                self.assertFalse(parsed.asks_for_incorrect_option)

        inverse = parse_question_text(
            "案情中没有程序错误。关于本案，下列哪一选项是不正确的？"
            "\nA.甲\nB.乙"
        )
        self.assertTrue(
            parse_request(inverse, runtime_date=date(2026, 1, 1))
            .asks_for_incorrect_option
        )

    def test_temporal_priority_prefers_hypothesis_then_request_metadata(self) -> None:
        hypothetical = parse_question_text(
            "原案于1997年6月23日判决。若本案发生在2023年，哪项规则应适用？"
            "\nA.甲\nB.乙"
        )
        parsed = parse_request(hypothetical, runtime_date=date(2026, 1, 1))
        self.assertEqual(parsed.request.as_of_date, date(2023, 12, 31))
        self.assertEqual(parsed.date_source, "question_hypothetical")
        self.assertIn("year_only_date_assumed_end", parsed.parse_warnings)

        metadata = parse_question_text(
            "案件材料记载2016年作出裁定。关于本案，哪项说法正确？"
            "\nA.甲\nB.乙",
            as_of_date=date(2024, 5, 1),
        )
        parsed = parse_request(metadata, runtime_date=date(2026, 1, 1))
        self.assertEqual(parsed.request.as_of_date, date(2024, 5, 1))
        self.assertEqual(parsed.date_source, "request_metadata")

        case_date = parse_question_text(
            "案件发生在2018年。关于本案，哪项说法正确？\nA.甲\nB.乙"
        )
        parsed = parse_request(case_date, runtime_date=date(2026, 1, 1))
        self.assertEqual(parsed.request.as_of_date, date(2018, 12, 31))
        self.assertEqual(parsed.date_source, "case_explicit")

    def test_invalid_or_duplicate_options_fail_before_model_work(self) -> None:
        with self.assertRaisesRegex(LegalAgentError, "consecutive"):
            parse_question_text("题目\nA.一\nA.二")
        with self.assertRaisesRegex(LegalAgentError, "at least two"):
            parse_question_text("题目\nA.一")

    def test_benchmark_adapter_separates_request_and_oracle(self) -> None:
        case = BenchmarkRecordAdapter.adapt(
            {
                "id": 7,
                "question": QUESTION,
                "golden_answers": ["C"],
                "meta_data": {"correct_option": "C", "original_index": 7},
            },
            source_dataset="fixture",
        )

        serialized_request = json.dumps(
            asdict(case.request), ensure_ascii=False, default=str
        )
        self.assertNotIn("golden", serialized_request)
        self.assertNotIn("correct_option", serialized_request)
        self.assertEqual(case.oracle.golden_answers, frozenset({"C"}))
        with self.assertRaisesRegex(LegalAgentError, "oracle_leakage"):
            LeakGuard.assert_clean(
                {"question": QUESTION, "golden_answers": ["C"]}
            )

    def test_controller_rejects_nested_answer_fields(self) -> None:
        payload = controller_payload()
        payload["nested"] = {"analysis": [{"selected_options": ["C"]}]}
        with self.assertRaisesRegex(Exception, "answer-bearing fields"):
            ControllerPlan.from_text(json.dumps(payload, ensure_ascii=False))

    def test_solver_v3_is_compact_and_v1_v2_remain_readable(self) -> None:
        v1_payload = solver_payload()
        v2_payload = solver_v2_payload()
        v3_payload = solver_v3_payload()
        v1 = SolverDecision.from_text(json.dumps(v1_payload, ensure_ascii=False))
        v2 = SolverDecision.from_text(json.dumps(v2_payload, ensure_ascii=False))
        v3 = SolverDecision.from_text(json.dumps(v3_payload, ensure_ascii=False))

        self.assertEqual(v1.selected_options, v2.selected_options)
        self.assertEqual(v2.selected_options, v3.selected_options)
        self.assertEqual(v2.raw["protocol_version"], SOLVER_PROTOCOL_V2_VERSION)
        self.assertEqual(v3.raw["protocol_version"], SOLVER_PROTOCOL_VERSION)
        self.assertEqual(v3.option_assessments[0].claims[0].claim_id, "A1")
        self.assertEqual(v2.issue_analyses[0].issue_id, "I1")
        self.assertEqual(v3.issue_analyses, ())
        self.assertLess(
            len(json.dumps(v2_payload, ensure_ascii=False)),
            len(json.dumps(v1_payload, ensure_ascii=False)),
        )
        self.assertLess(
            len(json.dumps(v3_payload, ensure_ascii=False)),
            len(json.dumps(v2_payload, ensure_ascii=False)),
        )

        invalid = solver_v2_payload()
        invalid["unexpected"] = "not allowed"
        with self.assertRaisesRegex(Exception, "fields mismatch"):
            SolverDecision.from_text(json.dumps(invalid, ensure_ascii=False))

        response_format = SolverDecision.response_format()
        self.assertEqual(response_format["type"], "json_schema")
        schema = response_format["json_schema"]
        self.assertTrue(schema["strict"])
        self.assertFalse(schema["schema"]["additionalProperties"])


class ThreeRoleWorkflowTest(unittest.TestCase):
    def agent(
        self,
        *,
        controller: FakeRoleModel,
        solver: FakeRoleModel,
        verifier: FakeRoleModel,
        mode: SolveMode = SolveMode.CLOSED_BOOK,
        evidence_provider=None,
        revisions: int = 1,
    ):
        request = parse_question_text(QUESTION, mode=mode)
        return LegalMCQAgent(
            controller_model=controller,
            solver_model=solver,
            verifier_model=verifier,
            controller_config=ModelConfig(
                **{**asdict(CONFIG), "model": "controller-cheap"}
            ),
            solver_config=ModelConfig(
                **{**asdict(CONFIG), "model": "solver-strong"}
            ),
            verifier_config=ModelConfig(
                **{**asdict(CONFIG), "model": "verifier-cheap"}
            ),
            policy=LegalMCQPolicy(max_revision_rounds=revisions),
            evidence_provider=evidence_provider,
            runtime_date_provider=lambda: date(2026, 1, 1),
        ), request

    def test_strong_solver_is_only_answering_role(self) -> None:
        controller = FakeRoleModel([controller_payload()])
        solver = FakeRoleModel([solver_payload()])
        verifier = FakeRoleModel([verifier_payload()])
        agent, request = self.agent(
            controller=controller, solver=solver, verifier=verifier
        )

        result = agent.solve(request)

        self.assertEqual(result.answer.selected_options, ("C",))
        self.assertFalse(result.answer.needs_review)
        self.assertEqual(len(result.answer.option_assessments), 4)
        self.assertEqual(controller.requests[0].model, "controller-cheap")
        self.assertEqual(solver.requests[0].model, "solver-strong")
        self.assertEqual(verifier.requests[0].model, "verifier-cheap")
        self.assertNotIn("golden_answers", solver.requests[0].messages[-1].content)
        self.assertEqual(result.answer.citations, ())
        self.assertEqual(
            result.skill_versions,
            {"legal-mcq-core": "1.0.0"},
        )
        self.assertNotIn(
            "legal-evaluation",
            controller.requests[0].messages[-1].content,
        )

    def test_compact_solver_v2_runs_through_validator_and_finalizer(self) -> None:
        controller = FakeRoleModel([controller_payload()])
        solver = FakeRoleModel([solver_v2_payload()])
        verifier = FakeRoleModel([verifier_payload()])
        agent, request = self.agent(
            controller=controller,
            solver=solver,
            verifier=verifier,
        )

        result = agent.solve(request)

        self.assertEqual(result.answer.selected_options, ("C",))
        self.assertEqual(
            result.solver_decision.raw["protocol_version"],
            SOLVER_PROTOCOL_V2_VERSION,
        )
        self.assertFalse(result.answer.needs_review)

    def test_invalid_json_is_repaired_with_one_bounded_retry(self) -> None:
        controller = FakeRoleModel(["not-json", controller_payload()])
        solver = FakeRoleModel([solver_payload()])
        verifier = FakeRoleModel([verifier_payload()])
        agent, request = self.agent(
            controller=controller, solver=solver, verifier=verifier
        )

        result = agent.solve(request)

        self.assertEqual(result.answer.selected_options, ("C",))
        self.assertEqual(len(controller.requests), 2)
        self.assertIn(
            "上一响应未通过结构校验",
            controller.requests[1].messages[-1].content,
        )

    def test_non_retryable_provider_error_is_classified_and_not_retried(self) -> None:
        controller = FakeRoleModel(
            [ModelCallError("quota_not_enough: balance exhausted")]
        )
        agent, request = self.agent(
            controller=controller,
            solver=FakeRoleModel([]),
            verifier=FakeRoleModel([]),
        )

        with self.assertRaises(LegalAgentError) as captured:
            agent.solve(request)

        self.assertEqual(captured.exception.code.value, "model_provider_error")
        self.assertEqual(len(controller.requests), 1)

    def test_multiple_choice_selection_matches_all_supported_options(self) -> None:
        controller = FakeRoleModel([controller_payload()])
        solver = FakeRoleModel(
            [
                solver_payload_for_verdicts(
                    ["A", "C"],
                    {
                        "A": "supported",
                        "B": "contradicted",
                        "C": "supported",
                        "D": "contradicted",
                    },
                )
            ]
        )
        verifier = FakeRoleModel([verifier_payload()])
        agent, _ = self.agent(
            controller=controller, solver=solver, verifier=verifier
        )
        request = parse_question_text(
            QUESTION,
            mode=SolveMode.CLOSED_BOOK,
            question_type=QuestionType.MULTIPLE_CHOICE,
        )

        result = agent.solve(request)

        self.assertEqual(result.answer.selected_options, ("A", "C"))
        self.assertFalse(result.answer.needs_review)

    def test_inverse_question_selects_the_unique_contradicted_option(self) -> None:
        question = QUESTION.replace("正确的是", "不正确的是")
        controller = FakeRoleModel([controller_payload()])
        solver = FakeRoleModel(
            [
                solver_payload_for_verdicts(
                    ["D"],
                    {
                        "A": "supported",
                        "B": "supported",
                        "C": "supported",
                        "D": "contradicted",
                    },
                )
            ]
        )
        verifier = FakeRoleModel([verifier_payload()])
        agent, _ = self.agent(
            controller=controller, solver=solver, verifier=verifier
        )

        result = agent.solve(parse_question_text(question))

        self.assertEqual(result.answer.selected_options, ("D",))
        self.assertFalse(result.answer.needs_review)

    def test_verifier_can_request_exactly_one_solver_revision(self) -> None:
        controller = FakeRoleModel([controller_payload()])
        solver = FakeRoleModel([solver_payload("D"), solver_payload("C")])
        verifier = FakeRoleModel(
            [verifier_payload(False, suggested="C"), verifier_payload(True)]
        )
        agent, request = self.agent(
            controller=controller, solver=solver, verifier=verifier
        )

        result = agent.solve(request)

        self.assertEqual(result.answer.selected_options, ("C",))
        self.assertEqual(result.revision_count, 1)
        self.assertEqual(len(solver.requests), 2)
        self.assertEqual(len(verifier.requests), 2)

    def test_unresolved_verifier_conflict_is_partial_not_an_infinite_loop(self) -> None:
        controller = FakeRoleModel([controller_payload()])
        solver = FakeRoleModel([solver_payload("D"), solver_payload("D")])
        verifier = FakeRoleModel(
            [
                verifier_payload(False, suggested="C"),
                verifier_payload(False, suggested="C"),
            ]
        )
        agent, request = self.agent(
            controller=controller, solver=solver, verifier=verifier
        )

        result = agent.solve(request)

        self.assertTrue(result.answer.needs_review)
        self.assertEqual(result.answer.status.value, "partial")
        self.assertEqual(result.revision_count, 1)
        self.assertEqual(len(solver.requests), 2)

    def test_open_book_requires_authoritative_normalized_evidence(self) -> None:
        controller = FakeRoleModel([controller_payload()])
        solver = FakeRoleModel([])
        verifier = FakeRoleModel([])
        agent, request = self.agent(
            controller=controller,
            solver=solver,
            verifier=verifier,
            mode=SolveMode.OPEN_BOOK,
        )
        with self.assertRaisesRegex(LegalAgentError, "retrieval_required"):
            agent.solve(request)
        self.assertEqual(solver.requests, [])

        evidence = LegalEvidence(
            evidence_id="law-1",
            title="个人独资企业法",
            url="https://www.gov.cn/law/1",
            publisher="全国人大",
            source_type="statute",
            authority_level=5,
            quote="投资人对受托人职权的限制，不得对抗善意第三人。",
            effective_from=date(2000, 1, 1),
        )
        controller = FakeRoleModel([controller_payload()])
        solver = FakeRoleModel([solver_payload(evidence_id="law-1")])
        verifier = FakeRoleModel([verifier_payload()])
        agent, request = self.agent(
            controller=controller,
            solver=solver,
            verifier=verifier,
            mode=SolveMode.OPEN_BOOK,
            evidence_provider=lambda *_: (evidence,),
        )

        result = agent.solve(request)

        self.assertEqual(result.answer.status.value, "completed")
        self.assertEqual(result.answer.citations[0].citation_id, "law-1")
        self.assertIn("https://www.gov.cn/law/1", result.answer.to_markdown())
        self.assertEqual(
            result.skill_versions,
            {
                "legal-mcq-core": "1.0.0",
                "legal-research": "1.0.0",
            },
        )

    def test_open_book_rejects_search_snippets_before_solver(self) -> None:
        evidence = LegalEvidence(
            evidence_id="snippet-1",
            title="搜索结果",
            url="https://example.invalid/search",
            publisher="search",
            source_type="search_snippet",
            authority_level=5,
            quote="这只是搜索摘要，不是抓取后的权威正文。",
        )
        controller = FakeRoleModel([controller_payload()])
        solver = FakeRoleModel([])
        verifier = FakeRoleModel([])
        agent, request = self.agent(
            controller=controller,
            solver=solver,
            verifier=verifier,
            mode=SolveMode.OPEN_BOOK,
            evidence_provider=lambda *_: (evidence,),
        )

        with self.assertRaisesRegex(LegalAgentError, "no_authoritative_source"):
            agent.solve(request)
        self.assertEqual(solver.requests, [])

    def test_verifier_unknown_option_cannot_be_accepted(self) -> None:
        controller = FakeRoleModel([controller_payload()])
        solver = FakeRoleModel([solver_payload()])
        verifier = FakeRoleModel(
            [
                {
                    "accepted": True,
                    "error_codes": [],
                    "challenged_options": ["Z"],
                    "explanation": "",
                    "suggested_selected_options": [],
                }
            ]
        )
        agent, request = self.agent(
            controller=controller,
            solver=solver,
            verifier=verifier,
            revisions=0,
        )

        result = agent.solve(request)

        self.assertTrue(result.answer.needs_review)
        self.assertIn("VERIFIER_UNKNOWN_OPTION", result.answer.warnings)

    def test_unknown_evidence_id_and_claim_conflict_require_review(self) -> None:
        evidence = LegalEvidence(
            evidence_id="law-1",
            title="个人独资企业法",
            url="https://www.gov.cn/law/1",
            publisher="全国人大",
            source_type="statute",
            authority_level=5,
            quote="投资人对受托人职权的限制，不得对抗善意第三人。",
            effective_from=date(2000, 1, 1),
        )
        decision = solver_payload(evidence_id="missing-law")
        assessments = decision["option_assessments"]
        assert isinstance(assessments, list)
        selected = next(
            item
            for item in assessments
            if isinstance(item, dict) and item["label"] == "C"
        )
        claims = selected["claims"]
        assert isinstance(claims, list)
        assert isinstance(claims[0], dict)
        claims[0]["verdict"] = "contradicted"

        controller = FakeRoleModel([controller_payload()])
        solver = FakeRoleModel([decision])
        verifier = FakeRoleModel([verifier_payload()])
        agent, request = self.agent(
            controller=controller,
            solver=solver,
            verifier=verifier,
            mode=SolveMode.OPEN_BOOK,
            evidence_provider=lambda *_: (evidence,),
            revisions=0,
        )

        result = agent.solve(request)

        self.assertTrue(result.answer.needs_review)
        self.assertIn("UNKNOWN_EVIDENCE_ID", result.answer.warnings)
        self.assertIn("CLAIM_OPTION_VERDICT_CONFLICT", result.answer.warnings)

    def test_runtime_rebinds_mispartitioned_controller_claims(self) -> None:
        plan = controller_payload()
        option_claims = plan["option_claims"]
        assert isinstance(option_claims, dict)
        option_claims["C"] = ["C1 第一必要命题", "C2 第二必要命题"]
        option_claims["D"] = ["D1 错挂到D的C项命题", "D2 D项命题"]
        controller = FakeRoleModel([plan])
        solver = FakeRoleModel([solver_payload()])
        verifier = FakeRoleModel([verifier_payload()])
        agent, request = self.agent(
            controller=controller,
            solver=solver,
            verifier=verifier,
            revisions=0,
        )

        result = agent.solve(request)

        self.assertFalse(result.answer.needs_review)
        self.assertEqual(
            result.controller_plan.option_claims,
            {
                option.label: (f"{option.label}1 {option.text}",)
                for option in request.options
            },
        )
        solver_context = json.loads(solver.requests[0].messages[1].content)
        self.assertEqual(
            solver_context["controller_plan"]["claim_binding"],
            "deterministic-option-text-v1",
        )

    def test_expired_or_low_authority_evidence_fails_closed(self) -> None:
        for evidence in (
            LegalEvidence(
                evidence_id="low",
                title="低权威来源",
                url="https://example.invalid/low",
                publisher="example",
                source_type="statute",
                authority_level=2,
                quote="低权威内容",
            ),
            LegalEvidence(
                evidence_id="expired",
                title="旧法",
                url="https://example.invalid/old",
                publisher="example",
                source_type="statute",
                authority_level=5,
                quote="已失效内容",
                effective_until=date(2020, 1, 1),
            ),
        ):
            with self.subTest(evidence=evidence.evidence_id):
                controller = FakeRoleModel([controller_payload()])
                agent, request = self.agent(
                    controller=controller,
                    solver=FakeRoleModel([]),
                    verifier=FakeRoleModel([]),
                    mode=SolveMode.OPEN_BOOK,
                    evidence_provider=lambda *_, value=evidence: (value,),
                )
                with self.assertRaisesRegex(
                    LegalAgentError,
                    "no_authoritative_source",
                ):
                    agent.solve(request)

    def test_evaluator_reads_oracle_only_after_agent_completion(self) -> None:
        case = BenchmarkRecordAdapter.adapt(
            {"id": 1, "question": QUESTION, "golden_answers": ["C"]}
        )
        controller = FakeRoleModel([controller_payload()])
        solver = FakeRoleModel([solver_payload()])
        verifier = FakeRoleModel([verifier_payload()])
        agent, _ = self.agent(
            controller=controller, solver=solver, verifier=verifier
        )
        completed: list[bool] = []

        record = evaluate_case(
            case,
            agent.solve,
            on_agent_completed=lambda: completed.append(True),
        )

        self.assertTrue(record.exact_match)
        self.assertEqual(completed, [True])


class ConfigAndIntegrationTest(unittest.TestCase):
    def test_config_loads_role_models_and_rejects_conflicting_route(self) -> None:
        payload = {
            "generation": {
                "backend": "openai",
                "backend_configs": {
                    "openai": {
                        "api_key": "",
                        "base_url": "https://offline.invalid/v1",
                        "model_name": "fallback",
                    }
                },
            },
            "legal_mcq": {
                "enabled": True,
                "mode": "closed_book",
                "controller_model": {"model_name": "controller-cheap"},
                "solver_model": {"model_name": "solver-strong"},
                "verifier_model": {"model_name": "verifier-cheap"},
            },
            "lgagent_plus": {"enabled": False},
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            path.write_text(yaml.safe_dump(payload), encoding="utf-8")
            config = load_lgagent_config(path, environ={})
            self.assertTrue(config.legal_mcq.enabled)
            assert config.legal_mcq.solver_model is not None
            self.assertEqual(config.legal_mcq.solver_model.model, "solver-strong")

            payload["lgagent_plus"]["enabled"] = True
            path.write_text(yaml.safe_dump(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "mutually exclusive"):
                load_lgagent_config(path, environ={})

    def test_unified_runner_uses_opt_in_route_and_switches_off_cleanly(self) -> None:
        controller = FakeRoleModel([controller_payload()])
        solver = FakeRoleModel([solver_payload()])
        verifier = FakeRoleModel([verifier_payload()])
        config = LGAgentConfig(
            generation=CONFIG,
            lgagent_plus=LGAgentPlusConfig(enabled=False),
            legal_mcq=LegalMCQSettings(
                enabled=True,
                controller_model=ModelConfig(
                    **{**asdict(CONFIG), "model": "controller-cheap"}
                ),
                solver_model=ModelConfig(
                    **{**asdict(CONFIG), "model": "solver-strong"}
                ),
                verifier_model=ModelConfig(
                    **{**asdict(CONFIG), "model": "verifier-cheap"}
                ),
            ),
        )

        result = LGAgentPlusRunner(
            config,
            reasoning_model=controller,
            evaluation_model=solver,
            verifier_model=verifier,
            legal_controller_model=controller,
            legal_solver_model=solver,
            legal_verifier_model=verifier,
        ).run(QUESTION)

        self.assertEqual(result.final_answer, "C")
        self.assertEqual(
            result.diagnostics["pipeline_route"], "legal-mcq-three-role"
        )
        self.assertEqual(
            result.diagnostics["model_policy"]["solver"], "solver-strong"
        )
        self.assertEqual(result.diagnostics["golden_leakage_incidents"], 0)


class RuntimeBoundaryTest(unittest.TestCase):
    def test_skill_catalog_hides_evaluation_skill(self) -> None:
        registry = LegalSkillRegistry.load()

        production = {item.name for item in registry.production_catalog()}
        self.assertEqual(production, {"legal-mcq-core", "legal-research"})
        self.assertEqual(
            [item.name for item in registry.preload(SolveMode.CLOSED_BOOK)],
            ["legal-mcq-core"],
        )
        self.assertEqual(
            [item.name for item in registry.preload(SolveMode.OPEN_BOOK)],
            ["legal-mcq-core", "legal-research"],
        )

    def test_role_clients_honor_distinct_endpoints_and_share_identical_ones(
        self,
    ) -> None:
        controller_config = replace(
            CONFIG,
            backend="openai",
            base_url="https://cheap.invalid/v1",
            api_key="cheap-key",
            model="controller",
        )
        solver_config = replace(
            CONFIG,
            backend="openai",
            base_url="https://strong.invalid/v1",
            api_key="strong-key",
            model="solver",
        )
        config = LGAgentConfig(
            generation=controller_config,
            lgagent_plus=LGAgentPlusConfig(enabled=False),
            legal_mcq=LegalMCQSettings(
                enabled=True,
                controller_model=controller_config,
                solver_model=solver_config,
                solver_fallback_model=replace(
                    controller_config,
                    model="solver-fallback",
                ),
                verifier_model=replace(controller_config, model="verifier"),
            ),
        )
        created: list[dict[str, object]] = []

        def factory(**kwargs):
            created.append(kwargs)
            return object()

        clients = build_openai_role_clients(config, client_factory=factory)

        self.assertIs(clients.controller, clients.verifier)
        self.assertIs(clients.controller, clients.solver_fallback)
        self.assertIsNot(clients.controller, clients.solver)
        self.assertEqual(len(created), 2)
        self.assertEqual(
            {str(item["base_url"]) for item in created},
            {"https://cheap.invalid/v1", "https://strong.invalid/v1"},
        )
        self.assertTrue(all(item["max_retries"] == 0 for item in created))
        self.assertTrue(all(item["timeout"] == 60.0 for item in created))

    def test_oath_adapter_deduplicates_audited_full_text_and_rejects_local_uri(
        self,
    ) -> None:
        item = AuditedEvidence(
            evidence_id="oath:1",
            label=AuditLabel.SUPPORT,
            exact_span="法律正文",
            retrieval_lanes=("support",),
            source_type="statute",
            law_name="示例法",
            article="第一条",
            clause=None,
            authority_level=5,
            effective_from=date(2020, 1, 1),
            effective_to=None,
            source_uri="https://law.example/1",
        )
        matrix = AuditedEvidenceMatrix(
            options={
                "A": OptionEvidenceAudit(support=(item,)),
                "B": OptionEvidenceAudit(support=(item,)),
            }
        )

        evidence = adapt_audited_evidence(matrix)

        self.assertEqual(len(evidence), 1)
        self.assertEqual(evidence[0].publisher, "law.example")
        self.assertTrue(evidence[0].content_hash)

        second_span = replace(item, exact_span="同一法条的另一段正文")
        split_evidence = adapt_audited_evidence(
            AuditedEvidenceMatrix(
                options={
                    "A": OptionEvidenceAudit(support=(item,)),
                    "B": OptionEvidenceAudit(refute=(second_span,)),
                }
            )
        )
        self.assertEqual(len(split_evidence), 2)
        self.assertTrue(
            all(":span:" in value.evidence_id for value in split_evidence)
        )

        invalid = replace(item, evidence_id="oath:2", source_uri="file:///law")
        with self.assertRaises(LegalEvidenceAdapterError):
            adapt_audited_evidence(
                AuditedEvidenceMatrix(
                    options={"A": OptionEvidenceAudit(support=(invalid,))}
                )
            )

    def test_impossible_revision_budget_blocks_before_any_provider_call(self) -> None:
        controller = FakeRoleModel([controller_payload()])
        solver = FakeRoleModel([solver_payload("D"), solver_payload("C")])
        verifier = FakeRoleModel([verifier_payload(False, suggested="C")])
        config = LGAgentConfig(
            generation=CONFIG,
            lgagent_plus=LGAgentPlusConfig(enabled=False),
            legal_mcq=LegalMCQSettings(
                enabled=True,
                max_model_calls=3,
                max_revision_rounds=1,
                controller_model=replace(CONFIG, model="controller"),
                solver_model=replace(CONFIG, model="solver"),
                verifier_model=replace(CONFIG, model="verifier"),
            ),
        )
        runner = LGAgentPlusRunner(
            config,
            reasoning_model=controller,
            evaluation_model=solver,
            verifier_model=verifier,
            legal_controller_model=controller,
            legal_solver_model=solver,
            legal_verifier_model=verifier,
        )

        with self.assertRaises(BudgetExceededError):
            runner.run(QUESTION)
        self.assertEqual(len(controller.requests), 0)
        self.assertEqual(len(solver.requests), 0)
        self.assertEqual(len(verifier.requests), 0)

    def test_disabled_feature_never_calls_legal_role_models(self) -> None:
        class BaselineReasoning:
            def complete(self, request: ModelRequest) -> ModelResponse:
                agent = request.metadata["agent"]
                if agent == "lawyer_a":
                    payload = {
                        "task_type": "single_choice",
                        "question_focus": "权限限制",
                        "legal_domain": "商法",
                        "jurisdiction": "CN",
                        "case_date": None,
                        "facts": [],
                        "option_claims": {
                            label: {
                                "claim": f"{label}项",
                                "elements": [],
                                "possible_exceptions": [],
                            }
                            for label in "ABCD"
                        },
                        "option_keywords": {
                            label: [label] for label in "ABCD"
                        },
                        "trap_signals": [],
                        "unknowns": [],
                    }
                elif agent == "judge":
                    payload = {
                        "need_retrieval": False,
                        "global_query": "内部权限限制",
                        "option_queries": {
                            label: label for label in "ABCD"
                        },
                        "evidence_requirements": {
                            label: {"support": "rule", "refute": "exception"}
                            for label in "ABCD"
                        },
                        "counterfactual_focus": "善意第三人",
                        "stop_rule": "all options checked",
                    }
                else:
                    raise AssertionError(f"unexpected reasoning agent: {agent}")
                return ModelResponse(json.dumps(payload, ensure_ascii=False))

        class BaselineEvaluation:
            def complete(self, request: ModelRequest) -> ModelResponse:
                agent = request.metadata["agent"]
                if agent == "lawyer_b0":
                    payload = {"initial_answer": "C", "confidence": 0.9}
                elif agent == "lawyer_b1":
                    payload = {
                        "final_answer": "C",
                        "verification": {
                            label: {
                                "status": (
                                    "SUPPORT" if label == "C" else "REFUTE"
                                ),
                                "score": 0.9,
                                "reason": "checked",
                            }
                            for label in "ABCD"
                        },
                        "initial_answer": "C",
                        "initial_confidence": 0.9,
                        "reasoning": "checked",
                    }
                else:
                    raise AssertionError(f"unexpected evaluation agent: {agent}")
                return ModelResponse(json.dumps(payload, ensure_ascii=False))

        class ForbiddenLegalModel:
            def complete(self, request: ModelRequest) -> ModelResponse:
                raise AssertionError(
                    f"disabled LegalMCQ role was called: {request.metadata}"
                )

        config = LGAgentConfig(
            generation=CONFIG,
            lgagent_plus=LGAgentPlusConfig(enabled=False),
            legal_mcq=LegalMCQSettings(enabled=False),
        )
        result = LGAgentPlusRunner(
            config,
            reasoning_model=BaselineReasoning(),
            evaluation_model=BaselineEvaluation(),
            legal_controller_model=ForbiddenLegalModel(),
            legal_solver_model=ForbiddenLegalModel(),
            legal_verifier_model=ForbiddenLegalModel(),
            max_dialogue_rounds=0,
        ).run(QUESTION)

        self.assertEqual(result.final_answer, "C")
        self.assertEqual(result.diagnostics["pipeline_route"], "corrected_baseline")


if __name__ == "__main__":
    unittest.main()
