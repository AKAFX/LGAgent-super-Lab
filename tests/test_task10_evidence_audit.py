from __future__ import annotations

import json
import sys
import unittest
from datetime import date
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from lgagent.config import ModelConfig
from lgagent.corpus import LegalEvidence
from lgagent.domain import run_single_question
from lgagent.evidence_audit import (
    AuditLabel,
    AuditedEvidence,
    AuditedEvidenceMatrix,
    EvidenceAuditConfig,
    EvidenceAuditError,
    EvidenceAuditPipeline,
    EvidenceAuditor,
    EvidenceRetrievalError,
    OptionEvidenceAudit,
    RetrievalFailureMode,
    RetrievalStatus,
    TemporalStatus,
    build_option_evidence_audit,
)
from lgagent.model import ModelRequest, ModelResponse
from lgagent.oath_rag import RetrievedEvidence
from lgagent.serialization import dumps_json
from lgagent.trace import RunTrace


CONFIG = ModelConfig(
    backend="fake",
    base_url="https://example.invalid",
    api_key="",
    model="fake-model",
    temperature=0.7,
    top_p=0.8,
    max_tokens=2048,
)


class FakeModel:
    def __init__(self, outputs: list[str]) -> None:
        self.outputs = list(outputs)
        self.requests: list[ModelRequest] = []

    def complete(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        return ModelResponse(self.outputs.pop(0))


def evidence(
    evidence_id: str,
    text: str,
    *,
    authority_level: int = 5,
    effective_from: date = date(2021, 1, 1),
    effective_to: date | None = None,
) -> LegalEvidence:
    return LegalEvidence(
        evidence_id=evidence_id,
        source_type="statute",
        law_name="Test Act",
        article=evidence_id,
        clause=None,
        version="v1",
        text=text,
        jurisdiction="CN",
        authority_level=authority_level,
        effective_from=effective_from,
        effective_to=effective_to,
        source_uri=f"https://example.invalid/{evidence_id}",
    )


def retrieved_matrix(items: list[LegalEvidence]):
    retrieved = tuple(RetrievedEvidence(item, score=1.0) for item in items)
    return {
        "A": {
            "support": retrieved,
            "refute": retrieved,
            "exception": retrieved,
        }
    }


def analysis() -> dict[str, object]:
    return {
        "option_claims": {"A": {"claim": "the transfer is valid"}},
        "case_date": "2024-01-01",
    }


def audited(
    evidence_id: str,
    label: AuditLabel,
    *,
    authority_level: int = 5,
) -> AuditedEvidence:
    return AuditedEvidence(
        evidence_id=evidence_id,
        label=label,
        exact_span=evidence_id,
        retrieval_lanes=("support",),
        source_type="statute",
        law_name="Test Act",
        article="1",
        clause=None,
        authority_level=authority_level,
        effective_from=date(2021, 1, 1),
        effective_to=None,
        source_uri=f"https://example.invalid/{evidence_id}",
    )


class EvidenceAuditorOfflineTest(unittest.TestCase):
    def test_labels_all_pairs_and_hard_gates_temporally_invalid_evidence(self) -> None:
        items = [
            evidence("support-id", "registration makes the transfer effective"),
            evidence("refute-id", "without registration the transfer is ineffective"),
            evidence("exception-id", "except for a protected good-faith acquirer"),
            evidence("irrelevant-id", "this provision governs filing fees"),
            evidence(
                "expired-id",
                "an expired historical transfer rule",
                effective_from=date(2010, 1, 1),
                effective_to=date(2020, 12, 31),
            ),
        ]
        model = FakeModel(
            [
                json.dumps(
                    {
                        "audits": [
                            {
                                "evidence_id": "support-id",
                                "label": "SUPPORT",
                                "exact_span": "transfer effective",
                            },
                            {
                                "evidence_id": "refute-id",
                                "label": "REFUTE",
                                "exact_span": "transfer is ineffective",
                            },
                            {
                                "evidence_id": "exception-id",
                                "label": "EXCEPTION",
                                "exact_span": "except for a protected good-faith acquirer",
                            },
                            {
                                "evidence_id": "irrelevant-id",
                                "label": "IRRELEVANT",
                                "exact_span": "filing fees",
                            },
                        ]
                    }
                )
            ]
        )

        trace = RunTrace()
        matrix = EvidenceAuditor(model, CONFIG).audit(
            analysis(),
            retrieved_matrix(items),
            trace=trace,
        )

        option = matrix.options["A"]
        self.assertEqual(option.support[0].evidence_id, "support-id")
        self.assertEqual(option.refute[0].evidence_id, "refute-id")
        self.assertEqual(option.exception[0].evidence_id, "exception-id")
        self.assertEqual(option.irrelevant[0].evidence_id, "irrelevant-id")
        self.assertEqual(
            option.temporally_invalid[0].evidence_id,
            "expired-id",
        )
        self.assertEqual(option.coverage, 1.0)
        self.assertAlmostEqual(option.conflict, 0.5)
        self.assertEqual(option.authority, 1.0)
        self.assertEqual(option.temporal, TemporalStatus.MIXED)
        self.assertNotIn("expired-id", model.requests[0].messages[-1].content)
        self.assertEqual(model.requests[0].temperature, 0.0)
        self.assertEqual(len(trace.model_calls), 1)
        self.assertEqual(trace.model_calls[0].agent, "evidence_auditor")

    def test_rejects_unknown_id_and_non_exact_span(self) -> None:
        item = evidence("provided-id", "only this exact source text is available")
        cases = (
            {
                "evidence_id": "invented-id",
                "label": "SUPPORT",
                "exact_span": "exact source",
            },
            {
                "evidence_id": "provided-id",
                "label": "SUPPORT",
                "exact_span": "paraphrased source",
            },
        )
        for payload in cases:
            with self.subTest(payload=payload):
                model = FakeModel([json.dumps({"audits": [payload]})])
                with self.assertRaises(EvidenceAuditError):
                    EvidenceAuditor(
                        model,
                        CONFIG,
                        config=EvidenceAuditConfig(max_attempts=1),
                    ).audit(
                        analysis(),
                        retrieved_matrix([item]),
                    )

    def test_retries_strict_output_validation_failures(self) -> None:
        item = evidence("provided-id", "only this exact source text is available")
        model = FakeModel(
            [
                json.dumps({"audits": []}),
                json.dumps(
                    {
                        "audits": [
                            {
                                "evidence_id": "provided-id",
                                "label": "SUPPORT",
                                "exact_span": "paraphrased source text",
                            }
                        ]
                    }
                ),
                json.dumps(
                    {
                        "audits": [
                            {
                                "evidence_id": "provided-id",
                                "label": "SUPPORT",
                                "exact_span": "exact source text",
                            }
                        ]
                    }
                ),
            ]
        )
        trace = RunTrace()

        matrix = EvidenceAuditor(model, CONFIG).audit(
            analysis(),
            retrieved_matrix([item]),
            trace=trace,
        )

        self.assertEqual(
            matrix.options["A"].support[0].evidence_id,
            "provided-id",
        )
        self.assertEqual(len(model.requests), 3)
        self.assertEqual(model.requests[1].metadata["attempt"], 2)
        self.assertEqual(model.requests[2].metadata["attempt"], 3)
        self.assertIn("omitted evidence IDs", model.requests[1].messages[-1].content)
        self.assertIn(
            "not an exact evidence substring",
            model.requests[2].messages[-1].content,
        )
        self.assertEqual([call.attempt for call in trace.model_calls], [1, 2, 3])
        self.assertEqual(trace.model_calls[0].error_type, "EvidenceAuditError")
        self.assertEqual(trace.model_calls[1].error_type, "EvidenceAuditError")
        self.assertIsNone(trace.model_calls[2].error_type)

    def test_missing_case_date_uses_injected_current_date(self) -> None:
        expired = evidence(
            "expired-id",
            "expired provision",
            effective_from=date(2010, 1, 1),
            effective_to=date(2020, 12, 31),
        )
        matrix = EvidenceAuditor(
            FakeModel([]),
            CONFIG,
            current_date=date(2024, 1, 1),
        ).audit(
            {"option_claims": {"A": "claim"}},
            retrieved_matrix([expired]),
        )

        self.assertEqual(
            matrix.options["A"].temporal,
            TemporalStatus.INVALID,
        )

    def test_status_calculation_is_pure_and_matrix_is_json_serializable(self) -> None:
        option = build_option_evidence_audit(
            [
                audited("support", AuditLabel.SUPPORT, authority_level=4),
                audited("refute", AuditLabel.REFUTE, authority_level=2),
                audited("exception", AuditLabel.EXCEPTION, authority_level=2),
            ],
            authority_ceiling=5,
        )
        matrix = AuditedEvidenceMatrix(options={"A": option})

        self.assertEqual(option.coverage, 1.0)
        self.assertEqual(option.conflict, 1.0)
        self.assertEqual(option.authority, 0.8)
        self.assertEqual(option.temporal, TemporalStatus.VALID)
        serialized = json.loads(dumps_json(matrix))
        self.assertEqual(serialized["options"]["A"]["support"][0]["evidence_id"], "support")
        self.assertEqual(serialized["options"]["A"]["temporal"], "VALID")


class RetrievalFailurePolicyTest(unittest.TestCase):
    class BrokenRetriever:
        def retrieve(
            self,
            analysis_value,
            *,
            jurisdiction=None,
            case_date=None,
        ):
            _ = analysis_value, jurisdiction, case_date
            raise RuntimeError("offline index unavailable")

    def test_retrieval_failure_is_fail_closed_by_default(self) -> None:
        auditor = EvidenceAuditor(FakeModel([]), CONFIG)
        pipeline = EvidenceAuditPipeline(self.BrokenRetriever(), auditor)

        with self.assertRaisesRegex(EvidenceRetrievalError, "offline index unavailable"):
            pipeline.run(analysis())

    def test_configured_fallback_returns_explicit_empty_matrix(self) -> None:
        auditor = EvidenceAuditor(
            FakeModel([]),
            CONFIG,
            config=EvidenceAuditConfig(
                retrieval_failure_mode=RetrievalFailureMode.EMPTY_EVIDENCE
            ),
        )
        matrix = EvidenceAuditPipeline(self.BrokenRetriever(), auditor).run(analysis())

        self.assertEqual(matrix.retrieval_status, RetrievalStatus.FALLBACK_EMPTY)
        self.assertIn("offline index unavailable", matrix.retrieval_error or "")
        self.assertEqual(matrix.options["A"].temporal, TemporalStatus.NO_EVIDENCE)


def lawyer_a_payload() -> dict[str, object]:
    return {
        "task_type": "single_choice",
        "question_focus": "focus",
        "legal_domain": "civil_law",
        "option_claims": {option: f"claim-{option}" for option in "ABCD"},
        "option_keywords": {option: [f"keyword-{option}"] for option in "ABCD"},
        "trap_signals": [],
        "unknowns": [],
    }


def judge_payload() -> dict[str, object]:
    return {
        "need_retrieval": True,
        "global_query": "query",
        "option_queries": {option: f"query-{option}" for option in "ABCD"},
        "evidence_requirements": {
            option: {"support": "support", "refute": "refute"}
            for option in "ABCD"
        },
        "counterfactual_focus": "focus",
        "stop_rule": "stop",
    }


def b1_payload() -> dict[str, object]:
    return {
        "final_answer": "A",
        "verification": {
            option: {
                "status": "SUPPORT" if option == "A" else "REFUTE",
                "score": 0.8,
                "reason": "audited",
            }
            for option in "ABCD"
        },
        "initial_answer": "A",
        "initial_confidence": 0.9,
        "reasoning": "matrix-grounded",
    }


class LawyerB1ContextTest(unittest.TestCase):
    def test_audited_matrix_replaces_unverified_documents_in_b1_context(self) -> None:
        reasoning = FakeModel(
            [json.dumps(lawyer_a_payload()), json.dumps(judge_payload())]
        )
        evaluation = FakeModel(
            [
                '{"initial_answer":"A","confidence":0.9}',
                json.dumps(b1_payload()),
            ]
        )
        options = {option: OptionEvidenceAudit() for option in "ABCD"}
        options["A"] = build_option_evidence_audit(
            [audited("stable-law-id", AuditLabel.SUPPORT)]
        )
        matrix = AuditedEvidenceMatrix(options=options)

        result = run_single_question(
            "Question A B C D",
            reasoning_model=reasoning,
            evaluation_model=evaluation,
            reasoning_config=CONFIG,
            evaluation_config=CONFIG,
            retrieved_docs=["UNVERIFIED DOCUMENT"],
            evidence_matrix=matrix,
        )

        b1_context = evaluation.requests[1].messages[-1].content
        self.assertIn("stable-law-id", b1_context)
        self.assertIn("exact_span", b1_context)
        self.assertNotIn("UNVERIFIED DOCUMENT", b1_context)
        self.assertEqual(
            result.diagnostics["evidence_matrix"]["options"]["A"]["coverage"],
            1 / 3,
        )

    def test_budgeted_web_provider_runs_after_b0_and_supplies_b1_context(self) -> None:
        reasoning = FakeModel(
            [json.dumps(lawyer_a_payload()), json.dumps(judge_payload())]
        )
        evaluation = FakeModel(
            [
                '{"initial_answer":"A","confidence":0.4}',
                json.dumps(b1_payload()),
            ]
        )
        observed: dict[str, object] = {}

        def provider(question, analysis_value, judge, b0, trace):
            observed.update(
                question=question,
                domain=analysis_value.legal_domain,
                query=judge.global_query,
                confidence=b0.confidence,
            )
            trace.add_route("web-search", "fixture")
            return (
                [
                    "[UNTRUSTED WEB EVIDENCE]\n"
                    "evidence_id: web:1\n"
                    "url: https://example.invalid/law\n"
                    "content: current legal rule"
                ],
                {
                    "searched": True,
                    "estimated_context_tokens": 30,
                },
            )

        result = run_single_question(
            "Question A B C D",
            reasoning_model=reasoning,
            evaluation_model=evaluation,
            reasoning_config=CONFIG,
            evaluation_config=CONFIG,
            retrieved_docs_provider=provider,
            max_dialogue_rounds=0,
        )

        self.assertEqual(observed["confidence"], 0.4)
        self.assertEqual(observed["query"], "query")
        self.assertIn("web:1", evaluation.requests[1].messages[-1].content)
        self.assertTrue(result.diagnostics["web_search"]["searched"])


if __name__ == "__main__":
    unittest.main()
