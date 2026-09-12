from __future__ import annotations

import json
import itertools
import sys
import tempfile
import unittest
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from lgagent.config import (
    CapeVSettings,
    ClexSettings,
    LGAgentConfig,
    LGAgentPlusConfig,
    ModelConfig,
    OathRagSettings,
    RiskSettings,
    WebSearchSettings,
)
from lgagent.corpus import CorpusValidationError
from lgagent.model import ModelRequest, ModelResponse
from lgagent.risk import BudgetLimits, RiskThresholds
from lgagent.runner import LGAgentPlusRunner


QUESTION = """Which ownership rule applies on 2024-06-01 in CN?

A. Alpha ownership
B. Beta ownership
C. Gamma ownership
D. Delta ownership"""

MODEL_CONFIG = ModelConfig(
    backend="fake",
    base_url="https://offline.invalid/v1",
    api_key="",
    model="fake",
    temperature=0.0,
    top_p=1.0,
    max_tokens=1024,
)


def lawyer_a_payload(*, legacy: bool = False) -> dict[str, object]:
    claims: dict[str, object]
    if legacy:
        claims = {
            option: f"{name} ownership"
            for option, name in zip("ABCD", ("alpha", "beta", "gamma", "delta"))
        }
    else:
        claims = {
            option: {
                "claim": f"{name} ownership",
                "elements": [f"{name} element"],
                "possible_exceptions": [f"{name} exception"],
            }
            for option, name in zip("ABCD", ("alpha", "beta", "gamma", "delta"))
        }
    payload: dict[str, object] = {
        "task_type": "single_choice",
        "question_focus": "ownership",
        "legal_domain": "civil_law",
        "option_claims": claims,
        "option_keywords": {
            option: [name]
            for option, name in zip("ABCD", ("alpha", "beta", "gamma", "delta"))
        },
        "trap_signals": [],
        "unknowns": [],
    }
    if not legacy:
        payload.update(
            jurisdiction="CN",
            case_date="2024-06-01",
            facts=[
                {
                    "fact_id": "F1",
                    "text": "The transfer occurred in CN.",
                    "legally_relevant": True,
                }
            ],
        )
    return payload


def judge_payload() -> dict[str, object]:
    return {
        "need_retrieval": True,
        "global_query": "ownership",
        "option_queries": {option: f"query-{option}" for option in "ABCD"},
        "evidence_requirements": {
            option: {"support": "support", "refute": "refute"}
            for option in "ABCD"
        },
        "counterfactual_focus": "exception",
        "stop_rule": "all options checked",
    }


def b1_payload(answer: str = "A") -> dict[str, object]:
    return {
        "final_answer": answer,
        "verification": {
            option: {
                "status": "SUPPORT" if option == answer else "REFUTE",
                "score": 0.9,
                "reason": "checked",
            }
            for option in "ABCD"
        },
        "initial_answer": "A",
        "initial_confidence": 0.9,
        "reasoning": "initial B1 result",
    }


def candidate_payload(answer: str) -> str:
    return json.dumps(
        {
            "answer": answer,
            "irac": {
                "issue": "ownership",
                "rule": [
                    {
                        "claim": f"{answer} rule",
                        "evidence_ids": [f"law-{answer.lower()}"],
                    }
                ],
                "application": [
                    {
                        "fact_ids": ["F1"],
                        "rule_index": 0,
                        "inference": "the rule applies",
                    }
                ],
                "conclusion": answer,
            },
            "option_verification": {
                option: {
                    "status": "SUPPORT" if option == answer else "REFUTE",
                    "score": 0.9,
                    "evidence_ids": [f"law-{option.lower()}"],
                }
                for option in "ABCD"
            },
        }
    )


class ReasoningModel:
    def __init__(self, events: list[str], *, legacy_analysis: bool = False) -> None:
        self.events = events
        self.legacy_analysis = legacy_analysis

    def complete(self, request: ModelRequest) -> ModelResponse:
        agent = str(request.metadata["agent"])
        self.events.append(agent)
        if agent == "lawyer_a":
            return ModelResponse(json.dumps(lawyer_a_payload(legacy=self.legacy_analysis)))
        if agent == "judge":
            return ModelResponse(json.dumps(judge_payload()))
        if agent == "evidence_auditor":
            payload = json.loads(request.messages[-1].content)
            return ModelResponse(
                json.dumps(
                    {
                        "audits": [
                            {
                                "evidence_id": item["evidence_id"],
                                "label": "SUPPORT",
                                "exact_span": item["text"],
                            }
                            for item in payload["evidence"]
                        ]
                    }
                )
            )
        raise AssertionError(f"unexpected reasoning agent: {agent}")


class EvaluationModel:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.cape_answers = itertools.cycle(("A", "B"))
        self.cape_requests: list[ModelRequest] = []
        self.requests: list[ModelRequest] = []

    def complete(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        agent = str(request.metadata["agent"])
        self.events.append(agent)
        if agent == "lawyer_b0":
            return ModelResponse('{"initial_answer":"A","confidence":0.9}')
        if agent == "lawyer_b1":
            return ModelResponse(json.dumps(b1_payload()))
        if agent == "cape_v_candidate":
            self.cape_requests.append(request)
            return ModelResponse(candidate_payload(next(self.cape_answers)))
        raise AssertionError(f"unexpected evaluation agent: {agent}")


class VerifierModel:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def complete(self, request: ModelRequest) -> ModelResponse:
        agent = str(request.metadata["agent"])
        self.events.append(agent)
        score = 0.9 if request.metadata["candidate_id"].endswith("2") else 0.4
        return ModelResponse(
            json.dumps(
                {
                    "pass": score >= 0.5,
                    "score": score,
                    "error_type": None if score >= 0.5 else "WEAK_CANDIDATE",
                    "reason": "deterministic fake verification",
                }
            )
        )


def write_corpus(path: Path) -> None:
    records = []
    for index, (option, name) in enumerate(
        zip("ABCD", ("alpha", "beta", "gamma", "delta")),
        start=1,
    ):
        records.append(
            {
                "evidence_id": f"law-{option.lower()}",
                "source_type": "statute",
                "law_name": "Test Act",
                "article": str(index),
                "clause": None,
                "version": "2024",
                "text": f"{name} ownership {name} element rule applies",
                "jurisdiction": "CN",
                "authority_level": 5,
                "effective_from": "2024-01-01",
                "effective_to": None,
                "source_uri": f"https://offline.invalid/law-{option.lower()}",
                "relations": [],
            }
        )
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def config(
    corpus_path: Path,
    *,
    enabled: bool,
    oath: bool,
    cape: bool,
    corpus_failure_mode: str = "fail_closed",
    initial_candidates: int = 2,
    permutation_count: int = 0,
    enable_counterfactual: bool = False,
    clex_calibration_path: Path | None = None,
    web_search: bool = False,
    risk_thresholds: RiskThresholds | None = None,
    risk_budget: BudgetLimits | None = None,
) -> LGAgentConfig:
    return LGAgentConfig(
        generation=MODEL_CONFIG,
        lgagent_plus=LGAgentPlusConfig(
            enabled=enabled,
            seed=7,
            oath_rag=OathRagSettings(
                enabled=oath,
                corpus_path=str(corpus_path),
                lexical_top_k=4,
                dense_top_k=4,
                final_top_k_per_lane=1,
                graph_hops=0,
                corpus_failure_mode=corpus_failure_mode,
            ),
            cape_v=CapeVSettings(
                enabled=cape,
                initial_candidates=initial_candidates,
                slow_path_candidates=1,
                permutation_count=permutation_count,
                enable_counterfactual=enable_counterfactual,
            ),
            risk=RiskSettings(
                thresholds=risk_thresholds or RiskThresholds(),
                budget=risk_budget or BudgetLimits(),
            ),
            clex=ClexSettings(
                enabled=clex_calibration_path is not None,
                calibration_path=(
                    str(clex_calibration_path)
                    if clex_calibration_path is not None
                    else ""
                ),
                alpha=0.1,
                min_calibration_size=30,
                group_field="domain",
            ),
            web_search=WebSearchSettings(enabled=web_search),
        ),
    )


class Task17IntegrationTest(unittest.TestCase):
    def run_route(
        self,
        corpus_path: Path,
        *,
        enabled: bool,
        oath: bool,
        cape: bool,
    ):
        events: list[str] = []
        evaluation = EvaluationModel(events)
        result = LGAgentPlusRunner(
            config(
                corpus_path,
                enabled=enabled,
                oath=oath,
                cape=cape,
                risk_thresholds=RiskThresholds(low=0.99, high=1.0),
                risk_budget=BudgetLimits(
                    max_calls=100,
                    max_tokens=1_000_000,
                    max_rounds=3,
                    max_seconds=120.0,
                ),
            ),
            reasoning_model=ReasoningModel(events),
            evaluation_model=evaluation,
            verifier_model=VerifierModel(events),
            project_root=ROOT_DIR,
            max_dialogue_rounds=0,
        ).run(QUESTION)
        return result, events, evaluation

    def test_original_oath_cape_and_joint_execute_real_routes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            corpus_path = Path(directory) / "corpus.jsonl"
            write_corpus(corpus_path)

            original, original_events, _ = self.run_route(
                corpus_path, enabled=False, oath=True, cape=True
            )
            oath_only, oath_events, _ = self.run_route(
                corpus_path, enabled=True, oath=True, cape=False
            )
            cape_only, cape_events, cape_model = self.run_route(
                corpus_path, enabled=True, oath=False, cape=True
            )
            joint, joint_events, joint_model = self.run_route(
                corpus_path, enabled=True, oath=True, cape=True
            )

        self.assertEqual(original.final_answer, "A")
        self.assertEqual(original.diagnostics["pipeline_route"], "corrected_baseline")
        self.assertNotIn("evidence_auditor", original_events)
        self.assertNotIn("cape_v_candidate", original_events)

        self.assertEqual(oath_only.final_answer, "A")
        self.assertEqual(oath_only.diagnostics["pipeline_route"], "oath-only")
        self.assertLess(
            oath_events.index("evidence_auditor"),
            oath_events.index("lawyer_b1"),
        )
        self.assertIn("evidence_matrix", oath_only.diagnostics)

        self.assertEqual(cape_only.final_answer, "B")
        self.assertEqual(cape_only.diagnostics["pipeline_route"], "cape-only")
        self.assertEqual(cape_events.count("cape_v_candidate"), 2)
        self.assertEqual(
            {
                event.removeprefix("cape_v_verifier_")
                for event in cape_events
                if event.startswith("cape_v_verifier_")
            },
            {"rule", "fact", "exception", "evidence", "entailment"},
        )
        self.assertNotIn(
            "audited_evidence_matrix",
            cape_model.cape_requests[0].messages[-1].content,
        )

        self.assertEqual(joint.final_answer, "B")
        self.assertEqual(joint.diagnostics["pipeline_route"], "joint")
        self.assertLess(
            joint_events.index("evidence_auditor"),
            joint_events.index("lawyer_b1"),
        )
        self.assertLess(
            joint_events.index("evidence_auditor"),
            joint_events.index("cape_v_candidate"),
        )
        self.assertIn(
            "audited_evidence_matrix",
            joint_model.cape_requests[0].messages[-1].content,
        )
        self.assertEqual(
            joint.diagnostics["risk_routing"]["selected_candidate_id"],
            "candidate-2",
        )
        self.assertEqual(len(joint.diagnostics["verification_reports"]), 2)

    def test_clex_uses_frozen_calibration_without_extra_model_calls(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corpus_path = root / "unused.jsonl"
            calibration_path = root / "clex.json"
            calibration_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "score_version": "clex-option-conformity-v1",
                        "alpha": 0.1,
                        "global_threshold": 0.45,
                        "global_count": 100,
                        "group_field": "domain",
                        "group_thresholds": {"civil_law": 0.45},
                        "group_counts": {"civil_law": 50},
                        "records_sha256": "fixture",
                    }
                ),
                encoding="utf-8",
            )
            events: list[str] = []
            result = LGAgentPlusRunner(
                config(
                    corpus_path,
                    enabled=True,
                    oath=False,
                    cape=True,
                    initial_candidates=2,
                    permutation_count=0,
                    clex_calibration_path=calibration_path,
                    risk_thresholds=RiskThresholds(low=0.99, high=1.0),
                    risk_budget=BudgetLimits(
                        max_calls=100,
                        max_tokens=1_000_000,
                        max_rounds=1,
                        max_seconds=120.0,
                    ),
                ),
                reasoning_model=ReasoningModel(events),
                evaluation_model=EvaluationModel(events),
                verifier_model=VerifierModel(events),
            ).run(QUESTION)

        self.assertEqual(result.final_answer, "B")
        self.assertEqual(result.diagnostics["clex"]["prediction_set"], ["B"])
        self.assertEqual(
            result.diagnostics["clex"]["threshold_source"],
            "domain:civil_law",
        )
        self.assertEqual(events.count("cape_v_candidate"), 2)
        self.assertIn("clex", [item.route for item in result.trace.routes])

    def test_clex_invalid_artifact_fails_before_any_model_call(self) -> None:
        events: list[str] = []
        missing = ROOT_DIR / "does-not-exist-clex-calibration.json"
        with self.assertRaises(FileNotFoundError):
            LGAgentPlusRunner(
                config(
                    ROOT_DIR / "unused-corpus.jsonl",
                    enabled=True,
                    oath=False,
                    cape=True,
                    clex_calibration_path=missing,
                ),
                reasoning_model=ReasoningModel(events),
                evaluation_model=EvaluationModel(events),
                verifier_model=VerifierModel(events),
            ).run(QUESTION)

        self.assertEqual(events, [])

    def test_web_search_runs_without_oath_or_cape(self) -> None:
        class FakeWebSearch:
            def retrieve(self, question, analysis, judge, b0, trace):
                _ = question, analysis, judge, b0
                trace.add_route("web-search", "fixture")
                return type(
                    "Outcome",
                    (),
                    {
                        "documents": (
                            "[UNTRUSTED WEB EVIDENCE]\n"
                            "evidence_id: web:1\n"
                            "url: https://gov.example/rule\n"
                            "content: current rule",
                        ),
                        "as_dict": lambda self: {
                            "searched": True,
                            "estimated_context_tokens": 20,
                        },
                    },
                )()

        with tempfile.TemporaryDirectory() as directory:
            events: list[str] = []
            evaluation = EvaluationModel(events)
            result = LGAgentPlusRunner(
                config(
                    Path(directory) / "unused.jsonl",
                    enabled=True,
                    oath=False,
                    cape=False,
                    web_search=True,
                ),
                reasoning_model=ReasoningModel(events),
                evaluation_model=evaluation,
                web_search_pipeline=FakeWebSearch(),
            ).run(QUESTION)

        self.assertEqual(result.diagnostics["pipeline_route"], "web-search-only")
        self.assertTrue(result.diagnostics["web_search"]["searched"])
        self.assertIn(
            "web:1",
            evaluation.requests[-1].messages[-1].content,
        )
        self.assertNotIn("evidence_auditor", events)
        self.assertNotIn("cape_v_candidate", events)

    def test_legacy_lawyer_a_output_remains_accepted(self) -> None:
        from lgagent.protocol import LawyerAOutput

        parsed = LawyerAOutput.from_text(json.dumps(lawyer_a_payload(legacy=True)))
        self.assertEqual(parsed.jurisdiction, "")
        self.assertIsNone(parsed.case_date)
        self.assertEqual(parsed.option_claims["A"].claim, "alpha ownership")
        self.assertEqual(parsed.option_claims["A"].elements, ())

    def test_oath_uses_default_jurisdiction_when_analysis_omits_it(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            corpus_path = Path(directory) / "corpus.jsonl"
            write_corpus(corpus_path)
            events: list[str] = []

            result = LGAgentPlusRunner(
                config(
                    corpus_path,
                    enabled=True,
                    oath=True,
                    cape=False,
                ),
                reasoning_model=ReasoningModel(events, legacy_analysis=True),
                evaluation_model=EvaluationModel(events),
            ).run(QUESTION)

        self.assertEqual(result.final_answer, "A")
        self.assertEqual(
            result.diagnostics["evidence_matrix"]["retrieval_status"],
            "SUCCESS",
        )
        self.assertIn("evidence_auditor", events)

    def test_permutation_probe_does_not_vote_or_run_five_verifiers(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            corpus_path = Path(directory) / "corpus.jsonl"
            write_corpus(corpus_path)
            events: list[str] = []
            result = LGAgentPlusRunner(
                config(
                    corpus_path,
                    enabled=True,
                    oath=False,
                    cape=True,
                    initial_candidates=1,
                    permutation_count=1,
                ),
                reasoning_model=ReasoningModel(events),
                evaluation_model=EvaluationModel(events),
                verifier_model=VerifierModel(events),
                max_dialogue_rounds=2,
            ).run(QUESTION)

        verifier_events = [
            event for event in events if event.startswith("cape_v_verifier_")
        ]
        self.assertEqual(events.count("cape_v_candidate"), 2)
        self.assertEqual(len(verifier_events), 5)
        self.assertEqual(result.final_answer, "A")
        self.assertEqual(result.diagnostics["dialogue_rounds"], 0)

    def test_strict_corpus_fails_or_uses_explicit_empty_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            legacy_path = Path(directory) / "legacy.jsonl"
            legacy_path.write_text(
                '{"id":"legacy-1","title":"Old Act","contents":"alpha ownership"}\n',
                encoding="utf-8",
            )
            events: list[str] = []
            with self.assertRaises(CorpusValidationError):
                LGAgentPlusRunner(
                    config(
                        legacy_path,
                        enabled=True,
                        oath=True,
                        cape=False,
                    ),
                    reasoning_model=ReasoningModel(events),
                    evaluation_model=EvaluationModel(events),
                ).run(QUESTION)

            events = []
            result = LGAgentPlusRunner(
                config(
                    legacy_path,
                    enabled=True,
                    oath=True,
                    cape=False,
                    corpus_failure_mode="empty_evidence",
                ),
                reasoning_model=ReasoningModel(events),
                evaluation_model=EvaluationModel(events),
            ).run(QUESTION)

        matrix = result.diagnostics["evidence_matrix"]
        self.assertEqual(matrix["retrieval_status"], "FALLBACK_EMPTY")
        self.assertIn("legacy contents records are disabled", matrix["retrieval_error"])
        self.assertNotIn("evidence_auditor", events)

    def test_medium_only_appends_candidates_but_high_risk_retrieves_again(self) -> None:
        budget = BudgetLimits(
            max_calls=100,
            max_tokens=1_000_000,
            max_rounds=1,
            max_seconds=120.0,
        )
        with tempfile.TemporaryDirectory() as directory:
            corpus_path = Path(directory) / "corpus.jsonl"
            write_corpus(corpus_path)

            medium_events: list[str] = []
            medium = LGAgentPlusRunner(
                config(
                    corpus_path,
                    enabled=True,
                    oath=True,
                    cape=True,
                    risk_thresholds=RiskThresholds(low=0.0, high=1.0),
                    risk_budget=budget,
                ),
                reasoning_model=ReasoningModel(medium_events),
                evaluation_model=EvaluationModel(medium_events),
                verifier_model=VerifierModel(medium_events),
            ).run(QUESTION)

            high_events: list[str] = []
            high_evaluation = EvaluationModel(high_events)
            high = LGAgentPlusRunner(
                config(
                    corpus_path,
                    enabled=True,
                    oath=True,
                    cape=True,
                    risk_thresholds=RiskThresholds(low=0.0, high=0.01),
                    risk_budget=budget,
                ),
                reasoning_model=ReasoningModel(high_events),
                evaluation_model=high_evaluation,
                verifier_model=VerifierModel(high_events),
            ).run(QUESTION)

        self.assertEqual(medium_events.count("evidence_auditor"), 4)
        self.assertEqual(high_events.count("evidence_auditor"), 8)
        self.assertEqual(medium.diagnostics["targeted_retrievals"], [])
        self.assertEqual(len(high.diagnostics["targeted_retrievals"]), 1)
        high_gaps = high.diagnostics["targeted_retrievals"][0]["gaps"]
        self.assertTrue(
            any("failed" in gap for gaps in high_gaps.values() for gap in gaps)
        )
        medium_routes = [item.route for item in medium.trace.routes]
        high_routes = [item.route for item in high.trace.routes]
        self.assertIn("verify-more", medium_routes)
        self.assertNotIn("targeted-retrieval-complete", medium_routes)
        self.assertIn("retrieve-and-reason", high_routes)
        self.assertIn("targeted-retrieval-complete", high_routes)
        slow_context = high_evaluation.cape_requests[-1].messages[-1].content
        self.assertIn("targeted_analysis", slow_context)
        self.assertIn("failed rule verification", slow_context)

    def test_cape_only_high_risk_records_retrieval_degradation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            corpus_path = Path(directory) / "unused.jsonl"
            events: list[str] = []
            result = LGAgentPlusRunner(
                config(
                    corpus_path,
                    enabled=True,
                    oath=False,
                    cape=True,
                    risk_thresholds=RiskThresholds(low=0.0, high=0.01),
                    risk_budget=BudgetLimits(
                        max_calls=100,
                        max_tokens=1_000_000,
                        max_rounds=1,
                        max_seconds=120.0,
                    ),
                ),
                reasoning_model=ReasoningModel(events),
                evaluation_model=EvaluationModel(events),
                verifier_model=VerifierModel(events),
            ).run(QUESTION)

        routes = [item.route for item in result.trace.routes]
        self.assertIn("retrieve-and-reason-degraded", routes)
        self.assertNotIn("evidence_auditor", events)
        self.assertEqual(result.diagnostics["targeted_retrievals"], [])

    def test_counterfactual_switch_controls_calls_and_records_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            corpus_path = Path(directory) / "unused.jsonl"
            disabled_events: list[str] = []
            disabled = LGAgentPlusRunner(
                config(
                    corpus_path,
                    enabled=True,
                    oath=False,
                    cape=True,
                    initial_candidates=1,
                    permutation_count=1,
                    enable_counterfactual=False,
                ),
                reasoning_model=ReasoningModel(disabled_events),
                evaluation_model=EvaluationModel(disabled_events),
                verifier_model=VerifierModel(disabled_events),
            ).run(QUESTION)

            enabled_events: list[str] = []
            enabled = LGAgentPlusRunner(
                config(
                    corpus_path,
                    enabled=True,
                    oath=False,
                    cape=True,
                    initial_candidates=1,
                    permutation_count=1,
                    enable_counterfactual=True,
                ),
                reasoning_model=ReasoningModel(enabled_events),
                evaluation_model=EvaluationModel(enabled_events),
                verifier_model=VerifierModel(enabled_events),
            ).run(QUESTION)

        self.assertEqual(disabled_events.count("cape_v_candidate"), 2)
        self.assertEqual(enabled_events.count("cape_v_candidate"), 3)
        self.assertEqual(disabled.diagnostics["counterfactual_observations"], [])
        observations = enabled.diagnostics["counterfactual_observations"]
        self.assertEqual(
            {
                item["transformation"]["transformation_type"]
                for item in observations
            },
            {"OPTION_REORDER", "HARMLESS_TEXT_NORMALIZATION"},
        )
        self.assertTrue(all(item["eligible"] for item in observations))
        self.assertTrue(all(not item["expected_change"] for item in observations))
        self.assertEqual(
            enabled.diagnostics["counterfactual_metrics"][
                "should_not_change_total"
            ],
            2,
        )
        routes = [item.route for item in enabled.trace.routes]
        self.assertIn("counterfactual-complete", routes)

    def test_cli_and_batch_import_the_unified_runner(self) -> None:
        for relative_path in (
            "tools/legal_multi_agent_prompt_demo.py",
            "tools/legal_multi_agent_eval_optimized.py",
        ):
            source = (ROOT_DIR / relative_path).read_text(encoding="utf-8")
            self.assertIn("LGAgentPlusRunner(", source)


if __name__ == "__main__":
    unittest.main()
