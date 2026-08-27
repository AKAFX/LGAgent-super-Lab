from __future__ import annotations

import json
import math
import socket
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import yaml

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from lgagent.cape_v import generate_option_permutations, parse_four_option_question
from lgagent.config import ConfigurationError, ModelConfig, load_lgagent_config
from lgagent.corpus import CorpusValidationError, load_jsonl_corpus
from lgagent.domain import run_single_question
from lgagent.evidence_audit import EvidenceAuditPipeline, EvidenceAuditor
from lgagent.model import ModelRequest, ModelResponse
from lgagent.oath_rag import OathRagConfig, OathRagRetriever
from lgagent.protocol import (
    B0Output,
    B1Output,
    StructuredOutputError,
    extract_json_object,
    should_request_clarification,
)
from lgagent.risk import RiskMetrics, RiskWeights, calculate_risk_score
from lgagent.serialization import dumps_json


CONFIG = ModelConfig(
    backend="fake",
    base_url="https://offline.invalid/v1",
    api_key="task15-test-secret",
    model="fake-model",
    temperature=0.0,
    top_p=1.0,
    max_tokens=1024,
    api_key_source="yaml",
)

QUESTION = """Which rule applies?

A. Alpha rule
B. Beta rule
C. Gamma rule
D. Delta rule"""


def lawyer_a_payload() -> dict[str, object]:
    return {
        "task_type": "single_choice",
        "question_focus": "ownership",
        "legal_domain": "civil_law",
        "option_claims": {option: f"{name} ownership" for option, name in zip("ABCD", ("alpha", "beta", "gamma", "delta"))},
        "option_keywords": {option: [name] for option, name in zip("ABCD", ("alpha", "beta", "gamma", "delta"))},
        "trap_signals": [],
        "unknowns": [],
    }


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


def b1_payload(
    statuses: tuple[str, str, str, str] = ("REFUTE", "SUPPORT", "REFUTE", "REFUTE"),
) -> dict[str, object]:
    return {
        "final_answer": "B",
        "verification": {
            option: {"status": status, "score": 0.9, "reason": "audited"}
            for option, status in zip("ABCD", statuses)
        },
        "initial_answer": "B",
        "initial_confidence": 0.9,
        "reasoning": "matrix-grounded",
    }


def corpus_record(
    evidence_id: str,
    article: str,
    text: str,
    **overrides: object,
) -> dict[str, object]:
    value: dict[str, object] = {
        "evidence_id": evidence_id,
        "source_type": "statute",
        "law_name": "Test Act",
        "article": article,
        "clause": None,
        "version": "2024",
        "text": text,
        "jurisdiction": "CN",
        "authority_level": 5,
        "effective_from": "2024-01-01",
        "effective_to": None,
        "source_uri": f"https://offline.invalid/{evidence_id}",
        "relations": [],
    }
    value.update(overrides)
    return value


def write_corpus(path: Path, records: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


class SequenceModel:
    def __init__(self, outputs: list[str]) -> None:
        self.outputs = list(outputs)
        self.requests: list[ModelRequest] = []

    def complete(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        return ModelResponse(self.outputs.pop(0))


class GroundedAuditModel:
    def __init__(self) -> None:
        self.requests: list[ModelRequest] = []

    def complete(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        payload = json.loads(request.messages[-1].content)
        audits = [
            {
                "evidence_id": item["evidence_id"],
                "label": "SUPPORT",
                "exact_span": item["text"],
            }
            for item in payload["evidence"]
        ]
        return ModelResponse(json.dumps({"audits": audits}))


class SchemaJsonConfigTest(unittest.TestCase):
    def test_json_and_agent_schema_reject_malformed_values(self) -> None:
        invalid_json = (
            '{"answer":"A",}',
            '{"answer":"A"} trailing',
            '```json\n{"answer":"A"}\n``` trailing',
        )
        for value in invalid_json:
            with self.subTest(value=value), self.assertRaises(StructuredOutputError):
                extract_json_object(value)

        invalid_b0 = (
            '{"initial_answer":"A","confidence":true}',
            '{"initial_answer":"E","confidence":0.5}',
            '{"initial_answer":"A","confidence":1.01}',
        )
        for value in invalid_b0:
            with self.subTest(value=value), self.assertRaises(StructuredOutputError):
                B0Output.from_text(value)

        duplicate_option = b1_payload()
        duplicate_option["verification"]["a"] = duplicate_option["verification"].pop("D")  # type: ignore[index,union-attr]
        with self.assertRaises(StructuredOutputError):
            B1Output.from_text(json.dumps(duplicate_option))

    def test_jsonl_rejects_duplicate_keys_nonfinite_numbers_and_non_objects(self) -> None:
        invalid_lines = (
            '{"evidence_id":"one","evidence_id":"two"}\n',
            '{"authority_level":NaN}\n',
            '["not-an-object"]\n',
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.jsonl"
            for raw in invalid_lines:
                with self.subTest(raw=raw):
                    path.write_text(raw, encoding="utf-8")
                    with self.assertRaises(CorpusValidationError):
                        load_jsonl_corpus(path)

    def test_config_rejects_unknown_fields_and_invalid_yaml(self) -> None:
        source = ROOT_DIR / "examples" / "parameter" / "legal2_rag_parameter.yaml"
        payload = yaml.safe_load(source.read_text(encoding="utf-8"))
        payload["lgagent_plus"]["risk"]["unexpected"] = True
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            path.write_text(yaml.safe_dump(payload), encoding="utf-8")
            with self.assertRaisesRegex(ConfigurationError, "unknown fields"):
                load_lgagent_config(path, environ={})

            path.write_text("generation: [unterminated", encoding="utf-8")
            with self.assertRaises(ConfigurationError):
                load_lgagent_config(path, environ={})


class DialogueBoundaryTest(unittest.TestCase):
    def test_nei_ratio_and_confidence_boundaries_are_exact(self) -> None:
        one_nei = B1Output.from_text(
            json.dumps(b1_payload(("NEI", "SUPPORT", "REFUTE", "REFUTE")))
        )
        two_nei = B1Output.from_text(
            json.dumps(b1_payload(("NEI", "NEI", "REFUTE", "REFUTE")))
        )

        self.assertFalse(should_request_clarification(one_nei.verification, 0.6))
        self.assertTrue(should_request_clarification(one_nei.verification, 0.599999))
        self.assertTrue(should_request_clarification(two_nei.verification, 1.0))

    def test_zero_dialogue_budget_reports_exhaustion_without_extra_calls(self) -> None:
        reasoning = SequenceModel(
            [json.dumps(lawyer_a_payload()), json.dumps(judge_payload())]
        )
        evaluation = SequenceModel(
            [
                '{"initial_answer":"B","confidence":0.59}',
                json.dumps(b1_payload()),
            ]
        )
        result = run_single_question(
            QUESTION,
            reasoning_model=reasoning,
            evaluation_model=evaluation,
            reasoning_config=CONFIG,
            evaluation_config=CONFIG,
            max_dialogue_rounds=0,
        )

        self.assertEqual(result.diagnostics["dialogue_rounds"], 0)
        self.assertTrue(result.diagnostics["dialogue_exhausted"])
        self.assertEqual(len(reasoning.requests) + len(evaluation.requests), 4)


class PermutationAndRiskPropertyTest(unittest.TestCase):
    def test_all_non_identity_permutations_are_bijections(self) -> None:
        parsed = parse_four_option_question(QUESTION)
        permutations = generate_option_permutations(parsed, count=23, seed=2026)
        orders = {
            tuple(permutation.displayed_to_original[label] for label in "ABCD")
            for permutation in permutations
        }

        self.assertEqual(len(permutations), math.factorial(4) - 1)
        self.assertEqual(len(orders), len(permutations))
        self.assertNotIn(tuple("ABCD"), orders)
        for permutation in permutations:
            self.assertEqual(set(permutation.displayed_to_original), set("ABCD"))
            self.assertEqual(set(permutation.displayed_to_original.values()), set("ABCD"))
            for displayed in "ABCD":
                original = permutation.map_answer_to_original(displayed)
                self.assertEqual(permutation.original_to_displayed[original], displayed)

    def test_risk_score_is_monotonic_in_every_risk_component(self) -> None:
        baseline = RiskMetrics(
            normalized_answer_entropy=0.0,
            top2_margin=1.0,
            low_top2_margin=0.0,
            evidence_coverage=1.0,
            missing_evidence_coverage=0.0,
            evidence_conflict=0.0,
            verifier_disagreement=0.0,
            permutation_instability=0.0,
        )
        weights = RiskWeights()
        components = tuple(baseline.risk_components())
        low_score = calculate_risk_score(baseline, weights)

        for component in components:
            with self.subTest(component=component):
                higher = replace(baseline, **{component: 0.25})
                self.assertGreater(calculate_risk_score(higher, weights), low_score)


class TemporalGraphAndEndToEndTest(unittest.TestCase):
    def test_temporal_boundaries_apply_before_graph_expansion(self) -> None:
        records = [
            corpus_record(
                "source",
                "1",
                "alpha ownership",
                effective_from="2023-01-01",
                effective_to="2024-01-01",
                relations=[
                    {"type": "REFERS_TO", "target_id": "boundary"},
                    {"type": "REFERS_TO", "target_id": "future"},
                ],
            ),
            corpus_record("boundary", "2", "supplemental boundary provision"),
            corpus_record(
                "future",
                "3",
                "future supplemental provision",
                effective_from="2024-01-02",
            ),
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "corpus.jsonl"
            write_corpus(path, records)
            retriever = OathRagRetriever(
                load_jsonl_corpus(path),
                config=OathRagConfig(final_top_k_per_lane=3, graph_hops=1),
            )
            matrix = retriever.retrieve(
                {
                    "legal_domain": "property",
                    "question_focus": "ownership",
                    "jurisdiction": "CN",
                    "case_date": "2024-01-01",
                    "facts": [],
                    "option_claims": {"A": "alpha ownership"},
                }
            )

        by_id = {item.evidence_id: item for item in matrix["A"]["support"]}
        self.assertIn("source", by_id)
        self.assertIn("boundary", by_id)
        self.assertEqual(by_id["boundary"].expanded_from, "source")
        self.assertNotIn("future", by_id)

    def test_fake_models_and_small_corpus_run_offline_end_to_end(self) -> None:
        records = [
            corpus_record(evidence_id, str(index), f"{name} ownership rule")
            for index, (evidence_id, name) in enumerate(
                zip(("law-a", "law-b", "law-c", "law-d"), ("alpha", "beta", "gamma", "delta")),
                start=1,
            )
        ]
        analysis = {
            "legal_domain": "property",
            "question_focus": "ownership",
            "jurisdiction": "CN",
            "case_date": "2024-01-01",
            "facts": [],
            "option_claims": lawyer_a_payload()["option_claims"],
        }
        audit_model = GroundedAuditModel()
        reasoning = SequenceModel(
            [json.dumps(lawyer_a_payload()), json.dumps(judge_payload())]
        )
        evaluation = SequenceModel(
            [
                '{"initial_answer":"B","confidence":0.9}',
                json.dumps(b1_payload()),
            ]
        )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "small-corpus.jsonl"
            write_corpus(path, records)
            retriever = OathRagRetriever(
                load_jsonl_corpus(path),
                config=OathRagConfig(final_top_k_per_lane=1, graph_hops=0),
            )
            pipeline = EvidenceAuditPipeline(
                retriever,
                EvidenceAuditor(audit_model, CONFIG),
            )
            with (
                patch.object(
                    socket,
                    "create_connection",
                    side_effect=AssertionError("network access attempted"),
                ),
                patch.object(
                    socket.socket,
                    "connect",
                    side_effect=AssertionError("network access attempted"),
                ),
            ):
                matrix = pipeline.run(analysis)
                result = run_single_question(
                    QUESTION,
                    reasoning_model=reasoning,
                    evaluation_model=evaluation,
                    reasoning_config=CONFIG,
                    evaluation_config=CONFIG,
                    evidence_matrix=matrix,
                )

        self.assertEqual(set(matrix.options), set("ABCD"))
        self.assertTrue(all(option.support for option in matrix.options.values()))
        self.assertEqual(result.final_answer, "B")
        self.assertEqual(set(result.diagnostics["evidence_matrix"]["options"]), set("ABCD"))

        serialized = dumps_json(
            {
                "config": CONFIG,
                "matrix": matrix,
                "result": result,
            }
        )
        request_text = dumps_json(
            [
                request
                for model in (audit_model, reasoning, evaluation)
                for request in model.requests
            ]
        )
        self.assertNotIn(CONFIG.api_key, serialized)
        self.assertNotIn(CONFIG.api_key, request_text)
        self.assertIn("[REDACTED]", serialized)


if __name__ == "__main__":
    unittest.main()
