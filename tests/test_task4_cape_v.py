from __future__ import annotations

import json
import math
import sys
import unittest
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from lgagent.cape_v import (
    CapeVConfig,
    CapeVRunner,
    OptionParseError,
    calculate_cape_v_metrics,
    generate_option_permutations,
    parse_four_option_question,
)
from lgagent.config import ModelConfig
from lgagent.model import ModelRequest, ModelResponse, TokenUsage

QUESTION = """Which rule applies?

A. Alpha rule
B. Beta rule
C. Gamma rule
D. Delta rule"""

CONFIG = ModelConfig(
    backend="fake",
    base_url="https://example.invalid/v1",
    api_key="",
    model="fake-model",
    temperature=0.7,
    top_p=0.8,
    max_tokens=2048,
)


def candidate_payload(answer: str) -> str:
    return json.dumps(
        {
            "answer": answer,
            "irac": {
                "issue": "Which rule controls?",
                "rule": [{"claim": "Applicable rule", "evidence_ids": []}],
                "application": [
                    {
                        "fact_ids": [],
                        "rule_index": 0,
                        "inference": "The facts satisfy the rule.",
                    }
                ],
                "conclusion": f"Option {answer} follows.",
            },
            "option_verification": {
                label: {
                    "status": "SUPPORT" if label == answer else "REFUTE",
                    "score": 0.9,
                    "evidence_ids": [],
                }
                for label in "ABCD"
            },
        }
    )


class TextIdentityModel:
    """Always selects the displayed label carrying the stable Beta option text."""

    def __init__(self) -> None:
        self.requests: list[ModelRequest] = []

    def complete(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        parsed = parse_four_option_question(request.messages[-1].content)
        answer = next(
            option.original_label
            for option in parsed.options
            if option.identity == "Beta rule"
        )
        return ModelResponse(
            candidate_payload(answer),
            TokenUsage(prompt_tokens=5, completion_tokens=7, total_tokens=12),
            request_id=f"fake-{len(self.requests)}",
        )


class ConstantModel:
    def __init__(self, answer: str = "A") -> None:
        self.answer = answer
        self.requests: list[ModelRequest] = []

    def complete(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        return ModelResponse(candidate_payload(self.answer))


class OptionParsingAndPermutationTest(unittest.TestCase):
    def test_parses_supported_markers_and_preserves_multiline_option_text(self) -> None:
        parsed = parse_four_option_question(
            """Question stem

(A) first line
continued line
B、second
C: third
D．fourth"""
        )

        self.assertEqual(parsed.stem, "Question stem")
        self.assertEqual(parsed.option("A").text, "first line\ncontinued line")
        self.assertEqual(parsed.option("A").identity, "first line continued line")
        self.assertEqual([option.original_label for option in parsed.options], list("ABCD"))

    def test_rejects_missing_markers_and_duplicate_text_identities(self) -> None:
        with self.assertRaisesRegex(OptionParseError, "exactly one"):
            parse_four_option_question("Question\nA. one\nB. two\nD. four")
        with self.assertRaisesRegex(OptionParseError, "not unique"):
            parse_four_option_question(
                "Question\nA. same text\nB. same   text\nC. three\nD. four"
            )

    def test_seeded_permutations_are_unique_and_bidirectional(self) -> None:
        parsed = parse_four_option_question(QUESTION)
        first = generate_option_permutations(parsed, count=8, seed=42)
        second = generate_option_permutations(parsed, count=8, seed=42)

        first_orders = [
            tuple(permutation.displayed_to_original[label] for label in "ABCD")
            for permutation in first
        ]
        second_orders = [
            tuple(permutation.displayed_to_original[label] for label in "ABCD")
            for permutation in second
        ]
        self.assertEqual(first_orders, second_orders)
        self.assertEqual(len(set(first_orders)), 8)
        self.assertNotIn(tuple("ABCD"), first_orders)

        for permutation in first:
            for original in "ABCD":
                displayed = permutation.original_to_displayed[original]
                self.assertEqual(permutation.map_answer_to_original(displayed), original)
                identity = permutation.original_to_identity[original]
                self.assertEqual(permutation.answer_identity(displayed), identity)


class CapeVMetricsTest(unittest.TestCase):
    def test_distribution_entropy_and_permutation_consistency(self) -> None:
        metrics = calculate_cape_v_metrics(
            ["A", "B", "C", "D"],
            ["B", "B", "B", "C"],
        )

        self.assertEqual(metrics.answer_counts, dict.fromkeys("ABCD", 1))
        self.assertEqual(metrics.answer_distribution, dict.fromkeys("ABCD", 0.25))
        self.assertTrue(math.isclose(metrics.normalized_answer_entropy, 1.0))
        self.assertEqual(metrics.permutation_consistency, 0.75)

    def test_unanimous_candidates_have_zero_entropy(self) -> None:
        metrics = calculate_cape_v_metrics(["C", "C", "C"])
        self.assertEqual(metrics.normalized_answer_entropy, 0.0)
        self.assertIsNone(metrics.permutation_consistency)


class CapeVRunnerTest(unittest.TestCase):
    def test_generates_independent_candidates_and_maps_permuted_answers(self) -> None:
        model = TextIdentityModel()
        runner = CapeVRunner(
            model,
            CONFIG,
            CapeVConfig(candidate_count=3, permutation_count=3, seed=7),
        )

        result = runner.run(QUESTION)

        self.assertEqual([candidate.answer for candidate in result.candidates], ["B"] * 3)
        self.assertEqual(
            [candidate.answer for candidate in result.permutation_candidates],
            ["B"] * 3,
        )
        self.assertTrue(
            all(
                candidate.answer_identity == "Beta rule"
                for candidate in (
                    *result.candidates,
                    *result.permutation_candidates,
                )
            )
        )
        self.assertEqual(result.metrics.answer_distribution["B"], 1.0)
        self.assertEqual(result.metrics.normalized_answer_entropy, 0.0)
        self.assertEqual(result.metrics.permutation_consistency, 1.0)
        self.assertEqual(len(model.requests), 6)
        self.assertTrue(all(len(request.messages) == 2 for request in model.requests))
        self.assertEqual(
            len({request.metadata["candidate_id"] for request in model.requests}),
            6,
        )
        self.assertEqual(result.trace.total_tokens, 72)

    def test_unparseable_question_disables_only_permutations_and_records_reason(self) -> None:
        model = ConstantModel("A")
        runner = CapeVRunner(
            model,
            CONFIG,
            CapeVConfig(candidate_count=2, permutation_count=3),
        )

        result = runner.run("Question without a reliable option block")

        self.assertFalse(result.permutation_enabled)
        self.assertIn("exactly one option marker", result.permutation_disabled_reason or "")
        self.assertEqual(len(result.candidates), 2)
        self.assertEqual(result.permutation_candidates, ())
        self.assertEqual(len(model.requests), 2)
        self.assertEqual(result.trace.routes[0].route, "cape_v_permutation_disabled")
        self.assertEqual(
            result.trace.routes[0].reason,
            result.permutation_disabled_reason,
        )

    def test_invalid_candidate_json_is_retried_explicitly(self) -> None:
        class RetryModel(ConstantModel):
            def complete(self, request: ModelRequest) -> ModelResponse:
                self.requests.append(request)
                if len(self.requests) == 1:
                    return ModelResponse('{"answer":"A"}')
                return ModelResponse(candidate_payload("A"))

        model = RetryModel()
        result = CapeVRunner(
            model,
            CONFIG,
            CapeVConfig(candidate_count=1, permutation_count=0, max_attempts=2),
        ).run(QUESTION)

        self.assertEqual(result.candidates[0].answer, "A")
        self.assertEqual(len(model.requests), 2)
        self.assertIn("不符合紧凑IRAC JSON协议", model.requests[1].messages[-1].content)
        self.assertEqual(result.trace.model_calls[0].error_type, "StructuredOutputError")


if __name__ == "__main__":
    unittest.main()
