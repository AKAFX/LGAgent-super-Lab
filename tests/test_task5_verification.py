from __future__ import annotations

import json
import sys
import time
import unittest
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from lgagent.cape_v import (
    CandidateOptionVerification,
    CompactIRAC,
    IRACApplication,
    IRACRule,
    ReasoningCandidate,
)
from lgagent.config import ModelConfig
from lgagent.model import ModelRequest, ModelResponse, TokenUsage
from lgagent.trace import RunTrace
from lgagent.verification import (
    VERIFIER_NAMES,
    VerificationContext,
    VerificationRunner,
    VerifierConfig,
)


def candidate() -> ReasoningCandidate:
    return ReasoningCandidate(
        candidate_id="candidate-1",
        answer="B",
        answer_identity="Beta rule",
        presented_answer="B",
        permutation_id="original",
        irac=CompactIRAC(
            issue="Which rule controls?",
            rule=(IRACRule("Rule claim", ("law:1",)),),
            application=(IRACApplication(("f1",), 0, "Fact satisfies rule."),),
            conclusion="B follows.",
        ),
        option_verification={
            label: CandidateOptionVerification(
                "SUPPORT" if label == "B" else "REFUTE",
                0.9,
                ("law:1",),
            )
            for label in "ABCD"
        },
    )


VERIFIER_CONFIG = ModelConfig(
    backend="fake",
    base_url="https://example.invalid/v1",
    api_key="",
    model="independent-verifier-model",
    temperature=0.0,
    top_p=1.0,
    max_tokens=256,
)


def enabled_only(name: str) -> dict[str, bool]:
    return {verifier: verifier == name for verifier in VERIFIER_NAMES}


class DimensionModel:
    def __init__(self) -> None:
        self.requests: list[ModelRequest] = []

    def complete(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        name = str(request.metadata["verifier"])
        failed = name == "exception"
        return ModelResponse(
            json.dumps(
                {
                    "pass": not failed,
                    "score": 0.4 if failed else 0.9,
                    "error_type": "MISSED_EXCEPTION" if failed else None,
                    "reason": f"{name} checked",
                }
            ),
            TokenUsage(prompt_tokens=3, completion_tokens=2, total_tokens=5),
            request_id=f"fake-{name}",
        )


class VerificationRunnerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.context = VerificationContext(
            question="Question\nA. Alpha\nB. Beta\nC. Gamma\nD. Delta",
            candidate=candidate(),
            facts=({"id": "f1", "text": "A fact"},),
            evidence=({"evidence_id": "law:1", "text": "Rule text"},),
            exception_hints=("Check exception X",),
        )

    def test_runs_five_independent_prompts_with_separate_model_config(self) -> None:
        model = DimensionModel()
        trace = RunTrace()
        report = VerificationRunner(
            model,
            VERIFIER_CONFIG,
            VerifierConfig(max_attempts=1),
        ).run(self.context, trace=trace)

        self.assertEqual(list(report.dimensions), list(VERIFIER_NAMES))
        self.assertAlmostEqual(report.overall_score, 0.8)
        self.assertEqual(report.failed_steps, ("exception",))
        self.assertEqual(report.error_types, ("MISSED_EXCEPTION",))
        self.assertAlmostEqual(report.disagreement, 0.4)
        self.assertEqual(len(model.requests), 5)
        self.assertTrue(
            all(request.model == "independent-verifier-model" for request in model.requests)
        )
        self.assertEqual(
            {request.metadata["verifier"] for request in model.requests},
            set(VERIFIER_NAMES),
        )
        self.assertEqual(report.as_dict()["dimensions"]["rule"]["pass"], True)
        self.assertEqual(trace.total_tokens, 25)

    def test_invalid_json_retries_then_reports_explicit_failure(self) -> None:
        class InvalidModel:
            def __init__(self) -> None:
                self.requests: list[ModelRequest] = []

            def complete(self, request: ModelRequest) -> ModelResponse:
                self.requests.append(request)
                return ModelResponse('{"pass": true}')

        model = InvalidModel()
        report = VerificationRunner(
            model,
            VERIFIER_CONFIG,
            VerifierConfig(
                enabled=enabled_only("rule"),
                max_attempts=2,
            ),
        ).run(self.context)

        self.assertEqual(len(model.requests), 2)
        self.assertIn("不符合验证JSON协议", model.requests[1].messages[-1].content)
        self.assertEqual(report.failed_steps, ("rule",))
        self.assertEqual(report.error_types, ("VERIFIER_INVALID_JSON",))
        self.assertEqual(report.overall_score, 0.0)

    def test_timeout_fails_closed_without_retrying_blocked_call(self) -> None:
        class SlowModel:
            def complete(self, request: ModelRequest) -> ModelResponse:
                _ = request
                time.sleep(0.05)
                return ModelResponse(
                    '{"pass":true,"score":1,"error_type":null,"reason":"late"}'
                )

        report = VerificationRunner(
            SlowModel(),
            VERIFIER_CONFIG,
            VerifierConfig(
                enabled=enabled_only("evidence"),
                timeout_seconds=0.005,
                max_attempts=3,
            ),
        ).run(self.context)

        result = report.dimensions["evidence"]
        self.assertFalse(result.passed)
        self.assertEqual(result.error_type, "VERIFIER_TIMEOUT")
        self.assertEqual(result.attempts, 1)


if __name__ == "__main__":
    unittest.main()
