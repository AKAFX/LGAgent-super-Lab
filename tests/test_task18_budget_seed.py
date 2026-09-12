from __future__ import annotations

import tempfile
import threading
import time
import unittest
from dataclasses import replace
from pathlib import Path

from lgagent.corpus import load_jsonl_corpus
from lgagent.evidence_audit import EvidenceAuditPipeline, EvidenceAuditor
from lgagent.model import (
    BudgetExceededError,
    BudgetedSeededChatModel,
    ChatMessage,
    ExecutionBudget,
    ModelRequest,
    ModelResponse,
    TokenUsage,
)
from lgagent.oath_rag import OathRagConfig, OathRagRetriever
from lgagent.risk import BudgetLimits
from lgagent.runner import LGAgentPlusRunner

from test_task17_integration import (
    EvaluationModel,
    MODEL_CONFIG,
    QUESTION,
    ReasoningModel,
    VerifierModel,
    config,
    write_corpus,
)


def request(
    *,
    agent: str = "test",
    max_tokens: int = 1,
    candidate_id: str | None = None,
) -> ModelRequest:
    metadata = {"agent": agent, "attempt": 1}
    if candidate_id is not None:
        metadata["candidate_id"] = candidate_id
    return ModelRequest(
        model="fake",
        messages=(ChatMessage("user", "question"),),
        max_tokens=max_tokens,
        metadata=metadata,
    )


class RecordingModel:
    def __init__(self, usages: list[int] | None = None) -> None:
        self.requests: list[ModelRequest] = []
        self.usages = list(usages or [])

    def complete(self, model_request: ModelRequest) -> ModelResponse:
        self.requests.append(model_request)
        tokens = self.usages.pop(0) if self.usages else 0
        return ModelResponse("ok", usage=TokenUsage(total_tokens=tokens))


class ExecutionBudgetTest(unittest.TestCase):
    def test_reasoning_effort_survives_budget_and_seed_wrapper(self) -> None:
        provider = RecordingModel()
        model = BudgetedSeededChatModel(
            provider,
            ExecutionBudget(max_calls=1, max_tokens=100, max_seconds=10),
            base_seed=7,
        )

        model.complete(replace(request(max_tokens=10), reasoning_effort="low"))

        self.assertEqual(provider.requests[0].reasoning_effort, "low")

    def test_future_call_reservation_blocks_before_provider_invocation(self) -> None:
        provider = RecordingModel()
        budget = ExecutionBudget(
            max_calls=2,
            max_tokens=100,
            max_seconds=10,
        )
        model = BudgetedSeededChatModel(provider, budget, base_seed=7)
        original = request()
        reserved = replace(
            original,
            metadata={
                **original.metadata,
                "budget_reserve_calls_after": 2,
            },
        )

        with self.assertRaisesRegex(BudgetExceededError, "max_calls") as raised:
            model.complete(reserved)

        self.assertEqual(provider.requests, [])
        self.assertEqual(budget.snapshot()["calls_used"], 0)
        self.assertEqual(raised.exception.diagnostics["reserve_calls_after"], 2)

    def test_inflight_call_is_cut_off_at_remaining_hard_deadline(self) -> None:
        release = threading.Event()

        class BlockingModel:
            def __init__(self) -> None:
                self.requests: list[ModelRequest] = []

            def complete(self, model_request: ModelRequest) -> ModelResponse:
                self.requests.append(model_request)
                release.wait(1.0)
                return ModelResponse("late")

        provider = BlockingModel()
        budget = ExecutionBudget(
            max_calls=1,
            max_tokens=100,
            max_seconds=0.05,
        )
        model = BudgetedSeededChatModel(provider, budget, base_seed=7)
        started = time.perf_counter()
        try:
            with self.assertRaisesRegex(
                BudgetExceededError, "max_seconds"
            ) as raised:
                model.complete(request(max_tokens=10))
        finally:
            release.set()

        self.assertLess(time.perf_counter() - started, 0.5)
        self.assertEqual(len(provider.requests), 1)
        self.assertGreater(provider.requests[0].timeout_seconds or 0.0, 0.0)
        self.assertLessEqual(provider.requests[0].timeout_seconds or 1.0, 0.05)
        self.assertTrue(
            raised.exception.diagnostics["deadline_exceeded_during_call"]
        )
        self.assertTrue(raised.exception.diagnostics["model_invoked"])
        self.assertEqual(budget.snapshot()["tokens_reserved"], 0)

    def test_n_plus_one_call_is_blocked_before_provider_invocation(self) -> None:
        diagnostics: list[tuple[str, dict[str, object]]] = []
        provider = RecordingModel()
        model = BudgetedSeededChatModel(
            provider,
            ExecutionBudget(
                max_calls=2,
                max_tokens=100,
                max_seconds=10,
                on_exhausted=lambda reason, details: diagnostics.append(
                    (reason, dict(details))
                ),
            ),
            base_seed=7,
        )

        model.complete(request(agent="first"))
        model.complete(request(agent="second"))
        with self.assertRaisesRegex(BudgetExceededError, "max_calls"):
            model.complete(request(agent="third"))

        self.assertEqual(len(provider.requests), 2)
        self.assertEqual(diagnostics[0][0], "max_calls")
        self.assertEqual(diagnostics[0][1]["calls_used"], 2)

    def test_reported_tokens_are_settled_and_next_max_tokens_is_reserved(self) -> None:
        provider = RecordingModel([6, 4])
        budget = ExecutionBudget(
            max_calls=3,
            max_tokens=10,
            max_seconds=10,
        )
        model = BudgetedSeededChatModel(provider, budget, base_seed=7)

        model.complete(request(agent="first", max_tokens=4))
        model.complete(request(agent="second", max_tokens=4))
        with self.assertRaisesRegex(BudgetExceededError, "max_tokens"):
            model.complete(request(agent="third", max_tokens=1))

        self.assertEqual(len(provider.requests), 2)
        self.assertEqual(budget.snapshot()["tokens_used"], 10)

    def test_response_that_crosses_token_limit_fails_with_reported_usage(self) -> None:
        provider = RecordingModel([11])
        budget = ExecutionBudget(
            max_calls=1,
            max_tokens=10,
            max_seconds=10,
        )
        model = BudgetedSeededChatModel(provider, budget, base_seed=7)

        with self.assertRaisesRegex(BudgetExceededError, "max_tokens") as raised:
            model.complete(request(max_tokens=1))

        self.assertEqual(len(provider.requests), 1)
        self.assertEqual(raised.exception.usage.total_tokens, 11)
        self.assertEqual(raised.exception.diagnostics["tokens_used"], 11)

    def test_wall_time_is_checked_before_provider_invocation_at_boundary(self) -> None:
        now = [0.0]
        provider = RecordingModel()
        model = BudgetedSeededChatModel(
            provider,
            ExecutionBudget(
                max_calls=2,
                max_tokens=10,
                max_seconds=1.0,
                clock=lambda: now[0],
            ),
            base_seed=7,
        )
        now[0] = 1.0

        with self.assertRaisesRegex(BudgetExceededError, "max_seconds"):
            model.complete(request())
        self.assertEqual(provider.requests, [])

    def test_seed_derivation_is_stable_and_distinguishes_inputs(self) -> None:
        def sequence(base_seed: int) -> list[int | None]:
            provider = RecordingModel()
            model = BudgetedSeededChatModel(
                provider,
                ExecutionBudget(
                    max_calls=4,
                    max_tokens=100,
                    max_seconds=10,
                ),
                base_seed=base_seed,
            )
            model.complete(request(agent="lawyer_a"))
            model.complete(
                request(agent="cape_v_candidate", candidate_id="candidate-1")
            )
            model.complete(
                request(agent="cape_v_candidate", candidate_id="candidate-2")
            )
            model.complete(request(agent="lawyer_a"))
            return [item.seed for item in provider.requests]

        first = sequence(42)
        second = sequence(42)
        changed = sequence(43)

        self.assertEqual(first, second)
        self.assertNotEqual(first, changed)
        self.assertNotEqual(first[1], first[2])
        self.assertNotEqual(first[0], first[3])

    def test_runner_budget_covers_oath_and_initial_cape(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            corpus_path = Path(directory) / "corpus.jsonl"
            write_corpus(corpus_path)
            events: list[str] = []
            runner = LGAgentPlusRunner(
                config(
                    corpus_path,
                    enabled=True,
                    oath=True,
                    cape=True,
                    initial_candidates=1,
                    risk_budget=BudgetLimits(
                        max_calls=9,
                        max_tokens=1_000_000,
                        max_rounds=1,
                        max_seconds=120.0,
                    ),
                ),
                reasoning_model=ReasoningModel(events),
                evaluation_model=EvaluationModel(events),
                verifier_model=VerifierModel(events),
            )

            with self.assertRaises(BudgetExceededError) as raised:
                runner.run(QUESTION)

        self.assertEqual(events.count("evidence_auditor"), 4)
        self.assertEqual(events.count("cape_v_candidate"), 1)
        self.assertFalse(
            any(event.startswith("cape_v_verifier_") for event in events)
        )
        trace = raised.exception.diagnostics["trace"]
        self.assertEqual(trace["routes"][-1]["route"], "budget-exhausted")
        self.assertEqual(
            trace["routes"][-1]["details"]["seed_requested"],
            trace["model_calls"][-1]["seed_requested"],
        )

    def test_injected_pipeline_reuses_retriever_with_budgeted_auditor(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            corpus_path = Path(directory) / "corpus.jsonl"
            write_corpus(corpus_path)
            events: list[str] = []
            reasoning = ReasoningModel(events)
            pipeline = EvidenceAuditPipeline(
                OathRagRetriever(
                    load_jsonl_corpus(corpus_path, allow_legacy=False),
                    config=OathRagConfig(
                        lexical_top_k=4,
                        dense_top_k=4,
                        final_top_k_per_lane=1,
                        graph_hops=0,
                    ),
                ),
                EvidenceAuditor(reasoning, MODEL_CONFIG),
            )
            runner = LGAgentPlusRunner(
                config(
                    corpus_path,
                    enabled=True,
                    oath=True,
                    cape=False,
                    risk_budget=BudgetLimits(
                        max_calls=2,
                        max_tokens=1_000_000,
                        max_rounds=1,
                        max_seconds=120.0,
                    ),
                ),
                reasoning_model=reasoning,
                evaluation_model=EvaluationModel(events),
                evidence_pipeline=pipeline,
            )

            with self.assertRaisesRegex(BudgetExceededError, "max_calls"):
                runner.run(QUESTION)

        self.assertEqual(events, ["lawyer_a", "evidence_auditor"])

    def test_runner_trace_records_requested_seed_without_guarantee_claim(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            corpus_path = Path(directory) / "unused.jsonl"
            events: list[str] = []
            result = LGAgentPlusRunner(
                config(
                    corpus_path,
                    enabled=False,
                    oath=False,
                    cape=False,
                ),
                reasoning_model=ReasoningModel(events),
                evaluation_model=EvaluationModel(events),
            ).run(QUESTION)

        seeds = [call.seed_requested for call in result.trace.model_calls]
        self.assertTrue(all(seed is not None for seed in seeds))
        self.assertEqual(len(seeds), len(set(seeds)))
        self.assertTrue(
            all(
                call.provider_seed_guarantee == "requested_not_guaranteed"
                for call in result.trace.model_calls
            )
        )
        self.assertEqual(
            result.diagnostics["reproduction"]["provider_seed_guarantee"],
            "requested_not_guaranteed",
        )


if __name__ == "__main__":
    unittest.main()
