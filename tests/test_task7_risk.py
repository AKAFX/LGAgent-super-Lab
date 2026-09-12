from __future__ import annotations

import math
import sys
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
    calculate_cape_v_metrics,
)
from lgagent.model import TokenUsage
from lgagent.risk import (
    AdaptiveRiskOrchestrator,
    BudgetCost,
    BudgetExceededError,
    BudgetGuard,
    BudgetLimits,
    CandidateSignals,
    RiskRoute,
    RiskSnapshot,
    RiskThresholds,
    RiskWeights,
    aggregate_candidates,
    calculate_evidence_metrics,
    calculate_risk_metrics,
    calculate_risk_score,
    calculate_top2_margin,
    select_risk_route,
)
from lgagent.trace import ModelCallTrace, RunTrace
from lgagent.verification import DimensionVerification, VerificationReport


def candidate(candidate_id: str, answer: str) -> ReasoningCandidate:
    return ReasoningCandidate(
        candidate_id=candidate_id,
        answer=answer,
        answer_identity=f"{answer} identity",
        presented_answer=answer,
        permutation_id="original",
        irac=CompactIRAC(
            issue="Issue",
            rule=(IRACRule("Rule", ("law:1",)),),
            application=(IRACApplication(("f1",), 0, "Inference"),),
            conclusion=f"{answer} follows",
        ),
        option_verification={
            label: CandidateOptionVerification(
                "SUPPORT" if label == answer else "REFUTE",
                0.9,
                ("law:1",),
            )
            for label in "ABCD"
        },
    )


def report(
    candidate_id: str,
    *,
    scores: tuple[float, ...] = (1.0, 1.0),
    passes: tuple[bool, ...] = (True, True),
) -> VerificationReport:
    names = ("rule", "fact")
    dimensions = {
        name: DimensionVerification(
            passed=passed,
            score=score,
            error_type=None if passed else f"{name.upper()}_FAILED",
            reason="checked",
        )
        for name, score, passed in zip(names, scores, passes)
    }
    return VerificationReport.build(candidate_id, dimensions)


def signals(
    candidate_id: str,
    answer: str,
    *,
    strength: float,
    verifier_score: float | None = None,
) -> CandidateSignals:
    item = candidate(candidate_id, answer)
    return CandidateSignals(
        candidate=item,
        evidence_coverage=strength,
        authority_score=strength,
        temporal_validity=strength,
        verifier_score=strength if verifier_score is None else verifier_score,
        permutation_consistency=strength,
    )


def call_trace(call_id: str, tokens: int) -> ModelCallTrace:
    return ModelCallTrace(
        call_id=call_id,
        agent="fake",
        model="fake",
        attempt=1,
        started_at="2026-01-01T00:00:00+00:00",
        duration_ms=1.0,
        usage=TokenUsage(total_tokens=tokens),
    )


class RiskMetricTest(unittest.TestCase):
    def test_calculates_all_normalized_metrics_from_existing_contracts(self) -> None:
        cape = calculate_cape_v_metrics(
            ["A", "A", "B", "C"],
            ["A", "A", "A", "B"],
        )
        matrix = {
            "A": {"support": ("e1",), "refute": (), "exception": (), "coverage": 1.0, "conflict": 0.0},
            "B": {"support": ("e2",), "refute": ("e3",), "exception": (), "coverage": 0.5, "conflict": 1.0},
        }
        metrics = calculate_risk_metrics(
            cape,
            evidence_matrix=matrix,
            verification_reports=(
                report("a", passes=(True, False)),
                report("b", passes=(True, True)),
            ),
        )

        self.assertAlmostEqual(metrics.top2_margin, 0.25)
        self.assertAlmostEqual(metrics.low_top2_margin, 0.75)
        self.assertAlmostEqual(metrics.evidence_coverage, 0.75)
        self.assertAlmostEqual(metrics.missing_evidence_coverage, 0.25)
        self.assertAlmostEqual(metrics.evidence_conflict, 0.5)
        self.assertAlmostEqual(metrics.verifier_disagreement, 0.5)
        self.assertAlmostEqual(metrics.permutation_instability, 0.25)
        self.assertTrue(0.0 <= metrics.normalized_answer_entropy <= 1.0)
        self.assertTrue(all(0.0 <= value <= 1.0 for value in vars(metrics).values()))

    def test_infers_coverage_and_conflict_for_oath_rag_lane_shape(self) -> None:
        coverage, conflict = calculate_evidence_metrics(
            {
                "A": {"support": ("e1",), "refute": (), "exception": ()},
                "B": {"support": ("e2",), "refute": ("e3",), "exception": ()},
                "C": {"support": (), "refute": (), "exception": ()},
                "D": {"support": (), "refute": ("e4",), "exception": ()},
            }
        )

        self.assertEqual(coverage, 0.75)
        self.assertEqual(conflict, 0.25)

    def test_missing_permutation_probe_is_not_treated_as_observed_instability(self) -> None:
        metrics = calculate_risk_metrics(calculate_cape_v_metrics(["A"]))
        self.assertEqual(metrics.permutation_instability, 0.0)
        self.assertEqual(metrics.top2_margin, 1.0)
        self.assertEqual(metrics.low_top2_margin, 0.0)

    def test_top2_margin_normalizes_non_unit_distribution(self) -> None:
        self.assertAlmostEqual(
            calculate_top2_margin({"A": 6.0 / 8.0, "B": 2.0 / 8.0}),
            0.5,
        )
        with self.assertRaisesRegex(ValueError, "positive mass"):
            calculate_top2_margin({})
        with self.assertRaisesRegex(ValueError, "between 0 and 1"):
            calculate_top2_margin({"A": 1.1})

    def test_weights_are_non_negative_normalized_and_reject_invalid_sets(self) -> None:
        weights = RiskWeights(
            normalized_answer_entropy=2.0,
            permutation_instability=1.0,
            verifier_disagreement=1.0,
            evidence_conflict=0.0,
            missing_evidence_coverage=0.0,
            low_top2_margin=0.0,
        )
        self.assertTrue(math.isclose(sum(weights.normalized.values()), 1.0))
        self.assertEqual(weights.normalized["normalized_answer_entropy"], 0.5)

        with self.assertRaisesRegex(ValueError, "non-negative"):
            RiskWeights(evidence_conflict=-0.01)
        with self.assertRaisesRegex(ValueError, "at least one"):
            RiskWeights(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        with self.assertRaisesRegex(ValueError, "finite"):
            RiskWeights(normalized_answer_entropy=math.inf)

    def test_risk_is_bounded_and_monotonic_for_each_component(self) -> None:
        low = calculate_risk_metrics(
            calculate_cape_v_metrics(["A", "A"]),
            evidence_matrix={"A": {"coverage": 1.0, "conflict": 0.0}},
        )
        high = calculate_risk_metrics(
            calculate_cape_v_metrics(["A", "B", "C", "D"], ["A", "B"]),
            evidence_matrix={"A": {"coverage": 0.0, "conflict": 1.0}},
            verification_reports=(report("x", passes=(True, False)),),
        )
        equal = RiskWeights()

        self.assertTrue(0.0 <= calculate_risk_score(low, equal) <= 1.0)
        self.assertGreater(calculate_risk_score(high, equal), calculate_risk_score(low, equal))


class RoutingTest(unittest.TestCase):
    def test_routes_all_threshold_boundaries(self) -> None:
        thresholds = RiskThresholds(low=0.3, high=0.7)
        self.assertIs(select_risk_route(0.2999, thresholds), RiskRoute.FAST)
        self.assertIs(select_risk_route(0.3, thresholds), RiskRoute.VERIFY_MORE)
        self.assertIs(select_risk_route(0.6999, thresholds), RiskRoute.VERIFY_MORE)
        self.assertIs(
            select_risk_route(0.7, thresholds),
            RiskRoute.RETRIEVE_AND_REASON,
        )
        with self.assertRaisesRegex(ValueError, "low < high"):
            RiskThresholds(low=0.5, high=0.5)


class DeterministicAggregationTest(unittest.TestCase):
    def test_minority_with_materially_stronger_evidence_can_win(self) -> None:
        result = aggregate_candidates(
            (
                signals("majority-1", "A", strength=0.3),
                signals("majority-2", "A", strength=0.3),
                signals("minority", "B", strength=1.0),
            )
        )

        self.assertEqual(result.selected.candidate_id, "minority")
        self.assertIn("minority answer selected", result.reason)
        self.assertGreater(result.ranked[0].score, result.ranked[1].score)

    def test_stable_tie_break_is_independent_of_input_order(self) -> None:
        first = signals("candidate-z", "B", strength=1.0)
        second = signals("candidate-a", "A", strength=1.0)

        forward = aggregate_candidates((first, second))
        reverse = aggregate_candidates((second, first))

        self.assertEqual(forward.selected.candidate_id, "candidate-a")
        self.assertEqual(reverse.selected.candidate_id, "candidate-a")
        self.assertIn("deterministic", forward.reason)

    def test_uses_matching_verification_report_and_rejects_mismatch(self) -> None:
        weak = CandidateSignals(
            candidate("weak", "A"),
            1.0,
            1.0,
            1.0,
            verification_report=report("weak", scores=(0.2, 0.2)),
        )
        strong = CandidateSignals(
            candidate("strong", "B"),
            1.0,
            1.0,
            1.0,
            verification_report=report("strong", scores=(0.9, 0.9)),
        )
        self.assertEqual(aggregate_candidates((weak, strong)).selected.candidate_id, "strong")

        with self.assertRaisesRegex(ValueError, "does not match"):
            CandidateSignals(
                candidate("candidate", "A"),
                1.0,
                1.0,
                1.0,
                verification_report=report("other"),
            )


class BudgetGuardTest(unittest.TestCase):
    def test_accounts_for_existing_trace_and_reservations(self) -> None:
        trace = RunTrace()
        trace.add_call(call_trace("initial", 10))
        guard = BudgetGuard(
            BudgetLimits(max_calls=3, max_tokens=30, max_rounds=2, max_seconds=10),
            trace,
        )
        guard.reserve(BudgetCost(calls=2, tokens=20))

        self.assertEqual(guard.calls_used, 3)
        self.assertEqual(guard.tokens_used, 30)
        self.assertEqual(guard.rounds_used, 1)
        self.assertEqual(
            guard.exhaustion_reason(BudgetCost(calls=1, tokens=0)),
            "max_calls",
        )

    def test_blocks_calls_tokens_rounds_and_elapsed_time_before_work(self) -> None:
        cases = (
            (
                BudgetLimits(1, 100, 2, 10),
                BudgetCost(2, 0),
                "max_calls",
            ),
            (
                BudgetLimits(2, 10, 2, 10),
                BudgetCost(1, 11),
                "max_tokens",
            ),
            (
                BudgetLimits(2, 100, 1, 10),
                (BudgetCost(1, 1), BudgetCost(0, 0)),
                "max_rounds",
            ),
        )
        for limits, costs, expected in cases:
            with self.subTest(expected=expected):
                guard = BudgetGuard(limits, RunTrace())
                sequence = costs if isinstance(costs, tuple) else (costs,)
                for cost in sequence[:-1]:
                    guard.reserve(cost)
                self.assertEqual(guard.exhaustion_reason(sequence[-1]), expected)
                with self.assertRaisesRegex(BudgetExceededError, expected):
                    guard.reserve(sequence[-1])

        now = [0.0]
        timed = BudgetGuard(
            BudgetLimits(2, 100, 2, 1.0),
            RunTrace(),
            clock=lambda: now[0],
        )
        now[0] = 1.0
        self.assertEqual(timed.exhaustion_reason(BudgetCost(0, 0)), "max_seconds")

    def test_detects_action_that_violates_declared_hard_total(self) -> None:
        trace = RunTrace()
        guard = BudgetGuard(BudgetLimits(1, 10, 1, 10), trace)
        guard.reserve(BudgetCost(1, 5))
        trace.add_call(call_trace("bad", 11))
        with self.assertRaisesRegex(BudgetExceededError, "max_tokens"):
            guard.assert_within_limits()

    def test_rejects_impossible_budgets_and_costs(self) -> None:
        with self.assertRaisesRegex(ValueError, "max_calls"):
            BudgetLimits(max_calls=0)
        with self.assertRaisesRegex(ValueError, "round cost"):
            BudgetCost(calls=0, tokens=0, rounds=0)


class AdaptiveOrchestratorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.weights = RiskWeights(
            normalized_answer_entropy=1.0,
            permutation_instability=0.0,
            verifier_disagreement=0.0,
            evidence_conflict=0.0,
            missing_evidence_coverage=0.0,
            low_top2_margin=0.0,
        )
        self.thresholds = RiskThresholds(low=0.2, high=0.8)

    @staticmethod
    def snapshot(answers: tuple[str, ...]) -> RiskSnapshot:
        return RiskSnapshot(
            candidates=tuple(
                signals(f"candidate-{index}", answer, strength=1.0)
                for index, answer in enumerate(answers)
            ),
            cape_metrics=calculate_cape_v_metrics(list(answers)),
            evidence_matrix={"A": {"coverage": 1.0, "conflict": 0.0}},
        )

    def test_fast_path_stops_without_running_an_action(self) -> None:
        called = []
        result = AdaptiveRiskOrchestrator(
            weights=self.weights,
            thresholds=self.thresholds,
        ).run(
            self.snapshot(("A", "A")),
            actions={
                RiskRoute.VERIFY_MORE: lambda *_: called.append(True),
            },
        )

        self.assertIs(result.route, RiskRoute.FAST)
        self.assertEqual(result.rounds, 0)
        self.assertEqual(called, [])
        self.assertEqual(result.trace.routes[-1].route, "fast")

    def test_medium_risk_verifies_more_then_reaggregates(self) -> None:
        calls = []

        def verify_more(snapshot, guard, trace):
            calls.append((guard.rounds_used, trace.routes[-1].route))
            return self.snapshot(("B", "B"))

        result = AdaptiveRiskOrchestrator(
            weights=self.weights,
            thresholds=self.thresholds,
            budget_limits=BudgetLimits(5, 100, 2, 10),
            action_costs={RiskRoute.VERIFY_MORE: BudgetCost(1, 10)},
        ).run(
            self.snapshot(("A", "A", "B")),
            actions={RiskRoute.VERIFY_MORE: verify_more},
        )

        self.assertEqual(calls, [(1, "verify-more")])
        self.assertIs(result.route, RiskRoute.FAST)
        self.assertEqual(result.selected.answer, "B")
        self.assertEqual(result.rounds, 1)
        self.assertEqual(
            [event.route for event in result.trace.routes],
            ["verify-more", "fast"],
        )

    def test_high_risk_uses_retrieve_and_reason(self) -> None:
        called = []

        def retrieve(snapshot, guard, trace):
            called.append(guard.as_dict())
            return self.snapshot(("C", "C"))

        result = AdaptiveRiskOrchestrator(
            weights=self.weights,
            thresholds=self.thresholds,
            budget_limits=BudgetLimits(5, 100, 2, 10),
            action_costs={
                RiskRoute.RETRIEVE_AND_REASON: BudgetCost(2, 20),
            },
        ).run(
            self.snapshot(("A", "B", "C", "D")),
            actions={RiskRoute.RETRIEVE_AND_REASON: retrieve},
        )

        self.assertEqual(len(called), 1)
        self.assertEqual(called[0]["calls_used"], 2)
        self.assertEqual(result.selected.answer, "C")
        self.assertEqual(result.trace.routes[0].route, "retrieve-and-reason")

    def test_budget_exhaustion_returns_best_verified_candidate_and_flag(self) -> None:
        initial = self.snapshot(("A", "A", "B"))
        result = AdaptiveRiskOrchestrator(
            weights=self.weights,
            thresholds=self.thresholds,
            budget_limits=BudgetLimits(1, 10, 1, 10),
            action_costs={RiskRoute.VERIFY_MORE: BudgetCost(2, 10)},
        ).run(
            initial,
            actions={RiskRoute.VERIFY_MORE: lambda *_: self.fail("must not run")},
        )

        self.assertTrue(result.budget_exhausted)
        self.assertEqual(result.budget_exhausted_reason, "max_calls")
        self.assertEqual(result.selected.candidate_id, "candidate-0")
        self.assertEqual(result.trace.routes[-1].route, "budget-exhausted")
        self.assertEqual(result.trace.routes[-1].details["requested_route"], "verify-more")

    def test_elapsed_budget_during_action_returns_new_best_candidate(self) -> None:
        now = [0.0]

        def verify_more(snapshot, guard, trace):
            now[0] = 1.1
            return self.snapshot(("D", "D"))

        result = AdaptiveRiskOrchestrator(
            weights=self.weights,
            thresholds=self.thresholds,
            budget_limits=BudgetLimits(5, 100, 2, 1.0),
            action_costs={RiskRoute.VERIFY_MORE: BudgetCost(1, 10)},
            clock=lambda: now[0],
        ).run(
            self.snapshot(("A", "A", "B")),
            actions={RiskRoute.VERIFY_MORE: verify_more},
        )

        self.assertTrue(result.budget_exhausted)
        self.assertEqual(result.budget_exhausted_reason, "max_seconds")
        self.assertEqual(result.selected.answer, "D")
        self.assertEqual(result.trace.routes[-1].route, "budget-exhausted")

    def test_budget_error_inside_optional_action_returns_existing_candidate(self) -> None:
        initial = self.snapshot(("A", "A", "B"))

        def verify_more(snapshot, guard, trace):
            raise BudgetExceededError("max_tokens")

        result = AdaptiveRiskOrchestrator(
            weights=self.weights,
            thresholds=self.thresholds,
            budget_limits=BudgetLimits(5, 100, 2, 10),
            action_costs={RiskRoute.VERIFY_MORE: BudgetCost(1, 10)},
        ).run(
            initial,
            actions={RiskRoute.VERIFY_MORE: verify_more},
        )

        self.assertTrue(result.budget_exhausted)
        self.assertEqual(result.budget_exhausted_reason, "max_tokens")
        self.assertEqual(result.selected.candidate_id, "candidate-0")
        self.assertEqual(result.rounds, 1)
        self.assertEqual(result.trace.routes[-1].route, "budget-exhausted")
        self.assertTrue(
            result.trace.routes[-1].details["fallback_to_existing_candidate"]
        )


if __name__ == "__main__":
    unittest.main()
