"""Risk scoring, deterministic aggregation, and bounded adaptive routing."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from time import perf_counter
from typing import Callable, Iterable, Mapping, Sequence

from .cape_v import OPTION_LABELS, CapeVMetrics, ReasoningCandidate
from .model import BudgetExceededError
from .oath_rag import EvidenceMatrix
from .protocol import normalize_option_label
from .trace import RunTrace
from .verification import VerificationReport

RISK_COMPONENTS = (
    "normalized_answer_entropy",
    "permutation_instability",
    "verifier_disagreement",
    "evidence_conflict",
    "missing_evidence_coverage",
    "low_top2_margin",
)


def _unit_interval(value: float, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number between 0 and 1")
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError(f"{name} must be a finite number between 0 and 1")
    return number


def _mean(values: Iterable[float], default: float = 0.0) -> float:
    collected = tuple(values)
    return sum(collected) / len(collected) if collected else default


@dataclass(frozen=True)
class RiskWeights:
    normalized_answer_entropy: float = 1.0
    permutation_instability: float = 1.0
    verifier_disagreement: float = 1.0
    evidence_conflict: float = 1.0
    missing_evidence_coverage: float = 1.0
    low_top2_margin: float = 1.0
    normalized: Mapping[str, float] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        raw = {
            name: float(getattr(self, name))
            for name in RISK_COMPONENTS
        }
        if any(not math.isfinite(value) or value < 0.0 for value in raw.values()):
            raise ValueError("risk weights must be finite and non-negative")
        total = sum(raw.values())
        if total <= 0.0:
            raise ValueError("at least one risk weight must be positive")
        object.__setattr__(
            self,
            "normalized",
            {name: value / total for name, value in raw.items()},
        )


@dataclass(frozen=True)
class RiskMetrics:
    normalized_answer_entropy: float
    top2_margin: float
    low_top2_margin: float
    evidence_coverage: float
    missing_evidence_coverage: float
    evidence_conflict: float
    verifier_disagreement: float
    permutation_instability: float

    def __post_init__(self) -> None:
        for name, value in vars(self).items():
            _unit_interval(value, name)

    def risk_components(self) -> dict[str, float]:
        return {name: getattr(self, name) for name in RISK_COMPONENTS}


def calculate_top2_margin(distribution: Mapping[str, float]) -> float:
    """Return the normalized probability gap between the two leading answers."""
    probabilities = [
        _unit_interval(distribution.get(label, 0.0), f"distribution.{label}")
        for label in OPTION_LABELS
    ]
    total = sum(probabilities)
    if total <= 0.0:
        raise ValueError("answer distribution must have positive mass")
    normalized = sorted((value / total for value in probabilities), reverse=True)
    return _unit_interval(normalized[0] - normalized[1], "top2_margin")


def calculate_evidence_metrics(
    evidence_matrix: EvidenceMatrix | Mapping[str, Mapping[str, object]] | None,
) -> tuple[float, float]:
    """Calculate coverage/conflict, preferring explicit audited matrix values."""
    if not evidence_matrix:
        return 0.0, 0.0

    coverage_values: list[float] = []
    conflict_values: list[float] = []
    for option, lanes in sorted(evidence_matrix.items()):
        if not isinstance(lanes, Mapping):
            raise ValueError(f"evidence matrix option {option} must be a mapping")

        explicit_coverage = lanes.get("coverage")
        if explicit_coverage is None:
            occupied = any(bool(lanes.get(lane, ())) for lane in ("support", "refute", "exception"))
            coverage_values.append(1.0 if occupied else 0.0)
        else:
            coverage_values.append(
                _unit_interval(float(explicit_coverage), f"{option}.coverage")
            )

        explicit_conflict = lanes.get("conflict")
        if explicit_conflict is None:
            conflict_values.append(
                1.0 if lanes.get("support") and lanes.get("refute") else 0.0
            )
        else:
            conflict_values.append(
                _unit_interval(float(explicit_conflict), f"{option}.conflict")
            )
    return _mean(coverage_values), _mean(conflict_values)


def calculate_verifier_disagreement(
    reports: Iterable[VerificationReport],
) -> float:
    return _unit_interval(
        _mean(report.disagreement for report in reports),
        "verifier_disagreement",
    )


def calculate_risk_metrics(
    cape_metrics: CapeVMetrics,
    *,
    evidence_matrix: EvidenceMatrix | Mapping[str, Mapping[str, object]] | None = None,
    verification_reports: Iterable[VerificationReport] = (),
) -> RiskMetrics:
    entropy = _unit_interval(
        cape_metrics.normalized_answer_entropy,
        "normalized_answer_entropy",
    )
    margin = calculate_top2_margin(cape_metrics.answer_distribution)
    coverage, conflict = calculate_evidence_metrics(evidence_matrix)
    disagreement = calculate_verifier_disagreement(verification_reports)
    consistency = (
        1.0
        if cape_metrics.permutation_consistency is None
        else _unit_interval(
            cape_metrics.permutation_consistency,
            "permutation_consistency",
        )
    )
    return RiskMetrics(
        normalized_answer_entropy=entropy,
        top2_margin=margin,
        low_top2_margin=1.0 - margin,
        evidence_coverage=coverage,
        missing_evidence_coverage=1.0 - coverage,
        evidence_conflict=conflict,
        verifier_disagreement=disagreement,
        permutation_instability=1.0 - consistency,
    )


def calculate_risk_score(
    metrics: RiskMetrics,
    weights: RiskWeights | None = None,
) -> float:
    resolved = weights or RiskWeights()
    score = sum(
        resolved.normalized[name] * value
        for name, value in metrics.risk_components().items()
    )
    return _unit_interval(score, "risk")


class RiskRoute(str, Enum):
    FAST = "fast"
    VERIFY_MORE = "verify-more"
    RETRIEVE_AND_REASON = "retrieve-and-reason"


@dataclass(frozen=True)
class RiskThresholds:
    low: float = 0.35
    high: float = 0.65

    def __post_init__(self) -> None:
        low = _unit_interval(self.low, "low threshold")
        high = _unit_interval(self.high, "high threshold")
        if low >= high:
            raise ValueError("risk thresholds must satisfy low < high")


def select_risk_route(
    risk: float,
    thresholds: RiskThresholds | None = None,
) -> RiskRoute:
    value = _unit_interval(risk, "risk")
    resolved = thresholds or RiskThresholds()
    if value < resolved.low:
        return RiskRoute.FAST
    if value < resolved.high:
        return RiskRoute.VERIFY_MORE
    return RiskRoute.RETRIEVE_AND_REASON


@dataclass(frozen=True)
class CandidateSignals:
    candidate: ReasoningCandidate
    evidence_coverage: float
    authority_score: float
    temporal_validity: float
    verifier_score: float | None = None
    verification_report: VerificationReport | None = None
    permutation_consistency: float = 1.0

    def __post_init__(self) -> None:
        for name in (
            "evidence_coverage",
            "authority_score",
            "temporal_validity",
            "permutation_consistency",
        ):
            _unit_interval(getattr(self, name), name)
        if self.verifier_score is not None:
            _unit_interval(self.verifier_score, "verifier_score")
        if (
            self.verification_report is not None
            and self.verification_report.candidate_id != self.candidate.candidate_id
        ):
            raise ValueError("verification report candidate_id does not match candidate")

    @property
    def resolved_verifier_score(self) -> float:
        if self.verifier_score is not None:
            return self.verifier_score
        if self.verification_report is not None:
            return _unit_interval(
                self.verification_report.overall_score,
                "verification_report.overall_score",
            )
        return 0.0


@dataclass(frozen=True)
class CandidateScore:
    candidate: ReasoningCandidate
    score: float
    vote_strength: float
    evidence_coverage: float
    authority_score: float
    temporal_validity: float
    verifier_score: float
    permutation_consistency: float


@dataclass(frozen=True)
class AggregationResult:
    selected: ReasoningCandidate
    ranked: tuple[CandidateScore, ...]
    reason: str


def aggregate_candidates(
    candidates: Sequence[CandidateSignals],
) -> AggregationResult:
    """Apply the specified multiplicative evidence score and stable tie-break."""
    if not candidates:
        raise ValueError("at least one candidate is required")
    ids = [item.candidate.candidate_id for item in candidates]
    if len(ids) != len(set(ids)):
        raise ValueError("candidate_id values must be unique")

    answers_by_id = {
        item.candidate.candidate_id: normalize_option_label(
            item.candidate.answer,
            "candidate answer",
        )
        for item in candidates
    }
    answers = [answers_by_id[item.candidate.candidate_id] for item in candidates]
    counts = Counter(answers)
    total = len(candidates)
    scored = []
    for item, answer in zip(candidates, answers):
        vote_strength = counts[answer] / total
        verifier_score = item.resolved_verifier_score
        score = math.prod(
            (
                vote_strength,
                item.evidence_coverage,
                item.authority_score,
                item.temporal_validity,
                verifier_score,
                item.permutation_consistency,
            )
        )
        scored.append(
            CandidateScore(
                candidate=item.candidate,
                score=score,
                vote_strength=vote_strength,
                evidence_coverage=item.evidence_coverage,
                authority_score=item.authority_score,
                temporal_validity=item.temporal_validity,
                verifier_score=verifier_score,
                permutation_consistency=item.permutation_consistency,
            )
        )

    option_order = {label: index for index, label in enumerate(OPTION_LABELS)}
    ranked = tuple(
        sorted(
            scored,
            key=lambda item: (
                -item.score,
                -item.verifier_score,
                -item.evidence_coverage,
                -item.authority_score,
                -item.temporal_validity,
                -item.permutation_consistency,
                option_order[answers_by_id[item.candidate.candidate_id]],
                item.candidate.candidate_id,
            ),
        )
    )
    winner = ranked[0]
    majority_answer = min(
        (
            answer
            for answer, count in counts.items()
            if count == max(counts.values())
        ),
        key=option_order.__getitem__,
    )
    if answers_by_id[winner.candidate.candidate_id] != majority_answer:
        reason = (
            "minority answer selected because its evidence-weighted verified "
            "candidate score exceeded the majority candidates"
        )
    elif len(ranked) > 1 and math.isclose(
        winner.score, ranked[1].score, rel_tol=0.0, abs_tol=1e-15
    ):
        reason = "equal scores resolved by deterministic signal and identity tie-break"
    else:
        reason = "highest evidence-weighted verified candidate score"
    return AggregationResult(winner.candidate, ranked, reason)


@dataclass(frozen=True)
class BudgetLimits:
    max_calls: int = 32
    max_tokens: int = 32768
    max_rounds: int = 2
    max_seconds: float = 120.0

    def __post_init__(self) -> None:
        for name in ("max_calls", "max_tokens", "max_rounds"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not math.isfinite(self.max_seconds) or self.max_seconds <= 0.0:
            raise ValueError("max_seconds must be finite and positive")


@dataclass(frozen=True)
class BudgetCost:
    calls: int
    tokens: int
    rounds: int = 1

    def __post_init__(self) -> None:
        for name in ("calls", "tokens", "rounds"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} cost must be a non-negative integer")
        if self.rounds == 0:
            raise ValueError("round cost must be positive")


class BudgetGuard:
    """Reserve worst-case action cost before executing any adaptive work."""

    def __init__(
        self,
        limits: BudgetLimits,
        trace: RunTrace,
        *,
        clock: Callable[[], float] = perf_counter,
    ) -> None:
        self.limits = limits
        self.trace = trace
        self._clock = clock
        self._started = clock()
        self._initial_calls = len(trace.model_calls)
        self._initial_tokens = trace.total_tokens
        self._reserved_calls = self._initial_calls
        self._reserved_tokens = self._initial_tokens
        self._rounds = 0

    @property
    def elapsed_seconds(self) -> float:
        return max(0.0, self._clock() - self._started)

    @property
    def calls_used(self) -> int:
        return max(self._reserved_calls, len(self.trace.model_calls))

    @property
    def tokens_used(self) -> int:
        return max(self._reserved_tokens, self.trace.total_tokens)

    @property
    def rounds_used(self) -> int:
        return self._rounds

    def exhaustion_reason(self, cost: BudgetCost) -> str | None:
        if self.elapsed_seconds >= self.limits.max_seconds:
            return "max_seconds"
        if self.calls_used + cost.calls > self.limits.max_calls:
            return "max_calls"
        if self.tokens_used + cost.tokens > self.limits.max_tokens:
            return "max_tokens"
        if self.rounds_used + cost.rounds > self.limits.max_rounds:
            return "max_rounds"
        return None

    def reserve(self, cost: BudgetCost) -> None:
        reason = self.exhaustion_reason(cost)
        if reason is not None:
            raise BudgetExceededError(f"adaptive budget exhausted: {reason}")
        self._reserved_calls += cost.calls
        self._reserved_tokens += cost.tokens
        self._rounds += cost.rounds

    def assert_within_limits(self) -> None:
        violation = self.violation_reason()
        if violation is not None:
            raise BudgetExceededError(
                f"adaptive action exceeded hard budget: {violation}"
            )

    def violation_reason(self) -> str | None:
        if len(self.trace.model_calls) > self.limits.max_calls:
            return "max_calls"
        if self.trace.total_tokens > self.limits.max_tokens:
            return "max_tokens"
        if self.rounds_used > self.limits.max_rounds:
            return "max_rounds"
        if self.elapsed_seconds > self.limits.max_seconds:
            return "max_seconds"
        return None

    def as_dict(self) -> dict[str, float | int]:
        return {
            "calls_used": self.calls_used,
            "tokens_used": self.tokens_used,
            "rounds_used": self.rounds_used,
            "elapsed_seconds": self.elapsed_seconds,
            "max_calls": self.limits.max_calls,
            "max_tokens": self.limits.max_tokens,
            "max_rounds": self.limits.max_rounds,
            "max_seconds": self.limits.max_seconds,
        }


@dataclass(frozen=True)
class RiskSnapshot:
    candidates: tuple[CandidateSignals, ...]
    cape_metrics: CapeVMetrics
    evidence_matrix: EvidenceMatrix | Mapping[str, Mapping[str, object]] | None = None

    @property
    def verification_reports(self) -> tuple[VerificationReport, ...]:
        return tuple(
            item.verification_report
            for item in self.candidates
            if item.verification_report is not None
        )


AdaptiveAction = Callable[[RiskSnapshot, BudgetGuard, RunTrace], RiskSnapshot]


@dataclass(frozen=True)
class AdaptiveResult:
    selected: ReasoningCandidate
    aggregation: AggregationResult
    candidates: tuple[CandidateSignals, ...]
    metrics: RiskMetrics
    risk: float
    route: RiskRoute
    rounds: int
    budget_exhausted: bool
    budget_exhausted_reason: str | None
    trace: RunTrace


class AdaptiveRiskOrchestrator:
    """Route measured risk and repeat only explicitly budgeted adaptive actions."""

    def __init__(
        self,
        *,
        weights: RiskWeights | None = None,
        thresholds: RiskThresholds | None = None,
        budget_limits: BudgetLimits | None = None,
        action_costs: Mapping[RiskRoute, BudgetCost] | None = None,
        clock: Callable[[], float] = perf_counter,
    ) -> None:
        self.weights = weights or RiskWeights()
        self.thresholds = thresholds or RiskThresholds()
        self.budget_limits = budget_limits or BudgetLimits()
        self.action_costs = dict(
            action_costs
            or {
                RiskRoute.VERIFY_MORE: BudgetCost(calls=5, tokens=4096),
                RiskRoute.RETRIEVE_AND_REASON: BudgetCost(calls=7, tokens=8192),
            }
        )
        if RiskRoute.FAST in self.action_costs:
            raise ValueError("fast route must not have an adaptive action cost")
        self.clock = clock

    def run(
        self,
        initial: RiskSnapshot,
        *,
        actions: Mapping[RiskRoute, AdaptiveAction] | None = None,
        trace: RunTrace | None = None,
        fixed_compute: bool = False,
    ) -> AdaptiveResult:
        run_trace = trace or RunTrace()
        guard = BudgetGuard(self.budget_limits, run_trace, clock=self.clock)
        snapshot = initial
        available = dict(actions or {})

        while True:
            aggregation = aggregate_candidates(snapshot.candidates)
            metrics = calculate_risk_metrics(
                snapshot.cape_metrics,
                evidence_matrix=snapshot.evidence_matrix,
                verification_reports=snapshot.verification_reports,
            )
            risk = calculate_risk_score(metrics, self.weights)
            route = (
                RiskRoute.RETRIEVE_AND_REASON
                if fixed_compute and guard.rounds_used < self.budget_limits.max_rounds
                else select_risk_route(risk, self.thresholds)
            )
            run_trace.add_route(
                route.value,
                f"normalized risk {risk:.6f}",
                {
                    "risk": risk,
                    "metrics": metrics.risk_components(),
                    "selected_candidate_id": aggregation.selected.candidate_id,
                    "aggregation_reason": aggregation.reason,
                    "budget": guard.as_dict(),
                },
            )
            initial_violation = guard.violation_reason()
            if initial_violation is not None:
                run_trace.add_route(
                    "budget-exhausted",
                    f"hard budget exhausted: {initial_violation}",
                    {
                        "requested_route": route.value,
                        "selected_candidate_id": aggregation.selected.candidate_id,
                        "budget": guard.as_dict(),
                    },
                )
                return AdaptiveResult(
                    selected=aggregation.selected,
                    aggregation=aggregation,
                    candidates=snapshot.candidates,
                    metrics=metrics,
                    risk=risk,
                    route=route,
                    rounds=guard.rounds_used,
                    budget_exhausted=True,
                    budget_exhausted_reason=initial_violation,
                    trace=run_trace,
                )
            if route is RiskRoute.FAST or route not in available:
                return AdaptiveResult(
                    selected=aggregation.selected,
                    aggregation=aggregation,
                    candidates=snapshot.candidates,
                    metrics=metrics,
                    risk=risk,
                    route=route,
                    rounds=guard.rounds_used,
                    budget_exhausted=False,
                    budget_exhausted_reason=None,
                    trace=run_trace,
                )

            cost = self.action_costs.get(route)
            if cost is None:
                raise ValueError(f"missing action cost for route {route.value}")
            exhausted = guard.exhaustion_reason(cost)
            if exhausted is not None:
                run_trace.add_route(
                    "budget-exhausted",
                    f"adaptive budget exhausted: {exhausted}",
                    {
                        "requested_route": route.value,
                        "selected_candidate_id": aggregation.selected.candidate_id,
                        "budget": guard.as_dict(),
                    },
                )
                return AdaptiveResult(
                    selected=aggregation.selected,
                    aggregation=aggregation,
                    candidates=snapshot.candidates,
                    metrics=metrics,
                    risk=risk,
                    route=route,
                    rounds=guard.rounds_used,
                    budget_exhausted=True,
                    budget_exhausted_reason=exhausted,
                    trace=run_trace,
                )

            guard.reserve(cost)
            try:
                next_snapshot = available[route](snapshot, guard, run_trace)
            except BudgetExceededError as exc:
                run_trace.add_route(
                    "budget-exhausted",
                    f"adaptive action exhausted hard budget: {exc.reason}",
                    {
                        "requested_route": route.value,
                        "selected_candidate_id": aggregation.selected.candidate_id,
                        "fallback_to_existing_candidate": True,
                        "budget": guard.as_dict(),
                    },
                )
                return AdaptiveResult(
                    selected=aggregation.selected,
                    aggregation=aggregation,
                    candidates=snapshot.candidates,
                    metrics=metrics,
                    risk=risk,
                    route=route,
                    rounds=guard.rounds_used,
                    budget_exhausted=True,
                    budget_exhausted_reason=exc.reason,
                    trace=run_trace,
                )
            snapshot = next_snapshot
            if not isinstance(snapshot, RiskSnapshot):
                raise TypeError("adaptive action must return RiskSnapshot")
            violation = guard.violation_reason()
            if violation is not None:
                aggregation = aggregate_candidates(snapshot.candidates)
                metrics = calculate_risk_metrics(
                    snapshot.cape_metrics,
                    evidence_matrix=snapshot.evidence_matrix,
                    verification_reports=snapshot.verification_reports,
                )
                risk = calculate_risk_score(metrics, self.weights)
                final_route = select_risk_route(risk, self.thresholds)
                run_trace.add_route(
                    "budget-exhausted",
                    f"adaptive budget exhausted: {violation}",
                    {
                        "requested_route": route.value,
                        "selected_candidate_id": aggregation.selected.candidate_id,
                        "budget": guard.as_dict(),
                    },
                )
                return AdaptiveResult(
                    selected=aggregation.selected,
                    aggregation=aggregation,
                    candidates=snapshot.candidates,
                    metrics=metrics,
                    risk=risk,
                    route=final_route,
                    rounds=guard.rounds_used,
                    budget_exhausted=True,
                    budget_exhausted_reason=violation,
                    trace=run_trace,
                )
