"""Offline evaluation, significance tests, and resumable experiment records."""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .serialization import to_jsonable


def _finite(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_hash(value: Any) -> str:
    encoded = json.dumps(
        to_jsonable(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class ExperimentMetadata:
    model_id: str
    endpoint_class: str
    prompt_version: str
    config_hash: str
    dataset_hash: str
    random_seed: int
    software_version: str

    def __post_init__(self) -> None:
        for name in (
            "model_id",
            "endpoint_class",
            "prompt_version",
            "config_hash",
            "dataset_hash",
            "software_version",
        ):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"{name} cannot be empty")
        if not isinstance(self.random_seed, int) or isinstance(self.random_seed, bool):
            raise ValueError("random_seed must be an integer")

    @property
    def experiment_id(self) -> str:
        return stable_hash(asdict(self))[:24]

    def as_dict(self) -> dict[str, Any]:
        return {**asdict(self), "experiment_id": self.experiment_id}


def evidence_recall_at_k(
    retrieved_ids: Sequence[str],
    relevant_ids: Iterable[str],
    k: int,
) -> float | None:
    if k <= 0:
        raise ValueError("k must be positive")
    relevant = set(relevant_ids)
    if not relevant:
        return None
    return len(set(retrieved_ids[:k]) & relevant) / len(relevant)


def reciprocal_rank(
    retrieved_ids: Sequence[str],
    relevant_ids: Iterable[str],
) -> float | None:
    relevant = set(relevant_ids)
    if not relevant:
        return None
    for rank, evidence_id in enumerate(retrieved_ids, start=1):
        if evidence_id in relevant:
            return 1.0 / rank
    return 0.0


def mean_reciprocal_rank(
    rankings: Iterable[tuple[Sequence[str], Iterable[str]]],
) -> float | None:
    scores = [
        score
        for retrieved, relevant in rankings
        if (score := reciprocal_rank(retrieved, relevant)) is not None
    ]
    return _mean(scores)


def ndcg_at_k(
    retrieved_ids: Sequence[str],
    relevance: Mapping[str, float] | Iterable[str],
    k: int,
) -> float | None:
    if k <= 0:
        raise ValueError("k must be positive")
    gains = (
        {str(key): _finite(value, "relevance") for key, value in relevance.items()}
        if isinstance(relevance, Mapping)
        else {str(key): 1.0 for key in relevance}
    )
    gains = {key: value for key, value in gains.items() if value > 0.0}
    if not gains:
        return None

    def dcg(values: Sequence[float]) -> float:
        return sum(
            (2.0**gain - 1.0) / math.log2(rank + 1)
            for rank, gain in enumerate(values, start=1)
        )

    seen: set[str] = set()
    ranked_gains = []
    for item in retrieved_ids[:k]:
        ranked_gains.append(0.0 if item in seen else gains.get(item, 0.0))
        seen.add(item)
    actual = dcg(ranked_gains)
    ideal = dcg(sorted(gains.values(), reverse=True)[:k])
    return actual / ideal if ideal else 0.0


@dataclass(frozen=True)
class CitationMetrics:
    precision: float | None
    recall: float | None
    f1: float | None
    predicted_count: int
    relevant_count: int
    true_positive_count: int


def citation_f1(
    cited_ids: Iterable[str],
    relevant_ids: Iterable[str],
) -> CitationMetrics:
    cited = set(cited_ids)
    relevant = set(relevant_ids)
    true_positives = len(cited & relevant)
    precision = true_positives / len(cited) if cited else None
    recall = true_positives / len(relevant) if relevant else None
    if not cited and not relevant:
        f1 = None
    elif not cited or not relevant:
        f1 = 0.0
    else:
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
    return CitationMetrics(
        precision=precision,
        recall=recall,
        f1=f1,
        predicted_count=len(cited),
        relevant_count=len(relevant),
        true_positive_count=true_positives,
    )


def permutation_consistency(mapped_answers: Sequence[str]) -> float | None:
    if not mapped_answers:
        return None
    counts: dict[str, int] = {}
    for answer in mapped_answers:
        counts[answer] = counts.get(answer, 0) + 1
    return max(counts.values()) / len(mapped_answers)


@dataclass(frozen=True)
class CounterfactualEvaluation:
    sensitivity: float | None
    specificity: float | None
    should_change_total: int
    should_change_correct: int
    should_preserve_total: int
    should_preserve_correct: int


def counterfactual_sensitivity_specificity(
    expected_change: Sequence[bool],
    observed_change: Sequence[bool],
) -> CounterfactualEvaluation:
    if len(expected_change) != len(observed_change):
        raise ValueError("counterfactual arrays must have equal length")
    change_total = sum(expected_change)
    preserve_total = len(expected_change) - change_total
    change_correct = sum(
        expected and observed
        for expected, observed in zip(expected_change, observed_change)
    )
    preserve_correct = sum(
        not expected and not observed
        for expected, observed in zip(expected_change, observed_change)
    )
    return CounterfactualEvaluation(
        sensitivity=change_correct / change_total if change_total else None,
        specificity=preserve_correct / preserve_total if preserve_total else None,
        should_change_total=change_total,
        should_change_correct=change_correct,
        should_preserve_total=preserve_total,
        should_preserve_correct=preserve_correct,
    )


def expected_calibration_error(
    confidences: Sequence[float],
    correctness: Sequence[bool | int],
    *,
    bins: int = 10,
) -> float | None:
    if len(confidences) != len(correctness):
        raise ValueError("confidence and correctness arrays must have equal length")
    if bins <= 0:
        raise ValueError("bins must be positive")
    if not confidences:
        return None
    totals = [0] * bins
    confidence_sums = [0.0] * bins
    correct_sums = [0] * bins
    for raw_confidence, raw_correct in zip(confidences, correctness):
        confidence = _finite(raw_confidence, "confidence")
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        index = min(int(confidence * bins), bins - 1)
        totals[index] += 1
        confidence_sums[index] += confidence
        correct_sums[index] += int(bool(raw_correct))
    count = len(confidences)
    return sum(
        totals[index]
        / count
        * abs(
            correct_sums[index] / totals[index]
            - confidence_sums[index] / totals[index]
        )
        for index in range(bins)
        if totals[index]
    )


def brier_score(
    confidences: Sequence[float],
    correctness: Sequence[bool | int],
) -> float | None:
    if len(confidences) != len(correctness):
        raise ValueError("confidence and correctness arrays must have equal length")
    if not confidences:
        return None
    scores = []
    for confidence, correct in zip(confidences, correctness):
        probability = _finite(confidence, "confidence")
        if not 0.0 <= probability <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        scores.append((probability - int(bool(correct))) ** 2)
    return sum(scores) / len(scores)


@dataclass(frozen=True)
class RiskCoveragePoint:
    coverage: float
    accuracy: float
    risk_threshold: float


@dataclass(frozen=True)
class RiskCoverageResult:
    points: tuple[RiskCoveragePoint, ...]
    aurc: float | None


def risk_coverage_curve(
    risks: Sequence[float],
    correctness: Sequence[bool | int],
) -> RiskCoverageResult:
    if len(risks) != len(correctness):
        raise ValueError("risk and correctness arrays must have equal length")
    if not risks:
        return RiskCoverageResult((), None)
    pairs = []
    for index, (raw_risk, correct) in enumerate(zip(risks, correctness)):
        risk = _finite(raw_risk, "risk")
        if not 0.0 <= risk <= 1.0:
            raise ValueError("risk must be between 0 and 1")
        pairs.append((risk, index, int(bool(correct))))
    pairs.sort()
    points = []
    correct_so_far = 0
    for accepted, (risk, _, correct) in enumerate(pairs, start=1):
        correct_so_far += correct
        points.append(
            RiskCoveragePoint(
                coverage=accepted / len(pairs),
                accuracy=correct_so_far / accepted,
                risk_threshold=risk,
            )
        )
    return RiskCoverageResult(
        tuple(points),
        sum(1.0 - point.accuracy for point in points) / len(points),
    )


@dataclass(frozen=True)
class UsageRecord:
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    cost: float = 0.0
    total_tokens: int = 0


def _usage_record(record: UsageRecord | Mapping[str, Any]) -> UsageRecord:
    normalized = (
        record
        if isinstance(record, UsageRecord)
        else UsageRecord(
            calls=int(record.get("calls", 0) or 0),
            prompt_tokens=int(record.get("prompt_tokens", 0) or 0),
            completion_tokens=int(record.get("completion_tokens", 0) or 0),
            latency_ms=float(record.get("latency_ms", 0.0) or 0.0),
            cost=float(record.get("cost", 0.0) or 0.0),
            total_tokens=int(record.get("total_tokens", 0) or 0),
        )
    )
    for name in ("calls", "prompt_tokens", "completion_tokens", "total_tokens"):
        value = getattr(normalized, name)
        if value < 0:
            raise ValueError(f"{name} cannot be negative")
    for name in ("latency_ms", "cost"):
        value = _finite(getattr(normalized, name), name)
        if value < 0.0:
            raise ValueError(f"{name} cannot be negative")
    return normalized


def calculate_usage_cost(
    prompt_tokens: int,
    completion_tokens: int,
    *,
    input_price_per_million: float = 0.0,
    output_price_per_million: float = 0.0,
) -> float:
    if prompt_tokens < 0 or completion_tokens < 0:
        raise ValueError("token counts cannot be negative")
    if input_price_per_million < 0 or output_price_per_million < 0:
        raise ValueError("token prices cannot be negative")
    return (
        prompt_tokens * input_price_per_million
        + completion_tokens * output_price_per_million
    ) / 1_000_000


def summarize_usage(records: Iterable[UsageRecord | Mapping[str, Any]]) -> dict[str, Any]:
    normalized = [_usage_record(record) for record in records]
    total_tokens = sum(
        item.total_tokens or item.prompt_tokens + item.completion_tokens
        for item in normalized
    )
    return {
        "examples": len(normalized),
        "calls": sum(item.calls for item in normalized),
        "prompt_tokens": sum(item.prompt_tokens for item in normalized),
        "completion_tokens": sum(item.completion_tokens for item in normalized),
        "total_tokens": total_tokens,
        "cost": sum(item.cost for item in normalized),
        "latency_ms": sum(item.latency_ms for item in normalized),
        "mean_latency_ms": (
            sum(item.latency_ms for item in normalized) / len(normalized)
            if normalized
            else None
        ),
    }


@dataclass(frozen=True)
class BootstrapResult:
    mean_difference: float
    confidence_interval: tuple[float, float]
    confidence_level: float
    p_value: float
    iterations: int


def paired_bootstrap(
    baseline: Sequence[float | bool | int],
    treatment: Sequence[float | bool | int],
    *,
    iterations: int = 10_000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> BootstrapResult:
    if len(baseline) != len(treatment) or not baseline:
        raise ValueError("paired samples must be non-empty and have equal length")
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be between 0 and 1")
    differences = [
        _finite(right, "treatment") - _finite(left, "baseline")
        for left, right in zip(baseline, treatment)
    ]
    rng = random.Random(seed)
    count = len(differences)
    samples = sorted(
        sum(differences[rng.randrange(count)] for _ in range(count)) / count
        for _ in range(iterations)
    )
    alpha = (1.0 - confidence_level) / 2.0

    def percentile(probability: float) -> float:
        position = probability * (iterations - 1)
        lower = int(math.floor(position))
        upper = int(math.ceil(position))
        if lower == upper:
            return samples[lower]
        fraction = position - lower
        return samples[lower] * (1.0 - fraction) + samples[upper] * fraction

    non_positive = sum(value <= 0.0 for value in samples)
    non_negative = sum(value >= 0.0 for value in samples)
    p_value = min(
        1.0,
        2.0 * (min(non_positive, non_negative) + 1) / (iterations + 1),
    )
    return BootstrapResult(
        mean_difference=sum(differences) / count,
        confidence_interval=(percentile(alpha), percentile(1.0 - alpha)),
        confidence_level=confidence_level,
        p_value=p_value,
        iterations=iterations,
    )


@dataclass(frozen=True)
class McNemarResult:
    baseline_only_correct: int
    treatment_only_correct: int
    discordant_pairs: int
    statistic: float
    p_value: float
    method: str = "exact-binomial"


def mcnemar_test(
    baseline_correct: Sequence[bool | int],
    treatment_correct: Sequence[bool | int],
) -> McNemarResult:
    if len(baseline_correct) != len(treatment_correct) or not baseline_correct:
        raise ValueError("paired outcomes must be non-empty and have equal length")
    baseline_only = sum(
        bool(left) and not bool(right)
        for left, right in zip(baseline_correct, treatment_correct)
    )
    treatment_only = sum(
        not bool(left) and bool(right)
        for left, right in zip(baseline_correct, treatment_correct)
    )
    discordant = baseline_only + treatment_only
    if not discordant:
        return McNemarResult(0, 0, 0, 0.0, 1.0)
    tail = min(baseline_only, treatment_only)
    probability = min(
        1.0,
        2.0
        * sum(math.comb(discordant, index) for index in range(tail + 1))
        / (2**discordant),
    )
    statistic = (max(0, abs(baseline_only - treatment_only) - 1) ** 2) / discordant
    return McNemarResult(
        baseline_only,
        treatment_only,
        discordant,
        statistic,
        probability,
    )


def paired_significance_tests(
    baseline_correct: Sequence[bool | int],
    treatment_correct: Sequence[bool | int],
    *,
    iterations: int = 10_000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> dict[str, Any]:
    """Run both required paired tests over the same example outcomes."""
    bootstrap = paired_bootstrap(
        baseline_correct,
        treatment_correct,
        iterations=iterations,
        confidence_level=confidence_level,
        seed=seed,
    )
    mcnemar = mcnemar_test(baseline_correct, treatment_correct)
    return {
        "paired_bootstrap": to_jsonable(bootstrap),
        "mcnemar": to_jsonable(mcnemar),
    }


def make_sample_id(index: int, question: str) -> str:
    return f"{index}:{hashlib.sha256(question.encode('utf-8')).hexdigest()[:16]}"


def build_result_record(
    *,
    sample_id: str,
    prediction: str,
    golden_answers: Sequence[str],
    candidates: Sequence[Any] = (),
    evidence_matrix: Mapping[str, Any] | None = None,
    verification_reports: Sequence[Any] = (),
    risk_routing: Mapping[str, Any] | Sequence[Any] | None = None,
    usage: UsageRecord | Mapping[str, Any] | None = None,
    reproduction: Mapping[str, Any] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    record = {
        "sample_id": sample_id,
        "prediction": prediction,
        "golden_answers": list(golden_answers),
        "candidates": to_jsonable(candidates),
        "evidence_matrix": to_jsonable(evidence_matrix or {}),
        "verification_reports": to_jsonable(verification_reports),
        "risk_routing": to_jsonable(risk_routing or {}),
        "usage": to_jsonable(usage or UsageRecord()),
        "reproduction": to_jsonable(reproduction or {}),
    }
    if extra:
        conflicts = record.keys() & extra.keys()
        if conflicts:
            names = ", ".join(sorted(conflicts))
            raise ValueError(f"extra cannot replace required result fields: {names}")
        record.update(to_jsonable(extra))
    return record


def evaluate_result_records(
    records: Sequence[Mapping[str, Any]],
    *,
    recall_ks: Sequence[int] = (1, 3, 5),
    calibration_bins: int = 10,
) -> dict[str, Any]:
    """Aggregate outcome, evidence, robustness, calibration, and cost metrics."""
    if not recall_ks or any(k <= 0 for k in recall_ks):
        raise ValueError("recall_ks must contain positive values")
    metric_ks = tuple(dict.fromkeys(recall_ks))

    def correct(record: Mapping[str, Any]) -> bool:
        prediction = str(
            record.get("prediction", record.get("lawyerB_pred_for_eval", ""))
        ).strip().casefold()
        return bool(prediction) and any(
            prediction == str(answer).strip().casefold()
            for answer in record.get("golden_answers", [])
        )

    outcomes = [correct(record) for record in records]
    accuracy = sum(outcomes) / len(outcomes) if outcomes else 0.0
    domains: dict[str, list[bool]] = {}
    for record, outcome in zip(records, outcomes):
        domain = str(record.get("domain", "")).strip()
        if domain:
            domains.setdefault(domain, []).append(outcome)

    evidence_rows = [
        (
            list(record.get("retrieved_evidence_ids", [])),
            list(record.get("relevant_evidence_ids", [])),
        )
        for record in records
        if record.get("relevant_evidence_ids")
    ]
    evidence = {
        f"recall@{k}": _mean(
            [
                score
                for retrieved, relevant in evidence_rows
                if (
                    score := evidence_recall_at_k(retrieved, relevant, k)
                )
                is not None
            ]
        )
        for k in metric_ks
    }
    evidence["mrr"] = mean_reciprocal_rank(evidence_rows)
    evidence.update(
        {
            f"ndcg@{k}": _mean(
                [
                    score
                    for retrieved, relevant in evidence_rows
                    if (score := ndcg_at_k(retrieved, relevant, k)) is not None
                ]
            )
            for k in metric_ks
        }
    )
    evidence["citation_f1"] = _mean(
        [
            score
            for record in records
            if record.get("relevant_evidence_ids")
            if (
                score := citation_f1(
                    record.get("cited_evidence_ids", []),
                    record.get("relevant_evidence_ids", []),
                ).f1
            )
            is not None
        ]
    )

    permutations = [
        score
        for record in records
        if (
            score := permutation_consistency(
                list(record.get("permutation_predictions", []))
            )
        )
        is not None
    ]
    expected_changes: list[bool] = []
    observed_changes: list[bool] = []
    for record in records:
        for observation in record.get("counterfactual_observations", []):
            if isinstance(observation, Mapping) and observation.get("eligible", True):
                expected_changes.append(bool(observation.get("expected_change")))
                observed_changes.append(bool(observation.get("observed_change")))

    calibrated = [
        (float(record["confidence"]), outcome)
        for record, outcome in zip(records, outcomes)
        if isinstance(record.get("confidence"), (int, float))
        and not isinstance(record.get("confidence"), bool)
        and 0.0 <= float(record["confidence"]) <= 1.0
    ]
    risk_rows = [
        (float(record["risk_score"]), outcome)
        for record, outcome in zip(records, outcomes)
        if isinstance(record.get("risk_score"), (int, float))
        and not isinstance(record.get("risk_score"), bool)
        and 0.0 <= float(record["risk_score"]) <= 1.0
    ]
    return {
        "avg_acc": accuracy,
        "avg_em": accuracy,
        "avg_f1": accuracy,
        "domain_accuracy": {
            domain: sum(values) / len(values)
            for domain, values in sorted(domains.items())
        },
        "evidence": evidence,
        "robustness": {
            "permutation_consistency": _mean(permutations),
            "counterfactual": to_jsonable(
                counterfactual_sensitivity_specificity(
                    expected_changes, observed_changes
                )
            ),
        },
        "calibration": {
            "ece": expected_calibration_error(
                [value for value, _ in calibrated],
                [outcome for _, outcome in calibrated],
                bins=calibration_bins,
            ),
            "brier": brier_score(
                [value for value, _ in calibrated],
                [outcome for _, outcome in calibrated],
            ),
        },
        "risk_coverage": to_jsonable(
            risk_coverage_curve(
                [value for value, _ in risk_rows],
                [outcome for _, outcome in risk_rows],
            )
        ),
        "cost": summarize_usage(record.get("usage", {}) for record in records),
    }


class CheckpointError(RuntimeError):
    """Raised when a checkpoint cannot safely resume an experiment."""


class ExperimentCheckpoint:
    """Atomic JSON checkpoint keyed by a stable experiment and sample identity."""

    def __init__(self, path: str | Path, metadata: ExperimentMetadata) -> None:
        self.path = Path(path)
        self.metadata = metadata
        self._records: dict[str, dict[str, Any]] = {}

    @property
    def records(self) -> tuple[dict[str, Any], ...]:
        def sort_key(item: Mapping[str, Any]) -> tuple[int, int | str, str]:
            index = item.get("idx")
            if isinstance(index, int) and not isinstance(index, bool):
                return (0, index, str(item["sample_id"]))
            return (1, str(index or ""), str(item["sample_id"]))

        return tuple(
            sorted(self._records.values(), key=sort_key)
        )

    @property
    def completed_ids(self) -> frozenset[str]:
        return frozenset(self._records)

    def load(self) -> tuple[dict[str, Any], ...]:
        if not self.path.exists():
            return ()
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise CheckpointError(f"cannot read checkpoint: {exc}") from exc
        if not isinstance(payload, dict):
            raise CheckpointError("checkpoint root must be an object")
        actual_id = payload.get("experiment_id")
        if actual_id != self.metadata.experiment_id:
            raise CheckpointError(
                "checkpoint experiment ID does not match current experiment"
            )
        records = payload.get("examples")
        if not isinstance(records, list):
            raise CheckpointError("checkpoint examples must be an array")
        loaded: dict[str, dict[str, Any]] = {}
        for record in records:
            if not isinstance(record, dict) or not record.get("sample_id"):
                raise CheckpointError("every checkpoint example needs sample_id")
            sample_id = str(record["sample_id"])
            if sample_id in loaded:
                raise CheckpointError(f"duplicate sample_id in checkpoint: {sample_id}")
            loaded[sample_id] = record
        self._records = loaded
        return self.records

    def add(self, record: Mapping[str, Any]) -> bool:
        sample_id = str(record.get("sample_id", "")).strip()
        if not sample_id:
            raise CheckpointError("checkpoint record needs sample_id")
        normalized = to_jsonable(record)
        assert isinstance(normalized, dict)
        if sample_id in self._records:
            if self._records[sample_id] != normalized:
                raise CheckpointError(
                    f"conflicting checkpoint record for sample_id: {sample_id}"
                )
            return False
        self._records[sample_id] = normalized
        return True

    def save(self, payload: Mapping[str, Any] | None = None) -> None:
        document = dict(to_jsonable(payload or {}))
        document.update(
            {
                "experiment_id": self.metadata.experiment_id,
                "reproducibility": self.metadata.as_dict(),
                "examples": list(self.records),
            }
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(
            prefix=f".{self.path.name}.",
            suffix=".tmp",
            dir=self.path.parent,
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                json.dump(document, stream, ensure_ascii=False, indent=2, sort_keys=True)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
