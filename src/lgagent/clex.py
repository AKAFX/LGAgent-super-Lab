"""Conformal legal option elimination over existing LGAgent++ signals."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .cape_v import OPTION_LABELS
from .evidence_audit import (
    AuditedEvidenceMatrix,
    OptionEvidenceAudit,
    TemporalStatus,
)
from .protocol import normalize_option_label
from .risk import CandidateSignals

CLEX_SCHEMA_VERSION = 1
CLEX_SCORE_VERSION = "clex-option-conformity-v1"


def _unit(value: float, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number between 0 and 1")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be a finite number between 0 and 1")
    return result


def _mean(values: Iterable[float]) -> float | None:
    collected = tuple(values)
    return sum(collected) / len(collected) if collected else None


def _weighted_mean(values: Mapping[str, tuple[float, float]]) -> float:
    active = tuple(
        (value, weight)
        for value, weight in values.values()
        if weight > 0.0
    )
    if not active:
        return 0.0
    return sum(value * weight for value, weight in active) / sum(
        weight for _, weight in active
    )


def _temporal_score(value: Any) -> float:
    if isinstance(value, TemporalStatus):
        status = value
    else:
        try:
            status = TemporalStatus(str(value))
        except ValueError:
            return 0.0
    return {
        TemporalStatus.VALID: 1.0,
        TemporalStatus.MIXED: 0.5,
        TemporalStatus.INVALID: 0.0,
        TemporalStatus.NO_EVIDENCE: 0.0,
    }[status]


def _option_audit(
    matrix: AuditedEvidenceMatrix | Mapping[str, Any] | None,
    option: str,
) -> OptionEvidenceAudit | Mapping[str, Any] | None:
    if matrix is None:
        return None
    options = matrix.options if isinstance(matrix, AuditedEvidenceMatrix) else matrix
    if "options" in options and isinstance(options["options"], Mapping):
        options = options["options"]
    value = options.get(option)
    return value if isinstance(value, (OptionEvidenceAudit, Mapping)) else None


def _audit_field(audit: OptionEvidenceAudit | Mapping[str, Any], name: str) -> Any:
    return audit.get(name) if isinstance(audit, Mapping) else getattr(audit, name)


def _evidence_conformity(
    matrix: AuditedEvidenceMatrix | Mapping[str, Any] | None,
    option: str,
) -> float | None:
    audit = _option_audit(matrix, option)
    if audit is None:
        return None
    support = len(_audit_field(audit, "support") or ())
    refute = len(_audit_field(audit, "refute") or ())
    exception = len(_audit_field(audit, "exception") or ())
    total = support + refute + exception
    if total == 0:
        return 0.0
    polarity = (support + 1.0) / (total + 2.0)
    coverage = _unit(float(_audit_field(audit, "coverage") or 0.0), "coverage")
    authority = _unit(float(_audit_field(audit, "authority") or 0.0), "authority")
    conflict = _unit(float(_audit_field(audit, "conflict") or 0.0), "conflict")
    temporal = _temporal_score(_audit_field(audit, "temporal"))
    quality = 0.4 * coverage + 0.3 * authority + 0.3 * temporal
    return _unit(polarity * quality * (1.0 - 0.5 * conflict), "evidence")


def _candidate_option_score(
    candidates: Sequence[CandidateSignals],
    option: str,
) -> float | None:
    values: list[float] = []
    for item in candidates:
        detail = item.candidate.option_verification.get(option)
        if detail is None:
            continue
        if detail.status == "SUPPORT":
            values.append(detail.score)
        elif detail.status == "REFUTE":
            values.append(1.0 - detail.score)
        else:
            values.append(0.5)
    return _mean(values)


@dataclass(frozen=True)
class ClexOptionScore:
    option: str
    conformity: float
    nonconformity: float
    components: Mapping[str, float | None]

    def __post_init__(self) -> None:
        normalize_option_label(self.option, "clex option")
        _unit(self.conformity, "conformity")
        _unit(self.nonconformity, "nonconformity")
        if not math.isclose(
            self.conformity + self.nonconformity,
            1.0,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("conformity and nonconformity must sum to one")

    def as_dict(self) -> dict[str, Any]:
        return {
            "option": self.option,
            "conformity": self.conformity,
            "nonconformity": self.nonconformity,
            "components": dict(self.components),
        }


def score_options(
    candidates: Sequence[CandidateSignals],
    *,
    evidence_matrix: AuditedEvidenceMatrix | Mapping[str, Any] | None = None,
    permutation_answers: Sequence[str] = (),
) -> dict[str, ClexOptionScore]:
    """Compute fixed, label-free option scores from existing inference signals."""
    if not candidates:
        raise ValueError("C-LEX requires at least one candidate")
    answers = [
        normalize_option_label(item.candidate.answer, "candidate answer")
        for item in candidates
    ]
    answer_counts = Counter(answers)
    permutation_counts = Counter(
        normalize_option_label(item, "permutation answer")
        for item in permutation_answers
    )
    total = len(candidates)
    permutation_total = len(permutation_answers)

    result: dict[str, ClexOptionScore] = {}
    for option in OPTION_LABELS:
        matching = [
            item for item, answer in zip(candidates, answers) if answer == option
        ]
        verifier = _mean(item.resolved_verifier_score for item in matching)
        candidate_option = _candidate_option_score(candidates, option)
        evidence = _evidence_conformity(evidence_matrix, option)
        components: dict[str, float | None] = {
            "answer_frequency": answer_counts[option] / total,
            "candidate_option_support": candidate_option,
            "verifier_support": verifier,
            "evidence_support": evidence,
            "permutation_support": (
                permutation_counts[option] / permutation_total
                if permutation_total
                else None
            ),
        }
        weighted = {
            "answer_frequency": (components["answer_frequency"] or 0.0, 0.25),
            "candidate_option_support": (
                components["candidate_option_support"] or 0.0,
                0.25 if candidate_option is not None else 0.0,
            ),
            "verifier_support": (
                components["verifier_support"] or 0.0,
                0.20 if verifier is not None else 0.0,
            ),
            "evidence_support": (
                components["evidence_support"] or 0.0,
                0.20 if evidence is not None else 0.0,
            ),
            "permutation_support": (
                components["permutation_support"] or 0.0,
                0.10 if permutation_total else 0.0,
            ),
        }
        conformity = _unit(_weighted_mean(weighted), f"{option}.conformity")
        result[option] = ClexOptionScore(
            option=option,
            conformity=conformity,
            nonconformity=1.0 - conformity,
            components=components,
        )
    return result


def conformal_quantile(scores: Sequence[float], alpha: float) -> float:
    """Return the finite-sample split-conformal quantile."""
    if not scores:
        raise ValueError("calibration scores cannot be empty")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be between 0 and 1")
    ordered = sorted(_unit(score, "calibration score") for score in scores)
    rank = min(len(ordered), math.ceil((len(ordered) + 1) * (1.0 - alpha)))
    return ordered[rank - 1]


@dataclass(frozen=True)
class ClexCalibration:
    alpha: float
    global_threshold: float
    global_count: int
    group_field: str = "domain"
    min_group_size: int = 30
    group_thresholds: Mapping[str, float] = field(default_factory=dict)
    group_counts: Mapping[str, int] = field(default_factory=dict)
    score_version: str = CLEX_SCORE_VERSION
    schema_version: int = CLEX_SCHEMA_VERSION
    records_sha256: str = ""

    def __post_init__(self) -> None:
        if not 0.0 < self.alpha < 1.0:
            raise ValueError("alpha must be between 0 and 1")
        _unit(self.global_threshold, "global_threshold")
        if self.global_count <= 0:
            raise ValueError("global_count must be positive")
        if not self.group_field.strip():
            raise ValueError("group_field cannot be empty")
        if self.min_group_size <= 0:
            raise ValueError("min_group_size must be positive")
        object.__setattr__(self, "group_thresholds", dict(self.group_thresholds or {}))
        object.__setattr__(self, "group_counts", dict(self.group_counts or {}))
        if self.score_version != CLEX_SCORE_VERSION:
            raise ValueError(
                f"unsupported C-LEX score version: {self.score_version}"
            )
        if self.schema_version != CLEX_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported C-LEX schema version: {self.schema_version}"
            )
        for group, threshold in self.group_thresholds.items():
            if not str(group).strip():
                raise ValueError("calibration group cannot be empty")
            _unit(threshold, f"group_thresholds.{group}")

    def threshold_for(self, group: str | None) -> tuple[float, str]:
        normalized = str(group or "").strip()
        if (
            normalized
            and normalized in self.group_thresholds
            and self.group_counts.get(normalized, 0) >= self.min_group_size
        ):
            return self.group_thresholds[normalized], f"{self.group_field}:{normalized}"
        return self.global_threshold, "global"

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ClexCalibration":
        return cls(
            alpha=float(value["alpha"]),
            global_threshold=float(value["global_threshold"]),
            global_count=int(value["global_count"]),
            group_field=str(value.get("group_field", "domain")),
            min_group_size=int(value.get("min_group_size", 30)),
            group_thresholds={
                str(key): float(threshold)
                for key, threshold in dict(
                    value.get("group_thresholds", {})
                ).items()
            },
            group_counts={
                str(key): int(count)
                for key, count in dict(value.get("group_counts", {})).items()
            },
            score_version=str(value.get("score_version", "")),
            schema_version=int(value.get("schema_version", 0)),
            records_sha256=str(value.get("records_sha256", "")),
        )

    @classmethod
    def load(cls, path: str | Path) -> "ClexCalibration":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(data, Mapping):
            raise ValueError("C-LEX calibration artifact must be a JSON object")
        return cls.from_dict(data)


def _record_scores(record: Mapping[str, Any]) -> Mapping[str, Any]:
    scores = record.get("clex_option_scores")
    if not isinstance(scores, Mapping):
        raise ValueError("record is missing clex_option_scores")
    return scores


def fit_calibration(
    records: Sequence[Mapping[str, Any]],
    *,
    alpha: float = 0.1,
    group_field: str = "domain",
    min_calibration_size: int = 30,
    min_group_size: int = 30,
) -> ClexCalibration:
    """Fit a split-conformal artifact from development result records."""
    if min_calibration_size <= 0 or min_group_size <= 0:
        raise ValueError("minimum calibration sizes must be positive")
    gold_scores: list[float] = []
    grouped: dict[str, list[float]] = {}
    canonical: list[Mapping[str, Any]] = []
    sample_ids: set[str] = set()
    for record in records:
        if record.get("status", "ok") != "ok":
            raise ValueError("C-LEX calibration cannot silently omit failed records")
        if record.get("split") != "dev":
            raise ValueError("C-LEX calibration accepts development records only")
        if record.get("clex_score_version") != CLEX_SCORE_VERSION:
            raise ValueError("record C-LEX score version is missing or incompatible")
        sample_id = str(record.get("sample_id", "")).strip()
        if not sample_id:
            raise ValueError("record sample_id cannot be empty")
        if sample_id in sample_ids:
            raise ValueError(
                "C-LEX calibration requires unique sample_id values; "
                "do not pool repeated seeds as independent samples"
            )
        sample_ids.add(sample_id)
        golden = record.get("golden_answers")
        if not isinstance(golden, Sequence) or isinstance(golden, (str, bytes)):
            raise ValueError("record golden_answers must be an array")
        labels = tuple(
            normalize_option_label(str(item), "golden answer") for item in golden
        )
        scores = _record_scores(record)
        values = []
        for label in labels:
            item = scores.get(label)
            if not isinstance(item, Mapping):
                raise ValueError(f"record is missing C-LEX score for gold option {label}")
            values.append(_unit(float(item["nonconformity"]), "gold nonconformity"))
        if not values:
            raise ValueError("record must contain at least one golden answer")
        score = min(values)
        gold_scores.append(score)
        group = str(record.get(group_field, "")).strip()
        if group:
            grouped.setdefault(group, []).append(score)
        canonical.append(
            {
                "sample_id": sample_id,
                "golden_answers": list(labels),
                "gold_nonconformity": score,
                group_field: group,
            }
        )
    if len(gold_scores) < min_calibration_size:
        raise ValueError(
            "insufficient development calibration records: "
            f"{len(gold_scores)} < {min_calibration_size}"
        )
    group_thresholds = {
        group: conformal_quantile(scores, alpha)
        for group, scores in sorted(grouped.items())
        if len(scores) >= min_group_size
    }
    encoded = json.dumps(
        canonical,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return ClexCalibration(
        alpha=alpha,
        global_threshold=conformal_quantile(gold_scores, alpha),
        global_count=len(gold_scores),
        group_field=group_field,
        min_group_size=min_group_size,
        group_thresholds=group_thresholds,
        group_counts={
            group: len(scores)
            for group, scores in sorted(grouped.items())
            if group in group_thresholds
        },
        records_sha256=hashlib.sha256(encoded).hexdigest(),
    )


@dataclass(frozen=True)
class ClexDecision:
    prediction_set: tuple[str, ...]
    selected_answer: str
    threshold: float
    threshold_source: str
    applied: bool
    fallback_reason: str | None
    eliminated: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "prediction_set": list(self.prediction_set),
            "selected_answer": self.selected_answer,
            "threshold": self.threshold,
            "threshold_source": self.threshold_source,
            "applied": self.applied,
            "fallback_reason": self.fallback_reason,
            "eliminated": list(self.eliminated),
        }


def apply_calibration(
    option_scores: Mapping[str, ClexOptionScore],
    calibration: ClexCalibration,
    *,
    current_answer: str,
    group: str | None = None,
    min_calibration_size: int = 30,
) -> ClexDecision:
    """Build a prediction set and select deterministically inside that set."""
    current = normalize_option_label(current_answer, "current answer")
    if calibration.global_count < min_calibration_size:
        return ClexDecision(
            prediction_set=OPTION_LABELS,
            selected_answer=current,
            threshold=calibration.global_threshold,
            threshold_source="global",
            applied=False,
            fallback_reason="insufficient_calibration",
            eliminated=(),
        )
    if set(option_scores) != set(OPTION_LABELS):
        raise ValueError("C-LEX option scores must contain exactly A-D")
    threshold, source = calibration.threshold_for(group)
    prediction_set = tuple(
        option
        for option in OPTION_LABELS
        if option_scores[option].nonconformity <= threshold
    )
    if not prediction_set:
        return ClexDecision(
            prediction_set=OPTION_LABELS,
            selected_answer=current,
            threshold=threshold,
            threshold_source=source,
            applied=False,
            fallback_reason="empty_prediction_set",
            eliminated=(),
        )
    if len(prediction_set) == len(OPTION_LABELS):
        return ClexDecision(
            prediction_set=prediction_set,
            selected_answer=current,
            threshold=threshold,
            threshold_source=source,
            applied=False,
            fallback_reason="no_options_eliminated",
            eliminated=(),
        )
    selected = max(
        prediction_set,
        key=lambda option: (
            option_scores[option].conformity,
            option == current,
            -OPTION_LABELS.index(option),
        ),
    )
    return ClexDecision(
        prediction_set=prediction_set,
        selected_answer=selected,
        threshold=threshold,
        threshold_source=source,
        applied=True,
        fallback_reason=None,
        eliminated=tuple(
            option for option in OPTION_LABELS if option not in prediction_set
        ),
    )
