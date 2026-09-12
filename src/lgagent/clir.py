"""Offline prototype for causal legal intervention routing (CLIR)."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

CLIR_SCHEMA_VERSION = 1
CLIR_POLICY_VERSION = "clir-group-uplift-hoeffding-v1"
CLIR_BASELINE_ACTION = "original"
_ACTION_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]*$")


def _action_name(value: Any, field_name: str = "action") -> str:
    action = str(value or "").strip().lower()
    if not _ACTION_PATTERN.fullmatch(action):
        raise ValueError(
            f"{field_name} must match {_ACTION_PATTERN.pattern!r}"
        )
    return action


def _probability(value: float, field_name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be between 0 and 1")
    result = float(value)
    if not math.isfinite(result) or not 0.0 < result < 1.0:
        raise ValueError(f"{field_name} must be between 0 and 1")
    return result


def _nonnegative(value: float, field_name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be a non-negative finite number")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{field_name} must be a non-negative finite number")
    return result


def _boolean(value: Any, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field_name} must be a boolean")
    return value


def _finite(value: float, field_name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field_name} must be a finite number")
    return result


def _correct(record: Mapping[str, Any]) -> bool:
    prediction = str(record.get("prediction", "")).strip().casefold()
    golden = record.get("golden_answers", ())
    if not isinstance(golden, Sequence) or isinstance(golden, (str, bytes)):
        raise ValueError("golden_answers must be an array")
    return bool(prediction) and any(
        prediction == str(answer).strip().casefold() for answer in golden
    )


def _usage(record: Mapping[str, Any]) -> dict[str, float | int]:
    raw = record.get("usage")
    usage = raw if isinstance(raw, Mapping) else {}
    return {
        "calls": int(usage.get("calls", 0) or 0),
        "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
        "completion_tokens": int(usage.get("completion_tokens", 0) or 0),
        "total_tokens": int(usage.get("total_tokens", 0) or 0),
        "latency_ms": float(usage.get("latency_ms", 0.0) or 0.0),
    }


@dataclass(frozen=True)
class InterventionOutcome:
    action: str
    prediction: str
    correct: bool
    status: str
    usage: Mapping[str, float | int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "action", _action_name(self.action))
        _boolean(self.correct, "correct")
        if self.status not in {"ok", "failed"}:
            raise ValueError("intervention status must be ok or failed")
        if self.status == "failed" and self.correct:
            raise ValueError("failed intervention cannot be correct")
        normalized_usage = _usage({"usage": self.usage})
        if any(float(value) < 0.0 for value in normalized_usage.values()):
            raise ValueError("intervention usage cannot be negative")
        object.__setattr__(self, "usage", normalized_usage)

    def as_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "prediction": self.prediction,
            "correct": self.correct,
            "status": self.status,
            "usage": dict(self.usage),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "InterventionOutcome":
        return cls(
            action=str(value["action"]),
            prediction=str(value.get("prediction", "")),
            correct=_boolean(value.get("correct", False), "correct"),
            status=str(value.get("status", "failed")),
            usage=(
                dict(value.get("usage", {}))
                if isinstance(value.get("usage", {}), Mapping)
                else {}
            ),
        )


@dataclass(frozen=True)
class InterventionRecord:
    sample_id: str
    split: str
    group: str
    golden_answers: tuple[str, ...]
    baseline: InterventionOutcome
    interventions: Mapping[str, InterventionOutcome]
    features: Mapping[str, Any] = field(default_factory=dict)
    provenance: Mapping[str, Any] = field(default_factory=dict)
    question: str = ""
    schema_version: int = CLIR_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.sample_id.strip():
            raise ValueError("sample_id cannot be empty")
        if self.split not in {"train", "dev", "test"}:
            raise ValueError("split must be train, dev, or test")
        if not self.golden_answers:
            raise ValueError("golden_answers cannot be empty")
        if self.baseline.action != CLIR_BASELINE_ACTION:
            raise ValueError(
                f"baseline action must be {CLIR_BASELINE_ACTION!r}"
            )
        normalized: dict[str, InterventionOutcome] = {}
        for name, outcome in self.interventions.items():
            action = _action_name(name, "intervention key")
            if action == CLIR_BASELINE_ACTION:
                raise ValueError("interventions cannot contain baseline action")
            if action != outcome.action:
                raise ValueError("intervention key and outcome action differ")
            if action in normalized:
                raise ValueError(f"duplicate intervention action: {action}")
            normalized[action] = outcome
        if not normalized:
            raise ValueError("at least one intervention action is required")
        if self.schema_version != CLIR_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported CLIR schema version: {self.schema_version}"
            )
        object.__setattr__(self, "interventions", normalized)
        object.__setattr__(self, "features", dict(self.features or {}))
        object.__setattr__(self, "provenance", dict(self.provenance or {}))

    def as_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "split": self.split,
            "group": self.group,
            "golden_answers": list(self.golden_answers),
            "question": self.question,
            "baseline": self.baseline.as_dict(),
            "interventions": {
                action: outcome.as_dict()
                for action, outcome in sorted(self.interventions.items())
            },
            "features": dict(self.features),
            "provenance": dict(self.provenance),
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "InterventionRecord":
        baseline = value.get("baseline")
        interventions = value.get("interventions")
        if not isinstance(baseline, Mapping):
            raise ValueError("baseline must be an object")
        if not isinstance(interventions, Mapping):
            raise ValueError("interventions must be an object")
        golden = value.get("golden_answers")
        if not isinstance(golden, Sequence) or isinstance(
            golden, (str, bytes)
        ):
            raise ValueError("golden_answers must be an array")
        parsed_interventions: dict[str, InterventionOutcome] = {}
        for action, outcome in interventions.items():
            if not isinstance(outcome, Mapping):
                raise ValueError("each intervention outcome must be an object")
            parsed_interventions[str(action)] = InterventionOutcome.from_dict(
                outcome
            )
        return cls(
            sample_id=str(value.get("sample_id", "")),
            split=str(value.get("split", "")),
            group=str(value.get("group", "")),
            golden_answers=tuple(str(item) for item in golden),
            question=str(value.get("question", "")),
            baseline=InterventionOutcome.from_dict(baseline),
            interventions=parsed_interventions,
            features=(
                dict(value.get("features", {}))
                if isinstance(value.get("features", {}), Mapping)
                else {}
            ),
            provenance=(
                dict(value.get("provenance", {}))
                if isinstance(value.get("provenance", {}), Mapping)
                else {}
            ),
            schema_version=int(value.get("schema_version", 0)),
        )


def _outcome(action: str, record: Mapping[str, Any]) -> InterventionOutcome:
    status = str(record.get("status", "ok"))
    if status not in {"ok", "failed"}:
        status = "failed"
    return InterventionOutcome(
        action=action,
        prediction=str(record.get("prediction", "")).strip(),
        correct=_correct(record) if status == "ok" else False,
        status=status,
        usage=_usage(record),
    )


def build_intervention_records(
    baseline_records: Sequence[Mapping[str, Any]],
    action_records: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    designated_split: str | None = None,
) -> tuple[InterventionRecord, ...]:
    """Pair result records by sample ID without dropping failed outcomes."""
    if not baseline_records:
        raise ValueError("baseline records cannot be empty")
    action_names = tuple(_action_name(name) for name in action_records)
    if not action_names:
        raise ValueError("at least one action result set is required")
    if CLIR_BASELINE_ACTION in action_names:
        raise ValueError("action result sets cannot use baseline action name")
    if len(set(action_names)) != len(action_names):
        raise ValueError("action names must be unique")
    if designated_split is not None and designated_split not in {
        "train",
        "dev",
        "test",
    }:
        raise ValueError("designated_split must be train, dev, or test")

    def index(
        records: Sequence[Mapping[str, Any]],
        label: str,
    ) -> dict[str, Mapping[str, Any]]:
        result: dict[str, Mapping[str, Any]] = {}
        for record in records:
            sample_id = str(record.get("sample_id", "")).strip()
            if not sample_id:
                raise ValueError(f"{label} record sample_id cannot be empty")
            if sample_id in result:
                raise ValueError(f"{label} contains duplicate sample_id")
            result[sample_id] = record
        return result

    baseline_by_id = index(baseline_records, CLIR_BASELINE_ACTION)
    actions_by_name = {
        name: index(records, name)
        for name, records in zip(action_names, action_records.values())
    }
    baseline_ids = set(baseline_by_id)
    for name, records in actions_by_name.items():
        if set(records) != baseline_ids:
            missing = sorted(baseline_ids - set(records))
            extra = sorted(set(records) - baseline_ids)
            raise ValueError(
                f"{name} sample IDs differ from baseline: "
                f"missing={missing[:3]}, extra={extra[:3]}"
            )

    paired: list[InterventionRecord] = []
    for sample_id, baseline in baseline_by_id.items():
        golden = baseline.get("golden_answers")
        if not isinstance(golden, Sequence) or isinstance(
            golden, (str, bytes)
        ):
            raise ValueError("baseline golden_answers must be an array")
        split = str(baseline.get("split", ""))
        output_split = designated_split or split
        question = str(baseline.get("question", ""))
        outcomes: dict[str, InterventionOutcome] = {}
        for name, records in actions_by_name.items():
            action_record = records[sample_id]
            if list(action_record.get("golden_answers", ())) != list(golden):
                raise ValueError(f"{name} golden answers differ for {sample_id}")
            if str(action_record.get("split", "")) != split:
                raise ValueError(f"{name} split differs for {sample_id}")
            if str(action_record.get("question", "")) != question:
                raise ValueError(f"{name} question differs for {sample_id}")
            baseline_reproduction = baseline.get("reproduction")
            action_reproduction = action_record.get("reproduction")
            if isinstance(baseline_reproduction, Mapping) and isinstance(
                action_reproduction, Mapping
            ):
                for field_name in (
                    "model_id",
                    "dataset_hash",
                    "random_seed",
                    "prompt_version",
                ):
                    baseline_value = baseline_reproduction.get(field_name)
                    action_value = action_reproduction.get(field_name)
                    if (
                        baseline_value is not None
                        and action_value is not None
                        and baseline_value != action_value
                    ):
                        raise ValueError(
                            f"{name} reproduction {field_name} differs "
                            f"for {sample_id}"
                        )
            outcomes[name] = _outcome(name, action_record)
        paired.append(
            InterventionRecord(
                sample_id=sample_id,
                split=output_split,
                group=str(baseline.get("domain", "")).strip(),
                golden_answers=tuple(str(item) for item in golden),
                question=question,
                baseline=_outcome(CLIR_BASELINE_ACTION, baseline),
                interventions=outcomes,
                features={
                    "domain": str(baseline.get("domain", "")).strip(),
                    "b0_confidence": baseline.get("b0_confidence"),
                    "dialogue_rounds": baseline.get("dialogue_rounds"),
                    "question_chars": len(question),
                },
                provenance={
                    field_name: baseline.get("reproduction", {}).get(field_name)
                    for field_name in (
                        "model_id",
                        "dataset_hash",
                        "random_seed",
                        "prompt_version",
                    )
                    if isinstance(baseline.get("reproduction"), Mapping)
                    and baseline["reproduction"].get(field_name) is not None
                }
                | (
                    {
                        "source_split": split,
                        "designated_split": output_split,
                    }
                    if designated_split is not None
                    else {}
                ),
            )
        )
    return tuple(paired)


def hoeffding_lower_bound(
    paired_uplifts: Sequence[int],
    *,
    alpha: float,
) -> float:
    """One-sided finite-sample lower bound for values in [-1, 1]."""
    _probability(alpha, "alpha")
    if not paired_uplifts:
        raise ValueError("paired_uplifts cannot be empty")
    if any(value not in {-1, 0, 1} for value in paired_uplifts):
        raise ValueError("paired uplifts must be -1, 0, or 1")
    mean = sum(paired_uplifts) / len(paired_uplifts)
    radius = 2.0 * math.sqrt(
        math.log(1.0 / alpha) / (2.0 * len(paired_uplifts))
    )
    return max(-1.0, mean - radius)


@dataclass(frozen=True)
class ActionEstimate:
    action: str
    count: int
    mean_uplift: float
    lower_bound: float
    corrected_count: int
    harmed_count: int
    unchanged_count: int
    mean_incremental_tokens: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "action", _action_name(self.action))
        if self.count <= 0:
            raise ValueError("action estimate count must be positive")
        if self.corrected_count + self.harmed_count + self.unchanged_count != self.count:
            raise ValueError("action estimate counts do not sum to count")
        for name in ("mean_uplift", "lower_bound"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not -1.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between -1 and 1")
        _finite(self.mean_incremental_tokens, "mean_incremental_tokens")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ActionEstimate":
        return cls(
            action=str(value["action"]),
            count=int(value["count"]),
            mean_uplift=float(value["mean_uplift"]),
            lower_bound=float(value["lower_bound"]),
            corrected_count=int(value["corrected_count"]),
            harmed_count=int(value["harmed_count"]),
            unchanged_count=int(value["unchanged_count"]),
            mean_incremental_tokens=float(
                value.get("mean_incremental_tokens", 0.0)
            ),
        )


@dataclass(frozen=True)
class ClirCalibration:
    alpha: float
    group_field: str
    min_group_size: int
    global_estimates: Mapping[str, ActionEstimate]
    group_estimates: Mapping[str, Mapping[str, ActionEstimate]]
    calibration_count: int
    records_sha256: str
    baseline_action: str = CLIR_BASELINE_ACTION
    policy_version: str = CLIR_POLICY_VERSION
    schema_version: int = CLIR_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _probability(self.alpha, "alpha")
        if not self.group_field.strip():
            raise ValueError("group_field cannot be empty")
        if self.min_group_size <= 0 or self.calibration_count <= 0:
            raise ValueError("calibration sizes must be positive")
        if self.baseline_action != CLIR_BASELINE_ACTION:
            raise ValueError("unsupported baseline action")
        if self.policy_version != CLIR_POLICY_VERSION:
            raise ValueError("unsupported CLIR policy version")
        if self.schema_version != CLIR_SCHEMA_VERSION:
            raise ValueError("unsupported CLIR schema version")
        if not self.global_estimates:
            raise ValueError("global_estimates cannot be empty")
        object.__setattr__(
            self,
            "global_estimates",
            dict(self.global_estimates),
        )
        object.__setattr__(
            self,
            "group_estimates",
            {
                str(group): dict(estimates)
                for group, estimates in self.group_estimates.items()
            },
        )
        for action, estimate in self.global_estimates.items():
            if _action_name(action) != estimate.action:
                raise ValueError(
                    "global estimate key and action differ"
                )
        for group, estimates in self.group_estimates.items():
            if not group.strip():
                raise ValueError("CLIR estimate group cannot be empty")
            if not set(estimates).issubset(self.global_estimates):
                raise ValueError(
                    "group estimates contain an unknown action"
                )
            for action, estimate in estimates.items():
                if _action_name(action) != estimate.action:
                    raise ValueError(
                        "group estimate key and action differ"
                    )

    def estimate_for(
        self,
        action: str,
        group: str | None,
    ) -> tuple[ActionEstimate, str]:
        normalized_action = _action_name(action)
        normalized_group = str(group or "").strip()
        grouped = self.group_estimates.get(normalized_group, {})
        estimate = grouped.get(normalized_action)
        if estimate is not None and estimate.count >= self.min_group_size:
            return estimate, f"{self.group_field}:{normalized_group}"
        try:
            return self.global_estimates[normalized_action], "global"
        except KeyError as exc:
            raise ValueError(f"unknown CLIR action: {normalized_action}") from exc

    def as_dict(self) -> dict[str, Any]:
        return {
            "alpha": self.alpha,
            "group_field": self.group_field,
            "min_group_size": self.min_group_size,
            "global_estimates": {
                action: estimate.as_dict()
                for action, estimate in sorted(self.global_estimates.items())
            },
            "group_estimates": {
                group: {
                    action: estimate.as_dict()
                    for action, estimate in sorted(estimates.items())
                }
                for group, estimates in sorted(self.group_estimates.items())
            },
            "calibration_count": self.calibration_count,
            "records_sha256": self.records_sha256,
            "baseline_action": self.baseline_action,
            "policy_version": self.policy_version,
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ClirCalibration":
        raw_global = value.get("global_estimates")
        raw_groups = value.get("group_estimates", {})
        if not isinstance(raw_global, Mapping) or not isinstance(
            raw_groups, Mapping
        ):
            raise ValueError("CLIR estimates must be objects")
        return cls(
            alpha=float(value["alpha"]),
            group_field=str(value["group_field"]),
            min_group_size=int(value["min_group_size"]),
            global_estimates={
                str(action): ActionEstimate.from_dict(estimate)
                for action, estimate in raw_global.items()
                if isinstance(estimate, Mapping)
            },
            group_estimates={
                str(group): {
                    str(action): ActionEstimate.from_dict(estimate)
                    for action, estimate in estimates.items()
                    if isinstance(estimate, Mapping)
                }
                for group, estimates in raw_groups.items()
                if isinstance(estimates, Mapping)
            },
            calibration_count=int(value["calibration_count"]),
            records_sha256=str(value.get("records_sha256", "")),
            baseline_action=str(
                value.get("baseline_action", CLIR_BASELINE_ACTION)
            ),
            policy_version=str(value.get("policy_version", "")),
            schema_version=int(value.get("schema_version", 0)),
        )

    @classmethod
    def load(cls, path: str | Path) -> "ClirCalibration":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError("CLIR calibration artifact must be an object")
        return cls.from_dict(payload)


def _estimate(
    records: Sequence[InterventionRecord],
    action: str,
    *,
    alpha: float,
) -> ActionEstimate:
    deltas: list[int] = []
    incremental_tokens: list[int] = []
    for record in records:
        outcome = record.interventions[action]
        delta = int(outcome.correct) - int(record.baseline.correct)
        deltas.append(delta)
        incremental_tokens.append(
            int(outcome.usage["total_tokens"])
            - int(record.baseline.usage["total_tokens"])
        )
    return ActionEstimate(
        action=action,
        count=len(records),
        mean_uplift=sum(deltas) / len(deltas),
        lower_bound=hoeffding_lower_bound(deltas, alpha=alpha),
        corrected_count=sum(value == 1 for value in deltas),
        harmed_count=sum(value == -1 for value in deltas),
        unchanged_count=sum(value == 0 for value in deltas),
        mean_incremental_tokens=sum(incremental_tokens) / len(incremental_tokens),
    )


def fit_clir_calibration(
    records: Sequence[InterventionRecord],
    *,
    alpha: float = 0.1,
    group_field: str = "domain",
    min_calibration_size: int = 30,
    min_group_size: int = 30,
) -> ClirCalibration:
    """Fit conservative paired-uplift bounds on development interventions."""
    _probability(alpha, "alpha")
    if min_calibration_size <= 0 or min_group_size <= 0:
        raise ValueError("minimum calibration sizes must be positive")
    if len(records) < min_calibration_size:
        raise ValueError(
            f"insufficient CLIR calibration records: {len(records)} "
            f"< {min_calibration_size}"
        )
    seen: set[str] = set()
    action_set: set[str] | None = None
    canonical: list[Mapping[str, Any]] = []
    for record in records:
        if record.split != "dev":
            raise ValueError("CLIR calibration accepts development records only")
        if record.sample_id in seen:
            raise ValueError("CLIR calibration requires unique sample_id values")
        seen.add(record.sample_id)
        current = set(record.interventions)
        if action_set is None:
            action_set = current
        elif current != action_set:
            raise ValueError("all CLIR records must contain identical actions")
        canonical.append(record.as_dict())
    assert action_set
    per_action_alpha = alpha / len(action_set)
    global_estimates = {
        action: _estimate(records, action, alpha=per_action_alpha)
        for action in sorted(action_set)
    }
    grouped_records: dict[str, list[InterventionRecord]] = {}
    for record in records:
        group = str(record.features.get(group_field, record.group)).strip()
        if group:
            grouped_records.setdefault(group, []).append(record)
    group_estimates = {
        group: {
            action: _estimate(group_records, action, alpha=per_action_alpha)
            for action in sorted(action_set)
        }
        for group, group_records in grouped_records.items()
    }
    encoded = json.dumps(
        canonical,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return ClirCalibration(
        alpha=alpha,
        group_field=group_field,
        min_group_size=min_group_size,
        global_estimates=global_estimates,
        group_estimates=group_estimates,
        calibration_count=len(records),
        records_sha256=hashlib.sha256(encoded).hexdigest(),
    )


@dataclass(frozen=True)
class ClirDecision:
    selected_action: str
    applied: bool
    utility: float
    lower_bound: float
    estimate_source: str
    action_scores: Mapping[str, Mapping[str, Any]]
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "selected_action": self.selected_action,
            "applied": self.applied,
            "utility": self.utility,
            "lower_bound": self.lower_bound,
            "estimate_source": self.estimate_source,
            "action_scores": {
                action: dict(score)
                for action, score in self.action_scores.items()
            },
            "reason": self.reason,
            "policy_version": CLIR_POLICY_VERSION,
        }


class ClirRouter:
    """Select only interventions with a positive conservative utility."""

    def __init__(
        self,
        calibration: ClirCalibration,
        *,
        minimum_uplift: float = 0.0,
        cost_weight: float = 0.0,
        token_scale: float = 100_000.0,
    ) -> None:
        if not -1.0 <= float(minimum_uplift) <= 1.0:
            raise ValueError("minimum_uplift must be between -1 and 1")
        _nonnegative(cost_weight, "cost_weight")
        if token_scale <= 0.0 or not math.isfinite(token_scale):
            raise ValueError("token_scale must be a positive finite number")
        self.calibration = calibration
        self.minimum_uplift = float(minimum_uplift)
        self.cost_weight = float(cost_weight)
        self.token_scale = float(token_scale)

    def decide(self, *, group: str | None = None) -> ClirDecision:
        scores: dict[str, dict[str, Any]] = {}
        for action in sorted(self.calibration.global_estimates):
            estimate, source = self.calibration.estimate_for(action, group)
            cost = max(0.0, estimate.mean_incremental_tokens)
            utility = estimate.lower_bound - self.cost_weight * (
                cost / self.token_scale
            )
            scores[action] = {
                "utility": utility,
                "lower_bound": estimate.lower_bound,
                "mean_uplift": estimate.mean_uplift,
                "mean_incremental_tokens": estimate.mean_incremental_tokens,
                "estimate_source": source,
                "count": estimate.count,
            }
        selected_action = max(
            scores,
            key=lambda action: (
                scores[action]["utility"],
                scores[action]["lower_bound"],
                action,
            ),
        )
        selected = scores[selected_action]
        if selected["utility"] <= self.minimum_uplift:
            return ClirDecision(
                selected_action=CLIR_BASELINE_ACTION,
                applied=False,
                utility=0.0,
                lower_bound=0.0,
                estimate_source="baseline",
                action_scores=scores,
                reason="no_positive_conservative_uplift",
            )
        return ClirDecision(
            selected_action=selected_action,
            applied=True,
            utility=float(selected["utility"]),
            lower_bound=float(selected["lower_bound"]),
            estimate_source=str(selected["estimate_source"]),
            action_scores=scores,
            reason="positive_conservative_uplift",
        )


def replay_clir_policy(
    records: Sequence[InterventionRecord],
    calibration: ClirCalibration,
    *,
    expected_split: str = "test",
    minimum_uplift: float = 0.0,
    cost_weight: float = 0.0,
    token_scale: float = 100_000.0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Replay a frozen CLIR policy using already-observed intervention arms."""
    if not records:
        raise ValueError("CLIR replay records cannot be empty")
    router = ClirRouter(
        calibration,
        minimum_uplift=minimum_uplift,
        cost_weight=cost_weight,
        token_scale=token_scale,
    )
    transformed: list[dict[str, Any]] = []
    corrected = harmed = selected_tokens = baseline_tokens = 0
    oracle_correct = routed_correct = 0
    action_counts: dict[str, int] = {}
    seen: set[str] = set()
    for record in records:
        if record.split != expected_split:
            raise ValueError(
                f"CLIR replay expected split {expected_split!r}, "
                f"got {record.split!r}"
            )
        if record.sample_id in seen:
            raise ValueError("CLIR replay requires unique sample_id values")
        seen.add(record.sample_id)
        group = str(
            record.features.get(calibration.group_field, record.group)
        ).strip()
        decision = router.decide(group=group)
        outcome = (
            record.baseline
            if decision.selected_action == CLIR_BASELINE_ACTION
            else record.interventions.get(decision.selected_action)
        )
        if outcome is None:
            raise ValueError(
                f"record {record.sample_id} is missing selected action "
                f"{decision.selected_action}"
            )
        action_counts[decision.selected_action] = (
            action_counts.get(decision.selected_action, 0) + 1
        )
        baseline_tokens += int(record.baseline.usage["total_tokens"])
        selected_tokens += int(outcome.usage["total_tokens"])
        corrected += int(not record.baseline.correct and outcome.correct)
        harmed += int(record.baseline.correct and not outcome.correct)
        routed_correct += int(outcome.correct)
        oracle_correct += int(
            record.baseline.correct
            or any(item.correct for item in record.interventions.values())
        )
        transformed.append(
            {
                "sample_id": record.sample_id,
                "split": record.split,
                "domain": record.group,
                "question": record.question,
                "golden_answers": list(record.golden_answers),
                "prediction": outcome.prediction,
                "status": outcome.status,
                "usage": dict(outcome.usage),
                "baseline_prediction": record.baseline.prediction,
                "clir": decision.as_dict(),
            }
        )
    count = len(records)
    baseline_correct = sum(record.baseline.correct for record in records)
    applied_count = sum(
        count
        for action, count in action_counts.items()
        if action != CLIR_BASELINE_ACTION
    )
    summary = {
        "records": count,
        "baseline_accuracy": baseline_correct / count if count else None,
        "routed_accuracy": routed_correct / count if count else None,
        "oracle_action_accuracy": oracle_correct / count if count else None,
        "application_rate": applied_count / count,
        "corrected_count": corrected,
        "harmed_count": harmed,
        "net_improvement_count": corrected - harmed,
        "net_improvement_rate": (
            (corrected - harmed) / count if count else None
        ),
        "action_counts": dict(sorted(action_counts.items())),
        "baseline_total_tokens": baseline_tokens,
        "routed_total_tokens": selected_tokens,
        "token_difference": selected_tokens - baseline_tokens,
        "calibration_records_sha256": calibration.records_sha256,
        "policy_version": CLIR_POLICY_VERSION,
    }
    return transformed, summary
