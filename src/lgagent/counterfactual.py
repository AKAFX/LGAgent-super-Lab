"""Safe counterfactual transformations and consistency metrics for CAPE-V."""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable, Mapping

from .cape_v import (
    OPTION_LABELS,
    OptionPermutation,
    generate_option_permutations,
    parse_four_option_question,
)
from .protocol import normalize_option_label


class TransformationExpectation(str, Enum):
    PRESERVE_LABEL = "PRESERVE_LABEL"
    CHANGE_LABEL = "CHANGE_LABEL"


class ProvenanceType(str, Enum):
    DETERMINISTIC_RULE = "DETERMINISTIC_RULE"
    TRUSTED_EVIDENCE = "TRUSTED_EVIDENCE"
    HUMAN_ANNOTATION = "HUMAN_ANNOTATION"
    LLM_GENERATED = "LLM_GENERATED"


_GROUND_TRUTH_PROVENANCE = frozenset(
    {
        ProvenanceType.DETERMINISTIC_RULE,
        ProvenanceType.TRUSTED_EVIDENCE,
        ProvenanceType.HUMAN_ANNOTATION,
    }
)


@dataclass(frozen=True)
class TransformationProvenance:
    provenance_id: str
    source_type: ProvenanceType
    source_ref: str
    verified: bool
    description: str = ""

    def __post_init__(self) -> None:
        if not self.provenance_id.strip():
            raise ValueError("provenance_id cannot be empty")
        if not self.source_ref.strip():
            raise ValueError("source_ref cannot be empty")

    @property
    def establishes_ground_truth(self) -> bool:
        return self.verified and self.source_type in _GROUND_TRUTH_PROVENANCE


@dataclass(frozen=True)
class DecisiveFactChange:
    fact_id: str
    original_fact: str
    transformed_fact: str
    legal_effect: str

    def __post_init__(self) -> None:
        for field_name in (
            "fact_id",
            "original_fact",
            "transformed_fact",
            "legal_effect",
        ):
            if not getattr(self, field_name).strip():
                raise ValueError(f"{field_name} cannot be empty")
        if self.original_fact.strip() == self.transformed_fact.strip():
            raise ValueError("decisive fact transformation must change the fact")


@dataclass(frozen=True)
class CounterfactualTransformation:
    transformation_id: str
    transformation_type: str
    original_question: str
    transformed_question: str
    original_answer: str
    expected_answer: str
    expectation: TransformationExpectation
    provenance: tuple[TransformationProvenance, ...]
    decisive_fact: DecisiveFactChange | None = None
    displayed_to_original: Mapping[str, str] = field(default_factory=dict)
    generated: bool = False

    def __post_init__(self) -> None:
        if not self.transformation_id.strip():
            raise ValueError("transformation_id cannot be empty")
        if not self.transformation_type.strip():
            raise ValueError("transformation_type cannot be empty")
        if not self.original_question.strip() or not self.transformed_question.strip():
            raise ValueError("counterfactual questions cannot be empty")
        original = normalize_option_label(self.original_answer, "original_answer")
        expected = normalize_option_label(self.expected_answer, "expected_answer")
        object.__setattr__(self, "original_answer", original)
        object.__setattr__(self, "expected_answer", expected)
        if self.expectation is TransformationExpectation.PRESERVE_LABEL:
            if expected != original:
                raise ValueError("label-preserving transformation must preserve answer")
            if self.decisive_fact is not None:
                raise ValueError(
                    "label-preserving transformation cannot contain decisive_fact"
                )
        else:
            if expected == original:
                raise ValueError("label-changing transformation must change answer")
            if self.decisive_fact is None:
                raise ValueError(
                    "label-changing transformation requires decisive_fact"
                )
        invalid_labels = (
            set(self.displayed_to_original)
            | set(self.displayed_to_original.values())
        ) - set(OPTION_LABELS)
        if invalid_labels:
            raise ValueError(
                f"displayed_to_original has invalid labels: {sorted(invalid_labels)}"
            )

    @property
    def is_metric_eligible(self) -> bool:
        if self.expectation is TransformationExpectation.PRESERVE_LABEL:
            return True
        return any(item.establishes_ground_truth for item in self.provenance)

    def map_prediction_to_original(self, answer: str) -> str:
        normalized = normalize_option_label(answer, "counterfactual prediction")
        return self.displayed_to_original.get(normalized, normalized)


@dataclass(frozen=True)
class CounterfactualConfig:
    enable_generative: bool = False


def _normalization_provenance(
    transformation_id: str,
) -> tuple[TransformationProvenance, ...]:
    return (
        TransformationProvenance(
            provenance_id=f"{transformation_id}:rule",
            source_type=ProvenanceType.DETERMINISTIC_RULE,
            source_ref="unicode-nfkc-and-whitespace-normalization-v1",
            verified=True,
            description="Only presentation whitespace and Unicode forms are normalized.",
        ),
    )


def harmless_text_normalization(
    question: str,
    *,
    original_answer: str,
    transformation_id: str = "normalize-1",
) -> CounterfactualTransformation:
    """Canonicalize presentation without changing option identity or legal facts."""
    parsed = parse_four_option_question(question)
    stem = " ".join(unicodedata.normalize("NFKC", parsed.stem).split())
    lines = [stem, ""]
    lines.extend(
        f"{option.original_label}. "
        + " ".join(unicodedata.normalize("NFKC", option.text).split())
        for option in parsed.options
    )
    normalized_answer = normalize_option_label(original_answer, "original_answer")
    return CounterfactualTransformation(
        transformation_id=transformation_id,
        transformation_type="HARMLESS_TEXT_NORMALIZATION",
        original_question=question,
        transformed_question="\n".join(lines),
        original_answer=normalized_answer,
        expected_answer=normalized_answer,
        expectation=TransformationExpectation.PRESERVE_LABEL,
        provenance=_normalization_provenance(transformation_id),
        displayed_to_original={label: label for label in OPTION_LABELS},
    )


def _permutation_transformation(
    question: str,
    original_answer: str,
    permutation: OptionPermutation,
) -> CounterfactualTransformation:
    return CounterfactualTransformation(
        transformation_id=permutation.permutation_id,
        transformation_type="OPTION_REORDER",
        original_question=question,
        transformed_question=permutation.question,
        original_answer=original_answer,
        expected_answer=original_answer,
        expectation=TransformationExpectation.PRESERVE_LABEL,
        provenance=(
            TransformationProvenance(
                provenance_id=f"{permutation.permutation_id}:rule",
                source_type=ProvenanceType.DETERMINISTIC_RULE,
                source_ref="stable-option-text-identity-v1",
                verified=True,
                description="Option text identity is mapped back after deterministic reorder.",
            ),
        ),
        displayed_to_original=permutation.displayed_to_original,
    )


def option_reorder_transformations(
    question: str,
    *,
    original_answer: str,
    count: int,
    seed: int,
) -> tuple[CounterfactualTransformation, ...]:
    """Create seeded label-preserving option reorder transformations."""
    normalized_answer = normalize_option_label(original_answer, "original_answer")
    parsed = parse_four_option_question(question)
    return tuple(
        _permutation_transformation(question, normalized_answer, permutation)
        for permutation in generate_option_permutations(
            parsed,
            count=count,
            seed=seed,
        )
    )


def decisive_fact_counterfactual(
    *,
    transformation_id: str,
    original_question: str,
    transformed_question: str,
    original_answer: str,
    expected_answer: str,
    decisive_fact: DecisiveFactChange,
    provenance: Iterable[TransformationProvenance],
    generated: bool = False,
) -> CounterfactualTransformation:
    """Construct a label-changing record; eligibility is derived from provenance."""
    return CounterfactualTransformation(
        transformation_id=transformation_id,
        transformation_type="DECISIVE_FACT_CHANGE",
        original_question=original_question,
        transformed_question=transformed_question,
        original_answer=original_answer,
        expected_answer=expected_answer,
        expectation=TransformationExpectation.CHANGE_LABEL,
        provenance=tuple(provenance),
        decisive_fact=decisive_fact,
        generated=generated,
    )


def enabled_transformations(
    transformations: Iterable[CounterfactualTransformation],
    config: CounterfactualConfig | None = None,
) -> tuple[CounterfactualTransformation, ...]:
    """Apply the default-off generative gate without changing metric eligibility."""
    resolved = config or CounterfactualConfig()
    return tuple(
        transformation
        for transformation in transformations
        if resolved.enable_generative or not transformation.generated
    )


@dataclass(frozen=True)
class CounterfactualObservation:
    transformation: CounterfactualTransformation
    predicted_answer: str

    def __post_init__(self) -> None:
        normalized = self.transformation.map_prediction_to_original(
            self.predicted_answer
        )
        object.__setattr__(self, "predicted_answer", normalized)

    @property
    def consistent(self) -> bool:
        return self.predicted_answer == self.transformation.expected_answer


@dataclass(frozen=True)
class CounterfactualMetrics:
    sensitivity: float | None
    specificity: float | None
    should_change_total: int
    should_change_correct: int
    should_not_change_total: int
    should_not_change_correct: int
    excluded_unverified_label_changing: int


def calculate_counterfactual_metrics(
    observations: Iterable[CounterfactualObservation],
) -> CounterfactualMetrics:
    """Calculate should-change sensitivity and should-not-change specificity."""
    should_change_total = 0
    should_change_correct = 0
    should_not_change_total = 0
    should_not_change_correct = 0
    excluded = 0

    for observation in observations:
        transformation = observation.transformation
        if transformation.expectation is TransformationExpectation.CHANGE_LABEL:
            if not transformation.is_metric_eligible:
                excluded += 1
                continue
            should_change_total += 1
            should_change_correct += int(observation.consistent)
        else:
            should_not_change_total += 1
            should_not_change_correct += int(observation.consistent)

    return CounterfactualMetrics(
        sensitivity=(
            should_change_correct / should_change_total
            if should_change_total
            else None
        ),
        specificity=(
            should_not_change_correct / should_not_change_total
            if should_not_change_total
            else None
        ),
        should_change_total=should_change_total,
        should_change_correct=should_change_correct,
        should_not_change_total=should_not_change_total,
        should_not_change_correct=should_not_change_correct,
        excluded_unverified_label_changing=excluded,
    )
