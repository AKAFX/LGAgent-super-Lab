"""Oracle-isolated evaluation utilities for LegalMCQ."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

from .adapter import request_only
from .models import EvaluationCase, LegalMCQRunResult


@dataclass(frozen=True)
class LegalMCQEvaluationRecord:
    question_id: str
    prediction: tuple[str, ...]
    golden_answers: tuple[str, ...]
    exact_match: bool
    status: str
    needs_review: bool
    all_options_covered: bool
    verifier_accepted: bool
    revision_count: int

    def as_dict(self) -> dict[str, object]:
        return {
            "question_id": self.question_id,
            "prediction": list(self.prediction),
            "golden_answers": list(self.golden_answers),
            "exact_match": self.exact_match,
            "status": self.status,
            "needs_review": self.needs_review,
            "all_options_covered": self.all_options_covered,
            "verifier_accepted": self.verifier_accepted,
            "revision_count": self.revision_count,
        }


def evaluate_case(
    case: EvaluationCase,
    solve: Callable[..., LegalMCQRunResult],
    *,
    on_agent_completed: Callable[[], None] | None = None,
) -> LegalMCQEvaluationRecord:
    """Run the agent before accessing the private evaluation oracle."""
    result = solve(request_only(case))
    if on_agent_completed is not None:
        on_agent_completed()

    # The oracle is intentionally first read after the solve call returns.
    golden = tuple(sorted(case.oracle.golden_answers))
    prediction = tuple(sorted(result.answer.selected_options))
    expected_labels = {option.label for option in case.request.options}
    covered_labels = {
        option.label for option in result.answer.option_assessments
    }
    return LegalMCQEvaluationRecord(
        question_id=case.request.question_id,
        prediction=prediction,
        golden_answers=golden,
        exact_match=prediction == golden,
        status=result.answer.status.value,
        needs_review=result.answer.needs_review,
        all_options_covered=covered_labels == expected_labels,
        verifier_accepted=result.verification.accepted,
        revision_count=result.revision_count,
    )


def summarize_records(
    records: Sequence[LegalMCQEvaluationRecord],
) -> dict[str, float | int]:
    total = len(records)
    if total == 0:
        return {
            "total": 0,
            "exact_match": 0.0,
            "option_coverage_rate": 0.0,
            "verifier_acceptance_rate": 0.0,
            "needs_review_rate": 0.0,
        }
    return {
        "total": total,
        "exact_match": sum(item.exact_match for item in records) / total,
        "option_coverage_rate": (
            sum(item.all_options_covered for item in records) / total
        ),
        "verifier_acceptance_rate": (
            sum(item.verifier_accepted for item in records) / total
        ),
        "needs_review_rate": sum(item.needs_review for item in records) / total,
    }
