"""Deterministic semantic checks around model-generated LegalMCQ decisions."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Sequence

from .models import (
    CONTROLLER_PROTOCOL_VERSION,
    CONTROLLER_PROTOCOL_V2_VERSION,
    Citation,
    ControllerPlan,
    LegalEvidence,
    ParsedLegalQuestion,
    QuestionType,
    SOLVER_PROTOCOL_VERSION,
    SolverDecision,
    SolveMode,
)


@dataclass(frozen=True)
class DecisionValidation:
    valid: bool
    error_codes: tuple[str, ...]
    warnings: tuple[str, ...] = ()


class DecisionValidator:
    """Enforce invariants that must never be delegated to an LLM."""

    def __init__(self, *, min_authority_level: int = 4) -> None:
        if not 0 <= min_authority_level <= 5:
            raise ValueError("min_authority_level must be between 0 and 5")
        self.min_authority_level = min_authority_level

    def validate(
        self,
        parsed: ParsedLegalQuestion,
        plan: ControllerPlan,
        decision: SolverDecision,
        evidence: Sequence[LegalEvidence],
        *,
        effective_mode: SolveMode,
    ) -> DecisionValidation:
        errors: list[str] = []
        warnings: list[str] = list(parsed.parse_warnings)
        expected = set(parsed.option_labels)
        observed = [item.label for item in decision.option_assessments]
        selected = decision.selected_options

        if len(observed) != len(set(observed)) or set(observed) != expected:
            errors.append("OPTION_COVERAGE_MISMATCH")
        if not set(selected).issubset(expected):
            errors.append("INVALID_SELECTED_OPTION")
        if len(selected) != len(set(selected)):
            errors.append("DUPLICATE_SELECTED_OPTION")
        if (
            parsed.request.question_type is QuestionType.SINGLE_CHOICE
            and len(selected) != 1
        ):
            errors.append("SINGLE_CHOICE_CARDINALITY")
        if (
            parsed.request.question_type is QuestionType.MULTIPLE_CHOICE
            and not selected
        ):
            errors.append("MULTIPLE_CHOICE_EMPTY")

        assessments = {item.label: item for item in decision.option_assessments}
        expected_issue_ids = {item.issue_id for item in plan.issues}
        observed_issue_ids = [item.issue_id for item in decision.issue_analyses]
        if decision.raw.get("protocol_version") != SOLVER_PROTOCOL_VERSION and (
            len(observed_issue_ids) != len(set(observed_issue_ids))
            or not expected_issue_ids.issubset(observed_issue_ids)
        ):
            errors.append("ISSUE_COVERAGE_MISMATCH")
        claim_ids: set[str] = set()
        for label, expected_claims in plan.option_claims.items():
            assessment = assessments.get(label)
            if assessment is None:
                continue
            if len(assessment.claims) < len(expected_claims):
                errors.append("CONTROLLER_CLAIM_COVERAGE_MISMATCH")
            if plan.raw.get("protocol_version") in {
                CONTROLLER_PROTOCOL_VERSION,
                CONTROLLER_PROTOCOL_V2_VERSION,
            }:
                expected_ids = {f"{label}{i}" for i in range(1, len(expected_claims) + 1)}
                if {claim.claim_id for claim in assessment.claims} != expected_ids:
                    errors.append("CONTROLLER_CLAIM_ID_MISMATCH")
            for claim in assessment.claims:
                if claim.claim_id in claim_ids:
                    errors.append("DUPLICATE_CLAIM_ID")
                claim_ids.add(claim.claim_id)
                if not claim.claim_id.upper().startswith(label):
                    errors.append("CLAIM_ID_OPTION_MISMATCH")
        target_verdict = (
            "contradicted" if parsed.asks_for_incorrect_option else "supported"
        )
        for label in selected:
            item = assessments.get(label)
            if item is not None and item.verdict != target_verdict:
                errors.append("QUESTION_POLARITY_MISMATCH")
        matching = [
            item.label
            for item in decision.option_assessments
            if item.verdict == target_verdict
        ]
        if (
            parsed.request.question_type is QuestionType.SINGLE_CHOICE
            and len(matching) != 1
        ):
            errors.append("AMBIGUOUS_OPTION_VERDICTS")
        if (
            parsed.request.question_type is QuestionType.MULTIPLE_CHOICE
            and set(selected) != set(matching)
        ):
            errors.append("MULTIPLE_CHOICE_SELECTION_MISMATCH")

        evidence_by_id = {item.evidence_id: item for item in evidence}
        cited_ids = {
            evidence_id
            for option in decision.option_assessments
            for claim in option.claims
            for evidence_id in claim.evidence_ids
        }
        if cited_ids - set(evidence_by_id):
            errors.append("UNKNOWN_EVIDENCE_ID")
        if effective_mode is SolveMode.CLOSED_BOOK and cited_ids:
            errors.append("CLOSED_BOOK_EXTERNAL_CITATION")
        if effective_mode is SolveMode.OPEN_BOOK:
            as_of_date = parsed.request.as_of_date or date.today()
            authoritative = [
                item
                for item in evidence
                if item.authority_level >= self.min_authority_level
                and item.covers(as_of_date)
            ]
            if not authoritative:
                errors.append("NO_AUTHORITATIVE_SOURCE")
            if not cited_ids:
                errors.append("OPEN_BOOK_MISSING_CITATION")
            for evidence_id in cited_ids.intersection(evidence_by_id):
                item = evidence_by_id[evidence_id]
                if item.authority_level < self.min_authority_level:
                    errors.append("CITED_SOURCE_NOT_AUTHORITATIVE")
                if not item.covers(as_of_date):
                    errors.append("SOURCE_VERSION_MISMATCH")
                if item.source_type.strip().lower() in {
                    "search_snippet",
                    "search-summary",
                    "search_summary",
                    "snippet",
                }:
                    errors.append("SEARCH_SNIPPET_NOT_DECISIVE")
            for label in selected:
                assessment = assessments.get(label)
                if assessment is None:
                    continue
                selected_citations = {
                    evidence_id
                    for claim in assessment.claims
                    for evidence_id in claim.evidence_ids
                }
                if not selected_citations:
                    errors.append("SELECTED_OPTION_MISSING_CITATION")
                    continue
                if not any(
                    evidence_id in evidence_by_id
                    and evidence_by_id[evidence_id].authority_level
                    >= self.min_authority_level
                    and evidence_by_id[evidence_id].covers(as_of_date)
                    and evidence_by_id[evidence_id].source_type.strip().lower()
                    not in {
                        "search_snippet",
                        "search-summary",
                        "search_summary",
                        "snippet",
                    }
                    for evidence_id in selected_citations
                ):
                    errors.append(
                        "SELECTED_OPTION_NO_AUTHORITATIVE_CITATION"
                    )

        for option in decision.option_assessments:
            if not option.claims:
                errors.append("COMPOUND_OPTION_NOT_DECOMPOSED")
                continue
            claim_verdicts = {claim.verdict for claim in option.claims}
            if option.verdict == "supported" and claim_verdicts != {"supported"}:
                errors.append("CLAIM_OPTION_VERDICT_CONFLICT")
            if (
                option.verdict == "contradicted"
                and "contradicted" not in claim_verdicts
            ):
                errors.append("CLAIM_OPTION_VERDICT_CONFLICT")

        return DecisionValidation(
            valid=not errors,
            error_codes=tuple(dict.fromkeys(errors)),
            warnings=tuple(dict.fromkeys(warnings)),
        )


def citations_for_decision(
    decision: SolverDecision,
    evidence: Sequence[LegalEvidence],
) -> tuple[Citation, ...]:
    cited_ids = {
        evidence_id
        for option in decision.option_assessments
        for claim in option.claims
        for evidence_id in claim.evidence_ids
    }
    return tuple(
        Citation(
            citation_id=item.evidence_id,
            title=item.title,
            url=item.url,
            publisher=item.publisher,
            article_number=item.article_number,
            quote=item.quote,
        )
        for item in evidence
        if item.evidence_id in cited_ids
    )
