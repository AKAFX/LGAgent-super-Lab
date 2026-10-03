"""Deterministic no-harm policy for Direct-anchor disagreements."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Sequence

from .models import (
    DirectAnchorDecision,
    LegalEvidence,
    SolverDecision,
    SolveMode,
)

_NON_DECISIVE_SOURCE_TYPES = frozenset(
    {
        "search_snippet",
        "search-summary",
        "search_summary",
        "snippet",
    }
)


@dataclass(frozen=True)
class NoHarmGateResult:
    action: str
    anchor_selected_options: tuple[str, ...]
    candidate_selected_options: tuple[str, ...]
    authoritative_evidence_ids: tuple[str, ...]
    reason: str

    @property
    def preserve_anchor(self) -> bool:
        return self.action == "preserve_anchor"


class NoHarmGate:
    """Allow an anchor override only when every changed option is evidenced."""

    def __init__(self, *, min_authority_level: int = 4) -> None:
        if not 0 <= min_authority_level <= 5:
            raise ValueError("min_authority_level must be between zero and five")
        self.min_authority_level = min_authority_level

    def evaluate(
        self,
        *,
        anchor: DirectAnchorDecision,
        candidate: SolverDecision,
        evidence: Sequence[LegalEvidence],
        effective_mode: SolveMode,
        as_of_date: date,
        candidate_valid: bool = True,
        verifier_accepted: bool = True,
    ) -> NoHarmGateResult:
        anchor_options = tuple(anchor.selected_options)
        candidate_options = tuple(candidate.selected_options)
        if set(anchor_options) == set(candidate_options):
            return NoHarmGateResult(
                action="matched",
                anchor_selected_options=anchor_options,
                candidate_selected_options=candidate_options,
                authoritative_evidence_ids=(),
                reason="structured candidate matches the independent Direct anchor",
            )
        if effective_mode is not SolveMode.OPEN_BOOK:
            return NoHarmGateResult(
                action="preserve_anchor",
                anchor_selected_options=anchor_options,
                candidate_selected_options=candidate_options,
                authoritative_evidence_ids=(),
                reason=(
                    "closed-book disagreement has no audited authority capable "
                    "of overriding the Direct anchor"
                ),
            )
        if not candidate_valid or not verifier_accepted:
            failed_checks = []
            if not candidate_valid:
                failed_checks.append("deterministic validation")
            if not verifier_accepted:
                failed_checks.append("independent verification")
            return NoHarmGateResult(
                action="preserve_anchor",
                anchor_selected_options=anchor_options,
                candidate_selected_options=candidate_options,
                authoritative_evidence_ids=(),
                reason=(
                    "structured candidate failed "
                    + " and ".join(failed_checks)
                ),
            )

        evidence_by_id = {item.evidence_id: item for item in evidence}
        assessments = {item.label: item for item in candidate.option_assessments}
        changed_labels = set(anchor_options).symmetric_difference(candidate_options)
        decisive_ids: set[str] = set()
        for label in sorted(changed_labels):
            assessment = assessments.get(label)
            if assessment is None:
                return self._preserve_for_missing_evidence(
                    anchor_options,
                    candidate_options,
                    decisive_ids,
                    label,
                )
            valid_for_label = {
                evidence_id
                for claim in assessment.claims
                for evidence_id in claim.evidence_ids
                if (
                    evidence_id in evidence_by_id
                    and evidence_by_id[evidence_id].authority_level
                    >= self.min_authority_level
                    and evidence_by_id[evidence_id].covers(as_of_date)
                    and evidence_by_id[evidence_id].source_type.strip().lower()
                    not in _NON_DECISIVE_SOURCE_TYPES
                )
            }
            if not valid_for_label:
                return self._preserve_for_missing_evidence(
                    anchor_options,
                    candidate_options,
                    decisive_ids,
                    label,
                )
            decisive_ids.update(valid_for_label)

        return NoHarmGateResult(
            action="allow_override",
            anchor_selected_options=anchor_options,
            candidate_selected_options=candidate_options,
            authoritative_evidence_ids=tuple(sorted(decisive_ids)),
            reason=(
                "every changed option cites authoritative in-force full-text "
                "evidence"
            ),
        )

    @staticmethod
    def _preserve_for_missing_evidence(
        anchor_options: tuple[str, ...],
        candidate_options: tuple[str, ...],
        decisive_ids: set[str],
        label: str,
    ) -> NoHarmGateResult:
        return NoHarmGateResult(
            action="preserve_anchor",
            anchor_selected_options=anchor_options,
            candidate_selected_options=candidate_options,
            authoritative_evidence_ids=tuple(sorted(decisive_ids)),
            reason=(
                f"changed option {label} lacks authoritative in-force "
                "full-text evidence"
            ),
        )
