"""Adapters from audited LGAgent evidence into the LegalMCQ contract."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Sequence
from urllib.parse import urlparse

from ..evidence_audit import (
    AuditedEvidence,
    AuditedEvidenceMatrix,
    EvidenceAuditPipeline,
)
from ..trace import RunTrace
from .models import (
    ControllerPlan,
    LegalEvidence,
    LegalQuestionRequest,
    ParsedLegalQuestion,
)


class LegalEvidenceAdapterError(ValueError):
    """Raised when audited evidence cannot satisfy LegalMCQ provenance rules."""


def _adapt_evidence(
    item: AuditedEvidence,
    *,
    evidence_id: str,
) -> LegalEvidence:
    parsed_url = urlparse(item.source_uri)
    if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
        raise LegalEvidenceAdapterError(
            f"{item.evidence_id}: decisive evidence must have an HTTP source URI"
        )
    article = item.article.strip()
    return LegalEvidence(
        evidence_id=evidence_id,
        title=f"{item.law_name} {article}".strip(),
        url=item.source_uri,
        publisher=parsed_url.netloc,
        source_type=item.source_type,
        authority_level=item.authority_level,
        quote=item.exact_span,
        law_name=item.law_name,
        article_number=article or None,
        effective_from=item.effective_from,
        effective_until=item.effective_to,
    )


def adapt_audited_evidence(
    matrix: AuditedEvidenceMatrix,
) -> tuple[LegalEvidence, ...]:
    """Convert relevant audited full-text spans and reject provenance conflicts."""
    grouped: dict[str, list[AuditedEvidence]] = {}
    for option in sorted(matrix.options):
        audit = matrix.options[option]
        relevant: Sequence[AuditedEvidence] = (
            *audit.support,
            *audit.refute,
            *audit.exception,
        )
        for item in relevant:
            grouped.setdefault(item.evidence_id, []).append(item)

    converted: list[LegalEvidence] = []
    for source_id in sorted(grouped):
        items = grouped[source_id]
        first = items[0]
        provenance = (
            first.source_type,
            first.law_name,
            first.article,
            first.clause,
            first.authority_level,
            first.effective_from,
            first.effective_to,
            first.source_uri,
        )
        if any(
            (
                item.source_type,
                item.law_name,
                item.article,
                item.clause,
                item.authority_level,
                item.effective_from,
                item.effective_to,
                item.source_uri,
            )
            != provenance
            for item in items[1:]
        ):
            raise LegalEvidenceAdapterError(
                f"{source_id}: conflicting audited evidence records"
            )
        by_span = {item.exact_span: item for item in items}
        for span in sorted(by_span):
            item = by_span[span]
            evidence_id = source_id
            if len(by_span) > 1:
                span_hash = hashlib.sha256(span.encode("utf-8")).hexdigest()[:12]
                evidence_id = f"{source_id}:span:{span_hash}"
            candidate = _adapt_evidence(item, evidence_id=evidence_id)
            if any(
                existing.evidence_id == candidate.evidence_id
                for existing in converted
            ):
                raise LegalEvidenceAdapterError(
                    f"{candidate.evidence_id}: conflicting audited evidence records"
                )
            converted.append(candidate)
    return tuple(sorted(converted, key=lambda item: item.evidence_id))


@dataclass
class OathAuditedEvidenceProvider:
    """Run the existing OATH pipeline and expose only audited full-text evidence."""

    pipeline: EvidenceAuditPipeline

    def __call__(
        self,
        request: LegalQuestionRequest,
        parsed: ParsedLegalQuestion,
        plan: ControllerPlan,
        trace: RunTrace,
    ) -> tuple[LegalEvidence, ...]:
        option_claims = {
            label: {"claim": "; ".join(claims)}
            for label, claims in plan.option_claims.items()
        }
        matrix = self.pipeline.run(
            {"option_claims": option_claims},
            jurisdiction=request.jurisdiction,
            case_date=parsed.request.as_of_date,
            trace=trace,
        )
        evidence = adapt_audited_evidence(matrix)
        trace.add_route(
            "legal-mcq-oath-adapter",
            "OATH audit matrix converted to LegalMCQ full-text evidence",
            {
                "retrieval_status": matrix.retrieval_status.value,
                "option_count": len(matrix.options),
                "evidence_count": len(evidence),
            },
        )
        return evidence
