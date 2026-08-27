"""CPU-first option-aligned hybrid retrieval for OATH-RAG."""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date
from enum import Enum
from typing import Any, Mapping, Protocol, Sequence

from .corpus import LegalEvidence, LoadedCorpus, RelationType

OPTION_LABELS = ("A", "B", "C", "D")
_ASCII_TOKEN = re.compile(r"[a-z0-9]+")
_CJK_RUN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]+")


class EvidenceLane(str, Enum):
    SUPPORT = "support"
    REFUTE = "refute"
    EXCEPTION = "exception"


@dataclass(frozen=True)
class OptionQueries:
    support: str
    refute: str
    exception: str

    def for_lane(self, lane: EvidenceLane) -> str:
        return getattr(self, lane.value)


@dataclass(frozen=True)
class DenseHit:
    evidence_id: str
    score: float


class DenseRetriever(Protocol):
    """Injectable dense search that may run locally or remotely."""

    def search(
        self,
        query: str,
        *,
        candidate_ids: Sequence[str],
        top_k: int,
    ) -> Sequence[DenseHit]:
        ...


@dataclass(frozen=True)
class OathRagConfig:
    lexical_top_k: int = 20
    dense_top_k: int = 20
    final_top_k_per_lane: int = 3
    rrf_k: int = 60
    min_authority_level: int = 1
    graph_hops: int = 1
    enabled_lanes: tuple[str, ...] = tuple(lane.value for lane in EvidenceLane)
    require_temporal_match: bool = True
    graph_decay: float = 0.75
    coverage_target: float = 0.8

    def __post_init__(self) -> None:
        for field in (
            "lexical_top_k",
            "dense_top_k",
            "final_top_k_per_lane",
            "rrf_k",
        ):
            if getattr(self, field) <= 0:
                raise ValueError(f"{field} must be positive")
        if self.min_authority_level < 0:
            raise ValueError("min_authority_level cannot be negative")
        if self.graph_hops not in (0, 1):
            raise ValueError("the first OATH-RAG implementation supports 0 or 1 graph hop")
        if (
            not self.enabled_lanes
            or len(set(self.enabled_lanes)) != len(self.enabled_lanes)
            or not set(self.enabled_lanes).issubset(
                {lane.value for lane in EvidenceLane}
            )
        ):
            raise ValueError("enabled_lanes must contain unique OATH-RAG lanes")
        if not isinstance(self.require_temporal_match, bool):
            raise ValueError("require_temporal_match must be a boolean")
        if not 0.0 <= self.graph_decay <= 1.0:
            raise ValueError("graph_decay must be between 0 and 1")
        if not 0.0 < self.coverage_target <= 1.0:
            raise ValueError("coverage_target must be in (0, 1]")


@dataclass(frozen=True)
class RetrievedEvidence:
    evidence: LegalEvidence
    score: float
    lexical_score: float = 0.0
    dense_score: float = 0.0
    relation_support: float = 0.0
    expanded_from: str | None = None
    relation_type: RelationType | None = None

    @property
    def evidence_id(self) -> str:
        return self.evidence.evidence_id


EvidenceMatrix = dict[str, dict[str, tuple[RetrievedEvidence, ...]]]


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return " ".join(unicodedata.normalize("NFKC", value).split())
    return _text(str(value))


def _field(value: Mapping[str, Any] | object, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _string_items(value: Any) -> list[str]:
    if isinstance(value, str):
        return [_text(value)] if _text(value) else []
    if not isinstance(value, Sequence) or isinstance(value, (bytes, bytearray)):
        return []
    result: list[str] = []
    for item in value:
        if isinstance(item, Mapping):
            if item.get("legally_relevant") is False:
                continue
            item = item.get("text", "")
        normalized = _text(item)
        if normalized:
            result.append(normalized)
    return result


def build_option_queries(
    analysis: Mapping[str, Any] | object,
) -> dict[str, OptionQueries]:
    """Build answer-neutral support, refute, and exception queries per option."""
    raw_claims = _field(analysis, "option_claims")
    if not isinstance(raw_claims, Mapping) or not raw_claims:
        raise ValueError("analysis.option_claims must be a non-empty mapping")

    context = [
        _text(_field(analysis, "legal_domain")),
        _text(_field(analysis, "question_focus")),
        _text(_field(analysis, "jurisdiction")),
        _text(_field(analysis, "case_date")),
    ]
    facts = _field(analysis, "facts", ())
    context.extend(_string_items(facts))
    context_text = " ".join(part for part in context if part)

    result: dict[str, OptionQueries] = {}
    for raw_option, raw_claim in sorted(raw_claims.items()):
        option = _text(raw_option).upper()
        if not option:
            raise ValueError("option label must not be empty")
        if isinstance(raw_claim, Mapping):
            claim = _text(raw_claim.get("claim"))
            elements = _string_items(raw_claim.get("elements", ()))
            exceptions = _string_items(raw_claim.get("possible_exceptions", ()))
        elif hasattr(raw_claim, "claim"):
            claim = _text(getattr(raw_claim, "claim"))
            elements = _string_items(getattr(raw_claim, "elements", ()))
            exceptions = _string_items(
                getattr(raw_claim, "possible_exceptions", ())
            )
        else:
            claim = _text(raw_claim)
            elements = []
            exceptions = []
        if not claim:
            raise ValueError(f"option {option} claim must not be empty")

        core = " ".join(part for part in (context_text, claim, *elements) if part)
        exception_hints = " ".join(exceptions)
        result[option] = OptionQueries(
            support=f"{core} 适用 构成要件 法律依据 支持",
            refute=f"{core} 不适用 不成立 排除 缺少要件 反驳",
            exception=" ".join(
                part
                for part in (
                    core,
                    exception_hints,
                    "例外 除外 但书 特别规定",
                )
                if part
            ),
        )
    return result


def _tokenize(text: str) -> tuple[str, ...]:
    normalized = unicodedata.normalize("NFKC", text).lower()
    tokens = _ASCII_TOKEN.findall(normalized)
    for run in _CJK_RUN.findall(normalized):
        tokens.extend(run)
        tokens.extend(run[index : index + 2] for index in range(len(run) - 1))
    return tuple(tokens)


class BM25Index:
    """Small dependency-free BM25 index suitable for CPU-only corpora."""

    def __init__(
        self,
        evidence: Sequence[LegalEvidence],
        *,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        if k1 <= 0:
            raise ValueError("k1 must be positive")
        if not 0.0 <= b <= 1.0:
            raise ValueError("b must be between 0 and 1")
        self._evidence = {item.evidence_id: item for item in evidence}
        self._k1 = k1
        self._b = b
        self._term_frequencies: dict[str, Counter[str]] = {}
        self._lengths: dict[str, int] = {}
        document_frequency: Counter[str] = Counter()
        for item in evidence:
            tokens = _tokenize(
                " ".join(
                    part
                    for part in (item.law_name, item.article, item.clause, item.text)
                    if part
                )
            )
            frequencies = Counter(tokens)
            self._term_frequencies[item.evidence_id] = frequencies
            self._lengths[item.evidence_id] = len(tokens)
            document_frequency.update(frequencies)
        count = max(len(evidence), 1)
        self._average_length = sum(self._lengths.values()) / count
        self._idf = {
            term: math.log(1.0 + (count - frequency + 0.5) / (frequency + 0.5))
            for term, frequency in document_frequency.items()
        }

    def search(
        self,
        query: str,
        *,
        candidate_ids: Sequence[str],
        top_k: int,
    ) -> tuple[DenseHit, ...]:
        query_terms = Counter(_tokenize(query))
        scores: list[DenseHit] = []
        for evidence_id in candidate_ids:
            frequencies = self._term_frequencies.get(evidence_id)
            if not frequencies:
                continue
            length = self._lengths[evidence_id]
            normalization = 1.0 - self._b
            if self._average_length:
                normalization += self._b * length / self._average_length
            score = 0.0
            for term, query_frequency in query_terms.items():
                frequency = frequencies.get(term, 0)
                if not frequency:
                    continue
                numerator = frequency * (self._k1 + 1.0)
                denominator = frequency + self._k1 * normalization
                score += (
                    self._idf.get(term, 0.0)
                    * numerator
                    / denominator
                    * (1.0 + math.log(query_frequency))
                )
            if score > 0:
                scores.append(DenseHit(evidence_id, score))
        scores.sort(key=lambda hit: (-hit.score, hit.evidence_id))
        return tuple(scores[:top_k])


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[DenseHit]],
    *,
    rrf_k: int = 60,
) -> dict[str, float]:
    if rrf_k <= 0:
        raise ValueError("rrf_k must be positive")
    fused: defaultdict[str, float] = defaultdict(float)
    for ranking in rankings:
        seen: set[str] = set()
        for rank, hit in enumerate(ranking, start=1):
            if hit.evidence_id in seen:
                continue
            seen.add(hit.evidence_id)
            fused[hit.evidence_id] += 1.0 / (rrf_k + rank)
    return dict(fused)


class OathRagRetriever:
    """Retrieve option-aligned legal evidence with deterministic hard gates."""

    def __init__(
        self,
        corpus: LoadedCorpus | Sequence[LegalEvidence],
        *,
        dense_retriever: DenseRetriever | None = None,
        config: OathRagConfig | None = None,
        current_date: date | None = None,
    ) -> None:
        evidence = corpus.evidence if isinstance(corpus, LoadedCorpus) else tuple(corpus)
        if not evidence:
            raise ValueError("corpus must not be empty")
        self.config = config or OathRagConfig()
        self.dense_retriever = dense_retriever
        self.current_date = current_date or date.today()
        self._evidence = {item.evidence_id: item for item in evidence}
        self._bm25 = BM25Index(evidence)
        self._neighbors = self._build_neighbors(evidence)

    @staticmethod
    def _build_neighbors(
        evidence: Sequence[LegalEvidence],
    ) -> dict[str, tuple[tuple[str, RelationType], ...]]:
        known = {item.evidence_id for item in evidence}
        neighbors: defaultdict[str, set[tuple[str, RelationType]]] = defaultdict(set)
        for item in evidence:
            for relation in item.relations:
                if relation.target_id not in known:
                    continue
                neighbors[item.evidence_id].add((relation.target_id, relation.type))
                neighbors[relation.target_id].add((item.evidence_id, relation.type))
        return {
            evidence_id: tuple(sorted(items, key=lambda pair: (pair[0], pair[1].value)))
            for evidence_id, items in neighbors.items()
        }

    def _eligible(
        self,
        *,
        jurisdiction: str,
        case_date: date | None,
    ) -> tuple[LegalEvidence, ...]:
        normalized_jurisdiction = _text(jurisdiction).upper()
        if not normalized_jurisdiction:
            raise ValueError("jurisdiction must not be empty")
        applicable_date = case_date or self.current_date
        return tuple(
            item
            for item in self._evidence.values()
            if item.jurisdiction == normalized_jurisdiction
            and item.authority_level >= self.config.min_authority_level
            and bool(item.evidence_id and item.text and item.source_uri)
            and (
                not self.config.require_temporal_match
                or
                item.effective_from is None
                or item.effective_from <= applicable_date
            )
            and (
                not self.config.require_temporal_match
                or
                item.effective_to is None
                or applicable_date <= item.effective_to
            )
        )

    def retrieve(
        self,
        analysis: Mapping[str, Any] | object,
        *,
        jurisdiction: str | None = None,
        case_date: date | str | None = None,
    ) -> EvidenceMatrix:
        queries = build_option_queries(analysis)
        selected_jurisdiction = jurisdiction or _text(_field(analysis, "jurisdiction"))
        raw_date = case_date if case_date is not None else _field(analysis, "case_date")
        selected_date = self._parse_date(raw_date)
        eligible = self._eligible(
            jurisdiction=selected_jurisdiction,
            case_date=selected_date,
        )
        candidate_ids = tuple(sorted(item.evidence_id for item in eligible))
        matrix: EvidenceMatrix = {}
        for option, option_queries in queries.items():
            lanes: dict[str, tuple[RetrievedEvidence, ...]] = {}
            for lane in EvidenceLane:
                if lane.value not in self.config.enabled_lanes:
                    continue
                query = option_queries.for_lane(lane)
                lanes[lane.value] = self._retrieve_lane(
                    query,
                    candidate_ids=candidate_ids,
                )
            matrix[option] = lanes
        return matrix

    @staticmethod
    def _parse_date(value: date | str | Any | None) -> date | None:
        if value is None or value == "":
            return None
        if isinstance(value, date):
            return value
        if not isinstance(value, str):
            raise ValueError("case_date must be an ISO date or date object")
        try:
            return date.fromisoformat(value.strip())
        except ValueError as exc:
            raise ValueError("case_date must be an ISO date (YYYY-MM-DD)") from exc

    def _retrieve_lane(
        self,
        query: str,
        *,
        candidate_ids: tuple[str, ...],
    ) -> tuple[RetrievedEvidence, ...]:
        if not candidate_ids:
            return ()
        lexical = self._bm25.search(
            query,
            candidate_ids=candidate_ids,
            top_k=self.config.lexical_top_k,
        )
        dense: Sequence[DenseHit] = ()
        if self.dense_retriever is not None:
            dense = self.dense_retriever.search(
                query,
                candidate_ids=candidate_ids,
                top_k=self.config.dense_top_k,
            )
            allowed = set(candidate_ids)
            dense = tuple(
                hit
                for hit in dense
                if hit.evidence_id in allowed and math.isfinite(hit.score)
            )

        fused = reciprocal_rank_fusion((lexical, dense), rrf_k=self.config.rrf_k)
        lexical_scores = {hit.evidence_id: hit.score for hit in lexical}
        dense_scores = {hit.evidence_id: hit.score for hit in dense}
        ranked = [
            RetrievedEvidence(
                evidence=self._evidence[evidence_id],
                score=score,
                lexical_score=lexical_scores.get(evidence_id, 0.0),
                dense_score=dense_scores.get(evidence_id, 0.0),
            )
            for evidence_id, score in fused.items()
        ]
        ranked.sort(key=self._sort_key)
        expanded = self._expand_once(ranked, allowed_ids=set(candidate_ids))
        deduplicated = self._deduplicate((*ranked, *expanded))
        return self._minimal_sufficient(query, deduplicated)

    @staticmethod
    def _sort_key(item: RetrievedEvidence) -> tuple[float, int, str]:
        return (-item.score, -item.evidence.authority_level, item.evidence_id)

    def _expand_once(
        self,
        ranked: Sequence[RetrievedEvidence],
        *,
        allowed_ids: set[str],
    ) -> tuple[RetrievedEvidence, ...]:
        if self.config.graph_hops == 0:
            return ()
        expanded: dict[str, RetrievedEvidence] = {}
        for source in ranked:
            for target_id, relation_type in self._neighbors.get(source.evidence_id, ()):
                if target_id not in allowed_ids:
                    continue
                candidate = RetrievedEvidence(
                    evidence=self._evidence[target_id],
                    score=source.score * self.config.graph_decay,
                    relation_support=1.0,
                    expanded_from=source.evidence_id,
                    relation_type=relation_type,
                )
                previous = expanded.get(target_id)
                if previous is None or self._sort_key(candidate) < self._sort_key(previous):
                    expanded[target_id] = candidate
        return tuple(sorted(expanded.values(), key=self._sort_key))

    def _deduplicate(
        self,
        candidates: Sequence[RetrievedEvidence],
    ) -> tuple[RetrievedEvidence, ...]:
        by_id: dict[str, RetrievedEvidence] = {}
        for candidate in candidates:
            previous = by_id.get(candidate.evidence_id)
            if previous is None or self._sort_key(candidate) < self._sort_key(previous):
                by_id[candidate.evidence_id] = candidate

        by_text: dict[str, RetrievedEvidence] = {}
        for candidate in sorted(by_id.values(), key=self._sort_key):
            normalized = _text(candidate.evidence.text).casefold()
            by_text.setdefault(normalized, candidate)
        return tuple(sorted(by_text.values(), key=self._sort_key))

    def _minimal_sufficient(
        self,
        query: str,
        candidates: Sequence[RetrievedEvidence],
    ) -> tuple[RetrievedEvidence, ...]:
        if not candidates:
            return ()
        limit = self.config.final_top_k_per_lane
        query_terms = set(_tokenize(query))
        available_terms = query_terms.intersection(
            token
            for candidate in candidates
            for token in _tokenize(candidate.evidence.text)
        )
        selected: list[RetrievedEvidence] = []
        remaining = list(candidates)
        covered: set[str] = set()

        while remaining and len(selected) < limit:
            def selection_key(item: RetrievedEvidence) -> tuple[float, float, int, str]:
                new_terms = set(_tokenize(item.evidence.text)) & available_terms - covered
                gain = len(new_terms) / max(len(available_terms), 1)
                relation_bonus = 1.0 if item.expanded_from is not None else 0.0
                return (
                    -(gain + relation_bonus * 0.05),
                    -item.score,
                    -item.evidence.authority_level,
                    item.evidence_id,
                )

            best = min(remaining, key=selection_key)
            new_terms = set(_tokenize(best.evidence.text)) & available_terms - covered
            if selected and not new_terms and best.expanded_from is None:
                break
            selected.append(best)
            remaining.remove(best)
            covered.update(new_terms)
            coverage = len(covered) / max(len(available_terms), 1)
            has_expansion = any(item.expanded_from is not None for item in selected)
            expansion_available = any(item.expanded_from is not None for item in remaining)
            if coverage >= self.config.coverage_target and (
                has_expansion or not expansion_available
            ):
                break

        return tuple(sorted(selected, key=self._sort_key))
