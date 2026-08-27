"""OATH-RAG legal evidence schema and deterministic JSONL corpus loading."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass
from datetime import date
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import quote

SCHEMA_VERSION = "oath-rag-corpus-v1"
_EVIDENCE_ID = re.compile(r"^[^\s]+$")
_MODERN_FIELDS = frozenset(
    {
        "evidence_id",
        "source_type",
        "law_name",
        "article",
        "clause",
        "version",
        "text",
        "jurisdiction",
        "authority_level",
        "effective_from",
        "effective_to",
        "source_uri",
        "relations",
    }
)
_MODERN_REQUIRED_FIELDS = _MODERN_FIELDS - {
    "evidence_id",
    "clause",
    "effective_to",
    "relations",
}
_LEGACY_FIELDS = frozenset({"id", "title", "contents"})


class CorpusValidationError(ValueError):
    """Raised when a corpus cannot satisfy the OATH-RAG contract."""

    def __init__(self, message: str, *, line: int | None = None) -> None:
        prefix = f"line {line}: " if line is not None else ""
        super().__init__(prefix + message)
        self.line = line


class RelationType(str, Enum):
    DEFINES = "DEFINES"
    REFERS_TO = "REFERS_TO"
    EXCEPTION_TO = "EXCEPTION_TO"
    AMENDS = "AMENDS"


def _normalized_text(value: Any, field: str, *, line: int) -> str:
    if not isinstance(value, str):
        raise CorpusValidationError(f"{field} must be a string", line=line)
    normalized = " ".join(unicodedata.normalize("NFKC", value).split())
    if not normalized:
        raise CorpusValidationError(f"{field} must not be empty", line=line)
    return normalized


def _optional_text(value: Any, field: str, *, line: int) -> str | None:
    if value is None:
        return None
    return _normalized_text(value, field, line=line)


def _date(value: Any, field: str, *, line: int) -> date | None:
    if value is None:
        return None
    normalized = _normalized_text(value, field, line=line)
    try:
        parsed = date.fromisoformat(normalized)
    except ValueError as exc:
        raise CorpusValidationError(
            f"{field} must be an ISO date (YYYY-MM-DD)", line=line
        ) from exc
    if parsed.isoformat() != normalized:
        raise CorpusValidationError(
            f"{field} must be an ISO date (YYYY-MM-DD)", line=line
        )
    return parsed


def _check_fields(
    value: Mapping[str, Any],
    *,
    allowed: frozenset[str],
    required: frozenset[str],
    line: int,
) -> None:
    fields = set(value)
    unknown = sorted(fields - allowed)
    missing = sorted(required - fields)
    if unknown:
        raise CorpusValidationError(f"unknown fields: {unknown}", line=line)
    if missing:
        raise CorpusValidationError(f"missing fields: {missing}", line=line)


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _parse_json_line(raw_line: str, *, line: int) -> Any:
    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise CorpusValidationError(
                    f"duplicate JSON object key {key!r}", line=line
                )
            result[key] = value
        return result

    def reject_non_finite(value: str) -> None:
        raise CorpusValidationError(
            f"non-finite JSON number {value!r} is not allowed", line=line
        )

    try:
        return json.loads(
            raw_line,
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=reject_non_finite,
        )
    except json.JSONDecodeError as exc:
        raise CorpusValidationError(
            f"invalid JSON at column {exc.colno}", line=line
        ) from exc


@dataclass(frozen=True, order=True)
class LegalRelation:
    type: RelationType
    target_id: str

    @classmethod
    def from_mapping(
        cls, value: Any, *, line: int, index: int
    ) -> "LegalRelation":
        field = f"relations[{index}]"
        if not isinstance(value, Mapping):
            raise CorpusValidationError(f"{field} must be an object", line=line)
        _check_fields(
            value,
            allowed=frozenset({"type", "target_id"}),
            required=frozenset({"type", "target_id"}),
            line=line,
        )
        relation_name = _normalized_text(value["type"], f"{field}.type", line=line)
        try:
            relation_type = RelationType(relation_name.upper())
        except ValueError as exc:
            allowed = ", ".join(item.value for item in RelationType)
            raise CorpusValidationError(
                f"{field}.type must be one of {allowed}", line=line
            ) from exc
        target_id = _normalized_text(
            value["target_id"], f"{field}.target_id", line=line
        )
        if _EVIDENCE_ID.fullmatch(target_id) is None:
            raise CorpusValidationError(
                f"{field}.target_id must not contain whitespace", line=line
            )
        return cls(type=relation_type, target_id=target_id)

    def as_dict(self) -> dict[str, str]:
        return {"type": self.type.value, "target_id": self.target_id}


@dataclass(frozen=True)
class LegalEvidence:
    evidence_id: str
    source_type: str
    law_name: str
    article: str
    clause: str | None
    version: str
    text: str
    jurisdiction: str
    authority_level: int
    effective_from: date | None
    effective_to: date | None
    source_uri: str
    relations: tuple[LegalRelation, ...] = ()

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, Any], *, line: int = 1
    ) -> "LegalEvidence":
        _check_fields(
            value,
            allowed=_MODERN_FIELDS,
            required=_MODERN_REQUIRED_FIELDS,
            line=line,
        )
        authority_level = value["authority_level"]
        if isinstance(authority_level, bool) or not isinstance(authority_level, int):
            raise CorpusValidationError(
                "authority_level must be an integer", line=line
            )
        if authority_level <= 0:
            raise CorpusValidationError(
                "authority_level must be positive", line=line
            )

        effective_from = _date(
            value["effective_from"], "effective_from", line=line
        )
        if effective_from is None:
            raise CorpusValidationError(
                "effective_from must not be null for a legal record", line=line
            )
        effective_to = _date(value.get("effective_to"), "effective_to", line=line)
        if effective_to is not None and effective_to < effective_from:
            raise CorpusValidationError(
                "effective_to must be on or after effective_from", line=line
            )

        raw_relations = value.get("relations", [])
        if not isinstance(raw_relations, list):
            raise CorpusValidationError("relations must be an array", line=line)
        relations = tuple(
            sorted(
                LegalRelation.from_mapping(item, line=line, index=index)
                for index, item in enumerate(raw_relations)
            )
        )
        if len(set(relations)) != len(relations):
            raise CorpusValidationError("relations must not contain duplicates", line=line)

        normalized = {
            "source_type": _normalized_text(
                value["source_type"], "source_type", line=line
            ).lower(),
            "law_name": _normalized_text(value["law_name"], "law_name", line=line),
            "article": _normalized_text(value["article"], "article", line=line),
            "clause": _optional_text(value.get("clause"), "clause", line=line),
            "version": _normalized_text(value["version"], "version", line=line),
            "text": _normalized_text(value["text"], "text", line=line),
            "jurisdiction": _normalized_text(
                value["jurisdiction"], "jurisdiction", line=line
            ).upper(),
            "authority_level": authority_level,
            "effective_from": effective_from,
            "effective_to": effective_to,
            "source_uri": _normalized_text(
                value["source_uri"], "source_uri", line=line
            ),
            "relations": relations,
        }
        evidence_id = value.get("evidence_id")
        if evidence_id is None:
            evidence_id = stable_evidence_id(normalized)
        else:
            evidence_id = _normalized_text(evidence_id, "evidence_id", line=line)
            if _EVIDENCE_ID.fullmatch(evidence_id) is None:
                raise CorpusValidationError(
                    "evidence_id must not contain whitespace", line=line
                )
        return cls(evidence_id=evidence_id, **normalized)

    def as_dict(self) -> dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "source_type": self.source_type,
            "law_name": self.law_name,
            "article": self.article,
            "clause": self.clause,
            "version": self.version,
            "text": self.text,
            "jurisdiction": self.jurisdiction,
            "authority_level": self.authority_level,
            "effective_from": (
                self.effective_from.isoformat() if self.effective_from else None
            ),
            "effective_to": self.effective_to.isoformat() if self.effective_to else None,
            "source_uri": self.source_uri,
            "relations": [relation.as_dict() for relation in self.relations],
        }


def stable_evidence_id(value: Mapping[str, Any] | LegalEvidence) -> str:
    """Return a stable ID from normalized evidence content, excluding relations."""
    if isinstance(value, LegalEvidence):
        payload = value.as_dict()
    else:
        payload = dict(value)
        payload = {
            key: (
                item.isoformat()
                if isinstance(item, date)
                else item
            )
            for key, item in payload.items()
        }
    payload.pop("evidence_id", None)
    payload.pop("relations", None)
    digest = hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()
    return f"oath:{digest}"


def _legacy_to_evidence(value: Mapping[str, Any], *, line: int) -> LegalEvidence:
    _check_fields(
        value,
        allowed=_LEGACY_FIELDS,
        required=frozenset({"id", "contents"}),
        line=line,
    )
    legacy_id = _normalized_text(value["id"], "id", line=line)
    contents = _normalized_text(value["contents"], "contents", line=line)
    title = _optional_text(value.get("title"), "title", line=line)
    parts = [part.strip() for part in legacy_id.split("-") if part.strip()]
    law_name = title or (parts[0] if len(parts) > 1 else legacy_id)
    article = parts[-1] if len(parts) > 1 else legacy_id
    payload: dict[str, Any] = {
        "source_type": "legacy",
        "law_name": law_name,
        "article": article,
        "clause": None,
        "version": "legacy-unversioned",
        "text": contents,
        "jurisdiction": "UNKNOWN",
        "authority_level": 0,
        "effective_from": None,
        "effective_to": None,
        "source_uri": f"legacy://{quote(legacy_id, safe='')}",
        "relations": (),
    }
    return LegalEvidence(evidence_id=stable_evidence_id(payload), **payload)


@dataclass(frozen=True)
class CorpusManifest:
    schema_version: str
    document_count: int
    legacy_document_count: int
    relation_count: int
    jurisdictions: dict[str, int]
    source_types: dict[str, int]
    content_sha256: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "document_count": self.document_count,
            "legacy_document_count": self.legacy_document_count,
            "relation_count": self.relation_count,
            "jurisdictions": dict(sorted(self.jurisdictions.items())),
            "source_types": dict(sorted(self.source_types.items())),
            "content_sha256": self.content_sha256,
        }


@dataclass(frozen=True)
class LoadedCorpus:
    evidence: tuple[LegalEvidence, ...]
    manifest: CorpusManifest


def _intervals_overlap(left: LegalEvidence, right: LegalEvidence) -> bool:
    if left.effective_to is not None and right.effective_from is not None:
        if left.effective_to < right.effective_from:
            return False
    if right.effective_to is not None and left.effective_from is not None:
        if right.effective_to < left.effective_from:
            return False
    return True


def _validate_corpus(evidence: Sequence[LegalEvidence]) -> None:
    by_id: dict[str, LegalEvidence] = {}
    by_clause: dict[tuple[str, str, str, str], list[LegalEvidence]] = {}
    for item in evidence:
        if item.evidence_id in by_id:
            raise CorpusValidationError(
                f"duplicate evidence_id {item.evidence_id!r}"
            )
        by_id[item.evidence_id] = item
        key = (
            item.jurisdiction,
            item.law_name,
            item.article,
            item.clause or "",
        )
        versions = by_clause.setdefault(key, [])
        for previous in versions:
            if previous.version == item.version:
                raise CorpusValidationError(
                    "duplicate clause version "
                    f"{item.law_name} {item.article} {item.clause or ''} "
                    f"({item.version})".strip()
                )
            if _intervals_overlap(previous, item):
                raise CorpusValidationError(
                    "overlapping version intervals for "
                    f"{item.law_name} {item.article} {item.clause or ''}: "
                    f"{previous.version} and {item.version}".strip()
                )
        versions.append(item)


def build_corpus_manifest(evidence: Sequence[LegalEvidence]) -> CorpusManifest:
    canonical_records = sorted(
        (_canonical_json(item.as_dict()) for item in evidence)
    )
    canonical_content = "\n".join(canonical_records)
    if canonical_content:
        canonical_content += "\n"
    jurisdictions: dict[str, int] = {}
    source_types: dict[str, int] = {}
    for item in evidence:
        jurisdictions[item.jurisdiction] = jurisdictions.get(item.jurisdiction, 0) + 1
        source_types[item.source_type] = source_types.get(item.source_type, 0) + 1
    return CorpusManifest(
        schema_version=SCHEMA_VERSION,
        document_count=len(evidence),
        legacy_document_count=source_types.get("legacy", 0),
        relation_count=sum(len(item.relations) for item in evidence),
        jurisdictions=jurisdictions,
        source_types=source_types,
        content_sha256=hashlib.sha256(canonical_content.encode("utf-8")).hexdigest(),
    )


def load_jsonl_corpus(
    path: str | Path,
    *,
    allow_legacy: bool = True,
    manifest_path: str | Path | None = None,
) -> LoadedCorpus:
    """Load and validate a legal corpus without network or model dependencies."""
    corpus_path = Path(path)
    evidence: list[LegalEvidence] = []
    try:
        lines = corpus_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise CorpusValidationError(f"cannot read corpus {corpus_path}: {exc}") from exc

    for line_number, raw_line in enumerate(lines, start=1):
        if not raw_line.strip():
            raise CorpusValidationError("blank lines are not allowed", line=line_number)
        value = _parse_json_line(raw_line, line=line_number)
        if not isinstance(value, Mapping):
            raise CorpusValidationError(
                "JSONL record must be an object", line=line_number
            )
        is_legacy = "contents" in value and "text" not in value
        if is_legacy:
            if not allow_legacy:
                raise CorpusValidationError(
                    "legacy contents records are disabled", line=line_number
                )
            item = _legacy_to_evidence(value, line=line_number)
        else:
            item = LegalEvidence.from_mapping(value, line=line_number)
        evidence.append(item)

    if not evidence:
        raise CorpusValidationError("corpus must contain at least one record")
    _validate_corpus(evidence)
    normalized = tuple(sorted(evidence, key=lambda item: item.evidence_id))
    manifest = build_corpus_manifest(normalized)
    if manifest_path is not None:
        output_path = Path(manifest_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(
                manifest.as_dict(),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    return LoadedCorpus(evidence=normalized, manifest=manifest)
