"""Reproducible conversion of the pinned just-laws repository to OATH JSONL."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import tempfile
from dataclasses import dataclass, replace
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Mapping, Sequence

from .corpus import LegalEvidence, LegalRelation, RelationType, build_corpus_manifest

JUST_LAWS_REPOSITORY = "https://github.com/imca0/just-laws.git"
JUST_LAWS_COMMIT = "b475e4e00e1640a927faf79c3ccae663c10c3bbd"
JUST_LAWS_LICENSE = "MIT"
SOURCE_BLOB_ROOT = (
    f"https://github.com/imca0/just-laws/blob/{JUST_LAWS_COMMIT}"
)

_CATEGORY_LINK = re.compile(r"\]\(\.\./(?P<law_id>[^)]+)/\)")
_TITLE = re.compile(r"^#\s+(?P<title>.+?)\s*$", re.MULTILINE)
_ARTICLE = re.compile(
    r"(?m)^\s*\*\*(?P<label>第[〇零一二三四五六七八九十百千万0-9]+条(?:之[〇零一二三四五六七八九十百千万0-9]+)?)\*\*[　 \t]*(?P<body>.*)$"
)
_EXPLICIT_EFFECTIVE = re.compile(
    r"本(?:法|条例|决定|规则|章程|通则|组织法|解释)"
    r"自(?P<year>[〇零一二三四五六七八九十0-9]{4})年"
    r"(?P<month>[〇零一二三四五六七八九十0-9]{1,3})月"
    r"(?P<day>[〇零一二三四五六七八九十0-9]{1,3})日起施行"
)
_LAW_REFERENCE = re.compile(r"《(?P<title>[^》\n]{2,80})》")
_MARKDOWN_LINK = re.compile(r"\[([^\]]+)\]\([^)]+\)")
_INLINE_MARKUP = re.compile(r"[`*_]+")
_FRONT_MATTER = re.compile(r"\A---\s*\n.*?\n---\s*\n", re.DOTALL)
_CHINESE_DIGITS = {
    "〇": 0,
    "零": 0,
    "一": 1,
    "二": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
}


class CorpusBuildError(RuntimeError):
    """Raised when source provenance or legal text is not safely parseable."""


@dataclass(frozen=True)
class SourceVersion:
    law_id: str
    law_name: str
    version: str
    effective_from: date
    effective_to: date | None
    files: tuple[Path, ...]
    date_source: str


@dataclass(frozen=True)
class ExcludedVersion:
    law_id: str
    law_name: str
    version: str
    reason: str


@dataclass(frozen=True)
class CorpusBuildResult:
    records: tuple[LegalEvidence, ...]
    excluded: tuple[ExcludedVersion, ...]
    source_law_count: int
    included_version_count: int


def _run_git(arguments: Sequence[str], *, cwd: Path | None = None) -> str:
    try:
        result = subprocess.run(
            ["git", *arguments],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        detail = getattr(exc, "stderr", "") or str(exc)
        raise CorpusBuildError(f"git command failed: {detail.strip()}") from exc
    return result.stdout.strip()


def verify_source_commit(source_dir: str | Path) -> Path:
    """Require a git checkout whose HEAD is exactly the pinned source commit."""
    root = Path(source_dir).resolve()
    if not (root / "docs").is_dir() or not (root / "LICENSE").is_file():
        raise CorpusBuildError("source directory is not a just-laws checkout")
    actual = _run_git(["rev-parse", "HEAD"], cwd=root)
    if actual != JUST_LAWS_COMMIT:
        raise CorpusBuildError(
            f"source HEAD must be {JUST_LAWS_COMMIT}, got {actual}"
        )
    return root


def clone_pinned_source(destination: str | Path) -> Path:
    """Clone only the requested source revision and verify the resulting HEAD."""
    target = Path(destination).resolve()
    if target.exists() and any(target.iterdir()):
        raise CorpusBuildError(f"clone destination is not empty: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    _run_git(
        [
            "clone",
            "--filter=blob:none",
            "--no-checkout",
            JUST_LAWS_REPOSITORY,
            str(target),
        ]
    )
    _run_git(["fetch", "--depth=1", "origin", JUST_LAWS_COMMIT], cwd=target)
    _run_git(["checkout", "--detach", JUST_LAWS_COMMIT], cwd=target)
    return verify_source_commit(target)


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _law_ids(docs_dir: Path) -> tuple[str, ...]:
    law_ids: set[str] = set()
    for category in sorted((docs_dir / "category").glob("*.md")):
        law_ids.update(match.group("law_id") for match in _CATEGORY_LINK.finditer(
            category.read_text(encoding="utf-8")
        ))
    if (docs_dir / "constitution" / "README.md").is_file():
        law_ids.add("constitution")
    return tuple(sorted(law_ids))


def _law_name(root_file: Path) -> str:
    text = _FRONT_MATTER.sub("", root_file.read_text(encoding="utf-8"), count=1)
    match = _TITLE.search(text)
    if match is None:
        raise CorpusBuildError(f"missing H1 law title: {root_file}")
    return " ".join(match.group("title").split())


def _direct_law_files(law_dir: Path) -> tuple[Path, ...]:
    return tuple(
        path
        for path in sorted(law_dir.glob("*.md"))
        if path.name != "versions.json"
    )


def _parse_date(value: Any, *, context: str) -> date:
    if not isinstance(value, str):
        raise CorpusBuildError(f"{context} must be an ISO date")
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise CorpusBuildError(f"{context} must be an ISO date") from exc
    if parsed.isoformat() != value:
        raise CorpusBuildError(f"{context} must be an ISO date")
    return parsed


def _chinese_integer(value: str) -> int:
    if value.isdigit():
        return int(value)
    if all(character in _CHINESE_DIGITS for character in value):
        return int("".join(str(_CHINESE_DIGITS[character]) for character in value))
    total = 0
    current = 0
    units = {"十": 10, "百": 100}
    for character in value:
        if character in _CHINESE_DIGITS:
            current = _CHINESE_DIGITS[character]
        elif character in units:
            total += (current or 1) * units[character]
            current = 0
        else:
            raise ValueError(value)
    return total + current


def _explicit_effective_date(text: str) -> date | None:
    parsed: set[date] = set()
    for match in _EXPLICIT_EFFECTIVE.finditer(text):
        try:
            parsed.add(
                date(
                    _chinese_integer(match.group("year")),
                    _chinese_integer(match.group("month")),
                    _chinese_integer(match.group("day")),
                )
            )
        except ValueError:
            continue
    return next(iter(parsed)) if len(parsed) == 1 else None


def _version_sources(
    docs_dir: Path,
    law_id: str,
) -> tuple[tuple[SourceVersion, ...], tuple[ExcludedVersion, ...]]:
    law_dir = docs_dir / law_id
    root_file = law_dir / "README.md"
    if not root_file.is_file():
        return (), (ExcludedVersion(law_id, law_id, "current", "missing README.md"),)
    law_name = _law_name(root_file)
    manifest_path = law_dir / "versions.json"
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise CorpusBuildError(f"invalid versions.json: {manifest_path}") from exc
        if (
            not isinstance(manifest, Mapping)
            or manifest.get("schemaVersion") != 1
            or manifest.get("lawId") != law_id
            or not isinstance(manifest.get("versions"), list)
        ):
            raise CorpusBuildError(f"invalid versions.json schema: {manifest_path}")
        versions: list[SourceVersion] = []
        excluded: list[ExcludedVersion] = []
        for raw in manifest["versions"]:
            if not isinstance(raw, Mapping):
                raise CorpusBuildError(f"invalid version entry: {manifest_path}")
            version_id = str(raw.get("id") or "").strip()
            entry = str(raw.get("entry") or "").strip()
            try:
                effective_from = _parse_date(
                    raw.get("effectiveFrom"),
                    context=f"{law_id}/{version_id}.effectiveFrom",
                )
                raw_to = raw.get("effectiveTo")
                effective_to = (
                    _parse_date(
                        raw_to, context=f"{law_id}/{version_id}.effectiveTo"
                    )
                    - timedelta(days=1)
                    if raw_to is not None
                    else None
                )
            except CorpusBuildError as exc:
                excluded.append(
                    ExcludedVersion(law_id, law_name, version_id, str(exc))
                )
                continue
            entry_path = (law_dir / entry).resolve()
            if law_dir.resolve() not in entry_path.parents:
                raise CorpusBuildError(f"version entry escapes law directory: {entry}")
            if entry == "README.md":
                files = _direct_law_files(law_dir)
            else:
                files = (entry_path,)
            if not files or any(not path.is_file() for path in files):
                excluded.append(
                    ExcludedVersion(
                        law_id, law_name, version_id, "version entry is missing"
                    )
                )
                continue
            versions.append(
                SourceVersion(
                    law_id,
                    str(manifest.get("title") or law_name).strip(),
                    version_id,
                    effective_from,
                    effective_to,
                    files,
                    "versions.json",
                )
            )
        return tuple(versions), tuple(excluded)

    files = _direct_law_files(law_dir)
    combined = "\n".join(path.read_text(encoding="utf-8") for path in files)
    effective_from = _explicit_effective_date(combined)
    if effective_from is None:
        return (), (
            ExcludedVersion(
                law_id,
                law_name,
                "current",
                "no unique explicit effective date in statutory text",
            ),
        )
    return (
        SourceVersion(
            law_id,
            law_name,
            f"effective-{effective_from.isoformat()}",
            effective_from,
            None,
            files,
            "explicit statutory commencement text",
        ),
    ), ()


def _plain_text(value: str) -> str:
    value = _MARKDOWN_LINK.sub(r"\1", value)
    value = _INLINE_MARKUP.sub("", value)
    value = re.sub(r"<[^>]+>", "", value)
    return " ".join(value.split())


def _articles(path: Path) -> tuple[tuple[str, str], ...]:
    text = _FRONT_MATTER.sub("", path.read_text(encoding="utf-8"), count=1)
    matches = list(_ARTICLE.finditer(text))
    articles: list[tuple[str, str]] = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        first_line = match.group("body")
        continuation = text[match.end():end]
        continuation = re.sub(r"(?m)^#{1,6}\s+.*$", "", continuation)
        body = _plain_text(f"{first_line}\n{continuation}")
        if body:
            articles.append((match.group("label"), body))
    return tuple(articles)


def _record_payload(
    source_root: Path,
    version: SourceVersion,
    path: Path,
    article: str,
    text: str,
) -> dict[str, Any]:
    relative = path.relative_to(source_root).as_posix()
    return {
        "source_type": "statute",
        "law_name": version.law_name,
        "article": article,
        "clause": None,
        "version": version.version,
        "text": text,
        "jurisdiction": "CN",
        "authority_level": 5,
        "effective_from": version.effective_from.isoformat(),
        "effective_to": (
            version.effective_to.isoformat() if version.effective_to else None
        ),
        "source_uri": f"{SOURCE_BLOB_ROOT}/{relative}",
        "relations": [],
    }


def _attach_refers_to(
    records: Sequence[LegalEvidence],
) -> tuple[LegalEvidence, ...]:
    first_by_law: dict[str, list[LegalEvidence]] = {}
    for item in records:
        first_by_law.setdefault(item.law_name, []).append(item)
    targets: dict[str, LegalEvidence] = {}
    for law_name, items in first_by_law.items():
        open_versions = [item for item in items if item.effective_to is None]
        candidates = open_versions or list(items)
        targets[law_name] = min(
            candidates,
            key=lambda item: (
                item.effective_from or date.min,
                item.article,
                item.evidence_id,
            ),
        )

    related: list[LegalEvidence] = []
    for item in records:
        relations = set(item.relations)
        for match in _LAW_REFERENCE.finditer(item.text):
            title = " ".join(match.group("title").split())
            target = targets.get(title)
            if target is not None and target.evidence_id != item.evidence_id:
                relations.add(
                    LegalRelation(RelationType.REFERS_TO, target.evidence_id)
                )
        related.append(replace(item, relations=tuple(sorted(relations))))
    return tuple(related)


def build_just_laws_corpus(source_dir: str | Path) -> CorpusBuildResult:
    """Build strict article-level evidence from a verified source checkout."""
    root = verify_source_commit(source_dir)
    docs_dir = root / "docs"
    law_ids = _law_ids(docs_dir)
    payloads: list[dict[str, Any]] = []
    excluded: list[ExcludedVersion] = []
    included_versions = 0
    for law_id in law_ids:
        versions, rejected = _version_sources(docs_dir, law_id)
        excluded.extend(rejected)
        for version in versions:
            version_payloads = [
                _record_payload(root, version, path, article, text)
                for path in version.files
                for article, text in _articles(path)
            ]
            if not version_payloads:
                excluded.append(
                    ExcludedVersion(
                        law_id, version.law_name, version.version, "no articles parsed"
                    )
                )
                continue
            payloads.extend(version_payloads)
            included_versions += 1

    records = tuple(
        LegalEvidence.from_mapping(payload, line=index)
        for index, payload in enumerate(payloads, start=1)
    )
    records = _attach_refers_to(records)
    records = tuple(sorted(records, key=lambda item: item.evidence_id))
    return CorpusBuildResult(
        records=records,
        excluded=tuple(sorted(excluded, key=lambda item: (item.law_id, item.version))),
        source_law_count=len(law_ids),
        included_version_count=included_versions,
    )


def write_corpus_artifacts(
    result: CorpusBuildResult,
    output_dir: str | Path,
    *,
    source_dir: str | Path,
) -> dict[str, Path]:
    """Write deterministic corpus, manifest, exclusions, and licensing metadata."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    paths = {
        "corpus": destination / "oath_just_laws_strict.jsonl",
        "manifest": destination / "manifest.json",
        "excluded": destination / "excluded_versions.json",
        "licensing": destination / "licensing.json",
        "license_text": destination / "LICENSE.just-laws.txt",
        "provenance": destination / "provenance.json",
    }
    corpus_text = "".join(
        _canonical_json(item.as_dict()) + "\n" for item in result.records
    )
    paths["corpus"].write_text(corpus_text, encoding="utf-8")
    manifest = build_corpus_manifest(result.records).as_dict()
    manifest.update(
        {
            "builder": "lgagent.corpus_builder",
            "source_law_count": result.source_law_count,
            "included_version_count": result.included_version_count,
            "excluded_version_count": len(result.excluded),
            "file_sha256": _sha256_file(paths["corpus"]),
            "source_repository": JUST_LAWS_REPOSITORY,
            "source_commit": JUST_LAWS_COMMIT,
            "source_license": JUST_LAWS_LICENSE,
        }
    )
    paths["manifest"].write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    paths["excluded"].write_text(
        json.dumps(
            {
                "policy": (
                    "Versions without versions.json effectiveFrom or one unique "
                    "explicit statutory commencement date are excluded."
                ),
                "excluded": [item.__dict__ for item in result.excluded],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    license_path = Path(source_dir) / "LICENSE"
    paths["license_text"].write_text(
        license_path.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    paths["licensing"].write_text(
        json.dumps(
            {
                "source_project": "just-laws",
                "repository": JUST_LAWS_REPOSITORY,
                "commit": JUST_LAWS_COMMIT,
                "license": JUST_LAWS_LICENSE,
                "license_source_uri": f"{SOURCE_BLOB_ROOT}/LICENSE",
                "license_sha256": _sha256_file(license_path),
                "notice": (
                    "Derived legal text retains source provenance. Verify official "
                    "promulgation sources before legal or production use."
                ),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    paths["provenance"].write_text(
        json.dumps(
            {
                "repository": JUST_LAWS_REPOSITORY,
                "commit": JUST_LAWS_COMMIT,
                "source_uri_template": f"{SOURCE_BLOB_ROOT}/{{path}}",
                "authority_policy": (
                    "authority_level=5 only for law directories explicitly listed "
                    "by just-laws category indexes (plus the Constitution)."
                ),
                "effective_date_policy": (
                    "versions.json effectiveFrom is preferred; otherwise exactly "
                    "one explicit statutory commencement date is required. "
                    "versions.json effectiveTo is converted from exclusive to "
                    "inclusive by subtracting one day."
                ),
                "relation_policy": (
                    "REFERS_TO is emitted only for exact 《法律名》 matches to a "
                    "uniquely indexed corpus law and targets that law's first "
                    "open-ended-version article."
                ),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return paths


def build_with_optional_clone(
    *,
    output_dir: str | Path,
    source_dir: str | Path | None = None,
    clone_dir: str | Path | None = None,
) -> tuple[CorpusBuildResult, dict[str, Path]]:
    """Build from a local checkout or clone the pinned revision first."""
    temporary: tempfile.TemporaryDirectory[str] | None = None
    if source_dir is None:
        if clone_dir is None:
            temporary = tempfile.TemporaryDirectory(prefix="just-laws-")
            clone_dir = Path(temporary.name) / "source"
        source_dir = clone_pinned_source(clone_dir)
    try:
        source = verify_source_commit(source_dir)
        result = build_just_laws_corpus(source)
        paths = write_corpus_artifacts(result, output_dir, source_dir=source)
        return result, paths
    finally:
        if temporary is not None:
            temporary.cleanup()
