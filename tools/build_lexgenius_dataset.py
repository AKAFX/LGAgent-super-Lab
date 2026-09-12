#!/usr/bin/env python3
"""Build a balanced, reproducible LexGenius subset from seven dimensions."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from lgagent.cape_v import OptionParseError, parse_four_option_question

DIMENSIONS = (
    (1, "Legal Understanding", "1_LegalUnderstanding_sample_200.jsonl"),
    (2, "Legal Reasoning", "2_LegalReasoning_sample_200.jsonl"),
    (3, "Legal Application", "3_LegalApplication_sample_200.jsonl"),
    (4, "Legal Ethics", "4_LegalEthics_sample_200.jsonl"),
    (5, "Legal Language", "5_LegalLanguage_sample_200.jsonl"),
    (6, "Law and Society", "6_LawAndSociety_sample_200.jsonl"),
    (7, "Judicial Practice", "7_JudicialPractice_sample_200.jsonl"),
)
BUILD_ALGORITHM = "per-dimension-sha256-seeded-sample-round-robin-v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalized_question(value: str) -> str:
    return " ".join(value.split())


def derived_seed(seed: int, dimension_id: int) -> int:
    digest = hashlib.sha256(f"{seed}:{dimension_id}".encode("ascii")).digest()
    return int.from_bytes(digest[:8], "big")


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT_DIR))
    except ValueError:
        return str(path.resolve())


@dataclass(frozen=True)
class SourceRecord:
    source_file: str
    source_line: int
    source_id: Any
    source_original_index: Any
    dimension_id: int
    dimension: str
    value: Mapping[str, Any]


def _read_jsonl(path: Path, dimension_id: int, dimension: str) -> list[SourceRecord]:
    records: list[SourceRecord] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, Mapping):
                raise ValueError(f"{path}:{line_number}: record must be an object")
            metadata = value.get("meta_data")
            metadata = metadata if isinstance(metadata, Mapping) else {}
            records.append(
                SourceRecord(
                    source_file=path.name,
                    source_line=line_number,
                    source_id=value.get("id"),
                    source_original_index=metadata.get("original_index"),
                    dimension_id=dimension_id,
                    dimension=dimension,
                    value=value,
                )
            )
    return records


def _validation_error(record: SourceRecord) -> str | None:
    question = record.value.get("question")
    if not isinstance(question, str) or not question.strip():
        return "missing_question"
    golden = record.value.get("golden_answers")
    if (
        not isinstance(golden, Sequence)
        or isinstance(golden, (str, bytes))
        or len(golden) != 1
        or golden[0] not in {"A", "B", "C", "D"}
    ):
        return "invalid_single_choice_answer"
    metadata = record.value.get("meta_data")
    if not isinstance(metadata, Mapping):
        return "missing_meta_data"
    if metadata.get("correct_option") != golden[0]:
        return "correct_option_mismatch"
    if "正确选项:" in question or "我的答案:" in question:
        return "answer_leakage"
    try:
        parse_four_option_question(question)
    except OptionParseError:
        return "unparseable_four_option_question"
    return None


def _atomic_json(path: Path, value: Any, *, jsonl: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            if jsonl:
                for item in value:
                    stream.write(
                        json.dumps(
                            item,
                            ensure_ascii=False,
                            sort_keys=True,
                            separators=(",", ":"),
                        )
                        + "\n"
                    )
            else:
                json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
                stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def build_lexgenius(
    input_dir: Path,
    *,
    sample_per_dimension: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if sample_per_dimension <= 0:
        raise ValueError("sample_per_dimension must be positive")

    seen_questions: dict[str, SourceRecord] = {}
    selected_by_dimension: list[list[SourceRecord]] = []
    sources: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []

    for dimension_id, dimension, filename in DIMENSIONS:
        path = input_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f"missing LexGenius dimension file: {path}")
        raw = _read_jsonl(path, dimension_id, dimension)
        valid: list[SourceRecord] = []
        for record in raw:
            error = _validation_error(record)
            normalized = (
                normalized_question(str(record.value.get("question", "")))
                if error is None
                else ""
            )
            if error is None and normalized in seen_questions:
                error = "duplicate_question"
            if error is not None:
                exclusions.append(
                    {
                        "dimension_id": dimension_id,
                        "dimension": dimension,
                        "source_file": filename,
                        "source_line": record.source_line,
                        "source_id": record.source_id,
                        "reason": error,
                    }
                )
                continue
            seen_questions[normalized] = record
            valid.append(record)
        if len(valid) < sample_per_dimension:
            raise ValueError(
                f"{dimension} has only {len(valid)} valid unique records; "
                f"cannot sample {sample_per_dimension}"
            )
        rng = random.Random(derived_seed(seed, dimension_id))
        selected = rng.sample(valid, sample_per_dimension)
        selected_by_dimension.append(selected)
        sources.append(
            {
                "dimension_id": dimension_id,
                "dimension": dimension,
                "path": display_path(path),
                "sha256": sha256_file(path),
                "raw_count": len(raw),
                "valid_unique_count": len(valid),
                "selected_count": len(selected),
                "selected_source_ids": [record.source_id for record in selected],
            }
        )

    output: list[dict[str, Any]] = []
    for position in range(sample_per_dimension):
        for selected in selected_by_dimension:
            source = selected[position]
            item = dict(source.value)
            metadata = dict(item.get("meta_data") or {})
            metadata.update(
                {
                    "dataset": "LexGenius",
                    "domain": source.dimension,
                    "dimension_id": source.dimension_id,
                    "dimension": source.dimension,
                    "source_file": source.source_file,
                    "source_line": source.source_line,
                    "source_id": source.source_id,
                    "source_original_index": source.source_original_index,
                    "sampling_seed": seed,
                }
            )
            item.update(
                {
                    "id": len(output),
                    "domain": source.dimension,
                    "meta_data": metadata,
                }
            )
            output.append(item)

    manifest = {
        "schema_version": 1,
        "dataset": "LexGenius",
        "variant": "balanced_80_per_dimension",
        "build_algorithm": BUILD_ALGORITHM,
        "seed": seed,
        "sample_per_dimension": sample_per_dimension,
        "total_samples": len(output),
        "dimensions": sources,
        "excluded_count": len(exclusions),
        "exclusions": exclusions,
    }
    return output, manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a balanced LexGenius JSONL dataset."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=ROOT_DIR / "data/dimension_jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT_DIR / "data/LexGenius.jsonl",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT_DIR / "data/LexGenius.manifest.json",
    )
    parser.add_argument("--sample-per-dimension", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    records, manifest = build_lexgenius(
        args.input_dir.resolve(),
        sample_per_dimension=args.sample_per_dimension,
        seed=args.seed,
    )
    _atomic_json(args.output, records, jsonl=True)
    manifest["output"] = {
        "path": display_path(args.output),
        "sha256": sha256_file(args.output),
    }
    _atomic_json(args.manifest, manifest)
    print(
        json.dumps(
            {
                "dataset": str(args.output),
                "manifest": str(args.manifest),
                "samples": len(records),
                "sample_per_dimension": args.sample_per_dimension,
                "excluded": manifest["excluded_count"],
                "sha256": manifest["output"]["sha256"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
