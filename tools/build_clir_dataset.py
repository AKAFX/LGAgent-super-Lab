#!/usr/bin/env python3
"""Build paired CLIR intervention records from frozen experiment results."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from lgagent.clir import build_intervention_records


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Pair one Original result JSONL with one or more intervention "
            "result JSONLs. Failed outcomes remain in the paired dataset."
        )
    )
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument(
        "--action",
        action="append",
        required=True,
        metavar="NAME=RESULTS_JSONL",
    )
    parser.add_argument(
        "--designated-split",
        choices=("train", "dev", "test"),
        help=(
            "Assign all paired rows to a predesignated CLIR split while "
            "preserving the original source split in provenance."
        ),
    )
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: record must be an object")
            records.append(value)
    return records


def _action_paths(values: list[str]) -> dict[str, Path]:
    actions: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError("--action must use NAME=RESULTS_JSONL")
        name, raw_path = value.split("=", 1)
        normalized = name.strip().lower()
        if not normalized or normalized in actions:
            raise ValueError("action names must be non-empty and unique")
        path = Path(raw_path.strip())
        if not raw_path.strip():
            raise ValueError("action result path cannot be empty")
        actions[normalized] = path
    return actions


def _atomic_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            for record in records:
                stream.write(
                    json.dumps(
                        record,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    paths = _action_paths(args.action)
    records = build_intervention_records(
        _read_jsonl(args.baseline),
        {name: _read_jsonl(path) for name, path in paths.items()},
        designated_split=args.designated_split,
    )
    serialized = [record.as_dict() for record in records]
    _atomic_jsonl(args.output, serialized)
    split_counts: dict[str, int] = {}
    for record in records:
        split_counts[record.split] = split_counts.get(record.split, 0) + 1
    print(
        json.dumps(
            {
                "output": str(args.output),
                "records": len(records),
                "actions": sorted(paths),
                "split_counts": dict(sorted(split_counts.items())),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
