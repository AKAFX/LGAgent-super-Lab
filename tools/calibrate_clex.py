#!/usr/bin/env python3
"""Fit a C-LEX split-conformal artifact from development result JSONL files."""

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

from lgagent.clex import fit_calibration


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit C-LEX thresholds from LGAgent++ development result records. "
            "Non-development records are rejected."
        )
    )
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--group-field", default="domain")
    parser.add_argument("--min-calibration-size", type=int, default=30)
    parser.add_argument("--min-group-size", type=int, default=30)
    return parser.parse_args(argv)


def _read_records(paths: list[Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in paths:
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(
                        f"{path}:{line_number}: result record must be an object"
                    )
                records.append(value)
    return records


def _atomic_write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    artifact = fit_calibration(
        _read_records(args.inputs),
        alpha=args.alpha,
        group_field=args.group_field,
        min_calibration_size=args.min_calibration_size,
        min_group_size=args.min_group_size,
    )
    _atomic_write(args.output, artifact.as_dict())
    print(
        json.dumps(
            {
                "output": str(args.output),
                "alpha": artifact.alpha,
                "global_count": artifact.global_count,
                "global_threshold": artifact.global_threshold,
                "groups": artifact.group_counts,
                "records_sha256": artifact.records_sha256,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
