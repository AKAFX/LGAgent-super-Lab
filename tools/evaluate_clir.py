#!/usr/bin/env python3
"""Replay a frozen CLIR policy over observed test intervention arms."""

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

from lgagent.clir import (
    ClirCalibration,
    InterventionRecord,
    replay_clir_policy,
)
from lgagent.evaluation import evaluate_result_records


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Offline CLIR replay. The selected action is evaluated from an "
            "already-observed intervention arm and makes no model calls."
        )
    )
    parser.add_argument("input", type=Path)
    parser.add_argument("--calibration", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--expected-split", choices=("dev", "test"), default="test")
    parser.add_argument("--minimum-uplift", type=float, default=0.0)
    parser.add_argument("--cost-weight", type=float, default=0.0)
    parser.add_argument("--token-scale", type=float, default=100_000.0)
    return parser.parse_args(argv)


def _read_split(path: Path, split: str) -> list[InterventionRecord]:
    records: list[InterventionRecord] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: record must be an object")
            record = InterventionRecord.from_dict(value)
            if record.split == split:
                records.append(record)
    return records


def _atomic_write(path: Path, value: Any, *, jsonl: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            if jsonl:
                for record in value:
                    stream.write(
                        json.dumps(
                            record,
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


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    calibration = ClirCalibration.load(args.calibration)
    records, clir_metrics = replay_clir_policy(
        _read_split(args.input, args.expected_split),
        calibration,
        expected_split=args.expected_split,
        minimum_uplift=args.minimum_uplift,
        cost_weight=args.cost_weight,
        token_scale=args.token_scale,
    )
    summary = {
        "method": "CLIR offline replay",
        "input": str(args.input),
        "calibration": str(args.calibration),
        "model_calls": 0,
        "clir": clir_metrics,
        "metrics": evaluate_result_records(records),
    }
    results_path = args.output_dir / "results.jsonl"
    summary_path = args.output_dir / "summary.json"
    _atomic_write(results_path, records, jsonl=True)
    _atomic_write(summary_path, summary)
    print(
        json.dumps(
            {
                "results": str(results_path),
                "summary": str(summary_path),
                "records": len(records),
                "model_calls": 0,
                "clir": clir_metrics,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
