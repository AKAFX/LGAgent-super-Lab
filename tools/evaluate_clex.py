#!/usr/bin/env python3
"""Apply a frozen C-LEX artifact to result records without model calls."""

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

from lgagent.clex import (
    CLEX_SCORE_VERSION,
    ClexCalibration,
    ClexOptionScore,
    apply_calibration,
)
from lgagent.evaluation import evaluate_result_records


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay C-LEX over frozen LGAgent++ result records."
    )
    parser.add_argument("input", type=Path)
    parser.add_argument("--calibration", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--expected-split", choices=("dev", "test"), default="test")
    parser.add_argument("--min-calibration-size", type=int, default=30)
    return parser.parse_args(argv)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: record must be an object")
            sample_id = str(value.get("sample_id", "")).strip()
            if not sample_id:
                raise ValueError(f"{path}:{line_number}: sample_id is required")
            if sample_id in seen:
                raise ValueError(
                    "C-LEX replay requires one record per sample; "
                    "evaluate random seeds separately"
                )
            seen.add(sample_id)
            records.append(value)
    return records


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


def _option_scores(record: dict[str, Any]) -> dict[str, ClexOptionScore]:
    if record.get("clex_score_version") != CLEX_SCORE_VERSION:
        raise ValueError("record C-LEX score version is missing or incompatible")
    raw = record.get("clex_option_scores")
    if not isinstance(raw, dict):
        raise ValueError("record is missing clex_option_scores")
    result: dict[str, ClexOptionScore] = {}
    for option, item in raw.items():
        if not isinstance(item, dict):
            raise ValueError("C-LEX option score must be an object")
        result[str(option)] = ClexOptionScore(
            option=str(option),
            conformity=float(item["conformity"]),
            nonconformity=float(item["nonconformity"]),
            components=dict(item.get("components", {})),
        )
    return result


def replay(
    records: list[dict[str, Any]],
    calibration: ClexCalibration,
    *,
    expected_split: str,
    min_calibration_size: int,
) -> list[dict[str, Any]]:
    transformed: list[dict[str, Any]] = []
    for record in records:
        if record.get("split") != expected_split:
            raise ValueError(
                f"C-LEX replay expected split {expected_split!r}, "
                f"got {record.get('split')!r}"
            )
        item = dict(record)
        if item.get("status", "ok") != "ok":
            transformed.append(item)
            continue
        original = str(item.get("prediction", "")).strip()
        decision = apply_calibration(
            _option_scores(item),
            calibration,
            current_answer=original,
            group=str(item.get(calibration.group_field, "")),
            min_calibration_size=min_calibration_size,
        )
        item["pre_clex_prediction"] = original
        item["prediction"] = decision.selected_answer
        item["clex"] = decision.as_dict()
        transformed.append(item)
    return transformed


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    calibration = ClexCalibration.load(args.calibration)
    records = replay(
        _read_jsonl(args.input),
        calibration,
        expected_split=args.expected_split,
        min_calibration_size=args.min_calibration_size,
    )
    output_dir = args.output_dir
    results_path = output_dir / "results.jsonl"
    summary_path = output_dir / "summary.json"
    summary = {
        "method": "C-LEX offline replay",
        "input": str(args.input),
        "calibration": str(args.calibration),
        "calibration_records_sha256": calibration.records_sha256,
        "model_calls": 0,
        "records": len(records),
        "metrics": evaluate_result_records(records),
    }
    _atomic_json(results_path, records, jsonl=True)
    _atomic_json(summary_path, summary)
    print(
        json.dumps(
            {
                "results": str(results_path),
                "summary": str(summary_path),
                "records": len(records),
                "model_calls": 0,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
