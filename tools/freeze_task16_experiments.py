#!/usr/bin/env python3
"""Generate the deterministic Task 16 dataset/config/seed freeze manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from lgagent.experiment_freeze import build_freeze_manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="examples/parameter/legal2_rag_parameter.yaml",
    )
    parser.add_argument(
        "--output",
        default="docs/task16_experiment_freeze.json",
    )
    args = parser.parse_args(argv)
    payload = build_freeze_manifest(ROOT_DIR, config_path=args.config)
    output = Path(args.output)
    if not output.is_absolute():
        output = ROOT_DIR / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"冻结清单已写入 {output}，freeze_id={payload['freeze_id']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
