#!/usr/bin/env python3
"""Validate or execute the frozen Task 16 experiment matrix."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from lgagent.experiment_freeze import EXPERIMENT_PROFILES
from lgagent.experiment_runner import Task16ExperimentRunner


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Task 16 真实实验执行器（默认不执行，必须显式选择 --dry-run 或 --execute）"
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="仅验证输入并估算调用量，不初始化 API 客户端",
    )
    mode.add_argument(
        "--execute",
        action="store_true",
        help="使用配置凭据执行真实 API 实验",
    )
    parser.add_argument(
        "--profile",
        choices=tuple(EXPERIMENT_PROFILES),
        default="pilot",
        help="分阶段实验配置（默认 pilot；完整矩阵必须显式指定 full）",
    )
    parser.add_argument(
        "--config",
        default=str(ROOT_DIR / "examples/parameter/legal2_rag_parameter.yaml"),
    )
    parser.add_argument(
        "--freeze",
        default=str(ROOT_DIR / "docs/task16_experiment_freeze.json"),
    )
    parser.add_argument(
        "--corpus",
        default=str(ROOT_DIR / "data/oath/oath_just_laws_strict.jsonl"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(ROOT_DIR / "output/task16"),
    )
    parser.add_argument(
        "--split",
        choices=("train", "dev", "test", "all"),
        help="覆盖 profile 的数据切分",
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        help="覆盖 profile 的 Task 14 experiment key 列表",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        help="覆盖 profile 的每数据集最大题数",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        help="覆盖 profile 的随机种子列表",
    )
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    return parser.parse_args(argv)


def resolve_plan(args: argparse.Namespace) -> dict[str, Any]:
    profile = EXPERIMENT_PROFILES[args.profile]
    return {
        "profile_name": args.profile,
        "split": args.split or profile.split,
        "experiment_keys": (
            tuple(args.experiments)
            if args.experiments is not None
            else profile.experiment_keys
        ),
        "max_examples": (
            args.max_examples
            if args.max_examples is not None
            else profile.max_examples
        ),
        "seeds": (
            tuple(args.seeds)
            if args.seeds is not None
            else profile.seeds
        ),
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.max_examples is not None and args.max_examples <= 0:
        raise SystemExit("--max-examples must be positive")
    if args.concurrency <= 0 or args.checkpoint_every <= 0:
        raise SystemExit("--concurrency and --checkpoint-every must be positive")
    if args.seeds is not None and len(set(args.seeds)) != len(args.seeds):
        raise SystemExit("--seeds must be unique")
    runner = Task16ExperimentRunner(
        project_root=ROOT_DIR,
        config_path=args.config,
        freeze_path=args.freeze,
        corpus_path=args.corpus,
        output_dir=args.output_dir,
    )
    common = resolve_plan(args)
    if args.dry_run:
        report = runner.dry_run(**common)
        print(
            json.dumps(
                {
                    "dry_run": True,
                    "jobs": report["jobs"],
                    "estimated_calls": report["estimated_calls"],
                    "cost_warning": report["cost_warning"],
                    "resolved_plan": report["resolved_plan"],
                    "accuracy": None,
                    "report": str(Path(args.output_dir) / "dry_run_report.json"),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    report = runner.run(
        **common,
        concurrency=args.concurrency,
        checkpoint_every=args.checkpoint_every,
    )
    print(
        json.dumps(
            {
                "dry_run": False,
                "completed_runs": len(report["runs"]),
                "summary": str(Path(args.output_dir) / "experiment_summary.json"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
