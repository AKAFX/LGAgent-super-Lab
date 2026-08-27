#!/usr/bin/env python3
"""Build the strict OATH corpus from the pinned just-laws revision."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from lgagent.corpus_builder import build_with_optional_clone


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="从固定 just-laws commit 构建严格 OATH LegalEvidence corpus"
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--source-dir",
        help="已检出固定 commit 的本地 just-laws 仓库",
    )
    source.add_argument(
        "--clone-dir",
        help="将固定 commit 克隆到该空目录；省略时使用临时目录",
    )
    parser.add_argument(
        "--output-dir",
        default=str(ROOT_DIR / "data" / "oath"),
        help="corpus、manifest、排除报告和来源信息输出目录",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result, paths = build_with_optional_clone(
        source_dir=args.source_dir,
        clone_dir=args.clone_dir,
        output_dir=args.output_dir,
    )
    print(
        f"严格 corpus 构建完成：{len(result.records)} 条，"
        f"{result.included_version_count} 个版本，"
        f"排除 {len(result.excluded)} 个版本。"
    )
    for name, path in paths.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
