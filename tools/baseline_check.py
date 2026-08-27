"""Reproducibility checks for the corrected original LGAgent baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any
from unittest.mock import patch

import yaml

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

MANIFEST_PATH = ROOT_DIR / "docs" / "baseline_data_manifest.json"
LFS_POINTER_PREFIX = b"version https://git-lfs.github.com/spec/v1\n"


class CredentialsUnavailable(RuntimeError):
    """Raised when an explicitly requested online check has no credential."""


def _json_type(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    raise TypeError(f"Unsupported JSON value type: {type(value).__name__}")


def inspect_jsonl(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    field_types: dict[str, set[str]] = defaultdict(set)
    sample_count = 0

    with path.open("rb") as raw:
        prefix = raw.read(len(LFS_POINTER_PREFIX))
        if prefix == LFS_POINTER_PREFIX:
            raise RuntimeError(
                f"{path.relative_to(ROOT_DIR)} is a Git LFS pointer; "
                "run scripts/bootstrap_git_lfs.sh"
            )
        raw.seek(0)
        for chunk in iter(lambda: raw.read(1024 * 1024), b""):
            digest.update(chunk)

    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"{path}:{line_number}: invalid JSON") from exc
            if not isinstance(record, dict):
                raise RuntimeError(f"{path}:{line_number}: expected a JSON object")
            for field, value in record.items():
                field_types[field].add(_json_type(value))
            sample_count += 1

    if sample_count == 0:
        raise RuntimeError(f"{path} contains no samples")

    return {
        "sha256": digest.hexdigest(),
        "bytes": path.stat().st_size,
        "samples": sample_count,
        "schema": {
            field: sorted(types)
            for field, types in sorted(field_types.items())
        },
    }


def verify_manifest() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    for relative_path, expected in manifest["datasets"].items():
        actual = inspect_jsonl(ROOT_DIR / relative_path)
        if actual != expected:
            raise RuntimeError(
                f"Dataset manifest mismatch for {relative_path}: "
                f"expected {expected}, got {actual}"
            )


def verify_dependency_contract() -> None:
    import tomllib

    project = tomllib.loads((ROOT_DIR / "pyproject.toml").read_text(encoding="utf-8"))
    metadata = project["project"]
    dependencies = metadata["dependencies"]
    if any("==" not in dependency for dependency in dependencies):
        raise RuntimeError("Every baseline dependency must use an exact version")
    if "openai==1.109.1" not in dependencies:
        raise RuntimeError("openai must remain pinned to 1.109.1")
    if "Flask==3.1.1" not in dependencies:
        raise RuntimeError("Flask must be an explicit baseline dependency")
    if metadata["optional-dependencies"]["all"] != [
        "lgagent[retriever]",
        "lgagent[generation]",
        "lgagent[corpus]",
    ]:
        raise RuntimeError("The all extra must reference the local lgagent package")


def verify_credential_precedence() -> None:
    from tools.baseline_eval import load_generation_config as load_baseline_config
    from tools.legal_multi_agent_prompt_demo import (
        load_generation_config as load_multi_agent_config,
    )

    base = {
        "generation": {
            "backend": "openai",
            "backend_configs": {
                "openai": {
                    "api_key": "yaml-test-value",
                    "model_name": "test-model",
                }
            },
        }
    }
    with tempfile.TemporaryDirectory() as directory:
        config_path = Path(directory) / "config.yaml"
        config_path.write_text(yaml.safe_dump(base), encoding="utf-8")
        with patch.dict(os.environ, {"LLM_API_KEY": "env-test-value"}, clear=False):
            for loader in (load_multi_agent_config, load_baseline_config):
                configured = loader(config_path)
                if configured["api_key"] != "yaml-test-value":
                    raise RuntimeError("YAML API key must take precedence")
                if configured["api_key_source"] != "yaml":
                    raise RuntimeError("YAML credential source was not recorded")

                base["generation"]["backend_configs"]["openai"]["api_key"] = ""
                config_path.write_text(yaml.safe_dump(base), encoding="utf-8")
                fallback = loader(config_path)
                if fallback["api_key"] != "env-test-value":
                    raise RuntimeError("LLM_API_KEY must be used when YAML is empty")
                if fallback["api_key_source"] != "environment":
                    raise RuntimeError("Environment credential source was not recorded")

                base["generation"]["backend_configs"]["openai"]["api_key"] = (
                    "yaml-test-value"
                )
                config_path.write_text(yaml.safe_dump(base), encoding="utf-8")


def run_offline_checks() -> None:
    verify_manifest()
    verify_dependency_contract()
    verify_credential_precedence()
    print("Offline baseline checks passed.")


def run_online_smoke() -> None:
    from tools.legal_multi_agent_prompt_demo import (
        PARAM_PATH,
        build_client,
        load_generation_config,
        run_judge,
        run_lawyer_answer,
        run_lawyer_parser,
    )

    config = load_generation_config(PARAM_PATH)
    if not config["api_key"]:
        raise CredentialsUnavailable(
            "No API credential is configured; online smoke test was not run."
        )

    dataset_path = ROOT_DIR / "data" / "sample_single_legal.jsonl"
    with dataset_path.open("r", encoding="utf-8") as stream:
        sample = json.loads(next(line for line in stream if line.strip()))
    question = sample["question"]

    client = build_client(config["base_url"], config["api_key"])
    parsed, _ = run_lawyer_parser(client, config, question)
    judge_result, _, _ = run_judge(client, config, question, parsed)
    answer, details = run_lawyer_answer(
        client,
        config,
        question,
        parsed,
        judge_result,
        reasoning_client=client,
        gen_conf=config,
        enable_dialogue=False,
    )
    if not answer or str(answer).startswith("[API_ERROR"):
        raise RuntimeError("Online single-question smoke test returned no answer")
    if not isinstance(details, dict):
        raise RuntimeError("Online single-question smoke test returned invalid details")
    print("Online single-question smoke test passed.")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--online-smoke",
        action="store_true",
        help="Run one real question through the baseline API pipeline.",
    )
    args = parser.parse_args()

    run_offline_checks()
    if args.online_smoke:
        try:
            run_online_smoke()
        except CredentialsUnavailable as exc:
            print(str(exc), file=sys.stderr)
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
