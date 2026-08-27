from __future__ import annotations

import csv
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
for path in (ROOT_DIR, SRC_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from lgagent.ablation import (
    EVIDENCE_LANES,
    REQUIRED_EXPERIMENT_KEYS,
    AblationMatrixError,
    build_task14_matrix,
    export_task14_matrix,
    task14_matrix_payload,
    validate_task14_matrix,
)
from lgagent.config import load_lgagent_config

PARAM_PATH = ROOT_DIR / "examples" / "parameter" / "legal2_rag_parameter.yaml"


def matrix():
    return build_task14_matrix(load_lgagent_config(PARAM_PATH, environ={}))


class Task14MatrixTest(unittest.TestCase):
    def test_enumerates_complete_matrix_deterministically(self) -> None:
        first = matrix()
        second = matrix()

        self.assertEqual(first, second)
        self.assertEqual(tuple(item.key for item in first), REQUIRED_EXPERIMENT_KEYS)
        self.assertEqual(task14_matrix_payload(first), task14_matrix_payload(second))
        self.assertEqual(len(first), 12)

    def test_main_configs_and_single_factor_ablations(self) -> None:
        by_key = {item.key: item for item in matrix()}

        self.assertEqual(
            (
                by_key["original"].lgagent_plus_enabled,
                by_key["original"].oath_rag_enabled,
                by_key["original"].cape_v_enabled,
            ),
            (False, False, False),
        )
        self.assertEqual(
            (
                by_key["oath-only"].oath_rag_enabled,
                by_key["oath-only"].cape_v_enabled,
            ),
            (True, False),
        )
        self.assertEqual(
            (
                by_key["cape-only"].oath_rag_enabled,
                by_key["cape-only"].cape_v_enabled,
            ),
            (False, True),
        )
        self.assertEqual(by_key["joint"].evidence_lanes, EVIDENCE_LANES)
        self.assertEqual(
            by_key["no-refute"].evidence_lanes, ("support", "exception")
        )
        self.assertEqual(
            by_key["no-exception"].evidence_lanes, ("support", "refute")
        )
        self.assertFalse(by_key["no-temporal"].require_temporal_match)
        self.assertEqual(by_key["no-graph-expansion"].graph_hops, 0)
        self.assertEqual(by_key["no-permutation"].permutation_count, 0)
        self.assertEqual(
            by_key["single-verifier"].enabled_verifiers, ("entailment",)
        )
        self.assertEqual(by_key["fixed-budget"].budget_policy, "fixed")

    def test_self_consistency_uses_joint_token_ceiling(self) -> None:
        by_key = {item.key: item for item in matrix()}
        joint = by_key["joint"]
        control = by_key["self-consistency-equal-token"]

        self.assertEqual(control.equal_token_reference, "joint")
        self.assertEqual(control.max_total_tokens, joint.max_total_tokens)
        self.assertLessEqual(
            control.self_consistency_samples
            * control.self_consistency_max_tokens,
            joint.max_total_tokens,
        )
        self.assertEqual(
            {item.max_total_tokens for item in by_key.values()},
            {joint.max_total_tokens},
        )

    def test_validation_rejects_duplicate_and_non_isolated_changes(self) -> None:
        experiments = matrix()
        with self.assertRaisesRegex(AblationMatrixError, "unique"):
            validate_task14_matrix((*experiments[:-1], experiments[0]))

        changed = list(experiments)
        index = REQUIRED_EXPERIMENT_KEYS.index("no-refute")
        changed[index] = replace(changed[index], graph_hops=0)
        with self.assertRaisesRegex(AblationMatrixError, "differ from joint only"):
            validate_task14_matrix(changed)

    def test_exports_json_csv_and_markdown_without_secrets(self) -> None:
        config = load_lgagent_config(PARAM_PATH, environ={})
        config = replace(
            config,
            generation=replace(config.generation, api_key="never-export-this"),
        )
        experiments = build_task14_matrix(config)

        with tempfile.TemporaryDirectory() as directory:
            paths = export_task14_matrix(experiments, directory)
            self.assertEqual(set(paths), {"json", "csv", "markdown"})
            payload = json.loads(paths["json"].read_text(encoding="utf-8"))
            self.assertEqual(payload["experiment_count"], 12)
            self.assertTrue(payload["offline_only"])
            with paths["csv"].open(encoding="utf-8", newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 12)
            markdown = paths["markdown"].read_text(encoding="utf-8")
            self.assertIn("Self-Consistency (equal-token)", markdown)
            for path in paths.values():
                self.assertNotIn(
                    "never-export-this", path.read_text(encoding="utf-8")
                )

    def test_cli_matrix_mode_does_not_build_an_api_client(self) -> None:
        from tools import legal_multi_agent_ablation_auto as cli

        with tempfile.TemporaryDirectory() as directory:
            argv = [
                "legal_multi_agent_ablation_auto.py",
                "--matrix-only",
                "--matrix-config",
                str(PARAM_PATH),
                "--output-dir",
                directory,
            ]
            with (
                patch.object(sys, "argv", argv),
                patch.object(
                    cli,
                    "build_client",
                    side_effect=AssertionError("API client must not be created"),
                ),
                redirect_stdout(io.StringIO()) as stdout,
            ):
                cli.main()

            self.assertIn("12 个配置", stdout.getvalue())
            self.assertTrue(
                (Path(directory) / "task14_ablation_matrix.json").exists()
            )

    def test_legacy_cli_arguments_remain_available(self) -> None:
        from tools import legal_multi_agent_ablation_auto as cli

        args = cli.parse_args(
            [
                "--datasets",
                "data/example.jsonl",
                "--eval-model",
                "model-a",
                "--eval-base-url",
                "https://example.invalid/v1",
                "--concurrency",
                "2",
                "--max-examples",
                "3",
                "--skip-configs",
                "ablation-none",
            ]
        )
        self.assertFalse(args.matrix_only)
        self.assertEqual(args.datasets, ["data/example.jsonl"])
        self.assertEqual(args.eval_model, "model-a")
        self.assertEqual(args.concurrency, 2)
        self.assertEqual(args.skip_configs, ["ablation-none"])


if __name__ == "__main__":
    unittest.main()
