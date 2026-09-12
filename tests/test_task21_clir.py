from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from lgagent.clir import (
    CLIR_BASELINE_ACTION,
    ClirCalibration,
    ClirRouter,
    InterventionOutcome,
    InterventionRecord,
    build_intervention_records,
    fit_clir_calibration,
    hoeffding_lower_bound,
    replay_clir_policy,
)


def result_record(
    index: int,
    *,
    split: str = "dev",
    prediction: str = "A",
    gold: str = "A",
    status: str = "ok",
    tokens: int = 100,
    domain: str = "civil",
) -> dict[str, object]:
    return {
        "sample_id": f"sample-{index}",
        "idx": index,
        "split": split,
        "status": status,
        "domain": domain,
        "question": f"question-{index}",
        "prediction": prediction if status == "ok" else "",
        "golden_answers": [gold],
        "b0_confidence": 0.5,
        "usage": {
            "calls": 4,
            "prompt_tokens": max(0, tokens - 10),
            "completion_tokens": min(10, tokens),
            "total_tokens": tokens,
            "latency_ms": 20.0,
        },
    }


def intervention_record(
    index: int,
    *,
    split: str = "dev",
    group: str = "civil",
    baseline_correct: bool = False,
    web_correct: bool = True,
    web_tokens: int = 120,
) -> InterventionRecord:
    gold = "A"
    return InterventionRecord(
        sample_id=f"sample-{index}",
        split=split,
        group=group,
        golden_answers=(gold,),
        question=f"question-{index}",
        baseline=InterventionOutcome(
            action=CLIR_BASELINE_ACTION,
            prediction=gold if baseline_correct else "B",
            correct=baseline_correct,
            status="ok",
            usage={"total_tokens": 100},
        ),
        interventions={
            "web-search": InterventionOutcome(
                action="web-search",
                prediction=gold if web_correct else "B",
                correct=web_correct,
                status="ok",
                usage={"total_tokens": web_tokens},
            )
        },
        features={"domain": group},
    )


class ClirDatasetTest(unittest.TestCase):
    def test_builds_strict_pairs_and_keeps_failed_action(self) -> None:
        baseline = [result_record(0), result_record(1)]
        web = [
            result_record(0, prediction="B", tokens=130),
            result_record(1, status="failed", tokens=0),
        ]

        paired = build_intervention_records(
            baseline,
            {"web-search": web},
        )

        self.assertEqual(len(paired), 2)
        self.assertFalse(paired[0].interventions["web-search"].correct)
        self.assertEqual(
            paired[1].interventions["web-search"].status,
            "failed",
        )
        self.assertFalse(paired[1].interventions["web-search"].correct)

        designated = build_intervention_records(
            baseline,
            {"web-search": web},
            designated_split="dev",
        )
        self.assertTrue(all(record.split == "dev" for record in designated))
        self.assertTrue(
            all(
                record.provenance["source_split"] == "dev"
                for record in designated
            )
        )

    def test_rejects_missing_or_misaligned_samples(self) -> None:
        baseline = [result_record(0), result_record(1)]
        with self.assertRaisesRegex(ValueError, "sample IDs differ"):
            build_intervention_records(
                baseline,
                {"web-search": [result_record(0)]},
            )

        changed = result_record(1)
        changed["question"] = "different"
        with self.assertRaisesRegex(ValueError, "question differs"):
            build_intervention_records(
                baseline,
                {"web-search": [result_record(0), changed]},
            )

        baseline[0]["reproduction"] = {
            "model_id": "model-a",
            "random_seed": 42,
        }
        web = [result_record(0), result_record(1)]
        web[0]["reproduction"] = {
            "model_id": "model-b",
            "random_seed": 42,
        }
        with self.assertRaisesRegex(
            ValueError,
            "reproduction model_id differs",
        ):
            build_intervention_records(
                baseline,
                {"web-search": web},
            )


class ClirCalibrationTest(unittest.TestCase):
    def test_hoeffding_bound_is_conservative(self) -> None:
        self.assertLess(hoeffding_lower_bound([1] * 30, alpha=0.1), 1.0)
        self.assertGreater(hoeffding_lower_bound([1] * 30, alpha=0.1), 0.0)
        self.assertLess(hoeffding_lower_bound([-1] * 30, alpha=0.1), 0.0)

    def test_routes_by_group_and_falls_back_for_harmful_action(self) -> None:
        records = [
            intervention_record(index, group="civil")
            for index in range(20)
        ] + [
            intervention_record(
                index,
                group="criminal",
                baseline_correct=True,
                web_correct=False,
            )
            for index in range(20, 40)
        ]
        calibration = fit_clir_calibration(
            records,
            min_calibration_size=30,
            min_group_size=20,
        )
        router = ClirRouter(calibration)

        civil = router.decide(group="civil")
        criminal = router.decide(group="criminal")
        unknown = router.decide(group="unknown")

        self.assertEqual(civil.selected_action, "web-search")
        self.assertTrue(civil.applied)
        self.assertEqual(civil.estimate_source, "domain:civil")
        self.assertEqual(criminal.selected_action, CLIR_BASELINE_ACTION)
        self.assertFalse(criminal.applied)
        self.assertEqual(unknown.selected_action, CLIR_BASELINE_ACTION)

    def test_rejects_test_calibration_and_incomplete_actions(self) -> None:
        records = [
            intervention_record(index, split="test")
            for index in range(30)
        ]
        with self.assertRaisesRegex(ValueError, "development records only"):
            fit_clir_calibration(records, min_calibration_size=30)

        records = [
            intervention_record(index) for index in range(30)
        ]
        records[0] = InterventionRecord(
            sample_id=records[0].sample_id,
            split="dev",
            group="civil",
            golden_answers=("A",),
            baseline=records[0].baseline,
            interventions={
                "symbolic-check": InterventionOutcome(
                    action="symbolic-check",
                    prediction="A",
                    correct=True,
                    status="ok",
                )
            },
        )
        with self.assertRaisesRegex(ValueError, "identical actions"):
            fit_clir_calibration(records, min_calibration_size=30)

    def test_artifact_round_trip_and_cost_penalty(self) -> None:
        calibration = fit_clir_calibration(
            [intervention_record(index, web_tokens=100_100) for index in range(30)],
            min_calibration_size=30,
        )
        loaded = ClirCalibration.from_dict(calibration.as_dict())

        self.assertEqual(loaded.records_sha256, calibration.records_sha256)
        self.assertEqual(
            ClirRouter(loaded).decide(group="civil").selected_action,
            "web-search",
        )
        self.assertEqual(
            ClirRouter(
                loaded,
                cost_weight=1.0,
                token_scale=100_000.0,
            ).decide(group="civil").selected_action,
            CLIR_BASELINE_ACTION,
        )


class ClirReplayTest(unittest.TestCase):
    def test_replay_reports_rescue_harm_and_token_delta(self) -> None:
        calibration = fit_clir_calibration(
            [intervention_record(index) for index in range(30)],
            min_calibration_size=30,
        )
        test_records = [
            intervention_record(
                index,
                split="test",
                baseline_correct=False,
                web_correct=True,
            )
            for index in range(4)
        ]

        replayed, summary = replay_clir_policy(test_records, calibration)

        self.assertEqual(len(replayed), 4)
        self.assertEqual(summary["baseline_accuracy"], 0.0)
        self.assertEqual(summary["routed_accuracy"], 1.0)
        self.assertEqual(summary["corrected_count"], 4)
        self.assertEqual(summary["harmed_count"], 0)
        self.assertEqual(summary["action_counts"], {"web-search": 4})
        self.assertEqual(summary["token_difference"], 80)

    def test_three_cli_pipeline_is_zero_call_and_atomic(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline_path = root / "baseline.jsonl"
            action_path = root / "web.jsonl"
            dataset_path = root / "interventions.jsonl"
            calibration_path = root / "calibration.json"
            output_dir = root / "replay"

            baseline_records = []
            web_records = []
            for index in range(35):
                split = "dev" if index < 30 else "test"
                baseline_records.append(
                    result_record(
                        index,
                        split=split,
                        prediction="B",
                        gold="A",
                        tokens=100,
                    )
                )
                web_records.append(
                    result_record(
                        index,
                        split=split,
                        prediction="A",
                        gold="A",
                        tokens=120,
                    )
                )
            baseline_path.write_text(
                "".join(json.dumps(item) + "\n" for item in baseline_records),
                encoding="utf-8",
            )
            action_path.write_text(
                "".join(json.dumps(item) + "\n" for item in web_records),
                encoding="utf-8",
            )

            subprocess.run(
                [
                    sys.executable,
                    str(ROOT_DIR / "tools/build_clir_dataset.py"),
                    "--baseline",
                    str(baseline_path),
                    "--action",
                    f"web-search={action_path}",
                    "--output",
                    str(dataset_path),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT_DIR / "tools/calibrate_clir.py"),
                    str(dataset_path),
                    "--output",
                    str(calibration_path),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            completed = subprocess.run(
                [
                    sys.executable,
                    str(ROOT_DIR / "tools/evaluate_clir.py"),
                    str(dataset_path),
                    "--calibration",
                    str(calibration_path),
                    "--output-dir",
                    str(output_dir),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            summary = json.loads(
                (output_dir / "summary.json").read_text(encoding="utf-8")
            )

        self.assertEqual(summary["model_calls"], 0)
        self.assertEqual(summary["clir"]["routed_accuracy"], 1.0)
        self.assertIn('"model_calls": 0', completed.stdout)


if __name__ == "__main__":
    unittest.main()
