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

from lgagent.cape_v import (
    CandidateOptionVerification,
    CompactIRAC,
    IRACApplication,
    IRACRule,
    ReasoningCandidate,
)
from lgagent.clex import (
    CLEX_SCORE_VERSION,
    ClexCalibration,
    ClexOptionScore,
    apply_calibration,
    conformal_quantile,
    fit_calibration,
    score_options,
)
from lgagent.risk import CandidateSignals


def candidate(candidate_id: str, answer: str, strength: float) -> CandidateSignals:
    reasoning = ReasoningCandidate(
        candidate_id=candidate_id,
        answer=answer,
        answer_identity=answer,
        presented_answer=answer,
        permutation_id="original",
        irac=CompactIRAC(
            issue="issue",
            rule=(IRACRule("rule", ("law:1",)),),
            application=(IRACApplication(("fact:1",), 0, "applies"),),
            conclusion=answer,
        ),
        option_verification={
            option: CandidateOptionVerification(
                status="SUPPORT" if option == answer else "REFUTE",
                score=0.9,
                evidence_ids=("law:1",),
            )
            for option in "ABCD"
        },
    )
    return CandidateSignals(
        candidate=reasoning,
        evidence_coverage=1.0,
        authority_score=1.0,
        temporal_validity=1.0,
        verifier_score=strength,
        permutation_consistency=1.0,
    )


def option_score(option: str, nonconformity: float) -> ClexOptionScore:
    return ClexOptionScore(
        option=option,
        conformity=1.0 - nonconformity,
        nonconformity=nonconformity,
        components={},
    )


class ClexScoringTest(unittest.TestCase):
    def test_scores_all_options_and_uses_verifier_strength(self) -> None:
        scores = score_options(
            (
                candidate("candidate-a", "A", 0.2),
                candidate("candidate-b", "B", 0.9),
            )
        )

        self.assertEqual(tuple(scores), tuple("ABCD"))
        self.assertGreater(scores["B"].conformity, scores["A"].conformity)
        self.assertGreater(scores["A"].conformity, scores["C"].conformity)
        self.assertTrue(
            all(
                abs(item.conformity + item.nonconformity - 1.0) < 1e-12
                for item in scores.values()
            )
        )

    def test_conformal_quantile_uses_finite_sample_correction(self) -> None:
        self.assertEqual(
            conformal_quantile([0.1, 0.2, 0.3, 0.4], alpha=0.25),
            0.4,
        )


class ClexCalibrationTest(unittest.TestCase):
    @staticmethod
    def records(count: int, *, split: str = "dev") -> list[dict[str, object]]:
        records = []
        for index in range(count):
            gold = "A" if index % 2 == 0 else "B"
            scores = {
                option: option_score(
                    option,
                    0.1 + index / 1000 if option == gold else 0.9,
                ).as_dict()
                for option in "ABCD"
            }
            records.append(
                {
                    "sample_id": f"sample-{index}",
                    "split": split,
                    "status": "ok",
                    "domain": "civil" if index < count // 2 else "criminal",
                    "golden_answers": [gold],
                    "clex_score_version": CLEX_SCORE_VERSION,
                    "clex_option_scores": scores,
                }
            )
        return records

    def test_fits_global_and_mondrian_thresholds(self) -> None:
        artifact = fit_calibration(
            self.records(40),
            alpha=0.1,
            min_calibration_size=30,
            min_group_size=20,
        )

        self.assertEqual(artifact.global_count, 40)
        self.assertEqual(artifact.group_counts, {"civil": 20, "criminal": 20})
        self.assertEqual(artifact.score_version, CLEX_SCORE_VERSION)
        self.assertEqual(len(artifact.records_sha256), 64)

    def test_rejects_non_development_records_and_small_sets(self) -> None:
        with self.assertRaisesRegex(ValueError, "development records only"):
            fit_calibration(
                self.records(30, split="test"),
                min_calibration_size=30,
            )
        with self.assertRaisesRegex(ValueError, "insufficient"):
            fit_calibration(self.records(2), min_calibration_size=30)

        failed = self.records(30)
        failed[0]["status"] = "failed"
        with self.assertRaisesRegex(ValueError, "failed records"):
            fit_calibration(failed, min_calibration_size=30)

        duplicated = self.records(30)
        duplicated[1]["sample_id"] = duplicated[0]["sample_id"]
        with self.assertRaisesRegex(ValueError, "unique sample_id"):
            fit_calibration(duplicated, min_calibration_size=30)

    def test_artifact_round_trip_and_cli(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "results.jsonl"
            source.write_text(
                "".join(
                    json.dumps(record) + "\n" for record in self.records(30)
                ),
                encoding="utf-8",
            )
            output = root / "calibration.json"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(ROOT_DIR / "tools/calibrate_clex.py"),
                    str(source),
                    "--output",
                    str(output),
                    "--min-group-size",
                    "15",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            loaded = ClexCalibration.load(output)

        self.assertEqual(loaded.global_count, 30)
        self.assertIn('"global_count": 30', completed.stdout)

    def test_offline_replay_preserves_input_and_makes_no_model_calls(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            calibration = fit_calibration(
                self.records(30),
                min_calibration_size=30,
                min_group_size=15,
            )
            calibration_path = root / "calibration.json"
            calibration_path.write_text(
                json.dumps(calibration.as_dict()),
                encoding="utf-8",
            )
            test_records = self.records(4, split="test")
            for record in test_records:
                record["prediction"] = "D"
                record["clex_option_scores"] = {
                    "A": option_score("A", 0.1).as_dict(),
                    "B": option_score("B", 0.8).as_dict(),
                    "C": option_score("C", 0.9).as_dict(),
                    "D": option_score("D", 0.95).as_dict(),
                }
            source = root / "test.jsonl"
            original = "".join(
                json.dumps(record) + "\n" for record in test_records
            )
            source.write_text(original, encoding="utf-8")
            output_dir = root / "replay"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(ROOT_DIR / "tools/evaluate_clex.py"),
                    str(source),
                    "--calibration",
                    str(calibration_path),
                    "--output-dir",
                    str(output_dir),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            replayed = [
                json.loads(line)
                for line in (output_dir / "results.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            summary = json.loads(
                (output_dir / "summary.json").read_text(encoding="utf-8")
            )
            preserved_source = source.read_text(encoding="utf-8")

        self.assertEqual(preserved_source, original)
        self.assertEqual(replayed[0]["pre_clex_prediction"], "D")
        self.assertEqual(replayed[0]["prediction"], "A")
        self.assertEqual(summary["model_calls"], 0)
        self.assertIn('"model_calls": 0', completed.stdout)


class ClexDecisionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.calibration = ClexCalibration(
            alpha=0.1,
            global_threshold=0.4,
            global_count=100,
            group_thresholds={"civil": 0.3},
            group_counts={"civil": 50},
        )
        self.scores = {
            "A": option_score("A", 0.55),
            "B": option_score("B", 0.20),
            "C": option_score("C", 0.70),
            "D": option_score("D", 0.80),
        }

    def test_applies_group_threshold_and_can_replace_current_answer(self) -> None:
        decision = apply_calibration(
            self.scores,
            self.calibration,
            current_answer="A",
            group="civil",
        )

        self.assertEqual(decision.prediction_set, ("B",))
        self.assertEqual(decision.selected_answer, "B")
        self.assertEqual(decision.threshold_source, "domain:civil")
        self.assertEqual(decision.eliminated, ("A", "C", "D"))
        self.assertTrue(decision.applied)

    def test_empty_set_and_insufficient_calibration_fail_closed(self) -> None:
        empty = apply_calibration(
            {
                option: option_score(option, 0.9)
                for option in "ABCD"
            },
            self.calibration,
            current_answer="A",
        )
        self.assertEqual(empty.selected_answer, "A")
        self.assertEqual(empty.fallback_reason, "empty_prediction_set")

        small = apply_calibration(
            self.scores,
            ClexCalibration(0.1, 0.4, 2),
            current_answer="A",
            min_calibration_size=30,
        )
        self.assertFalse(small.applied)
        self.assertEqual(small.fallback_reason, "insufficient_calibration")


if __name__ == "__main__":
    unittest.main()
