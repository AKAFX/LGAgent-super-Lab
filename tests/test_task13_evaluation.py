from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from lgagent.evaluation import (
    CheckpointError,
    ExperimentCheckpoint,
    ExperimentMetadata,
    UsageRecord,
    brier_score,
    build_result_record,
    calculate_usage_cost,
    citation_f1,
    counterfactual_sensitivity_specificity,
    evidence_recall_at_k,
    evaluate_result_records,
    expected_calibration_error,
    mcnemar_test,
    mean_reciprocal_rank,
    ndcg_at_k,
    paired_bootstrap,
    paired_significance_tests,
    permutation_consistency,
    reciprocal_rank,
    risk_coverage_curve,
    stable_hash,
    summarize_usage,
)


def metadata(seed: int = 42) -> ExperimentMetadata:
    return ExperimentMetadata(
        model_id="model-a",
        endpoint_class="openai-compatible",
        prompt_version="prompt-v1",
        config_hash="config-hash",
        dataset_hash="dataset-hash",
        random_seed=seed,
        software_version="0.1.0",
    )


class EvidenceMetricTest(unittest.TestCase):
    def test_retrieval_ranking_and_citation_metrics(self) -> None:
        retrieved = ["noise", "law-b", "law-a"]
        relevant = {"law-a", "law-b"}

        self.assertEqual(evidence_recall_at_k(retrieved, relevant, 2), 0.5)
        self.assertEqual(reciprocal_rank(retrieved, relevant), 0.5)
        self.assertEqual(
            mean_reciprocal_rank(
                [(retrieved, relevant), (["law-a"], {"law-a"})]
            ),
            0.75,
        )
        self.assertGreater(ndcg_at_k(retrieved, relevant, 3) or 0.0, 0.0)
        citations = citation_f1(["law-a", "fake"], relevant)
        self.assertEqual(citations.precision, 0.5)
        self.assertEqual(citations.recall, 0.5)
        self.assertEqual(citations.f1, 0.5)

    def test_undefined_metrics_are_explicit(self) -> None:
        self.assertIsNone(evidence_recall_at_k(["a"], [], 1))
        self.assertIsNone(reciprocal_rank(["a"], []))
        self.assertIsNone(ndcg_at_k(["a"], {}, 1))
        citations = citation_f1([], [])
        self.assertIsNone(citations.precision)
        self.assertIsNone(citations.recall)
        self.assertIsNone(citations.f1)
        missing_citation = citation_f1([], ["law-a"])
        self.assertIsNone(missing_citation.precision)
        self.assertEqual(missing_citation.recall, 0.0)
        self.assertEqual(missing_citation.f1, 0.0)


class RobustnessCalibrationCostTest(unittest.TestCase):
    def test_robustness_metrics(self) -> None:
        self.assertEqual(permutation_consistency(["B", "B", "A", "B"]), 0.75)
        result = counterfactual_sensitivity_specificity(
            [True, True, False, False],
            [True, False, False, False],
        )
        self.assertEqual(result.sensitivity, 0.5)
        self.assertEqual(result.specificity, 1.0)

    def test_calibration_and_risk_coverage(self) -> None:
        confidences = [0.9, 0.8]
        correctness = [True, False]
        self.assertAlmostEqual(
            expected_calibration_error(confidences, correctness, bins=2) or 0.0,
            0.35,
        )
        self.assertAlmostEqual(brier_score(confidences, correctness) or 0.0, 0.325)

        curve = risk_coverage_curve([0.2, 0.1], [False, True])
        self.assertEqual(curve.points[0].accuracy, 1.0)
        self.assertEqual(curve.points[-1].coverage, 1.0)
        self.assertEqual(curve.aurc, 0.25)

    def test_usage_summary_includes_calls_tokens_cost_and_latency(self) -> None:
        self.assertAlmostEqual(
            calculate_usage_cost(
                1_000_000,
                500_000,
                input_price_per_million=2.0,
                output_price_per_million=4.0,
            ),
            4.0,
        )
        summary = summarize_usage(
            [
                UsageRecord(2, 100, 20, 50.0, 0.01),
                {"calls": 1, "prompt_tokens": 5, "completion_tokens": 3,
                 "latency_ms": 10.0, "cost": 0.02},
            ]
        )
        self.assertEqual(summary["calls"], 3)
        self.assertEqual(summary["total_tokens"], 128)
        self.assertAlmostEqual(summary["cost"], 0.03)
        self.assertEqual(summary["latency_ms"], 60.0)
        with self.assertRaisesRegex(ValueError, "latency_ms cannot be negative"):
            summarize_usage([{"latency_ms": -1.0}])

    def test_full_record_report_keeps_legacy_and_extended_metrics(self) -> None:
        report = evaluate_result_records(
            [
                {
                    "prediction": "B",
                    "golden_answers": ["B"],
                    "domain": "civil",
                    "retrieved_evidence_ids": ["law-1"],
                    "relevant_evidence_ids": ["law-1"],
                    "cited_evidence_ids": ["law-1"],
                    "permutation_predictions": ["B", "B"],
                    "confidence": 0.8,
                    "risk_score": 0.2,
                    "usage": {"calls": 2, "total_tokens": 12},
                }
            ]
        )
        self.assertEqual(report["avg_acc"], 1.0)
        self.assertEqual(report["domain_accuracy"]["civil"], 1.0)
        self.assertEqual(report["evidence"]["recall@1"], 1.0)
        self.assertEqual(report["evidence"]["ndcg@1"], 1.0)
        self.assertEqual(report["evidence"]["ndcg@3"], 1.0)
        self.assertEqual(report["robustness"]["permutation_consistency"], 1.0)
        self.assertEqual(report["cost"]["total_tokens"], 12)

    def test_report_counts_missing_citations_as_zero(self) -> None:
        report = evaluate_result_records(
            [
                {
                    "prediction": "A",
                    "golden_answers": ["A"],
                    "retrieved_evidence_ids": ["law-1"],
                    "relevant_evidence_ids": ["law-1"],
                    "cited_evidence_ids": [],
                }
            ],
            recall_ks=(1,),
        )
        self.assertEqual(report["evidence"]["citation_f1"], 0.0)
        self.assertEqual(set(report["evidence"]), {"recall@1", "mrr", "ndcg@1", "citation_f1"})


class StatisticalTest(unittest.TestCase):
    def test_paired_bootstrap_is_seeded_and_detects_improvement(self) -> None:
        baseline = [0, 0, 0, 0]
        treatment = [1, 1, 1, 1]
        first = paired_bootstrap(baseline, treatment, iterations=999, seed=7)
        second = paired_bootstrap(baseline, treatment, iterations=999, seed=7)
        self.assertEqual(first, second)
        self.assertEqual(first.mean_difference, 1.0)
        self.assertEqual(first.confidence_interval, (1.0, 1.0))
        self.assertLess(first.p_value, 0.01)

    def test_exact_mcnemar_without_scipy(self) -> None:
        result = mcnemar_test(
            [True, False, False, False],
            [False, True, True, True],
        )
        self.assertEqual(result.baseline_only_correct, 1)
        self.assertEqual(result.treatment_only_correct, 3)
        self.assertAlmostEqual(result.p_value, 0.625)
        combined = paired_significance_tests(
            [True, False], [True, True], iterations=99
        )
        self.assertEqual(
            set(combined), {"paired_bootstrap", "mcnemar"}
        )


class CheckpointTest(unittest.TestCase):
    def test_experiment_id_is_stable_and_identity_sensitive(self) -> None:
        self.assertEqual(metadata().experiment_id, metadata().experiment_id)
        self.assertNotEqual(metadata().experiment_id, metadata(seed=43).experiment_id)
        self.assertEqual(stable_hash({"b": 2, "a": 1}), stable_hash({"a": 1, "b": 2}))

    def test_atomic_resume_deduplicates_and_rejects_wrong_experiment(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.json"
            checkpoint = ExperimentCheckpoint(path, metadata())
            record = build_result_record(
                sample_id="0:abc",
                prediction="B",
                golden_answers=["B"],
                candidates=[{"candidate_id": "c1", "answer": "B"}],
                evidence_matrix={"B": {"support": ["law-1"]}},
                verification_reports=[{"candidate_id": "c1"}],
                risk_routing={"route": "fast"},
                reproduction={"seed": 42},
                extra={"idx": 0},
            )
            self.assertTrue(checkpoint.add(record))
            self.assertFalse(checkpoint.add(record))
            changed = {**record, "prediction": "A"}
            with self.assertRaisesRegex(CheckpointError, "conflicting"):
                checkpoint.add(changed)
            checkpoint.save({"metrics": {"avg_acc": 1.0}})

            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload["experiment_id"], metadata().experiment_id)
            self.assertEqual(len(payload["examples"]), 1)
            resumed = ExperimentCheckpoint(path, metadata())
            self.assertEqual(len(resumed.load()), 1)
            self.assertFalse(resumed.add(record))
            with self.assertRaises(CheckpointError):
                ExperimentCheckpoint(path, metadata(seed=99)).load()

    def test_checkpoint_rejects_duplicate_records_on_disk(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.json"
            payload = {
                "experiment_id": metadata().experiment_id,
                "examples": [
                    {"sample_id": "same", "idx": 0},
                    {"sample_id": "same", "idx": 1},
                ],
            }
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(CheckpointError, "duplicate sample_id"):
                ExperimentCheckpoint(path, metadata()).load()

    def test_checkpoint_rejects_non_object_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.json"
            path.write_text("[]", encoding="utf-8")
            with self.assertRaisesRegex(CheckpointError, "root must be an object"):
                ExperimentCheckpoint(path, metadata()).load()

    def test_result_record_preserves_all_audit_artifacts(self) -> None:
        record = build_result_record(
            sample_id="0:abc",
            prediction="B",
            golden_answers=["B"],
            candidates=[{"candidate_id": "c1"}],
            evidence_matrix={"B": {"support": ["law-1"]}},
            verification_reports=[{"candidate_id": "c1", "passed": True}],
            risk_routing={"route": "slow"},
            usage=UsageRecord(calls=2, total_tokens=10, latency_ms=25.0),
            reproduction={"experiment_id": "experiment", "random_seed": 42},
        )
        self.assertEqual(
            {
                "candidates",
                "evidence_matrix",
                "verification_reports",
                "risk_routing",
                "usage",
                "reproduction",
            },
            set(record)
            - {"sample_id", "prediction", "golden_answers"},
        )
        with self.assertRaisesRegex(ValueError, "required result fields"):
            build_result_record(
                sample_id="0:abc",
                prediction="B",
                golden_answers=["B"],
                extra={"sample_id": "replacement"},
            )


if __name__ == "__main__":
    unittest.main()
