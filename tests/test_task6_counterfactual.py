from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from lgagent.cape_v import parse_four_option_question
from lgagent.counterfactual import (
    CounterfactualConfig,
    CounterfactualObservation,
    DecisiveFactChange,
    ProvenanceType,
    TransformationProvenance,
    calculate_counterfactual_metrics,
    decisive_fact_counterfactual,
    enabled_transformations,
    harmless_text_normalization,
    option_reorder_transformations,
)

QUESTION = """Which rule applies?

A. Alpha rule
B. Beta rule
C. Gamma rule
D. Delta rule"""


class LabelPreservingTransformationTest(unittest.TestCase):
    def test_reorder_maps_prediction_by_option_text_identity(self) -> None:
        transformations = option_reorder_transformations(
            QUESTION,
            original_answer="B",
            count=4,
            seed=17,
        )

        self.assertEqual(len(transformations), 4)
        self.assertEqual(
            len({item.transformed_question for item in transformations}),
            4,
        )
        for transformation in transformations:
            displayed = next(
                option.original_label
                for option in parse_four_option_question(
                    transformation.transformed_question
                ).options
                if option.identity == "Beta rule"
            )
            observation = CounterfactualObservation(transformation, displayed)
            self.assertEqual(observation.predicted_answer, "B")
            self.assertTrue(observation.consistent)
            self.assertTrue(transformation.is_metric_eligible)

    def test_harmless_normalization_preserves_expected_label(self) -> None:
        transformation = harmless_text_normalization(
            """Which   rule applies?

            Ａ． Alpha   rule
            Ｂ． Beta rule
            Ｃ． Gamma rule
            Ｄ． Delta rule""",
            original_answer="b",
        )

        self.assertEqual(transformation.expected_answer, "B")
        self.assertIn("A. Alpha rule", transformation.transformed_question)
        self.assertTrue(
            CounterfactualObservation(transformation, "B").consistent
        )


class CounterfactualSafetyAndMetricsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.fact_change = DecisiveFactChange(
            fact_id="f1",
            original_fact="The actor was under 16.",
            transformed_fact="The actor was 18.",
            legal_effect="The age element changes statutory applicability.",
        )

    def _changing(
        self,
        transformation_id: str,
        provenance: tuple[TransformationProvenance, ...],
        *,
        generated: bool,
    ):
        return decisive_fact_counterfactual(
            transformation_id=transformation_id,
            original_question=QUESTION,
            transformed_question=QUESTION.replace(
                "Which rule applies?", "The actor was 18. Which rule applies?"
            ),
            original_answer="B",
            expected_answer="C",
            decisive_fact=self.fact_change,
            provenance=provenance,
            generated=generated,
        )

    def test_unverified_generated_change_is_disabled_and_excluded(self) -> None:
        generated = self._changing(
            "generated-1",
            (
                TransformationProvenance(
                    provenance_id="llm-1",
                    source_type=ProvenanceType.LLM_GENERATED,
                    source_ref="generator-response-1",
                    verified=True,
                ),
            ),
            generated=True,
        )

        self.assertFalse(generated.is_metric_eligible)
        self.assertEqual(enabled_transformations([generated]), ())
        self.assertEqual(
            enabled_transformations(
                [generated],
                CounterfactualConfig(enable_generative=True),
            ),
            (generated,),
        )
        metrics = calculate_counterfactual_metrics(
            [CounterfactualObservation(generated, "C")]
        )
        self.assertIsNone(metrics.sensitivity)
        self.assertEqual(metrics.excluded_unverified_label_changing, 1)

    def test_only_verified_change_contributes_to_sensitivity_and_specificity(self) -> None:
        verified = self._changing(
            "verified-1",
            (
                TransformationProvenance(
                    provenance_id="evidence-1",
                    source_type=ProvenanceType.TRUSTED_EVIDENCE,
                    source_ref="law:article:version",
                    verified=True,
                ),
            ),
            generated=False,
        )
        unverified = self._changing(
            "unverified-1",
            (
                TransformationProvenance(
                    provenance_id="rule-1",
                    source_type=ProvenanceType.DETERMINISTIC_RULE,
                    source_ref="age-rule-v1",
                    verified=False,
                ),
            ),
            generated=False,
        )
        preserving = harmless_text_normalization(
            QUESTION,
            original_answer="B",
        )
        metrics = calculate_counterfactual_metrics(
            [
                CounterfactualObservation(verified, "C"),
                CounterfactualObservation(unverified, "C"),
                CounterfactualObservation(preserving, "B"),
            ]
        )

        self.assertEqual(metrics.sensitivity, 1.0)
        self.assertEqual(metrics.specificity, 1.0)
        self.assertEqual(metrics.should_change_total, 1)
        self.assertEqual(metrics.should_not_change_total, 1)
        self.assertEqual(metrics.excluded_unverified_label_changing, 1)


if __name__ == "__main__":
    unittest.main()
