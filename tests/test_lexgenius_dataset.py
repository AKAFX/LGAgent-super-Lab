from __future__ import annotations

import json
import sys
import unittest
from collections import Counter
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
for path in (ROOT_DIR, SRC_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from lgagent.cape_v import parse_four_option_question
from tools.build_lexgenius_dataset import (
    BUILD_ALGORITHM,
    DIMENSIONS,
    build_lexgenius,
    sha256_file,
)

DATASET_PATH = ROOT_DIR / "data/LexGenius.jsonl"
MANIFEST_PATH = ROOT_DIR / "data/LexGenius.manifest.json"
SOURCE_DIR = ROOT_DIR / "data/dimension_jsonl"


class LexGeniusDatasetTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.records = [
            json.loads(line)
            for line in DATASET_PATH.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        cls.manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

    def test_contains_eighty_valid_unique_questions_per_dimension(self) -> None:
        self.assertEqual(len(self.records), 560)
        self.assertEqual(
            Counter(record["domain"] for record in self.records),
            Counter({dimension: 80 for _, dimension, _ in DIMENSIONS}),
        )
        self.assertEqual(
            [record["id"] for record in self.records],
            list(range(560)),
        )
        self.assertEqual(
            len({" ".join(record["question"].split()) for record in self.records}),
            560,
        )
        for record in self.records:
            parse_four_option_question(record["question"])
            self.assertEqual(len(record["golden_answers"]), 1)
            self.assertIn(record["golden_answers"][0], "ABCD")
            self.assertEqual(record["meta_data"]["dataset"], "LexGenius")
            self.assertEqual(record["meta_data"]["domain"], record["domain"])

    def test_round_robin_order_keeps_dimension_prefixes_balanced(self) -> None:
        expected = [dimension_id for dimension_id, _, _ in DIMENSIONS]
        for start in range(0, len(self.records), 7):
            self.assertEqual(
                [
                    record["meta_data"]["dimension_id"]
                    for record in self.records[start : start + 7]
                ],
                expected,
            )

    def test_manifest_matches_sources_and_output(self) -> None:
        self.assertEqual(self.manifest["dataset"], "LexGenius")
        self.assertEqual(
            self.manifest["variant"],
            "balanced_80_per_dimension",
        )
        self.assertEqual(self.manifest["build_algorithm"], BUILD_ALGORITHM)
        self.assertEqual(self.manifest["seed"], 42)
        self.assertEqual(self.manifest["sample_per_dimension"], 80)
        self.assertEqual(self.manifest["total_samples"], 560)
        self.assertEqual(self.manifest["excluded_count"], 11)
        self.assertEqual(
            self.manifest["output"]["sha256"],
            sha256_file(DATASET_PATH),
        )
        for source in self.manifest["dimensions"]:
            self.assertEqual(
                source["sha256"],
                sha256_file(ROOT_DIR / source["path"]),
            )

    def test_rebuild_is_deterministic(self) -> None:
        rebuilt, manifest = build_lexgenius(
            SOURCE_DIR,
            sample_per_dimension=80,
            seed=42,
        )
        self.assertEqual(rebuilt, self.records)
        self.assertEqual(manifest["excluded_count"], 11)
        self.assertEqual(
            [item["selected_source_ids"] for item in manifest["dimensions"]],
            [
                item["selected_source_ids"]
                for item in self.manifest["dimensions"]
            ],
        )


if __name__ == "__main__":
    unittest.main()
