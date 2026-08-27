from __future__ import annotations

import json
import sys
import tempfile
import unittest
from datetime import date
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from lgagent.corpus import RelationType, load_jsonl_corpus
from lgagent.oath_rag import (
    DenseHit,
    OathRagConfig,
    OathRagRetriever,
    build_option_queries,
    reciprocal_rank_fusion,
)


def legal_record(evidence_id: str, article: str, text: str, **overrides: object) -> dict[str, object]:
    record: dict[str, object] = {
        "evidence_id": evidence_id,
        "source_type": "statute",
        "law_name": "Test Act",
        "article": article,
        "clause": None,
        "version": "2021",
        "text": text,
        "jurisdiction": "CN",
        "authority_level": 5,
        "effective_from": "2021-01-01",
        "effective_to": None,
        "source_uri": f"https://example.invalid/{evidence_id}",
        "relations": [],
    }
    record.update(overrides)
    return record


def load_records(records: list[dict[str, object]]):
    directory = tempfile.TemporaryDirectory()
    path = Path(directory.name) / "corpus.jsonl"
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    return directory, load_jsonl_corpus(path)


def analysis(claim: object = "alpha ownership") -> dict[str, object]:
    return {
        "legal_domain": "property law",
        "jurisdiction": "CN",
        "case_date": "2024-01-01",
        "question_focus": "ownership",
        "facts": [
            {"id": "f1", "text": "alpha transfer", "legally_relevant": True},
            {"id": "f2", "text": "secret answer B", "legally_relevant": False},
        ],
        "option_claims": {"A": claim},
        "predicted_answer": "B",
    }


class QueryAndFusionTest(unittest.TestCase):
    def test_builds_three_answer_neutral_queries_from_structured_claim(self) -> None:
        queries = build_option_queries(
            analysis(
                {
                    "claim": "transfer is valid",
                    "elements": ["registration"],
                    "possible_exceptions": ["good faith acquisition"],
                }
            )
        )["A"]

        self.assertIn("transfer is valid", queries.support)
        self.assertIn("registration", queries.refute)
        self.assertIn("good faith acquisition", queries.exception)
        self.assertNotIn("secret answer B", queries.support)
        self.assertNotIn("predicted_answer", " ".join(vars(queries).values()))

    def test_rrf_combines_rankings_and_ignores_duplicate_ids_per_ranking(self) -> None:
        fused = reciprocal_rank_fusion(
            (
                (DenseHit("a", 9.0), DenseHit("b", 8.0), DenseHit("a", 7.0)),
                (DenseHit("b", 0.9), DenseHit("c", 0.8)),
            ),
            rrf_k=10,
        )

        self.assertGreater(fused["b"], fused["a"])
        self.assertGreater(fused["a"], fused["c"])


class HardFilterAndHybridTest(unittest.TestCase):
    def test_lexical_only_selects_date_valid_jurisdiction_and_authority(self) -> None:
        temporary, corpus = load_records(
            [
                legal_record(
                    "old",
                    "1",
                    "alpha ownership historical rule",
                    version="2017",
                    effective_from="2017-01-01",
                    effective_to="2020-12-31",
                ),
                legal_record(
                    "current",
                    "1",
                    "alpha ownership current rule",
                    version="2021",
                ),
                legal_record(
                    "foreign",
                    "2",
                    "alpha ownership foreign rule",
                    jurisdiction="US",
                ),
                legal_record(
                    "weak",
                    "3",
                    "alpha ownership commentary",
                    authority_level=1,
                ),
            ]
        )
        self.addCleanup(temporary.cleanup)
        retriever = OathRagRetriever(
            corpus,
            config=OathRagConfig(min_authority_level=2, graph_hops=0),
            current_date=date(2024, 1, 1),
        )

        historical = retriever.retrieve(
            analysis(), case_date="2019-01-01"
        )["A"]["support"]
        current = retriever.retrieve(analysis())["A"]["support"]

        self.assertEqual({item.evidence_id for item in historical}, {"old"})
        self.assertEqual({item.evidence_id for item in current}, {"current"})

    def test_optional_dense_results_are_fused_and_hard_filtered(self) -> None:
        class FakeDense:
            def __init__(self) -> None:
                self.calls = 0

            def search(self, query: str, *, candidate_ids, top_k: int):
                self.calls += 1
                self.last_candidate_ids = tuple(candidate_ids)
                return (
                    DenseHit("dense", 0.99),
                    DenseHit("foreign", 1.0),
                )

        temporary, corpus = load_records(
            [
                legal_record("lexical", "1", "alpha transfer"),
                legal_record(
                    "dense",
                    "2",
                    "alpha ownership registration good-faith-acquisition",
                ),
                legal_record(
                    "foreign",
                    "3",
                    "alpha ownership foreign",
                    jurisdiction="US",
                ),
            ]
        )
        self.addCleanup(temporary.cleanup)
        dense = FakeDense()
        retriever = OathRagRetriever(
            corpus,
            dense_retriever=dense,
            config=OathRagConfig(final_top_k_per_lane=3, graph_hops=0),
        )

        dense_analysis = {
            "legal_domain": "property",
            "jurisdiction": "CN",
            "case_date": "2024-01-01",
            "question_focus": "ownership",
            "facts": [],
            "option_claims": {
                "A": {
                    "claim": "alpha registration",
                    "elements": ["good-faith-acquisition"],
                }
            },
        }
        lane = retriever.retrieve(dense_analysis)["A"]["support"]

        self.assertEqual(dense.calls, 3)
        self.assertNotIn("foreign", dense.last_candidate_ids)
        self.assertTrue(any(item.evidence_id == "dense" and item.dense_score > 0 for item in lane))
        self.assertNotIn("foreign", {item.evidence_id for item in lane})


class GraphExpansionAndSelectionTest(unittest.TestCase):
    def test_expands_each_supported_relation_type_by_one_hop(self) -> None:
        for relation_type in RelationType:
            with self.subTest(relation_type=relation_type):
                source = legal_record(
                    "source",
                    "1",
                    "alpha ownership",
                    relations=[{"type": relation_type.value, "target_id": "target"}],
                )
                target = legal_record("target", "2", "omega supplemental provision")
                temporary, corpus = load_records([source, target])
                try:
                    retriever = OathRagRetriever(
                        corpus,
                        config=OathRagConfig(final_top_k_per_lane=2),
                    )
                    lane = retriever.retrieve(analysis())["A"]["support"]
                finally:
                    temporary.cleanup()

                by_id = {item.evidence_id: item for item in lane}
                self.assertIn("target", by_id)
                self.assertEqual(by_id["target"].expanded_from, "source")
                self.assertEqual(by_id["target"].relation_type, relation_type)

    def test_deduplicates_equal_text_and_respects_evidence_budget(self) -> None:
        temporary, corpus = load_records(
            [
                legal_record("first", "1", "alpha ownership transfer registration"),
                legal_record("duplicate", "2", "alpha ownership transfer registration"),
                legal_record("extra", "3", "alpha ownership separate provision"),
            ]
        )
        self.addCleanup(temporary.cleanup)
        retriever = OathRagRetriever(
            corpus,
            config=OathRagConfig(final_top_k_per_lane=2, graph_hops=0),
        )

        lane = retriever.retrieve(analysis())["A"]["support"]

        self.assertLessEqual(len(lane), 2)
        self.assertEqual(
            len({item.evidence.text.casefold() for item in lane}),
            len(lane),
        )


if __name__ == "__main__":
    unittest.main()
