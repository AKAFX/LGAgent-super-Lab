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

from lgagent.corpus import (
    CorpusValidationError,
    RelationType,
    load_jsonl_corpus,
)


def legal_record(**overrides: object) -> dict[str, object]:
    record: dict[str, object] = {
        "source_type": "statute",
        "law_name": "中华人民共和国民法典",
        "article": "第一条",
        "clause": None,
        "version": "2020",
        "text": "为了保护民事主体的合法权益，制定本法。",
        "jurisdiction": "CN",
        "authority_level": 5,
        "effective_from": "2021-01-01",
        "effective_to": None,
        "source_uri": "https://example.invalid/civil-code/1",
        "relations": [],
    }
    record.update(overrides)
    return record


def write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


class CorpusSchemaAndNormalizationTest(unittest.TestCase):
    def test_loads_all_relation_types_and_normalizes_fields(self) -> None:
        relations = [
            {"type": relation.value.lower(), "target_id": f"target:{index}"}
            for index, relation in enumerate(RelationType)
        ]
        with tempfile.TemporaryDirectory() as directory:
            corpus_path = Path(directory) / "corpus.jsonl"
            write_jsonl(
                corpus_path,
                [
                    legal_record(
                        law_name=" 中华人民共和国  民法典 ",
                        jurisdiction="cn",
                        relations=relations,
                    )
                ],
            )

            loaded = load_jsonl_corpus(corpus_path)

        evidence = loaded.evidence[0]
        self.assertEqual(evidence.law_name, "中华人民共和国 民法典")
        self.assertEqual(evidence.jurisdiction, "CN")
        self.assertEqual(
            {relation.type for relation in evidence.relations},
            set(RelationType),
        )
        self.assertRegex(evidence.evidence_id, r"^oath:[0-9a-f]{64}$")
        self.assertEqual(loaded.manifest.document_count, 1)
        self.assertEqual(loaded.manifest.relation_count, 4)

    def test_stable_id_and_manifest_hash_ignore_spacing_and_record_order(self) -> None:
        first = legal_record(article="第一条", text="条文  内容")
        second = legal_record(
            article="第二条",
            text="第二条内容",
            source_uri="https://example.invalid/civil-code/2",
        )
        normalized_first = legal_record(
            article=" 第一条 ",
            text=" 条文\n内容 ",
            source_type="STATUTE",
            jurisdiction="cn",
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            left_path = root / "left.jsonl"
            right_path = root / "right.jsonl"
            write_jsonl(left_path, [first, second])
            write_jsonl(right_path, [second, normalized_first])

            left = load_jsonl_corpus(left_path)
            right = load_jsonl_corpus(right_path)

        self.assertEqual(
            {item.evidence_id for item in left.evidence},
            {item.evidence_id for item in right.evidence},
        )
        self.assertEqual(
            left.manifest.content_sha256,
            right.manifest.content_sha256,
        )

    def test_manifest_is_written_as_deterministic_json(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corpus_path = root / "corpus.jsonl"
            manifest_path = root / "build" / "manifest.json"
            write_jsonl(corpus_path, [legal_record()])

            loaded = load_jsonl_corpus(corpus_path, manifest_path=manifest_path)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        self.assertEqual(manifest, loaded.manifest.as_dict())
        self.assertRegex(manifest["content_sha256"], r"^[0-9a-f]{64}$")

    def test_rejects_unknown_fields_invalid_dates_and_missing_source(self) -> None:
        cases = [
            (legal_record(extra="unexpected"), "unknown fields"),
            (legal_record(effective_from="2021/01/01"), "ISO date"),
            (
                {
                    key: value
                    for key, value in legal_record().items()
                    if key != "source_uri"
                },
                "missing fields",
            ),
        ]
        with tempfile.TemporaryDirectory() as directory:
            corpus_path = Path(directory) / "invalid.jsonl"
            for record, message in cases:
                with self.subTest(message=message):
                    write_jsonl(corpus_path, [record])
                    with self.assertRaisesRegex(CorpusValidationError, message):
                        load_jsonl_corpus(corpus_path)


class LegacyAndCorpusIntegrityTest(unittest.TestCase):
    def test_converts_legacy_contents_and_can_disable_compatibility(self) -> None:
        legacy = {
            "id": "中华人民共和国专利法-第一章-第一条",
            "contents": " 为了保护专利权人的合法权益，  制定本法。 ",
        }
        with tempfile.TemporaryDirectory() as directory:
            corpus_path = Path(directory) / "legacy.jsonl"
            write_jsonl(corpus_path, [legacy])

            loaded = load_jsonl_corpus(corpus_path)
            with self.assertRaisesRegex(
                CorpusValidationError, "legacy contents records are disabled"
            ):
                load_jsonl_corpus(corpus_path, allow_legacy=False)

        evidence = loaded.evidence[0]
        self.assertEqual(evidence.source_type, "legacy")
        self.assertEqual(evidence.law_name, "中华人民共和国专利法")
        self.assertEqual(evidence.article, "第一条")
        self.assertEqual(evidence.authority_level, 0)
        self.assertIsNone(evidence.effective_from)
        self.assertTrue(evidence.source_uri.startswith("legacy://"))
        self.assertEqual(loaded.manifest.legacy_document_count, 1)

    def test_detects_duplicate_clause_versions(self) -> None:
        duplicate = legal_record(
            text="重复来源中的同版本条文",
            source_uri="https://example.invalid/mirror/1",
        )
        with tempfile.TemporaryDirectory() as directory:
            corpus_path = Path(directory) / "duplicates.jsonl"
            write_jsonl(corpus_path, [legal_record(), duplicate])

            with self.assertRaisesRegex(
                CorpusValidationError, "duplicate clause version"
            ):
                load_jsonl_corpus(corpus_path)

    def test_detects_overlapping_version_intervals(self) -> None:
        old_version = legal_record(
            version="2017",
            effective_from="2017-10-01",
            effective_to="2021-01-01",
        )
        new_version = legal_record(
            version="2021",
            text="新版本条文",
            effective_from="2021-01-01",
            source_uri="https://example.invalid/civil-code/1/2021",
        )
        with tempfile.TemporaryDirectory() as directory:
            corpus_path = Path(directory) / "overlap.jsonl"
            write_jsonl(corpus_path, [old_version, new_version])

            with self.assertRaisesRegex(
                CorpusValidationError, "overlapping version intervals"
            ):
                load_jsonl_corpus(corpus_path)

    def test_accepts_adjacent_non_overlapping_versions(self) -> None:
        old_version = legal_record(
            version="2017",
            effective_from="2017-10-01",
            effective_to="2020-12-31",
        )
        new_version = legal_record(
            version="2021",
            text="新版本条文",
            effective_from="2021-01-01",
            source_uri="https://example.invalid/civil-code/1/2021",
        )
        with tempfile.TemporaryDirectory() as directory:
            corpus_path = Path(directory) / "versions.jsonl"
            write_jsonl(corpus_path, [old_version, new_version])

            loaded = load_jsonl_corpus(corpus_path)

        self.assertEqual(loaded.manifest.document_count, 2)


if __name__ == "__main__":
    unittest.main()
