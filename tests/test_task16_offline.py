from __future__ import annotations

import json
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
for path in (ROOT_DIR, SRC_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from lgagent.ablation import build_task14_matrix
from lgagent.config import load_lgagent_config
from lgagent.corpus import LegalEvidence, load_jsonl_corpus
from lgagent.corpus_builder import (
    JUST_LAWS_COMMIT,
    build_just_laws_corpus,
    write_corpus_artifacts,
)
from lgagent.experiment_freeze import (
    EXPERIMENT_PROFILES,
    FrozenSample,
    assigned_split,
    validate_freeze_manifest,
)
from lgagent.experiment_runner import (
    _ApiKeyLeasePool,
    ExperimentRunError,
    Task16ExperimentRunner,
    _SharedRetrievalResources,
    _cited_evidence_ids,
    _config_with_api_key,
    _config_with_model,
    _evidence_ids,
    _load_api_key_pool,
    _model_clients,
    _run_sample,
    apply_experiment,
    estimate_calls,
    extract_sample_domain,
)
from lgagent.oath_rag import BM25Index
from tools.run_task16_experiments import parse_args, resolve_plan

CONFIG_PATH = ROOT_DIR / "examples/parameter/legal2_rag_parameter.yaml"
FREEZE_PATH = ROOT_DIR / "docs/task16_experiment_freeze.json"


def write_law(root: Path, law_id: str, title: str, body: str) -> None:
    law_dir = root / "docs" / law_id
    law_dir.mkdir(parents=True)
    (law_dir / "README.md").write_text(
        f"# {title}\n\n{body}\n", encoding="utf-8"
    )


class CorpusBuilderTest(unittest.TestCase):
    def test_builds_strict_records_and_reports_unknown_dates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "docs" / "category").mkdir(parents=True)
            (root / "LICENSE").write_text("MIT fixture", encoding="utf-8")
            (root / "docs" / "category" / "economic.md").write_text(
                "[甲法](../economic/law-a/)\n"
                "[乙法](../economic/law-b/)\n"
                "[丙法](../economic/law-c/)\n",
                encoding="utf-8",
            )
            write_law(
                root,
                "economic/law-a",
                "中华人民共和国甲法",
                "**第一条**　依照《中华人民共和国乙法》处理。\n\n"
                "**第二条**　本法自2024年1月2日起施行。",
            )
            write_law(
                root,
                "economic/law-b",
                "中华人民共和国乙法",
                "**第一条**　乙法规则。\n\n"
                "**第二条**　本法自二〇二三年十二月一日起施行。",
            )
            write_law(
                root,
                "economic/law-c",
                "中华人民共和国丙法",
                "**第一条**　公布之日起施行。",
            )

            with patch(
                "lgagent.corpus_builder.verify_source_commit",
                return_value=root,
            ):
                result = build_just_laws_corpus(root)

            self.assertEqual(len(result.records), 4)
            self.assertEqual(len(result.excluded), 1)
            self.assertIn("no unique explicit", result.excluded[0].reason)
            self.assertEqual({item.authority_level for item in result.records}, {5})
            self.assertTrue(
                all(JUST_LAWS_COMMIT in item.source_uri for item in result.records)
            )
            referring = next(
                item for item in result.records if item.law_name.endswith("甲法")
                and item.article == "第一条"
            )
            self.assertEqual(len(referring.relations), 1)

            paths = write_corpus_artifacts(result, root / "out", source_dir=root)
            loaded = load_jsonl_corpus(paths["corpus"], allow_legacy=False)
            self.assertEqual(loaded.manifest.document_count, 4)
            provenance = json.loads(paths["provenance"].read_text(encoding="utf-8"))
            self.assertEqual(provenance["commit"], JUST_LAWS_COMMIT)


class FreezeAndRunnerTest(unittest.TestCase):
    def test_api_key_pool_loads_deduplicated_keys_and_fallback(self) -> None:
        self.assertEqual(
            _load_api_key_pool(
                {"LGAGENT_REASONING_API_KEYS": " key-a,key-b,key-a ,, "}
            ),
            ("key-a", "key-b"),
        )
        self.assertEqual(
            _load_api_key_pool({}, fallback_key="single-key"),
            ("single-key",),
        )

    def test_api_key_pool_leases_exclusive_slots(self) -> None:
        pool = _ApiKeyLeasePool(("key-a", "key-b"))
        self.assertEqual(pool.size, 2)
        with pool.lease() as first:
            with pool.lease() as second:
                self.assertNotEqual(first[0], second[0])
                self.assertEqual({first[1], second[1]}, {"key-a", "key-b"})
        with pool.lease() as returned:
            self.assertIn(returned[1], {"key-a", "key-b"})

    def test_key_pool_override_does_not_change_noncredential_config(self) -> None:
        config = load_lgagent_config(
            CONFIG_PATH,
            environ={"LLM_API_KEY": "fallback-key"},
        )
        replaced_config = _config_with_api_key(config, "pool-key")

        self.assertEqual(replaced_config.generation.api_key, "pool-key")
        self.assertEqual(replaced_config.generation.api_key_source, "key_pool")
        self.assertEqual(
            replaced_config.generation.model,
            config.generation.model,
        )
        self.assertEqual(config.generation.api_key, "fallback-key")

    def test_model_override_updates_generation_and_verifier(self) -> None:
        config = load_lgagent_config(
            CONFIG_PATH,
            environ={"LLM_API_KEY": "test-key"},
        )
        replaced_config = _config_with_model(config, "qwen3-4b")

        self.assertEqual(replaced_config.generation.model, "qwen3-4b")
        self.assertEqual(
            replaced_config.lgagent_plus.cape_v.verifier_model.model,
            "qwen3-4b",
        )
        self.assertEqual(config.generation.model, "deepseek-v3")

    def test_domain_extraction_preserves_specific_field_priority(self) -> None:
        record = {
            "domain": "top-domain",
            "subject": "top-subject",
            "meta_data": {
                "domain": "meta-domain",
                "subject": "meta-subject",
                "source_file": "1-1.json",
                "type": "multiple-choice",
            },
        }
        self.assertEqual(extract_sample_domain(record), "meta-domain")
        del record["meta_data"]["domain"]
        self.assertEqual(extract_sample_domain(record), "top-domain")
        del record["domain"]
        self.assertEqual(extract_sample_domain(record), "meta-subject")
        del record["meta_data"]["subject"]
        self.assertEqual(extract_sample_domain(record), "top-subject")
        del record["subject"]
        self.assertEqual(extract_sample_domain(record), "multiple-choice")
        del record["meta_data"]["type"]
        self.assertEqual(
            extract_sample_domain(record, "model-inferred-domain"),
            "model-inferred-domain",
        )
        self.assertEqual(extract_sample_domain(record), "1-1.json")

    def test_domain_extraction_covers_current_dataset_shapes(self) -> None:
        records = {
            "Ability_merged": {
                "meta_data": {"correct_option": "A", "original_index": 1}
            },
            "CAIL2022": {
                "meta_data": {"correct_option": "B", "original_index": 2}
            },
            "lawbench_merged": {"meta_data": {"source_file": "1-1.json"}},
        }
        self.assertEqual(
            {
                name: extract_sample_domain(record, "model-inferred-domain")
                for name, record in records.items()
            },
            {
                "Ability_merged": "model-inferred-domain",
                "CAIL2022": "model-inferred-domain",
                "lawbench_merged": "model-inferred-domain",
            },
        )

    def test_result_record_uses_unified_domain_extraction(self) -> None:
        base = load_lgagent_config(CONFIG_PATH, environ={})
        experiment = next(
            item for item in build_task14_matrix(base) if item.key == "original"
        )
        configured = apply_experiment(
            base,
            experiment,
            seed=42,
            corpus_path="/tmp/unused.jsonl",
        )
        sample = FrozenSample(
            index=3,
            sample_id="sample-3",
            split="test",
            record={
                "question": "Question?",
                "golden_answers": ["A"],
                "meta_data": {"correct_option": "A", "original_index": 3},
            },
        )
        trace = SimpleNamespace(model_calls=[], as_dict=lambda: {"routes": []})
        result = SimpleNamespace(
            final_answer="A",
            lawyer_a_output=json.dumps(
                {
                    "task_type": "single_choice",
                    "question_focus": "ownership",
                    "legal_domain": "civil_law",
                    "option_claims": {
                        option: f"claim-{option}" for option in "ABCD"
                    },
                    "option_keywords": {
                        option: [f"keyword-{option}"] for option in "ABCD"
                    },
                    "trap_signals": [],
                    "unknowns": [],
                }
            ),
            diagnostics={
                "risk_score": 0.25,
                "b0_blind_answer": "B",
                "b0_confidence": 0.6,
                "counterfactual_observations": [{"eligible": True}],
                "evidence_matrix": {
                    "options": {
                        "A": {
                            "support": [{"evidence_id": "law:1"}],
                            "refute": [],
                            "exception": [],
                        }
                    }
                },
                "candidates": [
                    {
                        "irac": {
                            "rule": [{"evidence_ids": ["law:1", "law:2"]}]
                        }
                    }
                ],
            },
            trace=trace,
        )
        fake_runner = SimpleNamespace(run=lambda question: result)
        with (
            patch(
                "lgagent.experiment_runner._model_clients",
                return_value=(object(), object()),
            ) as model_clients,
            patch(
                "lgagent.experiment_runner.LGAgentPlusRunner",
                return_value=fake_runner,
            ),
        ):
            record = _run_sample(
                sample,
                config=configured,
                experiment=experiment,
                project_root=ROOT_DIR,
                reproduction={},
                retriever=None,
                api_key="pool-secret",
                api_key_slot=2,
            )

        effective_config = model_clients.call_args.args[0]
        self.assertEqual(effective_config.generation.api_key, "pool-secret")
        self.assertEqual(record["status"], "ok")
        self.assertEqual(record["reproduction"]["api_key_slot"], 2)
        self.assertNotIn("pool-secret", json.dumps(record))
        self.assertEqual(record["domain"], "civil_law")
        self.assertEqual(record["confidence"], 0.75)
        self.assertEqual(record["risk_score"], 0.25)
        self.assertEqual(record["b0_blind_answer"], "B")
        self.assertEqual(record["b0_confidence"], 0.6)
        self.assertEqual(record["retrieved_evidence_ids"], ["law:1"])
        self.assertEqual(record["cited_evidence_ids"], ["law:1", "law:2"])
        self.assertEqual(record["counterfactual_observations"], [{"eligible": True}])

    def test_evidence_id_extractors_are_stable_and_deduplicated(self) -> None:
        matrix = {
            "options": {
                "A": {
                    "support": [{"evidence_id": "law:1"}],
                    "refute": [{"evidence_id": "law:1"}],
                    "exception": [{"evidence_id": "law:2"}],
                }
            }
        }
        candidates = [
            {"irac": {"rule": [{"evidence_ids": ["law:2", "law:3", "law:2"]}]}}
        ]
        self.assertEqual(_evidence_ids(matrix), ["law:1", "law:2"])
        self.assertEqual(_cited_evidence_ids(candidates), ["law:2", "law:3"])

    def test_concurrent_resources_load_and_build_once_per_experiment(self) -> None:
        base = load_lgagent_config(CONFIG_PATH, environ={})
        experiments = {
            item.key: item for item in build_task14_matrix(base)
        }
        selected = (
            experiments["joint"],
            experiments["no-refute"],
            experiments["no-temporal"],
            experiments["no-graph-expansion"],
        )
        with tempfile.TemporaryDirectory() as directory:
            corpus_path = Path(directory) / "corpus.jsonl"
            record = LegalEvidence.from_mapping(
                {
                    "source_type": "statute",
                    "law_name": "中华人民共和国测试法",
                    "article": "第一条",
                    "clause": None,
                    "version": "2024",
                    "text": "测试规则。",
                    "jurisdiction": "CN",
                    "authority_level": 5,
                    "effective_from": "2024-01-01",
                    "effective_to": None,
                    "source_uri": "https://example.invalid/law",
                    "relations": [],
                }
            )
            corpus_path.write_text(
                json.dumps(record.as_dict(), ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            resources = _SharedRetrievalResources(corpus_path)
            original_bm25_init = BM25Index.__init__
            joint = selected[0]
            joint_config = apply_experiment(
                base,
                joint,
                seed=42,
                corpus_path=corpus_path,
            )
            with (
                patch(
                    "lgagent.experiment_runner.load_jsonl_corpus",
                    wraps=load_jsonl_corpus,
                ) as load_corpus,
                patch.object(
                    BM25Index,
                    "__init__",
                    autospec=True,
                    side_effect=original_bm25_init,
                ) as build_bm25,
            ):
                with ThreadPoolExecutor(max_workers=8) as pool:
                    retrievers = tuple(
                        pool.map(
                            lambda _: resources.retriever_for(
                                joint_config,
                                joint,
                            ),
                            range(24),
                        )
                    )

                self.assertEqual(len({id(item) for item in retrievers}), 1)
                self.assertEqual(load_corpus.call_count, 1)
                self.assertEqual(build_bm25.call_count, 1)

                distinct = []
                for experiment in selected[1:]:
                    configured = apply_experiment(
                        base,
                        experiment,
                        seed=42,
                        corpus_path=corpus_path,
                    )
                    distinct.append(
                        resources.retriever_for(configured, experiment)
                    )

                self.assertEqual(build_bm25.call_count, len(selected))
                self.assertEqual(
                    len({id(retrievers[0]), *(id(item) for item in distinct)}),
                    len(selected),
                )
                self.assertNotEqual(
                    distinct[0].config.enabled_lanes,
                    retrievers[0].config.enabled_lanes,
                )
                self.assertNotEqual(
                    distinct[1].config.require_temporal_match,
                    retrievers[0].config.require_temporal_match,
                )
                self.assertNotEqual(
                    distinct[2].config.graph_hops,
                    retrievers[0].config.graph_hops,
                )

    def test_frozen_manifest_matches_workspace_without_copying_data(self) -> None:
        manifest = json.loads(FREEZE_PATH.read_text(encoding="utf-8"))
        validate_freeze_manifest(manifest, ROOT_DIR)
        self.assertTrue({
            "src/lgagent/legal_mcq/agent.py",
            "src/lgagent/legal_mcq/models.py",
            "src/lgagent/legal_mcq/parser.py",
            "src/lgagent/config.py",
            "tools/run_legal_mcq.py",
        }.issubset(manifest["runs"]["prompt_source_sha256"]))
        self.assertEqual(manifest["runs"]["seeds"], [42, 43, 44])
        self.assertEqual(manifest["staged_protocol"]["default_profile"], "pilot")
        self.assertTrue(
            manifest["staged_protocol"]["full_requires_explicit_profile"]
        )
        self.assertEqual(manifest["split"]["materialization"].split(";")[0],
                         "indices are derived at runtime")
        first = assigned_split("sample", "a" * 64)
        self.assertEqual(first, assigned_split("sample", "a" * 64))

    def test_cli_profiles_are_bounded_and_allow_explicit_overrides(self) -> None:
        pilot = resolve_plan(parse_args(["--dry-run"]))
        self.assertEqual(pilot["profile_name"], "pilot")
        self.assertEqual(pilot["split"], "dev")
        self.assertEqual(pilot["max_examples"], 20)
        self.assertEqual(pilot["seeds"], (42,))
        self.assertEqual(
            pilot["experiment_keys"],
            ("original", "web-search", "oath-only", "cape-only", "joint"),
        )

        main = EXPERIMENT_PROFILES["main"]
        self.assertEqual(main.split, "test")
        self.assertEqual(main.max_examples, 200)
        self.assertEqual(main.seeds, (42, 43, 44))
        ablation = EXPERIMENT_PROFILES["ablation"]
        self.assertEqual(ablation.split, "dev")
        self.assertEqual(ablation.max_examples, 100)
        self.assertEqual(len(ablation.experiment_keys), 7)
        self.assertNotIn("self-consistency-equal-token", ablation.experiment_keys)

        full = resolve_plan(parse_args(["--dry-run", "--profile", "full"]))
        self.assertIsNone(full["max_examples"])
        self.assertEqual(len(full["experiment_keys"]), 13)
        overridden = resolve_plan(
            parse_args(
                [
                    "--dry-run",
                    "--profile",
                    "main",
                    "--split",
                    "dev",
                    "--max-examples",
                    "5",
                    "--experiments",
                    "original",
                    "joint",
                    "--datasets",
                    "data/LexGenius.jsonl",
                    "--seeds",
                    "7",
                    "--model",
                    "qwen3-4b",
                ]
            )
        )
        self.assertEqual(overridden["split"], "dev")
        self.assertEqual(overridden["max_examples"], 5)
        self.assertEqual(overridden["experiment_keys"], ("original", "joint"))
        self.assertEqual(
            overridden["dataset_paths"],
            ("data/LexGenius.jsonl",),
        )
        self.assertEqual(overridden["seeds"], (7,))
        self.assertEqual(overridden["model_id"], "qwen3-4b")

    def test_all_task14_variants_map_to_unified_runner_settings(self) -> None:
        base = load_lgagent_config(CONFIG_PATH, environ={})
        matrix = build_task14_matrix(base)
        for experiment in matrix:
            configured = apply_experiment(
                base,
                experiment,
                seed=43,
                corpus_path="/tmp/strict.jsonl",
            )
            self.assertEqual(
                configured.lgagent_plus.oath_rag.enabled,
                experiment.oath_rag_enabled,
            )
            self.assertEqual(
                configured.lgagent_plus.cape_v.enabled,
                experiment.cape_v_enabled,
            )
            self.assertEqual(configured.lgagent_plus.seed, 43)
            minimum, maximum = estimate_calls(experiment)
            self.assertGreater(minimum, 0)
            self.assertGreaterEqual(maximum, minimum)

    def test_model_clients_bound_transport_timeout_and_retries(self) -> None:
        config = load_lgagent_config(
            CONFIG_PATH,
            environ={
                "LLM_API_KEY": "test-key",
                "LGAGENT_VERIFIER_API_KEY": "test-key",
            },
        )
        client = object()
        with patch("openai.OpenAI", return_value=client) as constructor:
            reasoning, verifier = _model_clients(config)

        self.assertIs(reasoning, verifier)
        self.assertIs(reasoning._client, client)
        constructor.assert_called_once_with(
            api_key="test-key",
            base_url=config.generation.base_url,
            timeout=60.0,
            max_retries=2,
        )

    def test_dry_run_validates_without_model_client_or_accuracy_claim(self) -> None:
        record = LegalEvidence.from_mapping(
            {
                "source_type": "statute",
                "law_name": "中华人民共和国测试法",
                "article": "第一条",
                "clause": None,
                "version": "2024",
                "text": "测试规则。",
                "jurisdiction": "CN",
                "authority_level": 5,
                "effective_from": "2024-01-01",
                "effective_to": None,
                "source_uri": f"https://github.com/example/blob/{JUST_LAWS_COMMIT}/law.md",
                "relations": [],
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            corpus = temporary / "corpus.jsonl"
            corpus.write_text(
                json.dumps(record.as_dict(), ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            runner = Task16ExperimentRunner(
                project_root=ROOT_DIR,
                config_path=CONFIG_PATH,
                freeze_path=FREEZE_PATH,
                corpus_path=corpus,
                output_dir=temporary / "output",
            )
            with patch(
                "lgagent.experiment_runner._model_clients",
                side_effect=AssertionError("dry-run must not build a client"),
            ):
                report = runner.dry_run(
                    experiment_keys=["original", "joint"],
                    max_examples=1,
                )
            self.assertTrue(report["dry_run"])
            self.assertIsNone(report["accuracy"])
            self.assertEqual(report["jobs"], 18)
            self.assertEqual(len(report["by_experiment"]), 2)
            self.assertEqual(len(report["by_dataset"]), 3)
            self.assertTrue((temporary / "output/dry_run_report.json").is_file())
            lexgenius_only = runner.dry_run(
                experiment_keys=["joint"],
                dataset_paths=[str(ROOT_DIR / "data/LexGenius.jsonl")],
                max_examples=1,
                seeds=[42],
            )
            self.assertEqual(lexgenius_only["jobs"], 1)
            self.assertEqual(
                lexgenius_only["resolved_plan"]["datasets"],
                ["data/LexGenius.jsonl"],
            )
            with self.assertRaisesRegex(
                ExperimentRunError,
                "not present in the freeze manifest",
            ):
                runner.dry_run(dataset_paths=["data/not-frozen.jsonl"])
            full_report = runner.dry_run(max_examples=1)
            self.assertEqual(full_report["jobs"], 117)
            single_seed = runner.dry_run(
                experiment_keys=["original", "joint"],
                max_examples=1,
                seeds=[42],
                profile_name="pilot",
            )
            self.assertEqual(single_seed["jobs"], 6)
            self.assertEqual(single_seed["profile"], "pilot")
            original_matrix = runner.dry_run(profile_name="full")
            self.assertEqual(original_matrix["jobs"], 42_744)
            self.assertEqual(
                original_matrix["estimated_calls"],
                {
                    "minimum": 585_264,
                    "maximum_without_retries": 1_147_512,
                },
            )
            self.assertEqual(original_matrix["cost_warning"]["level"], "critical")

    def test_method_svg_is_valid_and_document_disclaims_results(self) -> None:
        ET.parse(ROOT_DIR / "docs/assets/lgagent-plus-task16.svg")
        methods = (ROOT_DIR / "docs/task16_methods.md").read_text(encoding="utf-8")
        self.assertIn("No paid API experiment was run", methods)
        self.assertIn("No accuracy", methods)
        self.assertIn("### Staged protocol", methods)
        self.assertIn("--profile full", methods)

    def test_yaml_points_oath_to_built_corpus_without_enabling_plus(self) -> None:
        config = load_lgagent_config(CONFIG_PATH, environ={})
        self.assertFalse(config.lgagent_plus.enabled)
        self.assertEqual(
            config.lgagent_plus.oath_rag.corpus_path,
            "data/oath/oath_just_laws_strict.jsonl",
        )


if __name__ == "__main__":
    unittest.main()
