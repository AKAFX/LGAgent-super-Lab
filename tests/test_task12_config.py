from __future__ import annotations

import copy
import sys
import tempfile
import unittest
from pathlib import Path

import yaml

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from lgagent.config import ConfigurationError, load_lgagent_config
from lgagent.orchestrator import (
    LGAgentPlusOrchestrator,
    resolve_legacy_entrypoint_settings,
    resolve_pipeline_route,
)


def valid_payload() -> dict[str, object]:
    return {
        "generation": {
            "backend": "openai",
            "backend_configs": {
                "openai": {
                    "api_key": "generation-yaml-key",
                    "base_url": "https://generation.invalid/v1",
                    "model_name": "reasoning-model",
                }
            },
            "sampling_params": {
                "temperature": 0.2,
                "top_p": 0.9,
                "max_tokens": 2048,
            },
        },
        "lgagent_plus": {
            "enabled": True,
            "seed": 7,
            "oath_rag": {
                "enabled": True,
                "corpus_path": "data/legal.jsonl",
                "lexical_top_k": 10,
                "dense_top_k": 8,
                "final_top_k_per_lane": 2,
                "graph_hops": 1,
                "require_temporal_match": True,
                "require_authoritative_source": True,
            },
            "cape_v": {
                "enabled": True,
                "initial_candidates": 1,
                "slow_path_candidates": 1,
                "permutation_count": 3,
                "enable_counterfactual": False,
                "candidate_max_tokens": 1024,
                "verifiers": {
                    "rule": True,
                    "fact": True,
                    "exception": True,
                    "evidence": True,
                    "entailment": True,
                },
                "verifier": {
                    "timeout_seconds": 15,
                    "max_attempts": 2,
                    "max_tokens": 512,
                    "weights": {"rule": 2.0, "fact": 1.0},
                },
                "verifier_model": {
                    "backend": "openai",
                    "base_url": "https://verifier.invalid/v1",
                    "api_key": "verifier-yaml-key",
                    "model_name": "independent-verifier",
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "max_tokens": 512,
                },
            },
            "risk": {
                "low_threshold": 0.3,
                "high_threshold": 0.65,
                "weights": {
                    "normalized_answer_entropy": 2.0,
                    "permutation_instability": 1.0,
                    "verifier_disagreement": 1.0,
                    "evidence_conflict": 1.0,
                    "missing_evidence_coverage": 1.0,
                    "low_top2_margin": 1.0,
                },
                "max_rounds": 2,
                "max_model_calls": 32,
                "max_total_tokens": 32768,
                "max_wall_time_seconds": 90,
            },
            "clex": {
                "enabled": False,
                "calibration_path": "docs/clex_calibration.json",
                "alpha": 0.1,
                "min_calibration_size": 30,
                "group_field": "domain",
            },
            "web_search": {
                "enabled": False,
                "provider": "tavily",
                "base_url": "https://api.tavily.com/search",
                "api_key_env": "TAVILY_API_KEY",
                "confidence_threshold": 0.65,
                "max_searches": 2,
                "candidate_results": 5,
                "max_results": 3,
                "chunks_per_source": 1,
                "max_query_chars": 240,
                "max_context_tokens": 1800,
                "max_chars_per_result": 600,
                "min_relevance_score": 0.35,
                "timeout_seconds": 15,
                "failure_mode": "closed_book",
            },
        },
    }


class ConfigFileMixin:
    def load(
        self,
        payload: dict[str, object],
        *,
        environ: dict[str, str] | None = None,
    ):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "parameter.yaml"
            path.write_text(
                yaml.safe_dump(payload, sort_keys=False), encoding="utf-8"
            )
            return load_lgagent_config(path, environ=environ or {})


class TypedConfigTest(ConfigFileMixin, unittest.TestCase):
    def test_loads_all_sections_and_independent_verifier_model(self) -> None:
        config = self.load(
            valid_payload(),
            environ={
                "LLM_API_KEY": "generation-env-key",
                "LGAGENT_VERIFIER_API_KEY": "verifier-env-key",
            },
        )

        self.assertEqual(config.generation.api_key, "generation-yaml-key")
        self.assertEqual(config.generation.api_key_source, "yaml")
        self.assertTrue(config.lgagent_plus.oath_rag.enabled)
        self.assertEqual(config.lgagent_plus.oath_rag.final_top_k_per_lane, 2)
        self.assertTrue(config.lgagent_plus.cape_v.enabled)
        self.assertEqual(config.lgagent_plus.cape_v.slow_path_candidates, 1)
        verifier_model = config.lgagent_plus.cape_v.verifier_model
        self.assertIsNotNone(verifier_model)
        assert verifier_model is not None
        self.assertEqual(verifier_model.model, "independent-verifier")
        self.assertEqual(verifier_model.api_key, "verifier-yaml-key")
        self.assertEqual(verifier_model.api_key_source, "yaml")
        self.assertAlmostEqual(
            sum(config.lgagent_plus.risk.weights.normalized.values()), 1.0
        )
        self.assertEqual(config.lgagent_plus.risk.budget.max_seconds, 90.0)
        self.assertFalse(config.lgagent_plus.clex.enabled)
        self.assertFalse(config.lgagent_plus.web_search.enabled)
        self.assertEqual(config.lgagent_plus.web_search.max_results, 3)
        self.assertEqual(
            config.lgagent_plus.clex.calibration_path,
            "docs/clex_calibration.json",
        )

    def test_clex_requires_cape_and_a_calibration_artifact(self) -> None:
        payload = valid_payload()
        payload["lgagent_plus"]["clex"]["enabled"] = True  # type: ignore[index]
        payload["lgagent_plus"]["clex"]["calibration_path"] = ""  # type: ignore[index]
        with self.assertRaisesRegex(ConfigurationError, "calibration_path"):
            self.load(payload)

        payload = valid_payload()
        payload["lgagent_plus"]["clex"]["enabled"] = True  # type: ignore[index]
        payload["lgagent_plus"]["cape_v"]["enabled"] = False  # type: ignore[index]
        with self.assertRaisesRegex(ConfigurationError, "requires"):
            self.load(payload)

    def test_clex_alpha_must_be_strictly_between_zero_and_one(self) -> None:
        for alpha in (0.0, 1.0):
            with self.subTest(alpha=alpha):
                payload = valid_payload()
                payload["lgagent_plus"]["clex"]["alpha"] = alpha  # type: ignore[index]
                with self.assertRaisesRegex(ConfigurationError, "strictly"):
                    self.load(payload)

    def test_web_search_isolated_from_oath_cape_and_clex(self) -> None:
        payload = valid_payload()
        payload["lgagent_plus"]["web_search"]["enabled"] = True  # type: ignore[index]
        with self.assertRaisesRegex(ConfigurationError, "isolated experiment"):
            self.load(payload)

        payload["lgagent_plus"]["oath_rag"]["enabled"] = False  # type: ignore[index]
        payload["lgagent_plus"]["cape_v"]["enabled"] = False  # type: ignore[index]
        config = self.load(payload)
        self.assertTrue(config.lgagent_plus.web_search.enabled)

    def test_web_search_candidate_pool_covers_final_evidence_limit(self) -> None:
        payload = valid_payload()
        payload["lgagent_plus"]["web_search"]["candidate_results"] = 2  # type: ignore[index]
        payload["lgagent_plus"]["web_search"]["max_results"] = 3  # type: ignore[index]
        with self.assertRaisesRegex(ConfigurationError, "at least max_results"):
            self.load(payload)

    def test_generation_yaml_key_precedence_remains_compatible(self) -> None:
        payload = valid_payload()
        config = self.load(payload, environ={"LLM_API_KEY": "environment-key"})
        self.assertEqual(config.generation.api_key, "generation-yaml-key")

        payload["generation"]["backend_configs"]["openai"]["api_key"] = ""  # type: ignore[index]
        fallback = self.load(
            payload,
            environ={
                "LLM_API_KEY": "environment-key",
                "LGAGENT_VERIFIER_API_KEY": "verifier-environment-key",
            },
        )
        self.assertEqual(fallback.generation.api_key, "environment-key")
        self.assertEqual(fallback.generation.api_key_source, "environment")

    def test_legal_role_reasoning_effort_is_typed_and_inherited(self) -> None:
        payload = valid_payload()
        payload["lgagent_plus"]["enabled"] = False  # type: ignore[index]
        payload["generation"]["sampling_params"]["reasoning_effort"] = "medium"  # type: ignore[index]
        payload["legal_mcq"] = {
            "enabled": True,
            "controller_model": {"model_name": "controller"},
            "solver_model": {
                "model_name": "solver",
                "reasoning_effort": "low",
            },
            "verifier_model": {"model_name": "verifier"},
        }

        config = self.load(payload)

        assert config.legal_mcq.controller_model is not None
        assert config.legal_mcq.solver_model is not None
        assert config.legal_mcq.verifier_model is not None
        self.assertEqual(
            config.legal_mcq.controller_model.reasoning_effort,
            "medium",
        )
        self.assertEqual(config.legal_mcq.solver_model.reasoning_effort, "low")
        self.assertEqual(
            config.legal_mcq.verifier_model.reasoning_effort,
            "medium",
        )

    def test_repository_yaml_is_offline_safe_and_defaults_to_baseline(self) -> None:
        config = load_lgagent_config(
            ROOT_DIR / "examples" / "parameter" / "legal2_rag_parameter.yaml",
            environ={},
        )
        route = resolve_pipeline_route(config)

        self.assertTrue(route.corrected_baseline)
        self.assertFalse(route.oath_rag_enabled)
        self.assertFalse(route.cape_v_enabled)
        self.assertEqual(
            config.lgagent_plus.oath_rag.default_jurisdiction,
            "CN",
        )
        self.assertEqual(config.generation.api_key, "")
        verifier_model = config.lgagent_plus.cape_v.verifier_model
        self.assertIsNotNone(verifier_model)
        assert verifier_model is not None
        self.assertEqual(verifier_model.api_key, "")
        legal = config.legal_mcq
        assert legal.controller_model is not None
        assert legal.solver_model is not None
        assert legal.solver_fallback_model is not None
        assert legal.verifier_model is not None
        self.assertEqual(legal.controller_model.model, "Qwen/Qwen3.7-Flash")
        self.assertEqual(legal.solver_model.model, "gpt-5.6-sol")
        self.assertEqual(
            legal.solver_fallback_model.model,
            "Qwen/Qwen3.8-Max-0902",
        )
        self.assertEqual(legal.verifier_model.model, "Qwen/Qwen3.7-Flash")
        self.assertEqual(legal.solver_model.reasoning_effort, "high")
        self.assertEqual(legal.solver_fallback_model.reasoning_effort, "medium")
        self.assertEqual(legal.max_total_tokens, 65536)
        self.assertEqual(legal.max_wall_time_seconds, 360)


class StrictValidationTest(ConfigFileMixin, unittest.TestCase):
    def assert_invalid(self, mutate, message: str) -> None:
        payload = copy.deepcopy(valid_payload())
        mutate(payload)
        with self.assertRaisesRegex(ConfigurationError, message):
            self.load(payload)

    def test_rejects_invalid_thresholds_and_non_numeric_thresholds(self) -> None:
        self.assert_invalid(
            lambda p: p["lgagent_plus"]["risk"].update(  # type: ignore[index]
                low_threshold=0.7, high_threshold=0.7
            ),
            "low_threshold < high_threshold",
        )
        self.assert_invalid(
            lambda p: p["lgagent_plus"]["risk"].update(low_threshold="0.3"),  # type: ignore[index]
            "must be a number",
        )

    def test_rejects_empty_default_jurisdiction(self) -> None:
        self.assert_invalid(
            lambda p: p["lgagent_plus"]["oath_rag"].update(  # type: ignore[index]
                default_jurisdiction=" "
            ),
            "default_jurisdiction must be a non-empty string",
        )

    def test_rejects_negative_and_all_zero_risk_weights(self) -> None:
        self.assert_invalid(
            lambda p: p["lgagent_plus"]["risk"]["weights"].update(  # type: ignore[index]
                evidence_conflict=-0.1
            ),
            "must be >= 0.0",
        )
        self.assert_invalid(
            lambda p: p["lgagent_plus"]["risk"].update(  # type: ignore[index]
                weights={
                    name: 0.0
                    for name in (
                        "normalized_answer_entropy",
                        "permutation_instability",
                        "verifier_disagreement",
                        "evidence_conflict",
                        "missing_evidence_coverage",
                        "low_top2_margin",
                    )
                }
            ),
            "positive weight",
        )

    def test_rejects_invalid_call_token_round_and_time_budgets(self) -> None:
        for field, value in (
            ("max_model_calls", 0),
            ("max_total_tokens", 0),
            ("max_rounds", 0),
            ("max_wall_time_seconds", 0),
        ):
            with self.subTest(field=field):
                self.assert_invalid(
                    lambda p, field=field, value=value: p["lgagent_plus"][  # type: ignore[index]
                        "risk"
                    ].update({field: value}),
                    field,
                )

    def test_rejects_impossible_fast_path_budgets(self) -> None:
        self.assert_invalid(
            lambda p: p["lgagent_plus"]["risk"].update(max_model_calls=26),  # type: ignore[index]
            "requires at least 27",
        )
        self.assert_invalid(
            lambda p: p["lgagent_plus"]["risk"].update(max_total_tokens=512),  # type: ignore[index]
            "requires at least 29952",
        )

    def test_rejects_legal_call_timeout_larger_than_question_deadline(self) -> None:
        def mutate(payload):
            payload["legal_mcq"] = {
                "enabled": False,
                "max_wall_time_seconds": 30,
                "model_call_timeout_seconds": 31,
            }

        self.assert_invalid(mutate, "cannot exceed")

    def test_rejects_unknown_solver_structured_output_mode(self) -> None:
        def mutate(payload):
            payload["legal_mcq"] = {
                "enabled": False,
                "solver_structured_output_mode": "best-effort",
            }

        self.assert_invalid(mutate, "solver_structured_output_mode")

    def test_rejects_invalid_output_budget_or_controller_mode(self) -> None:
        for key, value in (
            ("solver_visible_output_tokens", 0),
            ("solver_reasoning_allowance_tokens", -1),
            ("auxiliary_reasoning_reserve_tokens", True),
            ("controller_structured_output_mode", "best-effort"),
            ("verifier_structured_output_mode", "best-effort"),
            ("solver_timeout_retries", 1),
            ("solver_timeout_circuit_breaker", 0),
            ("skip_verifier_on_deterministic_errors", "true"),
        ):
            with self.subTest(key=key):
                def mutate(payload):
                    payload["legal_mcq"] = {key: value}
                self.assert_invalid(mutate, key)

    def test_rejects_invalid_role_timeout_and_verifier_retry_budget(self) -> None:
        def timeout(payload):
            payload["legal_mcq"] = {
                "max_wall_time_seconds": 60,
                "solver_call_timeout_seconds": 61,
            }

        self.assert_invalid(timeout, "cannot exceed")

        def verifier_tokens(payload):
            payload["legal_mcq"] = {
                "verifier_visible_output_tokens": 512,
                "verifier_length_retry_tokens": 384,
            }

        self.assert_invalid(verifier_tokens, "cannot be smaller")

    def test_solver_fallback_inherits_generation_key_and_requires_call_budget(self) -> None:
        payload = valid_payload()
        payload["lgagent_plus"]["enabled"] = False  # type: ignore[index]
        payload["legal_mcq"] = {
            "enabled": True,
            "max_model_calls": 6,
            "solver_fallback_model": {
                "model_name": "fallback",
                "max_tokens": 4096,
            },
        }
        config = self.load(payload)
        assert config.legal_mcq.solver_fallback_model is not None
        self.assertEqual(
            config.legal_mcq.solver_fallback_model.api_key,
            "generation-yaml-key",
        )

        payload["legal_mcq"]["max_model_calls"] = 5  # type: ignore[index]
        with self.assertRaisesRegex(ConfigurationError, "requires at least 6"):
            self.load(payload)

    def test_rejects_unknown_reasoning_effort(self) -> None:
        def mutate(payload):
            payload["lgagent_plus"]["enabled"] = False
            payload["legal_mcq"] = {
                "enabled": True,
                "solver_model": {
                    "model_name": "solver",
                    "reasoning_effort": "none",
                },
            }

        self.assert_invalid(mutate, "reasoning_effort")

    def test_rejects_negative_verifier_weights_and_non_boolean_switches(self) -> None:
        self.assert_invalid(
            lambda p: p["lgagent_plus"]["cape_v"]["verifier"][  # type: ignore[index]
                "weights"
            ].update(rule=-1),
            "must be >= 0.0",
        )
        self.assert_invalid(
            lambda p: p["lgagent_plus"]["oath_rag"].update(enabled="false"),  # type: ignore[index]
            "must be a boolean",
        )


class FeatureRoutingTest(ConfigFileMixin, unittest.TestCase):
    def test_legal_mcq_route_is_not_reported_as_corrected_baseline(self) -> None:
        payload = valid_payload()
        payload["lgagent_plus"]["enabled"] = False  # type: ignore[index]
        payload["legal_mcq"] = {"enabled": True}

        route = resolve_pipeline_route(self.load(payload))

        self.assertEqual(route.name, "legal-mcq-three-role")
        self.assertTrue(route.legal_mcq_enabled)
        self.assertFalse(route.corrected_baseline)
        self.assertFalse(route.oath_rag_enabled)
        self.assertFalse(route.cape_v_enabled)

    def test_master_switch_forces_corrected_baseline_in_legacy_batch(self) -> None:
        payload = valid_payload()
        payload["lgagent_plus"]["enabled"] = False  # type: ignore[index]
        route = resolve_pipeline_route(self.load(payload))
        settings = resolve_legacy_entrypoint_settings(
            route,
            enable_lawyer_a=False,
            enable_judge=False,
            enable_dialogue=False,
            use_rag=True,
        )

        self.assertTrue(route.corrected_baseline)
        self.assertTrue(settings.enable_lawyer_a)
        self.assertTrue(settings.enable_judge)
        self.assertTrue(settings.enable_dialogue)
        self.assertFalse(settings.use_rag)

    def test_oath_and_cape_can_be_routed_independently(self) -> None:
        for oath_enabled, cape_enabled in (
            (True, False),
            (False, True),
            (True, True),
            (False, False),
        ):
            with self.subTest(oath=oath_enabled, cape=cape_enabled):
                payload = valid_payload()
                plus = payload["lgagent_plus"]  # type: ignore[assignment]
                plus["oath_rag"]["enabled"] = oath_enabled
                plus["cape_v"]["enabled"] = cape_enabled
                route = resolve_pipeline_route(self.load(payload))
                self.assertEqual(route.oath_rag_enabled, oath_enabled)
                self.assertEqual(route.cape_v_enabled, cape_enabled)

    def test_orchestrator_skips_all_plus_stages_when_master_is_off(self) -> None:
        payload = valid_payload()
        payload["lgagent_plus"]["enabled"] = False  # type: ignore[index]
        events: list[str] = []
        result = LGAgentPlusOrchestrator(self.load(payload)).run(
            lambda: events.append("baseline") or "baseline-result",
            oath_rag=lambda value: events.append("oath") or value,
            cape_v=lambda value: events.append("cape") or value,
        )

        self.assertEqual(result, "baseline-result")
        self.assertEqual(events, ["baseline"])

    def test_orchestrator_runs_only_enabled_plus_stages(self) -> None:
        payload = valid_payload()
        payload["lgagent_plus"]["cape_v"]["enabled"] = False  # type: ignore[index]
        events: list[str] = []
        result = LGAgentPlusOrchestrator(self.load(payload)).run(
            lambda: events.append("baseline") or "result",
            oath_rag=lambda value: events.append("oath") or f"{value}+oath",
            cape_v=lambda value: events.append("cape") or f"{value}+cape",
        )

        self.assertEqual(result, "result+oath")
        self.assertEqual(events, ["baseline", "oath"])

    def test_enabled_stage_without_implementation_fails_before_model_work(self) -> None:
        events: list[str] = []
        with self.assertRaisesRegex(RuntimeError, "OATH-RAG"):
            LGAgentPlusOrchestrator(self.load(valid_payload())).run(
                lambda: events.append("baseline") or "result"
            )
        self.assertEqual(events, [])


if __name__ == "__main__":
    unittest.main()
