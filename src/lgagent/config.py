"""Typed configuration loading for LGAgent and LGAgent++."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping

import yaml

if TYPE_CHECKING:
    from .risk import BudgetLimits, RiskThresholds, RiskWeights
    from .verification import VerifierConfig

VERIFIER_NAMES = ("rule", "fact", "exception", "evidence", "entailment")
RISK_WEIGHT_NAMES = (
    "normalized_answer_entropy",
    "permutation_instability",
    "verifier_disagreement",
    "evidence_conflict",
    "missing_evidence_coverage",
    "low_top2_margin",
)


class ConfigurationError(ValueError):
    """Raised when an LGAgent configuration is missing or malformed."""


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ConfigurationError(f"{name} must be an object")
    return value


def _reject_unknown(
    value: Mapping[str, Any], allowed: set[str], name: str
) -> None:
    unknown = set(value) - allowed
    if unknown:
        raise ConfigurationError(f"{name} has unknown fields: {sorted(unknown)}")


def _boolean(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigurationError(f"{name} must be a boolean")
    return value


def _integer(
    value: Any,
    name: str,
    minimum: int,
    maximum: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigurationError(f"{name} must be an integer")
    if value < minimum or (maximum is not None and value > maximum):
        expected = (
            f"between {minimum} and {maximum}"
            if maximum is not None
            else f"at least {minimum}"
        )
        raise ConfigurationError(f"{name} must be {expected}")
    return value


def _number(
    value: Any,
    name: str,
    minimum: float,
    maximum: float | None = None,
    *,
    inclusive: bool = True,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigurationError(f"{name} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ConfigurationError(f"{name} must be finite")
    too_small = result < minimum if inclusive else result <= minimum
    if too_small:
        operator = ">=" if inclusive else ">"
        raise ConfigurationError(f"{name} must be {operator} {minimum}")
    if maximum is not None and result > maximum:
        raise ConfigurationError(f"{name} must be <= {maximum}")
    return result


@dataclass(frozen=True)
class ModelConfig:
    backend: str
    base_url: str
    api_key: str
    model: str
    temperature: float = 0.7
    top_p: float = 0.8
    max_tokens: int = 2048
    api_key_source: str = "environment"

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ModelConfig":
        try:
            max_tokens = int(value.get("max_tokens", 2048))
            temperature = float(value.get("temperature", 0.7))
            top_p = float(value.get("top_p", 0.8))
        except (TypeError, ValueError) as exc:
            raise ConfigurationError("sampling parameters must be numeric") from exc
        if max_tokens <= 0:
            raise ConfigurationError("max_tokens must be positive")
        if not 0.0 <= temperature <= 2.0:
            raise ConfigurationError("temperature must be between 0 and 2")
        if not 0.0 <= top_p <= 1.0:
            raise ConfigurationError("top_p must be between 0 and 1")
        return cls(
            backend=str(value.get("backend", "openai")),
            base_url=str(value.get("base_url", "https://api.zhizengzeng.com/v1")),
            api_key=str(value.get("api_key", "")),
            model=str(value.get("model", "gpt-4o")),
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            api_key_source=str(value.get("api_key_source", "environment")),
        )

    def as_legacy_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "base_url": self.base_url,
            "api_key": self.api_key,
            "api_key_source": self.api_key_source,
            "model": self.model,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
        }


@dataclass(frozen=True)
class OathRagSettings:
    enabled: bool = False
    corpus_path: str = "data/legal_corpus.jsonl"
    lexical_top_k: int = 20
    dense_top_k: int = 20
    final_top_k_per_lane: int = 3
    graph_hops: int = 1
    require_temporal_match: bool = True
    require_authoritative_source: bool = True
    corpus_failure_mode: str = "fail_closed"

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "OathRagSettings":
        prefix = "lgagent_plus.oath_rag"
        _reject_unknown(
            value,
            {
                "enabled", "corpus_path", "lexical_top_k", "dense_top_k",
                "final_top_k_per_lane", "graph_hops",
                "require_temporal_match", "require_authoritative_source",
                "corpus_failure_mode",
            },
            prefix,
        )
        corpus_path = value.get("corpus_path", "data/legal_corpus.jsonl")
        if not isinstance(corpus_path, str) or not corpus_path.strip():
            raise ConfigurationError(f"{prefix}.corpus_path must be a non-empty string")
        corpus_failure_mode = value.get("corpus_failure_mode", "fail_closed")
        if (
            not isinstance(corpus_failure_mode, str)
            or corpus_failure_mode not in {"fail_closed", "empty_evidence"}
        ):
            raise ConfigurationError(
                f"{prefix}.corpus_failure_mode must be fail_closed or empty_evidence"
            )
        return cls(
            enabled=_boolean(value.get("enabled", False), f"{prefix}.enabled"),
            corpus_path=corpus_path,
            lexical_top_k=_integer(
                value.get("lexical_top_k", 20), f"{prefix}.lexical_top_k", 1
            ),
            dense_top_k=_integer(
                value.get("dense_top_k", 20), f"{prefix}.dense_top_k", 1
            ),
            final_top_k_per_lane=_integer(
                value.get("final_top_k_per_lane", 3),
                f"{prefix}.final_top_k_per_lane",
                1,
            ),
            graph_hops=_integer(
                value.get("graph_hops", 1), f"{prefix}.graph_hops", 0, 1
            ),
            require_temporal_match=_boolean(
                value.get("require_temporal_match", True),
                f"{prefix}.require_temporal_match",
            ),
            require_authoritative_source=_boolean(
                value.get("require_authoritative_source", True),
                f"{prefix}.require_authoritative_source",
            ),
            corpus_failure_mode=corpus_failure_mode,
        )


def _default_verifier() -> "VerifierConfig":
    from .verification import VerifierConfig

    return VerifierConfig()


@dataclass(frozen=True)
class CapeVSettings:
    enabled: bool = False
    initial_candidates: int = 1
    slow_path_candidates: int = 1
    permutation_count: int = 3
    enable_counterfactual: bool = False
    candidate_max_tokens: int = 1024
    verifiers: Mapping[str, bool] = field(
        default_factory=lambda: dict.fromkeys(VERIFIER_NAMES, True)
    )
    verifier: "VerifierConfig" = field(default_factory=_default_verifier)
    verifier_model: ModelConfig | None = None

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
        *,
        generation: ModelConfig,
        environ: Mapping[str, str],
    ) -> "CapeVSettings":
        from .verification import VerifierConfig

        prefix = "lgagent_plus.cape_v"
        _reject_unknown(
            value,
            {
                "enabled", "initial_candidates", "slow_path_candidates",
                "permutation_count", "enable_counterfactual",
                "candidate_max_tokens", "verifiers", "verifier",
                "verifier_model",
            },
            prefix,
        )
        raw_enabled = _mapping(value.get("verifiers"), f"{prefix}.verifiers")
        _reject_unknown(raw_enabled, set(VERIFIER_NAMES), f"{prefix}.verifiers")
        enabled = {
            name: _boolean(raw_enabled.get(name, True), f"{prefix}.verifiers.{name}")
            for name in VERIFIER_NAMES
        }
        raw_verifier = _mapping(value.get("verifier"), f"{prefix}.verifier")
        _reject_unknown(
            raw_verifier,
            {"timeout_seconds", "max_attempts", "max_tokens", "weights"},
            f"{prefix}.verifier",
        )
        raw_weights = _mapping(
            raw_verifier.get("weights"), f"{prefix}.verifier.weights"
        )
        _reject_unknown(raw_weights, set(VERIFIER_NAMES), f"{prefix}.verifier.weights")
        weights = {
            name: _number(weight, f"{prefix}.verifier.weights.{name}", 0.0)
            for name, weight in raw_weights.items()
        }
        timeout = _number(
            raw_verifier.get("timeout_seconds", 30.0),
            f"{prefix}.verifier.timeout_seconds",
            0.0,
            inclusive=False,
        )
        attempts = _integer(
            raw_verifier.get("max_attempts", 2),
            f"{prefix}.verifier.max_attempts",
            1,
        )
        verifier_tokens = _integer(
            raw_verifier.get("max_tokens", 512),
            f"{prefix}.verifier.max_tokens",
            1,
        )
        verifier = VerifierConfig(
            enabled=enabled,
            timeout_seconds=timeout,
            max_attempts=attempts,
            max_tokens=verifier_tokens,
            weights=weights,
        )
        return cls(
            enabled=_boolean(value.get("enabled", False), f"{prefix}.enabled"),
            initial_candidates=_integer(
                value.get("initial_candidates", 1),
                f"{prefix}.initial_candidates",
                1,
            ),
            slow_path_candidates=_integer(
                value.get("slow_path_candidates", 1),
                f"{prefix}.slow_path_candidates",
                1,
            ),
            permutation_count=_integer(
                value.get("permutation_count", 3),
                f"{prefix}.permutation_count",
                0,
                23,
            ),
            enable_counterfactual=_boolean(
                value.get("enable_counterfactual", False),
                f"{prefix}.enable_counterfactual",
            ),
            candidate_max_tokens=_integer(
                value.get("candidate_max_tokens", 1024),
                f"{prefix}.candidate_max_tokens",
                1,
            ),
            verifiers=enabled,
            verifier=verifier,
            verifier_model=_load_verifier_model(
                _mapping(value.get("verifier_model"), f"{prefix}.verifier_model"),
                generation,
                environ,
                verifier_tokens,
            ),
        )


def _default_thresholds() -> "RiskThresholds":
    from .risk import RiskThresholds

    return RiskThresholds()


def _default_weights() -> "RiskWeights":
    from .risk import RiskWeights

    return RiskWeights()


def _default_budget() -> "BudgetLimits":
    from .risk import BudgetLimits

    return BudgetLimits()


@dataclass(frozen=True)
class RiskSettings:
    thresholds: "RiskThresholds" = field(default_factory=_default_thresholds)
    weights: "RiskWeights" = field(default_factory=_default_weights)
    budget: "BudgetLimits" = field(default_factory=_default_budget)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "RiskSettings":
        from .risk import BudgetLimits, RiskThresholds, RiskWeights

        prefix = "lgagent_plus.risk"
        _reject_unknown(
            value,
            {
                "low_threshold", "high_threshold", "weights", "max_rounds",
                "max_model_calls", "max_total_tokens",
                "max_wall_time_seconds",
            },
            prefix,
        )
        low = _number(
            value.get("low_threshold", 0.30),
            f"{prefix}.low_threshold",
            0.0,
            1.0,
        )
        high = _number(
            value.get("high_threshold", 0.65),
            f"{prefix}.high_threshold",
            0.0,
            1.0,
        )
        if low >= high:
            raise ConfigurationError(
                f"{prefix} thresholds must satisfy low_threshold < high_threshold"
            )
        raw_weights = _mapping(value.get("weights"), f"{prefix}.weights")
        _reject_unknown(raw_weights, set(RISK_WEIGHT_NAMES), f"{prefix}.weights")
        weights = {
            name: _number(
                raw_weights.get(name, 1.0), f"{prefix}.weights.{name}", 0.0
            )
            for name in RISK_WEIGHT_NAMES
        }
        if not any(weights.values()):
            raise ConfigurationError(f"{prefix}.weights must contain a positive weight")
        return cls(
            thresholds=RiskThresholds(low=low, high=high),
            weights=RiskWeights(**weights),
            budget=BudgetLimits(
                max_calls=_integer(
                    value.get("max_model_calls", 32),
                    f"{prefix}.max_model_calls",
                    1,
                ),
                max_tokens=_integer(
                    value.get("max_total_tokens", 32768),
                    f"{prefix}.max_total_tokens",
                    1,
                ),
                max_rounds=_integer(
                    value.get("max_rounds", 2), f"{prefix}.max_rounds", 1
                ),
                max_seconds=_number(
                    value.get("max_wall_time_seconds", 120.0),
                    f"{prefix}.max_wall_time_seconds",
                    0.0,
                    inclusive=False,
                ),
            ),
        )


@dataclass(frozen=True)
class LGAgentPlusConfig:
    enabled: bool = False
    seed: int = 42
    oath_rag: OathRagSettings = field(default_factory=OathRagSettings)
    cape_v: CapeVSettings = field(default_factory=CapeVSettings)
    risk: RiskSettings = field(default_factory=RiskSettings)

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
        *,
        generation: ModelConfig,
        environ: Mapping[str, str],
    ) -> "LGAgentPlusConfig":
        prefix = "lgagent_plus"
        _reject_unknown(
            value, {"enabled", "seed", "oath_rag", "cape_v", "risk"}, prefix
        )
        config = cls(
            enabled=_boolean(value.get("enabled", False), f"{prefix}.enabled"),
            seed=_integer(value.get("seed", 42), f"{prefix}.seed", 0),
            oath_rag=OathRagSettings.from_mapping(
                _mapping(value.get("oath_rag"), f"{prefix}.oath_rag")
            ),
            cape_v=CapeVSettings.from_mapping(
                _mapping(value.get("cape_v"), f"{prefix}.cape_v"),
                generation=generation,
                environ=environ,
            ),
            risk=RiskSettings.from_mapping(
                _mapping(value.get("risk"), f"{prefix}.risk")
            ),
        )
        config._validate_budget(generation)
        return config

    def _validate_budget(self, generation: ModelConfig) -> None:
        if not self.enabled or not self.cape_v.enabled:
            return
        verifier_count = sum(self.cape_v.verifiers.values())
        baseline_calls = 4
        evidence_audit_calls = 4 if self.oath_rag.enabled else 0
        candidate_calls = (
            self.cape_v.initial_candidates + self.cape_v.permutation_count
        )
        minimum_calls = (
            baseline_calls
            + evidence_audit_calls
            + candidate_calls
            + self.cape_v.initial_candidates * verifier_count
        )
        generation_tokens = generation.max_tokens
        minimum_tokens = (
            min(generation_tokens, 1024)
            + min(generation_tokens, 1024)
            + min(generation_tokens, 256)
            + min(generation_tokens, 1024)
            + evidence_audit_calls * min(generation_tokens, 2048)
            + candidate_calls * self.cape_v.candidate_max_tokens
            + self.cape_v.initial_candidates
            * verifier_count
            * self.cape_v.verifier.max_tokens
        )
        adaptive_calls = (
            self.cape_v.slow_path_candidates * (1 + verifier_count)
            + evidence_audit_calls
        )
        adaptive_tokens = (
            self.cape_v.slow_path_candidates
            * (
                self.cape_v.candidate_max_tokens
                + verifier_count * self.cape_v.verifier.max_tokens
            )
            + evidence_audit_calls * min(generation_tokens, 2048)
        )
        minimum_calls += adaptive_calls
        minimum_tokens += adaptive_tokens
        if self.risk.budget.max_calls < minimum_calls:
            raise ConfigurationError(
                "lgagent_plus.risk.max_model_calls is impossible for the CAPE-V "
                "fast path plus one high-risk adaptive round; "
                f"requires at least {minimum_calls}"
            )
        if self.risk.budget.max_tokens < minimum_tokens:
            raise ConfigurationError(
                "lgagent_plus.risk.max_total_tokens is impossible for the "
                "CAPE-V fast path plus one high-risk adaptive round; "
                f"requires at least {minimum_tokens}"
            )


@dataclass(frozen=True)
class LGAgentConfig:
    generation: ModelConfig
    lgagent_plus: LGAgentPlusConfig


def _load_verifier_model(
    value: Mapping[str, Any],
    generation: ModelConfig,
    environ: Mapping[str, str],
    max_tokens: int,
) -> ModelConfig:
    prefix = "lgagent_plus.cape_v.verifier_model"
    _reject_unknown(
        value,
        {
            "backend", "base_url", "api_key", "model_name", "temperature",
            "top_p", "max_tokens",
        },
        prefix,
    )
    yaml_key = str(value.get("api_key") or "")
    if yaml_key:
        api_key, source = yaml_key, "yaml"
    elif generation.api_key_source == "yaml" and generation.api_key:
        api_key, source = generation.api_key, "generation_yaml"
    elif environ.get("LGAGENT_VERIFIER_API_KEY"):
        api_key, source = environ["LGAGENT_VERIFIER_API_KEY"], "environment"
    else:
        api_key, source = generation.api_key, generation.api_key_source
    return ModelConfig.from_mapping(
        {
            "backend": value.get("backend", generation.backend),
            "base_url": value.get("base_url", generation.base_url),
            "api_key": api_key,
            "api_key_source": source,
            "model": value.get("model_name", generation.model),
            "temperature": value.get("temperature", 0.0),
            "top_p": value.get("top_p", 1.0),
            "max_tokens": value.get("max_tokens", max_tokens),
        }
    )


def _load_yaml(path: str | Path) -> Mapping[str, Any]:
    config_path = Path(path)
    try:
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigurationError(f"cannot load configuration: {config_path}") from exc
    if not isinstance(payload, Mapping):
        raise ConfigurationError("configuration root must be an object")
    return payload


def _load_generation(
    payload: Mapping[str, Any], environ: Mapping[str, str]
) -> ModelConfig:
    generation = _mapping(payload.get("generation"), "generation")
    backend = str(generation.get("backend", "openai"))
    backends = _mapping(
        generation.get("backend_configs"), "generation.backend_configs"
    )
    backend_config = _mapping(
        backends.get(backend), f"generation.backend_configs.{backend}"
    )
    sampling = _mapping(
        generation.get("sampling_params"), "generation.sampling_params"
    )
    yaml_key = str(backend_config.get("api_key") or "")
    return ModelConfig.from_mapping(
        {
            "backend": backend,
            "base_url": backend_config.get(
                "base_url", "https://api.zhizengzeng.com/v1"
            ),
            "api_key": yaml_key or environ.get("LLM_API_KEY", ""),
            "api_key_source": "yaml" if yaml_key else "environment",
            "model": backend_config.get("model_name", "gpt-4o"),
            "temperature": sampling.get("temperature", 0.7),
            "top_p": sampling.get("top_p", 0.8),
            "max_tokens": sampling.get("max_tokens", 2048),
        }
    )


def load_lgagent_config(
    path: str | Path,
    *,
    environ: Mapping[str, str] | None = None,
) -> LGAgentConfig:
    payload = _load_yaml(path)
    environment = os.environ if environ is None else environ
    generation = _load_generation(payload, environment)
    return LGAgentConfig(
        generation=generation,
        lgagent_plus=LGAgentPlusConfig.from_mapping(
            _mapping(payload.get("lgagent_plus"), "lgagent_plus"),
            generation=generation,
            environ=environment,
        ),
    )


def load_generation_config(
    path: str | Path,
    *,
    environ: Mapping[str, str] | None = None,
) -> ModelConfig:
    """Preserve the legacy generation-only API and YAML-key precedence."""
    return load_lgagent_config(path, environ=environ).generation
