"""Deterministic, offline experiment planning for the Task 14 ablations."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .config import LGAgentConfig, VERIFIER_NAMES
from .evaluation import stable_hash

EVIDENCE_LANES = ("support", "refute", "exception")
REQUIRED_EXPERIMENT_KEYS = (
    "original",
    "web-search",
    "oath-only",
    "cape-only",
    "joint",
    "no-refute",
    "no-exception",
    "no-temporal",
    "no-graph-expansion",
    "no-permutation",
    "single-verifier",
    "fixed-budget",
    "self-consistency-equal-token",
)


class AblationMatrixError(ValueError):
    """Raised when an experiment matrix is incomplete or internally inconsistent."""


@dataclass(frozen=True)
class AblationExperiment:
    key: str
    label: str
    family: str
    description: str
    strategy: str
    lgagent_plus_enabled: bool
    web_search_enabled: bool
    oath_rag_enabled: bool
    cape_v_enabled: bool
    evidence_lanes: tuple[str, ...]
    require_temporal_match: bool
    graph_hops: int
    permutation_count: int
    enabled_verifiers: tuple[str, ...]
    budget_policy: str
    max_model_calls: int
    max_total_tokens: int
    max_rounds: int
    self_consistency_samples: int = 0
    self_consistency_max_tokens: int = 0
    equal_token_reference: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _experiment(
    *,
    key: str,
    label: str,
    family: str,
    description: str,
    base: Mapping[str, Any],
    **overrides: Any,
) -> AblationExperiment:
    values = dict(base)
    values.update(overrides)
    return AblationExperiment(
        key=key,
        label=label,
        family=family,
        description=description,
        **values,
    )


def build_task14_matrix(config: LGAgentConfig) -> tuple[AblationExperiment, ...]:
    """Build the complete Task 14 matrix without initializing models or retrievers."""
    plus = config.lgagent_plus
    oath = plus.oath_rag
    cape = plus.cape_v
    budget = plus.risk.budget
    enabled_verifiers = tuple(VERIFIER_NAMES)
    baseline_worst_case_calls = 6
    self_consistency_samples = min(
        max(1, budget.max_calls // baseline_worst_case_calls),
        max(
            1,
            budget.max_tokens
            // (baseline_worst_case_calls * config.generation.max_tokens),
        ),
    )

    shared = {
        "strategy": "lgagent",
        "lgagent_plus_enabled": True,
        "web_search_enabled": False,
        "oath_rag_enabled": True,
        "cape_v_enabled": True,
        "evidence_lanes": EVIDENCE_LANES,
        "require_temporal_match": oath.require_temporal_match,
        "graph_hops": oath.graph_hops,
        "permutation_count": cape.permutation_count,
        "enabled_verifiers": enabled_verifiers,
        "budget_policy": "adaptive",
        "max_model_calls": budget.max_calls,
        "max_total_tokens": budget.max_tokens,
        "max_rounds": budget.max_rounds,
    }
    experiments = (
        _experiment(
            key="original",
            label="Original",
            family="main",
            description="Corrected original LGAgent without OATH-RAG or CAPE-V.",
            base=shared,
            lgagent_plus_enabled=False,
            oath_rag_enabled=False,
            cape_v_enabled=False,
            evidence_lanes=(),
            permutation_count=0,
            enabled_verifiers=(),
        ),
        _experiment(
            key="web-search",
            label="Budgeted Web Search",
            family="main",
            description=(
                "Corrected original LGAgent with gated real-time web evidence."
            ),
            base=shared,
            web_search_enabled=True,
            oath_rag_enabled=False,
            cape_v_enabled=False,
            evidence_lanes=(),
            permutation_count=0,
            enabled_verifiers=(),
        ),
        _experiment(
            key="oath-only",
            label="OATH-only",
            family="main",
            description="Corrected original LGAgent with OATH-RAG only.",
            base=shared,
            cape_v_enabled=False,
            permutation_count=0,
            enabled_verifiers=(),
        ),
        _experiment(
            key="cape-only",
            label="CAPE-only",
            family="main",
            description="Corrected original LGAgent with CAPE-V only.",
            base=shared,
            oath_rag_enabled=False,
            evidence_lanes=(),
        ),
        _experiment(
            key="joint",
            label="Joint",
            family="main",
            description="Joint OATH-RAG and CAPE-V system.",
            base=shared,
        ),
        _experiment(
            key="no-refute",
            label="no-refute",
            family="oath-ablation",
            description="Joint system without the OATH-RAG refute lane.",
            base=shared,
            evidence_lanes=("support", "exception"),
        ),
        _experiment(
            key="no-exception",
            label="no-exception",
            family="oath-ablation",
            description="Joint system without the OATH-RAG exception lane.",
            base=shared,
            evidence_lanes=("support", "refute"),
        ),
        _experiment(
            key="no-temporal",
            label="no-temporal",
            family="oath-ablation",
            description="Joint system without temporal matching.",
            base=shared,
            require_temporal_match=False,
        ),
        _experiment(
            key="no-graph-expansion",
            label="no-graph-expansion",
            family="oath-ablation",
            description="Joint system without one-hop legal graph expansion.",
            base=shared,
            graph_hops=0,
        ),
        _experiment(
            key="no-permutation",
            label="no-permutation",
            family="cape-ablation",
            description="Joint system without option permutation probes.",
            base=shared,
            permutation_count=0,
        ),
        _experiment(
            key="single-verifier",
            label="single-verifier",
            family="cape-ablation",
            description="Joint system using only the entailment verifier.",
            base=shared,
            enabled_verifiers=("entailment",),
        ),
        _experiment(
            key="fixed-budget",
            label="fixed-budget",
            family="compute-ablation",
            description="Joint system with fixed rather than risk-adaptive compute.",
            base=shared,
            budget_policy="fixed",
        ),
        _experiment(
            key="self-consistency-equal-token",
            label="Self-Consistency (equal-token)",
            family="control",
            description="Self-Consistency control under the Joint token ceiling.",
            base=shared,
            strategy="self-consistency",
            lgagent_plus_enabled=False,
            oath_rag_enabled=False,
            cape_v_enabled=False,
            evidence_lanes=(),
            permutation_count=0,
            enabled_verifiers=(),
            budget_policy="fixed",
            self_consistency_samples=self_consistency_samples,
            self_consistency_max_tokens=min(
                config.generation.max_tokens,
                budget.max_tokens
                // (baseline_worst_case_calls * self_consistency_samples),
            ),
            equal_token_reference="joint",
        ),
    )
    validate_task14_matrix(experiments)
    return experiments


def validate_task14_matrix(experiments: Sequence[AblationExperiment]) -> None:
    """Validate completeness, feature isolation, and fair token ceilings."""
    by_key = {experiment.key: experiment for experiment in experiments}
    if len(by_key) != len(experiments):
        raise AblationMatrixError("experiment keys must be unique")
    if tuple(by_key) != REQUIRED_EXPERIMENT_KEYS:
        missing = sorted(set(REQUIRED_EXPERIMENT_KEYS) - set(by_key))
        extra = sorted(set(by_key) - set(REQUIRED_EXPERIMENT_KEYS))
        raise AblationMatrixError(
            f"matrix keys/order mismatch; missing={missing}, extra={extra}"
        )

    for item in experiments:
        if item.strategy not in {"lgagent", "self-consistency"}:
            raise AblationMatrixError(f"{item.key} has an unknown strategy")
        if item.budget_policy not in {"adaptive", "fixed"}:
            raise AblationMatrixError(f"{item.key} has an unknown budget policy")
        if len(set(item.evidence_lanes)) != len(item.evidence_lanes) or not set(
            item.evidence_lanes
        ).issubset(EVIDENCE_LANES):
            raise AblationMatrixError(f"{item.key} has invalid evidence lanes")
        if not item.oath_rag_enabled and item.evidence_lanes:
            raise AblationMatrixError(
                f"{item.key} cannot use evidence lanes with OATH-RAG disabled"
            )
        if len(set(item.enabled_verifiers)) != len(item.enabled_verifiers) or not set(
            item.enabled_verifiers
        ).issubset(VERIFIER_NAMES):
            raise AblationMatrixError(f"{item.key} has invalid verifiers")
        if not item.cape_v_enabled and item.enabled_verifiers:
            raise AblationMatrixError(
                f"{item.key} cannot use verifiers with CAPE-V disabled"
            )
        if item.web_search_enabled and (
            item.oath_rag_enabled or item.cape_v_enabled
        ):
            raise AblationMatrixError(
                f"{item.key} cannot combine web search with OATH-RAG or CAPE-V"
            )
        integer_limits = {
            "graph_hops": (item.graph_hops, 0),
            "permutation_count": (item.permutation_count, 0),
            "max_model_calls": (item.max_model_calls, 1),
            "max_total_tokens": (item.max_total_tokens, 1),
            "max_rounds": (item.max_rounds, 1),
            "self_consistency_samples": (item.self_consistency_samples, 0),
            "self_consistency_max_tokens": (
                item.self_consistency_max_tokens,
                0,
            ),
        }
        for name, (value, minimum) in integer_limits.items():
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise AblationMatrixError(
                    f"{item.key}.{name} must be an integer >= {minimum}"
                )
        if item.graph_hops not in (0, 1):
            raise AblationMatrixError(f"{item.key}.graph_hops must be 0 or 1")
        if item.permutation_count > 23:
            raise AblationMatrixError(
                f"{item.key}.permutation_count must not exceed 23"
            )

    joint = by_key["joint"]
    if not (
        joint.lgagent_plus_enabled
        and joint.oath_rag_enabled
        and joint.cape_v_enabled
    ):
        raise AblationMatrixError("joint must enable LGAgent++, OATH-RAG, and CAPE-V")
    if joint.evidence_lanes != EVIDENCE_LANES:
        raise AblationMatrixError("joint must include all evidence lanes")
    if joint.permutation_count <= 0:
        raise AblationMatrixError("joint must include permutation probes")
    if set(joint.enabled_verifiers) != set(VERIFIER_NAMES):
        raise AblationMatrixError("joint must enable every verifier")
    web_search = by_key["web-search"]
    if not web_search.web_search_enabled:
        raise AblationMatrixError("web-search must enable budgeted web retrieval")
    if any(
        item.web_search_enabled
        for key, item in by_key.items()
        if key != "web-search"
    ):
        raise AblationMatrixError(
            "budgeted web retrieval must remain isolated to web-search"
        )

    expected_main = {
        "original": (False, False, False),
        "web-search": (True, False, False),
        "oath-only": (True, True, False),
        "cape-only": (True, False, True),
        "joint": (True, True, True),
    }
    for key, expected in expected_main.items():
        item = by_key[key]
        actual = (
            item.lgagent_plus_enabled,
            item.oath_rag_enabled,
            item.cape_v_enabled,
        )
        if actual != expected:
            raise AblationMatrixError(f"{key} has invalid main feature switches")

    if by_key["no-refute"].evidence_lanes != ("support", "exception"):
        raise AblationMatrixError("no-refute must disable only the refute lane")
    if by_key["no-exception"].evidence_lanes != ("support", "refute"):
        raise AblationMatrixError("no-exception must disable only the exception lane")
    if by_key["no-temporal"].require_temporal_match:
        raise AblationMatrixError("no-temporal must disable temporal matching")
    if by_key["no-graph-expansion"].graph_hops != 0:
        raise AblationMatrixError("no-graph-expansion must set graph_hops to zero")
    if by_key["no-permutation"].permutation_count != 0:
        raise AblationMatrixError("no-permutation must set permutation_count to zero")
    if len(by_key["single-verifier"].enabled_verifiers) != 1:
        raise AblationMatrixError("single-verifier must enable exactly one verifier")
    if by_key["fixed-budget"].budget_policy != "fixed":
        raise AblationMatrixError("fixed-budget must use the fixed budget policy")

    behavior_fields = (
        "strategy",
        "lgagent_plus_enabled",
        "web_search_enabled",
        "oath_rag_enabled",
        "cape_v_enabled",
        "evidence_lanes",
        "require_temporal_match",
        "graph_hops",
        "permutation_count",
        "enabled_verifiers",
        "budget_policy",
        "max_model_calls",
        "max_total_tokens",
        "max_rounds",
        "self_consistency_samples",
        "self_consistency_max_tokens",
        "equal_token_reference",
    )
    isolated_changes = {
        "no-refute": {"evidence_lanes"},
        "no-exception": {"evidence_lanes"},
        "no-temporal": {"require_temporal_match"},
        "no-graph-expansion": {"graph_hops"},
        "no-permutation": {"permutation_count"},
        "single-verifier": {"enabled_verifiers"},
        "fixed-budget": {"budget_policy"},
    }
    for key, expected_changes in isolated_changes.items():
        item = by_key[key]
        actual_changes = {
            field
            for field in behavior_fields
            if getattr(item, field) != getattr(joint, field)
        }
        if actual_changes != expected_changes:
            raise AblationMatrixError(
                f"{key} must differ from joint only in {sorted(expected_changes)}"
            )

    token_budgets = {item.max_total_tokens for item in experiments}
    if len(token_budgets) != 1:
        raise AblationMatrixError("all experiments must share one total token ceiling")
    control = by_key["self-consistency-equal-token"]
    if control.strategy != "self-consistency":
        raise AblationMatrixError("Self-Consistency control has the wrong strategy")
    if control.equal_token_reference != "joint":
        raise AblationMatrixError("Self-Consistency must reference joint")
    if control.self_consistency_samples <= 0:
        raise AblationMatrixError("Self-Consistency must schedule at least one sample")
    if (
        control.self_consistency_samples * control.self_consistency_max_tokens
        > joint.max_total_tokens
    ):
        raise AblationMatrixError("Self-Consistency exceeds the Joint token ceiling")


def task14_matrix_payload(
    experiments: Sequence[AblationExperiment],
) -> dict[str, Any]:
    validate_task14_matrix(experiments)
    rows = [experiment.as_dict() for experiment in experiments]
    return {
        "schema_version": 1,
        "matrix_id": stable_hash(rows)[:24],
        "offline_only": True,
        "experiment_count": len(rows),
        "experiments": rows,
    }


def export_task14_matrix(
    experiments: Sequence[AblationExperiment],
    output_dir: str | Path,
) -> dict[str, Path]:
    """Export the validated matrix as deterministic JSON, CSV, and Markdown."""
    payload = task14_matrix_payload(experiments)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    paths = {
        "json": destination / "task14_ablation_matrix.json",
        "csv": destination / "task14_ablation_matrix.csv",
        "markdown": destination / "task14_ablation_matrix.md",
    }
    paths["json"].write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    csv_fields = (
        "key",
        "label",
        "family",
        "strategy",
        "lgagent_plus_enabled",
        "web_search_enabled",
        "oath_rag_enabled",
        "cape_v_enabled",
        "evidence_lanes",
        "require_temporal_match",
        "graph_hops",
        "permutation_count",
        "enabled_verifiers",
        "budget_policy",
        "max_model_calls",
        "max_total_tokens",
        "max_rounds",
        "self_consistency_samples",
        "self_consistency_max_tokens",
        "equal_token_reference",
        "description",
    )
    with paths["csv"].open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=csv_fields)
        writer.writeheader()
        for experiment in experiments:
            row = experiment.as_dict()
            row["evidence_lanes"] = ",".join(experiment.evidence_lanes)
            row["enabled_verifiers"] = ",".join(experiment.enabled_verifiers)
            writer.writerow(row)

    lines = [
        "# Task 14 Ablation Matrix",
        "",
        f"- Matrix ID: `{payload['matrix_id']}`",
        f"- Experiments: {payload['experiment_count']}",
        "- Offline enumeration: yes",
        "",
        "| Experiment | Family | OATH | CAPE | Lanes | Permutations | Verifiers | Budget | Tokens |",
        "|---|---|---:|---:|---|---:|---|---|---:|",
    ]
    for item in experiments:
        lines.append(
            "| {label} | {family} | {oath} | {cape} | {lanes} | {permutations} | "
            "{verifiers} | {budget} | {tokens} |".format(
                label=item.label,
                family=item.family,
                oath=int(item.oath_rag_enabled),
                cape=int(item.cape_v_enabled),
                lanes=", ".join(item.evidence_lanes) or "-",
                permutations=item.permutation_count,
                verifiers=", ".join(item.enabled_verifiers) or "-",
                budget=item.budget_policy,
                tokens=item.max_total_tokens,
            )
        )
    paths["markdown"].write_text("\n".join(lines) + "\n", encoding="utf-8")
    return paths
