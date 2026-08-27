"""Credential-gated, resumable execution of the frozen Task 14 matrix."""

from __future__ import annotations

import json
import os
import tempfile
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping, Sequence

from .ablation import AblationExperiment, build_task14_matrix
from .config import LGAgentConfig, load_lgagent_config
from .corpus import LoadedCorpus, load_jsonl_corpus
from .evidence_audit import (
    EvidenceAuditConfig,
    EvidenceAuditPipeline,
    EvidenceAuditor,
    RetrievalFailureMode,
)
from .evaluation import (
    ExperimentCheckpoint,
    ExperimentMetadata,
    UsageRecord,
    build_result_record,
    evaluate_result_records,
    stable_hash,
)
from .experiment_freeze import (
    PROMPT_VERSION,
    FrozenSample,
    load_frozen_samples,
    validate_freeze_manifest,
)
from .model import OpenAIChatModel
from .oath_rag import OathRagConfig, OathRagRetriever
from .protocol import LawyerAOutput
from .runner import LGAgentPlusRunner
from .serialization import to_jsonable


class ExperimentRunError(RuntimeError):
    """Raised when an experiment cannot run without violating its contract."""


def _retriever_config(
    config: LGAgentConfig,
    experiment: AblationExperiment,
) -> OathRagConfig:
    settings = config.lgagent_plus.oath_rag
    return OathRagConfig(
        lexical_top_k=settings.lexical_top_k,
        dense_top_k=settings.dense_top_k,
        final_top_k_per_lane=settings.final_top_k_per_lane,
        min_authority_level=1 if settings.require_authoritative_source else 0,
        graph_hops=settings.graph_hops,
        enabled_lanes=experiment.evidence_lanes
        or ("support", "refute", "exception"),
        require_temporal_match=settings.require_temporal_match,
    )


class _SharedRetrievalResources:
    """Process-local immutable corpus and per-experiment retriever cache."""

    def __init__(self, corpus_path: Path) -> None:
        self._corpus_path = corpus_path
        self._lock = threading.RLock()
        self._corpus: LoadedCorpus | None = None
        self._retrievers: dict[
            tuple[str, OathRagConfig], OathRagRetriever
        ] = {}

    def corpus(self) -> LoadedCorpus:
        with self._lock:
            if self._corpus is None:
                self._corpus = load_jsonl_corpus(
                    self._corpus_path,
                    allow_legacy=False,
                )
            return self._corpus

    def retriever_for(
        self,
        config: LGAgentConfig,
        experiment: AblationExperiment,
    ) -> OathRagRetriever | None:
        if not (
            config.lgagent_plus.enabled
            and config.lgagent_plus.oath_rag.enabled
        ):
            return None
        retriever_config = _retriever_config(config, experiment)
        key = (experiment.key, retriever_config)
        with self._lock:
            retriever = self._retrievers.get(key)
            if retriever is None:
                retriever = OathRagRetriever(
                    self.corpus(),
                    config=retriever_config,
                )
                self._retrievers[key] = retriever
            return retriever


def extract_sample_domain(
    record: Mapping[str, Any],
    inferred_domain: str = "",
) -> str:
    """Extract the most specific available domain label from dataset metadata."""
    meta_data = record.get("meta_data")
    metadata = meta_data if isinstance(meta_data, Mapping) else {}
    candidates = (
        metadata.get("domain"),
        record.get("domain"),
        metadata.get("subject"),
        record.get("subject"),
        metadata.get("type"),
        inferred_domain,
        metadata.get("source_file"),
    )
    for value in candidates:
        if value is not None and (normalized := str(value).strip()):
            return normalized
    return ""


def _evidence_ids(matrix: Mapping[str, Any]) -> list[str]:
    found: list[str] = []
    options = matrix.get("options", {})
    if not isinstance(options, Mapping):
        return found
    for option in options.values():
        if not isinstance(option, Mapping):
            continue
        for lane in ("support", "refute", "exception", "temporally_invalid"):
            items = option.get(lane, ())
            if not isinstance(items, Sequence) or isinstance(items, (str, bytes)):
                continue
            for item in items:
                if isinstance(item, Mapping) and item.get("evidence_id"):
                    found.append(str(item["evidence_id"]))
    return list(dict.fromkeys(found))


def _cited_evidence_ids(candidates: Sequence[Any]) -> list[str]:
    found: list[str] = []
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            continue
        irac = candidate.get("irac", {})
        rules = irac.get("rule", ()) if isinstance(irac, Mapping) else ()
        if not isinstance(rules, Sequence) or isinstance(rules, (str, bytes)):
            continue
        for rule in rules:
            if not isinstance(rule, Mapping):
                continue
            evidence_ids = rule.get("evidence_ids", ())
            if isinstance(evidence_ids, Sequence) and not isinstance(
                evidence_ids, (str, bytes)
            ):
                found.extend(str(item) for item in evidence_ids if item)
    return list(dict.fromkeys(found))


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(
                to_jsonable(payload),
                stream,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _atomic_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            for record in records:
                stream.write(
                    json.dumps(
                        to_jsonable(record),
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def apply_experiment(
    base: LGAgentConfig,
    experiment: AblationExperiment,
    *,
    seed: int,
    corpus_path: str | Path,
) -> LGAgentConfig:
    plus = base.lgagent_plus
    oath = replace(
        plus.oath_rag,
        enabled=experiment.oath_rag_enabled,
        corpus_path=str(corpus_path),
        graph_hops=experiment.graph_hops,
        require_temporal_match=experiment.require_temporal_match,
    )
    enabled_verifiers = set(experiment.enabled_verifiers)
    cape = replace(
        plus.cape_v,
        enabled=experiment.cape_v_enabled,
        permutation_count=experiment.permutation_count,
        verifiers={
            name: name in enabled_verifiers for name in plus.cape_v.verifiers
        },
    )
    budget = replace(
        plus.risk.budget,
        max_calls=experiment.max_model_calls,
        max_tokens=experiment.max_total_tokens,
        max_rounds=experiment.max_rounds,
    )
    generation = (
        replace(
            base.generation,
            max_tokens=experiment.self_consistency_max_tokens,
        )
        if experiment.strategy == "self-consistency"
        else base.generation
    )
    return replace(
        base,
        generation=generation,
        lgagent_plus=replace(
            plus,
            enabled=experiment.lgagent_plus_enabled,
            seed=seed,
            oath_rag=oath,
            cape_v=cape,
            risk=replace(plus.risk, budget=budget),
        ),
    )


def estimate_calls(experiment: AblationExperiment) -> tuple[int, int]:
    """Conservative request-count range per example, excluding retries."""
    baseline_calls = 4
    if experiment.strategy == "self-consistency":
        return (
            baseline_calls * experiment.self_consistency_samples,
            (baseline_calls + 2 * experiment.max_rounds)
            * experiment.self_consistency_samples,
        )
    audit_calls = 4 if experiment.oath_rag_enabled else 0
    if not experiment.cape_v_enabled:
        calls = baseline_calls + audit_calls
        return calls, calls + 2 * experiment.max_rounds
    verifier_count = len(experiment.enabled_verifiers)
    # Permutation probes are candidate calls only: they measure stability and
    # neither vote nor run the five aspect verifiers.
    fast = (
        baseline_calls
        + audit_calls
        + 1
        + experiment.permutation_count
        + verifier_count
    )
    slow_round = 3 * (1 + verifier_count)
    return fast, min(
        experiment.max_model_calls,
        fast + experiment.max_rounds * slow_round,
    )


def _cost_warning(minimum: int, maximum: int) -> dict[str, Any]:
    if maximum >= 1_000_000:
        level = "critical"
    elif maximum >= 100_000:
        level = "high"
    elif maximum >= 5_000:
        level = "moderate"
    else:
        level = "low"
    messages = [
        f"Plan estimates {minimum:,}-{maximum:,} API requests before retries."
    ]
    if level in {"critical", "high"}:
        messages.append(
            "Paid execution has substantial cost; review per-configuration and "
            "per-dataset estimates before using --execute."
        )
    elif level == "moderate":
        messages.append(
            "Paid execution is non-trivial; confirm provider limits and budget "
            "before using --execute."
        )
    messages.append(
        "Request estimates exclude retries and are not token or currency estimates."
    )
    return {"level": level, "messages": messages}


def _usage_from_results(results: Sequence[Any], elapsed_ms: float) -> UsageRecord:
    traces = [result.trace for result in results]
    calls = [call for trace in traces for call in trace.model_calls]
    return UsageRecord(
        calls=len(calls),
        prompt_tokens=sum(call.usage.prompt_tokens for call in calls),
        completion_tokens=sum(call.usage.completion_tokens for call in calls),
        total_tokens=sum(call.usage.total_tokens for call in calls),
        latency_ms=elapsed_ms,
    )


def _model_clients(config: LGAgentConfig) -> tuple[OpenAIChatModel, OpenAIChatModel]:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ExperimentRunError("openai is required for non-dry-run execution") from exc
    generation = config.generation
    if not generation.api_key:
        raise ExperimentRunError(
            "no API key is configured; use --dry-run or set YAML api_key/LLM_API_KEY"
        )
    reasoning = OpenAIChatModel(
        OpenAI(api_key=generation.api_key, base_url=generation.base_url)
    )
    verifier_config = config.lgagent_plus.cape_v.verifier_model
    if verifier_config is None or (
        verifier_config.api_key == generation.api_key
        and verifier_config.base_url == generation.base_url
    ):
        verifier = reasoning
    else:
        if not verifier_config.api_key:
            raise ExperimentRunError("verifier model has no API key")
        verifier = OpenAIChatModel(
            OpenAI(
                api_key=verifier_config.api_key,
                base_url=verifier_config.base_url,
            )
        )
    return reasoning, verifier


def _majority_answer(answers: Sequence[str]) -> str:
    counts = Counter(answer for answer in answers if answer in "ABCD")
    if not counts:
        raise ExperimentRunError("all Self-Consistency samples failed")
    maximum = max(counts.values())
    return min(answer for answer, count in counts.items() if count == maximum)


def _redact_error(message: str, config: LGAgentConfig) -> str:
    keys = {
        config.generation.api_key,
        (
            config.lgagent_plus.cape_v.verifier_model.api_key
            if config.lgagent_plus.cape_v.verifier_model is not None
            else ""
        ),
    }
    for key in keys:
        if key:
            message = message.replace(key, "[REDACTED]")
    return message


def _run_sample(
    sample: FrozenSample,
    *,
    config: LGAgentConfig,
    experiment: AblationExperiment,
    project_root: Path,
    reproduction: Mapping[str, Any],
    retriever: OathRagRetriever | None = None,
) -> dict[str, Any]:
    question = str(sample.record["question"])
    golden = [str(value) for value in sample.record.get("golden_answers", [])]
    domain = extract_sample_domain(sample.record)
    started = perf_counter()
    results: list[Any] = []
    try:
        reasoning, verifier = _model_clients(config)
        evidence_pipeline: EvidenceAuditPipeline | None = None
        if config.lgagent_plus.enabled and config.lgagent_plus.oath_rag.enabled:
            if retriever is None:
                raise ExperimentRunError(
                    "enabled OATH-RAG experiment has no shared retriever"
                )
            failure_mode = (
                RetrievalFailureMode.FAIL_CLOSED
                if config.lgagent_plus.oath_rag.corpus_failure_mode
                == "fail_closed"
                else RetrievalFailureMode.EMPTY_EVIDENCE
            )
            evidence_pipeline = EvidenceAuditPipeline(
                retriever,
                EvidenceAuditor(
                    reasoning,
                    config.generation,
                    config=EvidenceAuditConfig(
                        retrieval_failure_mode=failure_mode
                    ),
                ),
            )
        repetitions = (
            experiment.self_consistency_samples
            if experiment.strategy == "self-consistency"
            else 1
        )
        for _ in range(repetitions):
            runner = LGAgentPlusRunner(
                config,
                reasoning_model=reasoning,
                evaluation_model=reasoning,
                verifier_model=verifier,
                evidence_pipeline=evidence_pipeline,
                project_root=project_root,
                evidence_lanes=experiment.evidence_lanes or (
                    "support",
                    "refute",
                    "exception",
                ),
                fixed_compute=experiment.budget_policy == "fixed",
            )
            results.append(runner.run(question))
        prediction = (
            _majority_answer([result.final_answer for result in results])
            if repetitions > 1
            else results[0].final_answer
        )
        diagnostics = results[-1].diagnostics
        try:
            inferred_domain = LawyerAOutput.from_text(
                results[-1].lawyer_a_output
            ).legal_domain
        except (AttributeError, ValueError):
            inferred_domain = ""
        domain = extract_sample_domain(sample.record, inferred_domain)
        if not domain:
            domain = "unclassified"
        candidates = diagnostics.get("candidates", [])
        evidence_matrix = diagnostics.get("evidence_matrix", {})
        risk_score = diagnostics.get("risk_score")
        confidence = (
            1.0 - float(risk_score)
            if isinstance(risk_score, (int, float))
            and not isinstance(risk_score, bool)
            else diagnostics.get("b0_confidence")
        )
        record = build_result_record(
            sample_id=sample.sample_id,
            prediction=prediction,
            golden_answers=golden,
            candidates=candidates,
            evidence_matrix=evidence_matrix,
            verification_reports=diagnostics.get("verification_reports", []),
            risk_routing=diagnostics.get("risk_routing", {}),
            usage=_usage_from_results(
                results, (perf_counter() - started) * 1000
            ),
            reproduction=reproduction,
            extra={
                "idx": sample.index,
                "question": question,
                "split": sample.split,
                "status": "ok",
                "domain": domain,
                "confidence": confidence,
                "risk_score": risk_score,
                "counterfactual_observations": diagnostics.get(
                    "counterfactual_observations", []
                ),
                "retrieved_evidence_ids": _evidence_ids(evidence_matrix),
                "cited_evidence_ids": _cited_evidence_ids(candidates),
                "permutation_predictions": diagnostics.get(
                    "permutation_predictions", []
                ),
                "trace": [result.trace.as_dict() for result in results],
            },
        )
    except Exception as exc:
        record = build_result_record(
            sample_id=sample.sample_id,
            prediction="",
            golden_answers=golden,
            usage=UsageRecord(latency_ms=(perf_counter() - started) * 1000),
            reproduction=reproduction,
            extra={
                "idx": sample.index,
                "question": question,
                "split": sample.split,
                "status": "failed",
                "domain": domain,
                "error": {
                    "type": type(exc).__name__,
                    "message": _redact_error(str(exc), config),
                },
            },
        )
    return record


class Task16ExperimentRunner:
    """Validate or execute all frozen matrix/dataset/seed combinations."""

    def __init__(
        self,
        *,
        project_root: str | Path,
        config_path: str | Path,
        freeze_path: str | Path,
        corpus_path: str | Path,
        output_dir: str | Path,
    ) -> None:
        self.root = Path(project_root).resolve()
        self.config_path = Path(config_path).resolve()
        self.freeze_path = Path(freeze_path).resolve()
        self.corpus_path = Path(corpus_path).resolve()
        self.output_dir = Path(output_dir).resolve()
        self._retrieval_resources = _SharedRetrievalResources(self.corpus_path)

    def _inputs(
        self,
    ) -> tuple[
        LGAgentConfig,
        Mapping[str, Any],
        tuple[AblationExperiment, ...],
        LoadedCorpus,
    ]:
        freeze = json.loads(self.freeze_path.read_text(encoding="utf-8"))
        validate_freeze_manifest(freeze, self.root)
        config = load_lgagent_config(self.config_path)
        corpus = self._retrieval_resources.corpus()
        return config, freeze, build_task14_matrix(config), corpus

    @staticmethod
    def _select_experiments(
        matrix: Sequence[AblationExperiment],
        experiment_keys: Sequence[str] | None,
    ) -> tuple[AblationExperiment, ...]:
        selected = set(experiment_keys or ())
        experiments = tuple(
            item for item in matrix if not selected or item.key in selected
        )
        if selected and selected != {item.key for item in experiments}:
            raise ExperimentRunError("unknown experiment key in selection")
        return experiments

    @staticmethod
    def _select_seeds(
        freeze: Mapping[str, Any],
        seeds: Sequence[int] | None,
    ) -> tuple[int, ...]:
        selected = tuple(
            int(seed) for seed in (
                seeds if seeds is not None else freeze["runs"]["seeds"]
            )
        )
        if not selected:
            raise ExperimentRunError("at least one seed is required")
        if len(set(selected)) != len(selected):
            raise ExperimentRunError("seeds must be unique")
        return selected

    def dry_run(
        self,
        *,
        split: str = "test",
        experiment_keys: Sequence[str] | None = None,
        max_examples: int | None = None,
        seeds: Sequence[int] | None = None,
        profile_name: str | None = None,
    ) -> dict[str, Any]:
        config, freeze, matrix, corpus = self._inputs()
        experiments = self._select_experiments(matrix, experiment_keys)
        run_seeds = self._select_seeds(freeze, seeds)
        rows = []
        by_experiment: dict[str, dict[str, Any]] = {}
        by_dataset: dict[str, dict[str, Any]] = {}
        total_calls_min = total_calls_max = total_jobs = 0
        for relative, dataset in freeze["primary_datasets"].items():
            samples = load_frozen_samples(
                self.root / relative,
                split=split,
                expected_sha256=dataset["sha256"],
                seed=int(freeze["split"]["seed"]),
                ratios=freeze["split"]["ratios_per_10000"],
            )
            count = min(len(samples), max_examples) if max_examples else len(samples)
            for experiment in experiments:
                minimum, maximum = estimate_calls(experiment)
                jobs = count * len(run_seeds)
                calls_min = jobs * minimum
                calls_max = jobs * maximum
                total_jobs += jobs
                total_calls_min += calls_min
                total_calls_max += calls_max
                rows.append(
                    {
                        "dataset": relative,
                        "split": split,
                        "samples_per_run": count,
                        "experiment": experiment.key,
                        "seeds": list(run_seeds),
                        "jobs": jobs,
                        "estimated_calls": {
                            "minimum": calls_min,
                            "maximum_without_retries": calls_max,
                        },
                        "max_tokens_per_example": experiment.max_total_tokens,
                    }
                )
                experiment_total = by_experiment.setdefault(
                    experiment.key,
                    {
                        "experiment": experiment.key,
                        "datasets": 0,
                        "jobs": 0,
                        "estimated_calls": {
                            "minimum": 0,
                            "maximum_without_retries": 0,
                        },
                    },
                )
                experiment_total["datasets"] += 1
                experiment_total["jobs"] += jobs
                experiment_total["estimated_calls"]["minimum"] += calls_min
                experiment_total["estimated_calls"][
                    "maximum_without_retries"
                ] += calls_max
                dataset_total = by_dataset.setdefault(
                    relative,
                    {
                        "dataset": relative,
                        "split": split,
                        "samples_per_run": count,
                        "experiments": 0,
                        "jobs": 0,
                        "estimated_calls": {
                            "minimum": 0,
                            "maximum_without_retries": 0,
                        },
                    },
                )
                dataset_total["experiments"] += 1
                dataset_total["jobs"] += jobs
                dataset_total["estimated_calls"]["minimum"] += calls_min
                dataset_total["estimated_calls"][
                    "maximum_without_retries"
                ] += calls_max
        report = {
            "dry_run": True,
            "makes_api_calls": False,
            "freeze_id": freeze["freeze_id"],
            "profile": profile_name,
            "resolved_plan": {
                "split": split,
                "experiment_keys": [item.key for item in experiments],
                "max_examples_per_dataset": max_examples,
                "seeds": list(run_seeds),
            },
            "matrix_id": stable_hash([item.as_dict() for item in experiments])[:24],
            "corpus": {
                "path": str(self.corpus_path),
                "records": corpus.manifest.document_count,
            },
            "jobs": total_jobs,
            "estimated_calls": {
                "minimum": total_calls_min,
                "maximum_without_retries": total_calls_max,
            },
            "cost_warning": _cost_warning(total_calls_min, total_calls_max),
            "by_experiment": list(by_experiment.values()),
            "by_dataset": list(by_dataset.values()),
            "accuracy": None,
            "note": "Validation and call estimates only; no accuracy was measured.",
            "runs": rows,
            "model": config.generation.model,
        }
        _atomic_json(self.output_dir / "dry_run_report.json", report)
        return report

    def run(
        self,
        *,
        split: str = "test",
        experiment_keys: Sequence[str] | None = None,
        max_examples: int | None = None,
        seeds: Sequence[int] | None = None,
        profile_name: str | None = None,
        concurrency: int = 1,
        checkpoint_every: int = 10,
    ) -> dict[str, Any]:
        config, freeze, matrix, _ = self._inputs()
        if not config.generation.api_key:
            raise ExperimentRunError(
                "no API key is configured; paid execution was not started"
            )
        experiments = self._select_experiments(matrix, experiment_keys)
        run_seeds = self._select_seeds(freeze, seeds)
        summaries: list[dict[str, Any]] = []
        for relative, dataset in freeze["primary_datasets"].items():
            samples = load_frozen_samples(
                self.root / relative,
                split=split,
                expected_sha256=dataset["sha256"],
                seed=int(freeze["split"]["seed"]),
                ratios=freeze["split"]["ratios_per_10000"],
            )
            if max_examples is not None:
                samples = samples[:max_examples]
            for experiment in experiments:
                for seed in run_seeds:
                    run_config = apply_experiment(
                        config,
                        experiment,
                        seed=int(seed),
                        corpus_path=self.corpus_path,
                    )
                    metadata = ExperimentMetadata(
                        model_id=run_config.generation.model,
                        endpoint_class="openai-compatible",
                        prompt_version=PROMPT_VERSION,
                        config_hash=stable_hash(
                            {
                                "base": freeze["config"]["sha256"],
                                "experiment": experiment.as_dict(),
                            }
                        ),
                        dataset_hash=dataset["sha256"],
                        random_seed=int(seed),
                        software_version="0.1.0",
                    )
                    reproduction = {
                        **metadata.as_dict(),
                        "seed_requested": int(seed),
                        "seed_derivation": (
                            "sha256(base_seed,metadata,occurrence)-31bit-v1"
                        ),
                        "provider_seed_guarantee": "requested_not_guaranteed",
                        "verifier_model_id": (
                            run_config.lgagent_plus.cape_v.verifier_model.model
                            if run_config.lgagent_plus.cape_v.verifier_model
                            else run_config.generation.model
                        ),
                        "experiment_key": experiment.key,
                        "freeze_id": freeze["freeze_id"],
                    }
                    run_dir = (
                        self.output_dir
                        / split
                        / Path(relative).stem
                        / experiment.key
                        / f"seed-{seed}"
                    )
                    checkpoint = ExperimentCheckpoint(
                        run_dir / "checkpoint.json", metadata
                    )
                    checkpoint.load()
                    pending = [
                        sample
                        for sample in samples
                        if sample.sample_id not in checkpoint.completed_ids
                    ]
                    shared_retriever = (
                        self._retrieval_resources.retriever_for(
                            run_config,
                            experiment,
                        )
                        if pending
                        else None
                    )
                    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
                        futures = {
                            pool.submit(
                                _run_sample,
                                sample,
                                config=run_config,
                                experiment=experiment,
                                project_root=self.root,
                                reproduction=reproduction,
                                retriever=shared_retriever,
                            ): sample
                            for sample in pending
                        }
                        for completed, future in enumerate(
                            as_completed(futures), start=1
                        ):
                            checkpoint.add(future.result())
                            if completed % max(1, checkpoint_every) == 0:
                                checkpoint.save({"complete": False})
                                _atomic_jsonl(
                                    run_dir / "results.jsonl",
                                    checkpoint.records,
                                )
                    records = checkpoint.records
                    metrics = evaluate_result_records(records)
                    failed = sum(
                        record.get("status") == "failed" for record in records
                    )
                    summary = {
                        "dataset": relative,
                        "experiment": experiment.key,
                        "seed": seed,
                        "experiment_id": metadata.experiment_id,
                        "samples": len(records),
                        "failed_samples": failed,
                        "metrics": metrics,
                    }
                    checkpoint.save({"complete": True, "summary": summary})
                    _atomic_jsonl(run_dir / "results.jsonl", records)
                    _atomic_json(run_dir / "summary.json", summary)
                    summaries.append(summary)
        report = {
            "dry_run": False,
            "freeze_id": freeze["freeze_id"],
            "profile": profile_name,
            "resolved_plan": {
                "split": split,
                "experiment_keys": [item.key for item in experiments],
                "max_examples_per_dataset": max_examples,
                "seeds": list(run_seeds),
            },
            "runs": summaries,
        }
        _atomic_json(self.output_dir / "experiment_summary.json", report)
        return report
