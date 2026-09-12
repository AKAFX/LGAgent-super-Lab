"""Deterministic dataset split and experiment identity freeze for Task 16."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .ablation import REQUIRED_EXPERIMENT_KEYS
from .config import load_lgagent_config
from .evaluation import sha256_file, stable_hash

SPLIT_ALGORITHM = "sha256(seed:dataset_sha256:sample_id)-mod-10000-v1"
DEFAULT_SPLIT_SEED = 20260826
DEFAULT_RUN_SEEDS = (42, 43, 44)
DEFAULT_RATIOS = {"train": 8000, "dev": 1000, "test": 1000}
DEFAULT_PRIMARY_DATASETS = (
    "data/LexGenius.jsonl",
    "data/CAIL2022.jsonl",
    "data/lawbench_merged.jsonl",
)
PROMPT_VERSION = "lgagent-plus-structured-prompts-v1"
MAIN_EXPERIMENT_KEYS = (
    "original",
    "web-search",
    "oath-only",
    "cape-only",
    "joint",
)
SINGLE_FACTOR_ABLATION_KEYS = (
    "no-refute",
    "no-exception",
    "no-temporal",
    "no-graph-expansion",
    "no-permutation",
    "single-verifier",
    "fixed-budget",
)


@dataclass(frozen=True)
class ExperimentProfile:
    split: str
    experiment_keys: tuple[str, ...]
    max_examples: int | None
    seeds: tuple[int, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "split": self.split,
            "experiment_keys": list(self.experiment_keys),
            "max_examples_per_dataset": self.max_examples,
            "seeds": list(self.seeds),
        }


EXPERIMENT_PROFILES = {
    "pilot": ExperimentProfile(
        split="dev",
        experiment_keys=MAIN_EXPERIMENT_KEYS,
        max_examples=20,
        seeds=(42,),
    ),
    "main": ExperimentProfile(
        split="test",
        experiment_keys=MAIN_EXPERIMENT_KEYS,
        max_examples=200,
        seeds=DEFAULT_RUN_SEEDS,
    ),
    "ablation": ExperimentProfile(
        split="dev",
        experiment_keys=SINGLE_FACTOR_ABLATION_KEYS,
        max_examples=100,
        seeds=(42,),
    ),
    "full": ExperimentProfile(
        split="test",
        experiment_keys=REQUIRED_EXPERIMENT_KEYS,
        max_examples=None,
        seeds=DEFAULT_RUN_SEEDS,
    ),
}


def staged_protocol_manifest() -> dict[str, Any]:
    return {
        "default_profile": "pilot",
        "full_requires_explicit_profile": True,
        "profiles": {
            name: profile.as_dict()
            for name, profile in EXPERIMENT_PROFILES.items()
        },
    }


class FreezeValidationError(ValueError):
    """Raised when data no longer matches the frozen experiment contract."""


@dataclass(frozen=True)
class FrozenSample:
    index: int
    sample_id: str
    split: str
    record: Mapping[str, Any]


def _sample_identity(index: int, record: Mapping[str, Any]) -> str:
    explicit = record.get("id")
    question = str(record.get("question") or "")
    value = {"index": index, "id": explicit, "question": question}
    return stable_hash(value)[:24]


def assigned_split(
    sample_id: str,
    dataset_sha256: str,
    *,
    seed: int = DEFAULT_SPLIT_SEED,
    ratios: Mapping[str, int] = DEFAULT_RATIOS,
) -> str:
    split_names = ("train", "dev", "test")
    if set(ratios) != set(split_names) or sum(ratios.values()) != 10_000:
        raise FreezeValidationError("split ratios must define train/dev/test and sum to 10000")
    digest = hashlib.sha256(
        f"{seed}:{dataset_sha256}:{sample_id}".encode("utf-8")
    ).digest()
    bucket = int.from_bytes(digest[:8], "big") % 10_000
    boundary = 0
    for name in split_names:
        size = ratios[name]
        boundary += size
        if bucket < boundary:
            return name
    raise AssertionError("unreachable split bucket")


def load_frozen_samples(
    path: str | Path,
    *,
    split: str,
    expected_sha256: str | None = None,
    seed: int = DEFAULT_SPLIT_SEED,
    ratios: Mapping[str, int] = DEFAULT_RATIOS,
) -> tuple[FrozenSample, ...]:
    dataset_path = Path(path)
    digest = sha256_file(dataset_path)
    if expected_sha256 is not None and digest != expected_sha256:
        raise FreezeValidationError(
            f"dataset hash mismatch for {dataset_path}: expected "
            f"{expected_sha256}, got {digest}"
        )
    samples: list[FrozenSample] = []
    with dataset_path.open(encoding="utf-8") as stream:
        for index, line in enumerate(stream):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise FreezeValidationError(
                    f"{dataset_path}:{index + 1}: invalid JSON"
                ) from exc
            if not isinstance(record, Mapping):
                raise FreezeValidationError(
                    f"{dataset_path}:{index + 1}: record must be an object"
                )
            if not str(record.get("question") or "").strip():
                raise FreezeValidationError(
                    f"{dataset_path}:{index + 1}: missing question"
                )
            sample_id = _sample_identity(index, record)
            assigned = assigned_split(
                sample_id, digest, seed=seed, ratios=ratios
            )
            if split == "all" or assigned == split:
                samples.append(FrozenSample(index, sample_id, assigned, record))
    return tuple(samples)


def build_freeze_manifest(
    project_root: str | Path,
    *,
    config_path: str | Path,
    datasets: Sequence[str] = DEFAULT_PRIMARY_DATASETS,
    split_seed: int = DEFAULT_SPLIT_SEED,
    run_seeds: Sequence[int] = DEFAULT_RUN_SEEDS,
) -> dict[str, Any]:
    root = Path(project_root)
    config = Path(config_path)
    if not config.is_absolute():
        config = root / config
    loaded_config = load_lgagent_config(config, environ={})
    prompt_sources = (
        "src/lgagent/domain.py",
        "src/lgagent/cape_v.py",
        "src/lgagent/verification.py",
        "src/lgagent/evidence_audit.py",
        "src/lgagent/counterfactual.py",
        "src/lgagent/clex.py",
        "src/lgagent/oath_rag.py",
        "src/lgagent/risk.py",
        "src/lgagent/model.py",
        "src/lgagent/protocol.py",
        "src/lgagent/runner.py",
        "src/lgagent/experiment_runner.py",
        "src/lgagent/web_search.py",
        "src/lgagent/config.py",
        "src/lgagent/trace.py",
        "tools/run_legal_mcq.py",
        *sorted(
            str(path.relative_to(root))
            for path in (root / "src/lgagent/legal_mcq").rglob("*")
            if path.suffix in {".py", ".yaml"}
        ),
    )
    dataset_entries: dict[str, Any] = {}
    for relative in datasets:
        path = root / relative
        digest = sha256_file(path)
        all_samples = load_frozen_samples(
            path, split="all", expected_sha256=digest, seed=split_seed
        )
        counts = {
            name: sum(sample.split == name for sample in all_samples)
            for name in DEFAULT_RATIOS
        }
        membership = {
            name: stable_hash(
                [sample.sample_id for sample in all_samples if sample.split == name]
            )
            for name in DEFAULT_RATIOS
        }
        dataset_entries[relative] = {
            "sha256": digest,
            "samples": len(all_samples),
            "split_counts": counts,
            "split_membership_sha256": membership,
        }
    payload = {
        "schema_version": 1,
        "primary_datasets": dataset_entries,
        "split": {
            "algorithm": SPLIT_ALGORITHM,
            "seed": split_seed,
            "ratios_per_10000": dict(DEFAULT_RATIOS),
            "materialization": "indices are derived at runtime; datasets are not copied",
        },
        "runs": {
            "seeds": list(run_seeds),
            "repeats": len(run_seeds),
            "prompt_version": PROMPT_VERSION,
            "prompt_source_sha256": {
                relative: sha256_file(root / relative)
                for relative in prompt_sources
            },
            "model": {
                "backend": loaded_config.generation.backend,
                "model_id": loaded_config.generation.model,
                "temperature": loaded_config.generation.temperature,
                "top_p": loaded_config.generation.top_p,
                "max_tokens": loaded_config.generation.max_tokens,
                "verifier_model_id": (
                    loaded_config.lgagent_plus.cape_v.verifier_model.model
                    if loaded_config.lgagent_plus.cape_v.verifier_model
                    else loaded_config.generation.model
                ),
            },
        },
        "staged_protocol": staged_protocol_manifest(),
        "config": {
            "path": str(config.relative_to(root)),
            "sha256": sha256_file(config),
        },
    }
    return {**payload, "freeze_id": stable_hash(payload)[:24]}


def validate_freeze_manifest(
    manifest: Mapping[str, Any],
    project_root: str | Path,
) -> None:
    root = Path(project_root)
    if manifest.get("schema_version") != 1:
        raise FreezeValidationError("unsupported freeze manifest schema")
    split = manifest.get("split")
    runs = manifest.get("runs")
    config = manifest.get("config")
    datasets = manifest.get("primary_datasets")
    if not all(isinstance(item, Mapping) for item in (split, runs, config, datasets)):
        raise FreezeValidationError("freeze manifest sections are missing")
    assert isinstance(split, Mapping)
    assert isinstance(runs, Mapping)
    assert isinstance(config, Mapping)
    assert isinstance(datasets, Mapping)
    if split.get("algorithm") != SPLIT_ALGORITHM:
        raise FreezeValidationError("split algorithm does not match implementation")
    if manifest.get("staged_protocol") != staged_protocol_manifest():
        raise FreezeValidationError("staged experiment protocol does not match implementation")
    seeds = runs.get("seeds")
    if (
        not isinstance(seeds, list)
        or len(seeds) < 3
        or any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds)
    ):
        raise FreezeValidationError("at least three integer run seeds are required")
    model = runs.get("model")
    prompt_hashes = runs.get("prompt_source_sha256")
    if not isinstance(model, Mapping) or not str(model.get("model_id") or ""):
        raise FreezeValidationError("frozen model identity is missing")
    if not isinstance(prompt_hashes, Mapping) or not prompt_hashes:
        raise FreezeValidationError("frozen prompt source hashes are missing")
    for relative, expected_hash in prompt_hashes.items():
        if sha256_file(root / str(relative)) != expected_hash:
            raise FreezeValidationError(f"prompt source hash mismatch: {relative}")
    config_path = root / str(config.get("path") or "")
    if sha256_file(config_path) != config.get("sha256"):
        raise FreezeValidationError("frozen config hash mismatch")
    ratios = split.get("ratios_per_10000")
    if not isinstance(ratios, Mapping):
        raise FreezeValidationError("split ratios are missing")
    seed = split.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise FreezeValidationError("split seed must be an integer")
    for relative, expected in datasets.items():
        if not isinstance(expected, Mapping):
            raise FreezeValidationError(f"invalid dataset entry: {relative}")
        samples = load_frozen_samples(
            root / str(relative),
            split="all",
            expected_sha256=str(expected.get("sha256") or ""),
            seed=seed,
            ratios={str(key): int(value) for key, value in ratios.items()},
        )
        counts = {
            name: sum(sample.split == name for sample in samples)
            for name in ratios
        }
        memberships = {
            name: stable_hash(
                [sample.sample_id for sample in samples if sample.split == name]
            )
            for name in ratios
        }
        if len(samples) != expected.get("samples"):
            raise FreezeValidationError(f"sample count mismatch: {relative}")
        if counts != expected.get("split_counts"):
            raise FreezeValidationError(f"split count mismatch: {relative}")
        if memberships != expected.get("split_membership_sha256"):
            raise FreezeValidationError(f"split membership mismatch: {relative}")
    recomputed = dict(manifest)
    freeze_id = recomputed.pop("freeze_id", None)
    if freeze_id != stable_hash(recomputed)[:24]:
        raise FreezeValidationError("freeze_id mismatch")
