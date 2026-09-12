# CLIR Offline Prototype

## Scope

The first CLIR prototype tests the core hypothesis before adding another paid
runtime branch:

> An intervention should be selected only when its paired accuracy uplift over
> Original LGAgent has a positive conservative lower bound.

It does not yet train an individual treatment-effect model and does not invoke
models or retrievers during policy replay.

## Data Contract

`build_clir_dataset.py` pairs Original and intervention result records by
`sample_id`. It requires identical sample membership, questions, labels, data
splits, model IDs, dataset hashes, requested seeds, and prompt versions.
Experiment keys may differ. Failed intervention outcomes remain in the dataset
and count as incorrect.

Each LawIntervene prototype row contains:

- Original outcome and usage.
- One or more intervention outcomes and usage.
- Split and legal domain.
- Label-free routing features such as B0 confidence and question length.

## Calibration

For action `a`, the paired outcome is:

```text
D_i(a) = correctness_i(a) - correctness_i(original)
```

where `D_i(a)` is `-1`, `0`, or `1`. The prototype estimates mean uplift and a
one-sided Hoeffding lower bound. The configured error probability is divided
by the number of actions using Bonferroni correction. Domain estimates are used
only when their sample count reaches `min_group_size`; otherwise routing falls
back to the global estimate.

The bound controls the mean paired uplift under independent, exchangeable
calibration examples. It is not an individual causal-effect guarantee and does
not justify causal claims when treatment arms were collected under incompatible
models, prompts, datasets, or seeds.

## Routing

For each action:

```text
utility = uplift_lower_bound
          - cost_weight * max(0, incremental_tokens) / token_scale
```

CLIR selects the highest-utility action only when utility is strictly greater
than `minimum_uplift`. Otherwise it returns `original`.

## Commands

Build paired intervention records:

```bash
python tools/build_clir_dataset.py \
  --baseline output/original/results.jsonl \
  --action web-search=output/web/results.jsonl \
  --action oath-only=output/oath/results.jsonl \
  --output output/clir/interventions.jsonl
```

Fit only the development rows:

```bash
python tools/calibrate_clir.py \
  output/clir/interventions.jsonl \
  --output output/clir/calibration.json \
  --alpha 0.1 \
  --min-calibration-size 30 \
  --min-group-size 30
```

Replay on already-observed test arms without model calls:

```bash
python tools/evaluate_clir.py \
  output/clir/interventions.jsonl \
  --calibration output/clir/calibration.json \
  --output-dir output/clir/test_replay \
  --expected-split test
```

## Current Smoke Result

The existing Qwen3-8B 100-question Original/Web experiment yielded 13 paired
development records and 15 paired test records:

- Development Web mean uplift: `+0.0769`.
- Development 90% Hoeffding lower bound: `-0.5183`.
- Test actions selected: Original `15`, Web `0`.
- Test baseline and routed accuracy: `0.40`.
- Additional routed tokens: `0`.

This is a pipeline smoke test, not an effectiveness claim. The development set
is below the intended minimum of 30 and includes only one intervention action.

## Next Prototype Stage

1. Collect at least 200 independent development questions for Original,
   authority retrieval, symbolic checking, and heterogeneous verification.
2. Replace domain-only strata with cross-fitted uplift models over label-free
   pre-intervention features.
3. Calibrate excess-risk bounds on a held-out calibration partition.
4. Add proof-certificate validation before an intervention can change B0.
