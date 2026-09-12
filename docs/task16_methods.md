# LGAgent++ Method and Reproduction Protocol

## Scope

This document freezes the executable method for Task 16. It does not report
experimental accuracy. No paid API experiment was run while preparing it.
`docs/task16_experiment_freeze.json` is the machine-readable source of truth for
dataset hashes, deterministic splits, prompt version, configuration hash, and
the three random seeds.

## Method

LGAgent++ couples two mechanisms:

1. **OATH-RAG** retrieves option-aligned support, refute, and exception evidence.
   It applies jurisdiction, authority, and legal-version gates before lexical
   retrieval, optional graph expansion, and evidence auditing.
2. **CAPE-V** generates answer candidates and option permutations, verifies rule,
   fact, exception, evidence, and entailment dimensions, and allocates additional
   computation according to measured risk and a hard budget.

For candidate \(c\), the deterministic aggregation score is

\[
S(c)=V(c)\,C(c)\,A(c)\,T(c)\,Q(c)\,P(c),
\]

where \(V\) is vote strength, \(C\) evidence coverage, \(A\) authority,
\(T\) temporal validity, \(Q\) verifier score, and \(P\) permutation
consistency. The implementation uses stable tie-breaking after these factors.

The architecture figure is available as
[`assets/lgagent-plus-task16.svg`](assets/lgagent-plus-task16.svg).

## OATH Corpus

The builder uses `imca0/just-laws` at commit
`b475e4e00e1640a927faf79c3ccae663c10c3bbd`. It accepts either a local checkout
at exactly that commit or performs a clone and detached checkout itself.

```bash
python tools/build_oath_corpus.py \
  --source-dir /path/to/just-laws \
  --output-dir data/oath
```

Without `--source-dir` or `--clone-dir`, the source is cloned into a temporary
directory. The output includes strict article-level JSONL, a corpus manifest,
an exclusion report, licensing metadata, and provenance policy.

Only law directories explicitly listed by the source repository's legal
category indexes, plus the Constitution, receive `authority_level=5`.
`effectiveFrom` in `versions.json` is accepted directly. For unversioned laws,
the builder requires exactly one explicit statutory statement of the form
“本法自 YYYY 年 MM 月 DD 日起施行” or its supported statutory-name variants.
Unknown or ambiguous dates are excluded and reported, never inferred.
`effectiveTo` in the upstream manifest is exclusive; OATH's inclusive
`effective_to` is therefore the preceding calendar day.

`REFERS_TO` is emitted only when an exact `《法律名》` reference resolves to a
known corpus title. Source URLs include the fixed commit.

## Frozen Evaluation

The primary files are not copied. Their SHA-256 hashes and deterministic split
membership hashes are frozen in `docs/task16_experiment_freeze.json`.
Assignment uses:

```text
sha256(split_seed : dataset_sha256 : stable_sample_id) mod 10000
```

Buckets are fixed to train/dev/test = 8000/1000/1000. The experiment split is
selected at runtime, and the runner rejects changed data or configuration.
Seeds are 42, 43, and 44. Paid execution follows the staged protocol below.

The frozen matrix contains all 13 configurations from `build_task14_matrix`:
Original, Budgeted Web Search, OATH-only, CAPE-only, Joint, seven single-factor
ablations, and the equal-token Self-Consistency control. Budgeted Web Search is
isolated from OATH-RAG, CAPE-V, and C-LEX. Every LGAgent configuration is
executed through `LGAgentPlusRunner`; Self-Consistency repeats its
corrected-original route and uses deterministic majority voting. Failed
samples remain in the result JSONL with an empty prediction and therefore
remain in the accuracy denominator.

### Staged protocol

Paid runs are deliberately staged. The CLI defaults to the recommended
`pilot` profile; the complete matrix is never selected implicitly.

| Profile | Split | Per-dataset cap | Configurations | Seeds |
| --- | --- | ---: | --- | --- |
| `pilot` | dev | 20 | 5 main configurations | 42 |
| `main` | test | 200 | 5 main configurations | 42, 43, 44 |
| `ablation` | dev | 100 | 7 single-factor ablations only | 42 |
| `full` | test | none | complete 13-configuration matrix | 42, 43, 44 |

Run `pilot` first, inspect failures, usage, and provider billing, then run
`main`; run `ablation` only after the main pipeline is stable. `full` preserves
the original complete test matrix for reproducibility and must be requested
explicitly with `--profile full`.

### Staged protocol

Paid runs are deliberately staged. The CLI defaults to the recommended
`pilot` profile; the complete matrix is never selected implicitly.

| Profile | Split | Per-dataset cap | Configurations | Seeds |
| --- | --- | ---: | --- | --- |
| `pilot` | dev | 20 | 5 main configurations | 42 |
| `main` | test | 200 | 5 main configurations | 42, 43, 44 |
| `ablation` | dev | 100 | 7 single-factor ablations only | 42 |
| `full` | test | none | complete 13-configuration matrix | 42, 43, 44 |

Run `pilot` first, inspect failures, usage, and provider billing, then run
`main`; run `ablation` only after the main pipeline is stable. `full` preserves
the original complete test matrix for reproducibility and must be requested
explicitly with `--profile full`.

Validate all inputs and estimate request volume without constructing an API
client:

```bash
python tools/run_task16_experiments.py --dry-run
```

The dry-run JSON reports the resolved profile, every dataset/configuration
combination, aggregates by dataset and by configuration, and a cost-warning
level. Request estimates exclude retries and are not token or currency
estimates.

Real execution requires an explicit flag:

```bash
python tools/run_task16_experiments.py --execute \
  --profile main --concurrency 4
```

Profile values can be overridden explicitly with `--split`, `--experiments`,
`--max-examples`, and `--seeds`. For example, a five-question smoke check is:

```bash
python tools/run_task16_experiments.py --dry-run \
  --profile pilot --max-examples 5 --experiments original joint
```

The YAML `generation.backend_configs.openai.api_key` takes precedence over
`LLM_API_KEY`. Results are written per split/dataset/configuration/seed as
`results.jsonl`, `checkpoint.json`, and `summary.json`. Restarting the same
command reuses records with matching experiment and sample identities.

## Experiment Table Template

Status for every row below is **NOT RUN**. Metric cells are intentionally blank;
dry-run estimates must never be inserted as measured results.

| Method | Family | EM | 95% CI | Tokens | Cost | Latency | Failures |
|---|---|---:|---:|---:|---:|---:|---:|
| Original | Main |  |  |  |  |  |  |
| OATH-only | Main |  |  |  |  |  |  |
| CAPE-only | Main |  |  |  |  |  |  |
| Joint | Main |  |  |  |  |  |  |
| no-refute | OATH ablation |  |  |  |  |  |  |
| no-exception | OATH ablation |  |  |  |  |  |  |
| no-temporal | OATH ablation |  |  |  |  |  |  |
| no-graph-expansion | OATH ablation |  |  |  |  |  |  |
| no-permutation | CAPE ablation |  |  |  |  |  |  |
| single-verifier | CAPE ablation |  |  |  |  |  |  |
| fixed-budget | Compute ablation |  |  |  |  |  |  |
| Self-Consistency (equal-token) | Control |  |  |  |  |  |  |

## Limitations

- `just-laws` is a community-maintained MIT-licensed transcription, not an
  official promulgation service. The pinned source provides reproducibility,
  not legal authenticity; production use must verify official sources.
- Laws without a reliable exact commencement date are deliberately absent.
  Coverage is therefore lower than the upstream repository.
- Exact-title `REFERS_TO` links are conservative and do not resolve aliases,
  article-level references, implicit references, amendment semantics, or repeal
  causality.
- The current dense lane is optional; the reproducible default remains
  CPU-compatible lexical retrieval.
- API providers may be nondeterministic even with fixed seeds. The runner
  records model identity, usage, traces, failures, and three repetitions but
  cannot guarantee byte-identical model responses.
- The fixed token ceiling is a configured maximum and actual provider-reported
  tokens are retained. Cross-provider tokenizers and hidden provider prompts can
  differ.
- No accuracy, significance, confidence interval, Pareto, or error-analysis
  claim exists until paid runs finish. Dry-run output is validation and a call
  estimate only.

## Sources

- just-laws repository and license:
  <https://github.com/imca0/just-laws/tree/b475e4e00e1640a927faf79c3ccae663c10c3bbd>
- Lewis et al., Retrieval-Augmented Generation (2020):
  <https://arxiv.org/abs/2005.11401>
- Wang et al., Self-Consistency Improves Chain of Thought Reasoning (2022):
  <https://arxiv.org/abs/2203.11171>
- McNemar, Note on the Sampling Error of the Difference between Correlated
  Proportions or Percentages (1947), *Psychometrika* 12:153-157.
