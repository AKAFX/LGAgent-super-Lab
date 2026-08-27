"""
多智能体法律评测（优化版）

项目方法与创新点（与现有实现严格一致）：
1) 角色分工明确的三智能体协作：
   - 律师A（Issue Spotter）：把题干+选项转为结构化要素 JSON，不允许直接给答案倾向。
   - 法官（Process Controller）：输出流程控制 JSON（是否检索、证据需求、反例方向）。
   - 律师B（Decision Maker）：两阶段提交（B0 盲答 + B1 核验）并输出最终选项。
2) 可选多轮澄清机制：
   - 律师B与法官可进行有限轮次澄清，减少 JSON 解析失败或证据不足导致的误判。
3) 可选RAG增强：
   - 仅当法官判定 need_retrieval=True 且启用 USE_RAG 时，才触发检索，降低无效检索开销。

本脚本相对旧版的工程优化：
- 全部路径基于项目根目录（避免写到错误磁盘路径）
- 支持命令行参数（模型/数据集/并发/开关）与默认值并存
- 定期 checkpoint 落盘（防止长跑中断后结果全丢）
- 线程异常样本保底写入，确保每题都有记录
- 增加错误统计（API_ERROR/TASK_ERROR/空预测）

用法示例（项目根目录）：
  python tools/legal_multi_agent_eval_optimized.py ^
    --dataset data/Ability_merged_500.jsonl ^
    --eval-model llama-3-8b-instruct ^
    --enable-lawyer-a true ^
    --enable-judge true ^
    --enable-dialogue true ^
    --use-rag false ^
    --concurrency 8
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List

# Ensure project root and src are importable.
ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
for _p in (ROOT_DIR, SRC_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from tools.key_pool import load_key_pool
from tools.legal_multi_agent_prompt_demo import (  # type: ignore
    PARAM_PATH,
    build_client,
)
from lgagent.protocol import JudgeOutput, StructuredOutputError
from lgagent.config import load_lgagent_config
from lgagent.evaluation import (
    CheckpointError,
    ExperimentCheckpoint,
    ExperimentMetadata,
    UsageRecord,
    build_result_record,
    calculate_usage_cost,
    evaluate_result_records,
    make_sample_id,
    sha256_file,
    stable_hash,
)
from lgagent.orchestrator import (
    resolve_legacy_entrypoint_settings,
    resolve_pipeline_route,
)
from lgagent.model import OpenAIChatModel
from lgagent.runner import LGAgentPlusRunner
from lgagent.serialization import to_jsonable


# ---- 默认配置（可被命令行覆盖）----
DEFAULT_DATASET = ROOT_DIR / "data" / "dimension_jsonl" / "4_LegalEthics_sample_200.jsonl"
DEFAULT_OUTPUT = ROOT_DIR / "output" / "legal_multi_agent_eval_optimized.json"
DEFAULT_LOG = ROOT_DIR / "output" / "legal_multi_agent_eval_optimized.log"
DEFAULT_MODEL = "glm-4-air"
DEFAULT_CONCURRENCY = 8
CHECKPOINT_EVERY = 50

# A/J 用 key 池（deepseek）
REASONING_KEYS_POOL = load_key_pool("LGAGENT_REASONING_API_KEYS")

# B 用 key 池（待测模型）
EVAL_KEYS_POOL = load_key_pool("LGAGENT_EVAL_API_KEYS")


class UsageMeter:
    """Collect OpenAI-compatible usage without changing the client contract."""

    def __init__(self, input_price_per_million: float, output_price_per_million: float):
        self.input_price = input_price_per_million
        self.output_price = output_price_per_million
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        self.latency_ms = 0.0
        self._lock = threading.Lock()

    def record(self, response: Any, elapsed_ms: float) -> None:
        usage = getattr(response, "usage", None)
        prompt = int(getattr(usage, "prompt_tokens", 0) or 0)
        completion = int(getattr(usage, "completion_tokens", 0) or 0)
        total = int(getattr(usage, "total_tokens", 0) or 0)
        with self._lock:
            self.calls += 1
            self.prompt_tokens += prompt
            self.completion_tokens += completion
            self.total_tokens += total or prompt + completion
            self.latency_ms += elapsed_ms

    def record_error(self, elapsed_ms: float) -> None:
        with self._lock:
            self.calls += 1
            self.latency_ms += elapsed_ms

    def snapshot(self, wall_latency_ms: float) -> UsageRecord:
        cost = calculate_usage_cost(
            self.prompt_tokens,
            self.completion_tokens,
            input_price_per_million=self.input_price,
            output_price_per_million=self.output_price,
        )
        return UsageRecord(
            calls=self.calls,
            prompt_tokens=self.prompt_tokens,
            completion_tokens=self.completion_tokens,
            latency_ms=wall_latency_ms,
            cost=cost,
            total_tokens=self.total_tokens,
        )


class _MeteredCreate:
    def __init__(self, create: Any, meter: UsageMeter):
        self._create = create
        self._meter = meter

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        started = perf_counter()
        try:
            response = self._create(*args, **kwargs)
        except Exception:
            self._meter.record_error((perf_counter() - started) * 1000)
            raise
        self._meter.record(response, (perf_counter() - started) * 1000)
        return response


class _MeteredCompletions:
    def __init__(self, completions: Any, meter: UsageMeter):
        self._raw = completions
        self.create = _MeteredCreate(completions.create, meter)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._raw, name)


class _MeteredChat:
    def __init__(self, chat: Any, meter: UsageMeter):
        self._raw = chat
        self.completions = _MeteredCompletions(chat.completions, meter)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._raw, name)


class MeteredClient:
    def __init__(self, client: Any, meter: UsageMeter):
        self._raw = client
        self.chat = _MeteredChat(client.chat, meter)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._raw, name)


class TeeLogger:
    def __init__(self, path: Path):
        self.stdout = sys.stdout
        self.stderr = sys.stderr
        self.f = path.open("w", encoding="utf-8")

    def write(self, s: str) -> None:
        self.stdout.write(s)
        self.f.write(s)
        self.f.flush()

    def flush(self) -> None:
        self.stdout.flush()
        self.f.flush()

    def close(self) -> None:
        self.f.close()


def normalize_text(text: str) -> str:
    import re
    import string as _string

    text = (text or "").strip().lower()
    table = str.maketrans({c: " " for c in _string.punctuation})
    text = text.translate(table)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_choice(answer: str) -> str:
    import re

    text = (answer or "").strip()
    if not text:
        return ""
    if text.startswith("[API_ERROR") or text.startswith("[ERROR") or text.startswith("[TASK_ERROR]"):
        return ""
    if re.match(r"^[A-D]$", text):
        return text
    m = re.search(r"[最终答案选项]*[：:]\s*([A-D])\b", text)
    if m:
        return m.group(1).strip()
    m2 = re.search(r"\b([A-D])\b", text.splitlines()[0] if text.splitlines() else text)
    if m2:
        return m2.group(1).strip()
    m3 = re.search(r"\b([A-D])\b", text)
    return m3.group(1).strip() if m3 else ""


def evaluate(gt_all: List[List[str]], pred_all: List[str]) -> Dict[str, float]:
    n = len(pred_all) or 1
    acc = 0.0
    for gts, pred in zip(gt_all, pred_all):
        p = normalize_text(pred)
        if p and any(p == normalize_text(g) for g in gts):
            acc += 1.0
    acc /= n
    # 选择题场景下 EM/F1 与 acc 一致性更强，这里保持简洁可解释
    return {"avg_acc": acc, "avg_em": acc, "avg_f1": acc}


def parse_bool(s: str) -> bool:
    return s.strip().lower() in {"1", "true", "yes", "y", "on"}


def _collect_string_values(value: Any, key: str) -> List[str]:
    found: List[str] = []
    if isinstance(value, dict):
        candidate = value.get(key)
        if isinstance(candidate, str):
            found.append(candidate)
        elif isinstance(candidate, list):
            found.extend(item for item in candidate if isinstance(item, str))
        for nested in value.values():
            found.extend(_collect_string_values(nested, key))
    elif isinstance(value, list):
        for nested in value:
            found.extend(_collect_string_values(nested, key))
    return list(dict.fromkeys(found))


def _metric_report(detail: List[Dict[str, Any]]) -> Dict[str, Any]:
    return evaluate_result_records(detail)


def save_checkpoint(
    checkpoint: ExperimentCheckpoint,
    dataset: Path,
    eval_conf: Dict[str, Any],
    settings: Dict[str, Any],
) -> None:
    detail = list(checkpoint.records)
    metrics = _metric_report(detail)
    err_api = sum(1 for d in detail if str(d.get("lawyerB_answer", "")).startswith("[API_ERROR"))
    err_task = sum(1 for d in detail if str(d.get("lawyerB_answer", "")).startswith("[TASK_ERROR"))
    empty = sum(1 for d in detail if not d.get("lawyerB_pred_for_eval"))
    payload = {
        "dataset": str(dataset),
        "metrics": metrics,
        "model_under_test": {"model": eval_conf["model"], "base_url": eval_conf["base_url"]},
        "settings": settings,
        "stats": {
            "finished_examples": len(detail),
            "api_error_count": err_api,
            "task_error_count": err_task,
            "empty_pred_count": empty,
        },
        "checkpoint_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    checkpoint.save(payload)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, default=str(DEFAULT_DATASET))
    ap.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT))
    ap.add_argument("--log", type=str, default=str(DEFAULT_LOG))
    ap.add_argument("--eval-model", type=str, default=DEFAULT_MODEL)
    ap.add_argument("--eval-base-url", type=str, default="")
    ap.add_argument("--max-examples", type=int, default=-1)
    ap.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    ap.add_argument("--enable-lawyer-a", type=str, default="true")
    ap.add_argument("--enable-judge", type=str, default="true")
    ap.add_argument("--enable-dialogue", type=str, default="true")
    ap.add_argument("--use-rag", type=str, default="false")
    ap.add_argument("--rag-mode", choices=("local", "remote"), default="local")
    ap.add_argument("--rag-fallback", type=str, default="true")
    ap.add_argument("--rag-top-k", type=int, default=3)
    ap.add_argument("--resume", type=str, default="true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--prompt-version", type=str, default="legacy-v1")
    ap.add_argument("--input-price-per-million", type=float, default=0.0)
    ap.add_argument("--output-price-per-million", type=float, default=0.0)
    args = ap.parse_args()

    dataset_path = Path(args.dataset)
    output_path = Path(args.output)
    log_path = Path(args.log)
    max_examples = None if args.max_examples < 0 else args.max_examples

    enable_lawyer_a = parse_bool(args.enable_lawyer_a)
    enable_judge = parse_bool(args.enable_judge)
    enable_dialogue = parse_bool(args.enable_dialogue)
    use_rag = parse_bool(args.use_rag)
    rag_fallback = parse_bool(args.rag_fallback)
    resume = parse_bool(args.resume)
    concurrency = max(1, int(args.concurrency))
    if args.input_price_per_million < 0 or args.output_price_per_million < 0:
        raise ValueError("token prices cannot be negative")

    app_conf = load_lgagent_config(PARAM_PATH)
    gen_conf = app_conf.generation.as_legacy_dict()
    pipeline_route = resolve_pipeline_route(app_conf)

    entrypoint = resolve_legacy_entrypoint_settings(
        pipeline_route,
        enable_lawyer_a=enable_lawyer_a,
        enable_judge=enable_judge,
        enable_dialogue=enable_dialogue,
        use_rag=use_rag,
    )
    enable_lawyer_a = entrypoint.enable_lawyer_a
    enable_judge = entrypoint.enable_judge
    enable_dialogue = entrypoint.enable_dialogue
    use_rag = entrypoint.use_rag

    log_path.parent.mkdir(parents=True, exist_ok=True)
    tee = TeeLogger(log_path)
    origin_out, origin_err = sys.stdout, sys.stderr
    sys.stdout = tee
    sys.stderr = tee

    try:
        eval_conf: Dict[str, Any] = {
            "model": args.eval_model,
            "base_url": args.eval_base_url or gen_conf["base_url"],
            "temperature": gen_conf["temperature"],
            "top_p": gen_conf["top_p"],
            "max_tokens": gen_conf["max_tokens"],
        }
        base_url_reasoning = gen_conf["base_url"]

        settings = {
            "ENABLE_LAWYER_A": enable_lawyer_a,
            "ENABLE_JUDGE": enable_judge,
            "ENABLE_DIALOGUE": enable_dialogue,
            "USE_RAG": use_rag,
            "RAG_MODE": args.rag_mode,
            "RAG_FALLBACK": rag_fallback,
            "CONCURRENCY": concurrency,
            "PIPELINE_ROUTE": pipeline_route.name,
            "OATH_RAG": pipeline_route.oath_rag_enabled,
            "CAPE_V": pipeline_route.cape_v_enabled,
            "SEED": args.seed,
            "MAX_EXAMPLES": max_examples,
        }

        print(f"日志文件：{log_path}")
        print(f"开始时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"数据集：{dataset_path}")
        print(f"模型（律师B）：{eval_conf['model']}")
        print(f"设置：{settings}")
        print("=" * 80)

        print("检索由 LGAgentPlusRunner 按配置管理")

        # 读取数据
        questions: List[str] = []
        golden_all: List[List[str]] = []
        dataset_rows: List[Dict[str, Any]] = []
        with dataset_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                q = obj.get("question", "")
                gts = obj.get("golden_answers") or []
                if not q or not gts:
                    continue
                questions.append(q)
                golden_all.append(gts)
                dataset_rows.append(obj)
                if max_examples is not None and len(questions) >= max_examples:
                    break

        total = len(questions)
        print(f"读取样本数：{total}")
        if total == 0:
            raise RuntimeError("数据集为空或字段不匹配（需要 question + golden_answers）。")

        try:
            software_version = importlib.metadata.version("lgagent")
        except importlib.metadata.PackageNotFoundError:
            software_version = "0.1.0"
        metadata = ExperimentMetadata(
            model_id=eval_conf["model"],
            endpoint_class="openai-compatible",
            prompt_version=args.prompt_version,
            config_hash=stable_hash(
                {
                    "settings": settings,
                    "sampling": {
                        "temperature": eval_conf["temperature"],
                        "top_p": eval_conf["top_p"],
                        "max_tokens": eval_conf["max_tokens"],
                    },
                }
            ),
            dataset_hash=sha256_file(dataset_path),
            random_seed=args.seed,
            software_version=software_version,
        )
        checkpoint = ExperimentCheckpoint(output_path, metadata)
        if resume and output_path.exists():
            try:
                resumed = checkpoint.load()
            except CheckpointError as exc:
                raise RuntimeError(
                    f"无法续跑现有输出 {output_path}：{exc}；"
                    "请更换 --output 或设置 --resume false"
                ) from exc
            print(
                f"续跑实验 {metadata.experiment_id}："
                f"已复用 {len(resumed)} 个样本"
            )
        elif output_path.exists():
            print(f"禁用续跑，将覆盖现有输出：{output_path}")

        # key 池准备
        reasoning_keys = (
            [gen_conf["api_key"]] * concurrency
            if gen_conf.get("api_key_source") == "yaml"
            else (
                REASONING_KEYS_POOL[:concurrency]
                if len(REASONING_KEYS_POOL) >= concurrency
                else [gen_conf.get("api_key", "")] * concurrency
            )
        )
        eval_keys = (
            [gen_conf["api_key"]] * concurrency
            if gen_conf.get("api_key_source") == "yaml"
            else (
                EVAL_KEYS_POOL[:concurrency]
                if len(EVAL_KEYS_POOL) >= concurrency
                else [gen_conf.get("api_key", "")] * concurrency
            )
        )
        if not any(eval_keys):
            raise RuntimeError("没有可用的律师B API key（EVAL_KEYS_POOL或parameter中的api_key为空）。")

        progress_lock = threading.Lock()
        completed = [0]

        def process_one(
            idx: int,
            q: str,
            gts: List[str],
            row: Dict[str, Any],
            thread_id: int,
        ) -> Dict[str, Any]:
            started = perf_counter()
            meter = UsageMeter(
                args.input_price_per_million,
                args.output_price_per_million,
            )
            rk = reasoning_keys[thread_id % len(reasoning_keys)]
            reason_client = MeteredClient(
                build_client(base_url_reasoning, rk), meter
            )
            ek = eval_keys[thread_id % len(eval_keys)]
            eval_client = MeteredClient(
                build_client(eval_conf["base_url"], ek), meter
            )

            parsed = ""
            judge_res = ""
            judge_need_retrieval = False
            retrieved_docs: List[str] = []
            lawyer_a_history: List[Dict[str, str]] = []
            judge_history: List[Dict[str, str]] = []

            try:
                verifier_settings = app_conf.lgagent_plus.cape_v.verifier_model
                verifier_client = eval_client
                if verifier_settings is not None:
                    verifier_client = MeteredClient(
                        build_client(
                            verifier_settings.base_url,
                            verifier_settings.api_key or ek,
                        ),
                        meter,
                    )
                result = LGAgentPlusRunner(
                    app_conf,
                    reasoning_model=OpenAIChatModel(reason_client),
                    evaluation_model=OpenAIChatModel(eval_client),
                    evaluation_config={
                        **eval_conf,
                        "api_key": ek,
                        "api_key_source": "yaml"
                        if gen_conf.get("api_key_source") == "yaml"
                        else "key_pool",
                    },
                    verifier_model=OpenAIChatModel(verifier_client),
                    project_root=ROOT_DIR,
                    max_dialogue_rounds=2 if enable_dialogue else 0,
                ).run(q)
                full_answer = result.final_answer
                internal_json = result.diagnostics
                parsed = result.lawyer_a_output
                judge_res = result.judge_output
                judge_need_retrieval = JudgeOutput.from_text(
                    judge_res
                ).need_retrieval
            except StructuredOutputError as e:
                full_answer = f"[TASK_ERROR] {e}"
                internal_json = {
                    "error_type": "structured_output",
                    "agent": e.agent,
                    "attempts": e.attempts,
                    "message": str(e),
                    "raw_output": e.raw_output,
                }
            except Exception as e:
                full_answer = f"[TASK_ERROR] {e}"
                internal_json = {"error_type": "task", "message": str(e)}

            pred = extract_choice(str(full_answer))
            raw = internal_json if isinstance(internal_json, dict) else {}
            candidates = raw.get("candidates")
            if not isinstance(candidates, list):
                candidates = [
                    {
                        "candidate_id": "legacy-b1",
                        "answer": pred,
                        "irac": raw.get("irac", {}),
                        "option_verification": raw.get(
                            "option_verification", raw.get("verification", {})
                        ),
                    }
                ]
            evidence_matrix = raw.get("evidence_matrix")
            if not isinstance(evidence_matrix, dict):
                evidence_matrix = {
                    "unscoped": {
                        "documents": retrieved_docs,
                        "coverage": 0.0,
                        "conflict": 0.0,
                    }
                }
            verification_reports = raw.get("verification_reports", [])
            if not isinstance(verification_reports, list):
                verification_reports = [verification_reports]
            if not verification_reports and isinstance(raw.get("verification"), dict):
                verification_reports = [
                    {
                        "candidate_id": "legacy-b1",
                        "dimensions": raw["verification"],
                    }
                ]
            risk_routing = raw.get("risk_routing", {})
            if not risk_routing and raw.get("route"):
                risk_routing = {"route": raw.get("route")}
            if not risk_routing:
                risk_routing = {
                    "route": pipeline_route.name,
                    "judge_need_retrieval": judge_need_retrieval,
                }
            confidence = raw.get("confidence", raw.get("b0_confidence"))
            risk_score = raw.get("risk_score")
            usage = meter.snapshot((perf_counter() - started) * 1000)
            sample_id = make_sample_id(idx, q)
            relevant_ids = (
                row.get("relevant_evidence_ids")
                or row.get("gold_evidence_ids")
                or []
            )
            cited_ids = raw.get("cited_evidence_ids") or _collect_string_values(
                candidates, "evidence_ids"
            )
            retrieved_ids = raw.get(
                "retrieved_evidence_ids"
            ) or _collect_string_values(evidence_matrix, "evidence_id")

            with progress_lock:
                completed[0] += 1
                cur = completed[0]
                pct = cur / total * 100.0
                print(f"\rEvaluating {cur}/{total} ({pct:5.1f}%)", end="", flush=True)

            return build_result_record(
                sample_id=sample_id,
                prediction=pred,
                golden_answers=gts,
                candidates=candidates,
                evidence_matrix=evidence_matrix,
                verification_reports=verification_reports,
                risk_routing=risk_routing,
                usage=usage,
                reproduction={
                    "experiment_id": metadata.experiment_id,
                    "sample_index": idx,
                    "random_seed": args.seed,
                },
                extra={
                    "idx": idx,
                    "question": q,
                    "domain": row.get("domain", row.get("category", "")),
                    "lawyerA_parsed": parsed,
                    "lawyerA_history": lawyer_a_history,
                    "judge_reasoning": judge_res,
                    "judge_history": judge_history,
                    "judge_need_retrieval": judge_need_retrieval,
                    "retrieved_docs": retrieved_docs,
                    "lawyerB_answer": str(full_answer),
                    "lawyerB_internal_json": internal_json,
                    "lawyerB_pred_for_eval": pred,
                    "confidence": confidence,
                    "risk_score": risk_score,
                    "retrieved_evidence_ids": retrieved_ids,
                    "relevant_evidence_ids": relevant_ids,
                    "cited_evidence_ids": cited_ids,
                    "permutation_predictions": raw.get(
                        "permutation_predictions", []
                    ),
                    "counterfactual_observations": raw.get(
                        "counterfactual_observations", []
                    ),
                },
            )

        expected_indices = {
            make_sample_id(idx, question): idx
            for idx, question in enumerate(questions)
        }
        unexpected_ids = checkpoint.completed_ids - set(expected_indices)
        if unexpected_ids:
            raise RuntimeError(
                "checkpoint 包含不属于当前数据切片的 sample_id："
                f"{sorted(unexpected_ids)[:3]}"
            )
        for item in checkpoint.records:
            if item.get("idx") != expected_indices[item["sample_id"]]:
                raise RuntimeError(
                    f"checkpoint 样本索引不匹配：{item['sample_id']}"
                )
        results_dict: Dict[int, Dict[str, Any]] = {
            int(item["idx"]): item for item in checkpoint.records
        }
        pending_indices = [
            idx
            for idx, question in enumerate(questions)
            if make_sample_id(idx, question) not in checkpoint.completed_ids
        ]
        completed[0] = total - len(pending_indices)
        with ThreadPoolExecutor(max_workers=concurrency) as ex:
            futures = {
                ex.submit(
                    process_one,
                    idx,
                    questions[idx],
                    golden_all[idx],
                    dataset_rows[idx],
                    idx % concurrency,
                ): idx
                for idx in pending_indices
            }
            for fut in as_completed(futures):
                idx = futures[fut]
                try:
                    results_dict[idx] = fut.result()
                except Exception as e:
                    failed = {
                        "idx": idx,
                        "sample_id": make_sample_id(idx, questions[idx]),
                        "question": questions[idx],
                        "golden_answers": golden_all[idx],
                        "lawyerA_parsed": "",
                        "lawyerA_history": [],
                        "judge_reasoning": "",
                        "judge_history": [],
                        "judge_need_retrieval": False,
                        "retrieved_docs": [],
                        "lawyerB_answer": f"[TASK_ERROR] {e}",
                        "lawyerB_internal_json": {"error": str(e)},
                        "lawyerB_pred_for_eval": "",
                        "candidates": [],
                        "evidence_matrix": {},
                        "verification_reports": [],
                        "risk_routing": {},
                        "usage": to_jsonable(UsageRecord()),
                        "reproduction": {
                            "experiment_id": metadata.experiment_id,
                            "sample_index": idx,
                            "random_seed": args.seed,
                        },
                    }
                    results_dict[idx] = failed

                # checkpoint：每收集一定数量就落盘
                checkpoint.add(results_dict[idx])
                if len(checkpoint.records) % CHECKPOINT_EVERY == 0:
                    save_checkpoint(
                        checkpoint, dataset_path, eval_conf, settings
                    )

        print()  # newline after progress

        detail = [results_dict[i] for i in range(total)]
        for item in detail:
            checkpoint.add(item)
        save_checkpoint(checkpoint, dataset_path, eval_conf, settings)
        metrics = _metric_report(detail)
        print("评估结果：", metrics)
        print(f"详细结果已保存到：{output_path}")
        print(f"结束时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)
    finally:
        sys.stdout = origin_out
        sys.stderr = origin_err
        tee.close()


if __name__ == "__main__":
    main()
