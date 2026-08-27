"""
多智能体组件消融自动化评测脚本（独立版本）

目标：
1) 自动评测单组件、两两组件、全组件、无组件（direct）配置
2) 自动在多个数据集上批量运行
3) 输出可直接用于论文表格的汇总结果（json/csv/md）

默认数据集：
- data/sample_legal_200.jsonl
- data/Ability_sample_200.jsonl
- data/generated_questions1234_sorted_100.jsonl
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
for _p in (ROOT_DIR, SRC_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from tools.key_pool import load_key_pool
from tools.legal_multi_agent_prompt_demo import (  # type: ignore
    PARAM_PATH,
    build_client,
    load_generation_config,
    run_judge,
    run_lawyer_answer,
    run_lawyer_parser,
)
from lgagent.protocol import JudgeOutput, StructuredOutputError
from lgagent.rag import RAGRetriever, RetrieverSettings
from lgagent.ablation import build_task14_matrix, export_task14_matrix
from lgagent.config import load_lgagent_config


REASONING_KEYS_POOL = load_key_pool("LGAGENT_REASONING_API_KEYS")
EVAL_KEYS_POOL = load_key_pool("LGAGENT_EVAL_API_KEYS")

DEFAULT_DATASETS = [
    ROOT_DIR / "data" / "sample_legal_200.jsonl",
    ROOT_DIR / "data" / "Ability_sample_200.jsonl",
    ROOT_DIR / "data" / "generated_questions1234_sorted_100.jsonl",
]
DEFAULT_OUTPUT_DIR = ROOT_DIR / "output" / "ablation_auto"
DEFAULT_MODEL = "deepseek-v3.2"
DEFAULT_CONCURRENCY = 8
CHECKPOINT_EVERY = 50


@dataclass(frozen=True)
class AblationConfig:
    name: str
    enable_lawyer_a: bool
    enable_judge: bool
    enable_dialogue: bool
    use_rag: bool = False


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
    return {"avg_acc": acc, "avg_em": acc, "avg_f1": acc}


def load_dataset(dataset_path: Path, max_examples: int | None) -> Tuple[List[str], List[List[str]]]:
    questions: List[str] = []
    golden_all: List[List[str]] = []
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
            if max_examples is not None and len(questions) >= max_examples:
                break
    if not questions:
        raise RuntimeError(f"数据集为空或字段不匹配: {dataset_path}")
    return questions, golden_all


def build_ablation_configs() -> List[AblationConfig]:
    # 3个核心组件：A(结构解析), J(流程控制), D(B与J多轮对话)
    # 这里穷举：none + 单组件 + 两两组件 + full
    return [
        AblationConfig("ablation-none", False, False, False, False),
        AblationConfig("ablation-A-only", True, False, False, False),
        AblationConfig("ablation-J-only", False, True, False, False),
        AblationConfig("ablation-D-only", False, True, True, False),
        AblationConfig("ablation-AJ", True, True, False, False),
        AblationConfig("ablation-AD", True, False, True, False),
        AblationConfig("ablation-JD", False, True, True, False),
        AblationConfig("ablation-full", True, True, True, False),
    ]


def save_run_output(
    output_path: Path,
    dataset: Path,
    config: AblationConfig,
    eval_conf: Dict[str, Any],
    detail: List[Dict[str, Any]],
    golden_all: List[List[str]],
    started_at: str,
) -> Dict[str, Any]:
    preds = [d.get("lawyerB_pred_for_eval", "") for d in detail]
    metrics = evaluate(golden_all, preds)
    err_api = sum(1 for d in detail if str(d.get("lawyerB_answer", "")).startswith("[API_ERROR"))
    err_task = sum(1 for d in detail if str(d.get("lawyerB_answer", "")).startswith("[TASK_ERROR"))
    empty = sum(1 for d in detail if not d.get("lawyerB_pred_for_eval"))
    payload = {
        "dataset": str(dataset),
        "ablation": config.__dict__,
        "metrics": metrics,
        "model_under_test": {"model": eval_conf["model"], "base_url": eval_conf["base_url"]},
        "stats": {
            "finished_examples": len(detail),
            "api_error_count": err_api,
            "task_error_count": err_task,
            "empty_pred_count": empty,
        },
        "examples": detail,
        "started_at": started_at,
        "finished_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return payload


def run_single_experiment(
    dataset_path: Path,
    config: AblationConfig,
    output_dir: Path,
    eval_model: str,
    eval_base_url: str,
    concurrency: int,
    max_examples: int | None,
) -> Dict[str, Any]:
    gen_conf = load_generation_config(PARAM_PATH)
    eval_conf: Dict[str, Any] = {
        "model": eval_model,
        "base_url": eval_base_url or gen_conf["base_url"],
        "temperature": gen_conf["temperature"],
        "top_p": gen_conf["top_p"],
        "max_tokens": gen_conf["max_tokens"],
    }
    base_url_reasoning = gen_conf["base_url"]

    questions, golden_all = load_dataset(dataset_path, max_examples)
    total = len(questions)
    started_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    retriever: RAGRetriever | None = None
    if config.use_rag and config.enable_judge:
        try:
            retriever = RAGRetriever(
                RetrieverSettings.load(
                    ROOT_DIR / "servers",
                    top_k=3,
                    project_root=ROOT_DIR,
                )
            )
            retriever.initialize()
        except Exception:
            retriever = None

    use_multi_agent = config.enable_lawyer_a or config.enable_judge
    if use_multi_agent:
        reasoning_keys = (
            [gen_conf["api_key"]] * concurrency
            if gen_conf.get("api_key_source") == "yaml"
            else (
                REASONING_KEYS_POOL[:concurrency]
                if len(REASONING_KEYS_POOL) >= concurrency
                else [gen_conf.get("api_key", "")] * concurrency
            )
        )
    else:
        reasoning_keys = []
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
        raise RuntimeError("没有可用的律师B API key。")

    dataset_tag = dataset_path.stem
    run_output = output_dir / dataset_tag / f"{config.name}.json"
    checkpoint_path = output_dir / dataset_tag / f"{config.name}.checkpoint.json"

    progress_lock = threading.Lock()
    completed = [0]
    results_dict: Dict[int, Dict[str, Any]] = {}

    def process_one(idx: int, q: str, gts: List[str], thread_id: int) -> Dict[str, Any]:
        reason_client = None
        if use_multi_agent:
            rk = reasoning_keys[thread_id % len(reasoning_keys)]
            reason_client = build_client(base_url_reasoning, rk)
        ek = eval_keys[thread_id % len(eval_keys)]
        eval_client = build_client(eval_conf["base_url"], ek)

        parsed = ""
        judge_res = ""
        judge_need_retrieval = False
        retrieved_docs: List[str] = []
        lawyer_a_history: List[Dict[str, str]] = []
        judge_history: List[Dict[str, str]] = []

        try:
            if config.enable_lawyer_a and reason_client is not None:
                parsed, lawyer_a_history = run_lawyer_parser(reason_client, gen_conf, q)

            if config.enable_judge and reason_client is not None:
                judge_res, judge_history, should_continue = run_judge(
                    reason_client, gen_conf, q, parsed, []
                )
                if should_continue and config.enable_lawyer_a:
                    parsed, lawyer_a_history = run_lawyer_parser(reason_client, gen_conf, q, lawyer_a_history)
                    judge_res, judge_history, _ = run_judge(
                        reason_client,
                        gen_conf,
                        q,
                        parsed,
                        judge_history,
                        need_clarification=True,
                        clarification_question="请补充关键缺口与证据需求。",
                    )
                judge_need_retrieval = JudgeOutput.from_text(judge_res).need_retrieval

            if config.enable_judge and retriever is not None and judge_need_retrieval:
                try:
                    retrieved_docs = retriever.search(q)
                except Exception:
                    retrieved_docs = []

            # 只有启用 Judge 才允许启用 dialogue，避免无效配置干扰
            effective_dialogue = config.enable_dialogue and config.enable_judge
            full_answer, internal_json = run_lawyer_answer(
                eval_client,
                eval_conf,
                q,
                parsed,
                judge_res,
                retrieved_docs,
                use_multi_agent=use_multi_agent,
                reasoning_client=reason_client if use_multi_agent else None,
                gen_conf=gen_conf if use_multi_agent else None,
                enable_dialogue=effective_dialogue,
            )
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
        with progress_lock:
            completed[0] += 1
            cur = completed[0]
            pct = cur / total * 100.0
            print(
                f"\r[{dataset_path.name} | {config.name}] {cur}/{total} ({pct:5.1f}%)",
                end="",
                flush=True,
            )

        return {
            "idx": idx,
            "question": q,
            "golden_answers": gts,
            "lawyerA_parsed": parsed,
            "lawyerA_history": lawyer_a_history,
            "judge_reasoning": judge_res,
            "judge_history": judge_history,
            "judge_need_retrieval": judge_need_retrieval,
            "retrieved_docs": retrieved_docs,
            "lawyerB_answer": str(full_answer),
            "lawyerB_internal_json": internal_json,
            "lawyerB_pred_for_eval": pred,
        }

    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futures = {
            ex.submit(process_one, idx, q, gts, idx % concurrency): idx
            for idx, (q, gts) in enumerate(zip(questions, golden_all))
        }
        for fut in as_completed(futures):
            idx = futures[fut]
            try:
                results_dict[idx] = fut.result()
            except Exception as e:
                results_dict[idx] = {
                    "idx": idx,
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
                }

            if len(results_dict) % CHECKPOINT_EVERY == 0:
                detail_now = [results_dict[i] for i in sorted(results_dict.keys())]
                save_run_output(
                    checkpoint_path,
                    dataset_path,
                    config,
                    eval_conf,
                    detail_now,
                    golden_all[: len(detail_now)],
                    started_at,
                )

    print()
    detail = [results_dict[i] for i in range(total)]
    payload = save_run_output(
        run_output,
        dataset_path,
        config,
        eval_conf,
        detail,
        golden_all,
        started_at,
    )
    if retriever is not None:
        retriever.close()
    return {
        "dataset": dataset_path.name,
        "dataset_path": str(dataset_path),
        "ablation": config.name,
        "enable_lawyer_a": config.enable_lawyer_a,
        "enable_judge": config.enable_judge,
        "enable_dialogue": config.enable_dialogue,
        "use_rag": config.use_rag,
        "avg_em": payload["metrics"]["avg_em"],
        "avg_acc": payload["metrics"]["avg_acc"],
        "avg_f1": payload["metrics"]["avg_f1"],
        "finished_examples": payload["stats"]["finished_examples"],
        "api_error_count": payload["stats"]["api_error_count"],
        "task_error_count": payload["stats"]["task_error_count"],
        "empty_pred_count": payload["stats"]["empty_pred_count"],
        "output_file": str(run_output),
    }


def write_summary_files(output_dir: Path, rows: List[Dict[str, Any]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_json = output_dir / "ablation_summary.json"
    summary_csv = output_dir / "ablation_summary.csv"
    summary_md = output_dir / "ablation_summary.md"

    with summary_json.open("w", encoding="utf-8") as f:
        json.dump({"generated_at": datetime.now().isoformat(), "rows": rows}, f, ensure_ascii=False, indent=2)

    csv_fields = [
        "dataset",
        "ablation",
        "enable_lawyer_a",
        "enable_judge",
        "enable_dialogue",
        "use_rag",
        "avg_em",
        "avg_acc",
        "avg_f1",
        "finished_examples",
        "api_error_count",
        "task_error_count",
        "empty_pred_count",
        "output_file",
    ]
    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in csv_fields})

    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        grouped.setdefault(r["dataset"], []).append(r)

    lines: List[str] = []
    lines.append("# Ablation Results")
    lines.append("")
    lines.append(f"- Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    for dataset, drows in grouped.items():
        drows_sorted = sorted(drows, key=lambda x: x["ablation"])
        lines.append(f"## {dataset}")
        lines.append("")
        lines.append("| Ablation | A | J | D | EM | API_ERR | TASK_ERR | EMPTY |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        for r in drows_sorted:
            lines.append(
                "| {ablation} | {a} | {j} | {d} | {em:.4f} | {api} | {task} | {empty} |".format(
                    ablation=r["ablation"],
                    a=int(bool(r["enable_lawyer_a"])),
                    j=int(bool(r["enable_judge"])),
                    d=int(bool(r["enable_dialogue"])),
                    em=float(r["avg_em"]),
                    api=r["api_error_count"],
                    task=r["task_error_count"],
                    empty=r["empty_pred_count"],
                )
            )
        lines.append("")

    with summary_md.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="多智能体组件消融自动化评测")
    ap.add_argument(
        "--datasets",
        nargs="*",
        default=[str(p) for p in DEFAULT_DATASETS],
        help="数据集路径列表（jsonl）",
    )
    ap.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    ap.add_argument("--eval-model", type=str, default=DEFAULT_MODEL)
    ap.add_argument("--eval-base-url", type=str, default="")
    ap.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    ap.add_argument("--max-examples", type=int, default=-1, help="每个数据集最多样本数，-1表示全量")
    ap.add_argument(
        "--skip-configs",
        nargs="*",
        default=[],
        help="可选：跳过的消融配置名称，如 ablation-none ablation-full",
    )
    ap.add_argument(
        "--matrix-only",
        action="store_true",
        help="仅离线校验并导出 Task 14 实验矩阵，不初始化客户端或调用 API",
    )
    ap.add_argument(
        "--matrix-config",
        type=str,
        default=str(PARAM_PATH),
        help="Task 14 实验矩阵使用的 YAML 配置路径",
    )
    return ap.parse_args(argv)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    if args.matrix_only:
        matrix_config = load_lgagent_config(args.matrix_config, environ={})
        matrix = build_task14_matrix(matrix_config)
        paths = export_task14_matrix(matrix, output_dir)
        print(f"Task 14 实验矩阵校验通过：{len(matrix)} 个配置")
        print(f"JSON：{paths['json']}")
        print(f"CSV：{paths['csv']}")
        print(f"Markdown：{paths['markdown']}")
        return

    log_path = output_dir / "ablation_run.log"
    output_dir.mkdir(parents=True, exist_ok=True)

    tee = TeeLogger(log_path)
    origin_out, origin_err = sys.stdout, sys.stderr
    sys.stdout = tee
    sys.stderr = tee
    try:
        datasets = [Path(x) for x in args.datasets]
        for ds in datasets:
            if not ds.is_absolute():
                ds = ROOT_DIR / ds
            if not ds.exists():
                raise FileNotFoundError(f"数据集不存在: {ds}")
        concurrency = max(1, int(args.concurrency))
        max_examples = None if int(args.max_examples) < 0 else int(args.max_examples)

        all_configs = build_ablation_configs()
        skip_set = set(args.skip_configs or [])
        configs = [c for c in all_configs if c.name not in skip_set]
        if not configs:
            raise RuntimeError("没有可运行的消融配置（可能被全部 skip 了）。")

        print("=" * 100)
        print("多智能体组件消融自动评测开始")
        print(f"开始时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"数据集数量：{len(datasets)}")
        print(f"配置数量：{len(configs)}")
        print(f"模型（律师B）：{args.eval_model}")
        print(f"并发数：{concurrency}")
        print("配置列表：", [c.name for c in configs])
        print("=" * 100)

        summary_rows: List[Dict[str, Any]] = []
        for ds in datasets:
            if not ds.is_absolute():
                ds = ROOT_DIR / ds
            print(f"\n{'#' * 30} DATASET: {ds.name} {'#' * 30}")
            for cfg in configs:
                print(f"\n---- Running {cfg.name} on {ds.name} ----")
                row = run_single_experiment(
                    dataset_path=ds,
                    config=cfg,
                    output_dir=output_dir,
                    eval_model=args.eval_model,
                    eval_base_url=args.eval_base_url,
                    concurrency=concurrency,
                    max_examples=max_examples,
                )
                summary_rows.append(row)
                print(
                    f"[DONE] {ds.name} | {cfg.name} | EM={row['avg_em']:.4f} | "
                    f"api_err={row['api_error_count']} | task_err={row['task_error_count']}"
                )

        write_summary_files(output_dir, summary_rows)
        print("\n" + "=" * 100)
        print("全部评测完成")
        print(f"结束时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"输出目录：{output_dir}")
        print(f"汇总JSON：{output_dir / 'ablation_summary.json'}")
        print(f"汇总CSV：{output_dir / 'ablation_summary.csv'}")
        print(f"汇总Markdown：{output_dir / 'ablation_summary.md'}")
        print("=" * 100)
    finally:
        sys.stdout = origin_out
        sys.stderr = origin_err
        tee.close()


if __name__ == "__main__":
    main()
