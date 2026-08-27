"""
DeepSeek-V3 Baseline 评估脚本（不使用多智能体）

功能：
- 直接让 deepseek-v3 回答问题，不使用律师A、法官等多智能体
- 批量评估 data/dimension_jsonl 目录下的数据集（从第二个到最后一个）
- 每个数据集生成一个评估结果 JSON
- 最后生成一个汇总结果 JSON

用法（在 UltraRAG 根目录执行）：
    python tools/baseline_eval.py
"""

import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import yaml
from dotenv import load_dotenv
from openai import OpenAI

# ==== 配置参数 ====
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from tools.key_pool import load_key_pool

DIMENSION_JSONL_DIR = ROOT_DIR / "data" / "dimension_jsonl" 
OUTPUT_DIR = ROOT_DIR / "output"
PARAM_PATH = ROOT_DIR / "examples" / "parameter" / "legal2_rag_parameter.yaml"
load_dotenv(ROOT_DIR / ".env")

# 模型配置
EVAL_MODEL = "glm-4-air"
EVAL_BASE_URL = None  # 若为 None，沿用 legal2_rag_parameter.yaml 中的 base_url

# API Key 池（用于并发）
API_KEYS_POOL = load_key_pool("LGAGENT_EVAL_API_KEYS")

CONCURRENCY = 8  # 并发数（必须与 API_KEYS_POOL 长度一致）
MAX_EXAMPLES = None  # 限制条数，None 表示全部


def load_generation_config(param_path: Path) -> Dict[str, Any]:
    """加载配置文件"""
    with open(param_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    gen_cfg = cfg.get("generation", {}) or {}
    backend = gen_cfg.get("backend", "openai")
    backend_cfgs = gen_cfg.get("backend_configs", {}) or {}
    backend_cfg = backend_cfgs.get(backend, {}) or {}

    sampling = gen_cfg.get("sampling_params", {}) or {}
    yaml_api_key = backend_cfg.get("api_key") or ""

    return {
        "backend": backend,
        "base_url": backend_cfg.get("base_url", "https://api.zhizengzeng.com/v1"),
        "api_key": yaml_api_key or os.environ.get("LLM_API_KEY", ""),
        "api_key_source": "yaml" if yaml_api_key else "environment",
        "model": backend_cfg.get("model_name", "gpt-4o"),
        "temperature": sampling.get("temperature", 0.7),
        "top_p": sampling.get("top_p", 0.8),
        "max_tokens": sampling.get("max_tokens", 2048),
    }


def build_client(base_url: str, api_key: str) -> OpenAI:
    """构建 OpenAI 客户端"""
    return OpenAI(api_key=api_key, base_url=base_url)


def chat(
    client: OpenAI,
    model: str,
    messages: List[Dict[str, str]],
    temperature: float = 0.7,
    top_p: float = 0.8,
    max_tokens: int = 1024,
) -> str:
    """调用模型 API"""
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )
        if hasattr(resp, 'error') and resp.error:
            error_msg = resp.error.get('message', 'Unknown API error')
            error_code = resp.error.get('code', 'unknown')
            return f"[API_ERROR: {error_code}] {error_msg}"
        if not hasattr(resp, 'choices') or not resp.choices:
            return "[API_ERROR: no_choices] API response has no choices"
        return resp.choices[0].message.content or ""
    except Exception as e:
        error_type = type(e).__name__
        error_msg = str(e)
        return f"[API_ERROR: {error_type}] {error_msg}"


def normalize_text(text: str) -> str:
    """简单归一化：小写、去标点、压缩空格"""
    import re
    import string as _string

    text = text.strip()
    text = text.lower()
    table = str.maketrans({c: " " for c in _string.punctuation})
    text = text.translate(table)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def accuracy_score(gt_list: List[str], pred: str) -> float:
    """只要 pred 包含任一标准答案的归一化形式，就记为 1"""
    pred_norm = normalize_text(pred)
    if not pred_norm:
        return 0.0
    gts = [normalize_text(g) for g in gt_list]
    return 1.0 if any(g in pred_norm or pred_norm in g for g in gts) else 0.0


def exact_match_score(gt_list: List[str], pred: str) -> float:
    """严格字符串 EM"""
    pred_norm = normalize_text(pred)
    gts = [normalize_text(g) for g in gt_list]
    return 1.0 if any(pred_norm == g for g in gts) else 0.0


def f1_score(gt_list: List[str], pred: str) -> float:
    """基于 token overlap 的 F1（取对所有 gt 的最高分）"""
    from collections import Counter

    def _f1(a: str, b: str) -> float:
        a_tokens = normalize_text(a).split()
        b_tokens = normalize_text(b).split()
        if not a_tokens or not b_tokens:
            return 0.0
        common = Counter(a_tokens) & Counter(b_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            return 0.0
        prec = num_same / len(b_tokens)
        rec = num_same / len(a_tokens)
        if prec + rec == 0:
            return 0.0
        return 2 * prec * rec / (prec + rec)

    scores = [_f1(g, pred) for g in gt_list]
    return max(scores) if scores else 0.0


def extract_choice_from_answer(answer: str) -> str:
    """从回答中提取选项字母（A/B/C/D）"""
    import re

    text = answer.strip()
    if not text:
        return text

    # 最优先：如果整个回答就是一个单独的选项字母
    if re.match(r"^[A-D]$", text):
        return text

    # 次优先：匹配类似 "最终选项：B" / "最终答案: C" / "答案是B" 这样的格式
    m = re.search(r"[最终答案选项]*[：:]\s*([A-D])\b", text)
    if m:
        return m.group(1).strip()

    # 再次：在首行中找第一个大写选项字母
    first_line = text.splitlines()[0].strip()
    m2 = re.search(r"^[^A-D]*([A-D])\b", first_line)
    if m2:
        return m2.group(1).strip()

    # 最后：在整个文本中找第一个出现的选项字母
    m3 = re.search(r"\b([A-D])\b", text)
    if m3:
        return m3.group(1).strip()

    return text


def evaluate_predictions(
    gt_all: List[List[str]],
    pred_all: List[str],
) -> Dict[str, float]:
    """计算评估指标"""
    accs, ems, f1s = [], [], []
    for gts, pred in zip(gt_all, pred_all):
        accs.append(accuracy_score(gts, pred))
        ems.append(exact_match_score(gts, pred))
        f1s.append(f1_score(gts, pred))
    n = len(pred_all) or 1
    return {
        "avg_acc": sum(accs) / n,
        "avg_em": sum(ems) / n,
        "avg_f1": sum(f1s) / n,
    }


def run_baseline_answer(
    client: OpenAI,
    eval_conf: Dict[str, Any],
    question: str,
) -> str:
    """直接让模型回答问题（baseline 模式）"""
    system_prompt = (
        "你是一个专业的法律问题解答助手。请仔细分析以下法律问题，并选择最正确的答案。\n\n"
        "【输出要求】\n"
        "只输出一个选项字母（A、B、C 或 D），不要输出其他内容。\n\n"
        "例如，如果答案是 B，你只需要输出：B"
    )

    user_prompt = f"请分析以下法律问题并给出答案：\n\n{question}"

    content = chat(
        client,
        model=eval_conf["model"],
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=eval_conf["temperature"],
        top_p=eval_conf["top_p"],
        max_tokens=min(eval_conf["max_tokens"], 128),  # baseline 只需要一个字母，限制 token
    )

    return content.strip()


def run_eval(
    dataset_path: Path,
    max_examples: int | None,
    eval_model: str | None,
    eval_base_url: str | None,
    output_path: Path,
    log_path: Path | None = None,
) -> Dict[str, float] | None:
    """运行评估"""
    # 设置日志
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    tee_logger = None

    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        tee_logger = TeeLogger(log_path)
        sys.stdout = tee_logger
        sys.stderr = tee_logger
        print(f"日志文件：{log_path}")
        print(f"开始时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)

    try:
        # 1. 加载配置
        gen_conf = load_generation_config(PARAM_PATH)
        eval_conf: Dict[str, Any] = {
            "model": eval_model or gen_conf["model"],
            "base_url": eval_base_url or gen_conf["base_url"],
            "temperature": gen_conf["temperature"],
            "top_p": gen_conf["top_p"],
            "max_tokens": gen_conf["max_tokens"],
        }

        # 2. 准备 API key 池
        if gen_conf["api_key_source"] == "yaml":
            eval_keys = [gen_conf["api_key"]] * CONCURRENCY
            print(f"使用 YAML API key：{CONCURRENCY} 组并行（可能触发限流）")
        elif API_KEYS_POOL and len(API_KEYS_POOL) >= CONCURRENCY:
            eval_keys = API_KEYS_POOL[:CONCURRENCY]
            print(f"使用多 API key 池模式：{len(eval_keys)} 个 key，并发数：{CONCURRENCY}")
        else:
            if not gen_conf["api_key"]:
                raise RuntimeError(
                    "未配置 API key 池且 legal2_rag_parameter.yaml 中 api_key 为空。\n"
                    "请配置 API_KEYS_POOL（8 个 deepseek API key）。"
                )
            eval_keys = [gen_conf["api_key"]] * CONCURRENCY
            print(f"使用单 API key 模式（回退）：{CONCURRENCY} 组并行（可能触发限流）")

        # 3. 读取数据集
        questions: List[str] = []
        golden_all: List[List[str]] = []
        with open(dataset_path, "r", encoding="utf-8") as f:
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

        total = len(questions)
        print(f"读取到 {total} 条样本，开始评估（Baseline 模式：直接回答）")
        print(f"模型：{eval_conf['model']}，并发数：{CONCURRENCY}")
        print("=" * 80)

        # 4. 线程安全的进度跟踪
        progress_lock = threading.Lock()
        completed = [0]

        def process_one(idx: int, q: str, gts: List[str], thread_id: int) -> Dict[str, Any]:
            """处理单道题目"""
            # 为当前线程分配 API key
            eval_key = eval_keys[thread_id % len(eval_keys)]
            eval_client = build_client(eval_conf["base_url"], eval_key)

            try:
                # 直接让模型回答问题
                full_answer = run_baseline_answer(eval_client, eval_conf, q)
            except Exception as e:
                full_answer = f"[ERROR] {e}"

            pred_for_eval = extract_choice_from_answer(full_answer)

            # 线程安全的进度更新
            with progress_lock:
                completed[0] += 1
                cur = completed[0]
                percent = cur / total * 100 if total else 0.0
                print(f"\rEvaluating {cur}/{total} ({percent:5.1f}%)", end="", flush=True)

            return {
                "idx": idx,
                "question": q,
                "golden_answers": gts,
                "model_answer": full_answer,
                "pred_for_eval": pred_for_eval,
            }

        # 5. 使用线程池并发处理
        results_dict: Dict[int, Dict[str, Any]] = {}
        with ThreadPoolExecutor(max_workers=CONCURRENCY) as executor:
            futures = {
                executor.submit(process_one, idx, q, gts, idx % CONCURRENCY): idx
                for idx, (q, gts) in enumerate(zip(questions, golden_all))
            }
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    results_dict[idx] = future.result()
                except Exception as e:
                    results_dict[idx] = {
                        "idx": idx,
                        "question": questions[idx] if idx < len(questions) else "",
                        "golden_answers": golden_all[idx] if idx < len(golden_all) else [],
                        "model_answer": f"[TASK_ERROR] {e}",
                        "pred_for_eval": "",
                    }

        # 6. 按原始顺序整理结果
        detail = [results_dict[i] for i in range(total)]
        preds = [d["pred_for_eval"] for d in detail]

        # 7. 计算整体指标
        metrics = evaluate_predictions(golden_all, preds)
        print()  # 换行
        print("评估结果：", metrics)

        # 8. 保存明细和指标
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "metrics": metrics,
                    "model_under_test": {
                        "model": eval_conf["model"],
                        "base_url": eval_conf["base_url"],
                    },
                    "mode": "baseline",
                    "examples": detail,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        print(f"详细结果已保存到：{output_path}")
        if log_path:
            print(f"结束时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("=" * 80)
    finally:
        # 恢复标准输出
        if tee_logger:
            sys.stdout = original_stdout
            sys.stderr = original_stderr
            tee_logger.close()

    return metrics


class TeeLogger:
    """同时输出到控制台和日志文件的 Logger"""

    def __init__(self, log_file_path: Path):
        self.log_file = open(log_file_path, "w", encoding="utf-8")
        self.stdout = sys.stdout
        self.stderr = sys.stderr

    def write(self, text: str):
        self.stdout.write(text)
        self.log_file.write(text)
        self.log_file.flush()

    def flush(self):
        self.stdout.flush()
        self.log_file.flush()

    def close(self):
        self.log_file.close()


def main():
    """批量评估多个维度数据集"""
    # 1. 获取 dimension_jsonl 目录下的所有 jsonl 文件
    jsonl_files = sorted([f for f in DIMENSION_JSONL_DIR.glob("*.jsonl")])
    
    if len(jsonl_files) < 2:
        print(f"错误：dimension_jsonl 目录下文件数量不足（需要至少 2 个文件，当前有 {len(jsonl_files)} 个）")
        return

    # 2. 从第二个文件开始（索引 1）到最后一个文件
    files_to_eval = jsonl_files[1:]  # 跳过第一个文件

    print(f"找到 {len(jsonl_files)} 个数据集文件，将从第 2 个开始评估（共 {len(files_to_eval)} 个）")
    print("=" * 80)

    # 3. 加载配置
    gen_conf = load_generation_config(PARAM_PATH)
    eval_base_url = EVAL_BASE_URL or gen_conf["base_url"]

    # 4. 逐个评估每个数据集
    all_metrics = {}
    for jsonl_file in files_to_eval:
        dataset_name = jsonl_file.stem  # 例如：2_LegalReasoning_sample_500
        print(f"\n开始评估数据集：{dataset_name}")

        output_path = OUTPUT_DIR / f"{dataset_name}_baseline_eval.json"
        log_path = OUTPUT_DIR / f"{dataset_name}_baseline.log"

        metrics = run_eval(
            dataset_path=jsonl_file,
            max_examples=MAX_EXAMPLES,
            eval_model=EVAL_MODEL,
            eval_base_url=eval_base_url,
            output_path=output_path,
            log_path=log_path,
        )

        if metrics:
            all_metrics[dataset_name] = metrics
            print(f"数据集 {dataset_name} 评估完成：{metrics}")

    # 5. 计算总体平均
    if all_metrics:
        overall_avg = {
            "avg_acc": sum(m["avg_acc"] for m in all_metrics.values()) / len(all_metrics),
            "avg_em": sum(m["avg_em"] for m in all_metrics.values()) / len(all_metrics),
            "avg_f1": sum(m["avg_f1"] for m in all_metrics.values()) / len(all_metrics),
        }
        all_metrics["OVERALL_AVERAGE"] = overall_avg

        # 6. 保存汇总结果
        summary_path = OUTPUT_DIR / "baseline_multi_eval_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(all_metrics, f, ensure_ascii=False, indent=2)
        print(f"\n汇总结果已保存到：{summary_path}")
        print("总体平均指标：", overall_avg)


if __name__ == "__main__":
    main()
