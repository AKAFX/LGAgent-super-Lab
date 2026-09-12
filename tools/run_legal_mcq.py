#!/usr/bin/env python3
"""Safely run the opt-in LegalMCQ route for one question or a JSONL dataset."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from lgagent.config import LGAgentConfig, load_lgagent_config
from lgagent.legal_mcq import (
    BenchmarkRecordAdapter,
    SolveMode,
    build_openai_role_clients,
    evaluate_case,
    parse_question_text,
)
from lgagent.runner import LGAgentPlusRunner
from lgagent.legal_mcq.observability import CallJournal, redact_telemetry
from lgagent.trace import RunTrace


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "LegalMCQ three-role runner. No API request is made unless "
            "--execute is explicitly supplied."
        )
    )
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--dry-run", action="store_true")
    action.add_argument("--execute", action="store_true")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--question")
    source.add_argument("--dataset", type=Path)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT_DIR / "examples/parameter/legal2_rag_parameter.yaml",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT_DIR / "output/legal_mcq/results.jsonl",
    )
    parser.add_argument("--max-examples", type=int)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument(
        "--log-model-output",
        action="store_true",
        help="Include credential-redacted visible model output in call logs; never hidden reasoning.",
    )
    return parser.parse_args(argv)


def _read_records(path: Path, max_examples: int | None) -> list[Mapping[str, Any]]:
    records: list[Mapping[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, Mapping):
                raise ValueError(f"{path}:{line_number}: row must be an object")
            records.append(value)
            if max_examples is not None and len(records) >= max_examples:
                break
    if not records:
        raise ValueError(f"{path}: dataset has no usable records")
    return records


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as stream:
        stream.write(text)
        temporary = Path(stream.name)
    temporary.replace(path)


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    successful = [item for item in rows if "error_type" not in item]
    correct = sum(bool(item.get("exact_match")) for item in successful)
    return {
        "total": total,
        "successful": len(successful),
        "failed": total - len(successful),
        "correct": correct,
        "exact_match": correct / total if total else 0.0,
        "successful_exact_match": (
            correct / len(successful) if successful else 0.0
        ),
        "option_coverage_rate": (
            sum(bool(item["all_options_covered"]) for item in successful) / total
            if total else 0.0
        ),
        "verifier_acceptance_rate": (
            sum(bool(item["verifier_accepted"]) for item in successful) / total
            if total else 0.0
        ),
        "needs_review_rate": (
            sum(bool(item["needs_review"]) for item in successful) / total
            if total else 0.0
        ),
        **_usage_summary(rows),
    }


def _usage_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    calls = [
        call
        for row in rows
        for call in row.get("diagnostics", {}).get("trace", {}).get("model_calls", [])
    ]
    invoked = [call for call in calls if call.get("model_invoked")]
    completion_tokens = sum(
        call.get("usage", {}).get("completion_tokens", 0) for call in calls
    )
    reasoning_tokens = sum(call.get("reasoning_tokens") or 0 for call in calls)
    return {
        "logical_model_calls": len(invoked),
        "budget_blocked_attempts": sum(
            call.get("outcome") == "budget_blocked" for call in calls
        ),
        "observed_usage": {
            field: sum(call.get("usage", {}).get(field, 0) for call in calls)
            for field in ("prompt_tokens", "completion_tokens", "total_tokens")
        },
        "observed_reasoning_tokens": reasoning_tokens,
        "observed_reasoning_token_rate": (
            reasoning_tokens / completion_tokens if completion_tokens else 0.0
        ),
        "reasoning_efforts": dict(
            Counter(
                call.get("reasoning_effort_requested") or "provider_default"
                for call in invoked
            )
        ),
        "usage_unknown_calls": sum(
            call.get("usage_reported") is not True for call in invoked
        ),
        "finish_reasons": dict(
            Counter(call.get("finish_reason") or "not_reported" for call in invoked)
        ),
        "call_outcomes": dict(
            Counter(call.get("outcome") or "not_reported" for call in calls)
        ),
    }


def _save_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    _atomic_write(
        path,
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in rows
        ),
    )


def _runner(config: LGAgentConfig) -> LGAgentPlusRunner:
    clients = build_openai_role_clients(config)
    return LGAgentPlusRunner(
        config,
        reasoning_model=clients.controller,
        evaluation_model=clients.solver,
        verifier_model=clients.verifier,
        legal_controller_model=clients.controller,
        legal_solver_model=clients.solver,
        legal_solver_fallback_model=clients.solver_fallback,
        legal_verifier_model=clients.verifier,
        project_root=ROOT_DIR,
    )


def _dry_run_payload(config: LGAgentConfig, count: int) -> dict[str, Any]:
    settings = config.legal_mcq
    controller = settings.controller_model or config.generation
    solver = settings.solver_model or config.generation
    verifier = settings.verifier_model or controller
    solver_fallback = settings.solver_fallback_model
    return {
        "dry_run": True,
        "makes_api_calls": False,
        "route": "legal-mcq-three-role",
        "items": count,
        "estimated_calls": {
            "minimum": count * 3,
            "maximum_without_schema_retries": count
            * (
                3
                + 2 * settings.max_revision_rounds
                + (1 if solver_fallback is not None else 0)
            ),
        },
        "models": {
            "controller": controller.model,
            "solver": solver.model,
            "verifier": verifier.model,
            "solver_fallback": (
                solver_fallback.model if solver_fallback is not None else None
            ),
        },
        "reasoning_effort": {
            "controller": controller.reasoning_effort or "provider_default",
            "solver": solver.reasoning_effort or "provider_default",
            "solver_fallback": (
                solver_fallback.reasoning_effort or "provider_default"
                if solver_fallback is not None
                else None
            ),
            "verifier": verifier.reasoning_effort or "provider_default",
        },
        "mode": settings.mode,
        "max_total_tokens_per_item": settings.max_total_tokens,
        "max_model_calls_per_item": settings.max_model_calls,
        "max_wall_time_seconds_per_item": settings.max_wall_time_seconds,
        "model_call_timeout_seconds": settings.model_call_timeout_seconds,
        "role_call_timeout_seconds": {
            "controller": settings.controller_call_timeout_seconds,
            "solver": settings.solver_call_timeout_seconds,
            "solver_fallback": settings.solver_fallback_timeout_seconds,
            "verifier": settings.verifier_call_timeout_seconds,
        },
        "solver_timeout_retries": settings.solver_timeout_retries,
        "solver_timeout_circuit_breaker": (
            settings.solver_timeout_circuit_breaker
        ),
        "sdk_automatic_retries": 0,
        "controller_structured_output_mode": settings.controller_structured_output_mode,
        "solver_structured_output_mode": (
            settings.solver_structured_output_mode
        ),
        "verifier_structured_output_mode": (
            settings.verifier_structured_output_mode
        ),
        "solver_completion_budget": {
            "visible_output_target": min(settings.solver_visible_output_tokens, solver.max_tokens),
            "reasoning_allowance_target": settings.solver_reasoning_allowance_tokens,
            "initial_max_tokens": min(
                solver.max_tokens,
                settings.solver_visible_output_tokens + settings.solver_reasoning_allowance_tokens,
            ),
            "hard_cap": solver.max_tokens,
            "provider_sub_budget_enforced": False,
        },
        "auxiliary_reasoning_reserve_tokens": settings.auxiliary_reasoning_reserve_tokens,
        "verifier_output_budget": {
            "visible_output_target": settings.verifier_visible_output_tokens,
            "length_retry_tokens": settings.verifier_length_retry_tokens,
        },
        "skip_verifier_on_deterministic_errors": (
            settings.skip_verifier_on_deterministic_errors
        ),
    }


def _redact_error(config: LGAgentConfig, message: str) -> str:
    return redact_telemetry(message, _credential_values(config))


def _credential_values(config: LGAgentConfig) -> tuple[str, ...]:
    role_configs = (
        config.legal_mcq.controller_model,
        config.legal_mcq.solver_model,
        config.legal_mcq.solver_fallback_model,
        config.legal_mcq.verifier_model,
    )
    return tuple({
        config.generation.api_key,
        *(item.api_key for item in role_configs if item is not None),
    })


def _call_trace(
    journal: CallJournal, question_id: str, dataset_index: int | None
) -> RunTrace:
    return RunTrace(
        _event_sink=lambda event: journal.write(
            {**event, "question_id": question_id, "dataset_index": dataset_index}
        )
    )


def _failure_record(
    config: LGAgentConfig,
    question_id: str,
    error: Exception,
    trace: RunTrace,
) -> dict[str, Any]:
    diagnostics = dict(getattr(error, "diagnostics", {}))
    details = getattr(error, "details", {})
    diagnostics["trace"] = trace.as_dict()
    return redact_telemetry(
        {
            "question_id": question_id,
            "trace_id": trace.run_id,
            "error_type": type(error).__name__,
            "message": str(error),
            "failed_agent": (
                diagnostics.get("failed_agent")
                or diagnostics.get("agent")
                or details.get("agent")
                or (trace.model_calls[-1].agent if trace.model_calls else None)
            ),
            "diagnostics": diagnostics,
        },
        _credential_values(config),
    )


def _check_output_paths(output: Path) -> Path:
    calls_path = output.with_suffix(".calls.jsonl")
    for path in (output, calls_path, output.with_suffix(".summary.json")):
        if path.exists():
            raise SystemExit(f"Output already exists; choose a new path: {path}")
    return calls_path


def main(argv: list[str] | None = None) -> int:
    load_dotenv(ROOT_DIR / ".env")
    args = parse_args(argv)
    if args.max_examples is not None and args.max_examples <= 0:
        raise SystemExit("--max-examples must be positive")
    if args.concurrency <= 0:
        raise SystemExit("--concurrency must be positive")

    config = load_lgagent_config(args.config)
    if not config.legal_mcq.enabled:
        raise SystemExit(
            "legal_mcq.enabled is false; enable the opt-in route in the config"
        )
    mode = SolveMode(config.legal_mcq.mode)
    if args.execute and mode is SolveMode.OPEN_BOOK:
        raise SystemExit(
            "The standalone CLI has no EvidenceProvider. Use closed_book/auto "
            "or inject OathAuditedEvidenceProvider through the Python API."
        )

    if args.question is not None:
        request = parse_question_text(args.question, mode=mode)
        if args.dry_run:
            print(
                json.dumps(
                    _dry_run_payload(config, 1),
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 0
        calls_path = _check_output_paths(args.output)
        with CallJournal(calls_path, secrets=_credential_values(config)) as journal:
            trace = _call_trace(journal, request.question_id, None)
            try:
                result = _runner(config).run_legal_request(
                    request, trace=trace, log_model_output=args.log_model_output
                )
            except Exception as exc:
                row = _failure_record(config, request.question_id, exc, trace)
                _save_rows(args.output, [row])
                print(row["message"], file=sys.stderr)
                return 1
            _save_rows(
                args.output,
                [{
                    "question_id": request.question_id,
                    "trace_id": trace.run_id,
                    "answer": result.answer.as_dict(),
                    "diagnostics": result.diagnostics(),
                }],
            )
        print(result.answer.to_markdown())
        return 0

    assert args.dataset is not None
    records = _read_records(args.dataset, args.max_examples)
    cases = [
        BenchmarkRecordAdapter.adapt(
            record,
            source_dataset=str(args.dataset),
            mode=mode,
        )
        for record in records
    ]
    if args.dry_run:
        print(
            json.dumps(
                _dry_run_payload(config, len(cases)),
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    calls_path = _check_output_paths(args.output)
    runner = _runner(config)
    journal = CallJournal(calls_path, secrets=_credential_values(config))
    batch_started = perf_counter()

    def run_case(index: int):
        started = perf_counter()
        captured = []
        trace = _call_trace(journal, cases[index].request.question_id, index)

        def solve(request):
            result = runner.run_legal_request(
                request, trace=trace, log_model_output=args.log_model_output
            )
            captured.append(result)
            return result

        try:
            evaluation = evaluate_case(cases[index], solve)
            result = captured[0]
            row = {
                **evaluation.as_dict(),
                "answer": result.answer.as_dict(),
                "diagnostics": result.diagnostics(),
                "trace_id": trace.run_id,
            }
        except Exception as exc:
            row = _failure_record(config, cases[index].request.question_id, exc, trace)
        row["dataset_index"] = index
        row["latency_seconds"] = perf_counter() - started
        return index, row

    completed: list[tuple[int, dict[str, Any]]] = []
    with journal, ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = {
            executor.submit(run_case, index): index
            for index in range(len(cases))
        }
        for future in as_completed(futures):
            index = futures[future]
            try:
                completed.append(future.result())
            except Exception as exc:
                completed.append(
                    (
                        index,
                        {
                            "question_id": cases[index].request.question_id,
                            "error_type": type(exc).__name__,
                            "message": _redact_error(config, str(exc)),
                        },
                    )
                )
            rows = [row for _, row in sorted(completed)]
            _save_rows(args.output, rows)
            progress = _summarize(rows)
            print(
                f"[{len(rows)}/{len(cases)}] "
                f"correct={progress['correct']} "
                f"returned={progress['successful']} "
                f"failed={progress['failed']} "
                f"elapsed={perf_counter() - batch_started:.1f}s",
                file=sys.stderr,
                flush=True,
            )
    rows = [row for _, row in sorted(completed)]
    summary = _summarize(rows)
    summary["elapsed_seconds"] = perf_counter() - batch_started
    summary["calls_output"] = str(calls_path)
    summary_path = args.output.with_suffix(".summary.json")
    _atomic_write(
        summary_path,
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "summary": str(summary_path),
                **summary,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 1 if summary["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
