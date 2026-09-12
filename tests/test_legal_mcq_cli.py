from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
spec = importlib.util.spec_from_file_location(
    "legal_mcq_cli_under_test", ROOT / "tools/run_legal_mcq.py"
)
assert spec is not None and spec.loader is not None
cli = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cli)


def evaluation_record(correct: bool) -> dict:
    return {
        "exact_match": correct,
        "all_options_covered": True,
        "verifier_accepted": True,
        "needs_review": False,
    }


def test_failed_cases_stay_in_accuracy_denominator():
    rows = [
        evaluation_record(True),
        evaluation_record(False),
        {"error_type": "BudgetExceededError"},
    ]
    summary = cli._summarize(rows)
    assert summary["correct"] == 1
    assert summary["successful"] == 2
    assert summary["failed"] == 1
    assert summary["exact_match"] == pytest.approx(1 / 3)
    assert summary["successful_exact_match"] == 0.5
    assert cli._summarize([])["exact_match"] == 0.0


def test_cli_saves_completed_cases_before_next_case_and_returns_failure(
    tmp_path, monkeypatch, capsys
):
    config = tmp_path / "config.yaml"
    config.write_text(
        "legal_mcq:\n  enabled: true\nlgagent_plus:\n  enabled: false\n",
        encoding="utf-8",
    )
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text(
        "".join(
            json.dumps(
                {
                    "id": index,
                    "question": "Question\nA. First\nB. Second",
                    "golden_answers": ["A"],
                }
            ) + "\n"
            for index in range(3)
        ),
        encoding="utf-8",
    )
    output = tmp_path / "results.jsonl"

    class Runner:
        def run_legal_request(self, request, *, trace, log_model_output):
            assert log_model_output is False
            trace.emit("model_call_started", {"call_id": request.question_id})
            if request.question_id == "1":
                raise RuntimeError("fixture failure")
            return SimpleNamespace(
                answer=SimpleNamespace(as_dict=lambda: {"selected_options": ["A"]}),
                diagnostics=lambda: {"trace": trace.as_dict()},
            )

    monkeypatch.setattr(cli, "load_dotenv", lambda *args, **kwargs: None)
    monkeypatch.setattr(cli, "_runner", lambda _: Runner())

    def evaluate(case, solve):
        solve(case.request)
        return SimpleNamespace(as_dict=lambda: evaluation_record(True))

    monkeypatch.setattr(cli, "evaluate_case", evaluate)
    snapshots = []
    save_rows = cli._save_rows

    def save(path, rows):
        save_rows(path, rows)
        snapshots.append(len(path.read_text(encoding="utf-8").splitlines()))

    monkeypatch.setattr(cli, "_save_rows", save)
    code = cli.main(
        [
            "--execute", "--config", str(config),
            "--dataset", str(dataset), "--output", str(output),
        ]
    )
    assert code == 1
    assert snapshots == [1, 2, 3]
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert [row["dataset_index"] for row in rows] == [0, 1, 2]
    assert rows[1]["error_type"] == "RuntimeError"
    assert all(row["latency_seconds"] >= 0 for row in rows)
    summary = json.loads(output.with_suffix(".summary.json").read_text())
    assert summary["exact_match"] == pytest.approx(2 / 3)
    calls = [
        json.loads(line)
        for line in output.with_suffix(".calls.jsonl").read_text().splitlines()
    ]
    assert len(calls) == 3
    assert {call["question_id"] for call in calls} == {"0", "1", "2"}
    assert rows[1]["diagnostics"]["trace"]["run_id"] == rows[1]["trace_id"]
    assert "[3/3]" in capsys.readouterr().err


def test_real_runner_cli_persists_response_and_error_telemetry(tmp_path, monkeypatch):
    from lgagent.model import ModelResponse, TokenUsage
    from test_legal_mcq_observability import ResponseModel, SECRETS, make_runner
    from test_task22_legal_mcq import QUESTION, controller_payload

    invalid = '{"incomplete": "' + SECRETS[0]
    controller = ResponseModel(
        ModelResponse(
            json.dumps(controller_payload()),
            usage=TokenUsage(10, 5, 15),
            finish_reason="stop",
            usage_reported=True,
        ),
        *[
            ModelResponse(
                invalid, usage=TokenUsage(10, 5, 15),
                finish_reason="length", usage_reported=True,
            )
            for _ in range(2)
        ],
    )
    runner = make_runner(controller=controller)
    monkeypatch.setattr(cli, "load_dotenv", lambda *args, **kwargs: None)
    monkeypatch.setattr(cli, "load_lgagent_config", lambda _: runner.config)
    monkeypatch.setattr(cli, "_runner", lambda _: runner)
    dataset = tmp_path / "data.jsonl"
    dataset.write_text(
        "".join(
            json.dumps({"id": index, "question": QUESTION, "golden_answers": ["C"]})
            + "\n"
            for index in range(2)
        )
    )
    output = tmp_path / "results.jsonl"
    code = cli.main([
        "--execute", "--dataset", str(dataset), "--output", str(output),
        "--log-model-output",
    ])
    assert code == 1
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert rows[0]["exact_match"] is True
    assert rows[1]["failed_agent"] == "legal_mcq_controller"
    assert rows[1]["diagnostics"]["execution_budget"]["calls_used"] == 2
    assert rows[1]["diagnostics"]["trace"]["model_calls"][-1]["finish_reason"] == "length"
    assert rows[1]["diagnostics"]["trace"]["model_calls"][-1]["output_text"].endswith(
        "[REDACTED]"
    )
    summary = json.loads(output.with_suffix(".summary.json").read_text())
    assert summary["logical_model_calls"] == 5
    assert summary["observed_usage"]["total_tokens"] == 75
    assert summary["usage_unknown_calls"] == 2
    assert summary["exact_match"] == 0.5
    calls_text = output.with_suffix(".calls.jsonl").read_text()
    events = [json.loads(line) for line in calls_text.splitlines()]
    assert len(events) == 10
    assert len({event["call_id"] for event in events}) == 5
    assert "golden_answers" not in calls_text
    assert not any(secret in calls_text + output.read_text() for secret in SECRETS)


@pytest.mark.parametrize("failure", [False, True])
def test_single_question_also_persists_trace(tmp_path, monkeypatch, failure):
    from lgagent.model import ModelCallError
    from test_legal_mcq_observability import ResponseModel, make_runner
    from test_task22_legal_mcq import QUESTION

    runner = make_runner(
        controller=(
            ResponseModel(ModelCallError("authentication error")) if failure else None
        )
    )
    monkeypatch.setattr(cli, "load_dotenv", lambda *args, **kwargs: None)
    monkeypatch.setattr(cli, "load_lgagent_config", lambda _: runner.config)
    monkeypatch.setattr(cli, "_runner", lambda _: runner)
    output = tmp_path / "single.jsonl"
    code = cli.main(["--execute", "--question", QUESTION, "--output", str(output)])
    assert code == int(failure)
    row = json.loads(output.read_text())
    assert row["diagnostics"]["trace"]["model_calls"]
    assert row["trace_id"] == row["diagnostics"]["trace"]["run_id"]
    events = [
        json.loads(line) for line in output.with_suffix(".calls.jsonl").read_text().splitlines()
    ]
    assert events[0]["event"] == "model_call_started"
    assert events[-1]["event"] == "model_call_finished"
    assert events[-1]["outcome"] == ("provider_error" if failure else "success")
    assert all(event.get("output_text") is None for event in events)


def test_dry_run_never_creates_log_files_or_clients(tmp_path, monkeypatch):
    from test_legal_mcq_observability import make_runner
    from test_task22_legal_mcq import QUESTION

    runner = make_runner()
    monkeypatch.setattr(cli, "load_dotenv", lambda *args, **kwargs: None)
    monkeypatch.setattr(cli, "load_lgagent_config", lambda _: runner.config)
    monkeypatch.setattr(
        cli, "_runner", lambda _: pytest.fail("dry-run must not create clients")
    )
    output = tmp_path / "dry-run.jsonl"
    assert cli.main(["--dry-run", "--question", QUESTION, "--output", str(output)]) == 0
    assert list(tmp_path.iterdir()) == []


def test_summary_keeps_failed_usage_and_excludes_pre_call_budget_blocks():
    completed = evaluation_record(True)
    completed["diagnostics"] = {"trace": {"model_calls": [{
        "model_invoked": True,
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        "usage_reported": True,
        "reasoning_tokens": 3,
        "finish_reason": "stop",
        "outcome": "success",
    }]}}
    failure = {
        "error_type": "BudgetExceededError",
        "diagnostics": {"trace": {"model_calls": [{
            "model_invoked": True, "usage": {"total_tokens": 20},
            "usage_reported": True, "finish_reason": "length",
            "outcome": "schema_error",
        }, {
            "model_invoked": False, "usage": {"total_tokens": 0},
            "outcome": "budget_blocked",
        }]}},
    }
    summary = cli._summarize([completed, failure])
    assert summary["exact_match"] == 0.5
    assert summary["logical_model_calls"] == 2
    assert summary["budget_blocked_attempts"] == 1
    assert summary["observed_usage"]["total_tokens"] == 35
    assert summary["observed_reasoning_tokens"] == 3
    assert summary["observed_reasoning_token_rate"] == 3 / 5
    assert summary["reasoning_efforts"] == {"provider_default": 2}
    assert summary["usage_unknown_calls"] == 0
    assert summary["finish_reasons"] == {"stop": 1, "length": 1}
