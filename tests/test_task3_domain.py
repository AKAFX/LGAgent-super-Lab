from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from lgagent.config import ModelConfig, load_generation_config
from lgagent.domain import run_single_question
from lgagent.model import (
    ChatMessage,
    ModelRequest,
    ModelResponse,
    OpenAIChatModel,
    TokenUsage,
)
from lgagent.serialization import dumps_json


def lawyer_a_payload() -> dict:
    return {
        "task_type": "single_choice",
        "question_focus": "focus",
        "legal_domain": "criminal_law",
        "option_claims": {option: f"claim-{option}" for option in "ABCD"},
        "option_keywords": {option: [f"keyword-{option}"] for option in "ABCD"},
        "trap_signals": ["trap"],
        "unknowns": ["unknown"],
    }


def judge_payload() -> dict:
    return {
        "need_retrieval": False,
        "global_query": "query",
        "option_queries": {option: f"query-{option}" for option in "ABCD"},
        "evidence_requirements": {
            option: {"support": f"support-{option}", "refute": f"refute-{option}"}
            for option in "ABCD"
        },
        "counterfactual_focus": "counterexample",
        "stop_rule": "stop",
    }


def b1_payload() -> dict:
    return {
        "final_answer": "B",
        "verification": {
            option: {
                "status": status,
                "score": 0.8,
                "reason": f"reason-{option}",
            }
            for option, status in zip(
                "ABCD", ("REFUTE", "SUPPORT", "REFUTE", "REFUTE")
            )
        },
        "initial_answer": "B",
        "initial_confidence": 0.9,
        "reasoning": "reasoning",
    }


class FakeChatModel:
    def __init__(self, outputs: list[str]) -> None:
        self.outputs = list(outputs)
        self.requests: list[ModelRequest] = []

    def complete(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        return ModelResponse(
            self.outputs.pop(0),
            TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            request_id=f"fake-{len(self.requests)}",
        )


CONFIG = ModelConfig(
    backend="fake",
    base_url="https://example.invalid/v1",
    api_key="secret-value",
    model="fake-model",
    temperature=0.0,
    top_p=1.0,
    max_tokens=1024,
    api_key_source="yaml",
)


class ConfigurationAndSerializationTest(unittest.TestCase):
    def test_yaml_api_key_precedes_environment_and_falls_back_when_empty(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            path.write_text(
                """
generation:
  backend: openai
  backend_configs:
    openai:
      api_key: yaml-key
      base_url: https://example.invalid/v1
      model_name: test-model
  sampling_params:
    temperature: 0
    top_p: 1
    max_tokens: 128
""",
                encoding="utf-8",
            )
            config = load_generation_config(path, environ={"LLM_API_KEY": "env-key"})
            self.assertEqual(config.api_key, "yaml-key")
            self.assertEqual(config.api_key_source, "yaml")

            path.write_text(
                path.read_text(encoding="utf-8").replace("api_key: yaml-key", "api_key: ''"),
                encoding="utf-8",
            )
            fallback = load_generation_config(path, environ={"LLM_API_KEY": "env-key"})
            self.assertEqual(fallback.api_key, "env-key")
            self.assertEqual(fallback.api_key_source, "environment")

    def test_serialization_redacts_secrets(self) -> None:
        serialized = dumps_json(CONFIG)
        self.assertNotIn("secret-value", serialized)
        self.assertIn("[REDACTED]", serialized)


class DomainWorkflowTest(unittest.TestCase):
    def test_openai_adapter_uses_the_shared_model_contract(self) -> None:
        captured: dict = {}

        def create(**kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                id="provider-request",
                choices=[SimpleNamespace(message=SimpleNamespace(content="answer"))],
                usage=SimpleNamespace(
                    prompt_tokens=7,
                    completion_tokens=2,
                    total_tokens=9,
                ),
            )

        client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        )
        response = OpenAIChatModel(client).complete(
            ModelRequest(
                model="test-model",
                messages=(ChatMessage("user", "question"),),
                max_tokens=32,
                seed=12345,
            )
        )

        self.assertEqual(response.content, "answer")
        self.assertEqual(response.usage.total_tokens, 9)
        self.assertEqual(captured["messages"], [{"role": "user", "content": "question"}])
        self.assertEqual(captured["seed"], 12345)
        self.assertEqual(response.seed_requested, 12345)
        self.assertEqual(
            response.provider_seed_guarantee,
            "requested_not_guaranteed",
        )

    def test_fake_models_run_end_to_end_and_record_trace(self) -> None:
        reasoning = FakeChatModel(
            [json.dumps(lawyer_a_payload()), json.dumps(judge_payload())]
        )
        evaluation = FakeChatModel(
            [
                '{"initial_answer":"B","confidence":0.9}',
                json.dumps(b1_payload()),
            ]
        )

        result = run_single_question(
            "Question with options A, B, C and D",
            reasoning_model=reasoning,
            evaluation_model=evaluation,
            reasoning_config=CONFIG,
            evaluation_config=CONFIG,
        )

        self.assertEqual(result.final_answer, "B")
        self.assertEqual(len(result.trace.model_calls), 4)
        self.assertEqual(result.trace.total_tokens, 60)
        self.assertEqual(result.diagnostics["dialogue_rounds"], 0)
        self.assertEqual(result.trace.routes[-1].route, "final")
        self.assertNotIn("secret-value", dumps_json(result))

    def test_invalid_output_is_retried_and_traced(self) -> None:
        reasoning = FakeChatModel(
            [
                "not json",
                json.dumps(lawyer_a_payload()),
                json.dumps(judge_payload()),
            ]
        )
        evaluation = FakeChatModel(
            [
                '{"initial_answer":"B","confidence":0.9}',
                json.dumps(b1_payload()),
            ]
        )

        result = run_single_question(
            "Question with options A, B, C and D",
            reasoning_model=reasoning,
            evaluation_model=evaluation,
            reasoning_config=CONFIG,
            evaluation_config=CONFIG,
        )

        lawyer_calls = [
            call for call in result.trace.model_calls if call.agent == "lawyer_a"
        ]
        self.assertEqual([call.attempt for call in lawyer_calls], [1, 2])
        self.assertEqual(lawyer_calls[0].error_type, "StructuredOutputError")
        self.assertEqual(result.trace.errors[0]["stage"], "lawyer_a")
        self.assertIn("不符合严格JSON协议", reasoning.requests[1].messages[-1].content)


if __name__ == "__main__":
    unittest.main()
