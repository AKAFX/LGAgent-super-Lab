from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
for path in (ROOT_DIR, SRC_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from lgagent.protocol import (
    B0Output,
    B1Output,
    JudgeOutput,
    LawyerAOutput,
    StructuredOutputError,
    extract_json_object,
    should_request_clarification,
)
from tools.legal_multi_agent_prompt_demo import (
    run_lawyer_answer_b0_blind,
    run_lawyer_judge_dialogue,
)


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
        "need_retrieval": True,
        "global_query": "query",
        "option_queries": {option: f"query-{option}" for option in "ABCD"},
        "evidence_requirements": {
            option: {"support": f"support-{option}", "refute": f"refute-{option}"}
            for option in "ABCD"
        },
        "counterfactual_focus": "counterexample",
        "stop_rule": "stop",
    }


def b1_payload(statuses: tuple[str, str, str, str]) -> dict:
    return {
        "final_answer": "A",
        "verification": {
            option: {"status": status, "score": 0.8, "reason": f"reason-{option}"}
            for option, status in zip("ABCD", statuses)
        },
        "initial_answer": "A",
        "initial_confidence": 0.8,
        "reasoning": "reasoning",
    }


CONFIG = {
    "model": "fake",
    "temperature": 0.0,
    "top_p": 1.0,
    "max_tokens": 1024,
}


class StrictJsonAndSchemaTest(unittest.TestCase):
    def test_extracts_complete_markdown_fence(self) -> None:
        self.assertEqual(extract_json_object('```json\n{"answer": "A"}\n```'), {"answer": "A"})
        self.assertEqual(extract_json_object('```\n{"answer": "A"}\n```'), {"answer": "A"})

    def test_rejects_surrounding_prose_and_non_object(self) -> None:
        with self.assertRaises(StructuredOutputError):
            extract_json_object('result: {"answer": "A"}')
        with self.assertRaises(StructuredOutputError):
            extract_json_object('```json\n{"answer": "A"}\n```\nextra')
        with self.assertRaises(StructuredOutputError):
            extract_json_object('["A"]')

    def test_all_agent_schemas_accept_valid_outputs(self) -> None:
        self.assertEqual(LawyerAOutput.from_text(json.dumps(lawyer_a_payload())).task_type, "single_choice")
        self.assertTrue(JudgeOutput.from_text(f"```json\n{json.dumps(judge_payload())}\n```").need_retrieval)
        self.assertEqual(B0Output.from_text('{"initial_answer":" a ","confidence":0.6}').initial_answer, "A")
        self.assertEqual(
            B1Output.from_text(json.dumps(b1_payload(("SUPPORT", "REFUTE", "NEI", "REFUTE")))).final_answer,
            "A",
        )

    def test_lawyer_a_normalizes_explicit_null_date_aliases(self) -> None:
        for value in ("null", "None", "不适用", "未知", "无"):
            with self.subTest(value=value):
                payload = lawyer_a_payload()
                payload["case_date"] = value
                self.assertIsNone(
                    LawyerAOutput.from_text(
                        json.dumps(payload, ensure_ascii=False)
                    ).case_date
                )

        payload = lawyer_a_payload()
        payload["case_date"] = "not-a-date"
        with self.assertRaisesRegex(StructuredOutputError, "ISO date"):
            LawyerAOutput.from_text(json.dumps(payload))

    def test_missing_field_and_illegal_option_are_rejected(self) -> None:
        missing = lawyer_a_payload()
        del missing["question_focus"]
        with self.assertRaises(StructuredOutputError):
            LawyerAOutput.from_text(json.dumps(missing))

        invalid = b1_payload(("SUPPORT", "REFUTE", "NEI", "REFUTE"))
        invalid["final_answer"] = "E"
        with self.assertRaises(StructuredOutputError):
            B1Output.from_text(json.dumps(invalid))

    def test_lawyer_a_rejects_explicit_answer_leakage(self) -> None:
        for field in ("answer", "final_answer", "correct_option", "答案"):
            with self.subTest(field=field):
                leaked = lawyer_a_payload()
                leaked[field] = "B"
                with self.assertRaisesRegex(
                    StructuredOutputError, "answer-bearing fields are forbidden"
                ):
                    LawyerAOutput.from_text(json.dumps(leaked, ensure_ascii=False))


class DialogueTriggerTest(unittest.TestCase):
    def test_two_of_four_nei_triggers_at_boundary(self) -> None:
        output = B1Output.from_text(json.dumps(b1_payload(("NEI", "NEI", "SUPPORT", "REFUTE"))))
        self.assertTrue(should_request_clarification(output.verification, 0.6))

    def test_complete_high_confidence_verification_stops(self) -> None:
        output = B1Output.from_text(json.dumps(b1_payload(("NEI", "SUPPORT", "SUPPORT", "REFUTE"))))
        self.assertFalse(should_request_clarification(output.verification, 0.6))

    def test_low_b0_confidence_triggers(self) -> None:
        output = B1Output.from_text(json.dumps(b1_payload(("SUPPORT", "SUPPORT", "REFUTE", "REFUTE"))))
        self.assertTrue(should_request_clarification(output.verification, 0.599))


class RetryAndDialogueProtocolTest(unittest.TestCase):
    @patch("tools.legal_multi_agent_prompt_demo.chat")
    def test_invalid_b0_is_retried_without_guessing_from_text(self, mock_chat) -> None:
        mock_chat.side_effect = [
            "I think the answer is B",
            '```json\n{"initial_answer":"b","confidence":0.7}\n```',
        ]

        answer, confidence = run_lawyer_answer_b0_blind(object(), CONFIG, "question")

        self.assertEqual((answer, confidence), ("B", 0.7))
        self.assertEqual(mock_chat.call_count, 2)
        retry_messages = mock_chat.call_args_list[1].kwargs["messages"]
        self.assertTrue(
            any(
                message["role"] == "user" and "不符合严格JSON协议" in message["content"]
                for message in retry_messages
            )
        )

    @patch("tools.legal_multi_agent_prompt_demo.chat")
    def test_api_failure_after_retry_is_explicit(self, mock_chat) -> None:
        mock_chat.side_effect = ["[API_ERROR: timeout] first", "[API_ERROR: timeout] second"]

        with self.assertRaises(StructuredOutputError) as raised:
            run_lawyer_answer_b0_blind(object(), CONFIG, "question")

        self.assertEqual(raised.exception.agent, "lawyer_b0")
        self.assertEqual(raised.exception.attempts, 2)
        self.assertIn("second", raised.exception.raw_output)

    @patch("tools.legal_multi_agent_prompt_demo.chat")
    def test_initial_b1_and_additional_round_are_counted_separately(self, mock_chat) -> None:
        initial_judge = json.dumps(judge_payload())
        updated_judge = judge_payload()
        updated_judge["global_query"] = "updated-query"
        mock_chat.side_effect = [
            json.dumps(b1_payload(("NEI", "NEI", "SUPPORT", "REFUTE"))),
            json.dumps(updated_judge),
            json.dumps(b1_payload(("SUPPORT", "REFUTE", "SUPPORT", "REFUTE"))),
        ]

        answer, report, _ = run_lawyer_judge_dialogue(
            object(),
            object(),
            CONFIG,
            CONFIG,
            "ORIGINAL QUESTION",
            '{"structure":"LAWYER A"}',
            initial_judge,
            ["EVIDENCE DOCUMENT"],
            "A",
            0.8,
            [],
            max_rounds=2,
        )

        self.assertEqual(answer, "A")
        self.assertTrue(report["initial_b1_completed"])
        self.assertEqual(report["dialogue_rounds"], 1)
        self.assertFalse(report["dialogue_exhausted"])
        self.assertEqual(mock_chat.call_count, 3)

        judge_prompt = "\n".join(
            message["content"]
            for message in mock_chat.call_args_list[1].kwargs["messages"]
            if message["role"] == "user"
        )
        for expected in (
            "ORIGINAL QUESTION",
            '{"structure":"LAWYER A"}',
            "EVIDENCE DOCUMENT",
            initial_judge,
            "律师B上一轮核验报告",
        ):
            self.assertIn(expected, judge_prompt)

        second_b1_prompt = "\n".join(
            message["content"]
            for message in mock_chat.call_args_list[2].kwargs["messages"]
            if message["role"] == "user"
        )
        for expected in (
            "ORIGINAL QUESTION",
            '{"structure":"LAWYER A"}',
            "EVIDENCE DOCUMENT",
            "updated-query",
            "本轮澄清问题",
        ):
            self.assertIn(expected, second_b1_prompt)

    @patch("tools.legal_multi_agent_prompt_demo.chat")
    def test_initial_b1_does_not_increment_dialogue_rounds(self, mock_chat) -> None:
        mock_chat.return_value = json.dumps(
            b1_payload(("SUPPORT", "REFUTE", "SUPPORT", "REFUTE"))
        )

        _, report, _ = run_lawyer_judge_dialogue(
            object(),
            object(),
            CONFIG,
            CONFIG,
            "question",
            "structure",
            json.dumps(judge_payload()),
            [],
            "A",
            0.8,
            [],
        )

        self.assertEqual(report["dialogue_rounds"], 0)
        self.assertEqual(mock_chat.call_count, 1)


if __name__ == "__main__":
    unittest.main()
