"""Evaluation boundary and oracle-leak prevention for LegalMCQ."""

from __future__ import annotations

import re
from dataclasses import fields, is_dataclass
from datetime import date
from enum import Enum
from typing import Any, Mapping, Sequence

from .models import (
    EvaluationCase,
    EvaluationOracle,
    LegalAgentError,
    LegalAgentErrorCode,
    LegalQuestionRequest,
    QuestionType,
    SolveMode,
)
from .parser import parse_question_text

_FORBIDDEN_FIELDS = frozenset(
    {
        "golden_answers",
        "correct_option",
        "correct_answer",
        "reference_answer",
        "annotation_note",
        "oracle",
    }
)
_FORBIDDEN_TEXT = re.compile(
    r"(?i)(?<![A-Za-z0-9_])"
    r"(golden_answers|correct_option|correct_answer|reference_answer|annotation_note)"
    r"(?![A-Za-z0-9_])"
)


class LeakGuard:
    """Reject evaluation-only fields at every production-agent boundary."""

    @classmethod
    def assert_clean(cls, value: Any, *, location: str = "agent_input") -> None:
        findings: list[str] = []
        cls._scan(value, "$", findings)
        if findings:
            raise LegalAgentError(
                LegalAgentErrorCode.ORACLE_LEAKAGE,
                f"evaluation-only data reached {location}",
                details={"paths": sorted(set(findings))},
            )

    @classmethod
    def assert_prompt_clean(cls, prompts: Sequence[str]) -> None:
        findings = [
            f"prompt[{index}]"
            for index, prompt in enumerate(prompts)
            if _FORBIDDEN_TEXT.search(prompt)
        ]
        if findings:
            raise LegalAgentError(
                LegalAgentErrorCode.ORACLE_LEAKAGE,
                "evaluation-only field name reached a model prompt",
                details={"paths": findings},
            )

    @classmethod
    def _scan(cls, value: Any, path: str, findings: list[str]) -> None:
        if is_dataclass(value) and not isinstance(value, type):
            for item in fields(value):
                name = item.name.lower()
                child = f"{path}.{item.name}"
                if name in _FORBIDDEN_FIELDS:
                    findings.append(child)
                cls._scan(getattr(value, item.name), child, findings)
            return
        if isinstance(value, Mapping):
            for raw_key, item in value.items():
                name = str(raw_key).strip().lower()
                child = f"{path}.{raw_key}"
                if name in _FORBIDDEN_FIELDS:
                    findings.append(child)
                cls._scan(item, child, findings)
            return
        if isinstance(value, (list, tuple, set, frozenset)):
            for index, item in enumerate(value):
                cls._scan(item, f"{path}[{index}]", findings)
            return
        if isinstance(value, Enum) or value is None:
            return


class BenchmarkRecordAdapter:
    """Split a raw benchmark row into an agent request and a private oracle."""

    @staticmethod
    def adapt(
        record: Mapping[str, Any],
        *,
        source_dataset: str | None = None,
        mode: SolveMode = SolveMode.CLOSED_BOOK,
        jurisdiction: str = "CN",
        as_of_date: date | None = None,
    ) -> EvaluationCase:
        if not isinstance(record, Mapping):
            raise LegalAgentError(
                LegalAgentErrorCode.INVALID_REQUEST,
                "benchmark record must be an object",
            )
        question = record.get("question")
        if not isinstance(question, str) or not question.strip():
            raise LegalAgentError(
                LegalAgentErrorCode.INVALID_REQUEST,
                "benchmark record has no question",
            )
        raw_answers = record.get("golden_answers")
        if (
            not isinstance(raw_answers, Sequence)
            or isinstance(raw_answers, (str, bytes))
            or not raw_answers
        ):
            raise LegalAgentError(
                LegalAgentErrorCode.INVALID_REQUEST,
                "benchmark record has no valid answer labels",
            )
        answers = frozenset(str(value).strip().upper() for value in raw_answers)
        metadata = record.get("meta_data")
        metadata = metadata if isinstance(metadata, Mapping) else {}
        identifier = str(record.get("id", metadata.get("original_index", ""))).strip()
        request = parse_question_text(
            question,
            question_id=identifier or None,
            mode=mode,
            jurisdiction=jurisdiction,
            as_of_date=as_of_date,
            question_type=(
                QuestionType.MULTIPLE_CHOICE
                if len(answers) > 1
                else QuestionType.SINGLE_CHOICE
            ),
        )
        oracle = EvaluationOracle(
            golden_answers=answers,
            source_dataset=source_dataset,
            original_index=(
                int(metadata["original_index"])
                if isinstance(metadata.get("original_index"), int)
                else None
            ),
            annotation_note=(
                str(metadata["annotation_note"])
                if metadata.get("annotation_note") is not None
                else None
            ),
        )
        LeakGuard.assert_clean(request, location="adapted request")
        return EvaluationCase(request=request, oracle=oracle)


def request_only(case: EvaluationCase) -> LegalQuestionRequest:
    """Make the evaluator-to-agent narrowing explicit and auditable."""
    LeakGuard.assert_clean(case.request, location="evaluator handoff")
    return case.request
