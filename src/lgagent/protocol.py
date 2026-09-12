"""Strict structured-output contracts for the corrected LGAgent baseline."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import date
from typing import Any, Mapping

OPTIONS = frozenset({"A", "B", "C", "D"})
VERIFICATION_STATUSES = frozenset({"SUPPORT", "REFUTE", "NEI"})
_ANSWER_LEAK_FIELDS = frozenset(
    {
        "answer",
        "final_answer",
        "initial_answer",
        "correct_answer",
        "correct_option",
        "best_option",
        "prediction",
        "答案",
        "正确答案",
    }
)
_FENCED_JSON = re.compile(
    r"\A\s*```(?:json)?[ \t]*\r?\n(?P<body>.*?)\r?\n```[ \t]*\s*\Z",
    re.IGNORECASE | re.DOTALL,
)


class StructuredOutputError(ValueError):
    """Raised when an agent response cannot satisfy its output contract."""

    def __init__(
        self,
        agent: str,
        message: str,
        *,
        attempts: int = 1,
        raw_output: str = "",
    ) -> None:
        super().__init__(f"{agent}: {message}")
        self.agent = agent
        self.attempts = attempts
        self.raw_output = raw_output


def extract_json_object(text: str) -> dict[str, Any]:
    """Parse a JSON object, optionally enclosed by one complete Markdown fence."""
    if not isinstance(text, str):
        raise StructuredOutputError("json", "output must be a string")

    candidate = text.strip()
    fenced = _FENCED_JSON.fullmatch(text)
    if candidate.startswith("```"):
        if fenced is None:
            raise StructuredOutputError("json", "malformed Markdown JSON fence", raw_output=text)
        candidate = fenced.group("body").strip()

    try:
        value = json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise StructuredOutputError(
            "json",
            f"invalid JSON at line {exc.lineno}, column {exc.colno}",
            raw_output=text,
        ) from exc
    if not isinstance(value, dict):
        raise StructuredOutputError("json", "top-level JSON value must be an object", raw_output=text)
    return value


def normalize_option_label(value: Any, field: str) -> str:
    if not isinstance(value, str):
        raise StructuredOutputError("schema", f"{field} must be a string")
    normalized = value.strip().upper()
    if normalized not in OPTIONS:
        raise StructuredOutputError("schema", f"{field} must be one of A, B, C, D")
    return normalized


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise StructuredOutputError("schema", f"{field} must be an object")
    return value


def _string(value: Any, field: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        qualifier = "a string" if allow_empty else "a non-empty string"
        raise StructuredOutputError("schema", f"{field} must be {qualifier}")
    return value.strip()


def _string_list(value: Any, field: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise StructuredOutputError("schema", f"{field} must be an array")
    return tuple(_string(item, f"{field}[{index}]") for index, item in enumerate(value))


def _number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise StructuredOutputError("schema", f"{field} must be a number")
    result = float(value)
    if not 0.0 <= result <= 1.0:
        raise StructuredOutputError("schema", f"{field} must be between 0 and 1")
    return result


def _option_map(value: Any, field: str) -> dict[str, Any]:
    raw = _mapping(value, field)
    normalized: dict[str, Any] = {}
    for key, item in raw.items():
        option = normalize_option_label(key, f"{field} key")
        if option in normalized:
            raise StructuredOutputError("schema", f"{field} contains duplicate option {option}")
        normalized[option] = item
    if set(normalized) != OPTIONS:
        missing = sorted(OPTIONS - set(normalized))
        extra = sorted(set(normalized) - OPTIONS)
        detail = f"missing={missing}, extra={extra}"
        raise StructuredOutputError("schema", f"{field} must contain exactly A-D ({detail})")
    return normalized


@dataclass(frozen=True)
class OptionClaim:
    claim: str
    elements: tuple[str, ...] = ()
    possible_exceptions: tuple[str, ...] = ()


@dataclass(frozen=True)
class LawyerAOutput:
    task_type: str
    question_focus: str
    legal_domain: str
    jurisdiction: str
    case_date: str | None
    facts: tuple[Mapping[str, Any], ...]
    option_claims: dict[str, OptionClaim]
    option_keywords: dict[str, tuple[str, ...]]
    trap_signals: tuple[str, ...]
    unknowns: tuple[str, ...]

    @classmethod
    def from_text(cls, text: str) -> "LawyerAOutput":
        data = extract_json_object(text)
        leaked_fields = sorted(
            str(key) for key in data if str(key).strip().lower() in _ANSWER_LEAK_FIELDS
        )
        if leaked_fields:
            raise StructuredOutputError(
                "lawyer_a",
                f"answer-bearing fields are forbidden: {leaked_fields}",
            )
        claims: dict[str, OptionClaim] = {}
        for option, value in _option_map(
            data.get("option_claims"), "option_claims"
        ).items():
            if isinstance(value, Mapping):
                claims[option] = OptionClaim(
                    claim=_string(value.get("claim"), f"option_claims.{option}.claim"),
                    elements=_string_list(
                        value.get("elements", []),
                        f"option_claims.{option}.elements",
                    ),
                    possible_exceptions=_string_list(
                        value.get("possible_exceptions", []),
                        f"option_claims.{option}.possible_exceptions",
                    ),
                )
            else:
                # Task 2-era responses used a plain claim string.
                claims[option] = OptionClaim(
                    claim=_string(value, f"option_claims.{option}")
                )
        keywords = {
            option: _string_list(value, f"option_keywords.{option}")
            for option, value in _option_map(data.get("option_keywords"), "option_keywords").items()
        }
        raw_case_date = data.get("case_date")
        case_date = None
        null_date_aliases = {"", "null", "none", "n/a", "不适用", "未知", "无"}
        normalized_case_date = (
            raw_case_date.strip().lower()
            if isinstance(raw_case_date, str)
            else raw_case_date
        )
        is_null_date = raw_case_date is None or (
            isinstance(normalized_case_date, str)
            and normalized_case_date in null_date_aliases
        )
        if not is_null_date:
            case_date = _string(raw_case_date, "case_date")
            try:
                date.fromisoformat(case_date)
            except ValueError as exc:
                raise StructuredOutputError(
                    "schema", "case_date must be an ISO date (YYYY-MM-DD)"
                ) from exc
        raw_facts = data.get("facts", [])
        if not isinstance(raw_facts, list):
            raise StructuredOutputError("schema", "facts must be an array")
        facts: list[Mapping[str, Any]] = []
        for index, value in enumerate(raw_facts):
            if isinstance(value, str):
                facts.append(
                    {
                        "fact_id": f"F{index + 1}",
                        "text": _string(value, f"facts[{index}]"),
                        "legally_relevant": True,
                    }
                )
                continue
            fact = _mapping(value, f"facts[{index}]")
            fact_id = _string(
                fact.get("fact_id", f"F{index + 1}"),
                f"facts[{index}].fact_id",
            )
            legally_relevant = fact.get("legally_relevant", True)
            if not isinstance(legally_relevant, bool):
                raise StructuredOutputError(
                    "schema", f"facts[{index}].legally_relevant must be a boolean"
                )
            facts.append(
                {
                    **dict(fact),
                    "fact_id": fact_id,
                    "text": _string(fact.get("text"), f"facts[{index}].text"),
                    "legally_relevant": legally_relevant,
                }
            )
        return cls(
            task_type=_string(data.get("task_type"), "task_type"),
            question_focus=_string(data.get("question_focus"), "question_focus"),
            legal_domain=_string(data.get("legal_domain"), "legal_domain"),
            jurisdiction=_string(
                data.get("jurisdiction", ""), "jurisdiction", allow_empty=True
            ),
            case_date=case_date,
            facts=tuple(facts),
            option_claims=claims,
            option_keywords=keywords,
            trap_signals=_string_list(data.get("trap_signals"), "trap_signals"),
            unknowns=_string_list(data.get("unknowns"), "unknowns"),
        )


@dataclass(frozen=True)
class EvidenceRequirement:
    support: str
    refute: str


@dataclass(frozen=True)
class JudgeOutput:
    need_retrieval: bool
    global_query: str
    option_queries: dict[str, str]
    evidence_requirements: dict[str, EvidenceRequirement]
    counterfactual_focus: str
    stop_rule: str

    @classmethod
    def from_text(cls, text: str) -> "JudgeOutput":
        data = extract_json_object(text)
        need_retrieval = data.get("need_retrieval")
        if not isinstance(need_retrieval, bool):
            raise StructuredOutputError("schema", "need_retrieval must be a boolean")
        queries = {
            option: _string(value, f"option_queries.{option}")
            for option, value in _option_map(data.get("option_queries"), "option_queries").items()
        }
        requirements: dict[str, EvidenceRequirement] = {}
        for option, value in _option_map(
            data.get("evidence_requirements"), "evidence_requirements"
        ).items():
            requirement = _mapping(value, f"evidence_requirements.{option}")
            requirements[option] = EvidenceRequirement(
                support=_string(requirement.get("support"), f"evidence_requirements.{option}.support"),
                refute=_string(requirement.get("refute"), f"evidence_requirements.{option}.refute"),
            )
        return cls(
            need_retrieval=need_retrieval,
            global_query=_string(data.get("global_query"), "global_query"),
            option_queries=queries,
            evidence_requirements=requirements,
            counterfactual_focus=_string(data.get("counterfactual_focus"), "counterfactual_focus"),
            stop_rule=_string(data.get("stop_rule"), "stop_rule"),
        )


@dataclass(frozen=True)
class B0Output:
    initial_answer: str
    confidence: float

    @classmethod
    def from_text(cls, text: str) -> "B0Output":
        data = extract_json_object(text)
        return cls(
            initial_answer=normalize_option_label(data.get("initial_answer"), "initial_answer"),
            confidence=_number(data.get("confidence"), "confidence"),
        )


@dataclass(frozen=True)
class OptionVerification:
    status: str
    score: float
    reason: str


@dataclass(frozen=True)
class B1Output:
    final_answer: str
    verification: dict[str, OptionVerification]
    initial_answer: str
    initial_confidence: float
    reasoning: str
    raw: dict[str, Any]

    @classmethod
    def from_text(cls, text: str) -> "B1Output":
        data = extract_json_object(text)
        verification: dict[str, OptionVerification] = {}
        for option, value in _option_map(data.get("verification"), "verification").items():
            item = _mapping(value, f"verification.{option}")
            status = _string(item.get("status"), f"verification.{option}.status").upper()
            if status not in VERIFICATION_STATUSES:
                raise StructuredOutputError(
                    "schema",
                    f"verification.{option}.status must be SUPPORT, REFUTE, or NEI",
                )
            verification[option] = OptionVerification(
                status=status,
                score=_number(item.get("score"), f"verification.{option}.score"),
                reason=_string(item.get("reason"), f"verification.{option}.reason"),
            )
        normalized = dict(data)
        normalized["final_answer"] = normalize_option_label(data.get("final_answer"), "final_answer")
        normalized["initial_answer"] = normalize_option_label(
            data.get("initial_answer"), "initial_answer"
        )
        normalized["initial_confidence"] = _number(
            data.get("initial_confidence"), "initial_confidence"
        )
        normalized["verification"] = {
            option: {
                "status": item.status,
                "score": item.score,
                "reason": item.reason,
            }
            for option, item in verification.items()
        }
        normalized["reasoning"] = _string(data.get("reasoning"), "reasoning")
        return cls(
            final_answer=normalized["final_answer"],
            verification=verification,
            initial_answer=normalized["initial_answer"],
            initial_confidence=normalized["initial_confidence"],
            reasoning=normalized["reasoning"],
            raw=normalized,
        )


def should_request_clarification(
    verification: Mapping[str, OptionVerification],
    calibrated_confidence: float,
) -> bool:
    if not verification:
        raise StructuredOutputError("dialogue", "verification cannot be empty")
    confidence = _number(calibrated_confidence, "calibrated_confidence")
    nei_count = sum(item.status == "NEI" for item in verification.values())
    return nei_count / len(verification) >= 0.5 or confidence < 0.6
