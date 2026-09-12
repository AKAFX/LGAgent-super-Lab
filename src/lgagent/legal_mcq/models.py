"""Typed contracts for the isolated LegalMCQ agent workflow."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from enum import Enum
from typing import Any, Mapping, Sequence

from ..protocol import StructuredOutputError, extract_json_object
from ..serialization import to_jsonable
from ..trace import RunTrace

_LABEL = re.compile(r"^[A-Z]$")
_VERDICTS = frozenset({"supported", "contradicted", "uncertain"})
_ERROR_CODE = re.compile(r"^[A-Z][A-Z0-9_]*$")
SOLVER_PROTOCOL_VERSION = "solver-v3"
SOLVER_PROTOCOL_V2_VERSION = "solver-v2"
CONTROLLER_PROTOCOL_VERSION = "controller-v3"
CONTROLLER_PROTOCOL_V2_VERSION = "controller-v2"
VERIFIER_PROTOCOL_VERSION = "verifier-v2"


class SolveMode(str, Enum):
    CLOSED_BOOK = "closed_book"
    OPEN_BOOK = "open_book"
    AUTO = "auto"


class QuestionType(str, Enum):
    SINGLE_CHOICE = "single_choice"
    MULTIPLE_CHOICE = "multiple_choice"


class AnswerStatus(str, Enum):
    COMPLETED = "completed"
    PARTIAL = "partial"
    BLOCKED = "blocked"


class LegalAgentErrorCode(str, Enum):
    INVALID_REQUEST = "invalid_request"
    INVALID_QUESTION_FORMAT = "invalid_question_format"
    UNSUPPORTED_QUESTION_TYPE = "unsupported_question_type"
    RETRIEVAL_REQUIRED_BUT_DISABLED = "retrieval_required_but_disabled"
    NO_AUTHORITATIVE_SOURCE = "no_authoritative_source"
    MODEL_PROVIDER_ERROR = "model_provider_error"
    MODEL_SCHEMA_ERROR = "model_schema_error"
    AMBIGUOUS_OPTIONS = "ambiguous_options"
    ORACLE_LEAKAGE = "oracle_leakage"
    INTERNAL_ERROR = "internal_error"


class LegalAgentError(RuntimeError):
    """Stable error returned by the LegalMCQ boundary."""

    def __init__(
        self,
        code: LegalAgentErrorCode,
        message: str,
        *,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        self.code = code
        self.details = dict(details or {})
        super().__init__(f"{code.value}: {message}")


def _text(value: Any, field_name: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise StructuredOutputError("legal_mcq", f"{field_name} must be a string")
    result = value.strip()
    if not allow_empty and not result:
        raise StructuredOutputError(
            "legal_mcq", f"{field_name} must be a non-empty string"
        )
    return result


def _confidence(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise StructuredOutputError("legal_mcq", f"{field_name} must be a number")
    result = float(value)
    if not 0.0 <= result <= 1.0:
        raise StructuredOutputError(
            "legal_mcq", f"{field_name} must be between 0 and 1"
        )
    return result


def _string_tuple(value: Any, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise StructuredOutputError("legal_mcq", f"{field_name} must be an array")
    return tuple(_text(item, f"{field_name}[{index}]") for index, item in enumerate(value))


def _label(value: Any, field_name: str) -> str:
    result = _text(value, field_name).upper()
    if _LABEL.fullmatch(result) is None:
        raise StructuredOutputError(
            "legal_mcq", f"{field_name} must be one uppercase option label"
        )
    return result


def _exact_keys(
    value: Mapping[str, Any],
    expected: set[str],
    field_name: str,
    *,
    stage: str = "legal_mcq_solver",
) -> None:
    observed = set(value)
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise StructuredOutputError(
            stage,
            f"{field_name} fields mismatch; missing={missing}, extra={extra}",
        )


def _answer_key_paths(value: Any, path: str = "$") -> tuple[str, ...]:
    forbidden = {
        "answer",
        "final_answer",
        "selected_options",
        "correct_option",
        "prediction",
    }
    findings: list[str] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            child = f"{path}.{key}"
            if str(key).strip().lower() in forbidden:
                findings.append(child)
            findings.extend(_answer_key_paths(item, child))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            findings.extend(_answer_key_paths(item, f"{path}[{index}]"))
    return tuple(findings)


@dataclass(frozen=True)
class LegalOption:
    label: str
    text: str

    def __post_init__(self) -> None:
        if _LABEL.fullmatch(self.label) is None:
            raise ValueError("option label must be one uppercase ASCII letter")
        if not self.text.strip():
            raise ValueError("option text cannot be empty")


@dataclass(frozen=True)
class LegalQuestionRequest:
    question_id: str
    stem: str
    options: tuple[LegalOption, ...]
    question_type: QuestionType = QuestionType.SINGLE_CHOICE
    jurisdiction: str = "CN"
    as_of_date: date | None = None
    mode: SolveMode = SolveMode.CLOSED_BOOK
    language: str = "zh-CN"
    explain_all_options: bool = True

    def __post_init__(self) -> None:
        if not self.question_id.strip():
            raise ValueError("question_id cannot be empty")
        if not self.stem.strip():
            raise ValueError("stem cannot be empty")
        if len(self.options) < 2:
            raise ValueError("at least two options are required")
        labels = [option.label for option in self.options]
        if len(set(labels)) != len(labels):
            raise ValueError("option labels must be unique")
        expected = [chr(ord("A") + index) for index in range(len(labels))]
        if labels != expected:
            raise ValueError(f"option labels must be consecutive: {expected}")
        if not self.jurisdiction.strip():
            raise ValueError("jurisdiction cannot be empty")

    def prompt_payload(self) -> dict[str, Any]:
        return {
            "question_id": self.question_id,
            "stem": self.stem,
            "options": [
                {"label": option.label, "text": option.text}
                for option in self.options
            ],
            "question_type": self.question_type.value,
            "jurisdiction": self.jurisdiction,
            "as_of_date": self.as_of_date.isoformat() if self.as_of_date else None,
            "language": self.language,
        }


@dataclass(frozen=True)
class EvaluationOracle:
    golden_answers: frozenset[str]
    source_dataset: str | None = None
    original_index: int | None = None
    annotation_note: str | None = None

    def __post_init__(self) -> None:
        if not self.golden_answers:
            raise ValueError("golden_answers cannot be empty")
        if any(_LABEL.fullmatch(label) is None for label in self.golden_answers):
            raise ValueError("golden_answers contains an invalid label")


@dataclass(frozen=True)
class EvaluationCase:
    request: LegalQuestionRequest
    oracle: EvaluationOracle


@dataclass(frozen=True)
class ParsedLegalQuestion:
    request: LegalQuestionRequest
    asks_for_incorrect_option: bool
    date_source: str
    legal_domains: tuple[str, ...] = ()
    named_laws: tuple[str, ...] = ()
    parse_warnings: tuple[str, ...] = ()

    @property
    def option_labels(self) -> tuple[str, ...]:
        return tuple(option.label for option in self.request.options)

    def prompt_payload(self) -> dict[str, Any]:
        payload = self.request.prompt_payload()
        payload.update(
            {
                "asks_for_incorrect_option": self.asks_for_incorrect_option,
                "date_source": self.date_source,
                "parse_warnings": list(self.parse_warnings),
            }
        )
        return payload


@dataclass(frozen=True)
class MaterialFact:
    fact_id: str
    text: str


@dataclass(frozen=True)
class LegalIssue:
    issue_id: str
    description: str
    decisive_for_options: tuple[str, ...]
    candidate_laws: tuple[str, ...] = ()


@dataclass(frozen=True)
class ControllerPlan:
    facts: tuple[MaterialFact, ...]
    issues: tuple[LegalIssue, ...]
    option_claims: Mapping[str, tuple[str, ...]]
    legal_domains: tuple[str, ...]
    named_laws: tuple[str, ...]
    trap_checks: tuple[str, ...]
    verification_checklist: tuple[str, ...]
    raw: Mapping[str, Any] = field(repr=False)

    @staticmethod
    def response_format(option_labels: tuple[str, ...]) -> dict[str, Any]:
        strings = {"type": "array", "items": {"type": "string"}}
        properties = {
            "protocol_version": {"type": "string", "enum": [CONTROLLER_PROTOCOL_VERSION]},
            "facts": strings,
            "issues": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "description": {"type": "string"},
                        "options": {
                            "type": "array",
                            "items": {"type": "string", "enum": list(option_labels)},
                        },
                    },
                    "required": ["description", "options"],
                },
            },
            "checks": strings,
        }
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "legal_mcq_controller_v3",
                "strict": True,
                "schema": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": properties,
                    "required": list(properties),
                },
            },
        }

    @staticmethod
    def _normalize_compact(
        data: Mapping[str, Any],
        *,
        version: str,
        has_option_claims: bool,
    ) -> dict[str, Any]:
        expected = {"protocol_version", "facts", "issues", "checks"}
        if has_option_claims:
            expected.add("option_claims")
        _exact_keys(
            data,
            expected,
            version,
            stage="legal_mcq_controller",
        )
        if data["protocol_version"] != version:
            raise StructuredOutputError(
                "legal_mcq_controller", "unsupported controller version"
            )
        facts = _string_tuple(data["facts"], "facts")
        checks = _string_tuple(data["checks"], "checks")
        raw_issues = data["issues"]
        if not isinstance(raw_issues, list):
            raise StructuredOutputError("legal_mcq_controller", "issues must be an array")
        issues = []
        for index, issue in enumerate(raw_issues, start=1):
            if not isinstance(issue, Mapping):
                raise StructuredOutputError(
                    "legal_mcq_controller", "issue must be an object"
                )
            _exact_keys(
                issue, {"description", "options"}, "controller issue",
                stage="legal_mcq_controller",
            )
            options = _string_tuple(issue["options"], "issue options")
            if not options or len(set(options)) != len(options):
                raise StructuredOutputError(
                    "legal_mcq_controller",
                    "issue options must be unique and non-empty",
                )
            issues.append(
                {
                    "issue_id": f"I{index}",
                    "description": issue["description"],
                    "decisive_for_options": list(options),
                }
            )
        claims = {}
        if has_option_claims:
            raw_claims = data["option_claims"]
            if not isinstance(raw_claims, Mapping):
                raise StructuredOutputError(
                    "legal_mcq_controller", "option_claims must be an object"
                )
            for label, values in raw_claims.items():
                if label != _label(label, "option label"):
                    raise StructuredOutputError(
                        "legal_mcq_controller", "option label must be uppercase"
                    )
                claims[label] = [
                    f"{label}{index} {text}"
                    for index, text in enumerate(
                        _string_tuple(values, f"claims.{label}"), start=1
                    )
                ]
        return {
            "facts": [
                {"fact_id": f"F{i}", "text": text}
                for i, text in enumerate(facts, start=1)
            ],
            "issues": issues,
            "option_claims": claims,
            "verification_checklist": list(checks),
        }

    @staticmethod
    def _normalize_v3(data: Mapping[str, Any]) -> dict[str, Any]:
        return ControllerPlan._normalize_compact(
            data,
            version=CONTROLLER_PROTOCOL_VERSION,
            has_option_claims=False,
        )

    @staticmethod
    def _normalize_v2(data: Mapping[str, Any]) -> dict[str, Any]:
        return ControllerPlan._normalize_compact(
            data,
            version=CONTROLLER_PROTOCOL_V2_VERSION,
            has_option_claims=True,
        )

    def bind_option_claims(
        self,
        options: Sequence[LegalOption],
    ) -> "ControllerPlan":
        """Bind one canonical claim per option from the trusted request."""
        claims = {
            option.label: (f"{option.label}1 {option.text.strip()}",)
            for option in options
        }
        if len(claims) != len(options):
            raise ValueError("option labels must be unique before claim binding")
        return ControllerPlan(
            facts=self.facts,
            issues=self.issues,
            option_claims=claims,
            legal_domains=self.legal_domains,
            named_laws=self.named_laws,
            trap_checks=self.trap_checks,
            verification_checklist=self.verification_checklist,
            raw=self.raw,
        )

    def prompt_payload(self) -> dict[str, Any]:
        if self.raw.get("protocol_version") not in {
            CONTROLLER_PROTOCOL_VERSION,
            CONTROLLER_PROTOCOL_V2_VERSION,
        }:
            payload = dict(self.raw)
            payload["option_claims"] = {
                label: list(claims)
                for label, claims in self.option_claims.items()
            }
            payload["claim_binding"] = "deterministic-option-text-v1"
            return payload
        return {
            "facts": [fact.text for fact in self.facts],
            "issues": [
                {
                    "issue_id": issue.issue_id,
                    "description": issue.description,
                    "decisive_for_options": list(issue.decisive_for_options),
                }
                for issue in self.issues
            ],
            "option_claims": {
                label: list(claims) for label, claims in self.option_claims.items()
            },
            "claim_binding": "deterministic-option-text-v1",
            "checks": list(self.verification_checklist),
        }

    @classmethod
    def from_text(cls, text: str) -> "ControllerPlan":
        source = extract_json_object(text)
        leaked = _answer_key_paths(source)
        if leaked:
            raise StructuredOutputError(
                "legal_mcq_controller",
                f"answer-bearing fields are forbidden: {leaked}",
            )
        version = source.get("protocol_version")
        if version == CONTROLLER_PROTOCOL_VERSION:
            data = cls._normalize_v3(source)
        elif version == CONTROLLER_PROTOCOL_V2_VERSION:
            data = cls._normalize_v2(source)
        elif version is not None:
            raise StructuredOutputError(
                "legal_mcq_controller", "unsupported controller version"
            )
        else:
            data = source

        raw_facts = data.get("facts")
        if not isinstance(raw_facts, list) or not raw_facts:
            raise StructuredOutputError(
                "legal_mcq_controller", "facts must be a non-empty array"
            )
        facts = []
        for index, raw in enumerate(raw_facts):
            if not isinstance(raw, Mapping):
                raise StructuredOutputError(
                    "legal_mcq_controller", f"facts[{index}] must be an object"
                )
            facts.append(
                MaterialFact(
                    fact_id=_text(
                        raw.get("fact_id", f"F{index + 1}"),
                        f"facts[{index}].fact_id",
                    ),
                    text=_text(raw.get("text"), f"facts[{index}].text"),
                )
            )

        raw_issues = data.get("issues")
        if not isinstance(raw_issues, list) or not raw_issues:
            raise StructuredOutputError(
                "legal_mcq_controller", "issues must be a non-empty array"
            )
        issues = []
        for index, raw in enumerate(raw_issues):
            if not isinstance(raw, Mapping):
                raise StructuredOutputError(
                    "legal_mcq_controller", f"issues[{index}] must be an object"
                )
            issues.append(
                LegalIssue(
                    issue_id=_text(
                        raw.get("issue_id", f"I{index + 1}"),
                        f"issues[{index}].issue_id",
                    ),
                    description=_text(
                        raw.get("description"), f"issues[{index}].description"
                    ),
                    decisive_for_options=tuple(
                        _label(item, f"issues[{index}].decisive_for_options")
                        for item in raw.get("decisive_for_options", [])
                    ),
                    candidate_laws=_string_tuple(
                        raw.get("candidate_laws", []),
                        f"issues[{index}].candidate_laws",
                    ),
                )
            )

        raw_claims = data.get("option_claims", {})
        if not isinstance(raw_claims, Mapping):
            raise StructuredOutputError(
                "legal_mcq_controller", "option_claims must be an object"
            )
        if version != CONTROLLER_PROTOCOL_VERSION and not raw_claims:
            raise StructuredOutputError(
                "legal_mcq_controller", "option_claims must be a non-empty object"
            )
        claims: dict[str, tuple[str, ...]] = {}
        for raw_key, raw_value in raw_claims.items():
            label = _label(raw_key, "option_claims key")
            values = _string_tuple(raw_value, f"option_claims.{label}")
            if not values:
                raise StructuredOutputError(
                    "legal_mcq_controller",
                    f"option_claims.{label} cannot be empty",
                )
            claims[label] = values

        return cls(
            facts=tuple(facts),
            issues=tuple(issues),
            option_claims=claims,
            legal_domains=_string_tuple(data.get("legal_domains", []), "legal_domains"),
            named_laws=_string_tuple(data.get("named_laws", []), "named_laws"),
            trap_checks=_string_tuple(data.get("trap_checks", []), "trap_checks"),
            verification_checklist=_string_tuple(
                data.get("verification_checklist", []),
                "verification_checklist",
            ),
            raw=source,
        )


@dataclass(frozen=True)
class LegalEvidence:
    evidence_id: str
    title: str
    url: str
    publisher: str
    source_type: str
    authority_level: int
    quote: str
    law_name: str | None = None
    article_number: str | None = None
    effective_from: date | None = None
    effective_until: date | None = None
    content_hash: str = ""

    def __post_init__(self) -> None:
        if not self.evidence_id.strip() or not self.quote.strip():
            raise ValueError("evidence_id and quote are required")
        if not self.url.startswith(("https://", "http://")):
            raise ValueError("evidence URL must use HTTP or HTTPS")
        if not 0 <= self.authority_level <= 5:
            raise ValueError("authority_level must be between 0 and 5")
        expected_hash = hashlib.sha256(self.quote.encode("utf-8")).hexdigest()
        if self.content_hash and self.content_hash != expected_hash:
            raise ValueError("content_hash does not match evidence quote")
        if not self.content_hash:
            object.__setattr__(self, "content_hash", expected_hash)

    def covers(self, as_of_date: date) -> bool:
        if self.effective_from and as_of_date < self.effective_from:
            return False
        return not self.effective_until or as_of_date <= self.effective_until


@dataclass(frozen=True)
class ClaimAssessment:
    claim_id: str
    text: str
    verdict: str
    reason: str
    rule: str
    evidence_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class OptionAssessment:
    label: str
    claims: tuple[ClaimAssessment, ...]
    verdict: str
    decisive_reason: str
    confidence: float


@dataclass(frozen=True)
class IssueAnalysis:
    issue_id: str
    issue: str
    application: str
    conclusion: str


@dataclass(frozen=True)
class SolverDecision:
    selected_options: tuple[str, ...]
    option_assessments: tuple[OptionAssessment, ...]
    issue_analyses: tuple[IssueAnalysis, ...]
    rationale: str
    confidence: float
    raw: Mapping[str, Any] = field(repr=False)

    @staticmethod
    def json_schema() -> dict[str, Any]:
        verdict = {
            "type": "string",
            "enum": ["supported", "contradicted", "uncertain"],
        }
        label = {"type": "string"}
        return {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "protocol_version": {
                    "type": "string",
                    "enum": [SOLVER_PROTOCOL_VERSION],
                },
                "selected_options": {
                    "type": "array",
                    "items": label,
                },
                "options": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "label": label,
                            "claims": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "additionalProperties": False,
                                    "properties": {
                                        "claim_id": {
                                            "type": "string",
                                        },
                                        "verdict": verdict,
                                        "reason": {
                                            "type": "string",
                                        },
                                        "evidence_ids": {
                                            "type": "array",
                                            "items": {
                                                "type": "string",
                                            },
                                        },
                                    },
                                    "required": [
                                        "claim_id",
                                        "verdict",
                                        "reason",
                                        "evidence_ids",
                                    ],
                                },
                            },
                            "verdict": verdict,
                        },
                        "required": [
                            "label",
                            "claims",
                            "verdict",
                        ],
                    },
                },
                "confidence": {"type": "number"},
            },
            "required": [
                "protocol_version",
                "selected_options",
                "options",
                "confidence",
            ],
        }

    @staticmethod
    def response_format() -> dict[str, Any]:
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "legal_mcq_solver_v3",
                "strict": True,
                "schema": SolverDecision.json_schema(),
            },
        }

    @staticmethod
    def _normalize_v2(data: Mapping[str, Any]) -> dict[str, Any]:
        _exact_keys(
            data,
            {
                "protocol_version",
                "selected_options",
                "issues",
                "options",
                "rationale",
                "confidence",
            },
            "solver-v2",
        )
        version = data.get("protocol_version")
        if version != SOLVER_PROTOCOL_V2_VERSION:
            raise StructuredOutputError(
                "legal_mcq_solver",
                f"protocol_version must be {SOLVER_PROTOCOL_V2_VERSION}",
            )
        raw_options = data.get("options")
        if not isinstance(raw_options, list):
            raise StructuredOutputError(
                "legal_mcq_solver", "options must be an array"
            )
        options = []
        for option_index, raw_option in enumerate(raw_options):
            if not isinstance(raw_option, Mapping):
                raise StructuredOutputError(
                    "legal_mcq_solver",
                    f"options[{option_index}] must be an object",
                )
            _exact_keys(
                raw_option,
                {"label", "claims", "verdict"},
                f"options[{option_index}]",
            )
            raw_claims = raw_option.get("claims")
            if not isinstance(raw_claims, list):
                raise StructuredOutputError(
                    "legal_mcq_solver",
                    f"options[{option_index}].claims must be an array",
                )
            claims = []
            for claim_index, raw_claim in enumerate(raw_claims):
                if not isinstance(raw_claim, Mapping):
                    raise StructuredOutputError(
                        "legal_mcq_solver",
                        f"options[{option_index}].claims[{claim_index}] "
                        "must be an object",
                    )
                _exact_keys(
                    raw_claim,
                    {"claim_id", "verdict", "reason", "evidence_ids"},
                    f"options[{option_index}].claims[{claim_index}]",
                )
                claims.append(
                    {
                        "claim_id": raw_claim.get("claim_id"),
                        "text": raw_claim.get("claim_id"),
                        "verdict": raw_claim.get("verdict"),
                        "reason": raw_claim.get("reason"),
                        "rule": "",
                        "evidence_ids": raw_claim.get("evidence_ids"),
                    }
                )
            options.append(
                {
                    "label": raw_option.get("label"),
                    "claims": claims,
                    "verdict": raw_option.get("verdict"),
                    "decisive_reason": "；".join(
                        str(claim.get("reason", "")).strip()
                        for claim in raw_claims
                        if str(claim.get("reason", "")).strip()
                    ),
                    "confidence": data.get("confidence"),
                }
            )
        raw_issues = data.get("issues")
        if not isinstance(raw_issues, list):
            raise StructuredOutputError(
                "legal_mcq_solver", "issues must be an array"
            )
        issues = []
        for index, raw_issue in enumerate(raw_issues):
            if not isinstance(raw_issue, Mapping):
                raise StructuredOutputError(
                    "legal_mcq_solver", f"issues[{index}] must be an object"
                )
            _exact_keys(
                raw_issue,
                {"issue_id", "analysis", "conclusion"},
                f"issues[{index}]",
            )
            issue_id = raw_issue.get("issue_id")
            issues.append(
                {
                    "issue_id": issue_id,
                    "issue": issue_id,
                    "application": raw_issue.get("analysis"),
                    "conclusion": raw_issue.get("conclusion"),
                }
            )
        return {
            "selected_options": data.get("selected_options"),
            "option_assessments": options,
            "issue_analyses": issues,
            "rationale": data.get("rationale"),
            "confidence": data.get("confidence"),
        }

    @staticmethod
    def _normalize_v3(data: Mapping[str, Any]) -> dict[str, Any]:
        _exact_keys(
            data,
            {
                "protocol_version",
                "selected_options",
                "options",
                "confidence",
            },
            "solver-v3",
        )
        if data.get("protocol_version") != SOLVER_PROTOCOL_VERSION:
            raise StructuredOutputError(
                "legal_mcq_solver",
                f"protocol_version must be {SOLVER_PROTOCOL_VERSION}",
            )
        selected = data.get("selected_options")
        raw_options = data.get("options")
        if not isinstance(selected, list) or not isinstance(raw_options, list):
            raise StructuredOutputError(
                "legal_mcq_solver",
                "selected_options and options must be arrays",
            )
        selected_labels = {str(label).strip().upper() for label in selected}
        reasons = []
        for option in raw_options:
            if not isinstance(option, Mapping):
                continue
            if str(option.get("label", "")).strip().upper() not in selected_labels:
                continue
            claims = option.get("claims")
            if isinstance(claims, list):
                reasons.extend(
                    str(claim.get("reason", "")).strip()
                    for claim in claims
                    if isinstance(claim, Mapping)
                    and str(claim.get("reason", "")).strip()
                )
        return SolverDecision._normalize_v2(
            {
                **data,
                "protocol_version": SOLVER_PROTOCOL_V2_VERSION,
                "issues": [],
                "rationale": "；".join(reasons) or "依据逐项命题判断。",
            }
        )

    @classmethod
    def from_text(cls, text: str) -> "SolverDecision":
        source_data = extract_json_object(text)
        version = source_data.get("protocol_version")
        if version == SOLVER_PROTOCOL_VERSION:
            data = cls._normalize_v3(source_data)
        elif version == SOLVER_PROTOCOL_V2_VERSION or (
            version is None and "options" in source_data
        ):
            data = cls._normalize_v2(source_data)
        elif version is not None:
            raise StructuredOutputError(
                "legal_mcq_solver",
                f"unsupported protocol_version: {version}",
            )
        else:
            data = source_data
        selected = tuple(
            _label(item, "selected_options") for item in data.get("selected_options", [])
        )
        if not selected:
            raise StructuredOutputError(
                "legal_mcq_solver", "selected_options cannot be empty"
            )

        raw_options = data.get("option_assessments")
        if not isinstance(raw_options, list) or not raw_options:
            raise StructuredOutputError(
                "legal_mcq_solver", "option_assessments must be a non-empty array"
            )
        assessments = []
        for option_index, raw_option in enumerate(raw_options):
            if not isinstance(raw_option, Mapping):
                raise StructuredOutputError(
                    "legal_mcq_solver",
                    f"option_assessments[{option_index}] must be an object",
                )
            label = _label(
                raw_option.get("label"), f"option_assessments[{option_index}].label"
            )
            raw_claims = raw_option.get("claims")
            if not isinstance(raw_claims, list) or not raw_claims:
                raise StructuredOutputError(
                    "legal_mcq_solver",
                    f"option_assessments.{label}.claims cannot be empty",
                )
            claims = []
            for claim_index, raw_claim in enumerate(raw_claims):
                if not isinstance(raw_claim, Mapping):
                    raise StructuredOutputError(
                        "legal_mcq_solver",
                        f"option_assessments.{label}.claims[{claim_index}] must be an object",
                    )
                verdict = _text(
                    raw_claim.get("verdict"),
                    f"option_assessments.{label}.claims[{claim_index}].verdict",
                ).lower()
                if verdict not in _VERDICTS:
                    raise StructuredOutputError(
                        "legal_mcq_solver",
                        f"invalid claim verdict: {verdict}",
                    )
                claims.append(
                    ClaimAssessment(
                        claim_id=_text(
                            raw_claim.get("claim_id", f"{label}{claim_index + 1}"),
                            "claim_id",
                        ),
                        text=_text(raw_claim.get("text"), "claim text"),
                        verdict=verdict,
                        reason=_text(raw_claim.get("reason"), "claim reason"),
                        rule=_text(
                            raw_claim.get("rule", ""),
                            "claim rule",
                            allow_empty=True,
                        ),
                        evidence_ids=_string_tuple(
                            raw_claim.get("evidence_ids", []), "evidence_ids"
                        ),
                    )
                )
            option_verdict = _text(
                raw_option.get("verdict"), f"option_assessments.{label}.verdict"
            ).lower()
            if option_verdict not in _VERDICTS:
                raise StructuredOutputError(
                    "legal_mcq_solver", f"invalid option verdict: {option_verdict}"
                )
            assessments.append(
                OptionAssessment(
                    label=label,
                    claims=tuple(claims),
                    verdict=option_verdict,
                    decisive_reason=_text(
                        raw_option.get("decisive_reason"),
                        f"option_assessments.{label}.decisive_reason",
                    ),
                    confidence=_confidence(
                        raw_option.get("confidence"),
                        f"option_assessments.{label}.confidence",
                    ),
                )
            )

        raw_issues = data.get("issue_analyses")
        if not isinstance(raw_issues, list):
            raise StructuredOutputError(
                "legal_mcq_solver", "issue_analyses must be an array"
            )
        issues = []
        for index, raw_issue in enumerate(raw_issues):
            if not isinstance(raw_issue, Mapping):
                raise StructuredOutputError(
                    "legal_mcq_solver", f"issue_analyses[{index}] must be an object"
                )
            issues.append(
                IssueAnalysis(
                    issue_id=_text(
                        raw_issue.get("issue_id", f"I{index + 1}"), "issue_id"
                    ),
                    issue=_text(raw_issue.get("issue"), "issue"),
                    application=_text(raw_issue.get("application"), "application"),
                    conclusion=_text(raw_issue.get("conclusion"), "conclusion"),
                )
            )
        return cls(
            selected_options=selected,
            option_assessments=tuple(assessments),
            issue_analyses=tuple(issues),
            rationale=_text(data.get("rationale"), "rationale"),
            confidence=_confidence(data.get("confidence"), "confidence"),
            raw=source_data,
        )


@dataclass(frozen=True)
class VerificationResult:
    accepted: bool
    error_codes: tuple[str, ...]
    challenged_options: tuple[str, ...]
    explanation: str
    suggested_selected_options: tuple[str, ...]
    raw: Mapping[str, Any] = field(repr=False)

    @staticmethod
    def response_format(option_labels: tuple[str, ...]) -> dict[str, Any]:
        label = {"type": "string", "enum": list(option_labels)}
        properties = {
            "protocol_version": {
                "type": "string",
                "enum": [VERIFIER_PROTOCOL_VERSION],
            },
            "accepted": {"type": "boolean"},
            "error_codes": {
                "type": "array",
                "items": {"type": "string"},
            },
            "challenged_options": {
                "type": "array",
                "items": label,
            },
            "suggested_selected_options": {
                "type": "array",
                "items": label,
            },
            "note": {"type": "string"},
        }
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "legal_mcq_verifier_v2",
                "strict": True,
                "schema": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": properties,
                    "required": list(properties),
                },
            },
        }

    @classmethod
    def from_text(cls, text: str) -> "VerificationResult":
        source = extract_json_object(text)
        is_v2 = "protocol_version" in source
        if is_v2:
            _exact_keys(
                source,
                {
                    "protocol_version",
                    "accepted",
                    "error_codes",
                    "challenged_options",
                    "suggested_selected_options",
                    "note",
                },
                "verifier-v2",
                stage="legal_mcq_verifier",
            )
            if source["protocol_version"] != VERIFIER_PROTOCOL_VERSION:
                raise StructuredOutputError(
                    "legal_mcq_verifier",
                    "unsupported verifier protocol version",
                )
            data = {
                "accepted": source["accepted"],
                "error_codes": source["error_codes"],
                "challenged_options": source["challenged_options"],
                "explanation": source["note"],
                "suggested_selected_options": source[
                    "suggested_selected_options"
                ],
            }
        else:
            data = source
        accepted = data.get("accepted")
        if not isinstance(accepted, bool):
            raise StructuredOutputError(
                "legal_mcq_verifier", "accepted must be a boolean"
            )
        error_codes = tuple(
            _text(item, "error_codes").upper()
            for item in data.get("error_codes", [])
        )
        if any(_ERROR_CODE.fullmatch(code) is None for code in error_codes):
            raise StructuredOutputError(
                "legal_mcq_verifier", "error_codes must use stable uppercase codes"
            )
        explanation = _text(
            data.get("explanation", ""),
            "explanation",
            allow_empty=accepted,
        )
        if is_v2 and len(explanation) > 160:
            raise StructuredOutputError(
                "legal_mcq_verifier",
                "explanation must not exceed 160 characters",
            )
        if is_v2 and accepted and (
            error_codes
            or data.get("challenged_options")
            or data.get("suggested_selected_options")
            or explanation
        ):
            raise StructuredOutputError(
                "legal_mcq_verifier",
                "accepted output must not contain challenges or explanation",
            )
        if is_v2 and not accepted and not error_codes:
            raise StructuredOutputError(
                "legal_mcq_verifier",
                "rejected output must include at least one error code",
            )
        return cls(
            accepted=accepted,
            error_codes=error_codes,
            challenged_options=tuple(
                _label(item, "challenged_options")
                for item in data.get("challenged_options", [])
            ),
            explanation=explanation,
            suggested_selected_options=tuple(
                _label(item, "suggested_selected_options")
                for item in data.get("suggested_selected_options", [])
            ),
            raw=source,
        )


@dataclass(frozen=True)
class Citation:
    citation_id: str
    title: str
    url: str
    publisher: str
    article_number: str | None = None
    quote: str | None = None


@dataclass(frozen=True)
class LegalAnswer:
    task_id: str
    question_id: str
    status: AnswerStatus
    selected_options: tuple[str, ...]
    concise_answer: str
    rationale: str
    option_assessments: tuple[OptionAssessment, ...]
    citations: tuple[Citation, ...]
    evidence_mode: str
    confidence: float
    needs_review: bool
    warnings: tuple[str, ...]
    trace_id: str

    def as_dict(self) -> dict[str, Any]:
        return to_jsonable(self)

    def to_markdown(self) -> str:
        choices = "、".join(self.selected_options)
        lines = [f"**答案：{choices}。**", "", self.rationale, "", "**选项判断**", ""]
        for item in self.option_assessments:
            lines.append(f"- {item.label}：{item.decisive_reason}")
        if self.citations:
            lines.extend(("", "**依据**", ""))
            for citation in self.citations:
                article = f"（{citation.article_number}）" if citation.article_number else ""
                lines.append(
                    f"- [{citation.title}]({citation.url}){article}，"
                    f"{citation.publisher}"
                )
        return "\n".join(lines)


@dataclass(frozen=True)
class LegalMCQRunResult:
    answer: LegalAnswer
    parsed_question: ParsedLegalQuestion
    controller_plan: ControllerPlan
    solver_decision: SolverDecision
    verification: VerificationResult
    revision_count: int
    prompt_version: str
    skill_versions: Mapping[str, str] = field(default_factory=dict)
    trace: RunTrace | None = field(default=None, repr=False, compare=False)
    execution_budget: Mapping[str, Any] = field(default_factory=dict)

    def diagnostics(self) -> dict[str, Any]:
        return {
            "answer": self.answer.as_dict(),
            "parsed_question": to_jsonable(self.parsed_question),
            "controller_plan": to_jsonable(self.controller_plan),
            "solver_decision": to_jsonable(self.solver_decision),
            "verification": to_jsonable(self.verification),
            "revision_count": self.revision_count,
            "prompt_version": self.prompt_version,
            "skill_versions": dict(self.skill_versions),
            "golden_leakage_incidents": 0,
            "trace": self.trace.as_dict() if self.trace is not None else {},
            "execution_budget": dict(self.execution_budget),
        }


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def compact_json(value: Any) -> str:
    return json.dumps(
        to_jsonable(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
