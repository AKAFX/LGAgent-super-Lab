"""CAPE-V candidate sampling, option permutation, and stability metrics."""

from __future__ import annotations

import itertools
import math
import random
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Mapping
from uuid import uuid4

from .config import ModelConfig
from .model import (
    BudgetExceededError,
    ChatMessage,
    ChatModel,
    ModelRequest,
    ModelResponse,
    TokenUsage,
)
from .protocol import OPTIONS, StructuredOutputError, extract_json_object, normalize_option_label
from .serialization import dumps_json, to_jsonable
from .trace import ModelCallTrace, RunTrace, utc_now

OPTION_LABELS = ("A", "B", "C", "D")
VERIFICATION_STATUSES = frozenset({"SUPPORT", "REFUTE", "NEI"})
_OPTION_MARKER = re.compile(
    r"(?m)^[ \t]*(?:"
    r"[\(\[（【]\s*(?P<bracket>[A-DＡ-Ｄ])\s*[\)\]）】]"
    r"|(?P<plain>[A-DＡ-Ｄ])\s*[.．、:：\)]"
    r")[ \t]*"
)

CANDIDATE_SYSTEM_PROMPT = """你是CAPE-V候选推理器。独立分析题目，不参考其他候选。
如提供 trusted_context，其中 audited_evidence_matrix 是唯一允许使用的外部法律证据，
不得补充矩阵之外的法律知识，引用必须使用其中的 evidence_id。
仅输出严格JSON：
{"answer":"A","irac":{"issue":"...","rule":[{"claim":"...","evidence_ids":[]}],
"application":[{"fact_ids":[],"rule_index":0,"inference":"..."}],"conclusion":"..."},
"option_verification":{"A":{"status":"SUPPORT","score":0.0,"evidence_ids":[]},
"B":{"status":"REFUTE","score":0.0,"evidence_ids":[]},"C":{"status":"NEI","score":0.0,"evidence_ids":[]},
"D":{"status":"REFUTE","score":0.0,"evidence_ids":[]}}}
保持IRAC紧凑；answer必须是当前题面中的A/B/C/D。"""


class OptionParseError(ValueError):
    """Raised when a four-option question cannot be parsed without ambiguity."""


def _normalize_option_text(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).split())


def _normalize_marker_label(label: str) -> str:
    return unicodedata.normalize("NFKC", label).upper()


@dataclass(frozen=True)
class QuestionOption:
    original_label: str
    text: str
    identity: str


@dataclass(frozen=True)
class ParsedQuestion:
    stem: str
    options: tuple[QuestionOption, ...]

    def option(self, label: str) -> QuestionOption:
        normalized = normalize_option_label(label, "option")
        return self.options[OPTION_LABELS.index(normalized)]


def parse_four_option_question(question: str) -> ParsedQuestion:
    """Parse exactly one ordered A-D option block and reject ambiguous identities."""
    if not isinstance(question, str) or not question.strip():
        raise OptionParseError("question is empty")

    markers = list(_OPTION_MARKER.finditer(question))
    labels = [
        _normalize_marker_label(match.group("bracket") or match.group("plain"))
        for match in markers
    ]
    if labels != list(OPTION_LABELS):
        raise OptionParseError(
            "expected exactly one option marker for each of A, B, C and D in order"
        )

    stem = question[: markers[0].start()].strip()
    if not stem:
        raise OptionParseError("question stem is empty")

    options: list[QuestionOption] = []
    for index, (label, marker) in enumerate(zip(OPTION_LABELS, markers)):
        end = markers[index + 1].start() if index + 1 < len(markers) else len(question)
        text = question[marker.end() : end].strip()
        identity = _normalize_option_text(text)
        if not identity:
            raise OptionParseError(f"option {label} is empty")
        options.append(QuestionOption(label, text, identity))

    identities = [option.identity for option in options]
    if len(set(identities)) != len(identities):
        raise OptionParseError("option texts are not unique after normalization")
    return ParsedQuestion(stem=stem, options=tuple(options))


@dataclass(frozen=True)
class OptionPermutation:
    permutation_id: str
    question: str
    displayed_to_identity: dict[str, str]
    identity_to_displayed: dict[str, str]
    identity_to_original: dict[str, str]
    original_to_identity: dict[str, str]

    @property
    def displayed_to_original(self) -> dict[str, str]:
        return {
            displayed: self.identity_to_original[identity]
            for displayed, identity in self.displayed_to_identity.items()
        }

    @property
    def original_to_displayed(self) -> dict[str, str]:
        return {
            original: self.identity_to_displayed[identity]
            for original, identity in self.original_to_identity.items()
        }

    def map_answer_to_original(self, displayed_answer: str) -> str:
        displayed = normalize_option_label(displayed_answer, "answer")
        identity = self.displayed_to_identity[displayed]
        return self.identity_to_original[identity]

    def answer_identity(self, displayed_answer: str) -> str:
        displayed = normalize_option_label(displayed_answer, "answer")
        return self.displayed_to_identity[displayed]


def _build_permutation(
    parsed: ParsedQuestion,
    order: tuple[str, ...],
    permutation_id: str,
) -> OptionPermutation:
    original_options = {option.original_label: option for option in parsed.options}
    displayed_to_identity = {
        displayed: original_options[original].identity
        for displayed, original in zip(OPTION_LABELS, order)
    }
    identity_to_displayed = {
        identity: displayed for displayed, identity in displayed_to_identity.items()
    }
    identity_to_original = {
        option.identity: option.original_label for option in parsed.options
    }
    original_to_identity = {
        option.original_label: option.identity for option in parsed.options
    }
    lines = [parsed.stem, ""]
    lines.extend(
        f"{displayed}. {original_options[original].text}"
        for displayed, original in zip(OPTION_LABELS, order)
    )
    return OptionPermutation(
        permutation_id=permutation_id,
        question="\n".join(lines),
        displayed_to_identity=displayed_to_identity,
        identity_to_displayed=identity_to_displayed,
        identity_to_original=identity_to_original,
        original_to_identity=original_to_identity,
    )


def identity_permutation(parsed: ParsedQuestion) -> OptionPermutation:
    return _build_permutation(parsed, OPTION_LABELS, "original")


def generate_option_permutations(
    parsed: ParsedQuestion,
    *,
    count: int,
    seed: int,
) -> tuple[OptionPermutation, ...]:
    """Generate deterministic, unique, non-identity option reorderings."""
    orders = [
        order
        for order in itertools.permutations(OPTION_LABELS)
        if order != OPTION_LABELS
    ]
    if count < 0:
        raise ValueError("permutation count cannot be negative")
    if count > len(orders):
        raise ValueError(f"permutation count cannot exceed {len(orders)}")
    random.Random(seed).shuffle(orders)
    return tuple(
        _build_permutation(parsed, order, f"perm-{index + 1}")
        for index, order in enumerate(orders[:count])
    )


def _non_empty_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise StructuredOutputError("cape_v_candidate", f"{field_name} must be non-empty")
    return value.strip()


def _string_tuple(value: Any, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise StructuredOutputError("cape_v_candidate", f"{field_name} must be an array")
    return tuple(
        _non_empty_string(item, f"{field_name}[{index}]")
        for index, item in enumerate(value)
    )


def _object(value: Any, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise StructuredOutputError("cape_v_candidate", f"{field_name} must be an object")
    return value


def _score(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise StructuredOutputError("cape_v_candidate", f"{field_name} must be a number")
    result = float(value)
    if not 0.0 <= result <= 1.0:
        raise StructuredOutputError(
            "cape_v_candidate", f"{field_name} must be between 0 and 1"
        )
    return result


@dataclass(frozen=True)
class IRACRule:
    claim: str
    evidence_ids: tuple[str, ...]


@dataclass(frozen=True)
class IRACApplication:
    fact_ids: tuple[str, ...]
    rule_index: int
    inference: str


@dataclass(frozen=True)
class CompactIRAC:
    issue: str
    rule: tuple[IRACRule, ...]
    application: tuple[IRACApplication, ...]
    conclusion: str


@dataclass(frozen=True)
class CandidateOptionVerification:
    status: str
    score: float
    evidence_ids: tuple[str, ...]


@dataclass(frozen=True)
class CandidatePayload:
    answer: str
    irac: CompactIRAC
    option_verification: dict[str, CandidateOptionVerification]

    @classmethod
    def from_text(cls, text: str) -> "CandidatePayload":
        data = extract_json_object(text)
        answer = normalize_option_label(data.get("answer"), "answer")
        irac_data = _object(data.get("irac"), "irac")

        raw_rules = irac_data.get("rule")
        if not isinstance(raw_rules, list) or not raw_rules:
            raise StructuredOutputError(
                "cape_v_candidate", "irac.rule must be a non-empty array"
            )
        rules = tuple(
            IRACRule(
                claim=_non_empty_string(
                    _object(item, f"irac.rule[{index}]").get("claim"),
                    f"irac.rule[{index}].claim",
                ),
                evidence_ids=_string_tuple(
                    _object(item, f"irac.rule[{index}]").get("evidence_ids"),
                    f"irac.rule[{index}].evidence_ids",
                ),
            )
            for index, item in enumerate(raw_rules)
        )

        raw_applications = irac_data.get("application")
        if not isinstance(raw_applications, list) or not raw_applications:
            raise StructuredOutputError(
                "cape_v_candidate", "irac.application must be a non-empty array"
            )
        applications: list[IRACApplication] = []
        for index, item in enumerate(raw_applications):
            application = _object(item, f"irac.application[{index}]")
            rule_index = application.get("rule_index")
            if (
                isinstance(rule_index, bool)
                or not isinstance(rule_index, int)
                or not 0 <= rule_index < len(rules)
            ):
                raise StructuredOutputError(
                    "cape_v_candidate",
                    f"irac.application[{index}].rule_index is out of range",
                )
            applications.append(
                IRACApplication(
                    fact_ids=_string_tuple(
                        application.get("fact_ids"),
                        f"irac.application[{index}].fact_ids",
                    ),
                    rule_index=rule_index,
                    inference=_non_empty_string(
                        application.get("inference"),
                        f"irac.application[{index}].inference",
                    ),
                )
            )

        raw_verification = _object(
            data.get("option_verification"), "option_verification"
        )
        verification: dict[str, CandidateOptionVerification] = {}
        for raw_label, item in raw_verification.items():
            label = normalize_option_label(raw_label, "option_verification key")
            if label in verification:
                raise StructuredOutputError(
                    "cape_v_candidate",
                    f"option_verification contains duplicate option {label}",
                )
            detail = _object(item, f"option_verification.{label}")
            status = _non_empty_string(
                detail.get("status"), f"option_verification.{label}.status"
            ).upper()
            if status not in VERIFICATION_STATUSES:
                raise StructuredOutputError(
                    "cape_v_candidate",
                    f"option_verification.{label}.status is invalid",
                )
            verification[label] = CandidateOptionVerification(
                status=status,
                score=_score(detail.get("score"), f"option_verification.{label}.score"),
                evidence_ids=_string_tuple(
                    detail.get("evidence_ids"),
                    f"option_verification.{label}.evidence_ids",
                ),
            )
        if set(verification) != OPTIONS:
            raise StructuredOutputError(
                "cape_v_candidate", "option_verification must contain exactly A-D"
            )

        return cls(
            answer=answer,
            irac=CompactIRAC(
                issue=_non_empty_string(irac_data.get("issue"), "irac.issue"),
                rule=rules,
                application=tuple(applications),
                conclusion=_non_empty_string(
                    irac_data.get("conclusion"), "irac.conclusion"
                ),
            ),
            option_verification=verification,
        )


@dataclass(frozen=True)
class ReasoningCandidate:
    candidate_id: str
    answer: str
    answer_identity: str | None
    presented_answer: str
    permutation_id: str
    irac: CompactIRAC
    option_verification: dict[str, CandidateOptionVerification]


@dataclass(frozen=True)
class CapeVMetrics:
    answer_counts: dict[str, int]
    answer_distribution: dict[str, float]
    normalized_answer_entropy: float
    permutation_consistency: float | None


def calculate_cape_v_metrics(
    candidate_answers: list[str] | tuple[str, ...],
    permutation_answers: list[str] | tuple[str, ...] = (),
) -> CapeVMetrics:
    normalized = [
        normalize_option_label(answer, "candidate answer") for answer in candidate_answers
    ]
    if not normalized:
        raise ValueError("at least one candidate answer is required")
    counts = Counter(normalized)
    total = len(normalized)
    answer_counts = {label: counts.get(label, 0) for label in OPTION_LABELS}
    distribution = {
        label: answer_counts[label] / total for label in OPTION_LABELS
    }
    entropy = -sum(
        probability * math.log(probability)
        for probability in distribution.values()
        if probability
    )
    normalized_entropy = entropy / math.log(len(OPTION_LABELS))

    mapped_permutations = [
        normalize_option_label(answer, "permutation answer")
        for answer in permutation_answers
    ]
    consistency = None
    if mapped_permutations:
        permutation_counts = Counter(mapped_permutations)
        consistency = max(permutation_counts.values()) / len(mapped_permutations)
    return CapeVMetrics(
        answer_counts=answer_counts,
        answer_distribution=distribution,
        normalized_answer_entropy=normalized_entropy,
        permutation_consistency=consistency,
    )


@dataclass(frozen=True)
class CapeVConfig:
    candidate_count: int = 1
    permutation_count: int = 3
    seed: int = 42
    max_attempts: int = 2
    max_tokens: int = 1024

    def __post_init__(self) -> None:
        if self.candidate_count <= 0:
            raise ValueError("candidate_count must be positive")
        if not 0 <= self.permutation_count <= 23:
            raise ValueError("permutation_count must be between 0 and 23")
        if self.max_attempts <= 0:
            raise ValueError("max_attempts must be positive")
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")


@dataclass(frozen=True)
class CapeVResult:
    candidates: tuple[ReasoningCandidate, ...]
    permutation_candidates: tuple[ReasoningCandidate, ...]
    permutations: tuple[OptionPermutation, ...]
    metrics: CapeVMetrics
    permutation_enabled: bool
    permutation_disabled_reason: str | None
    trace: RunTrace


class CapeVRunner:
    """Generate independent candidates and deterministic permutation probes."""

    def __init__(
        self,
        model: ChatModel,
        model_config: ModelConfig | Mapping[str, Any],
        cape_v_config: CapeVConfig | None = None,
    ) -> None:
        self.model = model
        self.model_config = (
            model_config
            if isinstance(model_config, ModelConfig)
            else ModelConfig.from_mapping(model_config)
        )
        self.cape_v_config = cape_v_config or CapeVConfig()

    def _invoke(
        self,
        question: str,
        candidate_id: str,
        permutation: OptionPermutation | None,
        trace: RunTrace,
        trusted_context: Mapping[str, Any] | None,
    ) -> ReasoningCandidate:
        context = self._context_for_permutation(trusted_context, permutation)
        user_content = f"请独立生成一个紧凑IRAC候选：\n\n{question}"
        if context:
            user_content += "\n\ntrusted_context:\n" + dumps_json(context)
        messages = [
            ChatMessage("system", CANDIDATE_SYSTEM_PROMPT),
            ChatMessage("user", user_content),
        ]
        last_output = ""
        last_error: BaseException | None = None
        last_error_message = ""
        payload: CandidatePayload | None = None

        for attempt in range(1, self.cape_v_config.max_attempts + 1):
            started_at = utc_now()
            started = perf_counter()
            response: ModelResponse | None = None
            error: Exception | None = None
            try:
                response = self.model.complete(
                    ModelRequest(
                        model=self.model_config.model,
                        messages=tuple(messages),
                        temperature=self.model_config.temperature,
                        top_p=self.model_config.top_p,
                        max_tokens=min(
                            self.model_config.max_tokens,
                            self.cape_v_config.max_tokens,
                        ),
                        metadata={
                            "agent": "cape_v_candidate",
                            "candidate_id": candidate_id,
                            "permutation_id": (
                                permutation.permutation_id if permutation else "unparsed"
                            ),
                            "attempt": attempt,
                        },
                    )
                )
                last_output = response.content.strip()
                payload = CandidatePayload.from_text(last_output)
            except Exception as exc:
                error = exc
                last_error = exc
                last_error_message = str(exc)
                if self.model_config.api_key:
                    last_error_message = last_error_message.replace(
                        self.model_config.api_key, "[REDACTED]"
                    )

            trace.add_call(
                ModelCallTrace(
                    call_id=uuid4().hex,
                    agent="cape_v_candidate",
                    model=self.model_config.model,
                    attempt=attempt,
                    started_at=started_at,
                    duration_ms=(perf_counter() - started) * 1000,
                    usage=(
                        response.usage
                        if response
                        else getattr(error, "usage", TokenUsage())
                    ),
                    request_id=response.request_id if response else None,
                    seed_requested=(
                        response.seed_requested
                        if response
                        else getattr(error, "diagnostics", {}).get("seed_requested")
                    ),
                    provider_seed_guarantee=(
                        response.provider_seed_guarantee
                        if response
                        else getattr(error, "diagnostics", {}).get(
                            "provider_seed_guarantee"
                        )
                    ),
                    error_type=type(error).__name__ if error else None,
                    error_message=last_error_message if error else None,
                )
            )
            if error is None:
                assert payload is not None
                break

            trace.add_error("cape_v_candidate", error, message=last_error_message)
            if isinstance(error, BudgetExceededError):
                error.diagnostics.setdefault("trace", trace.as_dict())
                raise error
            if attempt < self.cape_v_config.max_attempts:
                if last_output:
                    messages.append(ChatMessage("assistant", last_output))
                messages.append(
                    ChatMessage(
                        "user",
                        f"上一响应不符合紧凑IRAC JSON协议：{error}。"
                        "请仅重新输出完整、合法且字段齐全的JSON对象。",
                    )
                )
        else:
            raise StructuredOutputError(
                "cape_v_candidate",
                last_error_message or "model call failed",
                attempts=self.cape_v_config.max_attempts,
                raw_output=last_output,
            ) from last_error

        assert payload is not None
        answer = payload.answer
        answer_identity = None
        verification = payload.option_verification
        permutation_id = "unparsed"
        if permutation is not None:
            answer_identity = permutation.answer_identity(payload.answer)
            answer = permutation.map_answer_to_original(payload.answer)
            verification = {
                permutation.map_answer_to_original(label): detail
                for label, detail in payload.option_verification.items()
            }
            permutation_id = permutation.permutation_id
        return ReasoningCandidate(
            candidate_id=candidate_id,
            answer=answer,
            answer_identity=answer_identity,
            presented_answer=payload.answer,
            permutation_id=permutation_id,
            irac=payload.irac,
            option_verification=verification,
        )

    @staticmethod
    def _context_for_permutation(
        trusted_context: Mapping[str, Any] | None,
        permutation: OptionPermutation | None,
    ) -> dict[str, Any]:
        if not trusted_context:
            return {}
        context = to_jsonable(trusted_context)
        assert isinstance(context, dict)
        if permutation is None:
            return context
        matrix = context.get("audited_evidence_matrix")
        if not isinstance(matrix, Mapping):
            return context
        options = matrix.get("options")
        if not isinstance(options, Mapping):
            return context
        remapped = dict(matrix)
        remapped["options"] = {
            displayed: options.get(original, {})
            for displayed, original in permutation.displayed_to_original.items()
        }
        context["audited_evidence_matrix"] = remapped
        return context

    def run(
        self,
        question: str,
        *,
        trace: RunTrace | None = None,
        trusted_context: Mapping[str, Any] | None = None,
        candidate_prefix: str = "",
    ) -> CapeVResult:
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question cannot be empty")
        run_trace = trace or RunTrace()
        parsed: ParsedQuestion | None = None
        disabled_reason: str | None = None
        try:
            parsed = parse_four_option_question(question)
        except OptionParseError as exc:
            disabled_reason = str(exc)
            run_trace.add_route(
                "cape_v_permutation_disabled",
                disabled_reason,
                {"permutation_count": self.cape_v_config.permutation_count},
            )

        original_mapping = identity_permutation(parsed) if parsed is not None else None
        candidates = tuple(
            self._invoke(
                question,
                f"{candidate_prefix}candidate-{index + 1}",
                original_mapping,
                run_trace,
                trusted_context,
            )
            for index in range(self.cape_v_config.candidate_count)
        )

        permutations: tuple[OptionPermutation, ...] = ()
        permutation_candidates: tuple[ReasoningCandidate, ...] = ()
        if parsed is not None and self.cape_v_config.permutation_count:
            permutations = generate_option_permutations(
                parsed,
                count=self.cape_v_config.permutation_count,
                seed=self.cape_v_config.seed,
            )
            permutation_candidates = tuple(
                self._invoke(
                    permutation.question,
                    f"{candidate_prefix}permutation-candidate-{index + 1}",
                    permutation,
                    run_trace,
                    trusted_context,
                )
                for index, permutation in enumerate(permutations)
            )
            run_trace.add_route(
                "cape_v_permutation_complete",
                "deterministic permutation probes completed",
                {"count": len(permutation_candidates), "seed": self.cape_v_config.seed},
            )

        metrics = calculate_cape_v_metrics(
            [candidate.answer for candidate in candidates],
            [candidate.answer for candidate in permutation_candidates],
        )
        return CapeVResult(
            candidates=candidates,
            permutation_candidates=permutation_candidates,
            permutations=permutations,
            metrics=metrics,
            permutation_enabled=parsed is not None,
            permutation_disabled_reason=disabled_reason,
            trace=run_trace,
        )
