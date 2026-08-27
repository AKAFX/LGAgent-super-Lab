"""Independent prompt-based verification for CAPE-V reasoning candidates."""

from __future__ import annotations

import queue
import threading
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Mapping, Protocol
from uuid import uuid4

from .cape_v import ReasoningCandidate
from .config import ModelConfig
from .model import (
    BudgetExceededError,
    ChatMessage,
    ChatModel,
    ModelRequest,
    ModelResponse,
    TokenUsage,
)
from .protocol import StructuredOutputError, extract_json_object
from .serialization import dumps_json
from .trace import ModelCallTrace, RunTrace, utc_now

VERIFIER_NAMES = ("rule", "fact", "exception", "evidence", "entailment")

_PROMPTS = {
    "rule": """你是独立的法律规则验证器。只判断候选引用和表述的规则是否有效、适用且无明显过时。
仅输出严格JSON：{"pass":true,"score":0.0,"error_type":null,"reason":"..."}。
失败时error_type应为具体稳定代码，例如INVALID_RULE或INAPPLICABLE_RULE。""",
    "fact": """你是独立的事实依据验证器。只判断IRAC application使用的事实是否来自题目事实，且未虚构或歪曲。
仅输出严格JSON：{"pass":true,"score":0.0,"error_type":null,"reason":"..."}。
失败时error_type应为具体稳定代码，例如UNGROUNDED_FACT或FACT_DISTORTION。""",
    "exception": """你是独立的法律例外验证器。只判断候选是否识别并正确处理会改变结论的例外、但书或排除条件。
仅输出严格JSON：{"pass":true,"score":0.0,"error_type":null,"reason":"..."}。
失败时error_type应为具体稳定代码，例如MISSED_EXCEPTION。""",
    "evidence": """你是独立的证据验证器。只使用输入中的证据，核对候选证据ID、引用内容及其对主张的支持关系。
不得用未引用的模型知识补足证据。仅输出严格JSON：
{"pass":true,"score":0.0,"error_type":null,"reason":"..."}。
失败时error_type应为具体稳定代码，例如UNSUPPORTED_CITATION或UNKNOWN_EVIDENCE_ID。""",
    "entailment": """你是独立的结论蕴含验证器。判断候选的规则与事实应用是否逻辑上推出所选答案。
仅输出严格JSON：{"pass":true,"score":0.0,"error_type":null,"reason":"..."}。
失败时error_type应为具体稳定代码，例如INVALID_INFERENCE或CONCLUSION_MISMATCH。""",
}


def _safe_error_message(error: BaseException, api_key: str) -> str:
    message = str(error)
    return message.replace(api_key, "[REDACTED]") if api_key else message


def _score(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise StructuredOutputError("cape_v_verifier", "score must be a number")
    score = float(value)
    if not 0.0 <= score <= 1.0:
        raise StructuredOutputError(
            "cape_v_verifier", "score must be between 0 and 1"
        )
    return score


@dataclass(frozen=True)
class VerificationContext:
    question: str
    candidate: ReasoningCandidate
    facts: tuple[Mapping[str, Any], ...] = ()
    evidence: tuple[Any, ...] = ()
    exception_hints: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.question, str) or not self.question.strip():
            raise ValueError("question cannot be empty")


@dataclass(frozen=True)
class DimensionVerification:
    passed: bool
    score: float
    error_type: str | None
    reason: str
    attempts: int = 1

    @property
    def pass_(self) -> bool:
        """Expose the report's JSON `pass` field without using a keyword."""
        return self.passed

    @classmethod
    def from_text(cls, text: str, *, attempts: int = 1) -> "DimensionVerification":
        data = extract_json_object(text)
        passed = data.get("pass")
        if not isinstance(passed, bool):
            raise StructuredOutputError("cape_v_verifier", "pass must be a boolean")
        reason = data.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            raise StructuredOutputError(
                "cape_v_verifier", "reason must be a non-empty string"
            )
        error_type = data.get("error_type")
        if error_type is not None:
            if not isinstance(error_type, str) or not error_type.strip():
                raise StructuredOutputError(
                    "cape_v_verifier",
                    "error_type must be null or a non-empty string",
                )
            error_type = error_type.strip().upper()
        if passed and error_type is not None:
            raise StructuredOutputError(
                "cape_v_verifier", "passing result cannot have error_type"
            )
        if not passed and error_type is None:
            raise StructuredOutputError(
                "cape_v_verifier", "failing result must have error_type"
            )
        return cls(
            passed=passed,
            score=_score(data.get("score")),
            error_type=error_type,
            reason=reason.strip(),
            attempts=attempts,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "pass": self.passed,
            "score": self.score,
            "error_type": self.error_type,
            "reason": self.reason,
            "attempts": self.attempts,
        }


@dataclass(frozen=True)
class VerificationReport:
    candidate_id: str
    dimensions: dict[str, DimensionVerification]
    overall_score: float
    failed_steps: tuple[str, ...]
    error_types: tuple[str, ...]
    disagreement: float

    @classmethod
    def build(
        cls,
        candidate_id: str,
        dimensions: Mapping[str, DimensionVerification],
        *,
        weights: Mapping[str, float] | None = None,
    ) -> "VerificationReport":
        if not dimensions:
            return cls(
                candidate_id=candidate_id,
                dimensions={},
                overall_score=0.0,
                failed_steps=(),
                error_types=(),
                disagreement=0.0,
            )
        unknown = set(dimensions) - set(VERIFIER_NAMES)
        if unknown:
            raise ValueError(f"unknown verifier dimensions: {sorted(unknown)}")

        raw_weights = {
            name: float((weights or {}).get(name, 1.0)) for name in dimensions
        }
        if any(weight < 0.0 for weight in raw_weights.values()):
            raise ValueError("verifier weights must be non-negative")
        weight_total = sum(raw_weights.values())
        if weight_total <= 0.0:
            raise ValueError("at least one verifier weight must be positive")
        overall = sum(
            dimensions[name].score * raw_weights[name] for name in dimensions
        ) / weight_total

        results = list(dimensions.values())
        pairs = len(results) * (len(results) - 1) // 2
        disagreement = 0.0
        if pairs:
            disagreement = sum(
                left.passed != right.passed
                for index, left in enumerate(results)
                for right in results[index + 1 :]
            ) / pairs
        failed = tuple(name for name in VERIFIER_NAMES if name in dimensions and not dimensions[name].passed)
        errors = tuple(
            dict.fromkeys(
                dimensions[name].error_type
                for name in failed
                if dimensions[name].error_type is not None
            )
        )
        return cls(
            candidate_id=candidate_id,
            dimensions=dict(dimensions),
            overall_score=overall,
            failed_steps=failed,
            error_types=errors,
            disagreement=disagreement,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "dimensions": {
                name: result.as_dict() for name, result in self.dimensions.items()
            },
            "overall_score": self.overall_score,
            "failed_steps": list(self.failed_steps),
            "error_types": list(self.error_types),
            "disagreement": self.disagreement,
        }


class Verifier(Protocol):
    name: str

    def verify(
        self,
        context: VerificationContext,
        *,
        trace: RunTrace | None = None,
    ) -> DimensionVerification:
        """Verify one dimension of one reasoning candidate."""


@dataclass(frozen=True)
class VerifierConfig:
    enabled: Mapping[str, bool] = field(
        default_factory=lambda: {name: True for name in VERIFIER_NAMES}
    )
    timeout_seconds: float = 30.0
    max_attempts: int = 2
    max_tokens: int = 512
    weights: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        unknown = (set(self.enabled) | set(self.weights)) - set(VERIFIER_NAMES)
        if unknown:
            raise ValueError(f"unknown verifier names: {sorted(unknown)}")
        if self.timeout_seconds <= 0.0:
            raise ValueError("timeout_seconds must be positive")
        if self.max_attempts <= 0:
            raise ValueError("max_attempts must be positive")
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if any(float(weight) < 0.0 for weight in self.weights.values()):
            raise ValueError("verifier weights must be non-negative")


def _complete_with_timeout(
    model: ChatModel,
    request: ModelRequest,
    timeout_seconds: float,
) -> ModelResponse:
    result: queue.Queue[tuple[bool, Any]] = queue.Queue(maxsize=1)

    def invoke() -> None:
        try:
            result.put((True, model.complete(request)))
        except Exception as exc:
            result.put((False, exc))

    worker = threading.Thread(target=invoke, daemon=True)
    worker.start()
    try:
        succeeded, value = result.get(timeout=timeout_seconds)
    except queue.Empty as exc:
        raise TimeoutError(
            f"verifier model call exceeded {timeout_seconds:g} seconds"
        ) from exc
    if not succeeded:
        raise value
    if not isinstance(value, ModelResponse):
        raise TypeError("model.complete must return ModelResponse")
    return value


class PromptVerifier:
    """One independently configured verifier backed by a strict JSON prompt."""

    def __init__(
        self,
        name: str,
        model: ChatModel,
        model_config: ModelConfig | Mapping[str, Any],
        verifier_config: VerifierConfig,
    ) -> None:
        if name not in VERIFIER_NAMES:
            raise ValueError(f"unknown verifier name: {name}")
        self.name = name
        self.model = model
        self.model_config = (
            model_config
            if isinstance(model_config, ModelConfig)
            else ModelConfig.from_mapping(model_config)
        )
        self.verifier_config = verifier_config

    def _messages(self, context: VerificationContext) -> list[ChatMessage]:
        payload = {
            "question": context.question,
            "candidate": context.candidate,
            "facts": context.facts,
            "evidence": context.evidence,
            "exception_hints": context.exception_hints,
        }
        return [
            ChatMessage("system", _PROMPTS[self.name]),
            ChatMessage(
                "user",
                "请验证以下输入。候选之间相互不可见；证据验证只能依据evidence字段。\n"
                + dumps_json(payload),
            ),
        ]

    def verify(
        self,
        context: VerificationContext,
        *,
        trace: RunTrace | None = None,
    ) -> DimensionVerification:
        run_trace = trace or RunTrace()
        messages = self._messages(context)
        last_output = ""

        for attempt in range(1, self.verifier_config.max_attempts + 1):
            started_at = utc_now()
            started = perf_counter()
            response: ModelResponse | None = None
            error: BaseException | None = None
            try:
                response = _complete_with_timeout(
                    self.model,
                    ModelRequest(
                        model=self.model_config.model,
                        messages=tuple(messages),
                        temperature=self.model_config.temperature,
                        top_p=self.model_config.top_p,
                        max_tokens=min(
                            self.model_config.max_tokens,
                            self.verifier_config.max_tokens,
                        ),
                        metadata={
                            "agent": f"cape_v_verifier_{self.name}",
                            "verifier": self.name,
                            "candidate_id": context.candidate.candidate_id,
                            "attempt": attempt,
                        },
                    ),
                    self.verifier_config.timeout_seconds,
                )
                last_output = response.content.strip()
                parsed = DimensionVerification.from_text(
                    last_output, attempts=attempt
                )
            except Exception as exc:
                error = exc

            error_message = (
                _safe_error_message(error, self.model_config.api_key)
                if error is not None
                else None
            )
            run_trace.add_call(
                ModelCallTrace(
                    call_id=uuid4().hex,
                    agent=f"cape_v_verifier_{self.name}",
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
                    error_message=error_message,
                )
            )
            if error is None:
                return parsed

            run_trace.add_error(
                f"cape_v_verifier_{self.name}",
                error,
                message=error_message,
            )
            if isinstance(error, BudgetExceededError):
                error.diagnostics.setdefault("trace", run_trace.as_dict())
                raise error
            if isinstance(error, TimeoutError):
                return DimensionVerification(
                    passed=False,
                    score=0.0,
                    error_type="VERIFIER_TIMEOUT",
                    reason=error_message or "verifier timed out",
                    attempts=attempt,
                )
            if attempt < self.verifier_config.max_attempts:
                if last_output:
                    messages.append(ChatMessage("assistant", last_output))
                messages.append(
                    ChatMessage(
                        "user",
                        f"上一响应不符合验证JSON协议：{error_message}。"
                        "请仅输出完整、合法且字段齐全的JSON对象。",
                    )
                )
                continue
            error_type = (
                "VERIFIER_INVALID_JSON"
                if isinstance(error, StructuredOutputError)
                else "VERIFIER_MODEL_ERROR"
            )
            return DimensionVerification(
                passed=False,
                score=0.0,
                error_type=error_type,
                reason=error_message or "verifier failed",
                attempts=attempt,
            )

        raise AssertionError("unreachable verifier attempt state")


class RuleVerifier(PromptVerifier):
    def __init__(
        self,
        model: ChatModel,
        model_config: ModelConfig | Mapping[str, Any],
        verifier_config: VerifierConfig,
    ) -> None:
        super().__init__("rule", model, model_config, verifier_config)


class FactVerifier(PromptVerifier):
    def __init__(
        self,
        model: ChatModel,
        model_config: ModelConfig | Mapping[str, Any],
        verifier_config: VerifierConfig,
    ) -> None:
        super().__init__("fact", model, model_config, verifier_config)


class ExceptionVerifier(PromptVerifier):
    def __init__(
        self,
        model: ChatModel,
        model_config: ModelConfig | Mapping[str, Any],
        verifier_config: VerifierConfig,
    ) -> None:
        super().__init__("exception", model, model_config, verifier_config)


class EvidenceVerifier(PromptVerifier):
    def __init__(
        self,
        model: ChatModel,
        model_config: ModelConfig | Mapping[str, Any],
        verifier_config: VerifierConfig,
    ) -> None:
        super().__init__("evidence", model, model_config, verifier_config)


class EntailmentVerifier(PromptVerifier):
    def __init__(
        self,
        model: ChatModel,
        model_config: ModelConfig | Mapping[str, Any],
        verifier_config: VerifierConfig,
    ) -> None:
        super().__init__("entailment", model, model_config, verifier_config)


class VerificationRunner:
    """Run the enabled verifier registry and aggregate a common report."""

    def __init__(
        self,
        model: ChatModel,
        model_config: ModelConfig | Mapping[str, Any],
        verifier_config: VerifierConfig | None = None,
        *,
        registry: Mapping[str, Verifier] | None = None,
    ) -> None:
        self.config = verifier_config or VerifierConfig()
        self.model_config = (
            model_config
            if isinstance(model_config, ModelConfig)
            else ModelConfig.from_mapping(model_config)
        )
        self.registry: dict[str, Verifier] = (
            dict(registry)
            if registry is not None
            else {
                verifier.name: verifier
                for verifier in (
                    RuleVerifier(model, self.model_config, self.config),
                    FactVerifier(model, self.model_config, self.config),
                    ExceptionVerifier(model, self.model_config, self.config),
                    EvidenceVerifier(model, self.model_config, self.config),
                    EntailmentVerifier(model, self.model_config, self.config),
                )
            }
        )
        missing = {
            name
            for name in VERIFIER_NAMES
            if self.config.enabled.get(name, False) and name not in self.registry
        }
        if missing:
            raise ValueError(f"enabled verifiers missing from registry: {sorted(missing)}")

    def run(
        self,
        context: VerificationContext,
        *,
        trace: RunTrace | None = None,
    ) -> VerificationReport:
        run_trace = trace or RunTrace()
        dimensions = {
            name: self.registry[name].verify(context, trace=run_trace)
            for name in VERIFIER_NAMES
            if self.config.enabled.get(name, False)
        }
        return VerificationReport.build(
            context.candidate.candidate_id,
            dimensions,
            weights=self.config.weights,
        )
