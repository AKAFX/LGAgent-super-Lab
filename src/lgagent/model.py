"""Replaceable model-call protocol and OpenAI-compatible adapter."""

from __future__ import annotations

import hashlib
import json
import queue
import threading
from dataclasses import dataclass, field, replace
from math import ceil
from threading import Lock
from time import perf_counter
from typing import Any, Callable, Mapping, Protocol, Sequence

_SEED_METADATA_KEYS = (
    "agent",
    "candidate_id",
    "permutation_id",
    "attempt",
    "verifier",
    "option",
)


@dataclass(frozen=True)
class ChatMessage:
    role: str
    content: str


@dataclass(frozen=True)
class ModelRequest:
    model: str
    messages: tuple[ChatMessage, ...]
    temperature: float = 0.7
    top_p: float = 0.8
    max_tokens: int = 1024
    seed: int | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    timeout_seconds: float | None = None
    budget_tokens: int | None = None
    response_format: Mapping[str, Any] | None = None
    reasoning_effort: str | None = None


@dataclass(frozen=True)
class TokenUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass(frozen=True)
class ModelResponse:
    content: str
    usage: TokenUsage = field(default_factory=TokenUsage)
    request_id: str | None = None
    seed_requested: int | None = None
    provider_seed_guarantee: str | None = None
    finish_reason: str | None = None
    response_model: str | None = None
    content_state: str | None = None
    visible_content: str | None = field(default=None, repr=False)
    refusal_present: bool | None = None
    reasoning_present: bool | None = None
    tool_calls_present: bool | None = None
    reasoning_tokens: int | None = None
    cached_prompt_tokens: int | None = None
    usage_reported: bool | None = None


class ModelCallError(RuntimeError):
    """Raised when a model provider cannot produce a response."""

    def __init__(
        self,
        message: str,
        *,
        diagnostics: Mapping[str, Any] | None = None,
        response: ModelResponse | None = None,
    ) -> None:
        self.diagnostics = dict(diagnostics or {})
        self.response = response
        self.usage = response.usage if response is not None else TokenUsage()
        super().__init__(message)


class BudgetExceededError(RuntimeError):
    """Raised before a model request that would exceed a hard run budget."""

    def __init__(
        self,
        reason: str,
        *,
        diagnostics: Mapping[str, Any] | None = None,
        usage: TokenUsage | None = None,
        response: ModelResponse | None = None,
    ) -> None:
        self.reason = reason
        self.diagnostics = dict(diagnostics or {})
        self.usage = usage or TokenUsage()
        self.response = response
        super().__init__(f"execution budget exhausted: {reason}")


class ChatModel(Protocol):
    def complete(self, request: ModelRequest) -> ModelResponse:
        """Complete one chat request."""


def estimate_message_tokens(messages: Sequence[ChatMessage]) -> int:
    """Conservatively estimate prompt tokens without provider-specific tokenizers."""
    total = 0
    for message in messages:
        text = message.content
        cjk = sum(
            1
            for character in text
            if (
                "\u3400" <= character <= "\u4dbf"
                or "\u4e00" <= character <= "\u9fff"
                or "\uf900" <= character <= "\ufaff"
            )
        )
        non_cjk = len(text) - cjk
        total += cjk + ceil(non_cjk / 4) + 6
    return max(1, ceil(total * 1.15))


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _optional_count(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _visible_content(message: Any) -> tuple[str, str]:
    missing = object()
    content = _field(message, "content", missing)
    if content is missing:
        return "missing", ""
    if content is None:
        return "null", ""
    if isinstance(content, str):
        return ("text" if content.strip() else "empty"), content
    if isinstance(content, list):
        parts = [
            part if isinstance(part, str) else _field(part, "text", "")
            for part in content
            if isinstance(part, str)
            or _field(part, "type") in {None, "text", "output_text"}
        ]
        return "parts", "\n".join(part for part in parts if isinstance(part, str))
    return "unsupported", ""


def _message_content(message: Any) -> str:
    _, visible = _visible_content(message)
    return visible


class OpenAIChatModel:
    """Adapter for OpenAI and OpenAI-compatible synchronous clients."""

    def __init__(self, client: Any) -> None:
        self._client = client

    def complete(self, request: ModelRequest) -> ModelResponse:
        try:
            arguments = {
                "model": request.model,
                "messages": [
                    {"role": message.role, "content": message.content}
                    for message in request.messages
                ],
                "temperature": request.temperature,
                "top_p": request.top_p,
                "max_tokens": request.max_tokens,
            }
            if request.seed is not None:
                arguments["seed"] = request.seed
            if request.timeout_seconds is not None:
                arguments["timeout"] = request.timeout_seconds
            if request.response_format is not None:
                arguments["response_format"] = request.response_format
            if request.reasoning_effort is not None:
                arguments["reasoning_effort"] = request.reasoning_effort
            response = self._client.chat.completions.create(
                **arguments,
            )
        except Exception as exc:
            body = getattr(exc, "body", None)
            error_body = _field(body, "error", body)
            provider_error_code = _field(error_body, "code")
            if provider_error_code is None and "timeout" in type(exc).__name__.lower():
                provider_error_code = "request_timeout"
            raise ModelCallError(
                f"{type(exc).__name__}: {exc}",
                diagnostics={
                    "http_status": getattr(exc, "status_code", None),
                    "request_id": getattr(exc, "request_id", None),
                    "provider_error_code": provider_error_code,
                    "provider_error_type": _field(error_body, "type"),
                },
            ) from exc

        choices: Sequence[Any] = getattr(response, "choices", ()) or ()
        first_choice = choices[0] if choices else None
        message = _field(first_choice, "message")
        content_state, visible_content = _visible_content(message)
        raw_usage = getattr(response, "usage", None)
        usage = TokenUsage(
            prompt_tokens=int(_field(raw_usage, "prompt_tokens", 0) or 0),
            completion_tokens=int(_field(raw_usage, "completion_tokens", 0) or 0),
            total_tokens=int(_field(raw_usage, "total_tokens", 0) or 0),
        )
        observed = ModelResponse(
            content="",
            usage=usage,
            request_id=getattr(response, "id", None),
            seed_requested=request.seed,
            provider_seed_guarantee=(
                "requested_not_guaranteed" if request.seed is not None else None
            ),
            finish_reason=_field(first_choice, "finish_reason"),
            response_model=getattr(response, "model", None),
            content_state=content_state,
            visible_content=visible_content,
            refusal_present=bool(_field(message, "refusal")),
            reasoning_present=bool(
                _field(message, "reasoning_content") or _field(message, "reasoning")
            ),
            tool_calls_present=bool(_field(message, "tool_calls")),
            reasoning_tokens=_optional_count(
                _field(
                    _field(raw_usage, "completion_tokens_details"),
                    "reasoning_tokens",
                )
            ),
            cached_prompt_tokens=_optional_count(
                _field(_field(raw_usage, "prompt_tokens_details"), "cached_tokens")
            ),
            usage_reported=raw_usage is not None,
        )
        error = getattr(response, "error", None)
        if error:
            if isinstance(error, Mapping):
                detail = error.get("message", "unknown API error")
            else:
                detail = str(error)
            raise ModelCallError(
                detail,
                diagnostics={
                    "provider_error_code": _field(error, "code"),
                    "provider_error_type": _field(error, "type"),
                },
                response=observed,
            )
        if not choices:
            raise ModelCallError("API response has no choices", response=observed)
        if observed.refusal_present:
            raise ModelCallError(
                "assistant refused the request",
                diagnostics={"provider_error_code": "assistant_refusal"},
                response=observed,
            )
        if observed.content_state in {"missing", "null", "empty", "unsupported"}:
            code = (
                "unexpected_tool_calls"
                if observed.tool_calls_present
                else "empty_assistant_content"
            )
            raise ModelCallError(
                "API response has no visible assistant content "
                f"(state={observed.content_state}, "
                f"finish_reason={observed.finish_reason or 'not_reported'})",
                diagnostics={"provider_error_code": code},
                response=observed,
            )
        return replace(
            observed,
            content=_message_content(choices[0].message),
        )


def derive_request_seed(
    base_seed: int,
    metadata: Mapping[str, Any],
    *,
    occurrence: int = 0,
) -> int:
    """Derive a stable provider-compatible seed from call identity metadata."""
    payload = json.dumps(
        {
            "base_seed": base_seed,
            "metadata": {
                key: metadata[key]
                for key in _SEED_METADATA_KEYS
                if key in metadata
            },
            "occurrence": occurrence,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big") & 0x7FFFFFFF


class ExecutionBudget:
    """Thread-safe run-wide accounting at model request boundaries."""

    def __init__(
        self,
        *,
        max_calls: int,
        max_tokens: int,
        max_seconds: float,
        clock: Callable[[], float] = perf_counter,
        on_exhausted: Callable[[str, Mapping[str, Any]], None] | None = None,
    ) -> None:
        self.max_calls = max_calls
        self.max_tokens = max_tokens
        self.max_seconds = max_seconds
        self._clock = clock
        self._on_exhausted = on_exhausted
        self._started = clock()
        self._calls_used = 0
        self._tokens_used = 0
        self._tokens_reserved = 0
        self._lock = Lock()

    @property
    def elapsed_seconds(self) -> float:
        return max(0.0, self._clock() - self._started)

    def snapshot(self) -> dict[str, float | int]:
        with self._lock:
            return self._snapshot_unlocked()

    def _snapshot_unlocked(self) -> dict[str, float | int]:
        elapsed = self.elapsed_seconds
        return {
            "calls_used": self._calls_used,
            "tokens_used": self._tokens_used,
            "tokens_reserved": self._tokens_reserved,
            "elapsed_seconds": elapsed,
            "max_calls": self.max_calls,
            "max_tokens": self.max_tokens,
            "max_seconds": self.max_seconds,
            "calls_remaining": max(0, self.max_calls - self._calls_used),
            "tokens_remaining": max(
                0,
                self.max_tokens - self._tokens_used - self._tokens_reserved,
            ),
            "seconds_remaining": max(0.0, self.max_seconds - elapsed),
        }

    def _capacity_unlocked(
        self,
        request_tokens: int,
        *,
        reserve_calls_after: int,
        reserve_tokens_after: int,
        details: Mapping[str, Any] | None = None,
    ) -> tuple[str | None, dict[str, Any]]:
        diagnostics: dict[str, Any] = self._snapshot_unlocked()
        diagnostics.update(
            {
                "next_request_budget_tokens": request_tokens,
                "reserve_calls_after": reserve_calls_after,
                "reserve_tokens_after": reserve_tokens_after,
                **(details or {}),
            }
        )
        reason = None
        if self.elapsed_seconds >= self.max_seconds:
            reason = "max_seconds"
        elif self._calls_used + 1 + reserve_calls_after > self.max_calls:
            reason = "max_calls"
        elif (
            self._tokens_used
            + self._tokens_reserved
            + request_tokens
            + reserve_tokens_after
            > self.max_tokens
        ):
            reason = "max_tokens"
        return reason, diagnostics

    def capacity(
        self,
        request_tokens: int,
        *,
        reserve_calls_after: int = 0,
        reserve_tokens_after: int = 0,
        details: Mapping[str, Any] | None = None,
    ) -> tuple[bool, str | None, dict[str, Any]]:
        if min(request_tokens, reserve_calls_after, reserve_tokens_after) < 0:
            raise ValueError("budget reservations cannot be negative")
        with self._lock:
            reason, diagnostics = self._capacity_unlocked(
                request_tokens,
                reserve_calls_after=reserve_calls_after,
                reserve_tokens_after=reserve_tokens_after,
                details=details,
            )
            return reason is None, reason, diagnostics

    def reserve(
        self,
        request_tokens: int,
        *,
        reserve_calls_after: int = 0,
        reserve_tokens_after: int = 0,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        if min(request_tokens, reserve_calls_after, reserve_tokens_after) < 0:
            raise ValueError("budget reservations cannot be negative")
        with self._lock:
            reason, diagnostics = self._capacity_unlocked(
                request_tokens,
                reserve_calls_after=reserve_calls_after,
                reserve_tokens_after=reserve_tokens_after,
                details=details,
            )
            if reason is not None:
                if self._on_exhausted is not None:
                    self._on_exhausted(reason, diagnostics)
                raise BudgetExceededError(reason, diagnostics=diagnostics)
            self._calls_used += 1
            self._tokens_reserved += request_tokens

    def settle(self, request_tokens: int, usage: TokenUsage) -> None:
        with self._lock:
            self._tokens_reserved -= request_tokens
            self._tokens_used += max(0, usage.total_tokens)
            if self._tokens_used > self.max_tokens:
                diagnostics = self._snapshot_unlocked()
                diagnostics["response_total_tokens"] = usage.total_tokens
                if self._on_exhausted is not None:
                    self._on_exhausted("max_tokens", diagnostics)
                raise BudgetExceededError(
                    "max_tokens",
                    diagnostics=diagnostics,
                    usage=usage,
                )

    def remaining_seconds(self) -> float:
        return max(0.0, self.max_seconds - self.elapsed_seconds)


def _complete_with_deadline(
    model: ChatModel,
    request: ModelRequest,
    timeout_seconds: float,
    diagnostics: Mapping[str, Any],
) -> ModelResponse:
    result: queue.Queue[tuple[bool, Any]] = queue.Queue(maxsize=1)

    def invoke() -> None:
        try:
            result.put((True, model.complete(request)))
        except BaseException as exc:
            result.put((False, exc))

    worker = threading.Thread(target=invoke, daemon=True)
    worker.start()
    try:
        succeeded, value = result.get(timeout=timeout_seconds)
    except queue.Empty as exc:
        raise BudgetExceededError(
            "max_seconds",
            diagnostics={
                **diagnostics,
                "deadline_exceeded_during_call": True,
                "model_invoked": True,
                "provider_may_still_be_running": worker.is_alive(),
                "request_timeout_seconds": timeout_seconds,
            },
        ) from exc
    if not succeeded:
        raise value
    if not isinstance(value, ModelResponse):
        raise TypeError("model.complete must return ModelResponse")
    return value


class BudgetedSeededChatModel:
    """Apply one run budget and deterministic per-call seeds to a model."""

    provider_seed_guarantee = "requested_not_guaranteed"

    def __init__(
        self,
        model: ChatModel,
        budget: ExecutionBudget,
        *,
        base_seed: int,
    ) -> None:
        self.model = model
        self.budget = budget
        self.base_seed = base_seed
        self._occurrences: dict[str, int] = {}
        self._lock = Lock()

    def _seed(self, metadata: Mapping[str, Any]) -> int:
        identity = json.dumps(
            {
                key: metadata[key]
                for key in _SEED_METADATA_KEYS
                if key in metadata
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        with self._lock:
            occurrence = self._occurrences.get(identity, 0)
            self._occurrences[identity] = occurrence + 1
        return derive_request_seed(
            self.base_seed,
            metadata,
            occurrence=occurrence,
        )

    def complete(self, request: ModelRequest) -> ModelResponse:
        seed = self._seed(request.metadata)
        reserve_calls_after = int(
            request.metadata.get("budget_reserve_calls_after", 0)
        )
        reserve_tokens_after = int(
            request.metadata.get("budget_reserve_tokens_after", 0)
        )
        request_tokens = max(
            request.max_tokens,
            request.budget_tokens or request.max_tokens,
        )
        remaining_seconds = self.budget.remaining_seconds()
        timeout_seconds = min(
            remaining_seconds,
            (
                request.timeout_seconds
                if request.timeout_seconds is not None
                else remaining_seconds
            ),
        )
        seeded_request = ModelRequest(
            model=request.model,
            messages=request.messages,
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=request.max_tokens,
            seed=seed,
            metadata=request.metadata,
            timeout_seconds=timeout_seconds,
            budget_tokens=request_tokens,
            response_format=request.response_format,
            reasoning_effort=request.reasoning_effort,
        )
        self.budget.reserve(
            request_tokens,
            reserve_calls_after=reserve_calls_after,
            reserve_tokens_after=reserve_tokens_after,
            details={
                "agent": str(request.metadata.get("agent", "unknown")),
                "next_request_max_tokens": seeded_request.max_tokens,
                "seed_requested": seed,
                "provider_seed_guarantee": self.provider_seed_guarantee,
            },
        )
        response: ModelResponse | None = None
        try:
            response = _complete_with_deadline(
                self.model,
                seeded_request,
                timeout_seconds,
                self.budget.snapshot(),
            )
            response = replace(
                response,
                seed_requested=seed,
                provider_seed_guarantee=self.provider_seed_guarantee,
            )
            return response
        except Exception as exc:
            error_response = getattr(exc, "response", None)
            if response is None and isinstance(error_response, ModelResponse):
                response = error_response
            diagnostics = getattr(exc, "diagnostics", None)
            if isinstance(diagnostics, dict):
                diagnostics.setdefault("seed_requested", seed)
                diagnostics.setdefault(
                    "provider_seed_guarantee", self.provider_seed_guarantee
                )
            raise
        finally:
            try:
                self.budget.settle(
                    request_tokens,
                    (
                        TokenUsage(total_tokens=request_tokens)
                        if response is not None and response.usage_reported is False
                        else (
                            response.usage
                            if response is not None
                            else TokenUsage()
                        )
                    ),
                )
            except BudgetExceededError as exc:
                exc.response = response
                exc.diagnostics.update(
                    agent=str(request.metadata.get("agent", "unknown")),
                    seed_requested=seed,
                    provider_seed_guarantee=self.provider_seed_guarantee,
                )
                raise
