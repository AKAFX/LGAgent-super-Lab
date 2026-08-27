"""Replaceable model-call protocol and OpenAI-compatible adapter."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
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


class ModelCallError(RuntimeError):
    """Raised when a model provider cannot produce a response."""


class BudgetExceededError(RuntimeError):
    """Raised before a model request that would exceed a hard run budget."""

    def __init__(
        self,
        reason: str,
        *,
        diagnostics: Mapping[str, Any] | None = None,
        usage: TokenUsage | None = None,
    ) -> None:
        self.reason = reason
        self.diagnostics = dict(diagnostics or {})
        self.usage = usage or TokenUsage()
        super().__init__(f"execution budget exhausted: {reason}")


class ChatModel(Protocol):
    def complete(self, request: ModelRequest) -> ModelResponse:
        """Complete one chat request."""


def _message_content(message: Any) -> str:
    content = getattr(message, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, Mapping) and isinstance(part.get("text"), str):
                parts.append(part["text"])
        if parts:
            return "\n".join(parts)

    tool_calls = getattr(message, "tool_calls", None)
    if tool_calls:
        return json.dumps(
            [
                call.model_dump() if hasattr(call, "model_dump") else str(call)
                for call in tool_calls
            ],
            ensure_ascii=False,
        )
    if hasattr(message, "model_dump"):
        return json.dumps(message.model_dump(), ensure_ascii=False)
    return str(content or "")


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
            response = self._client.chat.completions.create(
                **arguments,
            )
        except Exception as exc:
            raise ModelCallError(f"{type(exc).__name__}: {exc}") from exc

        error = getattr(response, "error", None)
        if error:
            if isinstance(error, Mapping):
                detail = error.get("message", "unknown API error")
            else:
                detail = str(error)
            raise ModelCallError(detail)
        choices: Sequence[Any] = getattr(response, "choices", ())
        if not choices:
            raise ModelCallError("API response has no choices")

        raw_usage = getattr(response, "usage", None)
        usage = TokenUsage(
            prompt_tokens=int(getattr(raw_usage, "prompt_tokens", 0) or 0),
            completion_tokens=int(getattr(raw_usage, "completion_tokens", 0) or 0),
            total_tokens=int(getattr(raw_usage, "total_tokens", 0) or 0),
        )
        return ModelResponse(
            content=_message_content(choices[0].message),
            usage=usage,
            request_id=getattr(response, "id", None),
            seed_requested=request.seed,
            provider_seed_guarantee=(
                "requested_not_guaranteed" if request.seed is not None else None
            ),
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
        return {
            "calls_used": self._calls_used,
            "tokens_used": self._tokens_used,
            "tokens_reserved": self._tokens_reserved,
            "elapsed_seconds": self.elapsed_seconds,
            "max_calls": self.max_calls,
            "max_tokens": self.max_tokens,
            "max_seconds": self.max_seconds,
        }

    def reserve(
        self,
        max_tokens: int,
        *,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        with self._lock:
            reason = None
            if self.elapsed_seconds >= self.max_seconds:
                reason = "max_seconds"
            elif self._calls_used + 1 > self.max_calls:
                reason = "max_calls"
            elif (
                self._tokens_used + self._tokens_reserved + max_tokens
                > self.max_tokens
            ):
                reason = "max_tokens"
            if reason is not None:
                diagnostics = self._snapshot_unlocked()
                diagnostics["next_request_max_tokens"] = max_tokens
                diagnostics.update(details or {})
                if self._on_exhausted is not None:
                    self._on_exhausted(reason, diagnostics)
                raise BudgetExceededError(reason, diagnostics=diagnostics)
            self._calls_used += 1
            self._tokens_reserved += max_tokens

    def settle(self, max_tokens: int, usage: TokenUsage) -> None:
        with self._lock:
            self._tokens_reserved -= max_tokens
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
        seeded_request = ModelRequest(
            model=request.model,
            messages=request.messages,
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=request.max_tokens,
            seed=seed,
            metadata=request.metadata,
        )
        self.budget.reserve(
            seeded_request.max_tokens,
            details={
                "agent": str(request.metadata.get("agent", "unknown")),
                "seed_requested": seed,
                "provider_seed_guarantee": self.provider_seed_guarantee,
            },
        )
        response: ModelResponse | None = None
        try:
            response = self.model.complete(seeded_request)
            return ModelResponse(
                content=response.content,
                usage=response.usage,
                request_id=response.request_id,
                seed_requested=seed,
                provider_seed_guarantee=self.provider_seed_guarantee,
            )
        finally:
            self.budget.settle(
                seeded_request.max_tokens,
                response.usage if response is not None else TokenUsage(),
            )
