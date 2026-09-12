"""Structured, secret-free execution traces for LGAgent runs."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from threading import Lock
from typing import Any, Callable, Mapping
from uuid import uuid4

from .model import TokenUsage
from .serialization import to_jsonable


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class ModelCallTrace:
    call_id: str
    agent: str
    model: str
    attempt: int
    started_at: str
    duration_ms: float
    usage: TokenUsage
    request_id: str | None = None
    seed_requested: int | None = None
    provider_seed_guarantee: str | None = None
    error_type: str | None = None
    error_message: str | None = None
    max_tokens: int | None = None
    budget_tokens: int | None = None
    reserve_calls_after: int = 0
    reserve_tokens_after: int = 0
    timeout_seconds: float | None = None
    response_format_type: str | None = None
    response_schema_name: str | None = None
    reasoning_effort_requested: str | None = None
    visible_output_tokens: int | None = None
    reasoning_allowance_tokens: int | None = None
    completion_token_cap: int | None = None
    extra_reasoning_reserve_tokens: int = 0
    response_model: str | None = None
    finish_reason: str | None = None
    content_state: str | None = None
    content_chars: int | None = None
    content_sha256: str | None = None
    output_text: str | None = None
    refusal_present: bool | None = None
    reasoning_present: bool | None = None
    tool_calls_present: bool | None = None
    reasoning_tokens: int | None = None
    cached_prompt_tokens: int | None = None
    usage_reported: bool | None = None
    outcome: str | None = None
    schema_valid: bool | None = None
    model_invoked: bool | None = None
    http_status: int | None = None
    provider_error_code: str | None = None
    provider_error_type: str | None = None
    deadline_exceeded_during_call: bool | None = None
    provider_may_still_be_running: bool | None = None
    request_timeout_seconds: float | None = None
    budget_before: Mapping[str, Any] = field(default_factory=dict)
    budget_after: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RouteTrace:
    route: str
    reason: str
    occurred_at: str
    details: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class RunTrace:
    run_id: str = field(default_factory=lambda: uuid4().hex)
    started_at: str = field(default_factory=utc_now)
    model_calls: list[ModelCallTrace] = field(default_factory=list)
    routes: list[RouteTrace] = field(default_factory=list)
    errors: list[dict[str, str]] = field(default_factory=list)
    _event_sink: Callable[[Mapping[str, Any]], None] | None = field(
        default=None, repr=False, compare=False
    )
    _event_sequence: int = field(default=0, init=False, repr=False, compare=False)
    _lock: Lock = field(default_factory=Lock, init=False, repr=False, compare=False)

    def emit(self, event: str, details: Mapping[str, Any]) -> None:
        """Publish a correlated event without serializing prompts or callbacks."""
        with self._lock:
            self._event_sequence += 1
            sequence = self._event_sequence
        if self._event_sink is not None:
            self._event_sink(
                to_jsonable(
                    {
                        **details,
                        "trace_schema_version": 1,
                        "event": event,
                        "run_id": self.run_id,
                        "sequence": sequence,
                        "occurred_at": utc_now(),
                    }
                )
            )

    def add_call(self, call: ModelCallTrace) -> None:
        with self._lock:
            self.model_calls.append(call)
        self.emit("model_call_finished", to_jsonable(call))

    def add_route(
        self,
        route: str,
        reason: str,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        event = RouteTrace(route, reason, utc_now(), details or {})
        with self._lock:
            self.routes.append(event)

    def add_error(
        self,
        stage: str,
        error: BaseException,
        *,
        message: str | None = None,
    ) -> None:
        event = {
            "stage": stage,
            "error_type": type(error).__name__,
            "message": str(error) if message is None else message,
            "occurred_at": utc_now(),
        }
        with self._lock:
            self.errors.append(event)

    @property
    def total_tokens(self) -> int:
        return sum(call.usage.total_tokens for call in self.model_calls)

    def as_dict(self) -> dict[str, Any]:
        return to_jsonable(
            {
                "run_id": self.run_id,
                "started_at": self.started_at,
                "model_calls": self.model_calls,
                "routes": self.routes,
                "errors": self.errors,
                "total_tokens": self.total_tokens,
            }
        )
