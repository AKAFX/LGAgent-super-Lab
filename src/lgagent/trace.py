"""Structured, secret-free execution traces for LGAgent runs."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from threading import Lock
from typing import Any, Mapping
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
    _lock: Lock = field(default_factory=Lock, init=False, repr=False, compare=False)

    def add_call(self, call: ModelCallTrace) -> None:
        with self._lock:
            self.model_calls.append(call)

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
