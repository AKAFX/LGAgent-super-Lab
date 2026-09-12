"""Credential-safe call journaling for LegalMCQ, separate from model context."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from threading import Lock
from typing import Any, Mapping, Sequence

from ..model import BudgetExceededError, ModelCallError, ModelResponse
from ..serialization import to_jsonable


def redact_telemetry(value: Any, secrets: Sequence[str]) -> Any:
    """Redact known credential values as well as structured credential fields."""
    keys = sorted({key for key in secrets if key}, key=len, reverse=True)

    def clean(item: Any) -> Any:
        if isinstance(item, str):
            for key in keys:
                item = item.replace(key, "[REDACTED]")
            return item
        if isinstance(item, dict):
            return {clean(key): clean(value) for key, value in item.items()}
        if isinstance(item, list):
            return [clean(value) for value in item]
        return item

    return clean(to_jsonable(value))


def response_telemetry(
    response: ModelResponse | None,
    *,
    error: BaseException | None,
    schema_checked: bool,
    log_model_output: bool,
    secrets: Sequence[str],
) -> dict[str, Any]:
    """Describe the provider response without changing parsing or retry decisions."""
    diagnostics = getattr(error, "diagnostics", {})
    if not isinstance(diagnostics, Mapping):
        diagnostics = {}
    if response is None:
        response = getattr(error, "response", None)
    blocked = isinstance(error, BudgetExceededError) and (
        "next_request_max_tokens" in diagnostics
    ) and not diagnostics.get("deadline_exceeded_during_call")
    deadline_exceeded = bool(diagnostics.get("deadline_exceeded_during_call"))
    if error is None:
        outcome = "success"
    elif deadline_exceeded:
        outcome = "deadline_exceeded"
    elif blocked:
        outcome = "budget_blocked"
    elif isinstance(error, BudgetExceededError):
        outcome = "budget_exceeded_after_response"
    elif isinstance(error, ModelCallError):
        outcome = "provider_error"
    elif not isinstance(error, Exception):
        outcome = "cancelled"
    elif schema_checked:
        outcome = "schema_error"
    else:
        outcome = "error"

    fields: dict[str, Any] = {
        "outcome": outcome,
        "schema_valid": error is None if schema_checked else None,
        "model_invoked": diagnostics.get("model_invoked", not blocked),
        "http_status": diagnostics.get("http_status"),
        "provider_error_code": diagnostics.get("provider_error_code"),
        "provider_error_type": diagnostics.get("provider_error_type"),
        "request_id": diagnostics.get("request_id"),
        "seed_requested": diagnostics.get("seed_requested"),
        "provider_seed_guarantee": diagnostics.get("provider_seed_guarantee"),
        "deadline_exceeded_during_call": deadline_exceeded or None,
        "provider_may_still_be_running": diagnostics.get(
            "provider_may_still_be_running"
        ),
        "request_timeout_seconds": diagnostics.get("request_timeout_seconds"),
    }
    if response is not None:
        content_state = response.content_state or (
            "text" if response.content.strip() else "empty"
        )
        # Legacy/Fake responses have no raw-body metadata. A known null/missing
        # body must never fall back to an SDK envelope containing reasoning text.
        text = response.visible_content
        if text is None:
            text = response.content if response.content_state is None else ""
        safe_text = redact_telemetry(text, secrets)
        fields.update(
            request_id=response.request_id or fields["request_id"],
            seed_requested=response.seed_requested,
            provider_seed_guarantee=response.provider_seed_guarantee,
            response_model=response.response_model,
            finish_reason=response.finish_reason,
            content_state=content_state,
            content_chars=len(text),
            content_sha256=hashlib.sha256(safe_text.encode("utf-8")).hexdigest(),
            output_text=safe_text if log_model_output else None,
            refusal_present=response.refusal_present,
            reasoning_present=response.reasoning_present,
            tool_calls_present=response.tool_calls_present,
            reasoning_tokens=response.reasoning_tokens,
            cached_prompt_tokens=response.cached_prompt_tokens,
            usage_reported=response.usage_reported,
        )
    return redact_telemetry(fields, secrets)


class CallJournal:
    """Append and flush each event; share safely across concurrent questions."""

    def __init__(self, path: Path, *, secrets: Sequence[str] = ()) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        self._stream = os.fdopen(descriptor, "w", encoding="utf-8")
        self._secrets = tuple(secrets)
        self._lock = Lock()

    def __enter__(self) -> "CallJournal":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def close(self) -> None:
        with self._lock:
            self._stream.close()

    def write(self, event: Mapping[str, Any]) -> None:
        line = json.dumps(
            redact_telemetry(event, self._secrets),
            ensure_ascii=False,
            sort_keys=True,
        )
        with self._lock:
            self._stream.write(line + "\n")
            self._stream.flush()
