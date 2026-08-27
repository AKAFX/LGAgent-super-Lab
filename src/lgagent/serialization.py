"""Stable JSON serialization helpers with secret redaction."""

from __future__ import annotations

import dataclasses
import json
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import Any, Mapping

_SECRET_FIELDS = frozenset(
    {"api_key", "apikey", "authorization", "password", "secret", "token"}
)


def to_jsonable(value: Any, *, redact_secrets: bool = True) -> Any:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        value = {
            field.name: getattr(value, field.name)
            for field in dataclasses.fields(value)
            if not field.name.startswith("_")
        }
    if isinstance(value, Mapping):
        result = {}
        for key, item in value.items():
            name = str(key)
            if redact_secrets and name.lower() in _SECRET_FIELDS and item:
                result[name] = "[REDACTED]"
            else:
                result[name] = to_jsonable(item, redact_secrets=redact_secrets)
        return result
    if isinstance(value, (list, tuple, set, frozenset)):
        return [to_jsonable(item, redact_secrets=redact_secrets) for item in value]
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"cannot serialize value of type {type(value).__name__}")


def dumps_json(
    value: Any,
    *,
    indent: int | None = 2,
    redact_secrets: bool = True,
) -> str:
    return json.dumps(
        to_jsonable(value, redact_secrets=redact_secrets),
        ensure_ascii=False,
        indent=indent,
        sort_keys=True,
    )
