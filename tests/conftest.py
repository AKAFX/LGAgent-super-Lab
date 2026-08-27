from __future__ import annotations

import socket

import pytest


@pytest.fixture(autouse=True)
def deny_network_access(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail every pytest immediately if production code attempts network I/O."""

    def denied(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("network access is forbidden in the test suite")

    monkeypatch.setattr(socket, "create_connection", denied)
    monkeypatch.setattr(socket.socket, "connect", denied)
