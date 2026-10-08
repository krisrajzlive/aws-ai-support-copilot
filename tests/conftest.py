from __future__ import annotations

from typing import Any

from botocore.exceptions import ClientError


def client_error(code: str, operation: str = "Op") -> ClientError:
    return ClientError({"Error": {"Code": code, "Message": "x"}}, operation)


class FakeClient:
    """Any method succeeds unless a behavior is registered for it (exception or callable)."""

    def __init__(self, behaviors: dict[str, Any]):
        self._behaviors = behaviors

    def __getattr__(self, method: str):
        def call(**kwargs: Any):
            behavior = self._behaviors.get(method)
            if isinstance(behavior, Exception):
                raise behavior
            if callable(behavior):
                return behavior(**kwargs)
            return {}

        return call


class FakeSession:
    """Stands in for boto3.Session; behaviors are keyed by (client_name, method_name)."""

    def __init__(self, behaviors: dict[tuple[str, str], Any] | None = None):
        self._behaviors = behaviors or {}

    def client(self, name: str, **_: Any) -> FakeClient:
        return FakeClient({m: b for (c, m), b in self._behaviors.items() if c == name})
