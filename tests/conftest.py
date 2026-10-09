from __future__ import annotations

from typing import Any

import pytest
from botocore.exceptions import ClientError

from copilot.config import Settings


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


def converse_response(text: str) -> dict:
    """A Converse reply complete enough for both our parsers and langchain-aws."""
    return {
        "output": {"message": {"role": "assistant", "content": [{"text": text}]}},
        "stopReason": "end_turn",
        "usage": {"inputTokens": 1, "outputTokens": 1, "totalTokens": 2},
        "ResponseMetadata": {"HTTPStatusCode": 200},
    }


def router_converse(triage: dict, reply: str, calls: list | None = None):
    """Fake Converse: JSON for the triage prompt, plain text for a persona reply."""
    import json

    def call(**kw):
        if calls is not None:
            calls.append(kw)
        is_triage = "triage assistant" in kw["system"][0]["text"]
        text = json.dumps(triage) if is_triage else reply
        return converse_response(text)

    return call


@pytest.fixture(autouse=True)
def isolate_settings_from_dotenv(monkeypatch):
    """Tests must not depend on the developer's local .env (bucket, bot id, feature flags)."""
    monkeypatch.setitem(Settings.model_config, "env_file", None)
    for name in [n for n in __import__("os").environ if n.startswith("COPILOT_")]:
        monkeypatch.delenv(name)


class FakeS3:
    """In-memory S3 with just the calls the storage helpers use."""

    def __init__(self, objects: dict[str, bytes] | None = None):
        self.objects: dict[str, bytes] = dict(objects or {})

    def put_object(self, Bucket, Key, Body, **_):
        self.objects[Key] = Body if isinstance(Body, bytes) else Body.encode()

    def get_object(self, Bucket, Key):
        import io

        return {"Body": io.BytesIO(self.objects[Key])}

    def list_objects_v2(self, Bucket, Prefix, **_):
        return {"Contents": [{"Key": k} for k in sorted(self.objects) if k.startswith(Prefix)]}

    def delete_objects(self, Bucket, Delete):
        for item in Delete["Objects"]:
            self.objects.pop(item["Key"], None)
