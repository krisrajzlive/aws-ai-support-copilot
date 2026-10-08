from __future__ import annotations

import json
import re
from typing import Any

from botocore.exceptions import ClientError


def parse_json_object(text: str) -> dict[str, Any]:
    """Extract the first JSON object from a model reply, tolerating code fences and prose."""
    cleaned = re.sub(r"```(?:json)?", "", text)
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("no JSON object in model reply")
    return json.loads(cleaned[start : end + 1])


class BedrockService:
    """Thin Converse wrapper that walks a priority list of models until one answers."""

    def __init__(self, client: Any, model_ids: list[str]):
        self._client = client
        self._model_ids = model_ids
        self.model_id = ""

    def complete(self, system: str, user: str, max_tokens: int = 900) -> str:
        last_error: ClientError | None = None
        for model_id in self._model_ids:
            try:
                resp = self._client.converse(
                    modelId=model_id,
                    system=[{"text": system}],
                    messages=[{"role": "user", "content": [{"text": user}]}],
                    inferenceConfig={"maxTokens": max_tokens, "temperature": 0.2},
                )
            except ClientError as exc:
                last_error = exc
                continue
            self.model_id = model_id
            blocks = resp["output"]["message"]["content"]
            return "".join(b["text"] for b in blocks if "text" in b).strip()
        raise RuntimeError(f"no Bedrock model responded: {last_error}")

    def complete_json(self, system: str, user: str) -> dict[str, Any]:
        return parse_json_object(self.complete(system, user))
