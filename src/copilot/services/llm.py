"""Text generation through LangChain's ChatBedrockConverse with an ordered model fallback chain."""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from typing import Any

from langchain_aws import ChatBedrockConverse
from langchain_core.messages import HumanMessage, SystemMessage

_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)


def parse_json_object(text: str) -> dict[str, Any]:
    """Extract the first JSON object from a model reply, tolerating code fences and prose."""
    cleaned = re.sub(r"```(?:json)?", "", text)
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("no JSON object in model reply")
    return json.loads(cleaned[start : end + 1])


def message_text(content: Any) -> str:
    """Text of a chat message; reasoning blocks from thinking models are dropped."""
    if isinstance(content, str):
        return content
    parts = []
    for block in content or []:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text", ""))
    return "".join(parts)


class ChatLLM:
    """Walks a priority list of Bedrock models; the first that answers wins.

    maxTokens is always set explicitly so a call never reserves a model's full output quota.
    """

    def __init__(self, client: Any, model_ids: Sequence[str], region_name: str = "us-east-1"):
        if not model_ids:
            raise ValueError("at least one model id is required")
        self._client = client
        self._model_ids = list(model_ids)
        self._region = region_name
        self.model_id = ""

    def _chain(self, max_tokens: int, temperature: float):
        models = [
            ChatBedrockConverse(
                model=model_id,
                client=self._client,
                region_name=self._region,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            for model_id in self._model_ids
        ]
        return models[0].with_fallbacks(models[1:]) if len(models) > 1 else models[0]

    def complete(
        self, system: str, user: str, max_tokens: int = 900, temperature: float = 0.2
    ) -> str:
        try:
            reply = self._chain(max_tokens, temperature).invoke(
                [SystemMessage(content=system), HumanMessage(content=user)]
            )
        except Exception as exc:  # every model in the chain failed
            raise RuntimeError(f"no Bedrock model responded: {exc}") from exc
        self.model_id = str(reply.response_metadata.get("model_name", self._model_ids[0]))
        return _THINK.sub("", message_text(reply.content)).strip()

    def complete_json(
        self, system: str, user: str, max_tokens: int = 600, temperature: float = 0.0
    ) -> dict[str, Any]:
        return parse_json_object(self.complete(system, user, max_tokens, temperature))
