from __future__ import annotations

from dataclasses import dataclass
from typing import Any

BLOCKED_MESSAGE = "This request could not be processed automatically and has been escalated."


@dataclass(frozen=True)
class GuardrailResult:
    text: str
    intervened: bool = False
    blocked: bool = False


def _contains_blocked(node: Any) -> bool:
    """True if any assessment entry carries action == BLOCKED (anonymized spans are not blocks)."""
    if isinstance(node, dict):
        if node.get("action") == "BLOCKED":
            return True
        return any(_contains_blocked(v) for v in node.values())
    if isinstance(node, list):
        return any(_contains_blocked(v) for v in node)
    return False


class GuardrailService:
    """Amazon Bedrock Guardrails applied independently of any model via ApplyGuardrail."""

    def __init__(self, client: Any, guardrail_id: str, version: str = "DRAFT"):
        self._client = client
        self._id = guardrail_id
        self._version = version

    def apply(self, text: str, source: str) -> GuardrailResult:
        """source is 'INPUT' for user text or 'OUTPUT' for model text."""
        if not text.strip():
            return GuardrailResult(text)
        resp = self._client.apply_guardrail(
            guardrailIdentifier=self._id,
            guardrailVersion=self._version,
            source=source,
            content=[{"text": {"text": text}}],
        )
        if resp.get("action") != "GUARDRAIL_INTERVENED":
            return GuardrailResult(text)
        blocked = _contains_blocked(resp.get("assessments", []))
        outputs = resp.get("outputs", [])
        masked = outputs[0]["text"] if outputs and "text" in outputs[0] else text
        return GuardrailResult(BLOCKED_MESSAGE if blocked else masked, True, blocked)
