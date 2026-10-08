from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any


@dataclass
class LexTurn:
    messages: list[str] = field(default_factory=list)
    intent: str | None = None
    state: str | None = None
    action: str | None = None
    slots: dict[str, str] = field(default_factory=dict)

    @property
    def fulfilled(self) -> bool:
        return self.state == "Fulfilled"


def _slot_values(raw: dict[str, Any] | None) -> dict[str, str]:
    """Filled slots only, preferring what the customer typed over Lex's normalised value."""
    values: dict[str, str] = {}
    for name, slot in (raw or {}).items():
        value = (slot or {}).get("value") or {}
        text = value.get("originalValue") or value.get("interpretedValue")
        if text:
            values[name] = text
    return values


class LexIntake:
    """One conversation with the Amazon Lex V2 intake bot."""

    def __init__(
        self,
        runtime: Any,
        bot_id: str,
        alias_id: str = "TSTALIASID",
        locale: str = "en_US",
        session_id: str | None = None,
    ):
        self._runtime = runtime
        self._ids = {"botId": bot_id, "botAliasId": alias_id, "localeId": locale}
        self.session_id = session_id or uuid.uuid4().hex
        self.user_turns: list[str] = []

    def send(self, text: str) -> LexTurn:
        self.user_turns.append(text)
        resp = self._runtime.recognize_text(sessionId=self.session_id, text=text, **self._ids)
        state = resp.get("sessionState", {})
        intent = state.get("intent", {})
        return LexTurn(
            messages=[m["content"] for m in resp.get("messages", []) if m.get("content")],
            intent=intent.get("name"),
            state=intent.get("state"),
            action=state.get("dialogAction", {}).get("type"),
            slots=_slot_values(intent.get("slots")),
        )

    def case_text(self, turn: LexTurn) -> str:
        """Everything the customer said, plus the structured slots, as the pipeline's input text."""
        said = [t for t in self.user_turns if t.strip().lower() not in {"yes", "no", "y", "n"}]
        facts = ", ".join(f"{k} {v}" for k, v in turn.slots.items())
        return f"{' '.join(said)} ({facts})" if facts else " ".join(said)
