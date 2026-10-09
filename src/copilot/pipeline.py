"""Entry points over the LangGraph workflow in copilot.graph."""

from __future__ import annotations

from typing import Any

from copilot.aws import CLIENT_CONFIG
from copilot.config import Settings
from copilot.graph import CaseGraph
from copilot.knowledge import KnowledgeBase
from copilot.models import SupportCase
from copilot.personas import PersonaSet
from copilot.services.speech import synthesize_reply


def analyze_case(
    session: Any,
    settings: Settings,
    *,
    text: str = "",
    document: bytes | None = None,
    image: bytes | None = None,
    audio: bytes | None = None,
    audio_format: str = "wav",
    reply_language: str | None = None,
    persona: str | None = None,
    personas: PersonaSet | None = None,
    knowledge: KnowledgeBase | None = None,
) -> SupportCase:
    """Run one support request to completion (no human review pause)."""
    workflow = CaseGraph(session, settings, personas=personas, knowledge=knowledge)
    run = workflow.start(
        text=text,
        document=document,
        image=image,
        audio=audio,
        audio_format=audio_format,
        reply_language=reply_language,
        persona=persona,
    )
    if run.case is None:
        raise RuntimeError(
            "This case needs human review. Use CaseGraph.start/resume, or turn off "
            "COPILOT_REQUIRE_ESCALATION_REVIEW."
        )
    return run.case


def speak_reply(session: Any, settings: Settings, case: SupportCase) -> bytes:
    """Spoken MP3 of the drafted reply; only available when COPILOT_TTS_BACKEND=polly."""
    if settings.tts_backend != "polly":
        raise RuntimeError("Spoken replies are off. Set COPILOT_TTS_BACKEND=polly to enable them.")
    polly = session.client("polly", config=CLIENT_CONFIG)
    return synthesize_reply(polly, case.reply, case.reply_language or "en", settings.polly_voice)
