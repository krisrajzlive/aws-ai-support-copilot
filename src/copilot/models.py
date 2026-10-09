from __future__ import annotations

from pydantic import BaseModel, Field


class SupportCase(BaseModel):
    """Structured result of analysing one inbound support request."""

    case_id: str = ""
    human_reviewed: bool = False
    archive_key: str = ""

    source_language: str = "en"
    original_text: str = ""
    english_text: str = ""
    sentiment: str | None = None
    entities: list[str] = Field(default_factory=list)
    pii_types: list[str] = Field(default_factory=list)
    document_lines: list[str] = Field(default_factory=list)
    image_labels: list[str] = Field(default_factory=list)
    summary: str = ""
    category: str = ""
    category_source: str = ""
    category_confidence: float | None = None
    persona: str = ""
    persona_reason: str = ""
    priority: str = ""
    reply_redacted: str = ""  # the reply before customer details are restored; safe to archive
    policy_sources: list[str] = Field(default_factory=list)
    policy_excerpts: list[str] = Field(default_factory=list)
    reply_en: str = ""
    reply: str = ""
    reply_language: str = ""
    blocked: bool = False
    model_id: str = ""
    models: dict[str, str] = Field(default_factory=dict)
    backends: dict[str, str] = Field(default_factory=dict)
