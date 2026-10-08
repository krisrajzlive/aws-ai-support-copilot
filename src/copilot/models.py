from __future__ import annotations

from pydantic import BaseModel, Field


class SupportCase(BaseModel):
    """Structured result of analysing one inbound support request."""

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
    priority: str = ""
    reply_en: str = ""
    reply: str = ""
    reply_language: str = ""
    blocked: bool = False
    model_id: str = ""
    backends: dict[str, str] = Field(default_factory=dict)
