from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from botocore.exceptions import ClientError

SENTIMENT_LANGUAGES = {"en", "es", "fr", "de", "it", "pt", "ar", "hi", "ja", "ko", "zh", "zh-TW"}


@dataclass
class TextInsights:
    sentiment: str | None = None
    entities: list[str] = field(default_factory=list)
    pii_types: list[str] = field(default_factory=list)
    redacted_text: str = ""


def detect_language(client: Any, text: str) -> str:
    try:
        languages = client.detect_dominant_language(Text=text[:5000]).get("Languages", [])
    except ClientError:
        return "en"
    return languages[0]["LanguageCode"] if languages else "en"


def redact(text: str, pii_entities: list[dict[str, Any]]) -> str:
    """Replace each detected PII span with its type, working right-to-left to keep offsets valid."""
    for ent in sorted(pii_entities, key=lambda e: e["BeginOffset"], reverse=True):
        text = text[: ent["BeginOffset"]] + f"[{ent['Type']}]" + text[ent["EndOffset"] :]
    return text


def analyze_english(client: Any, text: str) -> TextInsights:
    """Sentiment, named entities and PII on English text; each call degrades independently."""
    insights = TextInsights(redacted_text=text)
    chunk = text[:4500]
    try:
        pii = client.detect_pii_entities(Text=chunk, LanguageCode="en").get("Entities", [])
        insights.pii_types = sorted({e["Type"] for e in pii})
        insights.redacted_text = redact(chunk, pii) + text[4500:]
    except ClientError:
        pass
    # Everything downstream of PII detection reads the redacted text so no raw value leaks.
    safe = insights.redacted_text[:4500]
    try:
        insights.sentiment = client.detect_sentiment(Text=safe, LanguageCode="en").get("Sentiment")
    except ClientError:
        pass
    try:
        ents = client.detect_entities(Text=safe, LanguageCode="en").get("Entities", [])
        insights.entities = sorted({e["Text"] for e in ents})
    except ClientError:
        pass
    return insights
