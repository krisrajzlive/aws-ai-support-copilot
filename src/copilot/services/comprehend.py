from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from botocore.exceptions import ClientError

SENTIMENT_LANGUAGES = {"en", "es", "fr", "de", "it", "pt", "ar", "hi", "ja", "ko", "zh", "zh-TW"}

_TOKEN = re.compile(r"\[\[([A-Z_]+)_(\d+)\]\]")


@dataclass
class TextInsights:
    sentiment: str | None = None
    entities: list[str] = field(default_factory=list)
    pii_types: list[str] = field(default_factory=list)
    redacted_text: str = ""
    # token such as "[[NAME_1]]" -> the original value, kept so replies can be personalised
    placeholders: dict[str, str] = field(default_factory=dict)


def detect_language(client: Any, text: str) -> str:
    try:
        languages = client.detect_dominant_language(Text=text[:5000]).get("Languages", [])
    except ClientError:
        return "en"
    return languages[0]["LanguageCode"] if languages else "en"


def redact(text: str, pii_entities: list[dict[str, Any]]) -> tuple[str, dict[str, str]]:
    """Replace each PII span with a numbered token, reusing one token per distinct value.

    Returns the redacted text and the token-to-original mapping.
    """
    ordered = sorted(pii_entities, key=lambda e: e["BeginOffset"])
    tokens: dict[tuple[str, str], str] = {}
    counts: dict[str, int] = {}
    for ent in ordered:
        value = text[ent["BeginOffset"] : ent["EndOffset"]]
        key = (ent["Type"], value)
        if key not in tokens:
            counts[ent["Type"]] = counts.get(ent["Type"], 0) + 1
            tokens[key] = f"[[{ent['Type']}_{counts[ent['Type']]}]]"
    for ent in reversed(ordered):
        value = text[ent["BeginOffset"] : ent["EndOffset"]]
        text = text[: ent["BeginOffset"]] + tokens[(ent["Type"], value)] + text[ent["EndOffset"] :]
    return text, {token: value for (_, value), token in tokens.items()}


def fill_placeholders(text: str, placeholders: dict[str, str]) -> str:
    """Put original values back into model output; unknown tokens are left as they are."""
    return _TOKEN.sub(lambda m: placeholders.get(m.group(0), m.group(0)), text)


def _is_token_artifact(entity_text: str) -> bool:
    return "[[" in entity_text or re.fullmatch(r"[A-Z_]+_\d+\]*", entity_text) is not None


def analyze_english(client: Any, text: str) -> TextInsights:
    """Sentiment, named entities and PII on English text; each call degrades independently."""
    insights = TextInsights(redacted_text=text)
    chunk = text[:4500]
    try:
        pii = client.detect_pii_entities(Text=chunk, LanguageCode="en").get("Entities", [])
        insights.pii_types = sorted({e["Type"] for e in pii})
        redacted, insights.placeholders = redact(chunk, pii)
        insights.redacted_text = redacted + text[4500:]
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
        insights.entities = sorted({e["Text"] for e in ents if not _is_token_artifact(e["Text"])})
    except ClientError:
        pass
    return insights
