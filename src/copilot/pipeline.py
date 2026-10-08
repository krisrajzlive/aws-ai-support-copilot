from __future__ import annotations

from typing import Any

from copilot.aws import CLIENT_CONFIG
from copilot.config import Settings
from copilot.models import SupportCase
from copilot.services import comprehend, vision
from copilot.services.bedrock import BedrockService
from copilot.services.translate import Translator

CATEGORIES = ["billing", "shipping", "product_defect", "account_access", "other"]
PRIORITIES = ["low", "medium", "high", "urgent"]

SYSTEM_PROMPT = (
    "You are a customer-support triage assistant. Using only the case facts provided, return a "
    "single JSON object with keys: summary (max 2 sentences), category (one of "
    f"{CATEGORIES}), priority (one of {PRIORITIES}), reply (a polite, concise draft reply in "
    "English that never repeats redacted values). Output JSON only."
)


def _case_prompt(case: SupportCase) -> str:
    parts = [f"Customer message (PII redacted): {case.english_text}"]
    if case.sentiment:
        parts.append(f"Detected sentiment: {case.sentiment}")
    if case.entities:
        parts.append(f"Entities: {', '.join(case.entities)}")
    if case.document_lines:
        parts.append("Attached document text:\n" + "\n".join(case.document_lines[:40]))
    if case.image_labels:
        parts.append(f"Attached photo shows: {', '.join(case.image_labels)}")
    return "\n".join(parts)


def analyze_case(
    session: Any,
    settings: Settings,
    *,
    text: str = "",
    document: bytes | None = None,
    image: bytes | None = None,
    reply_language: str | None = None,
) -> SupportCase:
    """Run one support request through language, vision and Bedrock services."""

    def client(name: str) -> Any:
        return session.client(name, config=CLIENT_CONFIG)

    bedrock = BedrockService(client("bedrock-runtime"), settings.bedrock_model_list)
    translator = Translator(client("translate"), bedrock)
    case = SupportCase(original_text=text)

    if document:
        case.document_lines = vision.extract_document_lines(client("textract"), document)
    if image:
        case.image_labels = vision.detect_image_labels(client("rekognition"), image)

    comp = client("comprehend")
    source = comprehend.detect_language(comp, text) if text.strip() else "en"
    case.source_language = source
    case.english_text = translator.translate(text, source, "en")

    insights = comprehend.analyze_english(comp, case.english_text)
    case.sentiment, case.entities, case.pii_types = (
        insights.sentiment,
        insights.entities,
        insights.pii_types,
    )
    case.english_text = insights.redacted_text

    result = bedrock.complete_json(SYSTEM_PROMPT, _case_prompt(case))
    case.summary = str(result.get("summary", ""))
    case.category = str(result.get("category", "other"))
    case.priority = str(result.get("priority", "medium"))
    case.reply_en = str(result.get("reply", ""))

    target = reply_language or source
    case.reply = translator.translate(case.reply_en, "en", target)
    case.model_id = bedrock.model_id
    case.backends = {"translate": translator.backend, "llm": "bedrock"}
    return case
