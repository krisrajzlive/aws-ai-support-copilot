from __future__ import annotations

from typing import Any

from copilot.aws import CLIENT_CONFIG
from copilot.config import Settings
from copilot.models import SupportCase
from copilot.personas import PersonaSet, load_personas
from copilot.services import comprehend, vision
from copilot.services.bedrock import BedrockService
from copilot.services.guardrails import GuardrailService
from copilot.services.speech import synthesize_reply
from copilot.services.transcribe import AwsTranscriber, Transcriber
from copilot.services.translate import Translator

CATEGORIES = ["billing", "shipping", "product_defect", "account_access", "other"]
PRIORITIES = ["low", "medium", "high", "urgent"]

TRIAGE_PROMPT = (
    "You are a customer-support triage assistant. Using only the case facts provided, return a "
    "single JSON object with keys: summary (max 2 sentences), category (one of "
    f"{CATEGORIES}) and priority (one of {PRIORITIES}). Priority guide: low = a general question; "
    "medium = a normal problem the customer wants fixed (late parcel, wrong charge, faulty item); "
    "high = repeated failure, major financial loss or a vulnerable customer; urgent = safety, "
    "legal threat or possible account compromise. Output JSON only."
)

BASE_REPLY_RULES = (
    "Write only the reply text addressed to the customer, in English, as plain text with no "
    "markdown and no preamble. Never state delivery times, prices, availability or company "
    "policies unless they appear in the case facts; when the customer needs one, say a team member "
    "will confirm it. Never invent order numbers, amounts or dates. Do not use bracketed template "
    "placeholders such as [Your Name]; sign off as 'Customer Support Team'."
)


def _reply_rules(placeholders: dict[str, str]) -> str:
    """Only advertise tokens that exist, so the model cannot copy an example that does not apply."""
    if not placeholders:
        return f"{BASE_REPLY_RULES} The message contains no customer details to personalise with."
    tokens = ", ".join(placeholders)
    return (
        f"{BASE_REPLY_RULES} You may address the customer using these tokens verbatim: {tokens}. "
        "Use no other bracketed tokens."
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
    audio: bytes | None = None,
    audio_format: str = "wav",
    reply_language: str | None = None,
    persona: str | None = None,
    personas: PersonaSet | None = None,
) -> SupportCase:
    """Run one support request through language, vision, triage and a routed persona."""

    def client(name: str) -> Any:
        return session.client(name, config=CLIENT_CONFIG)

    personas = personas or load_personas(settings.personas_file or None)
    runtime = client("bedrock-runtime")
    # Translation, triage and replies each get their own model priority list.
    translate_llm = BedrockService(runtime, settings.bedrock_model_list)
    triage_llm = BedrockService(runtime, settings.triage_model_list)
    translator = Translator(client("translate"), translate_llm, settings.translate_backend)
    guardrail = (
        GuardrailService(client("bedrock"), settings.guardrail_id, settings.guardrail_version)
        if settings.guardrail_id
        else None
    )
    backends: dict[str, str] = {}
    if audio:
        transcriber = (
            AwsTranscriber(client("transcribe"), client("s3"), settings.transcribe_bucket)
            if settings.transcribe_backend == "aws"
            else Transcriber(runtime)
        )
        text = f"{text} {transcriber.transcribe(audio, audio_format)}".strip()
        backends["transcribe"] = transcriber.backend
    case = SupportCase(original_text=text)

    if document:
        case.document_lines = vision.extract_document_lines(client("textract"), document)
    if image:
        case.image_labels = vision.detect_image_labels(client("rekognition"), image)

    comp = client("comprehend")
    source = comprehend.detect_language(comp, text) if text.strip() else "en"
    case.source_language = source
    case.english_text = translator.translate(text, source, "en")

    if guardrail:
        checked = guardrail.apply(case.english_text, "INPUT")
        case.english_text = checked.text
        if checked.blocked:
            case.blocked, case.reply, case.reply_en = True, checked.text, checked.text
            case.backends = {
                **backends,
                "translate": translator.backend,
                "guardrail": "bedrock-guardrails",
            }
            return case

    insights = comprehend.analyze_english(comp, case.english_text)
    case.sentiment, case.entities, case.pii_types = (
        insights.sentiment,
        insights.entities,
        insights.pii_types,
    )
    case.english_text = insights.redacted_text
    placeholders = insights.placeholders

    # Stage 1: a small, cheap model classifies the request.
    triage = triage_llm.complete_json(TRIAGE_PROMPT, _case_prompt(case))
    case.summary = str(triage.get("summary", ""))
    case.category = str(triage.get("category", "other"))
    case.priority = str(triage.get("priority", "medium"))

    # Stage 2: routing picks the persona, and with it the prompt, models and temperature.
    chosen, reason = personas.select(case.category, case.priority, case.sentiment, persona)
    case.persona, case.persona_reason = chosen.name, reason
    reply_llm = BedrockService(runtime, chosen.models)
    case.reply_en = reply_llm.complete(
        f"{chosen.prompt}\n\n{_reply_rules(placeholders)}",
        f"{_case_prompt(case)}\nTriage summary: {case.summary}\nPriority: {case.priority}",
        max_tokens=chosen.max_tokens,
        temperature=chosen.temperature,
    )
    if guardrail:
        case.reply_en = guardrail.apply(case.reply_en, "OUTPUT").text

    target = reply_language or source
    case.reply = comprehend.fill_placeholders(
        translator.translate(case.reply_en, "en", target), placeholders
    )
    case.reply_en = comprehend.fill_placeholders(case.reply_en, placeholders)
    case.reply_language = target
    case.model_id = reply_llm.model_id
    case.models = {"triage": triage_llm.model_id, "reply": reply_llm.model_id}
    case.backends = {
        **backends,
        "translate": translator.backend,
        "llm": "bedrock",
        "guardrail": "bedrock-guardrails" if guardrail else "comprehend-pii-redaction",
    }
    return case


def speak_reply(session: Any, settings: Settings, case: SupportCase) -> bytes:
    """Spoken MP3 of the drafted reply; only available when COPILOT_TTS_BACKEND=polly."""
    if settings.tts_backend != "polly":
        raise RuntimeError("Spoken replies are off. Set COPILOT_TTS_BACKEND=polly to enable them.")
    polly = session.client("polly", config=CLIENT_CONFIG)
    return synthesize_reply(polly, case.reply, case.reply_language or "en", settings.polly_voice)
