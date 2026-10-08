from __future__ import annotations

import json

from conftest import FakeSession, client_error
from copilot.config import Settings
from copilot.pipeline import analyze_case
from copilot.services.bedrock import parse_json_object
from copilot.services.comprehend import analyze_english, redact


def _converse(reply: str):
    def call(**_):
        return {"output": {"message": {"content": [{"text": reply}]}}}

    return call


MODEL_JSON = json.dumps(
    {"summary": "Broken item.", "category": "product_defect", "priority": "high", "reply": "Sorry."}
)


def test_redact_replaces_spans_right_to_left():
    text = "Hi Ann, mail ann@x.com"
    pii = [
        {"Type": "NAME", "BeginOffset": 3, "EndOffset": 6},
        {"Type": "EMAIL", "BeginOffset": 13, "EndOffset": 22},
    ]
    assert redact(text, pii) == "Hi [NAME], mail [EMAIL]"


def test_entities_and_sentiment_run_on_redacted_text():
    seen: dict[str, str] = {}

    def pii(**kw):
        return {"Entities": [{"Type": "NAME", "BeginOffset": 3, "EndOffset": 6}]}

    def entities(**kw):
        seen["entities"] = kw["Text"]
        return {"Entities": []}

    session = FakeSession(
        {
            ("comprehend", "detect_pii_entities"): pii,
            ("comprehend", "detect_entities"): entities,
        }
    )
    insights = analyze_english(session.client("comprehend"), "Hi Ann, order 1")
    assert "Ann" not in seen["entities"]
    assert insights.pii_types == ["NAME"]


def test_parse_json_object_tolerates_fences():
    assert parse_json_object('Here:\n```json\n{"a": 1}\n```') == {"a": 1}


def test_pipeline_falls_back_to_bedrock_translation_when_translate_denied():
    def comprehend_lang(**_):
        return {"Languages": [{"LanguageCode": "es"}]}

    session = FakeSession(
        {
            ("comprehend", "detect_dominant_language"): comprehend_lang,
            ("translate", "translate_text"): client_error("AccessDeniedException"),
            ("bedrock-runtime", "converse"): _converse(MODEL_JSON),
        }
    )
    case = analyze_case(
        session, Settings(aws_profile=None, bedrock_models="m.one"), text="Hola, pedido roto"
    )
    assert case.source_language == "es"
    assert case.backends["translate"] == "bedrock"
    assert case.category == "product_defect"
    assert case.model_id == "m.one"


def test_audio_is_transcribed_with_voxtral_before_analysis():
    def converse(**kw):
        if kw["modelId"].startswith("mistral.voxtral"):
            return {"output": {"message": {"content": [{"text": "My order is broken"}]}}}
        return {"output": {"message": {"content": [{"text": MODEL_JSON}]}}}

    session = FakeSession({("bedrock-runtime", "converse"): converse})
    case = analyze_case(
        session, Settings(aws_profile=None, bedrock_models="m.one"), audio=b"RIFFfake"
    )
    assert "order is broken" in case.original_text
    assert case.backends["transcribe"] == "bedrock-voxtral"
