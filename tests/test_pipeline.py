from __future__ import annotations

import json

from conftest import FakeSession, client_error, router_converse
from copilot.config import Settings
from copilot.pipeline import analyze_case
from copilot.services.bedrock import parse_json_object
from copilot.services.comprehend import analyze_english, fill_placeholders, redact


def _converse(reply: str):
    def call(**_):
        return {"output": {"message": {"content": [{"text": reply}]}}}

    return call


MODEL_JSON = json.dumps(
    {"summary": "Broken item.", "category": "product_defect", "priority": "high", "reply": "Sorry."}
)


def test_redact_numbers_tokens_and_reuses_one_per_distinct_value():
    text = "Hi Ann, tell Bob, Ann"
    pii = [
        {"Type": "NAME", "BeginOffset": 3, "EndOffset": 6},
        {"Type": "NAME", "BeginOffset": 13, "EndOffset": 16},
        {"Type": "NAME", "BeginOffset": 18, "EndOffset": 21},
    ]
    redacted, mapping = redact(text, pii)
    assert redacted == "Hi [[NAME_1]], tell [[NAME_2]], [[NAME_1]]"
    assert mapping == {"[[NAME_1]]": "Ann", "[[NAME_2]]": "Bob"}


def test_fill_placeholders_restores_known_tokens_and_drops_unknown():
    out = fill_placeholders("Dear [[NAME_1]], re [[ORDER_9]]", {"[[NAME_1]]": "Maria"})
    assert out == "Dear Maria, re"


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
        session,
        Settings(aws_profile=None, bedrock_models="m.one", triage_models="t.one"),
        text="Hola, pedido roto",
    )
    assert case.source_language == "es"
    assert case.backends["translate"] == "bedrock"
    assert case.category == "product_defect"
    assert case.models["triage"] == "t.one"


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


def test_translate_modes():
    from copilot.services.bedrock import BedrockService
    from copilot.services.translate import Translator

    def bedrock_client():
        return FakeSession({("bedrock-runtime", "converse"): _converse("hola")}).client(
            "bedrock-runtime"
        )

    denied = FakeSession({("translate", "translate_text"): client_error("AccessDeniedException")})
    ok = FakeSession({("translate", "translate_text"): lambda **_: {"TranslatedText": "hola-aws"}})

    def make(session, mode):
        return Translator(
            session.client("translate"), BedrockService(bedrock_client(), ["m"]), mode
        )

    assert make(ok, "aws").translate("hi", "en", "es") == "hola-aws"
    assert make(denied, "auto").translate("hi", "en", "es") == "hola"
    assert make(denied, "bedrock").translate("hi", "en", "es") == "hola"
    try:
        make(denied, "aws").translate("hi", "en", "es")
    except Exception as exc:  # strict aws mode must not silently fall back
        assert "AccessDenied" in str(exc)
    else:
        raise AssertionError("aws mode should raise when denied")


def test_reply_is_personalised_from_placeholders_without_sending_pii_to_the_model():
    calls: list[dict] = []

    def pii(**_):
        return {"Entities": [{"Type": "NAME", "BeginOffset": 8, "EndOffset": 13}]}

    session = FakeSession(
        {
            ("bedrock-runtime", "converse"): router_converse(
                {"summary": "s", "category": "other", "priority": "low"},
                "Dear [[NAME_1]], sorry.",
                calls,
            ),
            ("comprehend", "detect_pii_entities"): pii,
        }
    )
    case = analyze_case(
        session, Settings(aws_profile=None, bedrock_models="m.one"), text="My name Maria broke it"
    )
    assert case.reply_en == "Dear Maria, sorry."
    sent = [m["messages"][0]["content"][0]["text"] for m in calls]
    assert all("Maria" not in text for text in sent)
