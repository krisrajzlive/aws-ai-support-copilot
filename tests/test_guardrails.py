from __future__ import annotations

import json

from conftest import FakeSession
from copilot.config import Settings
from copilot.pipeline import analyze_case
from copilot.services.guardrails import BLOCKED_MESSAGE, GuardrailService

MODEL_JSON = json.dumps(
    {"summary": "s", "category": "other", "priority": "low", "reply": "Hello Ann@x.com"}
)


def _converse(**_):
    return {"output": {"message": {"content": [{"text": MODEL_JSON}]}}}


def test_no_intervention_returns_text_unchanged():
    svc = GuardrailService(
        FakeSession({("bedrock", "apply_guardrail"): lambda **_: {"action": "NONE"}}).client(
            "bedrock"
        ),
        "gr1",
    )
    result = svc.apply("hello", "INPUT")
    assert (result.text, result.intervened, result.blocked) == ("hello", False, False)


def test_anonymized_pii_returns_masked_text_not_blocked():
    resp = {
        "action": "GUARDRAIL_INTERVENED",
        "outputs": [{"text": "mail {EMAIL}"}],
        "assessments": [
            {
                "sensitiveInformationPolicy": {
                    "piiEntities": [{"type": "EMAIL", "action": "ANONYMIZED"}]
                }
            }
        ],
    }
    svc = GuardrailService(
        FakeSession({("bedrock", "apply_guardrail"): lambda **_: resp}).client("bedrock"), "gr1"
    )
    result = svc.apply("mail a@b.com", "INPUT")
    assert result.text == "mail {EMAIL}" and result.intervened and not result.blocked


def test_blocked_topic_short_circuits_pipeline_before_the_model_is_called():
    calls = {"converse": 0}

    def converse(**kw):
        calls["converse"] += 1
        return _converse()

    resp = {
        "action": "GUARDRAIL_INTERVENED",
        "outputs": [{"text": "x"}],
        "assessments": [{"topicPolicy": {"topics": [{"name": "Legal", "action": "BLOCKED"}]}}],
    }
    session = FakeSession(
        {
            ("bedrock", "apply_guardrail"): lambda **_: resp,
            ("bedrock-runtime", "converse"): converse,
        }
    )
    case = analyze_case(
        session,
        Settings(aws_profile=None, bedrock_models="m.one", guardrail_id="gr1"),
        text="Can I sue you?",
    )
    assert case.blocked and case.reply == BLOCKED_MESSAGE
    assert calls["converse"] == 0
