from __future__ import annotations

import pytest

from conftest import FakeSession, router_converse
from copilot.config import Settings
from copilot.personas import PersonaSet, _parse, load_personas
from copilot.pipeline import analyze_case


def test_default_personas_cover_every_category():
    ps = load_personas()
    for category in ("billing", "shipping", "product_defect", "account_access", "other"):
        persona, reason = ps.select(category, "medium", "NEUTRAL")
        assert category in persona.categories and reason == f"category: {category}"


def test_urgent_priority_escalates_regardless_of_category():
    persona, reason = load_personas().select("billing", "urgent", "NEUTRAL")
    assert persona.name == "escalation" and "urgent" in reason


def test_negative_sentiment_escalates_only_with_high_priority():
    ps = load_personas()
    assert ps.select("shipping", "high", "NEGATIVE")[0].name == "escalation"
    assert ps.select("shipping", "medium", "NEGATIVE")[0].name == "shipping"
    assert ps.select("shipping", "high", "POSITIVE")[0].name == "shipping"


def test_unknown_category_falls_back_to_default_persona():
    persona, reason = load_personas().select("weird", "low", None)
    assert persona.name == "general" and reason == "default"


def test_override_wins_and_unknown_override_is_rejected():
    ps = load_personas()
    assert ps.select("billing", "urgent", "NEGATIVE", override="shipping")[1] == "override"
    with pytest.raises(ValueError, match="unknown persona"):
        ps.select("billing", "low", None, override="nope")


def test_each_default_persona_has_distinct_prompt_and_models():
    ps: PersonaSet = load_personas()
    prompts = {p.prompt for p in ps.personas.values()}
    assert len(prompts) == len(ps.personas)
    assert all(p.models for p in ps.personas.values())


def test_invalid_persona_file_is_rejected():
    with pytest.raises(ValueError, match="needs a prompt"):
        _parse({"personas": {"general": {"models": []}}})
    with pytest.raises(ValueError, match="missing persona"):
        _parse({"personas": {"x": {"models": ["m"], "prompt": "p"}}})


def test_pipeline_routes_to_persona_and_uses_its_models_and_temperature():
    calls: list[dict] = []
    session = FakeSession(
        {
            ("bedrock-runtime", "converse"): router_converse(
                {"summary": "Wrong charge.", "category": "billing", "priority": "medium"},
                "We will review the charge.",
                calls,
            )
        }
    )
    case = analyze_case(session, Settings(aws_profile=None), text="I was charged twice")
    assert case.persona == "billing"
    assert case.persona_reason == "category: billing"
    reply_call = calls[-1]
    assert reply_call["modelId"] == "amazon.nova-pro-v1:0"
    assert reply_call["inferenceConfig"]["temperature"] == 0.1
    assert "billing specialist" in reply_call["system"][0]["text"]
    assert calls[0]["modelId"] == "amazon.nova-micro-v1:0"  # triage uses the cheap model
    assert case.models == {"triage": "amazon.nova-micro-v1:0", "reply": "amazon.nova-pro-v1:0"}


def test_persona_flag_overrides_routing():
    session = FakeSession(
        {
            ("bedrock-runtime", "converse"): router_converse(
                {"summary": "s", "category": "billing", "priority": "low"}, "ok"
            )
        }
    )
    case = analyze_case(session, Settings(aws_profile=None), text="hi", persona="shipping")
    assert case.persona == "shipping" and case.persona_reason == "override"
