from __future__ import annotations

import json

import pytest

from conftest import FakeSession, router_converse
from copilot.config import Settings
from copilot.graph import CaseGraph
from copilot.knowledge import KnowledgeBase
from copilot.pipeline import analyze_case

TRIAGE = {"summary": "Wrong charge.", "category": "billing", "priority": "medium"}


def _settings(**overrides) -> Settings:
    return Settings(aws_profile=None, **overrides)


def test_graph_has_the_expected_nodes_and_branches():
    graph = CaseGraph(FakeSession(), _settings())
    mermaid = graph.mermaid()
    for node in ("transcribe", "language", "triage", "classify", "retrieve", "review", "archive"):
        assert node in mermaid


def test_escalated_case_pauses_for_review_then_resumes_with_the_edited_reply():
    session = FakeSession(
        {
            ("bedrock-runtime", "converse"): router_converse(
                {"summary": "Furious customer.", "category": "billing", "priority": "urgent"},
                "We are very sorry and will call you.",
            )
        }
    )
    graph = CaseGraph(session, _settings(require_escalation_review=True))
    run = graph.start(text="This is outrageous, call me now")
    assert run.pending and run.case is None
    assert run.review["persona"] == "escalation"
    assert run.review["draft"] == "We are very sorry and will call you."

    done = graph.resume(run.thread_id, approved=True, reply="A senior agent will call you today.")
    assert done.case is not None and done.case.human_reviewed
    assert done.case.reply == "A senior agent will call you today."


def test_rejected_review_blocks_the_case_without_a_reply():
    session = FakeSession(
        {
            ("bedrock-runtime", "converse"): router_converse(
                {"summary": "s", "category": "billing", "priority": "urgent"}, "draft"
            )
        }
    )
    graph = CaseGraph(session, _settings(require_escalation_review=True))
    run = graph.start(text="urgent problem")
    done = graph.resume(run.thread_id, approved=False)
    assert done.case.blocked and done.case.reply == ""
    assert done.case.backends["review"] == "rejected by reviewer"


def test_review_is_skipped_for_ordinary_cases():
    session = FakeSession(
        {("bedrock-runtime", "converse"): router_converse(TRIAGE, "We will review the charge.")}
    )
    graph = CaseGraph(session, _settings(require_escalation_review=True))
    run = graph.start(text="I was charged twice")
    assert not run.pending and run.case.persona == "billing"


def test_analyze_case_refuses_to_hide_a_pending_review():
    session = FakeSession(
        {
            ("bedrock-runtime", "converse"): router_converse(
                {"summary": "s", "category": "billing", "priority": "urgent"}, "draft"
            )
        }
    )
    with pytest.raises(RuntimeError, match="human review"):
        analyze_case(session, _settings(require_escalation_review=True), text="urgent")


def test_case_is_archived_to_s3_with_redacted_fields_only():
    stored: dict[str, dict] = {}

    def put_object(**kw):
        stored[kw["Key"]] = json.loads(kw["Body"])

    def pii(**_):
        return {"Entities": [{"Type": "NAME", "BeginOffset": 8, "EndOffset": 13}]}

    session = FakeSession(
        {
            ("bedrock-runtime", "converse"): router_converse(TRIAGE, "Dear [[NAME_1]], sorry."),
            ("comprehend", "detect_pii_entities"): pii,
            ("s3", "put_object"): put_object,
        }
    )
    case = analyze_case(
        session,
        _settings(s3_bucket="bucket"),
        text="My name Maria was charged twice",
    )
    assert case.archive_key.startswith("cases/") and case.backends["archive"] == "s3"
    record = stored[case.archive_key]
    flattened = json.dumps(record)
    assert "Maria" not in flattened  # neither the raw message nor the restored reply is archived
    assert record["reply_redacted"] == "Dear [[NAME_1]], sorry."
    assert "original_text" not in record and "reply" not in record


def test_archive_failure_does_not_fail_the_case():
    from conftest import client_error

    session = FakeSession(
        {
            ("bedrock-runtime", "converse"): router_converse(TRIAGE, "ok"),
            ("s3", "put_object"): client_error("AccessDenied"),
        }
    )
    case = analyze_case(session, _settings(s3_bucket="bucket"), text="charged twice")
    assert case.reply and case.backends["archive"].startswith("failed")


def _policy_kb(tmp_path) -> KnowledgeBase:
    from llama_index.core.embeddings import MockEmbedding

    (tmp_path / "shipping.md").write_text("# Shipping\n\nStandard delivery takes 3 to 5 days.\n")
    (tmp_path / "refunds.md").write_text("# Refunds\n\nRefunds are paid within 5 business days.\n")
    return KnowledgeBase.build(tmp_path, MockEmbedding(embed_dim=8), top_k=2)


def test_policy_excerpts_are_given_to_the_model_and_recorded(tmp_path):
    calls: list[dict] = []
    session = FakeSession(
        {
            ("bedrock-runtime", "converse"): router_converse(
                TRIAGE, "Refunds take 5 business days.", calls
            )
        }
    )
    case = analyze_case(
        session, _settings(), text="when do I get my refund", knowledge=_policy_kb(tmp_path)
    )
    reply_prompt = calls[-1]["messages"][0]["content"][0]["text"]
    assert (
        "Policy excerpts" in reply_prompt
        and "Refunds are paid within 5 business days" in reply_prompt
    )
    assert "policy excerpts" in calls[-1]["system"][0]["text"]
    assert set(case.policy_sources) <= {"shipping.md", "refunds.md"} and case.policy_sources
    assert case.backends["knowledge"] == "llamaindex"


def test_without_a_knowledge_base_the_model_may_not_state_policy():
    calls: list[dict] = []
    session = FakeSession(
        {
            ("bedrock-runtime", "converse"): router_converse(
                TRIAGE, "A team member will confirm.", calls
            )
        }
    )
    case = analyze_case(session, _settings(), text="how long is delivery")
    system = calls[-1]["system"][0]["text"]
    assert "policy excerpts" not in system and "the case facts" in system
    assert case.policy_sources == [] and "knowledge" not in case.backends


def test_knowledge_base_failure_degrades_instead_of_failing_the_case():
    class Broken:
        def search_many(self, _):
            raise ConnectionError("embeddings unavailable")

    session = FakeSession({("bedrock-runtime", "converse"): router_converse(TRIAGE, "ok")})
    case = analyze_case(session, _settings(), text="charged twice", knowledge=Broken())
    assert case.reply and case.backends["knowledge"] == "unavailable (ConnectionError)"
