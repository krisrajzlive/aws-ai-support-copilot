from __future__ import annotations

from llama_index.core.embeddings import MockEmbedding

from conftest import FakeSession, converse_response
from copilot.config import Settings
from copilot.graph import CaseGraph
from copilot.knowledge import KnowledgeBase, Snippet, split_queries


def test_split_queries_keeps_the_whole_message_and_adds_each_substantial_sentence():
    single = "Where is my parcel?"
    assert split_queries(single) == [single]

    multi = "I was charged twice for my order. Also my parcel has not arrived yet. Thanks."
    queries = split_queries(multi)
    assert queries[0] == multi
    assert "I was charged twice for my order." in queries
    assert "Also my parcel has not arrived yet." in queries
    assert "Thanks." not in queries  # too short to be a useful query


def test_search_many_merges_deduplicates_and_keeps_the_best_score(tmp_path):
    (tmp_path / "a.md").write_text("# A\n\nFirst policy.\n")
    kb = KnowledgeBase.build(tmp_path, MockEmbedding(embed_dim=8), top_k=2)

    scores = {
        "q1": [Snippet("same passage", "a.md", 0.4), Snippet("only in q1", "b.md", 0.3)],
        "q2": [Snippet("same passage", "a.md", 0.7), Snippet("only in q2", "c.md", 0.5)],
    }
    kb.search = lambda q: scores[q]  # type: ignore[method-assign]
    merged = kb.search_many(["q1", "q2", "q1", "  "])
    assert [(s.text, s.score) for s in merged] == [("same passage", 0.7), ("only in q2", 0.5)]
    assert kb.search_many([]) == []


def test_each_query_of_a_multi_issue_message_reaches_the_knowledge_base():
    searched: list[str] = []

    class Knowledge:
        def search_many(self, queries):
            searched.extend(queries)
            return [Snippet("Duplicate charges reverse in 5 days.", "billing.md", 0.8)]

    session = FakeSession(
        {
            ("bedrock-runtime", "converse"): lambda **kw: converse_response(
                '{"summary": "s", "category": "billing", "priority": "low"}'
                if "triage assistant" in kw["system"][0]["text"]
                else "A team member will help."
            )
        }
    )
    graph = CaseGraph(session, Settings(aws_profile=None), knowledge=Knowledge())
    run = graph.start(text="I was charged twice for my order. Also my parcel has not arrived yet.")
    assert len(searched) == 3  # the whole message plus both sentences
    assert run.case.policy_sources == ["billing.md"]


def test_clients_are_created_once_and_reused():
    created: list[str] = []

    class CountingSession(FakeSession):
        def client(self, name, **kw):
            created.append(name)
            return super().client(name, **kw)

    graph = CaseGraph(CountingSession(), Settings(aws_profile=None))
    before = len(created)
    for _ in range(3):
        graph._client("s3")
    assert created[before:].count("s3") <= 1
