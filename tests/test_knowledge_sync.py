from __future__ import annotations

from pathlib import Path

from llama_index.core.embeddings import MockEmbedding

from conftest import FakeS3
from copilot.config import Settings
from copilot.knowledge import KnowledgeBase, sync_knowledge_base
from copilot.services import storage


def _write(directory: Path, name: str, body: str) -> None:
    (directory / name).write_text(f"# {name}\n\n{body}\n", encoding="utf-8")


def _kb(directory: Path) -> KnowledgeBase:
    return KnowledgeBase.build(directory, MockEmbedding(embed_dim=8), top_k=10)


def _sources(kb: KnowledgeBase, query: str = "policy") -> set[str]:
    return {s.source for s in kb.search(query)}


def test_sync_detects_new_edited_removed_and_unchanged_documents(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    _write(docs, "keep.md", "Unchanged policy text.")
    _write(docs, "edit.md", "Original wording.")
    _write(docs, "remove.md", "To be deleted.")
    kb = _kb(docs)

    _write(docs, "edit.md", "Completely new wording.")
    _write(docs, "add.md", "A brand new policy.")
    (docs / "remove.md").unlink()

    report = kb.sync(docs)
    assert (report.inserted, report.updated, report.deleted, report.unchanged) == (1, 1, 1, 1)
    assert _sources(kb) == {"keep.md", "edit.md", "add.md"}
    texts = " ".join(s.text for s in kb.search("policy"))
    assert "Completely new wording" in texts and "Original wording" not in texts


def test_a_second_sync_with_no_changes_is_a_no_op(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    _write(docs, "a.md", "Some text.")
    kb = _kb(docs)
    assert not kb.sync(docs).changed
    assert kb.sync(docs).unchanged == 1


def test_sync_survives_a_save_and_load_round_trip(tmp_path):
    docs, index = tmp_path / "docs", tmp_path / "index"
    docs.mkdir()
    _write(docs, "a.md", "First.")
    _kb(docs).save(index)

    loaded = KnowledgeBase.load(index, MockEmbedding(embed_dim=8), top_k=10)
    _write(docs, "b.md", "Second.")
    report = loaded.sync(docs)
    assert (report.inserted, report.unchanged) == (1, 1)
    assert _sources(loaded) == {"a.md", "b.md"}


def test_upload_prune_removes_stale_objects_and_download_prune_removes_stale_files(tmp_path):
    s3 = FakeS3({"p/stale.md": b"x", "other/keep.md": b"y"})
    local = tmp_path / "local"
    local.mkdir()
    (local / "fresh.md").write_text("fresh")
    storage.upload_dir(s3, "b", "p/", local, prune=True)
    assert sorted(s3.objects) == ["other/keep.md", "p/fresh.md"]  # other prefixes untouched

    target = tmp_path / "target"
    target.mkdir()
    (target / "leftover.md").write_text("old")
    storage.download_dir(s3, "b", "p/", target, prune=True)
    assert sorted(f.name for f in target.iterdir()) == ["fresh.md"]


def test_download_prune_never_wipes_local_files_when_the_bucket_prefix_is_empty(tmp_path):
    target = tmp_path / "target"
    target.mkdir()
    (target / "precious.md").write_text("keep me")
    assert storage.download_dir(FakeS3(), "b", "p/", target, prune=True) == 0
    assert (target / "precious.md").exists()


def test_sync_knowledge_base_mirrors_documents_and_index_to_s3(tmp_path):
    docs, index = tmp_path / "docs", tmp_path / "index"
    docs.mkdir()
    _write(docs, "a.md", "Policy A.")
    settings = Settings(
        aws_profile=None, s3_bucket="b", kb_docs_dir=str(docs), kb_index_dir=str(index)
    )
    s3 = FakeS3()
    embed = MockEmbedding(embed_dim=8)

    first = sync_knowledge_base(settings, embed_model=embed, s3=s3)
    assert first.inserted == 1
    assert "kb-docs/a.md" in s3.objects and "kb-index/docstore.json" in s3.objects

    _write(docs, "b.md", "Policy B.")
    second = sync_knowledge_base(settings, embed_model=embed, s3=s3)
    assert (second.inserted, second.unchanged) == (1, 1)
    assert "kb-docs/b.md" in s3.objects


def test_sync_from_s3_pulls_documents_edited_in_the_bucket(tmp_path):
    docs, index = tmp_path / "docs", tmp_path / "index"
    docs.mkdir()
    _write(docs, "a.md", "Old policy.")
    settings = Settings(
        aws_profile=None, s3_bucket="b", kb_docs_dir=str(docs), kb_index_dir=str(index)
    )
    s3 = FakeS3()
    embed = MockEmbedding(embed_dim=8)
    sync_knowledge_base(settings, embed_model=embed, s3=s3)

    s3.objects["kb-docs/a.md"] = b"# a.md\n\nPolicy edited in the bucket.\n"
    report = sync_knowledge_base(settings, embed_model=embed, s3=s3, from_s3=True)
    assert report.updated == 1
    assert "edited in the bucket" in (docs / "a.md").read_text()
