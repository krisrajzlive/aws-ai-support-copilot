"""Policy knowledge base: LlamaIndex vector index over the markdown documents in kb/policies.

Replies may only cite policy text that this module retrieves, which is what stops the model from
inventing delivery times, refund windows or website steps. The index is persisted locally and
mirrored to S3; `sync` updates it incrementally when documents are added, edited or removed.
"""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from llama_index.core import (
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.node_parser import MarkdownNodeParser
from llama_index.core.schema import Document

from copilot.config import Settings

DOCS_PREFIX = "kb-docs/"
INDEX_PREFIX = "kb-index/"


@dataclass(frozen=True)
class Snippet:
    text: str
    source: str
    score: float


@dataclass(frozen=True)
class SyncReport:
    inserted: int = 0
    updated: int = 0
    deleted: int = 0
    unchanged: int = 0

    @property
    def changed(self) -> bool:
        return bool(self.inserted or self.updated or self.deleted)


def bedrock_embeddings(settings: Settings) -> BaseEmbedding:
    from llama_index.embeddings.bedrock import BedrockEmbedding

    return BedrockEmbedding(
        model_name=settings.embedding_model,
        region_name=settings.aws_region,
        profile_name=settings.aws_profile or None,
    )


def read_documents(docs_dir: str | Path) -> list[Document]:
    """Load the markdown files with stable ids (their path relative to docs_dir).

    LlamaIndex change detection compares a document's hash under its id. Random ids would make
    every sync look like all-new documents, and absolute paths differ between machines.
    """
    root = Path(docs_dir).resolve()
    documents = SimpleDirectoryReader(str(root), required_exts=[".md"], recursive=True).load_data()
    for doc in documents:
        doc.doc_id = Path(doc.metadata["file_path"]).resolve().relative_to(root).as_posix()
    return documents


class KnowledgeBase:
    def __init__(self, index: VectorStoreIndex, top_k: int = 3, min_score: float = 0.0):
        self._index = index
        self._top_k = top_k
        self._min_score = min_score
        self._retriever = index.as_retriever(similarity_top_k=top_k)

    @classmethod
    def build(
        cls, docs_dir: str | Path, embed_model: BaseEmbedding, **kwargs: Any
    ) -> KnowledgeBase:
        index = VectorStoreIndex.from_documents(
            read_documents(docs_dir),
            embed_model=embed_model,
            transformations=[MarkdownNodeParser()],
            show_progress=False,
        )
        return cls(index, **kwargs)

    @classmethod
    def load(
        cls, persist_dir: str | Path, embed_model: BaseEmbedding, **kwargs: Any
    ) -> KnowledgeBase:
        storage = StorageContext.from_defaults(persist_dir=str(persist_dir))
        # The chunker must be restored too: a loaded index otherwise falls back to the default
        # sentence splitter and would chunk newly synced documents differently.
        index = load_index_from_storage(
            storage, embed_model=embed_model, transformations=[MarkdownNodeParser()]
        )
        return cls(index, **kwargs)

    def save(self, persist_dir: str | Path) -> None:
        self._index.storage_context.persist(persist_dir=str(persist_dir))

    def sync(self, docs_dir: str | Path) -> SyncReport:
        """Bring the index in line with the documents: insert new, re-embed edited, drop removed."""
        documents = read_documents(docs_dir)
        known = set(self._index.ref_doc_info)
        current = {d.doc_id for d in documents}

        refreshed = self._index.refresh_ref_docs(documents)
        inserted = sum(
            1 for d, r in zip(documents, refreshed, strict=True) if r and d.doc_id not in known
        )
        updated = sum(
            1 for d, r in zip(documents, refreshed, strict=True) if r and d.doc_id in known
        )

        removed = known - current
        for doc_id in removed:
            self._index.delete_ref_doc(doc_id, delete_from_docstore=True)
        self._retriever = self._index.as_retriever(similarity_top_k=self._top_k)
        return SyncReport(
            inserted=inserted,
            updated=updated,
            deleted=len(removed),
            unchanged=len(documents) - inserted - updated,
        )

    def search(self, query: str) -> list[Snippet]:
        results = self._retriever.retrieve(query)
        return [
            Snippet(
                text=r.node.get_content().strip(),
                source=r.node.metadata.get("file_name", "unknown"),
                score=round(float(r.score or 0.0), 3),
            )
            for r in results
            if (r.score or 0.0) >= self._min_score
        ]

    def search_many(self, queries: list[str], max_workers: int = 4) -> list[Snippet]:
        """Search several queries in parallel and merge: best score per passage, top_k overall."""
        unique = list(dict.fromkeys(q for q in queries if q.strip()))
        if len(unique) <= 1:
            return self.search(unique[0]) if unique else []
        with ThreadPoolExecutor(max_workers=min(max_workers, len(unique))) as pool:
            batches = list(pool.map(self.search, unique))
        best: dict[tuple[str, str], Snippet] = {}
        for batch in batches:
            for snippet in batch:
                key = (snippet.source, snippet.text)
                if key not in best or snippet.score > best[key].score:
                    best[key] = snippet
        return sorted(best.values(), key=lambda x: x.score, reverse=True)[: self._top_k]


SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+")


def split_queries(text: str, max_sentences: int = 3) -> list[str]:
    """The whole message, plus each substantial sentence when there are several.

    A message about two problems embeds as a blend of both; searching its sentences separately
    retrieves the policy for each one.
    """
    sentences = [s.strip() for s in SENTENCE_BREAK.split(text.strip()) if len(s.split()) >= 4]
    if len(sentences) < 2:
        return [text]
    return [text, *sentences[:max_sentences]]


def _options(settings: Settings) -> dict[str, Any]:
    return {"top_k": settings.kb_top_k, "min_score": settings.kb_min_score}


def load_or_build(
    settings: Settings, embed_model: BaseEmbedding | None = None, s3: Any = None
) -> KnowledgeBase:
    """Load the persisted index (local, then S3), or build it from the documents."""
    from copilot.services import storage

    embed = embed_model or bedrock_embeddings(settings)
    persist = Path(settings.kb_index_dir)
    if not (persist / "docstore.json").exists() and s3 and settings.s3_bucket:
        storage.download_dir(s3, settings.s3_bucket, INDEX_PREFIX, persist)
    if (persist / "docstore.json").exists():
        return KnowledgeBase.load(persist, embed, **_options(settings))
    kb = KnowledgeBase.build(settings.kb_docs_dir, embed, **_options(settings))
    kb.save(persist)
    if s3 and settings.s3_bucket:
        storage.upload_dir(s3, settings.s3_bucket, INDEX_PREFIX, persist, prune=True)
    return kb


def sync_knowledge_base(
    settings: Settings,
    embed_model: BaseEmbedding | None = None,
    s3: Any = None,
    from_s3: bool = False,
) -> SyncReport:
    """Incrementally update the index and mirror documents and index to S3.

    By default the local documents are the source of truth and are uploaded. With `from_s3` the
    bucket is the source of truth (policies edited there), and local copies are refreshed from it.
    """
    from copilot.services import storage

    embed = embed_model or bedrock_embeddings(settings)
    use_s3 = bool(s3 and settings.s3_bucket)
    docs_dir, persist = Path(settings.kb_docs_dir), Path(settings.kb_index_dir)

    if use_s3 and from_s3:
        storage.download_dir(s3, settings.s3_bucket, DOCS_PREFIX, docs_dir, prune=True)
    elif use_s3:
        storage.upload_dir(s3, settings.s3_bucket, DOCS_PREFIX, docs_dir, prune=True)

    if not (persist / "docstore.json").exists() and use_s3:
        storage.download_dir(s3, settings.s3_bucket, INDEX_PREFIX, persist)

    if (persist / "docstore.json").exists():
        kb = KnowledgeBase.load(persist, embed, **_options(settings))
        report = kb.sync(docs_dir)
    else:
        kb = KnowledgeBase.build(docs_dir, embed, **_options(settings))
        report = SyncReport(inserted=len(read_documents(docs_dir)))

    kb.save(persist)
    if use_s3:
        storage.upload_dir(s3, settings.s3_bucket, INDEX_PREFIX, persist, prune=True)
    return report
