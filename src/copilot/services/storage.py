"""S3 helpers: sync a directory (the knowledge-base index) and archive redacted cases."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from botocore.exceptions import ClientError


def list_keys(s3: Any, bucket: str, prefix: str) -> list[str]:
    """Every object key under the prefix (empty when the prefix is missing or unreadable)."""
    keys: list[str] = []
    token: str | None = None
    while True:
        kwargs: dict[str, Any] = {"Bucket": bucket, "Prefix": prefix}
        if token:
            kwargs["ContinuationToken"] = token
        try:
            page = s3.list_objects_v2(**kwargs)
        except ClientError:
            return keys
        keys += [obj["Key"] for obj in page.get("Contents", [])]
        token = page.get("NextContinuationToken")
        if not token:
            return keys


def upload_dir(
    s3: Any, bucket: str, prefix: str, directory: str | Path, prune: bool = False
) -> int:
    """Copy a directory to the prefix; with prune, delete remote objects no longer present."""
    root = Path(directory)
    sent: set[str] = set()
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        key = prefix + path.relative_to(root).as_posix()
        s3.put_object(Bucket=bucket, Key=key, Body=path.read_bytes())
        sent.add(key)
    if prune:
        stale = [{"Key": k} for k in list_keys(s3, bucket, prefix) if k not in sent]
        if stale:
            s3.delete_objects(Bucket=bucket, Delete={"Objects": stale})
    return len(sent)


def download_dir(
    s3: Any, bucket: str, prefix: str, directory: str | Path, prune: bool = False
) -> int:
    """Copy every object under the prefix into the directory; 0 when nothing is stored yet.

    With prune, local files that are not in the bucket are deleted, but only when the bucket
    actually has objects, so an empty or unreachable prefix can never wipe the local copy.
    """
    root = Path(directory)
    keys = list_keys(s3, bucket, prefix)
    for key in keys:
        target = root / key[len(prefix) :]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(s3.get_object(Bucket=bucket, Key=key)["Body"].read())
    if prune and keys and root.exists():
        remote = {(root / k[len(prefix) :]).resolve() for k in keys}
        for path in root.rglob("*"):
            if path.is_file() and path.resolve() not in remote:
                path.unlink()
    return len(keys)


# Fields that are safe to archive: they never contain the customer's raw message or restored PII.
ARCHIVE_FIELDS = (
    "case_id",
    "source_language",
    "english_text",
    "sentiment",
    "entities",
    "pii_types",
    "document_lines",
    "image_labels",
    "summary",
    "category",
    "category_source",
    "category_confidence",
    "persona",
    "persona_reason",
    "priority",
    "reply_redacted",
    "policy_sources",
    "models",
    "backends",
    "blocked",
)


def archive_case(s3: Any, bucket: str, case: Any) -> str:
    """Write the redacted case record to S3 and return its key."""
    record = {name: getattr(case, name) for name in ARCHIVE_FIELDS}
    now = datetime.now(UTC)
    key = f"cases/{now:%Y/%m/%d}/{case.case_id}.json"
    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=json.dumps({**record, "archived_at": now.isoformat()}, indent=2).encode(),
        ContentType="application/json",
    )
    return key
