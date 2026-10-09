"""Create (or reuse) the private S3 bucket used for the case archive and the knowledge-base index.

    uv run python scripts/create_bucket.py [--delete]

The bucket blocks all public access, encrypts objects by default and expires archived cases after
30 days. It verifies write and read access with a probe object, then prints COPILOT_S3_BUCKET for
.env. Bucket names are globally unique and include the account id, so keep the value out of git.
"""

from __future__ import annotations

import argparse

from botocore.exceptions import ClientError

from copilot.aws import make_session
from copilot.config import Settings

CASE_RETENTION_DAYS = 30


def bucket_name(account: str, region: str) -> str:
    return f"support-copilot-{account}-{region}"


def ensure_bucket(s3, name: str, region: str) -> bool:
    """Create the bucket if missing; return True when it was created."""
    try:
        s3.head_bucket(Bucket=name)
        return False
    except ClientError as exc:
        if exc.response["Error"]["Code"] not in ("404", "NoSuchBucket", "NotFound"):
            raise
    kwargs = (
        {}
        if region == "us-east-1"
        else {"CreateBucketConfiguration": {"LocationConstraint": region}}
    )
    s3.create_bucket(Bucket=name, **kwargs)
    return True


def harden(s3, name: str) -> None:
    s3.put_public_access_block(
        Bucket=name,
        PublicAccessBlockConfiguration={
            "BlockPublicAcls": True,
            "IgnorePublicAcls": True,
            "BlockPublicPolicy": True,
            "RestrictPublicBuckets": True,
        },
    )
    s3.put_bucket_encryption(
        Bucket=name,
        ServerSideEncryptionConfiguration={
            "Rules": [{"ApplyServerSideEncryptionByDefault": {"SSEAlgorithm": "AES256"}}]
        },
    )
    # Retention is hygiene, not security: some restricted accounts forbid lifecycle rules.
    try:
        s3.put_bucket_lifecycle_configuration(
            Bucket=name,
            LifecycleConfiguration={
                "Rules": [
                    {
                        "ID": "expire-archived-cases",
                        "Status": "Enabled",
                        "Filter": {"Prefix": "cases/"},
                        "Expiration": {"Days": CASE_RETENTION_DAYS},
                    }
                ]
            },
        )
    except ClientError as exc:
        if exc.response["Error"]["Code"] != "AccessDenied":
            raise
        print(
            f"WARNING: lifecycle rule not allowed here; archived cases will not expire after "
            f"{CASE_RETENTION_DAYS} days. Delete them with --delete when finished."
        )


def probe(s3, name: str) -> None:
    s3.put_object(Bucket=name, Key="probe/ok.txt", Body=b"ok")
    body = s3.get_object(Bucket=name, Key="probe/ok.txt")["Body"].read()
    s3.delete_object(Bucket=name, Key="probe/ok.txt")
    if body != b"ok":
        raise SystemExit("S3 probe read back unexpected content")


def empty_and_delete(s3, name: str) -> None:
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=name):
        objects = [{"Key": o["Key"]} for o in page.get("Contents", [])]
        if objects:
            s3.delete_objects(Bucket=name, Delete={"Objects": objects})
    s3.delete_bucket(Bucket=name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--delete", action="store_true", help="Empty and delete the bucket.")
    args = parser.parse_args()

    settings = Settings()
    session = make_session(settings)
    account = session.client("sts").get_caller_identity()["Account"]
    name = bucket_name(account, settings.aws_region)
    s3 = session.client("s3")

    if args.delete:
        empty_and_delete(s3, name)
        print(f"Deleted bucket {name}")
        return
    created = ensure_bucket(s3, name, settings.aws_region)
    harden(s3, name)
    probe(s3, name)
    print(f"{'Created' if created else 'Reused'} private bucket; write and read verified.")
    print(f"COPILOT_S3_BUCKET={name}")


if __name__ == "__main__":
    main()
