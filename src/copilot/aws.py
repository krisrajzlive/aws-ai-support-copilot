from __future__ import annotations

import boto3
from botocore.config import Config

from copilot.config import Settings

CLIENT_CONFIG = Config(
    retries={"max_attempts": 5, "mode": "adaptive"},
    connect_timeout=5,
    read_timeout=30,
)


def make_session(settings: Settings) -> boto3.Session:
    """Build a boto3 session from a named profile, or the default credential chain if unset."""
    return boto3.Session(
        profile_name=settings.aws_profile or None,
        region_name=settings.aws_region,
    )
