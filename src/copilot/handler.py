"""AWS Lambda entry point for the support-case pipeline.

Event (JSON, all fields optional but at least one input is required):
    {"text": "...", "document_b64": "...", "image_b64": "...",
     "audio_b64": "...", "audio_format": "wav", "reply_language": "es"}

Credentials come from the function's execution role; configuration from COPILOT_* variables.
"""

from __future__ import annotations

import base64
import binascii
import json
from typing import Any

from copilot.aws import make_session
from copilot.config import Settings
from copilot.pipeline import analyze_case

INPUT_FIELDS = ("text", "document_b64", "image_b64", "audio_b64")


def _response(status: int, body: dict[str, Any]) -> dict[str, Any]:
    return {
        "statusCode": status,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(body),
    }


def _decode(event: dict[str, Any], field: str) -> bytes | None:
    value = event.get(field)
    return base64.b64decode(value, validate=True) if value else None


def handle(
    event: dict[str, Any], session: Any = None, settings: Settings | None = None
) -> dict[str, Any]:
    # An API Gateway or Function URL proxy wraps the payload in a JSON string body.
    if isinstance(event.get("body"), str):
        try:
            event = json.loads(event["body"])
        except json.JSONDecodeError:
            return _response(400, {"error": "body is not valid JSON"})

    if not any(event.get(f) for f in INPUT_FIELDS):
        return _response(400, {"error": f"provide at least one of: {', '.join(INPUT_FIELDS)}"})

    try:
        document = _decode(event, "document_b64")
        image = _decode(event, "image_b64")
        audio = _decode(event, "audio_b64")
    except (binascii.Error, ValueError):
        return _response(400, {"error": "invalid base64 payload"})

    settings = settings or Settings(aws_profile=None)
    session = session or make_session(settings)
    try:
        case = analyze_case(
            session,
            settings,
            text=event.get("text", ""),
            document=document,
            image=image,
            audio=audio,
            audio_format=event.get("audio_format", "wav"),
            reply_language=event.get("reply_language"),
            persona=event.get("persona"),
        )
    except Exception as exc:  # surface the failure class without leaking internals
        return _response(502, {"error": f"{type(exc).__name__}: {exc}"})
    return _response(200, case.model_dump())


def handler(event: dict[str, Any], context: Any) -> dict[str, Any]:
    return handle(event)
