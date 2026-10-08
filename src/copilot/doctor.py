"""Preflight check: which AWS AI services can the current credentials actually use?

IAM evaluates permissions before request validation, so for most services a cheap call is
enough: an ``AccessDenied`` means the policy blocks it, while any other outcome (including a
validation error on a deliberately minimal request) means the action is authorized. Bedrock
models are probed strictly: only a successful Converse call counts as available.
"""

from __future__ import annotations

import struct
import zlib
from collections.abc import Callable
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any

from botocore.exceptions import BotoCoreError, ClientError

from copilot.aws import CLIENT_CONFIG
from copilot.config import Settings


class Status(StrEnum):
    AVAILABLE = "available"
    DENIED = "denied"
    ERROR = "error"


@dataclass(frozen=True)
class ProbeResult:
    service: str
    capability: str
    status: Status
    detail: str = ""


@dataclass(frozen=True)
class Probe:
    service: str
    capability: str
    client: str
    call: Callable[[Any], Any]
    strict: bool = False


@dataclass(frozen=True)
class DoctorReport:
    results: list[ProbeResult]

    @property
    def selected_model(self) -> str | None:
        """First Bedrock model that answered, in the configured priority order."""
        for r in self.results:
            if r.service == "bedrock" and r.status is Status.AVAILABLE:
                return r.capability
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "selected_model": self.selected_model,
            "results": [asdict(r) for r in self.results],
        }


DENIED_CODES = {"AccessDenied", "AccessDeniedException", "UnauthorizedOperation"}
CREDENTIAL_CODES = {
    "ExpiredToken",
    "ExpiredTokenException",
    "InvalidClientTokenId",
    "UnrecognizedClientException",
    "InvalidSignatureException",
    "SignatureDoesNotMatch",
}


def _blank_png(size: int = 128) -> bytes:
    """A valid all-white PNG, used as a minimal image payload for Textract and Rekognition."""

    def chunk(tag: bytes, data: bytes) -> bytes:
        body = tag + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))

    row = b"\x00" + b"\xff\xff\xff" * size
    header = struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(row * size))
        + chunk(b"IEND", b"")
    )


_PNG = _blank_png()
_HELLO = [{"role": "user", "content": [{"text": "Reply with the single word: ok"}]}]


def _bedrock_probe(model_id: str) -> Probe:
    return Probe(
        "bedrock",
        model_id,
        "bedrock-runtime",
        lambda c: c.converse(modelId=model_id, messages=_HELLO, inferenceConfig={"maxTokens": 20}),
        strict=True,
    )


STATIC_PROBES: tuple[Probe, ...] = (
    Probe(
        "textract",
        "DetectDocumentText",
        "textract",
        lambda c: c.detect_document_text(Document={"Bytes": _PNG}),
    ),
    Probe(
        "comprehend",
        "DetectSentiment",
        "comprehend",
        lambda c: c.detect_sentiment(Text="The support team was helpful.", LanguageCode="en"),
    ),
    Probe(
        "comprehend",
        "DetectEntities",
        "comprehend",
        lambda c: c.detect_entities(Text="Order 1234 was shipped to Seattle.", LanguageCode="en"),
    ),
    Probe(
        "comprehend",
        "DetectPiiEntities",
        "comprehend",
        lambda c: c.detect_pii_entities(Text="Call me at 555-0100.", LanguageCode="en"),
    ),
    Probe(
        "rekognition",
        "DetectLabels",
        "rekognition",
        lambda c: c.detect_labels(Image={"Bytes": _PNG}, MaxLabels=1),
    ),
    Probe(
        "rekognition",
        "DetectModerationLabels",
        "rekognition",
        lambda c: c.detect_moderation_labels(Image={"Bytes": _PNG}),
    ),
    Probe("guardrails", "ListGuardrails", "bedrock", lambda c: c.list_guardrails(maxResults=1)),
    Probe("lex", "ListBots", "lexv2-models", lambda c: c.list_bots(maxResults=1)),
    Probe(
        "sagemaker",
        "ListNotebookInstances",
        "sagemaker",
        lambda c: c.list_notebook_instances(MaxResults=1),
    ),
    Probe("sagemaker", "ListEndpoints", "sagemaker", lambda c: c.list_endpoints(MaxResults=1)),
    Probe("s3", "ListBuckets", "s3", lambda c: c.list_buckets()),
    Probe("lambda", "ListFunctions", "lambda", lambda c: c.list_functions(MaxItems=1)),
    Probe(
        "transcribe",
        "ListTranscriptionJobs",
        "transcribe",
        lambda c: c.list_transcription_jobs(MaxResults=1),
    ),
    Probe(
        "translate",
        "TranslateText",
        "translate",
        lambda c: c.translate_text(Text="hello", SourceLanguageCode="en", TargetLanguageCode="es"),
    ),
    Probe(
        "polly",
        "SynthesizeSpeech",
        "polly",
        lambda c: c.synthesize_speech(Text="ok", OutputFormat="mp3", VoiceId="Joanna"),
    ),
)

IDENTITY_PROBE = Probe(
    "sts", "GetCallerIdentity", "sts", lambda c: c.get_caller_identity(), strict=True
)


def classify(probe: Probe, session: Any) -> ProbeResult:
    """Run one probe and translate the outcome into a status."""
    client = session.client(probe.client, config=CLIENT_CONFIG)
    try:
        probe.call(client)
    except ClientError as exc:
        code = exc.response.get("Error", {}).get("Code", "Unknown")
        if code in CREDENTIAL_CODES:
            return ProbeResult(
                probe.service, probe.capability, Status.ERROR, "credentials invalid or expired"
            )
        if code in DENIED_CODES:
            return ProbeResult(
                probe.service, probe.capability, Status.DENIED, "blocked by IAM policy"
            )
        if probe.strict:
            return ProbeResult(probe.service, probe.capability, Status.ERROR, code)
        return ProbeResult(
            probe.service,
            probe.capability,
            Status.AVAILABLE,
            f"authorized (minimal request rejected: {code})",
        )
    except BotoCoreError as exc:
        return ProbeResult(probe.service, probe.capability, Status.ERROR, type(exc).__name__)
    return ProbeResult(probe.service, probe.capability, Status.AVAILABLE)


def run_doctor(session: Any, settings: Settings) -> DoctorReport:
    """Probe credentials first; if they fail, skip everything else."""
    identity = classify(IDENTITY_PROBE, session)
    if identity.status is not Status.AVAILABLE:
        return DoctorReport([identity])
    probes = [*(_bedrock_probe(m) for m in settings.bedrock_model_list), *STATIC_PROBES]
    return DoctorReport([identity, *(classify(p, session) for p in probes)])
