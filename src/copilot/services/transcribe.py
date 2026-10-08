from __future__ import annotations

import json
import time
import urllib.request
import uuid
from collections.abc import Callable
from typing import Any

from botocore.exceptions import ClientError

VOXTRAL_MODELS = ["mistral.voxtral-small-24b-2507", "mistral.voxtral-mini-3b-2507"]
PROMPT = "Transcribe this audio verbatim. Output only the transcript."


class Transcriber:
    """Speech to text through Voxtral on Bedrock."""

    backend = "bedrock-voxtral"

    def __init__(self, bedrock_runtime: Any, model_ids: list[str] | None = None):
        self._client = bedrock_runtime
        self._model_ids = model_ids or VOXTRAL_MODELS

    def transcribe(self, audio: bytes, audio_format: str = "wav") -> str:
        last_error: ClientError | None = None
        for model_id in self._model_ids:
            try:
                resp = self._client.converse(
                    modelId=model_id,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"audio": {"format": audio_format, "source": {"bytes": audio}}},
                                {"text": PROMPT},
                            ],
                        }
                    ],
                    inferenceConfig={"maxTokens": 600},
                )
            except ClientError as exc:
                last_error = exc
                continue
            blocks = resp["output"]["message"]["content"]
            return "".join(b["text"] for b in blocks if "text" in b).strip().strip('"')
        raise RuntimeError(f"no speech-to-text model responded: {last_error}")


def _fetch_transcript_json(uri: str) -> dict[str, Any]:
    with urllib.request.urlopen(uri, timeout=30) as resp:  # noqa: S310 - presigned AWS URL
        return json.load(resp)


class AwsTranscriber:
    """Speech to text through Amazon Transcribe: upload to S3, run a job, read the transcript."""

    backend = "aws-transcribe"

    def __init__(
        self,
        transcribe_client: Any,
        s3_client: Any,
        bucket: str,
        *,
        poll_seconds: float = 3.0,
        timeout_seconds: float = 180.0,
        sleep: Callable[[float], None] = time.sleep,
        fetch: Callable[[str], dict[str, Any]] = _fetch_transcript_json,
    ):
        if not bucket:
            raise ValueError("COPILOT_TRANSCRIBE_BUCKET is required for the aws transcribe backend")
        self._transcribe = transcribe_client
        self._s3 = s3_client
        self._bucket = bucket
        self._poll = poll_seconds
        self._timeout = timeout_seconds
        self._sleep = sleep
        self._fetch = fetch

    def transcribe(self, audio: bytes, audio_format: str = "wav") -> str:
        job = f"copilot-{uuid.uuid4().hex[:12]}"
        key = f"transcribe-input/{job}.{audio_format}"
        self._s3.put_object(Bucket=self._bucket, Key=key, Body=audio)
        try:
            self._transcribe.start_transcription_job(
                TranscriptionJobName=job,
                Media={"MediaFileUri": f"s3://{self._bucket}/{key}"},
                MediaFormat=audio_format,
                IdentifyLanguage=True,
            )
            waited = 0.0
            while True:
                data = self._transcribe.get_transcription_job(TranscriptionJobName=job)
                state = data["TranscriptionJob"]["TranscriptionJobStatus"]
                if state == "COMPLETED":
                    uri = data["TranscriptionJob"]["Transcript"]["TranscriptFileUri"]
                    return self._fetch(uri)["results"]["transcripts"][0]["transcript"].strip()
                if state == "FAILED":
                    reason = data["TranscriptionJob"].get("FailureReason", "unknown")
                    raise RuntimeError(f"Transcribe job failed: {reason}")
                if waited >= self._timeout:
                    raise TimeoutError(f"Transcribe job {job} did not finish in time")
                self._sleep(self._poll)
                waited += self._poll
        finally:
            self._s3.delete_object(Bucket=self._bucket, Key=key)
