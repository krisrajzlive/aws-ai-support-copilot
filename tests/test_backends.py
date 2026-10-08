from __future__ import annotations

import io

import pytest

from conftest import FakeSession
from copilot.config import Settings
from copilot.models import SupportCase
from copilot.pipeline import speak_reply
from copilot.services.speech import synthesize_reply
from copilot.services.transcribe import AwsTranscriber


def _job(status: str):
    job = {
        "TranscriptionJobStatus": status,
        "Transcript": {"TranscriptFileUri": "https://x/t.json"},
    }
    return {"TranscriptionJob": job}


def test_aws_transcriber_polls_then_reads_transcript_and_cleans_up():
    states = iter(["IN_PROGRESS", "COMPLETED"])
    deleted = []
    session = FakeSession(
        {
            ("transcribe", "get_transcription_job"): lambda **_: _job(next(states)),
            ("s3", "delete_object"): lambda **kw: deleted.append(kw["Key"]),
        }
    )
    svc = AwsTranscriber(
        session.client("transcribe"),
        session.client("s3"),
        "bucket",
        sleep=lambda _: None,
        fetch=lambda _: {"results": {"transcripts": [{"transcript": " hello "}]}},
    )
    assert svc.transcribe(b"audio", "wav") == "hello"
    assert len(deleted) == 1


def test_aws_transcriber_requires_bucket():
    with pytest.raises(ValueError):
        AwsTranscriber(None, None, "")


def test_polly_picks_language_voice():
    seen = {}

    def synth(**kw):
        seen.update(kw)
        return {"AudioStream": io.BytesIO(b"mp3")}

    polly = FakeSession({("polly", "synthesize_speech"): synth}).client("polly")
    assert synthesize_reply(polly, "hola", "es") == b"mp3"
    assert seen["VoiceId"] == "Lucia"


def test_speech_is_off_by_default():
    with pytest.raises(RuntimeError, match="off"):
        speak_reply(FakeSession(), Settings(aws_profile=None), SupportCase(reply="hi"))
