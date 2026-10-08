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


def test_lex_intake_collects_slots_and_builds_case_text():
    from copilot.services.lex import LexIntake

    replies = iter(
        [
            {
                "messages": [{"content": "What is your order number?"}],
                "sessionState": {
                    "dialogAction": {"type": "ElicitSlot"},
                    "intent": {
                        "name": "ReportProblem",
                        "state": "InProgress",
                        "slots": {"OrderId": None},
                    },
                },
            },
            {
                "messages": [{"content": "Thanks."}],
                "sessionState": {
                    "dialogAction": {"type": "Close"},
                    "intent": {
                        "name": "ReportProblem",
                        "state": "Fulfilled",
                        "slots": {
                            "OrderId": {
                                "value": {"originalValue": "A55120", "interpretedValue": "a55120"}
                            },
                            "Details": None,
                        },
                    },
                },
            },
        ]
    )
    session = FakeSession({("lexv2-runtime", "recognize_text"): lambda **_: next(replies)})
    lex = LexIntake(session.client("lexv2-runtime"), "BOT")
    first = lex.send("my order arrived broken")
    assert first.messages == ["What is your order number?"] and not first.fulfilled
    done = lex.send("A55120")
    assert done.fulfilled and done.slots == {"OrderId": "A55120"}
    assert lex.case_text(done) == "my order arrived broken A55120 (OrderId A55120)"


def _tiny_classifier(tmp_path):
    import json

    path = tmp_path / "model.json"
    path.write_text(
        json.dumps(
            {
                "classes": ["billing", "shipping"],
                "vocabulary": {"refund": 0, "charged": 1, "parcel": 2, "late": 3},
                "idf": [1.0, 1.0, 1.0, 1.0],
                "coef": [[4.0, 4.0, -4.0, -4.0], [-4.0, -4.0, 4.0, 4.0]],
                "intercept": [0.0, 0.0],
                "ngram_max": 1,
            }
        )
    )
    return path


def test_classifier_predicts_with_confidence(tmp_path):
    from copilot.services.classifier import TicketClassifier

    model = TicketClassifier.load(_tiny_classifier(tmp_path))
    billing = model.predict("I was charged and want a refund")
    shipping = model.predict("my parcel is late")
    unknown = model.predict("hello there")
    assert billing.label == "billing" and billing.confidence > 0.9
    assert shipping.label == "shipping" and shipping.confidence > 0.9
    assert unknown.confidence == 0.5  # no known words: undecided between two classes


def test_pipeline_uses_classifier_only_when_confident(tmp_path):
    from conftest import router_converse
    from copilot.pipeline import analyze_case

    triage = {"summary": "s", "category": "other", "priority": "low"}
    session = FakeSession({("bedrock-runtime", "converse"): router_converse(triage, "ok")})
    settings = Settings(
        aws_profile=None,
        classifier_path=str(_tiny_classifier(tmp_path)),
        classifier_min_confidence=0.8,
    )
    sure = analyze_case(session, settings, text="charged twice, refund please")
    assert (sure.category, sure.category_source) == ("billing", "classifier")
    assert sure.category_confidence and sure.category_confidence > 0.8
    unsure = analyze_case(session, settings, text="hello there")
    assert (unsure.category, unsure.category_source) == ("other", "llm")
