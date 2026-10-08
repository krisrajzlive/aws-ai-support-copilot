from __future__ import annotations

from conftest import FakeSession, client_error
from copilot.config import Settings
from copilot.doctor import Status, run_doctor


def _settings(models: str = "good.model,other.model") -> Settings:
    return Settings(aws_profile=None, bedrock_models=models)


def _by_key(report):
    return {(r.service, r.capability): r for r in report.results}


def test_credential_failure_short_circuits():
    session = FakeSession({("sts", "get_caller_identity"): client_error("ExpiredToken")})
    report = run_doctor(session, _settings())
    assert len(report.results) == 1
    assert report.results[0].status is Status.ERROR
    assert "expired" in report.results[0].detail


def test_denied_service_is_reported_denied():
    session = FakeSession({("polly", "synthesize_speech"): client_error("AccessDeniedException")})
    results = _by_key(run_doctor(session, _settings()))
    assert results[("polly", "SynthesizeSpeech")].status is Status.DENIED
    assert results[("translate", "TranslateText")].status is Status.AVAILABLE


def test_validation_error_on_minimal_request_counts_as_authorized():
    session = FakeSession(
        {("textract", "detect_document_text"): client_error("UnsupportedDocumentException")}
    )
    result = _by_key(run_doctor(session, _settings()))[("textract", "DetectDocumentText")]
    assert result.status is Status.AVAILABLE
    assert "UnsupportedDocumentException" in result.detail


def test_bedrock_selects_first_working_model_in_priority_order():
    def converse(modelId, **_):
        if modelId == "good.model":
            return {}
        raise client_error("AccessDeniedException")

    session = FakeSession({("bedrock-runtime", "converse"): converse})
    report = run_doctor(session, _settings("bad.model,good.model,other.model"))
    assert report.selected_model == "good.model"
    results = _by_key(report)
    assert results[("bedrock", "bad.model")].status is Status.DENIED


def test_bedrock_validation_error_is_not_availability():
    session = FakeSession({("bedrock-runtime", "converse"): client_error("ValidationException")})
    report = run_doctor(session, _settings("needs.profile"))
    assert report.selected_model is None
    assert _by_key(report)[("bedrock", "needs.profile")].status is Status.ERROR
