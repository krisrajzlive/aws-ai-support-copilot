from __future__ import annotations

import base64
import json

from conftest import FakeSession
from copilot.config import Settings
from copilot.handler import handle

MODEL_JSON = json.dumps(
    {"summary": "s", "category": "billing", "priority": "low", "reply": "Thanks."}
)


def _session():
    def converse(**_):
        return {"output": {"message": {"content": [{"text": MODEL_JSON}]}}}

    return FakeSession({("bedrock-runtime", "converse"): converse})


def _settings():
    return Settings(aws_profile=None, bedrock_models="m.one")


def test_text_request_returns_case():
    out = handle({"text": "Where is my refund?"}, _session(), _settings())
    assert out["statusCode"] == 200
    assert json.loads(out["body"])["category"] == "billing"


def test_proxy_style_body_is_unwrapped():
    event = {"body": json.dumps({"text": "hello"})}
    assert handle(event, _session(), _settings())["statusCode"] == 200


def test_missing_input_is_400():
    assert handle({}, _session(), _settings())["statusCode"] == 400


def test_invalid_base64_is_400():
    out = handle({"image_b64": "!!not-base64!!"}, _session(), _settings())
    assert out["statusCode"] == 400


def test_valid_base64_document_is_accepted():
    payload = base64.b64encode(b"png-bytes").decode()
    out = handle({"text": "x", "document_b64": payload}, _session(), _settings())
    assert out["statusCode"] == 200


def test_pipeline_failure_becomes_502_not_a_crash():
    def converse(**_):
        raise RuntimeError("boom")

    session = FakeSession({("bedrock-runtime", "converse"): converse})
    out = handle({"text": "hi"}, session, _settings())
    assert out["statusCode"] == 502
