from copilot.config import Settings


def test_defaults(monkeypatch):
    for var in ("COPILOT_AWS_PROFILE", "COPILOT_AWS_REGION", "COPILOT_BEDROCK_MODELS"):
        monkeypatch.delenv(var, raising=False)
    s = Settings(_env_file=None)
    assert s.aws_region == "us-east-1"
    assert s.bedrock_model_list[0] == "amazon.nova-lite-v1:0"


def test_model_list_parsing_ignores_blanks():
    s = Settings(bedrock_models=" a.b , ,c.d ")
    assert s.bedrock_model_list == ["a.b", "c.d"]
