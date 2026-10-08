from __future__ import annotations

from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_BEDROCK_MODELS = (
    "amazon.nova-lite-v1:0,"
    "amazon.nova-micro-v1:0,"
    "amazon.nova-pro-v1:0,"
    "qwen.qwen3-32b-v1:0,"
    "openai.gpt-oss-120b-1:0"
)


class Settings(BaseSettings):
    """Runtime configuration, read from COPILOT_* environment variables or a local .env.

    Defaults suit a restricted lab account: services that are commonly blocked there are off or
    routed to Bedrock. Switch them on per service in a fuller AWS account.
    """

    model_config = SettingsConfigDict(env_prefix="COPILOT_", env_file=".env", extra="ignore")

    aws_profile: str | None = "sandbox"
    aws_region: str = "us-east-1"
    bedrock_models: str = DEFAULT_BEDROCK_MODELS

    # bedrock: always use a Bedrock model; aws: Amazon Translate only; auto: Translate, then Bedrock
    translate_backend: Literal["bedrock", "aws", "auto"] = "bedrock"

    # bedrock: Voxtral on Bedrock; aws: Amazon Transcribe (needs transcribe_bucket)
    transcribe_backend: Literal["bedrock", "aws"] = "bedrock"
    transcribe_bucket: str = ""

    # off: no spoken reply; polly: Amazon Polly
    tts_backend: Literal["off", "polly"] = "off"
    polly_voice: str = "Joanna"

    # Empty disables Bedrock Guardrails; Comprehend PII redaction is always applied
    guardrail_id: str = ""
    guardrail_version: str = "DRAFT"

    @property
    def bedrock_model_list(self) -> list[str]:
        return [m.strip() for m in self.bedrock_models.split(",") if m.strip()]
