from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_BEDROCK_MODELS = (
    "amazon.nova-lite-v1:0,"
    "amazon.nova-micro-v1:0,"
    "amazon.nova-pro-v1:0,"
    "qwen.qwen3-32b-v1:0,"
    "openai.gpt-oss-120b-1:0"
)


class Settings(BaseSettings):
    """Runtime configuration, read from COPILOT_* environment variables or a local .env."""

    model_config = SettingsConfigDict(env_prefix="COPILOT_", env_file=".env", extra="ignore")

    aws_profile: str | None = "sandbox"
    aws_region: str = "us-east-1"
    bedrock_models: str = DEFAULT_BEDROCK_MODELS

    @property
    def bedrock_model_list(self) -> list[str]:
        return [m.strip() for m in self.bedrock_models.split(",") if m.strip()]
