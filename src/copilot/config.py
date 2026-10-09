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

    # Cheap, fast models for the triage step (category and priority), tried in order
    triage_models: str = "amazon.nova-micro-v1:0,amazon.nova-lite-v1:0"

    # Trained ticket classifier (ml/train.py output). When set and confident, it decides the
    # category and the LLM triage only supplies the summary and priority.
    classifier_path: str = ""
    classifier_min_confidence: float = 0.6

    # Path to a personas TOML file; empty uses the built-in personas
    personas_file: str = ""

    # bedrock: always use a Bedrock model; aws: Amazon Translate only; auto: Translate, then Bedrock
    translate_backend: Literal["bedrock", "aws", "auto"] = "bedrock"

    # bedrock: Voxtral on Bedrock; aws: Amazon Transcribe (needs transcribe_bucket)
    transcribe_backend: Literal["bedrock", "aws"] = "bedrock"
    transcribe_bucket: str = ""

    # off: no spoken reply; polly: Amazon Polly
    tts_backend: Literal["off", "polly"] = "off"
    polly_voice: str = "Joanna"

    # Amazon Lex intake bot, created by scripts/create_lex_bot.py (TSTALIASID = DRAFT test alias)
    lex_bot_id: str = ""
    lex_bot_alias_id: str = "TSTALIASID"
    lex_locale: str = "en_US"

    # S3 bucket for the case archive and the knowledge-base index (scripts/create_bucket.py)
    s3_bucket: str = ""
    archive_cases: bool = True  # only takes effect when s3_bucket is set

    # Policy knowledge base (LlamaIndex): documents, persisted index and retrieval settings
    embedding_model: str = "amazon.titan-embed-text-v2:0"
    kb_docs_dir: str = "kb/policies"
    kb_index_dir: str = "kb/index"
    kb_top_k: int = 3
    kb_min_score: float = 0.18  # drops chunks unrelated to the question
    kb_enabled: bool = False  # opt in once the index exists (copilot kb build)
    require_escalation_review: bool = False  # pause escalated cases for a human decision

    # Empty disables Bedrock Guardrails; Comprehend PII redaction is always applied
    guardrail_id: str = ""
    guardrail_version: str = "DRAFT"

    @property
    def triage_model_list(self) -> list[str]:
        return [m.strip() for m in self.triage_models.split(",") if m.strip()]

    @property
    def bedrock_model_list(self) -> list[str]:
        return [m.strip() for m in self.bedrock_models.split(",") if m.strip()]
