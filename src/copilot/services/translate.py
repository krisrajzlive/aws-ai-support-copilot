from __future__ import annotations

from typing import Any

from botocore.exceptions import ClientError

from copilot.doctor import DENIED_CODES
from copilot.services.bedrock import BedrockService


class Translator:
    """Translation with a configurable backend.

    mode "aws" uses Amazon Translate only, "bedrock" uses a Bedrock model only, and "auto" tries
    Amazon Translate first and falls back to Bedrock when the account denies it.
    """

    def __init__(self, translate_client: Any, bedrock: BedrockService, mode: str = "bedrock"):
        self._client = translate_client
        self._bedrock = bedrock
        self._use_aws = mode in ("aws", "auto")
        self._fallback = mode == "auto"
        self.backend = "none"

    def translate(self, text: str, source: str, target: str) -> str:
        if not text.strip() or source == target:
            return text
        if self._use_aws:
            try:
                translated = self._client.translate_text(
                    Text=text, SourceLanguageCode=source, TargetLanguageCode=target
                )["TranslatedText"]
                self.backend = "aws-translate"
                return translated
            except ClientError as exc:
                denied = exc.response.get("Error", {}).get("Code") in DENIED_CODES
                if not (denied and self._fallback):
                    raise
                self._use_aws = False
        self.backend = "bedrock"
        return self._bedrock.complete(
            "You are a professional translator. Output only the translation, nothing else. "
            "Keep any token in double square brackets, such as [[NAME_1]], exactly unchanged.",
            f"Translate from language code '{source}' to '{target}':\n\n{text}",
        )
