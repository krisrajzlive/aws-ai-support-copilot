from __future__ import annotations

from typing import Any

from botocore.exceptions import ClientError

from copilot.doctor import DENIED_CODES
from copilot.services.bedrock import BedrockService


class Translator:
    """Amazon Translate when the account allows it, otherwise a Bedrock model."""

    def __init__(self, translate_client: Any, bedrock: BedrockService):
        self._client = translate_client
        self._bedrock = bedrock
        self.backend = "aws-translate"

    def translate(self, text: str, source: str, target: str) -> str:
        if not text.strip() or source == target:
            return text
        if self.backend == "aws-translate":
            try:
                return self._client.translate_text(
                    Text=text, SourceLanguageCode=source, TargetLanguageCode=target
                )["TranslatedText"]
            except ClientError as exc:
                if exc.response.get("Error", {}).get("Code") not in DENIED_CODES:
                    raise
                self.backend = "bedrock"
        return self._bedrock.complete(
            "You are a professional translator. Output only the translation, nothing else.",
            f"Translate from language code '{source}' to '{target}':\n\n{text}",
        )
