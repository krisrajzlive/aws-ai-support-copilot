from __future__ import annotations

from typing import Any

from botocore.exceptions import ClientError

VOXTRAL_MODELS = ["mistral.voxtral-small-24b-2507", "mistral.voxtral-mini-3b-2507"]
PROMPT = "Transcribe this audio verbatim. Output only the transcript."


class Transcriber:
    """Speech to text through Voxtral on Bedrock, used where Amazon Transcribe is unavailable."""

    backend = "bedrock-voxtral"

    def __init__(self, bedrock_runtime: Any, model_ids: list[str] | None = None):
        self._client = bedrock_runtime
        self._model_ids = model_ids or VOXTRAL_MODELS

    def transcribe(self, audio: bytes, audio_format: str = "wav") -> str:
        last_error: ClientError | None = None
        for model_id in self._model_ids:
            try:
                resp = self._client.converse(
                    modelId=model_id,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"audio": {"format": audio_format, "source": {"bytes": audio}}},
                                {"text": PROMPT},
                            ],
                        }
                    ],
                    inferenceConfig={"maxTokens": 600},
                )
            except ClientError as exc:
                last_error = exc
                continue
            blocks = resp["output"]["message"]["content"]
            return "".join(b["text"] for b in blocks if "text" in b).strip().strip('"')
        raise RuntimeError(f"no speech-to-text model responded: {last_error}")
