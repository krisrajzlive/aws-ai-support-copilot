from __future__ import annotations

from typing import Any

# Standard-engine voices by language code; unknown languages use the configured default.
VOICES = {
    "en": "Joanna",
    "es": "Lucia",
    "fr": "Celine",
    "de": "Marlene",
    "it": "Carla",
    "pt": "Camila",
    "ja": "Mizuki",
}


def synthesize_reply(polly: Any, text: str, language: str, default_voice: str = "Joanna") -> bytes:
    """Spoken MP3 of the reply via Amazon Polly."""
    resp = polly.synthesize_speech(
        Text=text,
        OutputFormat="mp3",
        VoiceId=VOICES.get(language.split("-")[0], default_voice),
    )
    return resp["AudioStream"].read()
