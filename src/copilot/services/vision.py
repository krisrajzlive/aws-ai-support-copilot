from __future__ import annotations

from typing import Any


def extract_document_lines(textract: Any, document: bytes) -> list[str]:
    """Line-level text from a scanned page or image via Textract."""
    blocks = textract.detect_document_text(Document={"Bytes": document}).get("Blocks", [])
    return [b["Text"] for b in blocks if b["BlockType"] == "LINE"]


def detect_image_labels(rekognition: Any, image: bytes, min_confidence: float = 80.0) -> list[str]:
    labels = rekognition.detect_labels(
        Image={"Bytes": image}, MaxLabels=10, MinConfidence=min_confidence
    ).get("Labels", [])
    return [label["Name"] for label in labels]
