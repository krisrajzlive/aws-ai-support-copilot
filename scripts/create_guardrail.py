"""Create (or reuse) the support-copilot guardrail and print its id and version.

Run in an account that allows bedrock:CreateGuardrail:

    uv run python scripts/create_guardrail.py

Then set COPILOT_GUARDRAIL_ID and COPILOT_GUARDRAIL_VERSION in .env.
"""

from __future__ import annotations

from copilot.aws import make_session
from copilot.config import Settings

NAME = "support-copilot-guardrail"
BLOCKED = "This request could not be processed automatically and has been escalated."

PII_TYPES = ["NAME", "EMAIL", "PHONE", "ADDRESS", "CREDIT_DEBIT_CARD_NUMBER"]
FILTERS = ["HATE", "INSULTS", "SEXUAL", "VIOLENCE", "MISCONDUCT"]


def build_config() -> dict:
    return {
        "name": NAME,
        "description": "PII masking, harmful-content filters and denied advice topics.",
        "blockedInputMessaging": BLOCKED,
        "blockedOutputsMessaging": BLOCKED,
        "sensitiveInformationPolicyConfig": {
            "piiEntitiesConfig": [{"type": t, "action": "ANONYMIZE"} for t in PII_TYPES]
        },
        "contentPolicyConfig": {
            "filtersConfig": [
                {"type": t, "inputStrength": "HIGH", "outputStrength": "HIGH"} for t in FILTERS
            ]
            + [{"type": "PROMPT_ATTACK", "inputStrength": "HIGH", "outputStrength": "NONE"}]
        },
        "topicPolicyConfig": {
            "topicsConfig": [
                {
                    "name": "LegalAndMedicalAdvice",
                    "definition": "Providing legal, medical or financial advice to customers.",
                    "examples": ["Can I sue the seller?", "What dose should I take?"],
                    "type": "DENY",
                }
            ]
        },
    }


def main() -> None:
    client = make_session(Settings()).client("bedrock")
    existing = [g for g in client.list_guardrails()["guardrails"] if g["name"] == NAME]
    guardrail_id = (
        existing[0]["id"] if existing else client.create_guardrail(**build_config())["guardrailId"]
    )
    version = client.create_guardrail_version(guardrailIdentifier=guardrail_id)["version"]
    print(f"COPILOT_GUARDRAIL_ID={guardrail_id}")
    print(f"COPILOT_GUARDRAIL_VERSION={version}")


if __name__ == "__main__":
    main()
