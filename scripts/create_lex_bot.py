"""Create (or recreate) the SupportIntake Amazon Lex V2 bot and print its settings.

    uv run python scripts/create_lex_bot.py [--recreate] [--role-arn ARN]
    uv run python scripts/create_lex_bot.py --delete

The bot collects an order id, an issue type and optional details, then confirms. Those slots feed
the support pipeline (`copilot intake`). By default the Lex service-linked role of the current
account is used; pass --role-arn to use another role.
"""

from __future__ import annotations

import argparse
import time

from botocore.exceptions import ClientError

from copilot.aws import make_session
from copilot.config import Settings

BOT_NAME = "SupportIntake"
LOCALE = "en_US"
VERSION = "DRAFT"

ISSUE_TYPES = {
    "billing": ["charge", "refund", "invoice", "payment", "overcharged"],
    "shipping": ["delivery", "parcel", "package", "late", "tracking"],
    "product_defect": ["broken", "damaged", "faulty", "defective", "not working"],
    "account_access": ["login", "password", "locked out", "cannot sign in"],
    "other": ["question", "something else"],
}


def say(text: str) -> dict:
    return {"messageGroups": [{"message": {"plainTextMessage": {"value": text}}}]}


def prompt(text: str, retries: int = 2) -> dict:
    return {**say(text), "maxRetries": retries, "allowInterrupt": True}


INTENTS = {
    "ReportProblem": {
        "utterances": [
            "I have a problem with my order",
            "my order {OrderId} arrived broken",
            "I need help with order {OrderId}",
            "I want to report a problem",
            "something is wrong with my order",
            "I was charged twice",
            "my parcel is late",
            "I cannot log in to my account",
        ],
        "slots": [
            ("OrderId", "AMAZON.AlphaNumeric", "Required", "What is your order number?"),
            (
                "IssueType",
                "IssueType",
                "Required",
                "What kind of problem is it: billing, shipping, faulty product or account access?",
            ),
            ("Details", "AMAZON.FreeFormInput", "Optional", "Briefly describe what happened."),
        ],
        "confirm": "To confirm: a {IssueType} problem with order {OrderId}. Should I open a case?",
        "decline": "Okay, I have not opened a case. Tell me if you need anything else.",
        "closing": "Thanks. I have opened a case for order {OrderId} and our team will reply soon.",
    },
    "OrderStatus": {
        "utterances": [
            "where is my order",
            "track order {OrderId}",
            "what is the status of order {OrderId}",
            "has my order shipped",
        ],
        "slots": [
            ("OrderId", "AMAZON.AlphaNumeric", "Required", "What is your order number?"),
        ],
        "confirm": None,
        "decline": None,
        "closing": "I have passed order {OrderId} to our logistics team to confirm the status.",
    },
}


def settings_for(intent: dict) -> dict:
    out: dict = {
        "intentClosingSetting": {"closingResponse": say(intent["closing"]), "active": True}
    }
    if intent["confirm"]:
        out["intentConfirmationSetting"] = {
            "promptSpecification": prompt(intent["confirm"]),
            "declinationResponse": say(intent["decline"]),
            "active": True,
        }
    return out


def find_bot(lex, name: str) -> str | None:
    token: str | None = None
    while True:
        kwargs = {"nextToken": token} if token else {}
        page = lex.list_bots(**kwargs)
        for bot in page["botSummaries"]:
            if bot["botName"] == name:
                return bot["botId"]
        token = page.get("nextToken")
        if not token:
            return None


def delete_bot(lex, bot_id: str) -> None:
    lex.delete_bot(botId=bot_id, skipResourceInUseCheck=True)
    for _ in range(30):  # deletion is asynchronous; wait until the name is free again
        if find_bot(lex, BOT_NAME) is None:
            break
        time.sleep(2)
    print(f"Deleted bot {bot_id}")


def build(lex, role_arn: str) -> str:
    bot = lex.create_bot(
        botName=BOT_NAME,
        description="Collects order id and issue type before opening a support case.",
        roleArn=role_arn,
        dataPrivacy={"childDirected": False},
        idleSessionTTLInSeconds=300,
    )
    bot_id = bot["botId"]
    lex.get_waiter("bot_available").wait(botId=bot_id)
    lex.create_bot_locale(
        botId=bot_id, botVersion=VERSION, localeId=LOCALE, nluIntentConfidenceThreshold=0.4
    )
    lex.get_waiter("bot_locale_created").wait(botId=bot_id, botVersion=VERSION, localeId=LOCALE)
    base = {"botId": bot_id, "botVersion": VERSION, "localeId": LOCALE}

    slot_type = lex.create_slot_type(
        slotTypeName="IssueType",
        valueSelectionSetting={"resolutionStrategy": "TopResolution"},
        slotTypeValues=[
            {"sampleValue": {"value": v}, "synonyms": [{"value": s} for s in syns]}
            for v, syns in ISSUE_TYPES.items()
        ],
        **base,
    )
    type_ids = {"IssueType": slot_type["slotTypeId"]}

    for name, spec in INTENTS.items():
        common = {
            "sampleUtterances": [{"utterance": u} for u in spec["utterances"]],
            **settings_for(spec),
        }
        intent_id = lex.create_intent(intentName=name, **common, **base)["intentId"]
        priorities = []
        for order, (slot, type_name, constraint, ask) in enumerate(spec["slots"], start=1):
            slot_id = lex.create_slot(
                slotName=slot,
                intentId=intent_id,
                slotTypeId=type_ids.get(type_name, type_name),
                valueElicitationSetting={
                    "slotConstraint": constraint,
                    "promptSpecification": prompt(ask),
                },
                **base,
            )["slotId"]
            priorities.append({"priority": order, "slotId": slot_id})
        lex.update_intent(
            intentId=intent_id, intentName=name, slotPriorities=priorities, **common, **base
        )

    lex.build_bot_locale(**{k: v for k, v in base.items()})
    lex.get_waiter("bot_locale_built").wait(**base)
    return bot_id


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--recreate", action="store_true", help="Delete an existing bot first.")
    parser.add_argument("--delete", action="store_true", help="Delete the bot and exit.")
    parser.add_argument(
        "--role-arn", help="Role for the bot; defaults to the Lex service-linked role."
    )
    args = parser.parse_args()

    settings = Settings()
    session = make_session(settings)
    lex = session.client("lexv2-models")
    existing = find_bot(lex, BOT_NAME)

    if args.delete:
        if existing:
            delete_bot(lex, existing)
        return
    if existing and not args.recreate:
        print(f"Bot {BOT_NAME} already exists. Use --recreate to rebuild it.")
        print(f"COPILOT_LEX_BOT_ID={existing}")
        return
    if existing:
        delete_bot(lex, existing)

    account = session.client("sts").get_caller_identity()["Account"]
    role = args.role_arn or (
        f"arn:aws:iam::{account}:role/aws-service-role/lexv2.amazonaws.com/"
        "AWSServiceRoleForLexV2Bots"
    )
    try:
        bot_id = build(lex, role)
    except ClientError as exc:
        raise SystemExit(f"Lex build failed: {exc}") from exc
    print(f"COPILOT_LEX_BOT_ID={bot_id}")
    print("COPILOT_LEX_BOT_ALIAS_ID=TSTALIASID  # the built-in test alias of the DRAFT version")


if __name__ == "__main__":
    main()
