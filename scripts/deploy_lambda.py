"""Deploy, invoke or delete the support-copilot Lambda function.

    uv run python scripts/deploy_lambda.py deploy --role-arn arn:aws:iam::123456789012:role/NAME
    uv run python scripts/deploy_lambda.py invoke --text "My order arrived broken"
    uv run python scripts/deploy_lambda.py delete

The role must allow the AI services you enable (Bedrock, Comprehend, Textract, ...). Settings are
copied from the current COPILOT_* configuration into the function's environment variables.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from botocore.exceptions import ClientError

from copilot.aws import make_session
from copilot.config import Settings

FUNCTION_NAME = "support-copilot"
ZIP_PATH = Path(__file__).resolve().parent.parent / "dist" / "support-copilot.zip"


def function_environment(settings: Settings) -> dict[str, str]:
    """Pass every COPILOT_* setting except the local profile, which has no meaning in Lambda."""
    values = settings.model_dump(exclude={"aws_profile"})
    return {f"COPILOT_{key.upper()}": str(value) for key, value in values.items()}


def deploy(client, role_arn: str, settings: Settings) -> None:
    code = ZIP_PATH.read_bytes()
    config = {
        "Runtime": "python3.12",
        "Role": role_arn,
        "Handler": "copilot.handler.handler",
        "Timeout": 60,
        "MemorySize": 512,
        "Environment": {"Variables": function_environment(settings)},
    }
    try:
        client.get_function(FunctionName=FUNCTION_NAME)
    except ClientError as exc:
        if exc.response["Error"]["Code"] != "ResourceNotFoundException":
            raise
        client.create_function(FunctionName=FUNCTION_NAME, Code={"ZipFile": code}, **config)
        client.get_waiter("function_active_v2").wait(FunctionName=FUNCTION_NAME)
        print(f"Created {FUNCTION_NAME}")
        return
    client.update_function_code(FunctionName=FUNCTION_NAME, ZipFile=code)
    client.get_waiter("function_updated_v2").wait(FunctionName=FUNCTION_NAME)
    client.update_function_configuration(FunctionName=FUNCTION_NAME, **config)
    client.get_waiter("function_updated_v2").wait(FunctionName=FUNCTION_NAME)
    print(f"Updated {FUNCTION_NAME}")


def invoke(client, payload: dict) -> None:
    resp = client.invoke(FunctionName=FUNCTION_NAME, Payload=json.dumps(payload).encode())
    print(json.dumps(json.loads(resp["Payload"].read()), indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    dep = sub.add_parser("deploy")
    dep.add_argument("--role-arn", required=True)
    inv = sub.add_parser("invoke")
    inv.add_argument("--text", default="")
    inv.add_argument("--event-file", type=Path, help="JSON event; overrides --text")
    sub.add_parser("delete")
    args = parser.parse_args()

    settings = Settings()
    client = make_session(settings).client("lambda")
    if args.command == "deploy":
        deploy(client, args.role_arn, settings)
    elif args.command == "invoke":
        payload = (
            json.loads(args.event_file.read_text()) if args.event_file else {"text": args.text}
        )
        invoke(client, payload)
    else:
        client.delete_function(FunctionName=FUNCTION_NAME)
        print(f"Deleted {FUNCTION_NAME}")


if __name__ == "__main__":
    main()
