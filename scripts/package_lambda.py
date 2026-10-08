"""Build dist/support-copilot.zip for AWS Lambda (python3.12, x86_64) from any OS.

    uv run python scripts/package_lambda.py

Dependencies are installed for the Linux target platform; CLI-only packages are left out.
"""

from __future__ import annotations

import shutil
import subprocess
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BUILD = ROOT / "build" / "lambda"
ZIP_PATH = ROOT / "dist" / "support-copilot.zip"
RUNTIME_DEPS = ["boto3", "pydantic", "pydantic-settings"]
SKIP_DIRS = {"__pycache__", "tests"}


def main() -> None:
    shutil.rmtree(BUILD, ignore_errors=True)
    BUILD.mkdir(parents=True)
    subprocess.run(
        [
            "uv",
            "pip",
            "install",
            "--quiet",
            "--target",
            str(BUILD),
            "--python-platform",
            "x86_64-manylinux2014",
            "--python-version",
            "3.12",
            *RUNTIME_DEPS,
        ],
        check=True,
    )
    shutil.copytree(
        ROOT / "src" / "copilot", BUILD / "copilot", ignore=shutil.ignore_patterns("__pycache__")
    )

    ZIP_PATH.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(BUILD.rglob("*")):
            if path.is_file() and not (SKIP_DIRS & set(path.relative_to(BUILD).parts)):
                zf.write(path, path.relative_to(BUILD).as_posix())
    size_mb = ZIP_PATH.stat().st_size / 1_048_576
    print(f"Wrote {ZIP_PATH} ({size_mb:.1f} MB)")
    if size_mb > 50:
        raise SystemExit("Package exceeds Lambda's 50 MB direct-upload limit")


if __name__ == "__main__":
    main()
