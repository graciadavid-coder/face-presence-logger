"""Reject staged or tracked runtime files that may contain private data."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import PurePosixPath

ALLOWED_FILES = {"data/.gitkeep", "models/.gitkeep"}
BLOCKED_DIRECTORIES = ("data/", "models/", ".venv/", "captures/")
BLOCKED_SUFFIXES = {
    ".db",
    ".db-wal",
    ".db-shm",
    ".sqlite",
    ".sqlite3",
    ".onnx",
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".mp4",
    ".avi",
    ".mov",
    ".csv",
}


def _git_paths(staged: bool) -> list[str]:
    if staged:
        command = ["git", "diff", "--cached", "--name-only", "--diff-filter=ACMR"]
    else:
        command = ["git", "ls-files"]

    result = subprocess.run(
        command,
        capture_output=True,
        check=True,
        text=True,
    )
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def find_violations(paths: list[str]) -> list[str]:
    violations: list[str] = []
    for path in paths:
        normalized = path.replace("\\", "/")
        if normalized in ALLOWED_FILES:
            continue

        in_blocked_directory = any(
            normalized == directory.rstrip("/") or normalized.startswith(directory)
            for directory in BLOCKED_DIRECTORIES
        )
        suffix = PurePosixPath(normalized).suffix.lower()
        if in_blocked_directory or suffix in BLOCKED_SUFFIXES:
            violations.append(path)
    return violations


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fail when private runtime files are tracked or staged."
    )
    parser.add_argument(
        "--staged",
        action="store_true",
        help="Check only the files staged for the next commit.",
    )
    args = parser.parse_args()

    try:
        paths = _git_paths(args.staged)
    except subprocess.CalledProcessError as exc:
        print(f"Could not read Git paths: {exc}", file=sys.stderr)
        return 1

    violations = find_violations(paths)
    if not violations:
        print("Repository hygiene check passed.")
        return 0

    print("Blocked files detected:", file=sys.stderr)
    for path in violations:
        print(f"  {path}", file=sys.stderr)
    print(
        "Runtime data, databases, model binaries and captured media must stay out of Git.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
