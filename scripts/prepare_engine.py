#!/usr/bin/env python3
"""Fetch and verify the exact SGLang source used by the inference launcher.

This command does not install SGLang or its dependencies, import the engine,
or run any GPU code. Keep this checkout separate from pip/build operations.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile


UPSTREAM = "https://github.com/sgl-project/sglang.git"
COMMIT = "29481685462732237d80d86076d6563e1f658102"
DIGEST = "2bd3879198c21a3c7448ff4e98ca35666110048c613d6a299ac83321d8edd427"
FILE_COUNT = 3124


def git(directory, *args, capture=False):
    command = [
        "git", "-c", "core.hooksPath=/dev/null", "-c", "core.autocrlf=false",
        "-c", "http.version=HTTP/1.1",
        "-C", str(directory), *args,
    ]
    return subprocess.run(command, check=True, text=True, capture_output=capture, timeout=600)


def verify(directory):
    if not (directory / ".git").is_dir():
        raise ValueError(f"Not a dedicated Git checkout: {directory}")
    actual_commit = git(directory, "rev-parse", "HEAD", capture=True).stdout.strip()
    if actual_commit != COMMIT:
        raise ValueError(f"Unexpected SGLang commit: {actual_commit}")
    git(directory, "diff", "--quiet", "HEAD", "--")
    python_root = directory / "python"
    required = python_root / "sglang/srt/speculative/dspark_components/dspark_draft.py"
    if not required.is_file():
        raise ValueError("The pinned DSpark implementation is missing.")
    paths = sorted(python_root.rglob("*.py"))
    lines = [
        f"{path.relative_to(python_root).as_posix()}\0"
        f"{hashlib.sha256(path.read_bytes()).hexdigest()}"
        for path in paths
    ]
    actual_digest = hashlib.sha256("\n".join(lines).encode()).hexdigest()
    if len(paths) != FILE_COUNT or actual_digest != DIGEST:
        raise ValueError(
            f"Source verification failed: {len(paths)} Python files, SHA256 {actual_digest}. "
            "Use a clean checkout; do not install/build into this source directory."
        )
    return python_root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=Path("third_party/sglang"))
    parser.add_argument("--environment", type=Path, default=Path("environment.json"))
    parser.add_argument("--verify-only", action="store_true", help="Verify an existing checkout without network access")
    args = parser.parse_args()
    environment = json.loads(args.environment.read_text())
    expected = (COMMIT, DIGEST, FILE_COUNT)
    recorded = tuple(environment.get(key) for key in (
        "sglang_commit", "engine_python_sha256", "engine_python_file_count"
    ))
    if recorded != expected:
        raise ValueError("environment.json does not match this release's pinned engine.")
    destination = args.destination.expanduser().resolve()
    if destination.exists():
        python_root = verify(destination)
    else:
        if args.verify_only:
            raise ValueError(f"Checkout does not exist: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=".ready-sglang-", dir=destination.parent))
        try:
            git(staging, "init", "--quiet", "--template=")
            git(staging, "remote", "add", "origin", UPSTREAM)
            git(staging, "fetch", "--depth=1", "origin", COMMIT)
            git(staging, "checkout", "--quiet", "--detach", "FETCH_HEAD")
            verify(staging)
            # A second process must never cause replacement of an existing checkout.
            if destination.exists():
                raise ValueError(f"Destination appeared while fetching: {destination}")
            staging.rename(destination)
            python_root = destination / "python"
        except Exception:
            print(f"Fetch/verification failed; temporary checkout preserved at {staging}", file=sys.stderr)
            raise
    print(f"Verified SGLang commit: {COMMIT}")
    print(f"Verified Python files: {FILE_COUNT}")
    print(f"Verified Python-source SHA256: {DIGEST}")
    print(f"Use --sglang-source {python_root}")
    print("Runtime dependencies were not installed or validated by this command.")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
