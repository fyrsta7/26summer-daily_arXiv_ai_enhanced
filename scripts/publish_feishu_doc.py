#!/usr/bin/env python3
"""Publish one idempotent daily arXiv digest as a Feishu cloud document."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator


TITLE_PREFIX = "Daily arXiv"


@contextmanager
def publication_lock(folder_token: str, title: str) -> Iterator[None]:
    """Serialize same-title publishers running on this server."""
    lock_key = hashlib.sha256(f"{folder_token}\0{title}".encode()).hexdigest()
    lock_path = Path(tempfile.gettempdir()) / f"daily-arxiv-feishu-{lock_key}.lock"
    with lock_path.open("w") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        yield


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", required=True)
    parser.add_argument("--folder-token", required=True)
    parser.add_argument("--markdown")
    parser.add_argument("--categories", default="cs.SE")
    parser.add_argument("--no-new-content", action="store_true")
    parser.add_argument("--identity", choices=("user", "bot"), default="user")
    return parser.parse_args()


def _parse_json_envelope(output: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    for index, character in enumerate(output):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(output[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and "ok" in value:
            return value
    raise RuntimeError("lark-cli did not return a JSON envelope")


def run_lark(arguments: list[str], *, input_text: str | None = None) -> dict[str, Any]:
    env = os.environ.copy()
    env["LARKSUITE_CLI_NO_UPDATE_NOTIFIER"] = "1"
    env["LARKSUITE_CLI_NO_SKILLS_NOTIFIER"] = "1"
    completed = subprocess.run(
        ["lark-cli", *arguments],
        input=input_text,
        text=True,
        capture_output=True,
        check=False,
        env=env,
    )
    combined = "\n".join(part for part in (completed.stdout, completed.stderr) if part)
    try:
        envelope = _parse_json_envelope(combined)
    except RuntimeError:
        if completed.returncode:
            raise RuntimeError(f"lark-cli failed with exit code {completed.returncode}") from None
        raise
    if completed.returncode or envelope.get("ok") is not True:
        error = envelope.get("error") or {}
        message = error.get("message") or f"exit code {completed.returncode}"
        hint = error.get("hint")
        raise RuntimeError(f"lark-cli failed: {message}" + (f"; {hint}" if hint else ""))
    return envelope


def find_existing_document(
    folder_token: str, title: str, identity: str
) -> dict[str, Any] | None:
    page_token = ""
    while True:
        arguments = [
            "drive",
            "files",
            "list",
            "--as",
            identity,
            "--folder-token",
            folder_token,
            "--page-size",
            "200",
            "--format",
            "json",
        ]
        if page_token:
            arguments.extend(["--page-token", page_token])
        data = run_lark(arguments).get("data") or {}
        for item in data.get("files") or []:
            if item.get("type") == "docx" and item.get("name") == title:
                return item
        if not data.get("has_more"):
            return None
        page_token = str(data.get("next_page_token") or "")
        if not page_token:
            raise RuntimeError("Feishu Drive pagination returned has_more without next_page_token")


def normalize_digest_markdown(markdown: str) -> str:
    """Remove HTML-only wrappers that Feishu's Markdown importer cannot preserve."""
    output: list[str] = []
    for line in markdown.splitlines():
        stripped = line.strip()
        if re.fullmatch(r"<div\b[^>]*></div>", stripped, flags=re.IGNORECASE):
            continue
        if stripped.lower() in {"<details>", "</details>"}:
            continue
        if stripped.lower() == "<summary>details</summary>":
            output.append("#### 详细信息")
            continue
        output.append(line)
    return "\n".join(output).strip() + "\n"


def build_content(args: argparse.Namespace) -> str:
    if args.no_new_content:
        return (
            f"# {TITLE_PREFIX} {args.date}\n\n"
            "今日的 Daily arXiv 任务已正常运行。\n\n"
            f"在去重后没有发现新的 {args.categories} 论文，因此今天没有论文条目。\n"
        )
    if not args.markdown:
        raise ValueError("--markdown is required unless --no-new-content is set")
    if args.markdown == "-":
        return normalize_digest_markdown(sys.stdin.read())
    path = Path(args.markdown)
    if not path.is_file():
        raise FileNotFoundError(f"Markdown digest does not exist: {path}")
    return normalize_digest_markdown(path.read_text(encoding="utf-8"))


def publish(args: argparse.Namespace, markdown_text: str | None = None) -> str:
    title = f"{TITLE_PREFIX} {args.date}"
    with publication_lock(args.folder_token, title):
        existing = find_existing_document(args.folder_token, title, args.identity)
        if existing:
            url = existing.get("url") or ""
            print(
                f"Feishu document already exists; skipping duplicate: {title} {url}".rstrip()
            )
            return str(url)

        content = (
            normalize_digest_markdown(markdown_text)
            if markdown_text is not None
            else build_content(args)
        )
        envelope = run_lark(
            [
                "docs",
                "+create",
                "--as",
                args.identity,
                "--parent-token",
                args.folder_token,
                "--title",
                title,
                "--doc-format",
                "markdown",
                "--content",
                "-",
                "--format",
                "json",
            ],
            input_text=content,
        )
        document = (envelope.get("data") or {}).get("document") or {}
        document_id = str(document.get("document_id") or "")
        url = str(document.get("url") or "")
        if not document_id:
            raise RuntimeError("Feishu document creation succeeded without a document_id")
        print(f"Created Feishu document: {title} {url}".rstrip())
        return url


def main() -> None:
    publish(parse_args())


if __name__ == "__main__":
    main()
