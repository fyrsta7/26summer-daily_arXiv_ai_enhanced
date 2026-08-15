#!/usr/bin/env python3
"""Append one Daily arXiv date to the Feishu reading tracker idempotently."""

from __future__ import annotations

import argparse
import re
import time

try:
    from scripts.publish_feishu_doc import run_lark
except ModuleNotFoundError:  # Direct execution from scripts/
    from publish_feishu_doc import run_lark


DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--doc", required=True)
    parser.add_argument("--date", required=True)
    parser.add_argument("--identity", choices=("user", "bot"), default="user")
    return parser.parse_args()


def append_content(date: str) -> str:
    month_heading = f"<h1>{date[:7]}</h1>" if date.endswith("-01") else ""
    return f"{month_heading}<checkbox done=\"false\">{date}</checkbox>"


def run_with_retry(arguments: list[str], identity: str, input_text: str | None = None) -> dict:
    last_error: RuntimeError | None = None
    for attempt in range(4):
        try:
            return run_lark(arguments, input_text=input_text)
        except RuntimeError as error:
            last_error = error
            if attempt < 3:
                time.sleep(5 * (attempt + 1))
    assert last_error is not None
    raise last_error


def main() -> None:
    args = parse_args()
    if not DATE_RE.fullmatch(args.date):
        raise ValueError(f"Invalid date: {args.date}")

    current = run_with_retry(
        [
            "docs",
            "+fetch",
            "--as",
            args.identity,
            "--doc",
            args.doc,
            "--doc-format",
            "xml",
            "--scope",
            "keyword",
            "--keyword",
            args.date,
            "--format",
            "json",
        ],
        args.identity,
    )
    content = ((current.get("data") or {}).get("document") or {}).get("content", "")
    if args.date in content:
        print(f"Reading tracker already contains {args.date}; skipping duplicate")
        return

    result = run_with_retry(
        [
            "docs",
            "+update",
            "--as",
            args.identity,
            "--doc",
            args.doc,
            "--command",
            "append",
            "--doc-format",
            "xml",
            "--content",
            "-",
            "--format",
            "json",
        ],
        args.identity,
        input_text=append_content(args.date),
    )
    if ((result.get("data") or {}).get("result")) != "success":
        raise RuntimeError(f"Reading tracker update did not succeed: {result}")
    print(f"Added {args.date} to the Feishu reading tracker")


if __name__ == "__main__":
    main()
