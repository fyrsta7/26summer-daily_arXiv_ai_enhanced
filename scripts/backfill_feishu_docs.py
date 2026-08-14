#!/usr/bin/env python3
"""Backfill date-named Markdown digests from a Git ref into Feishu docs."""

from __future__ import annotations

import argparse
import re
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

try:
    from scripts.publish_feishu_doc import publish
except ModuleNotFoundError:  # Direct execution: python scripts/backfill_feishu_docs.py
    from publish_feishu_doc import publish


DATE_FILE = re.compile(r"data/(\d{4}-\d{2}-\d{2})\.md$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder-token", required=True)
    parser.add_argument("--repo", default=".")
    parser.add_argument("--git-ref", default="origin/main")
    parser.add_argument("--start-date")
    parser.add_argument("--end-date")
    parser.add_argument("--categories", default="cs.AI,cs.SE")
    parser.add_argument("--identity", choices=("user", "bot"), default="user")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--delay-seconds", type=float, default=1.0)
    parser.add_argument("--list-only", action="store_true")
    return parser.parse_args()


def list_digest_dates(repo: Path, git_ref: str) -> list[str]:
    output = subprocess.check_output(
        ["git", "-C", str(repo), "ls-tree", "-r", "--name-only", git_ref, "data"],
        text=True,
    )
    return sorted(
        match.group(1)
        for line in output.splitlines()
        if (match := DATE_FILE.fullmatch(line))
    )


def select_dates(dates: list[str], start_date: str | None, end_date: str | None) -> list[str]:
    return [
        date
        for date in dates
        if (not start_date or date >= start_date) and (not end_date or date <= end_date)
    ]


def read_digest(repo: Path, git_ref: str, date: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), "show", f"{git_ref}:data/{date}.md"],
        text=True,
    )


def publish_date(args: argparse.Namespace, repo: Path, date: str) -> str:
    markdown = read_digest(repo, args.git_ref, date)
    publish_args = argparse.Namespace(
        date=date,
        folder_token=args.folder_token,
        markdown=None,
        categories=args.categories,
        no_new_content=False,
        identity=args.identity,
    )
    last_error: Exception | None = None
    for attempt in range(6):
        try:
            result = publish(publish_args, markdown_text=markdown)
            if args.delay_seconds:
                time.sleep(args.delay_seconds)
            return result
        except Exception as error:  # Continue the batch and make retry resumable.
            last_error = error
            if attempt < 5:
                if "frequency limit" in str(error).casefold():
                    time.sleep(30 * (attempt + 1))
                else:
                    time.sleep(2**attempt)
    assert last_error is not None
    raise last_error


def main() -> None:
    args = parse_args()
    if not 1 <= args.workers <= 8:
        raise ValueError("--workers must be between 1 and 8")
    if args.delay_seconds < 0:
        raise ValueError("--delay-seconds must be non-negative")
    repo = Path(args.repo).resolve()
    dates = select_dates(
        list_digest_dates(repo, args.git_ref), args.start_date, args.end_date
    )
    if not dates:
        raise RuntimeError("No date-named Markdown digests matched the requested range")
    print(f"Selected {len(dates)} digests: {dates[0]} through {dates[-1]}")
    if args.list_only:
        return

    failures: list[tuple[str, str]] = []
    completed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(publish_date, args, repo, date): date for date in dates}
        for future in as_completed(futures):
            date = futures[future]
            try:
                future.result()
            except Exception as error:
                failures.append((date, str(error)))
                print(f"FAILED {date}: {error}")
            else:
                completed += 1
                print(f"PROGRESS {completed}/{len(dates)}: {date}")

    if failures:
        failed_dates = ", ".join(date for date, _ in sorted(failures))
        raise RuntimeError(f"Feishu backfill failed for {len(failures)} dates: {failed_dates}")
    print(f"Feishu backfill complete: {completed} date documents")


if __name__ == "__main__":
    main()
