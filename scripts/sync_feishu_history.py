#!/usr/bin/env python3
"""Overwrite existing historical Feishu digests from a local data directory."""

from __future__ import annotations

import argparse
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from publish_feishu_doc import list_folder_items, normalize_digest_markdown, run_lark


TITLE_RE = re.compile(r"^Daily arXiv (\d{4}-\d{2}-\d{2})$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder-token", required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--identity", choices=("user", "bot"), default="user")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--retries", type=int, default=6)
    return parser.parse_args()


def find_documents(folder_token: str, identity: str) -> dict[str, list[dict[str, Any]]]:
    documents: dict[str, list[dict[str, Any]]] = {}
    month_folders = [
        item
        for item in list_folder_items(folder_token, identity)
        if item.get("type") == "folder" and re.fullmatch(r"\d{4}-\d{2}", str(item.get("name", "")))
    ]
    for month_folder in month_folders:
        for item in list_folder_items(str(month_folder["token"]), identity):
            match = TITLE_RE.fullmatch(str(item.get("name", "")))
            if item.get("type") == "docx" and match:
                documents.setdefault(match.group(1), []).append(item)
    return documents


def overwrite_document(
    document: dict[str, Any], content: str, identity: str, retries: int
) -> None:
    token = str(document["token"])
    for attempt in range(retries):
        try:
            envelope = run_lark(
                [
                    "docs",
                    "+update",
                    "--as",
                    identity,
                    "--doc",
                    token,
                    "--command",
                    "overwrite",
                    "--doc-format",
                    "markdown",
                    "--content",
                    "-",
                    "--format",
                    "json",
                ],
                input_text=content,
            )
            data = envelope.get("data") or {}
            if data.get("result") != "success":
                raise RuntimeError(f"unexpected update result: {data!r}")
            return
        except RuntimeError:
            if attempt + 1 == retries:
                raise
            time.sleep(min(2 ** (attempt + 1), 30))


def main() -> None:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be positive")

    markdown_files = {
        path.stem: path
        for path in sorted(args.data_dir.glob("????-??-??.md"))
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", path.stem)
    }
    documents = find_documents(args.folder_token, args.identity)
    missing = sorted(set(markdown_files) - set(documents))
    if missing:
        raise RuntimeError(f"Feishu documents missing for {len(missing)} dates: {missing}")

    jobs: list[tuple[str, dict[str, Any], str]] = []
    for date, path in markdown_files.items():
        content = normalize_digest_markdown(path.read_text(encoding="utf-8"))
        jobs.extend((date, document, content) for document in documents[date])

    completed = 0
    progress_lock = threading.Lock()
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                overwrite_document, document, content, args.identity, args.retries
            ): (date, document)
            for date, document, content in jobs
        }
        for future in as_completed(futures):
            date, document = futures[future]
            future.result()
            with progress_lock:
                completed += 1
                print(
                    f"Synced {completed}/{len(jobs)}: {date} {document['token']}",
                    flush=True,
                )

    duplicate_dates = sorted(date for date, items in documents.items() if len(items) > 1)
    print(
        f"Completed: {len(markdown_files)} dates, {len(jobs)} documents; "
        f"duplicate dates updated: {duplicate_dates}",
        flush=True,
    )


if __name__ == "__main__":
    main()
