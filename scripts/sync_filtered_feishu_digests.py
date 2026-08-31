#!/usr/bin/env python3
"""Replace generated Feishu daily digests with their filtered Markdown source."""

from __future__ import annotations

import argparse
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

try:
    from scripts.publish_feishu_doc import list_folder_items, normalize_digest_markdown, run_lark
except ModuleNotFoundError:  # Direct execution from scripts/
    from publish_feishu_doc import list_folder_items, normalize_digest_markdown, run_lark


MONTH = re.compile(r"\d{4}-\d{2}")
DATE = re.compile(r"\d{4}-\d{2}-\d{2}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--folder-token", required=True)
    parser.add_argument("--start-date")
    parser.add_argument("--end-date")
    parser.add_argument("--workers", type=int, default=4, help="Concurrent independent document updates (1-8)")
    parser.add_argument("--identity", choices=("user", "bot"), default="user")
    return parser.parse_args()


def markdown_paths(args: argparse.Namespace) -> list[Path]:
    paths = []
    for path in sorted(Path(args.data_dir).glob("????-??-??.md")):
        date = path.stem
        if (not args.start_date or date >= args.start_date) and (not args.end_date or date <= args.end_date):
            paths.append(path)
    return paths


def retry(arguments: list[str], input_text: str | None = None) -> dict[str, Any]:
    last_error: RuntimeError | None = None
    for attempt in range(5):
        try:
            return run_lark(arguments, input_text=input_text)
        except RuntimeError as error:
            last_error = error
            if attempt < 4:
                time.sleep(2 ** attempt)
    assert last_error is not None
    raise last_error


def map_documents(root_token: str, identity: str) -> dict[str, dict[str, Any]]:
    months = {
        item["name"]: item
        for item in list_folder_items(root_token, identity)
        if item.get("type") == "folder" and MONTH.fullmatch(str(item.get("name") or ""))
    }
    documents: dict[str, dict[str, Any]] = {}
    for month, folder in months.items():
        token = str(folder.get("token") or folder.get("folder_token") or "")
        if not token:
            continue
        for item in list_folder_items(token, identity):
            name = str(item.get("name") or "")
            date = name.removeprefix("Daily arXiv ")
            if item.get("type") == "docx" and DATE.fullmatch(date) and date.startswith(month):
                documents[date] = item
    return documents


def main() -> None:
    args = parse_args()
    if not 1 <= args.workers <= 8:
        raise ValueError("--workers must be between 1 and 8")
    paths = markdown_paths(args)
    if not paths:
        raise RuntimeError("no daily Markdown digests matched the requested date range")
    documents = map_documents(args.folder_token, args.identity)
    targets: list[tuple[str, Path, str]] = []
    missing_dates: list[str] = []
    for path in paths:
        date = path.stem
        document = documents.get(date)
        if not document:
            missing_dates.append(date)
            continue
        doc = str(document.get("token") or "")
        if not doc:
            missing_dates.append(date)
            continue
        targets.append((date, path, doc))

    def sync_one(date: str, path: Path, doc: str) -> str:
        # Read immediately before the intentional complete replacement so the
        # target is verified as the expected existing generated digest.
        current = retry([
            "docs", "+fetch", "--as", args.identity, "--doc", doc,
            "--doc-format", "markdown", "--scope", "outline", "--format", "json",
        ])
        if not ((current.get("data") or {}).get("document") or {}).get("content"):
            raise RuntimeError(f"could not read existing Feishu digest for {date}")
        content = normalize_digest_markdown(path.read_text(encoding="utf-8"))
        result = retry([
            "docs", "+update", "--as", args.identity, "--doc", doc,
            "--command", "overwrite", "--doc-format", "markdown", "--content", "-", "--format", "json",
        ], input_text=content)
        if ((result.get("data") or {}).get("result")) != "success":
            raise RuntimeError(f"Feishu overwrite did not succeed for {date}: {result}")
        return date

    updated = 0
    failures: list[dict[str, str]] = []
    with ThreadPoolExecutor(max_workers=min(args.workers, len(targets))) as executor:
        futures = {
            executor.submit(sync_one, date, path, doc): date
            for date, path, doc in targets
        }
        for future in as_completed(futures):
            target_date = futures[future]
            try:
                date = future.result()
            except Exception as error:
                # Let the remaining independent documents finish.  A later
                # targeted invocation can retry only the failed dates.
                failures.append({"date": target_date, "error": str(error)})
                print(f"FEISHU_ERROR {target_date}: {error}", flush=True)
                continue
            updated += 1
            print(f"FEISHU_PROGRESS {updated}/{len(paths)} {date}", flush=True)
    summary = {
        "documents_seen": len(paths),
        "updated": updated,
        "failed": len(failures),
        "failures": failures,
        "missing": len(missing_dates),
        "missing_dates": missing_dates,
    }
    print(json.dumps(summary, ensure_ascii=False))
    if failures:
        raise RuntimeError(f"{len(failures)} Feishu digest update(s) failed")


if __name__ == "__main__":
    main()
