#!/usr/bin/env python3
"""Append the rendered SE selection to existing date-named Feishu digest docs."""

from __future__ import annotations

import argparse
import html
import json
import re
import time
from pathlib import Path
from typing import Any

try:
    from scripts.publish_feishu_doc import list_folder_items, run_lark
except ModuleNotFoundError:  # Direct execution from scripts/
    from publish_feishu_doc import list_folder_items, run_lark


HEADING = "SE 精选（SemOpt / Coding Agent 相关）"
MONTH = re.compile(r"\d{4}-\d{2}")
DATE = re.compile(r"\d{4}-\d{2}-\d{2}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--folder-token", required=True)
    parser.add_argument("--start-date")
    parser.add_argument("--end-date")
    parser.add_argument("--identity", choices=("user", "bot"), default="user")
    return parser.parse_args()


def selected_paths(args: argparse.Namespace) -> list[Path]:
    paths = []
    for path in sorted(Path(args.data_dir).glob("????-??-??.jsonl")):
        date = path.stem
        if (not args.start_date or date >= args.start_date) and (not args.end_date or date <= args.end_date):
            paths.append(path)
    return paths


def section_xml(items: list[dict[str, Any]]) -> str:
    selected = [item for item in items if "cs.SE" in set(item.get("categories") or []) and item.get("se_selection", {}).get("relevant")]
    selected.sort(key=lambda item: (-float(item["se_selection"].get("score", 0)), item.get("title", "")))
    chunks = [f"<h2>{HEADING}</h2>", "<p>以下为从原始 cs.SE 列表中筛出的 SemOpt / Coding Agent 相关论文。</p>"]
    if not selected:
        return "".join(chunks + ["<p>本日没有筛选出与 SemOpt / Coding Agent 明显相关的 cs.SE 论文。</p>"])
    chunks.append("<ol>")
    for item in selected:
        decision = item["se_selection"]
        title = html.escape(str(item.get("title") or "Untitled"))
        url = html.escape(str(item.get("abs") or ""), quote=True)
        reason = html.escape(str(decision.get("reason_zh") or ""))
        score = float(decision.get("score", 0))
        chunks.append(f'<li><p><a type="url-preview" href="{url}">{title}</a><br/>相关度 {score:.2f}：{reason}</p></li>')
    chunks.append("</ol>")
    return "".join(chunks)


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
    months = {item["name"]: item for item in list_folder_items(root_token, identity) if item.get("type") == "folder" and MONTH.fullmatch(str(item.get("name") or ""))}
    documents: dict[str, dict[str, Any]] = {}
    for month, folder in months.items():
        token = str(folder.get("token") or folder.get("folder_token") or "")
        if not token:
            continue
        for item in list_folder_items(token, identity):
            name = str(item.get("name") or "")
            if item.get("type") == "docx" and name.startswith("Daily arXiv "):
                date = name.removeprefix("Daily arXiv ")
                if DATE.fullmatch(date) and date.startswith(month):
                    documents[date] = item
    return documents


def main() -> None:
    args = parse_args()
    paths = selected_paths(args)
    if not paths:
        raise RuntimeError("no date-named JSONL files matched the requested range")
    documents = map_documents(args.folder_token, args.identity)
    updated = skipped = missing = 0
    missing_dates: list[str] = []
    for index, path in enumerate(paths, 1):
        date = path.stem
        document = documents.get(date)
        if not document:
            missing += 1
            missing_dates.append(date)
            continue
        doc = str(document.get("token") or "")
        current = retry(["docs", "+fetch", "--as", args.identity, "--doc", doc, "--doc-format", "xml", "--scope", "keyword", "--keyword", HEADING, "--format", "json"])
        content = str(((current.get("data") or {}).get("document") or {}).get("content") or "")
        if HEADING in content:
            skipped += 1
            continue
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        result = retry(["docs", "+update", "--as", args.identity, "--doc", doc, "--command", "append", "--doc-format", "xml", "--content", "-", "--format", "json"], input_text=section_xml(records))
        if ((result.get("data") or {}).get("result")) != "success":
            raise RuntimeError(f"Feishu update did not succeed for {date}: {result}")
        updated += 1
        print(f"FEISHU_PROGRESS {index}/{len(paths)} {date}", flush=True)
    print(json.dumps({"documents_seen": len(paths), "updated": updated, "already_updated": skipped, "missing": missing, "missing_dates": missing_dates}, ensure_ascii=False))


if __name__ == "__main__":
    main()
