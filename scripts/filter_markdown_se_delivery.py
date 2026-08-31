#!/usr/bin/env python3
"""Remove unselected cs.SE paper blocks from a rendered daily Markdown digest."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


MARKER_SECTION = re.compile(
    r"\n*<!-- semopt-se-selection:start -->.*?<!-- semopt-se-selection:end -->\n*", re.DOTALL
)
PAPER_START = re.compile(r"(?m)^### \[\d+\] .*$")
ARXIV_ID = re.compile(r"https?://arxiv\.org/abs/([^\s)]+)")
SE_TOTAL = re.compile(r"(?im)^- \[cs\.SE\]\(#cs\.se\) \[Total: \d+\]$")
PAPER_NUMBER = re.compile(r"(?m)^### \[\d+\]")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--markdown", required=True)
    return parser.parse_args()


def selected_se_ids(records: list[dict[str, Any]]) -> set[str]:
    ids = set()
    for record in records:
        if "cs.SE" not in set(record.get("categories") or []):
            continue
        decision = record.get("se_selection")
        if not isinstance(decision, dict) or "relevant" not in decision:
            raise ValueError(f"missing cs.SE selection decision for {record.get('id', '<unknown>')}")
        if decision["relevant"] is True:
            ids.add(str(record["id"]))
    return ids


def filter_markdown(markdown: str, selected_ids: set[str]) -> tuple[str, int]:
    markdown = MARKER_SECTION.sub("\n", markdown).rstrip() + "\n"
    starts = list(PAPER_START.finditer(markdown))
    output: list[str] = []
    cursor = 0
    removed = 0
    selected_se_count = 0
    for index, start in enumerate(starts):
        end = starts[index + 1].start() if index + 1 < len(starts) else len(markdown)
        block = markdown[start.start():end]
        category_is_se = "Main category: cs.SE" in block
        arxiv = ARXIV_ID.search(block)
        if category_is_se:
            if not arxiv:
                raise ValueError(f"could not find arXiv ID for cs.SE block: {block[:120]!r}")
            if arxiv.group(1) not in selected_ids:
                output.append(markdown[cursor:start.start()])
                cursor = end
                removed += 1
                continue
            selected_se_count += 1
    output.append(markdown[cursor:])
    markdown = "".join(output)
    markdown = SE_TOTAL.sub(f"- [cs.SE](#cs.se) [Total: {selected_se_count}]", markdown)
    number = 0

    def renumber(match: re.Match[str]) -> str:
        nonlocal number
        number += 1
        return f"### [{number}]"

    markdown = PAPER_NUMBER.sub(renumber, markdown)
    return markdown.rstrip() + "\n", removed


def main() -> None:
    args = parse_args()
    data_path = Path(args.data)
    markdown_path = Path(args.markdown)
    records = [json.loads(line) for line in data_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    updated, removed = filter_markdown(markdown_path.read_text(encoding="utf-8"), selected_se_ids(records))
    markdown_path.write_text(updated, encoding="utf-8")
    print(json.dumps({"markdown": str(markdown_path), "cs.SE_removed": removed}, ensure_ascii=False))


if __name__ == "__main__":
    main()
