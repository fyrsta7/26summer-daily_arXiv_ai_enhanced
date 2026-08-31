#!/usr/bin/env python3
"""Keep only SemOpt-relevant cs.SE papers in delivery artifacts.

The AI relevance filter already removes irrelevant cs.AI papers.  This script
applies the corresponding decision for cs.SE papers before enhancement and
delivery, so Markdown, email, and Feishu contain one consistently filtered
paper set.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Filtered daily JSONL containing se_selection decisions")
    parser.add_argument("--enhanced", help="Optional AI-enhanced JSONL to filter using retained arXiv IDs")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records), encoding="utf-8")


def keep_for_delivery(record: dict[str, Any]) -> bool:
    """AI is already filtered upstream; cs.SE additionally needs its own decision."""
    categories = set(record.get("categories") or [])
    if "cs.SE" not in categories:
        return True
    decision = record.get("se_selection")
    if not isinstance(decision, dict) or "relevant" not in decision:
        raise ValueError(f"missing cs.SE selection decision for arXiv {record.get('id', '<unknown>')}")
    return decision["relevant"] is True


def main() -> None:
    args = parse_args()
    data_path = Path(args.data)
    records = read_jsonl(data_path)
    retained = [record for record in records if keep_for_delivery(record)]
    retained_ids = {str(record["id"]) for record in retained}
    removed_se = sum("cs.SE" in set(record.get("categories") or []) for record in records) - sum(
        "cs.SE" in set(record.get("categories") or []) for record in retained
    )
    write_jsonl(data_path, retained)

    enhanced_retained = None
    if args.enhanced:
        enhanced_path = Path(args.enhanced)
        enhanced_records = read_jsonl(enhanced_path)
        enhanced_retained = [record for record in enhanced_records if str(record.get("id")) in retained_ids]
        write_jsonl(enhanced_path, enhanced_retained)

    print(
        json.dumps(
            {
                "records_before": len(records),
                "records_after": len(retained),
                "cs.SE_removed": removed_se,
                "enhanced_records_after": len(enhanced_retained) if enhanced_retained is not None else None,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
