#!/usr/bin/env python3
"""Apply existing relevance decisions to history; no LLM/API calls.

By default only reports the proposed change. --apply removes explicitly dated,
tracked files before --start-date and synchronizes the retained daily artifacts.
Deleted data remains recoverable from the recorded Git parent commit.
"""

import argparse
from collections import Counter
from datetime import date
import json
import os
from pathlib import Path
import re
import subprocess
import sys

from prune_se_delivery_papers import keep_for_delivery, read_jsonl, sync_enhanced_records


ROOT = Path(__file__).resolve().parents[1]
DATED_FILE = re.compile(r"^(\d{4}-\d{2}-\d{2})(?:[_.])")


def prepare(root: Path, start_date: str, audit_source: str | None = None):
    """Validate every retained day before changing any repository file."""
    date.fromisoformat(start_date)
    data = root / "data"
    tracked = set(subprocess.check_output(
        ["git", "ls-files", "data"], cwd=root, text=True
    ).splitlines())
    deleted = []
    for path in sorted(data.iterdir()):
        match = DATED_FILE.match(path.name)
        if not path.is_file() or not match or match[1] >= start_date:
            continue
        if path.is_symlink() or str(path.relative_to(root)) not in tracked:
            raise ValueError(f"Refusing to delete untracked or linked file: {path}")
        deleted.append(path)

    updates = {}
    days = []
    report_days = []
    counts = Counter()
    def before_records(path, current):
        if audit_source is None:
            return current
        original = subprocess.check_output(
            ["git", "show", f"{audit_source}:{path.relative_to(root)}"], cwd=root, text=True
        )
        return [json.loads(line) for line in original.splitlines() if line.strip()]

    for raw_path in sorted(data.glob("????-??-??.jsonl")):
        day = raw_path.stem
        if day < start_date:
            continue
        records = read_jsonl(raw_path)
        ids = [str(record["id"]) for record in records]
        if len(ids) != len(set(ids)):
            raise ValueError(f"Duplicate IDs in {raw_path}")
        for record in records:
            if "cs.SE" not in record.get("categories", []):
                if record.get("selection", {}).get("policy") != "cs.AI_semopt_relevance":
                    raise ValueError(f"Missing AI relevance decision: {day} {record['id']}")
            elif type(record.get("se_selection", {}).get("relevant")) is not bool:
                raise ValueError(f"Missing SE relevance decision: {day} {record['id']}")
        retained = [record for record in records if keep_for_delivery(record)]
        original = before_records(raw_path, records)
        if {str(record['id']) for record in original if keep_for_delivery(record)} != {str(record['id']) for record in retained}:
            raise ValueError(f"Audit source has different selection results: {day}")
        removed = [record for record in original if not keep_for_delivery(record)]
        updates[raw_path] = retained
        enhanced_paths = sorted(data.glob(f"{day}_AI_enhanced_*.jsonl"))
        if not enhanced_paths:
            raise ValueError(f"Missing enhanced data for {day}")
        for enhanced_path in enhanced_paths:
            enhanced = read_jsonl(enhanced_path)
            if {str(record["id"]) for record in enhanced} != set(ids):
                raise ValueError(f"Raw/enhanced ID mismatch: {enhanced_path}")
            selected = sync_enhanced_records(enhanced, retained)
            for record in selected:
                required = {"abstract_zh", "tldr", "motivation", "method", "result", "conclusion"}
                if not required.issubset(record.get("AI") or {}):
                    raise ValueError(f"Incomplete AI content: {enhanced_path} {record['id']}")
            counts["enhanced_before"] += len(before_records(enhanced_path, enhanced))
            counts["enhanced_after"] += len(selected)
            updates[enhanced_path] = selected
        counts["raw_before"] += len(original)
        counts["raw_after"] += len(retained)
        counts["removed_se_occurrences"] += len(removed)
        days.append((day, enhanced_paths))
        report_days.append({
            "date": day, "before": len(original), "after": len(retained),
            "removed": [{"id": record["id"], "decision": record["se_selection"]} for record in removed],
        })
    if not days:
        raise ValueError("No retained daily data found")
    report = {
        "start_date": start_date,
        "source_commit": subprocess.check_output(["git", "rev-parse", audit_source or "HEAD"], cwd=root, text=True).strip(),
        "deleted_files": [str(path.relative_to(root)) for path in deleted],
        "deleted_bytes": sum(path.stat().st_size for path in deleted),
        "retained_dates": len(days), "counts": dict(counts), "days": report_days,
    }
    return deleted, updates, days, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--audit-source", help="Git commit for before counts when resuming an interrupted sync")
    args = parser.parse_args()
    deleted, updates, days, report = prepare(ROOT, args.start_date, args.audit_source)
    print(json.dumps({
        "apply": args.apply, "deleted_files": len(deleted),
        "deleted_bytes": report["deleted_bytes"], "retained_dates": len(days),
        **report["counts"],
    }))
    if not args.apply:
        return
    # Record the full plan before mutation so an interrupted run remains auditable.
    (ROOT / "data" / "history_selection_sync_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    for path, records in updates.items():
        text = "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records)
        if path.read_text(encoding="utf-8") != text:
            path.write_text(text, encoding="utf-8")
    for day, enhanced_paths in days:
        enhanced = next((path for path in enhanced_paths if "_Chinese." in path.name), enhanced_paths[0])
        # Relative data path avoids underscores in repository directory names.
        subprocess.run(
            [sys.executable, "convert.py", "--data", f"../data/{enhanced.name}"],
            cwd=ROOT / "to_md", env={**os.environ, "CATEGORIES": "cs.AI,cs.SE"},
            check=True, capture_output=True, text=True,
        )
        normalize = lambda value: re.sub(r"v\d+$", "", value)
        expected = {normalize(str(record["id"])) for record in updates[enhanced]}
        markdown = (ROOT / "data" / f"{day}.md").read_text(encoding="utf-8")
        actual = {normalize(value) for value in re.findall(r"^### \[\d+\].*?https?://arxiv\.org/abs/([^\s)]+)", markdown, re.M)}
        if actual != expected:
            raise ValueError(f"Rendered Markdown does not match selected papers: {day}")
    # All retained artifacts have been rebuilt and checked before deletion.
    for path in deleted:
        path.unlink()
    file_list = sorted(path.name for path in (ROOT / "data").glob("*.jsonl"))
    (ROOT / "assets" / "file-list.txt").write_text("\n".join(file_list) + "\n", encoding="utf-8")
    print("Synchronized raw JSONL, enhanced JSONL, Markdown, and date index.")


if __name__ == "__main__":
    main()
