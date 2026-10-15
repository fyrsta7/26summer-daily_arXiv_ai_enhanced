#!/usr/bin/env python3
"""Replay a failed daily crawl from its log, preserving the original digest date."""

import argparse
import ast
import json
import os
import re
import subprocess
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo


def recover_records(log, target_date):
    expected = re.findall(
        rf"Stored jsonl feed \((\d+) items\) in: ../data/{re.escape(target_date)}\.jsonl",
        log,
    )
    if len(expected) != 1:
        raise ValueError("Expected exactly one completed crawl for the target date")
    records = []
    for line in log.splitlines():
        payload = line.partition(" ")[2]
        if payload.startswith("{'id':"):
            record = ast.literal_eval(payload)
            if not isinstance(record, dict) or not all(
                key in record for key in ("id", "title", "summary", "categories")
            ):
                raise ValueError("Invalid paper record in crawl log")
            records.append(record)
    if len(records) != int(expected[0]):
        raise ValueError(f"Incomplete crawl log: recovered {len(records)}, expected {expected[0]}")
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True, type=int)
    parser.add_argument("--date", required=True)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--history-ref", required=True)
    args = parser.parse_args()
    target = date.fromisoformat(args.date)
    repo = os.environ["GITHUB_REPOSITORY"]

    def api(path):
        return subprocess.check_output(["gh", "api", f"repos/{repo}/{path}"], text=True)

    run = json.loads(api(f"actions/runs/{args.run_id}"))
    original_day = datetime.fromisoformat(run["created_at"].replace("Z", "+00:00")).astimezone(
        ZoneInfo("Asia/Shanghai")
    ).date()
    if original_day != target or run["name"] != "arXiv-daily-email" or run["conclusion"] != "failure":
        raise ValueError("Source run must be a failed daily digest for the requested date")
    jobs = json.loads(api(f"actions/runs/{args.run_id}/attempts/1/jobs"))["jobs"]
    build = next(job for job in jobs if job["name"] == "build")
    if any("email" in step["name"].lower() and step["conclusion"] == "success" for step in build["steps"]):
        raise ValueError("Source run already sent email; refusing duplicate delivery")
    records = recover_records(api(f"actions/jobs/{build['id']}/logs"), args.date)

    history_ids = set()
    for offset in range(1, 8):
        historical_date = (target - timedelta(days=offset)).isoformat()
        revision_path = f"{args.history_ref}:data/{historical_date}.jsonl"
        exists = subprocess.run(["git", "cat-file", "-e", revision_path], capture_output=True)
        if exists.returncode:
            continue
        content = subprocess.check_output(["git", "show", revision_path], text=True)
        history_ids.update(json.loads(line)["id"] for line in content.splitlines() if line.strip())
    kept = [item for item in records if item["id"] not in history_ids]
    args.data_dir.mkdir(parents=True, exist_ok=True)
    output = args.data_dir / f"{args.date}.jsonl"
    output.write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in kept), encoding="utf-8")
    print(json.dumps({"source_run_id": args.run_id, "date": args.date, "recovered_records": len(records),
                      "history_removed": len(records) - len(kept), "remaining_records": len(kept)}))


if __name__ == "__main__":
    main()
