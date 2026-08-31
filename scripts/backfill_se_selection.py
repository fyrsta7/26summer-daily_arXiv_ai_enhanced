#!/usr/bin/env python3
"""Classify cs.SE papers for SemOpt relevance and append a selected list to digests.

The classifier operates on unique arXiv IDs, then writes each decision back to
every daily occurrence.  This keeps a historical backfill inexpensive while
preserving the complete, unfiltered cs.SE section already present in a digest.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import requests


SE_SELECTION_HEADING = "SE 精选（SemOpt / Coding Agent 相关）"
MARKER_START = "<!-- semopt-se-selection:start -->"
MARKER_END = "<!-- semopt-se-selection:end -->"
DATE_FILE = re.compile(r"(\d{4}-\d{2}-\d{2})\.jsonl$")

SYSTEM_PROMPT = """你是 LLM Agent 与自动代码优化方向的论文筛选专家。请判断一篇 cs.SE
论文是否值得加入 SemOpt / Coding Agent 的每日精选。筛选是宽松的：有明确迁移价值即可。

保留条件（满足任一即可）：
1. 自动程序性能优化、编译器/运行时优化、profiling、调优、benchmark、正确性或 speedup 验证；
2. coding agent 或软件工程 agent：代码生成、修复、测试、审查、检索、规划、工具反馈、验证、长程执行；
3. 面向程序分析、调试、测试、代码检索、软件工程评测或工程化 agent 的通用方法、数据集或 benchmark；
4. 虽不直接研究 SemOpt，但提出可迁移到代码 agent 的可靠性、记忆、反思、工作流、多 agent 协作或评测机制。

排除：纯业务系统实现、一般项目管理/教育/人因研究、没有可迁移方法贡献的单一垂直应用、
与代码或 agent 无关的优化、以及仅因摘要出现 AI/agent/optimization 词汇而主题无关的论文。

score 表示对用户研究的价值：0.9-1.0 直接相关，0.7-0.89 明显可迁移，0.5-0.69 有参考价值。
relevant=true 通常要求 score>=0.5。只输出有效 JSON，不要 Markdown：
{"decisions":[{"id":"arxiv id","relevant":true,"score":0.0,"reason_zh":"一句具体理由"}]}
必须对输入中的每个 id 恰好输出一次。"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--start-date")
    parser.add_argument("--end-date")
    parser.add_argument("--workers", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--report", default="data/se_selection_backfill_report.json")
    parser.add_argument("--render-only", action="store_true")
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--only-with-markdown", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def selected_data_paths(data_dir: Path, start_date: str | None, end_date: str | None) -> list[Path]:
    paths: list[Path] = []
    for path in sorted(data_dir.glob("????-??-??.jsonl")):
        match = DATE_FILE.search(path.name)
        if not match:
            continue
        date = match.group(1)
        if start_date and date < start_date:
            continue
        if end_date and date > end_date:
            continue
        paths.append(path)
    return paths


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def is_se(item: dict[str, Any]) -> bool:
    return "cs.SE" in set(item.get("categories") or [])


def normalize_decision(value: dict[str, Any]) -> dict[str, Any]:
    score = max(0.0, min(1.0, float(value.get("score", 0.0))))
    return {
        "policy": "cs.SE_semopt_relevance",
        "relevant": value.get("relevant") is True or str(value.get("relevant")).lower() == "true",
        "score": score,
        "reason_zh": str(value.get("reason_zh") or "与 SemOpt / Coding Agent 研究相关。"),
    }


def classify_batch(batch: list[dict[str, Any]], model: str, base_url: str, api_key: str) -> tuple[list[dict[str, Any]], dict[str, int]]:
    papers = [
        {
            "id": item["id"],
            "title": item.get("title", ""),
            "abstract": str(item.get("summary") or "")[:2400],
            "categories": item.get("categories", []),
        }
        for item in batch
    ]
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps({"papers": papers}, ensure_ascii=False)},
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0,
    }
    expected = {item["id"] for item in papers}
    last_error: Exception | None = None
    for attempt in range(6):
        try:
            response = requests.post(
                base_url.rstrip("/") + "/chat/completions",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json=payload,
                timeout=180,
            )
            response.raise_for_status()
            body = response.json()
            decisions = json.loads(body["choices"][0]["message"]["content"])["decisions"]
            received = [str(decision.get("id")) for decision in decisions]
            if set(received) != expected or len(received) != len(set(received)):
                raise ValueError(f"invalid decision IDs: expected={expected}, received={received}")
            usage = {
                key: int(body.get("usage", {}).get(key, 0) or 0)
                for key in ("prompt_tokens", "completion_tokens", "total_tokens")
            }
            return decisions, usage
        except Exception as error:
            last_error = error
            if attempt == 5:
                break
            time.sleep(min(30, 2**attempt) + random.random())
    assert last_error is not None
    raise RuntimeError(f"batch classification failed after 6 attempts: {last_error}")


def classify_unique(items: list[dict[str, Any]], workers: int, batch_size: int) -> tuple[dict[str, dict[str, Any]], dict[str, int]]:
    if not 1 <= workers <= 128:
        raise ValueError("--workers must be between 1 and 128")
    if not 1 <= batch_size <= 50:
        raise ValueError("--batch-size must be between 1 and 50")
    if not items:
        return {}, {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    model = os.environ["MODEL_NAME"]
    base_url = os.environ["OPENAI_BASE_URL"]
    api_key = os.environ["OPENAI_API_KEY"]
    batches = [items[index:index + batch_size] for index in range(0, len(items), batch_size)]
    decisions: dict[str, dict[str, Any]] = {}
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    with ThreadPoolExecutor(max_workers=min(workers, len(batches))) as executor:
        futures = {
            executor.submit(classify_batch, batch, model, base_url, api_key): batch
            for batch in batches
        }
        for index, future in enumerate(as_completed(futures), 1):
            result, batch_usage = future.result()
            decisions.update({str(value["id"]): normalize_decision(value) for value in result})
            for key in usage:
                usage[key] += batch_usage[key]
            print(f"CLASSIFY_PROGRESS {index}/{len(batches)}", flush=True)
    return decisions, usage


def selection_markdown(items: list[dict[str, Any]]) -> str:
    selected = [item for item in items if is_se(item) and item.get("se_selection", {}).get("relevant")]
    selected.sort(key=lambda item: (-float(item["se_selection"].get("score", 0)), item.get("title", "")))
    lines = [MARKER_START, f"## {SE_SELECTION_HEADING}", "", "以下为从原始 cs.SE 列表中筛出的 SemOpt / Coding Agent 相关论文。", ""]
    if not selected:
        lines.append("本日没有筛选出与 SemOpt / Coding Agent 明显相关的 cs.SE 论文。")
    else:
        for item in selected:
            decision = item["se_selection"]
            lines.append(
                f"- [{item.get('title', 'Untitled')}]({item.get('abs', '')}) — "
                f"相关度 {float(decision.get('score', 0)):.2f}：{decision.get('reason_zh', '')}"
            )
    lines.extend(["", MARKER_END, ""])
    return "\n".join(lines)


def replace_selection_section(markdown: str, section: str) -> str:
    start = markdown.find(MARKER_START)
    if start >= 0:
        end = markdown.find(MARKER_END, start)
        if end < 0:
            raise ValueError("found an unterminated SE selection section")
        markdown = markdown[:start].rstrip() + "\n"
    return markdown.rstrip() + "\n\n" + section


def render_markdown(path: Path, items: list[dict[str, Any]]) -> bool:
    markdown_path = path.with_suffix(".md")
    if not markdown_path.exists():
        return False
    old = markdown_path.read_text(encoding="utf-8")
    new = replace_selection_section(old, selection_markdown(items))
    if new == old:
        return False
    markdown_path.write_text(new, encoding="utf-8")
    return True


def main() -> None:
    args = parse_args()
    if args.render_only and args.no_render:
        raise ValueError("--render-only and --no-render cannot be used together")
    data_dir = Path(args.data_dir)
    paths = selected_data_paths(data_dir, args.start_date, args.end_date)
    if args.only_with_markdown:
        paths = [path for path in paths if path.with_suffix(".md").exists()]
    if not paths:
        raise RuntimeError("no date-named JSONL files matched the requested range")
    loaded = {path: load_jsonl(path) for path in paths}
    unique: dict[str, dict[str, Any]] = {}
    occurrences = 0
    for records in loaded.values():
        for item in records:
            if is_se(item):
                occurrences += 1
                unique.setdefault(str(item["id"]), item)
    print(f"Found {occurrences} cs.SE occurrences / {len(unique)} unique arXiv IDs in {len(paths)} digests")
    if args.dry_run:
        return

    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    if not args.render_only:
        decisions, usage = classify_unique(list(unique.values()), args.workers, args.batch_size)
        for path, records in loaded.items():
            changed = False
            for item in records:
                if is_se(item):
                    decision = decisions[str(item["id"])]
                    if item.get("se_selection") != decision:
                        item["se_selection"] = decision
                        changed = True
            if changed:
                path.write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in records), encoding="utf-8")

        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(
                {
                    "policy": "cs.SE_semopt_relevance",
                    "model": os.environ["MODEL_NAME"],
                    "documents": len(paths),
                    "cs.SE_occurrences": occurrences,
                    "unique_arxiv_ids": len(unique),
                    "selected_unique": sum(value["relevant"] for value in decisions.values()),
                    "usage": usage,
                },
                ensure_ascii=False,
                indent=2,
            ) + "\n",
            encoding="utf-8",
        )
        print(f"Selected {sum(value['relevant'] for value in decisions.values())}/{len(unique)} unique cs.SE papers")

    if not args.no_render:
        rendered = sum(render_markdown(path, records) for path, records in loaded.items())
        print(f"Rendered SE selection sections into {rendered}/{len(paths)} Markdown digests")


if __name__ == "__main__":
    main()
