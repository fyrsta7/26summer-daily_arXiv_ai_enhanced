#!/usr/bin/env python3
"""Fetch broad arXiv candidates for the automatic-code-optimization filter."""

from __future__ import annotations

import argparse
import json
import re
import time
import xml.etree.ElementTree as ET
from datetime import date
from pathlib import Path

import requests


API_URL = "https://export.arxiv.org/api/query"
ATOM = "{http://www.w3.org/2005/Atom}"
ARXIV = "{http://arxiv.org/schemas/atom}"
OPENSEARCH = "{http://a9.com/-/spec/opensearch/1.1/}"
USER_AGENT = "semopt-auto-code-optimization-review/1.0"

# The daily feed covers cs.AI/cs.SE.  These deliberately overlapping queries
# add compiler, performance-engineering, and accelerator-code terminology from
# the rest of arXiv; the strict downstream reviewer removes false positives.
QUERIES = {
    "systems_optimization": (
        "(cat:cs.PL OR cat:cs.SE OR cat:cs.DC OR cat:cs.PF OR cat:cs.AR OR cat:cs.MS) "
        "AND (all:optimization OR all:optimizing OR all:autotuning OR all:superoptimization) "
        "AND (all:code OR all:program OR all:compiler OR all:kernel OR all:software "
        "OR all:LLVM OR all:CUDA OR all:Triton)"
    ),
    "code_performance": (
        "(all:\"code performance\" OR all:\"program performance\" "
        "OR all:\"software performance optimization\" OR all:\"performance-improving code\")"
    ),
    "accelerator_kernels": (
        "(all:\"GPU kernel\" OR all:\"NPU kernel\" OR all:\"tensor program\") "
        "AND (all:generation OR all:optimization OR all:autotuning OR all:agent OR all:LLM)"
    ),
    "compiler_techniques": (
        "(all:\"compiler pass\" OR all:\"phase ordering\" OR all:vectorization "
        "OR all:inlining OR all:\"loop optimization\" OR all:superoptimization) "
        "AND (all:automatic OR all:learning OR all:search OR all:agent OR all:LLM)"
    ),
    "language_model_optimization": (
        "(all:\"coding agent\" OR all:\"LLM agent\" OR all:\"language model\") "
        "AND (all:performance OR all:speedup OR all:runtime OR all:latency) "
        "AND (all:code OR all:program OR all:compiler OR all:kernel)"
    ),
    "profiling": (
        "(all:profiling OR all:hotspot) AND (all:code OR all:program OR all:compiler) "
        "AND (all:automatic OR all:optimization OR all:agent OR all:LLM)"
    ),
    "benchmarks": (
        "(all:benchmark OR all:dataset) AND (all:\"code optimization\" "
        "OR all:\"software performance\" OR all:\"compiler optimization\" "
        "OR all:\"kernel optimization\")"
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--page-size", type=int, default=200)
    parser.add_argument("--request-delay", type=float, default=3.0)
    return parser.parse_args()


def text(node: ET.Element, name: str) -> str:
    return " ".join((node.findtext(name) or "").split())


def parse_feed(xml: bytes, query_name: str) -> tuple[list[dict], int]:
    root = ET.fromstring(xml)
    total = int(root.findtext(f"{OPENSEARCH}totalResults") or 0)
    papers = []
    for entry in root.findall(f"{ATOM}entry"):
        abs_url = text(entry, f"{ATOM}id")
        match = re.search(r"/abs/([^v]+)(?:v\d+)?$", abs_url)
        if not match:
            continue
        paper_id = match.group(1)
        links = {
            link.attrib.get("title"): link.attrib.get("href", "")
            for link in entry.findall(f"{ATOM}link")
        }
        papers.append(
            {
                "id": paper_id,
                "abs": abs_url,
                "pdf": links.get("pdf", f"https://arxiv.org/pdf/{paper_id}"),
                "title": text(entry, f"{ATOM}title"),
                "summary": text(entry, f"{ATOM}summary"),
                "authors": [text(author, f"{ATOM}name") for author in entry.findall(f"{ATOM}author")],
                "categories": [node.attrib.get("term", "") for node in entry.findall(f"{ATOM}category")],
                "comment": text(entry, f"{ARXIV}comment") or None,
                "published": text(entry, f"{ATOM}published"),
                "updated": text(entry, f"{ATOM}updated"),
                "candidate_queries": [query_name],
            }
        )
    return papers, total


def get_page(session: requests.Session, search_query: str, start: int, page_size: int) -> bytes:
    params = {
        "search_query": search_query,
        "start": start,
        "max_results": page_size,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }
    last_error: Exception | None = None
    for attempt in range(5):
        try:
            response = session.get(API_URL, params=params, timeout=90)
            response.raise_for_status()
            return response.content
        except Exception as error:
            last_error = error
            if attempt < 4:
                time.sleep(2 ** attempt)
    raise RuntimeError(f"arXiv API request failed after retries: {last_error}")


def main() -> None:
    args = parse_args()
    start = date.fromisoformat(args.start_date)
    end = date.fromisoformat(args.end_date)
    if end < start:
        raise ValueError("end-date must not precede start-date")
    if not 1 <= args.page_size <= 2000:
        raise ValueError("page-size must be between 1 and 2000")
    date_clause = f"submittedDate:[{start:%Y%m%d}0000 TO {end:%Y%m%d}2359]"
    by_id: dict[str, dict] = {}
    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT
    request_count = 0
    for query_name, query in QUERIES.items():
        search_query = f"{date_clause} AND ({query})"
        offset = 0
        query_total = None
        while query_total is None or offset < query_total:
            if request_count:
                time.sleep(args.request_delay)
            payload = get_page(session, search_query, offset, args.page_size)
            request_count += 1
            papers, query_total = parse_feed(payload, query_name)
            for paper in papers:
                existing = by_id.get(paper["id"])
                if existing:
                    existing["candidate_queries"] = list(
                        dict.fromkeys(existing["candidate_queries"] + paper["candidate_queries"])
                    )
                else:
                    published_date = paper.get("published", "")[:10]
                    paper["source_dates"] = [published_date] if published_date else []
                    by_id[paper["id"]] = paper
            print(
                json.dumps(
                    {
                        "query": query_name,
                        "total": query_total,
                        "received": len(papers),
                        "unique_so_far": len(by_id),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            if not papers:
                break
            offset += len(papers)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in by_id.values()),
        encoding="utf-8",
    )
    print(json.dumps({"output": str(output), "unique_candidates": len(by_id), "requests": request_count}))


if __name__ == "__main__":
    main()
