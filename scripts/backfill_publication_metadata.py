#!/usr/bin/env python3
"""Backfill arXiv publication metadata into historical JSONL and Markdown."""

from __future__ import annotations

import argparse
import json
import os
import re
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import requests


ATOM = "{http://www.w3.org/2005/Atom}"
ARXIV = "{http://arxiv.org/schemas/atom}"
DATE_MARKDOWN = re.compile(r"\d{4}-\d{2}-\d{2}\.md$")
PAPER_START = re.compile(
    r"(?m)^### \[\d+\] \[[^\n]+\]\(https://arxiv\.org/abs/([^\s)]+)\)"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--arxiv-batch-size", type=int, default=100)
    parser.add_argument("--openalex-batch-size", type=int, default=50)
    parser.add_argument("--arxiv-delay", type=float, default=3.0)
    parser.add_argument("--openalex-delay", type=float, default=0.2)
    parser.add_argument("--cache", default="publication-metadata-cache.json")
    return parser.parse_args()


def normalize_id(value: str) -> str:
    value = value.rsplit("/", 1)[-1]
    return re.sub(r"v\d+$", "", value)


def plausible_official_affiliation(value: str) -> bool:
    """Reject obvious author-metadata corruption without guessing replacements."""
    value = value.strip()
    if not value:
        return False
    words = value.split()
    if len(words) > 1:
        return True
    word = words[0]
    return word.isupper() or any(character.isupper() for character in word[1:])


def request_with_retry(
    session: requests.Session, url: str, *, params: dict[str, Any], attempts: int = 5
) -> requests.Response:
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            response = session.get(url, params=params, timeout=90)
            response.raise_for_status()
            return response
        except Exception as error:
            last_error = error
            if attempt + 1 < attempts:
                time.sleep(min(30, 2**attempt))
    raise RuntimeError(f"metadata request failed after {attempts} attempts: {last_error}")


def parse_arxiv_feed(xml: bytes) -> dict[str, dict[str, Any]]:
    root = ET.fromstring(xml)
    output: dict[str, dict[str, Any]] = {}
    for entry in root.findall(f"{ATOM}entry"):
        entry_id = normalize_id(entry.findtext(f"{ATOM}id", ""))
        if not entry_id:
            continue
        affiliations = []
        for author in entry.findall(f"{ATOM}author"):
            name = (author.findtext(f"{ATOM}name", "") or "").strip()
            affiliation = (author.findtext(f"{ARXIV}affiliation", "") or "").strip()
            if plausible_official_affiliation(affiliation):
                affiliations.append({"author": name, "affiliation": affiliation})
        output[entry_id] = {
            "comment": (entry.findtext(f"{ARXIV}comment", "") or "").strip() or None,
            "journal_ref": (entry.findtext(f"{ARXIV}journal_ref", "") or "").strip()
            or None,
            "doi": (entry.findtext(f"{ARXIV}doi", "") or "").strip() or None,
            "author_affiliations": affiliations,
        }
    return output


def fetch_arxiv_metadata(
    ids: list[str], batch_size: int, delay: float
) -> dict[str, dict[str, Any]]:
    session = requests.Session()
    session.headers["User-Agent"] = (
        "daily-arXiv-ai-enhanced/1.0 "
        "(https://github.com/fyrsta7/26summer-daily_arXiv_ai_enhanced)"
    )
    output: dict[str, dict[str, Any]] = {}
    for start in range(0, len(ids), batch_size):
        batch = ids[start : start + batch_size]
        response = request_with_retry(
            session,
            "https://export.arxiv.org/api/query",
            params={"id_list": ",".join(batch), "max_results": len(batch)},
        )
        output.update(parse_arxiv_feed(response.content))
        print(
            f"ARXIV_PROGRESS {min(start + len(batch), len(ids))}/{len(ids)} ",
            f"returned={len(output)}",
            flush=True,
        )
        if start + len(batch) < len(ids):
            time.sleep(delay)
    return output


def parse_openalex_work(work: dict[str, Any]) -> list[dict[str, str]]:
    affiliations = []
    for authorship in work.get("authorships") or []:
        author = str((authorship.get("author") or {}).get("display_name") or "").strip()
        institutions = []
        seen = set()
        for institution in authorship.get("institutions") or []:
            name = str(institution.get("display_name") or "").strip()
            if name and name not in seen:
                seen.add(name)
                institutions.append(name)
        if institutions:
            affiliations.append({"author": author, "affiliation": " / ".join(institutions)})
    return affiliations


def fetch_openalex_affiliations(
    metadata: dict[str, dict[str, Any]], batch_size: int, delay: float
) -> int:
    doi_to_ids: dict[str, list[str]] = {}
    for arxiv_id, item in metadata.items():
        doi = str(item.get("doi") or "").strip().lower()
        if doi and not item.get("author_affiliations"):
            doi_to_ids.setdefault(doi, []).append(arxiv_id)
    dois = sorted(doi_to_ids)
    if not dois:
        return 0

    session = requests.Session()
    session.headers["User-Agent"] = "daily-arXiv-ai-enhanced/1.0"
    api_key = os.environ.get("OPENALEX_API_KEY")
    matched = 0
    for start in range(0, len(dois), batch_size):
        batch = dois[start : start + batch_size]
        params: dict[str, Any] = {
            "filter": "doi:" + "|".join(batch),
            "per-page": len(batch),
            "select": "doi,authorships",
        }
        if api_key:
            params["api_key"] = api_key
        try:
            response = request_with_retry(
                session, "https://api.openalex.org/works", params=params, attempts=3
            )
        except RuntimeError as error:
            print(f"OPENALEX_SKIPPED batch={start // batch_size + 1}: {error}", flush=True)
            continue
        for work in response.json().get("results") or []:
            doi = str(work.get("doi") or "").removeprefix("https://doi.org/").lower()
            affiliations = parse_openalex_work(work)
            if not affiliations:
                continue
            for arxiv_id in doi_to_ids.get(doi, []):
                metadata[arxiv_id]["author_affiliations"] = affiliations
                metadata[arxiv_id]["affiliation_source"] = "openalex_doi"
                matched += 1
        print(
            f"OPENALEX_PROGRESS {min(start + len(batch), len(dois))}/{len(dois)} ",
            f"matched={matched}",
            flush=True,
        )
        if start + len(batch) < len(dois):
            time.sleep(delay)
    return matched


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def merge_metadata(record: dict[str, Any], metadata: dict[str, Any]) -> None:
    for key in ("comment", "journal_ref", "doi"):
        if metadata.get(key) is not None:
            record[key] = metadata[key]
        elif key not in record:
            record[key] = None
    record["author_affiliations"] = metadata.get("author_affiliations") or []
    if metadata.get("affiliation_source"):
        record["affiliation_source"] = metadata["affiliation_source"]


def format_metadata(item: dict[str, Any]) -> str:
    lines = []
    comment = str(item.get("comment") or "").strip()
    if comment:
        lines.append(f"Comments: {comment}")
    journal_ref = str(item.get("journal_ref") or "").strip()
    if journal_ref:
        lines.append(f"Journal reference: {journal_ref}")
    doi = str(item.get("doi") or "").strip()
    if doi:
        lines.append(f"DOI: [{doi}](https://doi.org/{doi})")
    affiliations = []
    for entry in item.get("author_affiliations") or []:
        author = str(entry.get("author") or "").strip()
        affiliation = str(entry.get("affiliation") or "").strip()
        if affiliation:
            affiliations.append(f"{author}: {affiliation}" if author else affiliation)
    if affiliations:
        lines.append("Author affiliations: " + "; ".join(affiliations))
    return "\n\n".join(lines)


def strip_existing_metadata(text: str) -> str:
    return re.sub(
        r"(?m)^(?:Comments|Journal reference|DOI|Author affiliations):[^\n]*\n(?:\n)?",
        "",
        text,
    )


def patch_markdown(markdown: str, by_id: dict[str, dict[str, Any]]) -> str:
    matches = list(PAPER_START.finditer(markdown))
    if not matches:
        return markdown
    output = [markdown[: matches[0].start()]]
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(markdown)
        block = strip_existing_metadata(markdown[match.start() : end])
        arxiv_id = normalize_id(match.group(1))
        metadata = format_metadata(by_id.get(arxiv_id, {}))
        if metadata:
            block = re.sub(
                r"(?m)^(Main category: [^\n]+\n)",
                lambda category: f"{category.group(1)}\n{metadata}\n",
                block,
                count=1,
            )
        output.append(block)
    return "".join(output)


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir).resolve()
    markdown_paths = sorted(path for path in data_dir.glob("*.md") if DATE_MARKDOWN.fullmatch(path.name))
    ids = sorted(
        {
            normalize_id(match.group(1))
            for path in markdown_paths
            for match in PAPER_START.finditer(path.read_text(encoding="utf-8"))
        }
    )
    print(f"Selected {len(markdown_paths)} dates and {len(ids)} unique arXiv papers", flush=True)

    cache_path = data_dir / args.cache
    cached = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
    missing = [arxiv_id for arxiv_id in ids if arxiv_id not in cached]
    if missing:
        cached.update(fetch_arxiv_metadata(missing, args.arxiv_batch_size, args.arxiv_delay))
        cache_path.write_text(json.dumps(cached, ensure_ascii=False), encoding="utf-8")
    metadata = {arxiv_id: cached.get(arxiv_id, {}) for arxiv_id in ids}
    for item in metadata.values():
        if item.get("affiliation_source") == "openalex_doi":
            continue
        item["author_affiliations"] = [
            entry
            for entry in item.get("author_affiliations") or []
            if plausible_official_affiliation(str(entry.get("affiliation") or ""))
        ]
    openalex_matches = fetch_openalex_affiliations(
        metadata, args.openalex_batch_size, args.openalex_delay
    )
    cache_path.write_text(json.dumps(cached, ensure_ascii=False), encoding="utf-8")

    updated_jsonl = 0
    for path in markdown_paths:
        date = path.stem
        for jsonl_path in (data_dir / f"{date}.jsonl", data_dir / f"{date}_AI_enhanced_Chinese.jsonl"):
            records = load_jsonl(jsonl_path)
            for record in records:
                arxiv_id = normalize_id(str(record.get("id") or ""))
                if arxiv_id in metadata:
                    merge_metadata(record, metadata[arxiv_id])
            write_jsonl(jsonl_path, records)
            updated_jsonl += 1

        enhanced = {
            normalize_id(str(record.get("id") or "")): record
            for record in load_jsonl(data_dir / f"{date}_AI_enhanced_Chinese.jsonl")
        }
        path.write_text(
            patch_markdown(path.read_text(encoding="utf-8"), enhanced),
            encoding="utf-8",
        )

    cache_path.unlink(missing_ok=True)
    with_affiliations = sum(bool(item.get("author_affiliations")) for item in metadata.values())
    with_journal = sum(bool(item.get("journal_ref")) for item in metadata.values())
    print(
        f"BACKFILL_DONE dates={len(markdown_paths)} jsonl={updated_jsonl} "
        f"journal={with_journal} affiliations={with_affiliations} "
        f"openalex_affiliations={openalex_matches}",
        flush=True,
    )


if __name__ == "__main__":
    main()
