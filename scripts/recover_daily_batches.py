#!/usr/bin/env python3
"""Replay verified crawler batches, serially, without email or other delivery."""
from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
import lzma
import math
import os
from pathlib import Path
import re
import subprocess
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
FIELDS = ('abstract_zh', 'tldr', 'motivation', 'method', 'result', 'conclusion')
BAD_TEXT = ('translation failed', 'generation failed', 'extraction failed',
            'analysis unavailable', 'processing failed', 'compliance test',
            'has been hidden', '未通过合规', '内容已隐藏')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temp.replace(path)


def canonical_hash(value):
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()
    return hashlib.sha256(raw).hexdigest()


def parse_crawler_log(content):
    records = []
    for line in content.splitlines():
        line = re.sub(r'^\ufeff?\d{4}-\d\d-\d\dT\S+\s+', '', line)
        if line.startswith("{'id':"):
            item = ast.literal_eval(line)
            if not isinstance(item, dict):
                raise ValueError('Invalid original crawler record')
            records.append(item)
    return records


def verify_records(records, spec):
    if len(records) != spec['raw_records'] or len({p['id'] for p in records}) != spec['unique_ids']:
        raise ValueError(f"Original candidate count mismatch: {spec['date']}")
    if canonical_hash(records) != spec['records_sha256']:
        raise ValueError(f"Original candidate hash mismatch: {spec['date']}")
    by_id = {}
    for item in records:
        if not all(item.get(key) for key in ('id', 'title', 'summary', 'authors', 'categories', 'abs', 'pdf')):
            raise ValueError('Incomplete original paper')
        if item['id'] in by_id and item != by_id[item['id']]:
            raise ValueError('Conflicting duplicate original paper')
        by_id[item['id']] = item


def load_sources(manifest, cache):
    sources = {}
    archive = None
    if manifest.get('source_file'):
        raw = (ROOT / manifest['source_file']).read_bytes()
        if hashlib.sha256(raw).hexdigest() != manifest['source_sha256']:
            raise ValueError('Verified source archive hash mismatch')
        archive = json.loads(lzma.decompress(raw))
    for spec in manifest['dates']:
        path = cache / ('source-' + spec['date'] + '.json')
        if archive is not None:
            records = archive[spec['date']]
        elif path.exists():
            records = read_json(path)
        else:
            # gh authenticates only to GitHub; raw logs and credentials are never printed.
            endpoint = f"repos/{os.environ['GITHUB_REPOSITORY']}/actions/jobs/{spec['job_id']}/logs"
            result = subprocess.run(['gh', 'api', endpoint], capture_output=True, text=True)
            if result.returncode:
                raise RuntimeError(f"Unable to read original job {spec['job_id']}")
            records = parse_crawler_log(result.stdout)
        verify_records(records, spec)
        write_json(path, records)
        sources[spec['date']] = records
    return sources


def assign_unique(sources, published_ids=()):
    seen = set(published_ids)
    result = {}
    for date, records in sorted(sources.items()):
        result[date] = []
        for item in records:
            if item['id'] not in seen:
                result[date].append(copy.deepcopy(item))
                seen.add(item['id'])
    return result


def constant_prompt(path):
    for node in ast.parse(Path(path).read_text(encoding='utf-8')).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'SYSTEM_PROMPT' for t in node.targets):
            return ast.literal_eval(node.value)
    raise ValueError('Missing relevance system prompt')


class SerialModel:
    def __init__(self, model):
        self.model = model
        self.endpoint = os.environ['OPENAI_BASE_URL'].rstrip('/') + '/chat/completions'
        if not self.endpoint.startswith('https://'):
            raise ValueError('Model endpoint must use HTTPS')
        self.key = os.environ['OPENAI_API_KEY']
        self.next_request = 0.0

    def call(self, system, user, validate):
        payload = {'model': self.model, 'messages': [
            {'role': 'system', 'content': system}, {'role': 'user', 'content': user}],
            'response_format': {'type': 'json_object'}, 'temperature': 0}
        for attempt in range(8):
            time.sleep(max(0, self.next_request - time.monotonic()))
            retry_after = 0
            try:
                request = Request(self.endpoint, data=json.dumps(payload).encode(), headers={
                    'Authorization': 'Bearer ' + self.key, 'Content-Type': 'application/json'})
                with urlopen(request, timeout=180) as response:
                    body = json.load(response)
                self.next_request = time.monotonic() + 1
                value = json.loads(body['choices'][0]['message']['content'])
                validate(value)
                return value
            except HTTPError as error:
                status = error.code
                if status not in (408, 409, 429) and status < 500:
                    raise RuntimeError(f'Model HTTP {status}; stopping without publishing') from None
                try:
                    retry_after = float(error.headers.get('Retry-After', '0'))
                except ValueError:
                    pass
                reason = f'HTTP {status}'
            except (URLError, TimeoutError, OSError):
                reason = 'temporary network error'
            except (ValueError, KeyError, TypeError, IndexError):
                reason = 'incomplete or invalid structured response'
            if attempt == 7:
                raise RuntimeError(f'Model failed after 8 serial attempts: {reason}') from None
            delay = max(min(120, 5 * 2**attempt), min(600, retry_after))
            self.next_request = time.monotonic() + delay
            print(f'Retrying {reason}; wait {delay:.0f}s (attempt {attempt + 2}/8)', flush=True)
        raise AssertionError('unreachable')


def validate_decisions(value, expected):
    rows = value['decisions']
    ids = [row['id'] for row in rows]
    if len(ids) != len(set(ids)) or set(ids) != set(expected):
        raise ValueError('Classification must cover every input exactly once')
    for row in rows:
        if type(row.get('relevant')) is not bool:
            raise ValueError('Relevance must be boolean')
        score = row['score']
        if not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError('Invalid relevance score')
        if not isinstance(row.get('reason_zh'), str) or len(row['reason_zh'].strip()) < 3:
            raise ValueError('Missing relevance reason')


def classify(items, kind, client, cache):
    path = cache / f'{kind}-decisions.json'
    decisions = read_json(path) if path.exists() else {}
    prompt = constant_prompt(ROOT / ('scripts/backfill_se_selection.py' if kind == 'se' else 'ai/filter_relevance.py'))
    for item in items:
        if item['id'] in decisions:
            validate_decisions({'decisions': [decisions[item['id']]]}, [item['id']])
    pending = [item for item in items if item['id'] not in decisions]
    for start in range(0, len(pending), 10):
        batch = pending[start:start + 10]
        papers = [{'id': p['id'], 'title': p['title'], 'abstract': p['summary'][:2400], 'categories': p['categories']} for p in batch]
        expected = {p['id'] for p in batch}
        result = client.call(prompt, json.dumps({'papers': papers}, ensure_ascii=False), lambda value: validate_decisions(value, expected))
        decisions.update({row['id']: row for row in result['decisions']})
        write_json(path, decisions)
        print(f'{kind.upper()} classified {min(start + 10, len(pending))}/{len(pending)} pending papers', flush=True)
    return decisions


def validate_ai(ai, paper):
    if not isinstance(ai, dict):
        raise ValueError('AI output must be an object')
    for key in FIELDS:
        text = ai.get(key)
        if not isinstance(text, str) or not text.strip() or any(bad in text.lower() for bad in BAD_TEXT):
            raise ValueError('Missing or placeholder analysis')
    abstract = ai['abstract_zh']
    if len(re.findall('[\u4e00-\u9fff]', abstract)) < 20 or len(abstract) < len(paper['summary']) * 0.12:
        raise ValueError('Incomplete Chinese abstract')


def enhance(items, client, cache):
    system = (ROOT / 'ai/system.txt').read_text(encoding='utf-8').replace('{language}', 'Chinese')
    template = (ROOT / 'ai/template.txt').read_text(encoding='utf-8')
    enhanced = []
    for index, item in enumerate(items, 1):
        path = cache / 'enhanced' / (item['id'] + '.json')
        if path.exists():
            ai = read_json(path)
            validate_ai(ai, item)
        else:
            ai = client.call(system, template.replace('{content}', item['summary']), lambda value: validate_ai(value, item))
            write_json(path, ai)
        enhanced.append({**item, 'AI': ai})
        print(f'Enhanced {index}/{len(items)}: {item["id"]}', flush=True)
    return enhanced


def load_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]


def write_jsonl(path, items):
    Path(path).write_text(''.join(json.dumps(p, ensure_ascii=False) + '\n' for p in items), encoding='utf-8')


def verify_selection_report(report, candidates):
    decisions = {}
    for key, is_se in (('cs.SE_decisions', True), ('cs.AI_decisions', False)):
        expected = [p['id'] for p in candidates if ('cs.SE' in p['categories']) == is_se]
        rows = report[key]
        validate_decisions({'decisions': rows}, expected)
        decisions.update({row['id']: row for row in rows})
    selected = [p['id'] for p in candidates if decisions[p['id']]['relevant']]
    if selected != report['recovery']['selected_ids'] or len(selected) != report['output_total']:
        raise ValueError('Selection report disagrees with candidate decisions')
    return selected


def validate_output(date, data_dir, expected_ids):
    raw = load_jsonl(data_dir / f'{date}.jsonl')
    enhanced = load_jsonl(data_dir / f'{date}_AI_enhanced_Chinese.jsonl')
    ids = [p['id'] for p in raw]
    if len(ids) != len(set(ids)) or set(ids) != set(expected_ids) or [p['id'] for p in enhanced] != ids:
        raise ValueError(f'Output coverage mismatch: {date}')
    markdown = (data_dir / f'{date}.md').read_text(encoding='utf-8')
    if not markdown.strip():
        raise ValueError('Empty Markdown')
    for source, item in zip(raw, enhanced):
        if {k: v for k, v in item.items() if k != 'AI'} != source:
            raise ValueError('Enhanced source metadata changed')
        validate_ai(item['AI'], item)
        if markdown.count(f']({item["abs"]})') != 1:
            raise ValueError('Paper missing from Markdown or rendered twice')
    report = read_json(data_dir / f'{date}_filter_report.json')
    if report['output_total'] != len(ids) or report['recovery']['status'] != 'complete':
        raise ValueError('Incomplete recovery report')


def run(manifest, cache, verify_only=False):
    data_dir = ROOT / 'data'
    data_dir.mkdir(exist_ok=True)
    target_dates = {row['date'] for row in manifest['dates']}
    sources = load_sources(manifest, cache)
    if len({p['id'] for rows in sources.values() for p in rows}) != manifest['expected_unique_candidates']:
        raise ValueError('Candidate union mismatch')
    published = set()
    for path in data_dir.glob('????-??-??_AI_enhanced_*.jsonl'):
        if path.name[:10] not in target_dates:
            published.update(p['id'] for p in load_jsonl(path))
    assignments = assign_unique(sources, published)
    summary = []
    client = None
    for spec in manifest['dates']:
        date = spec['date']
        candidates = assignments[date]
        report_path = data_dir / f'{date}_filter_report.json'
        expected_report = {'source_records_sha256': spec['records_sha256'], 'candidate_ids': [p['id'] for p in candidates]}
        existing = read_json(report_path) if report_path.exists() else {}
        complete = all(existing.get('recovery', {}).get(k) == v for k, v in expected_report.items()) and existing.get('recovery', {}).get('status') == 'complete'
        if complete:
            expected = verify_selection_report(existing, candidates)
            validate_output(date, data_dir, expected)
            originals = {p['id']: p for p in candidates}
            for item in load_jsonl(data_dir / f'{date}.jsonl'):
                if {k: v for k, v in item.items() if k not in ('selection', 'se_selection')} != originals[item['id']]:
                    raise ValueError('Published source metadata differs from original crawler batch')
            summary.append({'date': date, 'candidates': len(candidates), 'selected': len(expected), 'status': 'verified_existing'})
            continue
        if verify_only:
            raise ValueError(f'Recovery not complete: {date}')
        if any((data_dir / f'{date}{suffix}').exists() for suffix in ('.jsonl', '.md', '_AI_enhanced_Chinese.jsonl', '_filter_report.json')):
            raise ValueError(f'Refusing to overwrite existing unverified digest: {date}')
        if candidates and client is None:
            if os.environ.get('MODEL_NAME') != manifest['model']:
                raise ValueError('Configured model differs from approved recovery model')
            client = SerialModel(manifest['model'])
        se = [p for p in candidates if 'cs.SE' in p['categories']]
        ai = [p for p in candidates if 'cs.SE' not in p['categories'] and 'cs.AI' in p['categories']]
        if len(se) + len(ai) != len(candidates):
            raise ValueError('Unsupported candidate categories')
        se_decisions = classify(se, 'se', client, cache) if se else {}
        ai_decisions = classify(ai, 'ai', client, cache) if ai else {}
        selected = []
        for paper in candidates:
            is_se = 'cs.SE' in paper['categories']
            decision = (se_decisions if is_se else ai_decisions)[paper['id']]
            if decision['relevant']:
                selection = {'policy': 'cs.SE_semopt_relevance' if is_se else 'cs.AI_semopt_relevance', **{k: decision[k] for k in ('relevant', 'score', 'reason_zh')}}
                paper['selection'] = selection
                if is_se:
                    paper['se_selection'] = selection
                selected.append(paper)
        enhanced = enhance(selected, client, cache)
        # Outputs are written only after every selected analysis validates.
        write_jsonl(data_dir / f'{date}.jsonl', selected)
        enhanced_path = data_dir / f'{date}_AI_enhanced_Chinese.jsonl'
        write_jsonl(enhanced_path, enhanced)
        if enhanced:
            subprocess.run(['python3', 'convert.py', '--data', '../data/' + enhanced_path.name], cwd=ROOT / 'to_md', check=True)
        else:
            (data_dir / f'{date}.md').write_text(f'# {date} arXiv 日报补档\n\n' + ('原始抓取批次中的论文已在前两日补档中去重收录，本日没有新增论文。\n' if not candidates else '本日新增候选经 SemOpt / Coding Agent 相关性筛选后，没有入选论文。\n'), encoding='utf-8')
        report = {'model': manifest['model'], 'policy': 'cs.SE_and_cs.AI_semopt_relevance',
            'input_records': len(sources[date]), 'input_total': len(candidates),
            'cs.SE_candidates': len(se), 'cs.SE_kept': sum('cs.SE' in p['categories'] for p in selected),
            'cs.AI_candidates': len(ai), 'cs.AI_kept': sum('cs.SE' not in p['categories'] for p in selected),
            'output_total': len(selected),
            'cs.SE_decisions': [se_decisions[p['id']] for p in se],
            'cs.AI_decisions': [ai_decisions[p['id']] for p in ai],
            'recovery': {**expected_report, 'status': 'complete', 'source_run_url': spec['url'],
                'source_unique': spec['unique_ids'], 'selected_ids': [p['id'] for p in selected],
                'duplicate_records_removed': len(sources[date]) - len(candidates),
                'max_concurrent_model_requests': 1, 'external_delivery': False}}
        write_json(report_path, report)
        verify_selection_report(report, candidates)
        validate_output(date, data_dir, [p['id'] for p in selected])
        summary.append({'date': date, 'candidates': len(candidates), 'selected': len(selected), 'status': 'recovered'})
        print(json.dumps(summary[-1]), flush=True)
    all_ids = [p['id'] for date in sorted(target_dates) for p in load_jsonl(data_dir / f'{date}.jsonl')]
    if len(all_ids) != len(set(all_ids)) or set(all_ids) & published:
        raise ValueError('Cross-digest duplicate publication')
    write_json(cache / 'verification.json', summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--cache', default='.recovery-cache/oct31-nov05-2026')
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    run(read_json(args.manifest), Path(args.cache), args.verify_only)


if __name__ == '__main__':
    main()
