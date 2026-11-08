#!/usr/bin/env python3
"""Complete the approved six-date recovery with adaptive Qwen concurrency."""
import argparse
import copy
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from threading import Condition, Lock
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, build_opener, HTTPRedirectHandler

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))
from scripts import recover_daily_batches as original

MODEL = 'ali/qwen3.7-plus'
LEGACY = 'huawei/deepseek-v4.1-flash'
MANIFEST = ROOT / 'recovery/oct31-nov05-2026/manifest.json'


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class AdaptiveModel:
    def __init__(self, cache):
        self.key = os.environ['OPENAI_API_KEY']
        self.base = os.environ['OPENAI_BASE_URL'].rstrip('/')
        if not self.base.startswith('https://'):
            raise ValueError('Existing model endpoint must use HTTPS')
        self.cache = cache
        self.condition = Condition()
        self.active = 0
        self.limit = 2
        self.good = 0
        self.ramp = True
        self.abort = False
        self.blocked_until = 0.0
        self.metrics = {'model': MODEL, 'maximum_allowed': 8, 'peak_inflight': 0,
                        'successes': 0, 'attempts': 0, 'http_errors': {},
                        'network_errors': 0, 'invalid_responses': 0,
                        'limit_changes': [{'limit': 2, 'reason': 'initial test'}]}

    def snapshot(self):
        with self.condition:
            return copy.deepcopy(self.metrics)

    def persist(self):
        # Caller holds condition, preventing concurrent metrics-file writes.
        original.write_json(self.cache / 'qwen-concurrency.json', self.metrics)

    def enter(self):
        with self.condition:
            while self.active >= self.limit or time.monotonic() < self.blocked_until:
                if self.abort:
                    raise RuntimeError('Stopped after a permanent model error')
                self.condition.wait(timeout=1)
            if self.abort:
                raise RuntimeError('Stopped after a permanent model error')
            self.active += 1
            self.metrics['attempts'] += 1
            self.metrics['peak_inflight'] = max(self.metrics['peak_inflight'], self.active)

    def leave(self):
        with self.condition:
            self.active -= 1
            self.condition.notify_all()

    def success(self):
        with self.condition:
            self.metrics['successes'] += 1
            self.good += 1
            if self.ramp and self.limit < 8 and self.good >= self.limit:
                self.limit *= 2
                self.good = 0
                event = {'limit': self.limit, 'reason': 'successful lower-concurrency requests'}
                self.metrics['limit_changes'].append(event)
                print('CONCURRENCY ' + json.dumps(event), flush=True)
            self.persist()
            self.condition.notify_all()

    def error(self, kind, delay, permanent=False):
        with self.condition:
            if isinstance(kind, int):
                counts = self.metrics['http_errors']
                counts[str(kind)] = counts.get(str(kind), 0) + 1
            else:
                self.metrics[kind] += 1
            if permanent:
                self.abort = True
            elif kind != 'invalid_responses':
                self.limit = max(1, self.limit // 2)
                self.ramp = False
                self.blocked_until = max(self.blocked_until, time.monotonic() + delay)
                event = {'limit': self.limit, 'reason': str(kind), 'cooldown_seconds': delay}
                self.metrics['limit_changes'].append(event)
                print('CONCURRENCY ' + json.dumps(event), flush=True)
            self.persist()
            self.condition.notify_all()

    def call(self, system, user, validate):
        payload = {'model': MODEL, 'messages': [{'role': 'system', 'content': system},
                    {'role': 'user', 'content': user}], 'temperature': 0,
                   'response_format': {'type': 'json_object'}}
        for attempt in range(8):
            delay = min(120, 5 * 2**attempt)
            permanent = False
            self.enter()
            try:
                request = Request(self.base + '/chat/completions', data=json.dumps(payload).encode(),
                                  headers={'Authorization': 'Bearer ' + self.key, 'Content-Type': 'application/json'})
                with build_opener(NoRedirect).open(request, timeout=180) as response:
                    body = json.load(response)
                value = json.loads(body['choices'][0]['message']['content'])
                validate(value)
            except HTTPError as error:
                kind = error.code
                permanent = kind not in (408, 409, 429) and kind < 500
                try:
                    retry_after = float(error.headers.get('Retry-After', 0))
                    if math.isfinite(retry_after):
                        delay = max(delay, min(600, max(0, retry_after)))
                except (ValueError, TypeError):
                    pass
                reason = f'HTTP {kind}'
                self.error(kind, delay, permanent)
            except (URLError, TimeoutError, OSError):
                kind, reason = 'network_errors', 'temporary connection failure'
                self.error(kind, delay)
            except (ValueError, KeyError, TypeError, IndexError):
                kind, reason = 'invalid_responses', 'invalid or incomplete structured response'
                self.error(kind, delay)
            else:
                self.success()
                return value
            finally:
                self.leave()
            if permanent or attempt == 7:
                raise RuntimeError(f'Model failed: {reason}; attempts={attempt + 1}') from None
            print(f'RETRY {reason}; attempt={attempt + 2}/8; wait={delay:.0f}s', flush=True)
            time.sleep(delay)
        raise AssertionError('unreachable')


def one_file(directory, name):
    paths = list(Path(directory).rglob(name))
    if len(paths) != 1:
        raise ValueError(f'Expected exactly one {name} in input artifact')
    return paths[0]


def seed_verified_cache(cache, legacy_dir, probe_dir, sources):
    source_path = one_file(legacy_dir, 'source-2026-10-31.json')
    baseline = source_path.parent
    if original.read_json(source_path) != sources['2026-10-31']:
        raise ValueError('Legacy cache source does not match verified archive')
    unique = {p['id']: p for p in sources['2026-10-31']}
    selected = set()
    for kind, is_se in [('se', True), ('ai', False)]:
        incoming = original.read_json(baseline / f'{kind}-decisions.json')
        expected = [p['id'] for p in unique.values() if ('cs.SE' in p['categories']) == is_se]
        original.validate_decisions({'decisions': list(incoming.values())}, expected)
        target = cache / f'{kind}-decisions.json'
        current = original.read_json(target) if target.exists() else {}
        for paper_id, row in incoming.items():
            if row['relevant']:
                selected.add(paper_id)
            old = current.get(paper_id)
            if old and {k: v for k, v in old.items() if k != 'model'} != row:
                raise ValueError('Conflicting cached classification')
            current[paper_id] = {**row, 'model': LEGACY}
        original.write_json(target, current)
    files = list((baseline / 'enhanced').glob('*.json'))
    if len(files) != 46:
        raise ValueError('Expected the 46 independently validated legacy analyses')
    legacy_ids = set()
    for path in files:
        paper_id = path.stem
        if paper_id not in selected:
            raise ValueError('Legacy analysis was not selected')
        ai = original.read_json(path)
        original.validate_ai(ai, unique[paper_id])
        target = cache / 'enhanced' / path.name
        if target.exists():
            current = original.read_json(target)
            if {k: v for k, v in current.items() if k != 'model'} != ai:
                raise ValueError('Conflicting cached legacy analysis')
        original.write_json(target, {**ai, 'model': LEGACY})
        legacy_ids.add(paper_id)
    probe_report = original.read_json(one_file(probe_dir, 'original-prompt-paper.json'))
    if probe_report.get('requested_model') != MODEL or probe_report.get('http_status') != 200 or probe_report.get('structured_validation_passed') is not True or probe_report.get('prompt_changed') is not False:
        raise ValueError('Qwen probe did not pass the unchanged prompt')
    ai = original.read_json(one_file(probe_dir, 'paper-2610.38215_AI.json'))
    original.validate_ai(ai, unique['2610.38215'])
    target = cache / 'enhanced' / '2610.38215.json'
    if target.exists() and {k: v for k, v in original.read_json(target).items() if k != 'model'} != ai:
        raise ValueError('Conflicting cached probe analysis')
    original.write_json(target, {**ai, 'model': MODEL})
    return legacy_ids


def parallel_jobs(jobs, function):
    errors = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(function, job) for job in jobs]
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as error:
                errors.append(type(error).__name__ + ': ' + str(error))
    if errors:
        # Messages originate from our sanitized client and local validators.
        raise RuntimeError(f'{len(errors)} work items failed: {errors[0]}')


def classify(items, kind, model, cache):
    path = cache / f'{kind}-decisions.json'
    decisions = original.read_json(path) if path.exists() else {}
    for paper in items:
        if paper['id'] in decisions:
            row = decisions[paper['id']]
            original.validate_decisions({'decisions': [row]}, [paper['id']])
            if row.get('model') not in (MODEL, LEGACY):
                raise ValueError('Unattributed cached classification')
    pending = [p for p in items if p['id'] not in decisions]
    batches = [pending[i:i + 10] for i in range(0, len(pending), 10)]
    prompt = original.constant_prompt(ROOT / ('scripts/backfill_se_selection.py' if kind == 'se' else 'ai/filter_relevance.py'))
    lock = Lock()
    def process(batch):
        papers = [{'id': p['id'], 'title': p['title'], 'abstract': p['summary'][:2400], 'categories': p['categories']} for p in batch]
        expected = [p['id'] for p in batch]
        value = model.call(prompt, json.dumps({'papers': papers}, ensure_ascii=False),
                           lambda value: original.validate_decisions(value, expected))
        with lock:
            decisions.update({row['id']: {**row, 'model': MODEL} for row in value['decisions']})
            original.write_json(path, decisions)
            print(f'CLASSIFIED {kind}: {len(expected)} new decisions; checkpoint saved', flush=True)
    parallel_jobs(batches, process)
    return decisions


def enhance(items, model, cache):
    # Deliberately use the existing prompt unchanged; Qwen passed this exact test.
    system = (ROOT / 'ai/system.txt').read_text(encoding='utf-8').replace('{language}', 'Chinese')
    template = (ROOT / 'ai/template.txt').read_text(encoding='utf-8')
    results = {}
    lock = Lock()
    def process(paper):
        path = cache / 'enhanced' / (paper['id'] + '.json')
        if path.exists():
            value = original.read_json(path)
            original.validate_ai(value, paper)
            if value.get('model') not in (MODEL, LEGACY):
                raise ValueError('Unattributed cached analysis')
        else:
            value = model.call(system, template.replace('{content}', paper['summary']),
                               lambda value: original.validate_ai(value, paper))
            value['model'] = MODEL
            original.write_json(path, value)
        with lock:
            results[paper['id']] = value
            print(f'ENHANCED {len(results)}/{len(items)}: {paper["id"]}; model={value["model"]}', flush=True)
    parallel_jobs(items, process)
    return [{**paper, 'AI': results[paper['id']]} for paper in items]


def count_models(rows):
    result = {}
    for row in rows:
        model = row['model']
        result[model] = result.get(model, 0) + 1
    return result


def verify_date(spec, candidates, data, legacy_ids):
    date = spec['date']
    report = original.read_json(data / f'{date}_filter_report.json')
    meta = report['recovery']
    if meta.get('source_records_sha256') != spec['records_sha256'] or meta.get('candidate_ids') != [p['id'] for p in candidates]:
        raise ValueError('Report differs from globally assigned original candidates')
    selected = original.verify_selection_report(report, candidates)
    original.validate_output(date, data, selected)
    source = {p['id']: p for p in candidates}
    raw = original.load_jsonl(data / f'{date}.jsonl')
    enhanced = original.load_jsonl(data / f'{date}_AI_enhanced_Chinese.jsonl')
    decisions = {r['id']: r for key in ('cs.SE_decisions', 'cs.AI_decisions') for r in report[key]}
    expected_classifier = LEGACY if date == '2026-10-31' else MODEL
    if any(row.get('model') != expected_classifier for row in decisions.values()):
        raise ValueError('Incorrect classification-model provenance')
    for paper, full in zip(raw, enhanced):
        if {k: v for k, v in paper.items() if k not in ('selection', 'se_selection')} != source[paper['id']]:
            raise ValueError('Original metadata changed')
        expected_model = LEGACY if paper['id'] in legacy_ids else MODEL
        if full['AI'].get('model') != expected_model:
            raise ValueError('Incorrect analysis-model provenance')
        selection = paper.get('se_selection') or paper['selection']
        if selection.get('model') != decisions[paper['id']].get('model'):
            raise ValueError('Incorrect selection-model provenance')
    if report['generation_models'] != count_models([p['AI'] for p in enhanced]):
        raise ValueError('Analysis model totals mismatch')
    if report['classification_models'] != count_models(list(decisions.values())):
        raise ValueError('Classification model totals mismatch')
    return report


def git(*args):
    subprocess.run(['git', *args], cwd=ROOT, check=True)


def publish(date):
    data = ROOT / 'data'
    (ROOT / 'assets/file-list.txt').write_text(''.join(p.name + '\n' for p in sorted(data.glob('*.jsonl'))), encoding='utf-8')
    git('config', 'user.name', os.environ['NAME'])
    git('config', 'user.email', os.environ['EMAIL'])
    paths = [f'data/{date}{suffix}' for suffix in ('.jsonl', '.md', '_AI_enhanced_Chinese.jsonl', '_filter_report.json')]
    git('add', *paths, 'assets/file-list.txt')
    if subprocess.run(['git', 'diff', '--staged', '--quiet'], cwd=ROOT).returncode:
        git('commit', '-m', f'backfill: recover {date} with Qwen and verified cached results')
        git('push', 'origin', 'HEAD:main')


def run(cache, legacy_dir, probe_dir, publish_enabled=True):
    manifest = original.read_json(MANIFEST)
    if not manifest.get('source_file') or not manifest.get('source_sha256'):
        raise ValueError('This continuation requires the approved verified source archive')
    sources = original.load_sources(manifest, cache)
    union = {p['id'] for rows in sources.values() for p in rows}
    if len(union) != 713 or manifest['expected_unique_candidates'] != 713:
        raise ValueError('Original 713-candidate union mismatch')
    legacy_ids = seed_verified_cache(cache, legacy_dir, probe_dir, sources)
    targets = {s['date'] for s in manifest['dates']}
    data = ROOT / 'data'
    published_ids = set()
    for path in data.glob('????-??-??_AI_enhanced_*.jsonl'):
        if path.name[:10] not in targets:
            published_ids.update(p['id'] for p in original.load_jsonl(path))
    assignments = original.assign_unique(sources, published_ids)
    model = AdaptiveModel(cache)
    summary = []
    for spec in manifest['dates']:
        date = spec['date']
        candidates = assignments[date]
        paths = [data / f'{date}{suffix}' for suffix in ('.jsonl', '.md', '_AI_enhanced_Chinese.jsonl', '_filter_report.json')]
        if all(p.exists() for p in paths):
            report = verify_date(spec, candidates, data, legacy_ids)
            summary.append({'date': date, 'candidates': len(candidates), 'selected': report['output_total'], 'status': 'verified_existing'})
            continue
        if any(p.exists() for p in paths):
            raise ValueError(f'Refusing to overwrite incomplete existing digest: {date}')
        se = [p for p in candidates if 'cs.SE' in p['categories']]
        ai = [p for p in candidates if 'cs.SE' not in p['categories'] and 'cs.AI' in p['categories']]
        if len(se) + len(ai) != len(candidates):
            raise ValueError('Unsupported original candidate category')
        se_decisions = classify(se, 'se', model, cache)
        ai_decisions = classify(ai, 'ai', model, cache)
        selected = []
        for source in candidates:
            paper = copy.deepcopy(source)
            is_se = 'cs.SE' in paper['categories']
            decision = (se_decisions if is_se else ai_decisions)[paper['id']]
            if decision['relevant']:
                selection = {'policy': 'cs.SE_semopt_relevance' if is_se else 'cs.AI_semopt_relevance',
                             **{k: decision[k] for k in ('relevant', 'score', 'reason_zh', 'model')}}
                paper['selection'] = selection
                if is_se:
                    paper['se_selection'] = selection
                selected.append(paper)
        enhanced = enhance(selected, model, cache)
        decisions_se = [se_decisions[p['id']] for p in se]
        decisions_ai = [ai_decisions[p['id']] for p in ai]
        report = {'model': MODEL, 'model_scope': 'New requests; per-record model provenance retains validated cached results.',
                  'policy': 'cs.SE_and_cs.AI_semopt_relevance', 'input_records': len(sources[date]),
                  'input_total': len(candidates), 'cs.SE_candidates': len(se),
                  'cs.SE_kept': sum('cs.SE' in p['categories'] for p in selected),
                  'cs.AI_candidates': len(ai), 'cs.AI_kept': sum('cs.SE' not in p['categories'] for p in selected),
                  'output_total': len(selected), 'cs.SE_decisions': decisions_se, 'cs.AI_decisions': decisions_ai,
                  'generation_models': count_models([p['AI'] for p in enhanced]),
                  'classification_models': count_models(decisions_se + decisions_ai),
                  'recovery': {'source_records_sha256': spec['records_sha256'], 'candidate_ids': [p['id'] for p in candidates],
                               'selected_ids': [p['id'] for p in selected], 'source_run_url': spec['url'],
                               'source_unique': spec['unique_ids'], 'status': 'complete',
                               'duplicate_records_removed': len(sources[date]) - len(candidates),
                               'external_delivery': False, 'prompt_changed': False,
                               'concurrency': model.snapshot()}}
        with tempfile.TemporaryDirectory(prefix='qwen-stage-') as directory:
            stage = Path(directory); stage_data = stage / 'data'; stage_data.mkdir()
            stage_converter = stage / 'to_md'; stage_converter.mkdir()
            for name in ('convert.py', 'paper_template.md'):
                shutil.copy2(ROOT / 'to_md' / name, stage_converter / name)
            original.write_jsonl(stage_data / f'{date}.jsonl', selected)
            original.write_jsonl(stage_data / f'{date}_AI_enhanced_Chinese.jsonl', enhanced)
            if selected:
                subprocess.run([sys.executable, 'convert.py', '--data', f'../data/{date}_AI_enhanced_Chinese.jsonl'], cwd=stage_converter, check=True)
            else:
                text = '原始抓取批次与之前日期的候选重复，本日没有新增候选论文。' if not candidates else '本日新增候选经相关性筛选后，没有入选论文。'
                (stage_data / f'{date}.md').write_text(f'# {date} arXiv 日报补档\n\n{text}\n', encoding='utf-8')
            original.write_json(stage_data / f'{date}_filter_report.json', report)
            verify_date(spec, candidates, stage_data, legacy_ids)
            for path in stage_data.iterdir():
                shutil.copy2(path, data / path.name)
        if publish_enabled:
            publish(date)
        summary.append({'date': date, 'candidates': len(candidates), 'selected': len(selected),
                        'generation_models': report['generation_models'], 'classification_models': report['classification_models'],
                        'status': 'published' if publish_enabled else 'validated'})
        original.write_json(cache / 'qwen-progress.json', summary)
        print('DATE_DONE ' + json.dumps(summary[-1], ensure_ascii=False), flush=True)
    all_ids = []
    for spec in manifest['dates']:
        report = verify_date(spec, assignments[spec['date']], data, legacy_ids)
        all_ids.extend(report['recovery']['selected_ids'])
    if len(all_ids) != len(set(all_ids)) or set(all_ids) & published_ids:
        raise ValueError('Cross-date duplicate publication')
    result = {'dates': summary, 'candidate_union': 713, 'selected_total': len(all_ids),
              'all_dates_verified': True, 'cross_date_duplicates': 0, 'concurrency': model.snapshot(), 'external_delivery': False}
    original.write_json(cache / 'qwen-final-verification.json', result)
    print('FINAL_VERIFIED ' + json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache', type=Path, default=Path('.recovery-cache/oct31-nov05-2026'))
    parser.add_argument('--legacy', required=True, type=Path)
    parser.add_argument('--probe', required=True, type=Path)
    args = parser.parse_args()
    run(args.cache, args.legacy, args.probe)
