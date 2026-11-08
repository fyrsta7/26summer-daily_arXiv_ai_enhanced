import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import recover_daily_batches as recovery


def paper(identifier='2610.1'):
    return {'id': identifier, 'title': 'Example', 'summary': 'An abstract explaining a code analysis method.',
            'authors': ['Author'], 'categories': ['cs.SE'], 'abs': 'https://arxiv.org/abs/' + identifier,
            'pdf': 'https://arxiv.org/pdf/' + identifier}


def decision(identifier='2610.1'):
    return {'id': identifier, 'relevant': True, 'score': 0.8, 'reason_zh': '与代码分析方法相关'}


def analysis():
    return {key: '本文提出一种新的代码分析方法，通过系统实验研究程序优化的有效性和正确性。' for key in recovery.FIELDS}


class RecoveryTests(unittest.TestCase):
    def test_parse_uses_literal_not_execution(self):
        p = paper()
        self.assertEqual(recovery.parse_crawler_log('2026-10-31T01:01:01Z ' + repr(p)), [p])
        with self.assertRaises((SyntaxError, ValueError)):
            recovery.parse_crawler_log("2026-10-31T01:01:01Z {'id': __import__('os').system('false')}")

    def test_source_integrity(self):
        rows = [paper(), paper()]
        spec = {'date': '2026-10-31', 'raw_records': 2, 'unique_ids': 1, 'records_sha256': recovery.canonical_hash(rows)}
        recovery.verify_records(rows, spec)
        rows[1]['title'] = 'Tampered'
        with self.assertRaises(ValueError):
            recovery.verify_records(rows, spec)

    def test_earliest_assignment_and_published_exclusion(self):
        sources = {'2026-11-02': [paper(), paper('2610.2')], '2026-11-01': [paper(), paper()],
                   '2026-11-03': [paper('2610.3')]}
        result = recovery.assign_unique(sources, ['2610.3'])
        self.assertEqual({day: [p['id'] for p in rows] for day, rows in result.items()},
                         {'2026-11-01': ['2610.1'], '2026-11-02': ['2610.2'], '2026-11-03': []})

    def test_complete_decisions_required(self):
        recovery.validate_decisions({'decisions': [decision()]}, ['2610.1'])
        for rows in ([], [decision(), decision()], [decision('2610.2')]):
            with self.assertRaises(ValueError):
                recovery.validate_decisions({'decisions': rows}, ['2610.1'])

    def test_decision_types_strict(self):
        for field, value in [('relevant', 'false'), ('score', float('nan')), ('score', 1.1), ('reason_zh', '')]:
            d = decision(); d[field] = value
            with self.assertRaises(ValueError):
                recovery.validate_decisions({'decisions': [d]}, ['2610.1'])

    def test_fallback_and_refusal_rejected(self):
        recovery.validate_ai(analysis(), paper())
        for bad in ('Processing failed', 'This content has not passed the compliance test and has been hidden.', ''):
            ai = analysis(); ai['method'] = bad
            with self.assertRaises(ValueError):
                recovery.validate_ai(ai, paper())

    def test_incomplete_translation_rejected(self):
        ai = analysis(); ai['abstract_zh'] = '短摘要'
        with self.assertRaises(ValueError):
            recovery.validate_ai(ai, paper())

    def test_atomic_cache_roundtrip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'child' / 'test.json'
            recovery.write_json(path, {'ok': True})
            self.assertEqual(recovery.read_json(path), {'ok': True})
            self.assertFalse(path.with_suffix('.json.tmp').exists())

    def test_output_coverage_and_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory); p = paper(); date = '2026-11-01'
            recovery.write_jsonl(data / f'{date}.jsonl', [p])
            recovery.write_jsonl(data / f'{date}_AI_enhanced_Chinese.jsonl', [{**p, 'AI': analysis()}])
            (data / f'{date}.md').write_text(f'### [1] [Example]({p["abs"]})')
            recovery.write_json(data / f'{date}_filter_report.json', {'output_total': 1, 'recovery': {'status': 'complete'}})
            recovery.validate_output(date, data, [p['id']])
            (data / f'{date}.md').write_text('Missing paper')
            with self.assertRaises(ValueError):
                recovery.validate_output(date, data, [p['id']])

    def test_prompts_available_without_importing_dependencies(self):
        self.assertIn('decisions', recovery.constant_prompt(recovery.ROOT / 'ai/filter_relevance.py'))
        self.assertIn('decisions', recovery.constant_prompt(recovery.ROOT / 'scripts/backfill_se_selection.py'))

    def test_selected_ids_must_match_all_candidate_decisions(self):
        report = {'cs.SE_decisions': [decision()], 'cs.AI_decisions': [], 'output_total': 1,
                  'recovery': {'selected_ids': ['2610.1']}}
        self.assertEqual(recovery.verify_selection_report(report, [paper()]), ['2610.1'])
        report['cs.SE_decisions'][0]['relevant'] = False
        with self.assertRaises(ValueError):
            recovery.verify_selection_report(report, [paper()])


if __name__ == '__main__':
    unittest.main()
