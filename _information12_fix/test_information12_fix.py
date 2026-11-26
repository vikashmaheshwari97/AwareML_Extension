"""Installation and data-preservation checks; standard library only."""
import ast
from collections import Counter
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('info12_fix', str(HERE / 'APPLY_INFORMATION_SEEKING_12_FIX.py'))
fix = importlib.util.module_from_spec(spec); spec.loader.exec_module(fix)


def parent():
    return json.loads((HERE / 'fixtures/source_demo.json').read_text(encoding='utf-8'))


def subset():
    data = (HERE / 'fixtures/source_demo.json').read_bytes()
    return fix.build_subset(json.loads(data), fix.digest(data))


def source():
    return (HERE / 'fixtures/source_ui.py').read_text(encoding='utf-8')


def project(root):
    for rel, data in [(fix.SOURCE_REL, (HERE / 'fixtures/source_demo.json').read_bytes()), (fix.UI_REL, source().encode('utf-8'))]:
        p = root / rel; p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(data)


def loader():
    tree = ast.parse(fix.UI_HELPERS)
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_load_information_demo12')
    mod = ast.Module(body=[fn], type_ignores=[])
    env = {'Path': Path, 'json': json, 'Counter': Counter}
    exec(compile(mod, '<installed demo loader>', 'exec'), env)
    return env['_load_information_demo12']


class Information12Tests(unittest.TestCase):
    def test_group_counts_and_one_response_per_case(self):
        r = subset()
        self.assertEqual(Counter(p['experience_group'] for p in r['participants']), dict(fix.GROUPS))
        self.assertEqual(len(r['participants']), 12)
        self.assertEqual(len(r['information_seeking']), 12)
        self.assertEqual(len(set(x['participant_code'] for x in r['information_seeking'])), 12)
        self.assertNotIn('trust_trials', r)

    def test_selected_records_and_events_retained_verbatim(self):
        before = parent(); r = subset(); codes = set(r['selection']['selected_codes'])
        self.assertEqual(r['participants'], [p for p in before['participants'] if p['participant_code'] in codes])
        self.assertEqual(r['information_seeking'], [p for p in before['information_seeking'] if p['participant_code'] in codes])
        self.assertEqual(r['scripted_events'], [e for e in before['scripted_events'] if e['track'] == 'information_seeking' and e['participant_code'] in codes])
        self.assertEqual(before, parent())

    def test_selection_does_not_use_ratings_or_decisions(self):
        a = parent(); b = deepcopy(a)
        for p in b['participants']:
            p['mean_trust_correct'] = 1; p['mean_trust_incorrect'] = 7
        for r in b['information_seeking']:
            r['decision_confidence'] = 1
        self.assertEqual(fix.build_subset(a, 'same')['selection'], fix.build_subset(b, 'same')['selection'])

    def test_exact_aggregate_totals_and_event_counts(self):
        r = subset(); s = fix.summarize(r)
        self.assertEqual((s['questions'], s['evidence_accesses_including_repeats'], s['repeat_accesses']), (14, 29, 5))
        self.assertEqual([s['final_decisions'][k]['count'] for k in ['Accept', 'Override', 'Reject']], [7, 3, 2])
        self.assertEqual(len(r['scripted_events']), 55)
        self.assertEqual(sum(v['all_accesses'] for v in s['evidence_by_area'].values()), 29)
        self.assertEqual(sum(v['repeats'] for v in s['evidence_by_area'].values()), 5)

    def test_rejects_real_or_unlabelled_parent_data(self):
        for key in [None, 'participants', 'trust_trials', 'post_study', 'information_seeking', 'scripted_events']:
            r = parent(); target = r if key is None else r[key][0]
            target['is_synthetic'] = False
            with self.assertRaises(ValueError): fix.build_subset(r, 'test')

    def test_requires_previously_fixed_parent_groups(self):
        r = parent(); r['participants'][0]['experience_group'] = 'Practitioner'
        with self.assertRaises(ValueError): fix.build_subset(r, 'test')

    def test_rejects_inconsistent_counts_or_cross_track_events(self):
        for kind in ['question', 'access', 'repeat', 'final', 'track', 'group']:
            r = subset()
            if kind == 'question': r['information_seeking'][0]['question_count'] += 1
            if kind == 'access': r['information_seeking'][0]['evidence_view_count'] += 1
            if kind == 'repeat': r['information_seeking'][0]['revisit_count'] += 1
            if kind == 'final': r['scripted_events'].pop()
            if kind == 'track': r['scripted_events'][0]['track'] = 'trust_calibration'
            if kind == 'group': r['participants'][0]['experience_group'] = 'Practitioner'
            with self.assertRaises(ValueError): fix.validate_subset(r)

    def test_ui_is_idempotent_supports_crlf_and_preserves_trust_functions(self):
        for original in [source(), source().replace('\n', '\r\n')]:
            after = fix.patch_ui(original)
            self.assertEqual(fix.patch_ui(after), after)
            self.assertEqual(after.count(fix.MARKER), 1)
            self.assertNotIn(fix.OLD_WARNING, after)
            old = {n.name: n for n in ast.parse(original).body if isinstance(n, ast.FunctionDef)}
            new = {n.name: n for n in ast.parse(after).body if isinstance(n, ast.FunctionDef)}
            for name, node in old.items():
                if name not in ['render_information_demo', '_downloads']:
                    self.assertEqual(ast.dump(node), ast.dump(new[name]), name)
            self.assertEqual([ast.dump(n) for n in old['_downloads'].body], [ast.dump(n) for n in new['_downloads'].body[1:]])

    def test_installed_loader_validates_current_subset(self):
        load = loader()
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'info12.json'; p.write_text(json.dumps(subset()), encoding='utf-8')
            self.assertEqual(load(p), subset())
            r = subset(); r['information_seeking'][0]['revisit_count'] += 1
            p.write_text(json.dumps(r), encoding='utf-8')
            with self.assertRaises(ValueError): load(p)
            r = subset(); r['is_synthetic'] = False; p.write_text(json.dumps(r), encoding='utf-8')
            with self.assertRaises(ValueError): load(p)

    def test_install_check_idempotence_rollback_and_protected_files(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); project(root)
            protected = [fix.SOURCE_REL, 'artifacts/study.db', 'artifacts/assessor_reviews.csv', 'data/journal/trust_stimulus_bank_v3/frozen/stimuli_participant.json', 'awareml/studies/information_seeking.py']
            for rel in protected[1:]:
                p = root / rel; p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(b'protected original data')
            original = {rel: (root / rel).read_bytes() for rel in protected + [fix.UI_REL]}
            backup = fix.install(root)
            self.assertEqual(fix.prepare(root)[1], {})
            self.assertIsNone(fix.install(root))
            for rel in protected: self.assertEqual((root / rel).read_bytes(), original[rel])
            fix.rollback(root, backup)
            self.assertFalse((root / fix.DATA_REL).exists())
            for rel, data in original.items(): self.assertEqual((root / rel).read_bytes(), data)

    def test_existing_valid_subset_can_be_restored(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); project(root)
            p = root / fix.DATA_REL; original = json.dumps(subset()).encode('utf-8'); p.write_bytes(original)
            backup = fix.install(root); fix.rollback(root, backup)
            self.assertEqual(p.read_bytes(), original)

    def test_unknown_ui_refused_before_any_mutation(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); project(root)
            (root / fix.UI_REL).write_text('def render_information_demo():\n    pass\n', encoding='utf-8')
            original = (root / fix.SOURCE_REL).read_bytes()
            with self.assertRaises(ValueError): fix.install(root)
            self.assertEqual((root / fix.SOURCE_REL).read_bytes(), original)
            self.assertFalse((root / fix.DATA_REL).exists())
            self.assertFalse((root / 'artifacts/information_demo12_backups').exists())

    def test_unlabelled_destination_refused_without_overwriting(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); project(root)
            p = root / fix.DATA_REL; p.write_text('{"is_synthetic": false}', encoding='utf-8')
            before = (root / fix.UI_REL).read_bytes()
            with self.assertRaises(ValueError): fix.install(root)
            self.assertEqual((root / fix.UI_REL).read_bytes(), before)
            self.assertEqual(p.read_text(), '{"is_synthetic": false}')

    def test_rollback_refuses_later_edits_before_restoring_anything(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); project(root); backup = fix.install(root)
            p = root / fix.UI_REL; p.write_bytes(p.read_bytes() + b'\n# later edit\n')
            before = (root / fix.DATA_REL).read_bytes()
            with self.assertRaises(ValueError): fix.rollback(root, backup)
            self.assertEqual((root / fix.DATA_REL).read_bytes(), before)

    def test_failed_install_restores_previously_written_files(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); project(root); before = (root / fix.UI_REL).read_bytes()
            real = fix.atomic_write; failed = [False]
            def fail_once(path, data):
                if path == root / fix.UI_REL and not failed[0]:
                    failed[0] = True; raise OSError('injected write failure')
                return real(path, data)
            with patch.object(fix, 'atomic_write', side_effect=fail_once):
                with self.assertRaises(OSError): fix.install(root)
            self.assertEqual((root / fix.UI_REL).read_bytes(), before)
            self.assertFalse((root / fix.DATA_REL).exists())

    def test_python38_syntax_and_preview_agree(self):
        ast.parse((HERE / 'APPLY_INFORMATION_SEEKING_12_FIX.py').read_text(encoding='utf-8'), feature_version=8)
        ast.parse(fix.patch_ui(source()), feature_version=8)
        preview = json.loads((HERE / 'preview/Information_Seeking_12_Generated_Cases.json').read_text(encoding='utf-8'))
        self.assertEqual(preview, subset())
        self.assertEqual(json.loads((HERE / 'preview/SUMMARY.json').read_text()), fix.summarize(preview))


if __name__ == '__main__':
    unittest.main()
