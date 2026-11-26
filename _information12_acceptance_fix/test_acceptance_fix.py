import ast
from collections import Counter
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
spec = importlib.util.spec_from_file_location('acceptance_fix', str(HERE / 'APPLY_INFORMATION_SEEKING_ACCEPTANCE_FIX.py'))
fix = importlib.util.module_from_spec(spec); spec.loader.exec_module(fix)


def original():
    return json.loads((HERE / 'fixtures/source_information12.json').read_text(encoding='utf-8'))


def source():
    return (HERE / 'fixtures/source_ui.py').read_text(encoding='utf-8')


def project(root):
    for rel, content in [(fix.DATA_REL, json.dumps(original()).encode('utf-8')), (fix.UI_REL, source().encode('utf-8'))]:
        p = root / rel; p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(content)


class AcceptanceTests(unittest.TestCase):
    def test_integer_counts_percentages_and_composition(self):
        r = fix.update_demo(original()); s = fix.core.summarize(r)
        self.assertEqual(fix.counts(r), {'Accept': 9, 'Override': 2, 'Reject': 1})
        self.assertEqual(['{:.2f}'.format(s['final_decisions'][k]['percent']) for k in ['Accept', 'Override', 'Reject']], ['75.00', '16.67', '8.33'])
        self.assertEqual(s['group_counts'], {'University / CS student': 4, 'Practitioner': 3, 'ML / AutoML expert': 5})
        self.assertEqual(s['cases'], 12)
        self.assertEqual(min(range(13), key=lambda n: abs(100 * n / 12 - 73.46)), 9)

    def test_two_demo_decisions_and_linked_events_change(self):
        before = original(); after = fix.update_demo(before)
        self.assertEqual(len(after[fix.POLICY_KEY]['changed_cases']), 2)
        self.assertEqual([x['participant_code'] for x in after[fix.POLICY_KEY]['changed_cases']], ['DEMO_P032', 'DEMO_P034'])
        for a, b in zip(before['information_seeking'], after['information_seeking']):
            aa = deepcopy(a); bb = deepcopy(b)
            for k in ['recommendation_action', 'override_text', 'rationale']:
                aa.pop(k); bb.pop(k)
            self.assertEqual(aa, bb)
        self.assertEqual(before['participants'], after['participants'])
        self.assertEqual(before['selection'], after['selection'])
        self.assertEqual(before['information_seeking_context'], after['information_seeking_context'])
        self.assertEqual([e for e in before['scripted_events'] if e['event_type'] != 'scripted_final_decision'], [e for e in after['scripted_events'] if e['event_type'] != 'scripted_final_decision'])
        fix.core.validate_subset(after)

    def test_questions_evidence_repeats_and_confidence_preserved(self):
        r = fix.update_demo(original()); s = fix.core.summarize(r)
        self.assertEqual((s['questions'], s['evidence_accesses_including_repeats'], s['repeat_accesses']), (14, 29, 5))
        self.assertEqual(sum(x['decision_confidence'] for x in r['information_seeking']) / 12, 6.5)
        self.assertTrue(r['is_synthetic']); self.assertFalse(r['eligible_for_main_study'])

    def test_transform_and_ui_are_idempotent(self):
        r = fix.update_demo(original()); self.assertEqual(fix.update_demo(r), r)
        for s in [source(), source().replace('\n', '\r\n')]:
            updated = fix.patch_ui(s)
            self.assertEqual(updated, fix.patch_ui(updated))
            self.assertNotIn(fix.core.NEW_WARNING, updated)
            self.assertIn(fix.WARNING, updated)
            self.assertIn('Decision (%)', updated)

    def test_other_functions_unchanged(self):
        before = {n.name: ast.dump(n) for n in ast.parse(source()).body if isinstance(n, ast.FunctionDef)}
        after = {n.name: ast.dump(n) for n in ast.parse(fix.patch_ui(source())).body if isinstance(n, ast.FunctionDef)}
        for name, value in before.items():
            if name != 'render_information_demo': self.assertEqual(after[name], value)

    def test_real_or_unlabelled_data_rejected(self):
        for key in [None, 'participants', 'information_seeking', 'scripted_events']:
            r = original(); target = r if key is None else r[key][0]
            target['is_synthetic'] = False
            with self.assertRaises(ValueError): fix.update_demo(r)

    def test_unknown_or_edited_policy_rejected(self):
        r = fix.update_demo(original()); r[fix.POLICY_KEY]['version'] = 'unknown'
        with self.assertRaises(ValueError): fix.update_demo(r)
        r = fix.update_demo(original()); r[fix.POLICY_KEY]['actual_counts']['Accept'] = 8
        with self.assertRaises(ValueError): fix.update_demo(r)

    def test_install_check_idempotence_rollback_and_protected_files(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); project(root)
            protected = ['data/illustrative_demo/awareml_34_cases.json', 'artifacts/study.db', 'artifacts/assessor_reviews.csv', 'data/journal/trust_stimulus_bank_v3/frozen/stimuli_participant.json', 'awareml/studies/information_seeking.py']
            for rel in protected:
                p = root / rel; p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(b'unchanged protected file')
            before = {rel: (root / rel).read_bytes() for rel in [fix.DATA_REL, fix.UI_REL] + protected}
            backup = fix.install(root)
            self.assertEqual(fix.prepare(root)[1], {})
            self.assertIsNone(fix.install(root))
            for rel in protected: self.assertEqual((root / rel).read_bytes(), before[rel])
            fix.rollback(root, backup)
            for rel, data in before.items(): self.assertEqual((root / rel).read_bytes(), data)

    def test_unknown_ui_stops_before_mutation(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); project(root)
            p = root / fix.UI_REL; p.write_text('def render_information_demo():\n    pass\n')
            before = (root / fix.DATA_REL).read_bytes()
            with self.assertRaises(ValueError): fix.install(root)
            self.assertEqual((root / fix.DATA_REL).read_bytes(), before)
            self.assertFalse((root / 'artifacts/information_acceptance_demo_backups').exists())

    def test_rollback_refuses_later_edits(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); project(root); backup = fix.install(root)
            p = root / fix.UI_REL; p.write_bytes(p.read_bytes() + b'\n# local edit\n')
            before = (root / fix.DATA_REL).read_bytes()
            with self.assertRaises(ValueError): fix.rollback(root, backup)
            self.assertEqual((root / fix.DATA_REL).read_bytes(), before)

    def test_interrupted_install_restores_files(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); project(root)
            before = {rel: (root / rel).read_bytes() for rel in [fix.DATA_REL, fix.UI_REL]}
            real = fix.core.atomic_write; failed = [False]
            def fail_once(path, data):
                if path == root / fix.UI_REL and not failed[0]:
                    failed[0] = True; raise OSError('injected write failure')
                return real(path, data)
            with patch.object(fix.core, 'atomic_write', side_effect=fail_once):
                with self.assertRaises(OSError): fix.install(root)
            for rel, data in before.items(): self.assertEqual((root / rel).read_bytes(), data)

    def test_python38_syntax_and_generated_preview(self):
        for p in [HERE / 'APPLY_INFORMATION_SEEKING_ACCEPTANCE_FIX.py', HERE / 'information12_core.py']:
            ast.parse(p.read_text(encoding='utf-8'), feature_version=8)
        ast.parse(fix.patch_ui(source()), feature_version=8)
        self.assertEqual(json.loads((HERE / 'preview/Information_Seeking_12_Updated.json').read_text()), fix.update_demo(original()))

    def test_direct_upgrade_from_34_case_demo(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            for rel, fixture in [(fix.core.SOURCE_REL, 'source_parent34.json'), (fix.UI_REL, 'source_ui34.py')]:
                p = root / rel; p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes((HERE / 'fixtures' / fixture).read_bytes())
            original_parent = (root / fix.core.SOURCE_REL).read_bytes(); original_ui = (root / fix.UI_REL).read_bytes()
            backup = fix.install(root)
            r, changes = fix.prepare(root)
            self.assertFalse(changes); self.assertEqual(fix.counts(r), fix.TARGET)
            self.assertEqual((root / fix.core.SOURCE_REL).read_bytes(), original_parent)
            fix.rollback(root, backup)
            self.assertFalse((root / fix.DATA_REL).exists())
            self.assertEqual((root / fix.UI_REL).read_bytes(), original_ui)


if __name__ == '__main__':
    unittest.main()
