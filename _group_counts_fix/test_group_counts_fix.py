"""Focused installer and data-integrity checks; standard library only."""
import ast
from collections import Counter
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('group_fix', str(HERE / 'APPLY_GROUP_COUNTS_FIX.py'))
fix = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fix)
AFTER = HERE / 'preview/Generated_Cases_34_Updated.json'
AUDIT = HERE / 'preview/CHANGE_REPORT.json'
UI_SOURCE = """from collections import Counter
import streamlit as st
def _banner(raw):
    st.warning('Generated examples only.')
    st.caption('Read-only demo files. Main Study databases, assessor reviews, readiness gates and inferential analyses are not used by this view.')
    st.write('Existing cards retained')
def render_information_demo():
    st.warning('Generated examples, not collected behavior. All 34 cases demonstrate this optional activity; the real study sampling plan is unchanged.')
    st.write('Existing information cards retained')
def untouched():
    return 'No change'
"""


def original_data():
    raw = json.loads(AFTER.read_text(encoding='utf-8'))
    report = json.loads(AUDIT.read_text(encoding='utf-8'))
    before = {r['participant_code']: r['before'] for r in report['reassigned_cases']}
    for person in raw['participants']:
        if person['participant_code'] in before:
            person['experience_group'] = before[person['participant_code']]
    raw.pop('case_group_assignment')
    return raw


def project(root):
    data = root / fix.DATA_REL
    ui = root / fix.UI_REL
    data.parent.mkdir(parents=True)
    ui.parent.mkdir(parents=True)
    data.write_text(json.dumps(original_data()), encoding='utf-8')
    ui.write_text(UI_SOURCE, encoding='utf-8')


class GroupCountTests(unittest.TestCase):
    def test_counts_and_only_group_fields_change(self):
        before = original_data()
        after = fix.update_groups(before)
        self.assertEqual(Counter(p['experience_group'] for p in after['participants']), fix.TARGET)
        changed = 0
        for old, new in zip(before['participants'], after['participants']):
            old_copy, new_copy = deepcopy(old), deepcopy(new)
            changed += old_copy.pop('experience_group') != new_copy.pop('experience_group')
            self.assertEqual(old_copy, new_copy)
        self.assertEqual(changed, 7)
        restored = deepcopy(after)
        restored.pop('case_group_assignment')
        for old, new in zip(before['participants'], restored['participants']):
            new['experience_group'] = old['experience_group']
        self.assertEqual(before, restored)
        self.assertEqual(after, fix.update_groups(after))

    def test_existing_numerical_summaries_retained(self):
        raw = fix.update_groups(original_data())
        for condition, expected in [('known_correct', 6.627450980392157), ('known_incorrect', 2.802941176470588)]:
            rows = [r for r in raw['trust_trials'] if r['condition'] == condition]
            self.assertAlmostEqual(sum(r['trust_mean'] for r in rows) / 340, expected)
        self.assertEqual(Counter(p['arm'] for p in raw['participants']), {'A': 17, 'B': 17})

    def test_rejects_real_unlabelled_and_unknown_groups(self):
        for record_type in ['top', 'participants', 'trust_trials', 'information_seeking']:
            raw = original_data()
            target = raw if record_type == 'top' else raw[record_type][0]
            target['is_synthetic'] = False
            with self.assertRaises(ValueError):
                fix.update_groups(raw)
        raw = original_data()
        raw['participants'][0]['experience_group'] = 'Unknown'
        with self.assertRaises(ValueError):
            fix.update_groups(raw)

    def test_ui_idempotence_and_crlf_support(self):
        for source in [UI_SOURCE, UI_SOURCE.replace('\n', '\r\n')]:
            updated = fix.patch_ui(source)
            self.assertEqual(updated.count(fix.MARKER), 1)
            self.assertEqual(updated.count(fix.INFO_MARKER), 1)
            self.assertEqual(fix.patch_ui(updated), updated)
            old_tree, new_tree = ast.parse(source), ast.parse(updated)
            self.assertEqual(ast.dump(old_tree.body[-1]), ast.dump(new_tree.body[-1]))

    def test_install_check_repeat_rollback_and_live_files_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            project(root)
            protected = ['data/journal/trust_stimulus_bank_v3/frozen/stimuli_participant.json', 'data/journal/trust_stimulus_bank_v3/frozen/manifest.json', 'artifacts/main_study.db', 'artifacts/assessor_reviews.csv']
            for relative in protected:
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b'protected live file')
            before = {relative: (root / relative).read_bytes() for relative in [fix.DATA_REL, fix.UI_REL]}
            backup = fix.install(root)
            self.assertIsNotNone(backup)
            self.assertEqual(fix.prepare(root)[2], {})
            self.assertIsNone(fix.install(root))
            self.assertEqual(len(list((root / 'artifacts/generated_case_group_backups').iterdir())), 1)
            for relative in protected:
                self.assertEqual((root / relative).read_bytes(), b'protected live file')
            fix.rollback(root, backup)
            for relative, contents in before.items():
                self.assertEqual((root / relative).read_bytes(), contents)

    def test_unknown_ui_stops_before_any_write(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            project(root)
            (root / fix.UI_REL).write_text('def another_banner():\n    pass\n', encoding='utf-8')
            before = (root / fix.DATA_REL).read_bytes()
            with self.assertRaises(ValueError):
                fix.install(root)
            self.assertEqual((root / fix.DATA_REL).read_bytes(), before)
            self.assertFalse((root / 'artifacts/generated_case_group_backups').exists())

    def test_rollback_refuses_later_edits(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            project(root)
            backup = fix.install(root)
            path = root / fix.UI_REL
            path.write_bytes(path.read_bytes() + b'\n# local edit\n')
            before = (root / fix.DATA_REL).read_bytes()
            with self.assertRaises(ValueError):
                fix.rollback(root, backup)
            self.assertEqual((root / fix.DATA_REL).read_bytes(), before)

    def test_preview_matches_transform(self):
        actual = json.loads(AFTER.read_text(encoding='utf-8'))
        self.assertEqual(fix.update_groups(original_data()), actual)
        self.assertTrue(actual['is_synthetic'])
        self.assertFalse(actual['eligible_for_main_study'])


if __name__ == '__main__':
    unittest.main()
