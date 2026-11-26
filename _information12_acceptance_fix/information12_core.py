"""Install an isolated 12-case Information-Seeking demo. Python 3.8+."""
import argparse
import ast
from collections import Counter
from copy import deepcopy
import datetime
import hashlib
import json
from pathlib import Path

VERSION = 'information_seeking_demo12_4_3_5_v1'
SOURCE_REL = 'data/illustrative_demo/awareml_34_cases.json'
DATA_REL = 'data/illustrative_demo/awareml_information_seeking_12_cases.json'
UI_REL = 'awareml/ui_v2/illustrative_study_results.py'
GROUPS = [('University / CS student', 4), ('Practitioner', 3), ('ML / AutoML expert', 5)]
PARENT_COUNTS = {'University / CS student': 7, 'Practitioner': 12, 'ML / AutoML expert': 15}
AREAS = ['Performance', 'Fairness', 'Sustainability', 'Drift', 'Explainability']
MARKER = '# awareml_information_demo12_4_3_5_v1'
OLD_WARNING = "Generated examples, not collected behavior. All 34 cases demonstrate this optional activity; the real study sampling plan is unchanged."
NEW_WARNING = 'Generated examples, not collected behavior. These 12 cases demonstrate the optional Information-Seeking activity as a subset of the 34-case Trust Calibration demo.'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def generated(record):
    return record.get('is_synthetic') is True and record.get('eligible_for_main_study') is False


def validate_parent(raw):
    if not generated(raw):
        raise ValueError('Only explicitly generated demo data can be used.')
    for key in ['participants', 'trust_trials', 'post_study', 'information_seeking', 'scripted_events']:
        if not isinstance(raw.get(key), list) or not raw[key] or any(not generated(r) for r in raw[key]):
            raise ValueError('Real, unlabelled or missing records rejected: ' + key)
    people = raw['participants']
    codes = [p['participant_code'] for p in people]
    if len(codes) != 34 or len(set(codes)) != 34:
        raise ValueError('Expected 34 distinct parent demo cases.')
    if Counter(p['experience_group'] for p in people) != PARENT_COUNTS:
        raise ValueError('Install the earlier 7 / 12 / 15 group-count fix first. Parent data will not be relabelled.')
    if len(raw['trust_trials']) != 680:
        raise ValueError('Expected the existing 680 Trust Calibration trials.')
    for code in codes:
        rows = [r for r in raw['trust_trials'] if r['participant_code'] == code]
        if len(rows) != 20 or Counter(r['condition'] for r in rows) != {'known_correct': 10, 'known_incorrect': 10}:
            raise ValueError('Invalid parent trial balance: ' + code)
    for key in ['post_study', 'information_seeking']:
        if Counter(r['participant_code'] for r in raw[key]) != Counter(codes):
            raise ValueError('Invalid parent case linkage: ' + key)


def build_subset(raw, source_sha256):
    validate_parent(raw)
    selected = set()
    for group, quota in GROUPS:
        people = sorted([p for p in raw['participants'] if p['experience_group'] == group], key=lambda p: p['participant_code'])
        # Deterministic, outcome-independent example selection across the code range.
        for i in range(quota):
            selected.add(people[i * (len(people) - 1) // (quota - 1)]['participant_code'])
    result = {
        'format_version': VERSION,
        'is_synthetic': True,
        'eligible_for_main_study': False,
        'data_origin': 'computer_generated_demonstration',
        'purpose': 'Separate 12-case descriptive Information-Seeking demo; no human responses.',
        'source_file': SOURCE_REL,
        'source_sha256': source_sha256,
        'selection': {
            'parent_cases': 34, 'information_seeking_cases': 12,
            'requested_groups': dict(GROUPS),
            'selected_codes': sorted(selected),
            'rule': 'Within each group, sort case codes and select evenly spaced indices i*(n-1)//(quota-1). Ratings, decisions and access counts are not selection criteria.',
            'record_policy': 'Selected records and events are copied without changing their contents.'
        },
        'participants': [deepcopy(p) for p in raw['participants'] if p['participant_code'] in selected],
        'information_seeking': [deepcopy(r) for r in raw['information_seeking'] if r['participant_code'] in selected],
        'scripted_events': [deepcopy(e) for e in raw['scripted_events'] if e['track'] == 'information_seeking' and e['participant_code'] in selected],
        'information_seeking_context': deepcopy(raw['information_seeking_context'])
    }
    validate_subset(result)
    return result


def validate_subset(raw):
    if not generated(raw) or raw.get('format_version') != VERSION:
        raise ValueError('Expected a labelled 12-case generated demo.')
    for key in ['participants', 'information_seeking', 'scripted_events']:
        if not isinstance(raw.get(key), list) or not raw[key] or any(not generated(r) for r in raw[key]):
            raise ValueError('Missing or unlabelled subset records: ' + key)
    people = raw['participants']; rows = raw['information_seeking']; events = raw['scripted_events']
    codes = [p['participant_code'] for p in people]
    if len(codes) != 12 or len(set(codes)) != 12 or Counter(r['participant_code'] for r in rows) != Counter(codes):
        raise ValueError('Expected exactly 12 distinct cases and one final response per case.')
    if Counter(p['experience_group'] for p in people) != dict(GROUPS):
        raise ValueError('Expected 4 students, 3 practitioners and 5 experts.')
    if raw['selection']['selected_codes'] != sorted(codes):
        raise ValueError('Selection metadata does not match the records.')
    if any(e['participant_code'] not in codes or e['track'] != 'information_seeking' for e in events):
        raise ValueError('Events from another case or track rejected.')
    for row in rows:
        counts = Counter(row['evidence_sequence'])
        if set(counts) - set(AREAS):
            raise ValueError('Unknown evidence area.')
        if row['evidence_view_count'] != sum(counts.values()) or row['unique_evidence_areas'] != len(counts):
            raise ValueError('Evidence-access total does not match its sequence.')
        if row['revisit_count'] != sum(max(0, n - 1) for n in counts.values()):
            raise ValueError('Repeat-access total does not match its sequence.')
        if row['question_count'] != len(row['questions']):
            raise ValueError('Question count does not match question records.')
        if not 1 <= row['decision_confidence'] <= 7:
            raise ValueError('Decision confidence must be on the 1 to 7 scale.')
        if row['recommendation_action'].split(' — ')[0] not in ['Accept', 'Override', 'Reject']:
            raise ValueError('Unknown final decision.')
        case_events = [e for e in events if e['participant_code'] == row['participant_code']]
        by_type = {typ: [e for e in case_events if e['event_type'] == typ] for typ in ['scripted_evidence_view', 'scripted_question', 'scripted_final_decision']}
        if sum(map(len, by_type.values())) != len(case_events):
            raise ValueError('Unsupported Information-Seeking event type.')
        if [e['payload']['area'] for e in by_type['scripted_evidence_view']] != row['evidence_sequence']:
            raise ValueError('Evidence event sequence does not match the response.')
        if [e['payload'] for e in by_type['scripted_question']] != row['questions']:
            raise ValueError('Question events do not match the response.')
        final = by_type['scripted_final_decision']
        if len(final) != 1 or final[0]['payload']['action'] != row['recommendation_action']:
            raise ValueError('Expected one matching final-decision event per case.')


def summarize(raw):
    validate_subset(raw)
    views = Counter(); repeats = Counter(); decisions = Counter()
    for row in raw['information_seeking']:
        c = Counter(row['evidence_sequence']); views.update(c)
        repeats.update({k: max(0, n - 1) for k, n in c.items()})
        decisions[row['recommendation_action'].split(' — ')[0]] += 1
    return {
        'is_synthetic': True, 'eligible_for_main_study': False,
        'cases': 12, 'final_responses': 12,
        'group_counts': dict(Counter(p['experience_group'] for p in raw['participants'])),
        'questions': sum(r['question_count'] for r in raw['information_seeking']),
        'evidence_accesses_including_repeats': sum(views.values()),
        'repeat_accesses': sum(repeats.values()),
        'evidence_by_area': {k: {'all_accesses': views[k], 'repeats': repeats[k]} for k in AREAS},
        'final_decisions': {k: {'count': decisions[k], 'percent': 100 * decisions[k] / 12} for k in ['Accept', 'Override', 'Reject']}
    }


# Installed inside the existing researcher-only UI; uses its existing imports.
UI_HELPERS = r'''
# awareml_information_demo12_4_3_5_v1
def _load_information_demo12(path=None):
    target = Path(path) if path is not None else DATA_PATH.with_name('awareml_information_seeking_12_cases.json')
    raw = json.loads(target.read_text(encoding='utf-8-sig'))
    if raw.get('is_synthetic') is not True or raw.get('eligible_for_main_study') is not False or raw.get('format_version') != 'information_seeking_demo12_4_3_5_v1':
        raise ValueError('Only the labelled 12-case generated Information-Seeking dataset is allowed.')
    for name in ['participants', 'information_seeking', 'scripted_events']:
        if not isinstance(raw.get(name), list) or not raw[name]:
            raise ValueError('Missing generated Information-Seeking records: ' + name)
        if any(r.get('is_synthetic') is not True or r.get('eligible_for_main_study') is not False for r in raw[name]):
            raise ValueError('Real or unlabelled records rejected: ' + name)
    codes = [p['participant_code'] for p in raw['participants']]
    if len(codes) != 12 or len(set(codes)) != 12 or Counter(r['participant_code'] for r in raw['information_seeking']) != Counter(codes):
        raise ValueError('Expected 12 distinct generated cases with one final response each.')
    if Counter(p['experience_group'] for p in raw['participants']) != {'University / CS student': 4, 'Practitioner': 3, 'ML / AutoML expert': 5}:
        raise ValueError('Expected 4 students, 3 practitioners and 5 ML / AutoML experts.')
    if raw['selection']['selected_codes'] != sorted(codes):
        raise ValueError('Information-Seeking selection metadata does not match.')
    if any(e['participant_code'] not in codes or e['track'] != 'information_seeking' for e in raw['scripted_events']):
        raise ValueError('Event from another track or case rejected.')
    areas = {'Performance', 'Fairness', 'Sustainability', 'Drift', 'Explainability'}
    for row in raw['information_seeking']:
        c = Counter(row['evidence_sequence'])
        if set(c) - areas or row['evidence_view_count'] != sum(c.values()) or row['unique_evidence_areas'] != len(c):
            raise ValueError('Evidence-access count mismatch.')
        if row['revisit_count'] != sum(max(0, n - 1) for n in c.values()) or row['question_count'] != len(row['questions']):
            raise ValueError('Question or repeat-access count mismatch.')
        if not 1 <= row['decision_confidence'] <= 7 or row['recommendation_action'].split(' — ')[0] not in ['Accept', 'Override', 'Reject']:
            raise ValueError('Invalid final response.')
        events = [e for e in raw['scripted_events'] if e['participant_code'] == row['participant_code']]
        views = [e['payload']['area'] for e in events if e['event_type'] == 'scripted_evidence_view']
        questions = [e['payload'] for e in events if e['event_type'] == 'scripted_question']
        final = [e for e in events if e['event_type'] == 'scripted_final_decision']
        if len(events) != len(views) + len(questions) + len(final) or views != row['evidence_sequence'] or questions != row['questions']:
            raise ValueError('Scripted events do not match the response.')
        if len(final) != 1 or final[0]['payload']['action'] != row['recommendation_action']:
            raise ValueError('Expected one matching final-decision event per case.')
    return raw


def _information_demo12_downloads(raw):
    with st.expander('Demo records and downloads'):
        for name in ['participants', 'information_seeking', 'scripted_events']:
            frame = pd.DataFrame(raw[name])
            st.dataframe(frame, use_container_width=True, hide_index=True)
            st.download_button('Download illustrative ' + name, frame.to_csv(index=False).encode('utf-8'), file_name='ILLUSTRATIVE_INFORMATION_12_' + name + '.csv', mime='text/csv', key='demo12_download_information_' + name)
        st.download_button('Download complete 12-case Information-Seeking demo JSON', json.dumps(raw, indent=2).encode('utf-8'), file_name='ILLUSTRATIVE_INFORMATION_12_cases.json', mime='application/json', key='demo12_json_information')
'''


def function_source(source, name):
    nodes = [n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == name]
    if len(nodes) != 1:
        raise ValueError('Expected exactly one UI function: ' + name)
    node = nodes[0]; lines = source.splitlines(keepends=True)
    return node, ''.join(lines[node.lineno - 1:node.end_lineno])


def patch_ui(source):
    source = source.replace('\r\n', '\n')
    if MARKER in source:
        helper = UI_HELPERS.strip()
        if helper not in source:
            raise ValueError('The installed demo12 helper was edited; no files changed.')
        _, render = function_source(source, 'render_information_demo')
        _, downloads = function_source(source, '_downloads')
        if 'try:raw=_load_information_demo12()' not in render or NEW_WARNING not in render or '_information_demo12_downloads(raw)' not in downloads:
            raise ValueError('Incomplete or modified Information-Seeking demo12 patch.')
        return source
    if 'from collections import Counter' not in source:
        raise ValueError('Unsupported demo UI imports; no files changed.')
    node, render = function_source(source, 'render_information_demo')
    if render.count('try:raw=load_demo()') != 1 or render.count(OLD_WARNING) != 1:
        raise ValueError('Unsupported Information-Seeking demo UI version. Existing files retained.')
    if '_researcher_access()' not in render or "_downloads(raw,'information')" not in render:
        raise ValueError('Expected the existing researcher-only generated-results tab.')
    replacement = render.replace('try:raw=load_demo()', 'try:raw=_load_information_demo12()', 1).replace(OLD_WARNING, NEW_WARNING, 1)
    # Existing 7/12/15 fix already derives these from raw; raw is now the 12-case file.
    if 'group_counts = Counter(' not in replacement:
        anchor = "    st.warning('" + NEW_WARNING + "')\n"
        if anchor not in replacement:
            raise ValueError('Unknown Information-Seeking banner placement.')
        replacement = replacement.replace(anchor, anchor + "    group_counts = Counter(p['experience_group'] for p in raw['participants'])\n    st.caption('Generated case groups: {0} students | {1} practitioners | {2} ML / AutoML experts'.format(group_counts['University / CS student'], group_counts['Practitioner'], group_counts['ML / AutoML expert']))\n", 1)
    anchor = "    rows=raw['information_seeking'];cards=st.columns(4)"
    if replacement.count(anchor) != 1:
        raise ValueError('Unknown Information-Seeking metric layout.')
    replacement = replacement.replace(anchor, "    st.caption('12 final responses, one per case. Evidence accesses count individual openings; repeat openings are included in the access total.')\n" + anchor, 1)
    replacement = replacement.replace("key='compact_demo_case'", "key='compact_demo_case_info12'")
    replacement = replacement.replace("fig.update_yaxes(title='Generated cases',dtick=5)", "fig.update_yaxes(title='Generated cases',dtick=1,range=[0,max(counts.values())+2])")
    replacement = replacement.replace("fig.update_yaxes(title='Scripted accesses',dtick=5)", "fig.update_yaxes(title='Scripted accesses',dtick=1)")
    lines = source.splitlines(keepends=True)
    lines[node.lineno - 1:node.end_lineno] = [replacement + '\n']
    updated = ''.join(lines)
    node, downloads = function_source(updated, '_downloads')
    dlines = downloads.splitlines(keepends=True)
    dlines[1:1] = ["    if track == 'information':\n        return _information_demo12_downloads(raw)\n"]
    lines = updated.splitlines(keepends=True)
    lines[node.lineno - 1:node.end_lineno] = [''.join(dlines) + '\n']
    result = ''.join(lines).rstrip() + '\n\n' + UI_HELPERS.strip() + '\n'
    ast.parse(result)
    # Ensure every original top-level function except the two intended ones is unchanged.
    before = {n.name: ast.dump(n) for n in ast.parse(source).body if isinstance(n, ast.FunctionDef)}
    after = {n.name: ast.dump(n) for n in ast.parse(result).body if isinstance(n, ast.FunctionDef)}
    for name, dump in before.items():
        if name not in ['render_information_demo', '_downloads'] and after.get(name) != dump:
            raise ValueError('Unexpected edit outside Information-Seeking: ' + name)
    return result


def prepare(root):
    if not (root / SOURCE_REL).is_file() or not (root / UI_REL).is_file():
        raise ValueError('Run from the AwareML project root after installing the existing demo tabs and the 7 / 12 / 15 group fix.')
    parent_bytes = (root / SOURCE_REL).read_bytes()
    raw = json.loads(parent_bytes.decode('utf-8-sig'))
    subset = build_subset(raw, digest(parent_bytes))
    source = (root / UI_REL).read_bytes().decode('utf-8-sig').replace('\r\n', '\n')
    updated = patch_ui(source)
    if (root / DATA_REL).exists():
        current = json.loads((root / DATA_REL).read_text(encoding='utf-8-sig'))
        validate_subset(current)
    proposed = {DATA_REL: (json.dumps(subset, ensure_ascii=True, indent=2) + '\n').encode('utf-8'), UI_REL: updated.encode('utf-8')}
    changes = {rel: value for rel, value in proposed.items() if not (root / rel).is_file() or (root / rel).read_bytes() != value}
    return subset, changes


def atomic_write(path, contents):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.info12.tmp')
    try:
        temp.write_bytes(contents); temp.replace(path)
    finally:
        if temp.exists():
            temp.unlink()


def install(root):
    subset, changes = prepare(root)
    if not changes:
        print('Already installed: Information Seeking demo has 12 cases (4 / 3 / 5).')
        return None
    backup = root / 'artifacts/information_demo12_backups' / datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    backup.mkdir(parents=True)
    entries = []; originals = {}
    for rel, contents in changes.items():
        path = root / rel; existed = path.is_file(); old = path.read_bytes() if existed else None
        originals[rel] = old
        if existed:
            saved = backup / rel; saved.parent.mkdir(parents=True, exist_ok=True); saved.write_bytes(old)
        entries.append({'path': rel, 'existed': existed, 'previous_sha256': digest(old) if existed else None, 'installed_sha256': digest(contents)})
    manifest = {'patch': VERSION, 'source_sha256': digest((root / SOURCE_REL).read_bytes()), 'summary': summarize(subset), 'files': entries}
    (backup / 'restore_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    try:
        for rel, contents in changes.items():
            atomic_write(root / rel, contents)
        if digest((root / SOURCE_REL).read_bytes()) != manifest['source_sha256']:
            raise ValueError('The parent demo file changed during installation.')
    except Exception:
        for rel, old in originals.items():
            if old is None:
                if (root / rel).exists(): (root / rel).unlink()
            else: atomic_write(root / rel, old)
        raise
    print('Information-Seeking demo: 12 cases = 4 students + 3 practitioners + 5 experts.')
    print('Recomputed totals: ' + json.dumps(summarize(subset), sort_keys=True))
    print('Trust Calibration retains 34 cases and 680 trials. Parent JSON and live study files were not modified.')
    print('Backup: ' + str(backup))
    print('Restart Streamlit, unlock the researcher workspace, then open Information Seeking > Illustrative demo results.')
    return backup


def rollback(root, backup):
    manifest = json.loads((backup / 'restore_manifest.json').read_text(encoding='utf-8'))
    if manifest.get('patch') != VERSION:
        raise ValueError('Unknown rollback manifest.')
    entries = manifest['files']
    if not entries or len(set(e['path'] for e in entries)) != len(entries):
        raise ValueError('Invalid rollback entries.')
    for e in entries:
        if e['path'] not in {DATA_REL, UI_REL}:
            raise ValueError('Unexpected rollback path.')
        path = root / e['path']
        if not path.is_file() or digest(path.read_bytes()) != e['installed_sha256']:
            raise ValueError('File was edited after installation; rollback stopped: ' + e['path'])
        if e['existed']:
            saved = backup / e['path']
            if not saved.is_file() or digest(saved.read_bytes()) != e['previous_sha256']:
                raise ValueError('Backup integrity check failed: ' + e['path'])
    for e in entries:
        if e['existed']: atomic_write(root / e['path'], (backup / e['path']).read_bytes())
        else: (root / e['path']).unlink()
    print('Information-Seeking demo restored. Parent demo and live study files remain unchanged.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path.cwd())
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--rollback', type=Path)
    args = parser.parse_args()
    try:
        root = args.root.resolve()
        if args.rollback:
            rollback(root, args.rollback.resolve())
        elif args.check:
            subset, changes = prepare(root)
            if changes: raise ValueError('The Information-Seeking 12-case patch is not fully installed or the parent source changed.')
            print('PASS: 12 cases; group counts 4 / 3 / 5; response/event counts and UI wiring verified.')
            print(json.dumps(summarize(subset), indent=2))
        else: install(root)
    except (ValueError, OSError, KeyError, TypeError, SyntaxError) as error:
        parser.exit(1, 'Update stopped: ' + str(error) + '\n')


if __name__ == '__main__':
    main()
