#!/usr/bin/env python3
from pathlib import Path
import shutil


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    package_root = Path(__file__).resolve().parents[1]

    src = package_root / 'awareml' / 'ui_v2' / 'phase16_research_visuals.py'
    dst = repo / 'awareml' / 'ui_v2' / 'phase16_research_visuals.py'
    test_src = package_root / 'tests' / 'test_phase16_research_visuals_v2.py'
    test_dst = repo / 'tests' / 'test_phase16_research_visuals_v2.py'
    doc_src = package_root / 'docs' / 'PHASE16_RESEARCHER_VISUALS_V2.md'
    doc_dst = repo / 'docs' / 'PHASE16_RESEARCHER_VISUALS_V2.md'
    readme_src = package_root / 'README_PHASE16_RESEARCHER_VISUALS_V2.md'
    readme_dst = repo / 'README_PHASE16_RESEARCHER_VISUALS_V2.md'

    for path in [src, test_src, doc_src, readme_src]:
        if not path.exists():
            raise SystemExit(f'Missing expected package file: {path}')

    dst.parent.mkdir(parents=True, exist_ok=True)
    test_dst.parent.mkdir(parents=True, exist_ok=True)
    doc_dst.parent.mkdir(parents=True, exist_ok=True)

    shutil.copy2(src, dst)
    shutil.copy2(test_src, test_dst)
    shutil.copy2(doc_src, doc_dst)
    shutil.copy2(readme_src, readme_dst)

    print('[copied] awareml/ui_v2/phase16_research_visuals.py')
    print('[copied] tests/test_phase16_research_visuals_v2.py')
    print('[copied] docs/PHASE16_RESEARCHER_VISUALS_V2.md')
    print('[copied] README_PHASE16_RESEARCHER_VISUALS_V2.md')
    print('')
    print('Phase-16 Researcher Workspace visual redesign v2 applied.')
    print('This refreshes chart styling, colors, label layout, and readability only.')
    print('It does not change study logic, frozen stimuli, analysis formulas, or collected responses.')
    print('')
    print('Suggested verification:')
    print('  python -m pytest -q .\\tests\\test_phase16_research_visuals_v2.py')
    print('  streamlit run app.py')


if __name__ == '__main__':
    main()
