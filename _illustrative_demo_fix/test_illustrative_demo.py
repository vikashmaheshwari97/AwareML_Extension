import ast,importlib.util,json,tempfile,unittest
from pathlib import Path
HERE=Path(__file__).resolve().parent

def load(name,path):
 s=importlib.util.spec_from_file_location(name,str(path));m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
installer=load('demo_installer',HERE/'APPLY_ILLUSTRATIVE_DEMO.py')
ui=load('demo_ui',HERE/'payload/awareml/ui_v2/illustrative_study_results.py')
DATA=HERE/'payload/data/illustrative_demo/awareml_34_cases.json'
class DemoTests(unittest.TestCase):
 def test_data_and_adapters(self):
  raw=ui.load_demo(DATA);study,summary=ui.trust_adapter(raw)
  rows=study.store.response_rows('demo');self.assertEqual(len(rows),680);self.assertEqual(len(summary['pair_coverage']),20)
  self.assertTrue(all(p['correct']==p['incorrect']==17 for p in summary['pair_coverage']))
  events,sessions,analysis=ui.information_adapter(raw);self.assertEqual(len(sessions),34)
  self.assertTrue(sessions['total_time_to_final_decision_sec'].isna().all());self.assertTrue(events['created_at'].isna().all())
  self.assertEqual(analysis['manual_theme_counts'],{});self.assertEqual(sessions['session_hash'].nunique(),34)
  self.assertAlmostEqual(sum(p['mean_trust_correct'] for p in raw['participants'])/34,6.627450980392156)
 def test_real_data_rejected(self):
  raw=ui.load_demo(DATA);raw['participants'][0]['is_synthetic']=False
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'data.json';p.write_text(json.dumps(raw))
   with self.assertRaises(ValueError):ui.load_demo(p)
 def test_install_idempotent_and_rollback(self):
  source='import streamlit as st\n\ndef trust_calibration_v2_page():\n    from .phase16_morten_v2 import phase16_trust_calibration_page_v2\n    return phase16_trust_calibration_page_v2()\n\ndef information_seeking_v2_page():\n    from .phase17_morten_v2 import information_seeking_research_page_v2\n    return information_seeking_research_page_v2()\n'
  self.assertEqual(installer.hook(installer.hook(source)),installer.hook(source))
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);target=root/'awareml/ui_v2/pages_specialist.py';target.parent.mkdir(parents=True);target.write_text(source)
   sentinel=root/'artifacts/study.db';sentinel.parent.mkdir();sentinel.write_bytes(b'real study sentinel')
   installer.install(root);first=target.read_bytes();installer.install(root);self.assertEqual(first,target.read_bytes());self.assertEqual(sentinel.read_bytes(),b'real study sentinel')
   backup=next((root/'artifacts/illustrative_demo_backups').iterdir());installer.rollback(root,backup);self.assertEqual(target.read_text(),source);self.assertEqual(sentinel.read_bytes(),b'real study sentinel')
 def test_unknown_source_rejected(self):
  with self.assertRaises(ValueError):installer.hook('def trust_calibration_v2_page():\n    print("unknown")\n')
 def test_streamlit_chart_rendering(self):
  from streamlit.testing.v1 import AppTest
  module=HERE/'payload/awareml/ui_v2/illustrative_study_results.py'
  code="import importlib.util\nfrom pathlib import Path\ns=importlib.util.spec_from_file_location('demo_test_ui',"+repr(str(module))+" )\nm=importlib.util.module_from_spec(s)\ns.loader.exec_module(m)\nm.render_trust_demo()\nm.render_information_demo()\n"
  at=AppTest.from_string(code)
  at.session_state['p16_task_context_r1_researcher_unlocked']=True
  at.run(timeout=60)
  self.assertEqual(len(at.exception),0)
  self.assertGreaterEqual(len(at.get('plotly_chart')),35)
  self.assertEqual(len(at.metric),8)
 def test_locked_demo_has_no_charts(self):
  from streamlit.testing.v1 import AppTest
  module=HERE/'payload/awareml/ui_v2/illustrative_study_results.py'
  code="import importlib.util\ns=importlib.util.spec_from_file_location('locked_demo',"+repr(str(module))+" )\nm=importlib.util.module_from_spec(s)\ns.loader.exec_module(m)\nm.render_trust_demo()\nm.render_information_demo()\n"
  at=AppTest.from_string(code).run(timeout=60)
  self.assertEqual(len(at.exception),0)
  self.assertEqual(len(at.get('plotly_chart')),0)
  self.assertEqual(len(at.metric),0)
 def test_no_store_imports(self):
  source=(HERE/'payload/awareml/ui_v2/illustrative_study_results.py').read_text();tree=ast.parse(source)
  for n in ast.walk(tree):
   if isinstance(n,ast.ImportFrom):self.assertFalse((n.module or '').startswith('awareml.studies'))
if __name__=='__main__':unittest.main()
