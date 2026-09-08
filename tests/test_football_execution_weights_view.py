"""The existing deep-report card must not call nested pipeline steps unused."""
import json
from pathlib import Path
import shutil
import subprocess
import unittest


@unittest.skipUnless(shutil.which('node'), 'Node is needed to execute the actual renderer')
class ExecutionWeightViewTests(unittest.TestCase):
    def test_final_stage_weights_keep_five_rows_and_report_actual_execution(self):
        source = Path('web/index.html').read_text(encoding='utf-8')
        start = source.index('function renderModelWeights(')
        end = source.index('function renderCalibrationEffect(', start)
        script = "const esc = x => String(x); const pct = x => (x*100)+'%';\n" + source[start:end]
        script += "\nconsole.log(JSON.stringify([renderModelWeights({scope:'final_stage_mixture',existing_pipeline:.95,ml:.05}),renderModelWeights({scope:'final_stage_mixture',existing_pipeline:1,ml:0})]));"
        completed = subprocess.run(['node', '-e', script], text=True, encoding='utf-8', capture_output=True, check=True)
        applied, fallback = json.loads(completed.stdout)
        self.assertEqual(applied.count('class="zhixuan"'), 5)
        self.assertIn('95%', applied)
        self.assertIn('5%', applied)
        self.assertIn('实际最终融合', applied)
        self.assertNotIn('未参与', applied)
        self.assertIn('100%', fallback)
        self.assertEqual(fallback.count('未参与'), 1)
