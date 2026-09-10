import re
import shutil
import subprocess
import unittest
from pathlib import Path
from unittest.mock import patch


class RecordRecommendations(unittest.TestCase):
    def test_api_returns_saved_gates_without_reconstructing_them(self):
        from src.football import result_sync
        snapshot = {'accuracy_gate': {'spf': {'selected': False, 'candidate': '胜',
                                              'reasons': ['历史筛选未通过']}},
                    'decision_gate': {'official_bet_allowed': False}}
        record = {'match_id': 'saved-gate', 'professional_snapshot': snapshot,
                  'predicted_1x2': {'H': .7, 'D': .2, 'A': .1}}
        with patch.object(result_sync._global_history, 'records', [record]):
            row = result_sync.get_prediction_records(include_hidden=True)[0]
        self.assertEqual(row['recommendation_snapshot'], snapshot)
        self.assertIsNot(row['recommendation_snapshot']['accuracy_gate'], snapshot['accuracy_gate'])

    @unittest.skipUnless(shutil.which('node'), 'Node required')
    def test_record_uses_the_same_recommendation_states_and_renderer(self):
        html = Path('web/index.html').read_text(encoding='utf-8')
        names = ('calculateFootballPredictionReliability', 'accuracyGateReasonText',
                 'getFootballMarketContext', 'getFootballTotalCandidate',
                 'formatAccuracyGateValidation', 'getAccuracyGatePresentation',
                 'footballMarketDirectionsCompatible', 'getFootballResultTier',
                 'renderWebMarketRecommendation', 'renderProbabilityStrip',
                 'renderPredictionRecordMarkets')
        functions = []
        for name in names:
            start = html.index('function ' + name + '(')
            end = re.search(r'\n(?:async )?function ', html[start + 1:])
            functions.append(html[start:start + 1 + end.start()])
        script = '\n'.join(functions) + r'''
const assert = require('node:assert/strict');
function esc(v) { return String(v ?? ''); }
const record = {predicted_1x2:{H:.55,D:.25,A:.2},
  predicted_rqspf:{'让胜':.3,'让平':.2,'让负':.5}, lottery_handicap:-1,
  recommendation_snapshot:{accuracy_gate:{
    spf:{selected:false,candidate:'胜',reasons:['未通过历史筛选']},
    rqspf:{selected:false,candidate:'让负',reasons:['优势不足']}}}};
let out = renderPredictionRecordMarkets(record);
assert(!out.includes('<strong>胜</strong>'));
assert(!out.includes('<strong>让负</strong>'));
assert(out.includes('不推荐') && out.includes('未通过历史筛选'));
assert(out.includes('55.0%'));
record.recommendation_snapshot.accuracy_gate.spf.selected = true;
record.recommendation_snapshot.accuracy_gate.rqspf.selected = true;
out = renderPredictionRecordMarkets(record);
assert(!out.includes('<strong>胜</strong>'));
assert(!out.includes('<strong>让负</strong>'));
record.lottery_handicap = -2;
out = renderPredictionRecordMarkets(record);
assert(out.includes('<strong>胜</strong>') && out.includes('<strong>让负</strong>'));
assert(out.includes('研究精选'));
record.predicted_1x2 = {H:.2,D:.25,A:.55};
record.predicted_rqspf = {'让胜':.5,'让平':.2,'让负':.3};
record.recommendation_snapshot.accuracy_gate.spf.candidate = '负';
record.recommendation_snapshot.accuracy_gate.rqspf.candidate = '让胜';
record.lottery_handicap = 1;
out = renderPredictionRecordMarkets(record);
assert(!out.includes('<strong>负</strong>') && !out.includes('<strong>让胜</strong>'));
record.lottery_handicap = 2;
out = renderPredictionRecordMarkets(record);
assert(out.includes('<strong>负</strong>') && out.includes('<strong>让胜</strong>'));
record.recommendation_snapshot.accuracy_gate.rqspf.validation_status = 'pending_independent_validation';
out = renderPredictionRecordMarkets(record);
assert(!out.includes('<strong>让胜</strong>'));
delete record.recommendation_snapshot;
out = renderPredictionRecordMarkets(record);
assert(out.includes('未保存当时的推荐筛选结果'));
assert(!out.includes('<strong>胜</strong>'));
'''
        result = subprocess.run([shutil.which('node'), '-'], input=script,
                                text=True, encoding='utf-8', capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
