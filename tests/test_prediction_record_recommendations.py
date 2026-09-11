import re
import shutil
import subprocess
import unittest
from pathlib import Path
from unittest.mock import patch


class RecordRecommendations(unittest.TestCase):
    def test_legacy_record_restores_the_exact_recommendation_list_analysis(self):
        from copy import deepcopy
        from src.football import result_sync
        from src.domain.sports.football.lottery import lottery_market_probabilities
        for line, scores in [(1, {'1-0': .219, '1-1': .241, '0-1': .24, '0-2': .30}),
                             (-1, {'1-0': .24, '2-0': .30, '1-1': .241, '0-1': .219})]:
            with self.subTest(line=line):
                odds = {'offer_matched': True, 'spf_available': True, 'rqspf_available': True,
                        'spf_odds': {'胜': 4.5, '平': 3.6, '负': 1.8} if line > 0
                        else {'胜': 1.8, '平': 3.6, '负': 4.5},
                        'rqspf_odds': {'让胜': 2.1, '让平': 3.5, '让负': 3.0}}
                candidates = [(tuple(map(int, score.split('-'))), p) for score, p in scores.items()]
                expected = lottery_market_probabilities(candidates, line,
                    spf_odds=odds['spf_odds'], rqspf_odds=odds['rqspf_odds'])
                record = {'match_id': 'legacy', 'predicted_scores': scores,
                          'predicted_1x2': {'H': .219, 'D': .241, 'A': .54},
                          'predicted_rqspf': expected['handicap']['probabilities'],
                          'lottery_handicap': line, 'odds_snapshot': {'lottery': odds}}
                original = deepcopy(record)
                with patch.object(result_sync._global_history, 'records', [record]):
                    row = result_sync.get_prediction_records(include_hidden=True)[0]
                self.assertTrue(row['lottery_direction_analysis']['available'])
                self.assertEqual(row['lottery_direction_analysis'], expected['direction_analysis'])
                self.assertEqual(row['lottery_analysis']['standard'], expected['standard'])
                self.assertEqual(row['lottery_analysis']['handicap'], expected['handicap'])
                self.assertEqual(record, original)
                # A modern snapshot must keep its original market weights and probabilities.
                record['odds_snapshot']['lottery'].update(expected)
                self.assertEqual(result_sync._record_lottery_analysis(record, record['odds_snapshot']['lottery'], line)['direction_analysis'], expected['direction_analysis'])

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

    def test_api_exposes_saved_direction_without_changing_market_probabilities(self):
        from src.football import result_sync
        direction = {'available': True, 'conditional_probabilities': {'让胜': 0, '让平': .4, '让负': .6}}
        record = {'match_id': 'saved-direction', 'predicted_1x2': {'H': .2, 'D': .25, 'A': .55},
                  'predicted_rqspf': {'让胜': .5, '让平': .2, '让负': .3},
                  'lottery_handicap': 1, 'odds_snapshot': {'lottery': {
                      'offer_matched': True, 'spf_available': True, 'spf_odds': {'胜': 4},
                      'rqspf_available': True, 'rqspf_odds': {'让胜': 2},
                      'direction_analysis': direction}}}
        with patch.object(result_sync._global_history, 'records', [record]):
            row = result_sync.get_prediction_records(include_hidden=True)[0]
        self.assertEqual(row['lottery_direction_analysis'], direction)
        self.assertIsNot(row['lottery_direction_analysis']['conditional_probabilities'], direction['conditional_probabilities'])
        self.assertEqual(row['predicted_rqspf'], record['predicted_rqspf'])

    @unittest.skipUnless(shutil.which('node'), 'Node required')
    def test_record_uses_the_same_recommendation_states_and_renderer(self):
        html = Path('web/index.html').read_text(encoding='utf-8')
        names = ('calculateFootballPredictionReliability', 'accuracyGateReasonText',
                 'getFootballMarketContext', 'getFootballTotalCandidate',
                 'formatAccuracyGateValidation', 'getAccuracyGatePresentation',
                 'footballMarketDirectionsCompatible', 'getFootballResultTier',
                 'renderWebMarketRecommendation', 'renderProbabilityStrip',
                 'getFootballDirectionState', 'renderFootballHandicapProbabilities',
                 'renderFootballDirectionAnalysis',
                 'predictionRecordActualScore', 'renderPredictionMarketOutcome',
                 'renderPredictionScoreReference',
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
for (const [pick, line, probs, conditional, compatible, incompatible] of [
  ['负', 1, {H:.219,D:.241,A:.54}, {'让胜':0,'让平':.4,'让负':.6}, ['让平','让负'], ['让胜']],
  ['胜', -1, {H:.54,D:.241,A:.219}, {'让胜':.6,'让平':.4,'让负':0}, ['让胜','让平'], ['让负']],
]) {
  const lottery = {standard:{prediction:pick, probabilities:pick === '负' ? {'胜':.219,'平':.241,'负':.54} : {'胜':.54,'平':.241,'负':.219}},
    handicap:{handicap:line, probabilities:record.predicted_rqspf},
    direction_analysis:{available:true, standard_prediction:pick, standard_probability:.54,
      handicap:line, probability_basis:'conditional_on_standard_result',
      conditional_probabilities:conditional,
      joint_probabilities:Object.fromEntries(Object.entries(conditional).map(([k,v]) => [k,v*.54])),
      compatible_handicap_predictions:compatible, incompatible_handicap_predictions:incompatible}};
  const saved = {...record, predicted_1x2:probs, lottery_handicap:line,
    lottery_direction_analysis:lottery.direction_analysis};
  out = renderPredictionRecordMarkets(saved);
  assert(out.includes(renderFootballHandicapProbabilities(lottery, true)));
  assert(out.includes(renderFootballDirectionAnalysis(lottery, true)));
  assert(out.includes('该情景不成立'));
  assert(!out.includes(`${pick}＋${incompatible[0]}`));
  const modelDiffers = {...saved, predicted_1x2:{H:.8,D:.1,A:.1},
    lottery_analysis:{standard:lottery.standard, handicap:lottery.handicap}};
  assert(renderPredictionRecordMarkets(modelDiffers).includes(renderFootballHandicapProbabilities(lottery, true)));
  delete saved.lottery_direction_analysis;
  out = renderPredictionRecordMarkets(saved);
  assert(!out.includes('data-probability-basis="conditional_on_standard_result"'));
  assert(out.includes('全场概率参考'));
}
const barca = {settled:true, actual_score:'5-1', lottery_handicap:-3,
  predicted_scores:{'4-0':.106,'3-0':.087,'4-1':.068}};
let outcome = renderPredictionMarketOutcome(barca, 'rqspf', {status:'abstain'},
  {'让胜':.410,'让平':.199,'让负':.391});
assert(outcome.includes('模型参考（非推荐）：让胜'));
assert(outcome.includes('✅ 命中'));
assert(!outcome.includes('推荐结果：'));
outcome = renderPredictionScoreReference(barca);
assert(outcome.includes('Top1：❌ 未命中') && outcome.includes('Top3：❌ 未命中'));
const stuttgart = {settled:true, actual_score:'3-1', lottery_handicap:-2,
  predicted_scores:{'3-0':.095,'4-0':.088,'3-1':.075}};
outcome = renderPredictionMarketOutcome(stuttgart, 'rqspf', {status:'abstain'},
  {'让胜':.405,'让平':.213,'让负':.382});
assert(outcome.includes('❌ 未命中') && outcome.includes('实际让平'));
outcome = renderPredictionScoreReference(stuttgart);
assert(outcome.includes('Top1：❌ 未命中') && outcome.includes('Top3：✅ 命中（第3位）'));
outcome = renderPredictionMarketOutcome(stuttgart, 'rqspf', {status:'selected',pick:'让平'},
  {'让胜':.405,'让平':.213,'让负':.382});
assert(outcome.includes('推荐结果：让平') && outcome.includes('✅ 命中'));
assert.equal(renderPredictionMarketOutcome({...barca,settled:false}, 'rqspf', {status:'abstain'}, {}), '');
assert(!renderPredictionScoreReference({...barca,settled:false}).includes('命中'));
assert(renderPredictionMarketOutcome({...barca,lottery_handicap:null}, 'rqspf',
  {status:'selected',pick:'让胜'}, {}).includes('暂无法判定'));
assert(renderPredictionScoreReference({...barca,actual_score:null}).includes('暂无法判定'));
'''
        result = subprocess.run([shutil.which('node'), '-'], input=script,
                                text=True, encoding='utf-8', capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
