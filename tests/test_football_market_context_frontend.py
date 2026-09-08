"""Exercise the actual browser helpers against contradictory market snapshots."""
import re
import shutil
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(shutil.which("node"), "Node.js is needed to execute frontend helpers")
class FootballMarketContextTests(unittest.TestCase):
    def test_conflict_and_missing_evidence_are_not_reported_as_agreement(self):
        html = (ROOT / "web/index.html").read_text(encoding="utf-8")
        names = ("getFootballMarketContext", "getFootballTotalCandidate")
        functions = []
        for name in names:
            start = html.index("function " + name + "(")
            end = re.search(r"\n(?:async )?function ", html[start + 1:])
            self.assertIsNotNone(end)
            functions.append(html[start:start + 1 + end.start()])
        start = html.index('  function renderMatchItem(item) {')
        end = html.index("    if (footballSort === 'time')", start)
        functions.append(html[start:end])
        script = "\n".join(functions) + r"""
const assert = require('node:assert/strict');
const zeroMovement = {handicap_signal:0, asian_water_signal:0, euro_signal:0, conflict:false};
let view = getFootballMarketContext({
  anomaly:{euro_asian_deviation:{abs_deviation:.695}},
  model:{joint_market_state:zeroMovement}
});
assert.equal(view.label, '盘口强度分歧');
assert.equal(view.conflict, true);
assert.ok(view.detail.includes('0.69') || view.detail.includes('0.70'));
view = getFootballMarketContext({model:{joint_market_state:zeroMovement}});
assert.equal(view.label, '暂无明确走势信号');
assert.ok(view.detail.includes('暂无有效数据'));
assert.equal(getFootballMarketContext({}).label, '暂无明确走势信号');
view = getFootballMarketContext({model:{joint_market_state:{conflict:true}},
  anomaly:{euro_asian_deviation:{abs_deviation:.2}}});
assert.equal(view.label, '盘口走势存在分歧');
view = getFootballMarketContext({model:{joint_market_state:{handicap_signal:.3, euro_signal:.2, conflict:false}}});
assert.equal(view.label, '走势信号同向');
view = getFootballMarketContext({lottery:{accuracy_gate:{spf:{reasons:['欧赔与亚盘明显冲突']}}}});
assert.equal(view.label, '盘口强度分歧');
view = getFootballMarketContext({anomaly:{euro_asian_deviation:{abs_deviation:.01, fit_failed:true}}});
assert.equal(view.label, '盘口强度校验失败');
assert.ok(view.detail.includes('暂不放行推荐'));
assert.equal(getFootballTotalCandidate({candidate:'over',line:3.5}), '大3.5');
assert.equal(getFootballTotalCandidate({candidate:'under',line:2.5}), '小2.5');
assert.equal(getFootballTotalCandidate({candidate:'under',line:null}), '小球（盘口待确认）');
assert.equal(getFootballTotalCandidate({}), '-');
function esc(value) { return String(value); }
function uiIcon() { return ''; }
function renderFixtureTeams() { return ''; }
function getLotteryLinkedSelections() { return null; }
function getFootballTopScoreReferences() { return []; }
function renderProbabilityStrip() { return ''; }
function renderWebMarketRecommendation() { return ''; }
function renderMarketEvidence() { return ''; }
let tier = {tier:'abstain', label:'观望', probability:.798, prediction:'主胜', market:'胜平负',
  downgradeReasons:[], markets:{spf:{status:'abstain'}, rqspf:{status:'abstain'}, total_goals:{status:'abstain'}}};
function getFootballResultTier() { return tier; }
const item = {match:{home:'主队', away:'客队'}, result:{
  lottery:{standard:{prediction:'胜', probabilities:{'胜':.798, '平':.128, '负':.074}}},
  model:{joint_market_state:zeroMovement}, anomaly:{euro_asian_deviation:{abs_deviation:.695}},
  upset:{confident:true, favorite:'胜', favorite_prob:.798, gap:.67}
}};
let card = renderMatchItem(item);
assert.ok(card.includes('盘口强度分歧'));
assert.ok(card.includes('暂无通过筛选的推荐'));
assert.ok(!card.includes('胜平负 主胜 <b>80%</b>'));
assert.ok(!card.includes('倾向较集中'));
assert.ok(!card.includes('热门稳胆'));
assert.ok(!card.includes('真实冷门率'));
tier = {...tier, tier:'watch', label:'观察'};
card = renderMatchItem(item);
assert.ok(card.includes('暂无通过筛选的推荐'));
assert.ok(!card.includes('胜平负 主胜 <b>80%</b>'));
tier = {...tier, tier:'selected', markets:{...tier.markets, spf:{status:'selected'}}};
item.result.anomaly.euro_asian_deviation.abs_deviation = .1;
card = renderMatchItem(item);
assert.ok(card.includes('胜平负倾向较集中'));
assert.ok(card.includes('胜平负 主胜 <b>80%</b>'));
console.log('market context cases passed');
"""
        result = subprocess.run([shutil.which("node"), "-e", script],
                                capture_output=True, text=True, encoding="utf-8")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("market context cases passed", result.stdout)

    def test_market_reasons_remain_complete_when_evidence_is_deduplicated(self):
        html = (ROOT / "web/index.html").read_text(encoding="utf-8")
        functions = []
        for name in ("renderWebMarketRecommendation", "renderMarketEvidence"):
            start = html.index("function " + name + "(")
            end = re.search(r"\n(?:async )?function ", html[start + 1:])
            self.assertIsNotNone(end)
            functions.append(html[start:start + 1 + end.start()])
        script = "\n".join(functions) + r"""
const assert = require('node:assert/strict');
function esc(value) { return String(value); }
const reasons = ['样本不足', '盘口资料缺失', '走势存在分歧', '独立验证尚未完成'];
const validation = '样本外 70.0% / 100场';
const state = {status:'abstain', label:'观望', pick:'胜', reasons, validation};
for (const status of ['selected', 'watch', 'abstain']) {
  const row = renderWebMarketRecommendation('胜平负', {...state, status}, '胜', '概率分布');
  for (const reason of reasons) assert.ok(row.includes(reason), `${status} omitted ${reason}`);
  assert.ok(row.includes('概率分布'), `${status} hid probability distribution`);
  assert.ok(!row.includes('模型候选'));
  if (status === 'selected') assert.ok(row.includes('<strong>胜</strong>'));
  else {
    assert.ok(row.includes('<strong>不推荐</strong>'));
    assert.ok(!row.includes('<strong>胜</strong>'));
  }
}
for (const pick of ['让负', '让胜', '大2.5']) {
  const row = renderWebMarketRecommendation('玩法', state, pick, '概率分布');
  assert.ok(!row.includes(pick), `unselected candidate leaked: ${pick}`);
}
const missingStateRow = renderWebMarketRecommendation('胜平负', null, '胜');
assert.ok(missingStateRow.includes('<strong>不推荐</strong>'));
const displayedRow = renderWebMarketRecommendation('胜平负', state, '胜', '概率分布');
const displayedEvidence = renderMarketEvidence([['胜平负', state, true]]);
assert.ok(displayedEvidence.includes(validation), 'deduplication must preserve validation');
for (const reason of reasons) {
  assert.ok(!displayedEvidence.includes(reason), `already displayed reason repeated: ${reason}`);
  assert.equal((displayedRow + displayedEvidence).split(reason).length - 1, 1);
}
const missingState = {...state, reasons:['让球盘口不可用', '该玩法未开售'], validation:''};
for (const entry of [['让球胜平负', missingState, false], ['让球胜平负', missingState]]) {
  const evidence = renderMarketEvidence([entry]);
  for (const reason of missingState.reasons) {
    assert.ok(evidence.includes(reason), `unrendered market lost reason: ${reason}`);
  }
}
const mixedEvidence = renderMarketEvidence([
  ['胜平负', state, true], ['让球胜平负', {...missingState, validation}, false]
]);
assert.ok(mixedEvidence.includes(validation));
assert.ok(!mixedEvidence.includes(reasons[0]));
assert.ok(mixedEvidence.includes(missingState.reasons[0]));
console.log('market evidence completeness cases passed');
"""
        result = subprocess.run([shutil.which("node"), "-e", script],
                                capture_output=True, text=True, encoding="utf-8")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("market evidence completeness cases passed", result.stdout)

    def test_independently_validated_handicap_can_be_the_selected_market(self):
        html = (ROOT / "web/index.html").read_text(encoding="utf-8")
        functions = []
        for name in ("calculateFootballPredictionReliability", "accuracyGateReasonText",
                     "getFootballMarketContext", "getFootballTotalCandidate",
                     "formatAccuracyGateValidation", "getAccuracyGatePresentation",
                     "getFootballResultTier"):
            start = html.index("function " + name + "(")
            end = re.search(r"\n(?:async )?function ", html[start + 1:])
            self.assertIsNotNone(end)
            functions.append(html[start:start + 1 + end.start()])
        script = "\n".join(functions) + r"""
const assert = require('node:assert/strict');
const item = {result:{lottery:{
  standard:{probabilities:{'胜':.3, '平':.3, '负':.4}},
  accuracy_gate:{
    spf:{selected:false, candidate:'负', reasons:['概率不足']},
    total_goals:{selected:false},
    rqspf:{selected:true, candidate:'让胜', probability:.85, validation_status:'validated', reasons:[]}
  }
}}};
let tier = getFootballResultTier(item);
assert.equal(tier.tier, 'selected');
assert.equal(tier.market, '让球胜平负');
assert.equal(tier.prediction, '让胜');
assert.equal(tier.probability, .85);
item.result.lottery.accuracy_gate.rqspf.validation_status = 'pending_independent_validation';
tier = getFootballResultTier(item);
assert.equal(tier.tier, 'watch');
assert.equal(tier.markets.rqspf.status, 'watch');
item.result.lottery.accuracy_gate.rqspf.selected = false;
tier = getFootballResultTier(item);
assert.equal(tier.tier, 'abstain');
console.log('handicap selection presentation cases passed');
"""
        result = subprocess.run([shutil.which("node"), "-e", script],
                                capture_output=True, text=True, encoding="utf-8")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("handicap selection presentation cases passed", result.stdout)

    def test_expanded_cards_use_marginal_probabilities_and_gate_picks(self):
        html = (ROOT / "web/index.html").read_text(encoding="utf-8")
        start = html.index('  function renderMatchItem(item) {')
        end = html.index("    if (footballSort === 'time')", start)
        renderer = html[start:end].replace('const compactFootballView = true;',
                                           'const compactFootballView = false;')
        script = renderer + r"""
const assert = require('node:assert/strict');
function esc(value) { return String(value); }
function uiIcon() { return ''; }
function renderFixtureTeams() { return ''; }
function getLotteryHandicapExplanation() { return ''; }
function getLotteryLinkedSelections() {
  return {primaryPrediction:'胜', standardPredictions:['胜'], handicapPredictions:['让平'],
    handicapConditionalProbabilities:{'让胜':0, '让平':1, '让负':0}};
}
const states = {spf:{status:'abstain', pick:'胜'}, rqspf:{status:'selected', pick:'让胜'},
  total_goals:{status:'abstain'}};
function getFootballResultTier() { return {tier:'selected', label:'研究精选',
  market:'让球胜平负', prediction:'让胜', probability:.85, downgradeReasons:[], markets:states}; }
const item = {match:{home:'主队',away:'客队'}, result:{lottery:{
  standard:{prediction:'胜', probabilities:{'胜':.6,'平':.2,'负':.2}},
  handicap:{handicap:-1, prediction:'让负', probabilities:{'让胜':.3,'让平':.25,'让负':.45}}
}}};
let card = renderMatchItem(item);
assert.ok(card.includes('30%'));
assert.ok(card.includes('25%'));
assert.ok(card.includes('45%'));
assert.ok(!card.includes('100%'), 'conditional certainty must never replace marginal probability');
assert.ok(!card.includes('⇒ 首选'));
// Outcome markup has whitespace after its opening element.
const highlighted = [...card.matchAll(/font-weight:bold">\s*<div>([^<]+)<\/div>/g)].map(m => m[1]);
assert.deepEqual(highlighted, ['让胜'], 'highlight must follow selected gate.pick');
states.rqspf.status = 'watch';
card = renderMatchItem(item);
assert.ok(!/font-weight:bold">\s*<div>让/.test(card));
console.log('expanded market probability cases passed');
"""
        result = subprocess.run([shutil.which("node"), "-e", script],
                                capture_output=True, text=True, encoding="utf-8")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("expanded market probability cases passed", result.stdout)

    def test_detail_page_does_not_promote_unvalidated_markets_to_recommendations(self):
        html = (ROOT / "web/index.html").read_text(encoding="utf-8")
        start = html.index('function renderResult(r) {')
        end = html.index('function switchTab(tab)', start)
        script = html[start:end] + r"""
const assert = require('node:assert/strict');
const app = {innerHTML:''};
function esc(value) { return String(value || ''); }
function pct(value) { return (Number(value || 0) * 100).toFixed(1) + '%'; }
function heatLabel() { return ''; }
function backBtn() { return ''; }
function bindBack() {}
function renderModelStatus() { return ''; }
function renderModelWeights() { return ''; }
function renderCalibrationEffect() { return ''; }
function renderSimilarMarketDetail() { return ''; }
function probBar() { return ''; }
function getGoalDistributionFromCandidates() {
  return [{goals:3,label:'3球',probability:.6},{goals:2,label:'2球',probability:.4}];
}
let tier = 'abstain';
function getFootballResultTier() { return {tier, label:tier === 'selected' ? '研究精选' : '观望',
  probability:.6, predictionReliability:.5, market:'胜平负', prediction:'胜'}; }
const score = {home:2, away:1, prob:.12, recommend_score:.9};
const data = {match:{home:'主队',away:'客队'}, asian:{source:'model_proxy'},
  euro:{close:{home:.6,draw:.2,away:.2},changes:[]}, total:{close_prob:{over:.6,under:.4}},
  confidence:{score:.6,level:'low'},
  model:{top_scores:[score],recommend:[score],candidates:[[[2,1],.6],[[1,1],.4]],
    half_full_time:{probs:[{code:'HH',name:'半胜全胜',probability:35}]}}
};
for (tier of ['abstain', 'watch', 'selected']) {
  renderResult(data);
  const detail = app.innerHTML;
  assert.ok(detail.includes('比分参考（非推荐）'));
  assert.ok(detail.includes('比分概率 12.0%'));
  assert.ok(detail.includes('半全场参考（非推荐）'));
  assert.ok(detail.includes('进球数参考（非推荐）'));
  assert.ok(detail.includes('半场概率参考'));
  assert.ok(!detail.includes('最终策略推荐'));
  assert.ok(!detail.includes('第1推荐'));
  assert.ok(!detail.includes('class="star"'));
  assert.ok(!detail.includes('半全场推荐'));
  assert.ok(!detail.includes('进球数推荐'));
  if (tier === 'selected') assert.ok(detail.includes('胜平负 胜'));
  else {
    assert.ok(detail.includes('暂无通过筛选的推荐'));
    assert.ok(!detail.includes('胜平负 胜'));
  }
}
console.log('detail recommendation presentation cases passed');
"""
        result = subprocess.run([shutil.which("node"), "-"], input=script,
                                capture_output=True, text=True, encoding="utf-8")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("detail recommendation presentation cases passed", result.stdout)
