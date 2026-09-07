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
assert.ok(!card.includes('倾向较集中'));
assert.ok(!card.includes('热门稳胆'));
assert.ok(!card.includes('真实冷门率'));
tier = {...tier, tier:'selected', markets:{...tier.markets, spf:{status:'selected'}}};
item.result.anomaly.euro_asian_deviation.abs_deviation = .1;
card = renderMatchItem(item);
assert.ok(card.includes('胜平负倾向较集中'));
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
}
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
