"""Keep full-match percentages visible and conditional analysis optional."""
import re
import shutil
import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(shutil.which("node"), "Node.js is needed to execute frontend helpers")
class FootballDirectionFrontendTests(unittest.TestCase):
    def run_renderer(self, assertions):
        html = (ROOT / "web/index.html").read_text(encoding="utf-8")
        functions = []
        for name in ("getFootballMarketContext", "getFootballTotalCandidate",
                     "renderProbabilityStrip", "renderWebMarketRecommendation",
                     "renderFootballDirectionAnalysis",
                     "renderMarketEvidence"):
            start = html.index("function " + name + "(")
            end = re.search(r"\n(?:async )?function ", html[start + 1:])
            functions.append(html[start:start + 1 + end.start()])
        start = html.index('  function renderMatchItem(item) {')
        end = html.index("    if (footballSort === 'time')", start)
        functions.append(html[start:end])
        script = "\n".join(functions) + r"""
const assert = require('node:assert/strict');
function esc(value) { return String(value ?? ''); }
function uiIcon() { return ''; }
function renderFixtureTeams() { return ''; }
function getLotteryLinkedSelections() { return null; }
function getLotteryHandicapExplanation() { return ''; }
function getFootballTopScoreReferences() { return []; }
function getFootballResultTier(item) {
  return {tier:'abstain', label:'观望', probability:0, downgradeReasons:[], markets:{
    spf:{status:'abstain', label:'观望', reasons:['模型概率不足']},
    rqspf:{status:'abstain', label:'观望', reasons:['独立验证尚未完成']},
    total_goals:{status:'abstain', label:'观望', reasons:[]}
  }};
}
function makeItem(standard, probability, handicap, conditional, compatible, incompatible) {
  return {match:{home:'主队', away:'客队'}, result:{lottery:{
    standard:{prediction:standard, probabilities:standard === '负'
      ? {'胜':.276, '平':.253, '负':probability}
      : standard === '平' ? {'胜':.3, '平':probability, '负':.3}
      : {'胜':probability, '平':.236, '负':.218}},
    handicap:{handicap, prediction:handicap > 0 ? '让胜' : '让负', probabilities:handicap > 0
      ? {'让胜':.523, '让平':.222, '让负':.256}
      : {'让胜':.316, '让平':.238, '让负':.446}},
    direction_analysis:{available:true, standard_prediction:standard, standard_probability:probability,
      handicap, conditional_probabilities:conditional,
      joint_probabilities:Object.fromEntries(Object.entries(conditional).map(([key,value]) => [key, value * probability])),
      compatible_handicap_predictions:compatible, incompatible_handicap_predictions:incompatible,
      probability_basis:'conditional_on_standard_result'}
  }}};
}
function parts(card) {
  const match = card.match(/<details class="football-direction-reference">[\s\S]*?<\/details>/);
  assert.ok(match, 'conditional scenario analysis belongs in an optional collapsed reference');
  assert.ok(match[0].includes('<summary>胜平负与让球的对应分析</summary>'));
  assert.ok(!match[0].slice(0, match[0].indexOf('>')).includes('open'));
  return {reference:match[0], main:card.replace(match[0], '')};
}
""" + assertions
        result = subprocess.run([shutil.which("node"), "-"], input=script,
                                capture_output=True, text=True, encoding="utf-8")
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_screenshot_away_win_uses_only_compatible_plus_one_branches(self):
        self.run_renderer(r"""
const item = makeItem('负', .471, 1, {'让胜':0, '让平':.45, '让负':.55}, ['让平','让负'], ['让胜']);
const card = renderMatchItem(item);
const {main, reference} = parts(card);
assert.ok(reference.includes('客胜（负） 47.1%'));
assert.ok(reference.includes('假设客胜成立 · 主队让球 +1'));
assert.ok(reference.includes('条件概率 55.0%'));
assert.ok(reference.includes('负＋让负 · 联合估计 25.9%'));
assert.ok(reference.includes('让平 · 客队赢1球'));
assert.ok(reference.includes('让负 · 客队赢至少2球'));
assert.ok(reference.includes('让胜：该情景下不可能'));
assert.ok(!reference.includes('data-handicap-result="让胜"'));
assert.ok(!reference.includes('负＋让胜'));
for (const percentage of ['27.6%', '25.3%', '47.1%', '52.3%', '22.2%', '25.6%']) {
  assert.ok(main.includes(`<strong>${percentage}</strong>`), `${percentage} requires expansion`);
}
assert.ok(!main.includes('模型候选'));
assert.ok(!main.includes('class="market-candidate"'));
assert.ok(!main.includes('条件概率'));
assert.ok(!main.includes('football-marginal-reference'));
assert.ok(card.includes('暂无通过筛选的推荐'));
assert.ok(card.includes('模型概率不足'));
assert.ok(card.includes('独立验证尚未完成'));
assert.ok(main.includes('<h3>玩法分析</h3><span>赛前概率参考</span>'));
assert.ok(!main.includes('比赛方向分析'));
assert.ok(card.indexOf('aria-label="大小球"') < card.indexOf('football-direction-reference'));
""")

    def test_screenshot_home_win_uses_only_compatible_minus_one_branches(self):
        self.run_renderer(r"""
const item = makeItem('胜', .546, -1, {'让胜':.57, '让平':.43, '让负':0}, ['让胜','让平'], ['让负']);
const {main, reference} = parts(renderMatchItem(item));
assert.ok(reference.includes('主胜（胜） 54.6%'));
assert.ok(reference.includes('假设主胜成立 · 主队让球 -1'));
assert.ok(reference.includes('条件概率 57.0%'));
assert.ok(reference.includes('胜＋让胜 · 联合估计 31.1%'));
assert.ok(reference.includes('让平 · 主队赢1球'));
assert.ok(reference.includes('让胜 · 主队赢至少2球'));
assert.ok(reference.includes('让负：该情景下不可能'));
assert.ok(!reference.includes('data-handicap-result="让负"'));
assert.ok(!reference.includes('胜＋让负'));
for (const percentage of ['54.6%', '23.6%', '21.8%', '31.6%', '23.8%', '44.6%']) {
  assert.ok(main.includes(`<strong>${percentage}</strong>`), `${percentage} requires expansion`);
}
assert.ok(!main.includes('is-pick'), 'watch status must not highlight a candidate');
""")

    def test_conditional_certainty_is_not_promoted_to_full_match_probability(self):
        self.run_renderer(r"""
const item = makeItem('平', .4, 1, {'让胜':1, '让平':0, '让负':0}, ['让胜'], ['让平','让负']);
const {main, reference} = parts(renderMatchItem(item));
assert.ok(reference.includes('平局（平） 40.0%'));
assert.ok(reference.includes('假设平局成立'));
assert.ok(reference.includes('让胜 · 双方打平'));
assert.ok(reference.includes('条件概率 100.0%'));
assert.ok(reference.includes('平＋让胜 · 联合估计 40.0%'));
assert.ok(reference.includes('条件概率 100% 也不代表整场必然命中'));
assert.ok(!main.includes('100.0%'));
assert.ok(main.includes('52.3%'), 'true marginal probability stays visible without expansion');
assert.ok(main.includes('<strong>40.0%</strong>'));
assert.ok(!main.includes('fixture-pick">让球胜平负'));
""")

    def test_missing_old_or_invalid_joint_analysis_never_falls_back_to_marginal_maximum(self):
        self.run_renderer(r"""
const original = makeItem('负', .471, 1, {'让胜':0, '让平':.45, '让负':.55}, ['让平','让负'], ['让胜']);
for (const value of [undefined, {available:false,reasons:['比分样本不完整']},
    {...original.result.lottery.direction_analysis, probability_basis:'marginal'},
    {...original.result.lottery.direction_analysis, compatible_handicap_predictions:['让胜','让平','让负'], incompatible_handicap_predictions:[]},
    {...original.result.lottery.direction_analysis, joint_probabilities:{'让胜':0,'让平':.8,'让负':.9}},
    {...original.result.lottery.direction_analysis, conditional_probabilities:{'让胜':null,'让平':.45,'让负':.55}}]) {
  const item = JSON.parse(JSON.stringify(original));
  item.result.lottery.direction_analysis = value;
  item.result.lottery.linked_recommendation = {handicap_prediction:'让胜',standard_prediction:'负'};
  const {main, reference} = parts(renderMatchItem(item));
  assert.ok(reference.includes('待重新分析'));
  assert.ok(!reference.includes('data-handicap-result'));
  assert.ok(!reference.includes('47.1%'));
  assert.ok(!reference.includes('52.3%'));
  assert.ok(main.includes('47.1%'));
  assert.ok(main.includes('52.3%'));
}
""")

    def test_unsold_standard_market_does_not_create_a_joint_direction(self):
        self.run_renderer(r"""
const item = makeItem('负', .471, 1, {'让胜':0, '让平':.45, '让负':.55}, ['让平','让负'], ['让胜']);
Object.assign(item.result.lottery, {offer_matched:true, spf_available:false, rqspf_available:true});
const {main, reference} = parts(renderMatchItem(item));
assert.ok(main.includes('竞彩胜平负未开售'));
assert.ok(main.includes('52.3%'));
assert.ok(!main.includes('假设客胜成立'));
assert.ok(!main.includes('客胜（负） 47.1%'));
assert.ok(reference.includes('竞彩胜平负未开售'));
assert.ok(!reference.includes('假设客胜成立'));
assert.ok(!main.includes('aria-label="胜平负"'), 'unsold standard probabilities must not be presented as available');
""")

    def test_handicap_margin_explanations_work_beyond_one_goal(self):
        self.run_renderer(r"""
for (const [standard, handicap] of [['胜',-2], ['负',2], ['胜',-3], ['负',3]]) {
  const item = makeItem(standard, standard === '胜' ? .546 : .471, handicap,
    {'让胜':.4, '让平':.3, '让负':.3}, ['让胜','让平','让负'], []);
  const {reference} = parts(renderMatchItem(item));
  const team = standard === '胜' ? '主队' : '客队';
  const line = Math.abs(handicap);
  assert.ok(reference.includes(`让平 · ${team}赢${line}球`));
  assert.ok(reference.includes(`${team}赢至少${line + 1}球`));
  assert.ok(reference.includes(line === 2 ? `${team}赢1球` : `${team}赢1至2球`));
}
""")

    def test_mixed_snapshot_anchor_or_handicap_is_rejected(self):
        self.run_renderer(r"""
const original = makeItem('负', .471, 1, {'让胜':0, '让平':.45, '让负':.55}, ['让平','让负'], ['让胜']);
for (const mutate of [
  lottery => { lottery.handicap.handicap = -1; },
  lottery => { lottery.standard.prediction = '胜'; },
  lottery => { lottery.standard.probabilities = {'胜':.3, '平':.3, '负':.4}; },
  lottery => { lottery.standard.probabilities = {'胜':.5, '平':.029, '负':.471}; },
  lottery => {
    const analysis = lottery.direction_analysis;
    lottery.handicap.handicap = analysis.handicap = 6;
    analysis.compatible_handicap_predictions = ['让胜','让平','让负'];
    analysis.incompatible_handicap_predictions = [];
    analysis.conditional_probabilities = {'让胜':.3, '让平':.3, '让负':.4};
    analysis.joint_probabilities = {'让胜':.471 * .3, '让平':.471 * .3, '让负':.471 * .4};
  }
]) {
  const item = JSON.parse(JSON.stringify(original));
  mutate(item.result.lottery);
  const {reference, main} = parts(renderMatchItem(item));
  assert.ok(reference.includes('待重新分析'));
  assert.ok(!reference.includes('data-handicap-result'));
  assert.ok(main.includes('52.3%'), 'a rejected auxiliary analysis must not hide original market data');
}
""")
