"""Keep the original layout and probability data with coherent handicap scenarios."""
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
                     "getFootballDirectionState", "renderFootballHandicapProbabilities",
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
    standard:{prediction:standard, probabilities:Object.fromEntries(['胜','平','负'].map(key => [key, key === standard ? probability : (1-probability)/2]))},
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
function parts(card, item) {
  for (const removed of ['football-direction-reference', 'class="match-context"', 'class="market-evidence"']) {
    assert.ok(!card.includes(removed), `removed panel still rendered: ${removed}`);
  }
  // Exercise the pure scenario helper separately; it is no longer a card section.
  return {reference:renderFootballDirectionAnalysis(item.result.lottery, item.result.lottery.spf_available !== false), main:card};
}
function marketRow(main, label) {
  const rows = [...main.matchAll(/<section class="market-row"[^>]*aria-label="([^"]+)"[\s\S]*?<\/section>/g)];
  const row = rows.find(match => match[1].startsWith(label));
  assert.ok(row, `missing market row: ${label}`);
  const visible = row[0].split('<details')[0];
  assert.ok(visible.includes('probability-strip') || !row[0].includes('<details'), `${label} full-match probabilities require expansion`);
  return row[0];
}
""" + assertions
        result = subprocess.run([shutil.which("node"), "-"], input=script,
                                capture_output=True, text=True, encoding="utf-8")
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_linked_results_exclude_impossible_combinations(self):
        self.run_renderer(r"""
for (const [pick, line, conditional, compatible, incompatible] of [
  ['胜', -1, {'让胜':.45,'让平':.55,'让负':0}, ['让胜','让平'], ['让负']],
  ['负', -1, {'让胜':0,'让平':0,'让负':1}, ['让负'], ['让胜','让平']],
  ['平', -1, {'让胜':0,'让平':0,'让负':1}, ['让负'], ['让胜','让平']],
  ['负', 1, {'让胜':0,'让平':.45,'让负':.55}, ['让平','让负'], ['让胜']],
  ['胜', 1, {'让胜':1,'让平':0,'让负':0}, ['让胜'], ['让平','让负']],
  ['胜', -2, {'让胜':.4,'让平':.3,'让负':.3}, ['让胜','让平','让负'], []],
  ['负', 2, {'让胜':.3,'让平':.3,'让负':.4}, ['让胜','让平','让负'], []],
]) {
  const item = makeItem(pick, .54, line, conditional, compatible, incompatible);
  const before = JSON.stringify(item);
  const {main} = parts(renderMatchItem(item), item);
  const rq = marketRow(main, '让球胜平负');
  assert.equal(JSON.stringify(item), before);
  assert(rq.includes('data-probability-basis="conditional_on_standard_result"'));
  assert(!rq.includes('data-probability-basis="full_match"'));
  assert(!rq.includes('handicap-marginal-reference'));
  assert(!rq.includes('<details'));
  for (const key of compatible) {
    assert(rq.includes(`data-handicap-result="${key}"`));
    assert(rq.includes(`<strong>${(conditional[key]*100).toFixed(1)}%</strong>`));
    assert(rq.includes('联合估计 ' + (conditional[key]*.54*100).toFixed(1) + '%'));
  }
  for (const key of incompatible) {
    assert(!rq.includes(`data-handicap-result="${key}"`));
    assert(!rq.includes(`<span>${key}</span>`));
  }
  assert(rq.includes('条件100%不代表整场命中'));
  assert(!rq.includes('is-pick'));
}
""")

    def test_missing_or_invalid_analysis_does_not_restore_independent_results(self):
        self.run_renderer(r"""
const original = makeItem('胜', .54, -1, {'让胜':.4,'让平':.6,'让负':0}, ['让胜','让平'], ['让负']);
for (const mutate of [
  lottery => { delete lottery.direction_analysis; },
  lottery => { lottery.direction_analysis.available = false; },
  lottery => { lottery.handicap.handicap = 1; },
  lottery => { lottery.standard.prediction = '负'; },
  lottery => { lottery.direction_analysis.joint_probabilities['让胜'] = .9; },
  lottery => { lottery.direction_analysis.conditional_probabilities['让负'] = .1; },
]) {
  const item = JSON.parse(JSON.stringify(original));
  mutate(item.result.lottery);
  const {main} = parts(renderMatchItem(item), item);
  const rq = marketRow(main, '让球胜平负');
  assert(rq.includes('待重新分析'));
  assert(!rq.includes('data-handicap-result'));
  assert(!rq.includes('52.3%'));
  assert(!rq.includes('data-probability-basis="full_match"'));
}
""")

    def test_unsold_standard_keeps_the_only_available_independent_market(self):
        self.run_renderer(r"""
const item = makeItem('负', .54, 1, {'让胜':0,'让平':.4,'让负':.6}, ['让平','让负'], ['让胜']);
Object.assign(item.result.lottery, {offer_matched:true, spf_available:false, rqspf_available:true});
const {main} = parts(renderMatchItem(item), item);
assert(!main.includes('aria-label="胜平负"'));
const rq = marketRow(main, '让球胜平负');
assert(rq.includes('独立玩法 · 全场概率'));
assert(!rq.includes('条件概率'));
""")
