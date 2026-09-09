import os
import unittest


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class WebProfessionalModalTests(unittest.TestCase):
    def test_professional_metrics_are_modal_only(self):
        path = os.path.join(ROOT, 'web', 'index.html')
        with open(path, encoding='utf-8') as handle:
            html = handle.read()
        self.assertIn('onclick="openFootballProfessionalModal()"', html)
        self.assertIn('id="football-pro-modal-body"', html)
        self.assertIn("let html = '';", html)
        self.assertNotIn(
            'let html = renderFootballProfessionalStatus(footballProfessionalStatus);',
            html,
        )
        self.assertIn("openFootballEvidenceModal", html)
        self.assertRegex(
            html,
            r'<button\b[^>]*onclick="openFootballEvidenceModal\([^"]*"[^>]*>[^<]*证据审计',
        )
        self.assertIn("deriveFootballEvidence", html)

    def test_web_recommendations_use_accuracy_gates(self):
        path = os.path.join(ROOT, 'web', 'index.html')
        with open(path, encoding='utf-8') as handle:
            html = handle.read()
        self.assertIn("let footballQualityFilter = 'all';", html)
        self.assertIn("requestAnimationFrame(scrollToFootballResults);", html)
        self.assertIn("id=\"football-results-start\"", html)
        self.assertIn("已加载 ${results.length} 场", html)
        self.assertIn("getAccuracyGatePresentation", html)
        self.assertIn("return getFootballResultTier(item).tier === footballQualityFilter", html)
        self.assertIn("研究精选", html)
        self.assertIn("观察", html)
        self.assertIn("观望", html)
        self.assertIn("精选项已通过历史验证筛选", html)
        self.assertIn("观察与观望项仅供参考", html)
        self.assertIn("footballJobRequestWithRetry('/api/predict/batch/start'", html)
        self.assertIn("const FOOTBALL_JOB_MAX_MATCHES = 80;", html)
        self.assertIn("const FOOTBALL_JOB_POLL_MS = 1000;", html)
        self.assertIn("'/api/predict/batch/status?job_id='", html)
        self.assertIn("timeoutError.code = 'ANALYSIS_TIMEOUT'", html)
        self.assertIn("已完成结果先行展示", html)
        self.assertIn("已隔离 ${footballAnalysisFailures.length} 场超时/失败分析", html)
        self.assertIn('class="fixture-completeness">信息完整度', html)
        self.assertIn("预测可信度", html)
        self.assertIn("本场专业证据审计", html)
        self.assertIn("胜平负预测", html)
        self.assertIn("让球胜平负", html)
        self.assertIn("赛后比分", html)
        self.assertIn("setFootballQualityFilter('upset')", html)
        self.assertIn("防冷情景（非推荐）：", html)
        self.assertNotIn("防冷方向：", html)

    def test_professional_status_falls_back_to_static_backtest(self):
        path = os.path.join(ROOT, 'web', 'index.html')
        with open(path, encoding='utf-8') as handle:
            html = handle.read()
        self.assertIn(
            "fetchJson('/reports/professional_football_backtest.json')",
            html,
        )
        self.assertIn("normalizeProfessionalBacktest", html)
        self.assertIn("bundledProfessionalBacktest", html)
        self.assertIn("bundled_audited_baseline", html)
        self.assertIn("距离专业生产系统的差距", html)
        self.assertIn("生产预测闭环监控", html)
        self.assertIn("league_spf_validation", html)
        self.assertIn("生产联赛门禁（数据库历史 · 时间后段冻结验证）", html)
        self.assertIn("95%区间", html)
        self.assertIn("数据读取失败，不代表样本为0场", html)

    def test_web_requests_and_renders_all_repaired_football_markets(self):
        path = os.path.join(ROOT, 'web', 'index.html')
        with open(path, encoding='utf-8') as handle:
            html = handle.read()

        self.assertIn("if (m.rqspf && !m.rqspf.error)", html)
        self.assertIn("🎯 让球胜平负", html)
        self.assertIn("🎯 让球胜平负（主队", html)

    def test_football_list_uses_compact_joint_market_summary(self):
        path = os.path.join(ROOT, 'web', 'index.html')
        with open(path, encoding='utf-8') as handle:
            html = handle.read()

        self.assertIn('const compactFootballView = true;', html)
        self.assertIn('class="football-compact-summary"', html)
        self.assertIn('renderFootballDirectionAnalysis(r.lottery, spfPredictionEnabled)', html)
        self.assertIn('<details class="football-direction-reference">', html)
        self.assertIn('<summary>胜平负与让球的对应分析</summary>', html)
        self.assertNotIn('renderFootballMarginalReference', html)
        self.assertIn('<h3>玩法分析</h3><span>赛前概率参考</span>', html)
        self.assertIn("compactProbLine(standardPrediction.probs, ['胜','平','负'])", html)
        self.assertIn("getFootballDirectionState(r.lottery, spfPredictionEnabled)", html)
        self.assertIn("renderFootballHandicapProbabilities(r.lottery, spfPredictionEnabled)", html)
        self.assertIn('class="football-score-reference"', html)
        self.assertIn("比分参考 Top3（非主推）", html)
        self.assertIn("getFootballScoreSettlement(score, lotteryHandicap)", html)
        self.assertIn('class="football-score-reference-panel"', html)
        self.assertIn("全部场次比分参考 Top3", html)
        self.assertIn("当前筛选下暂无主推，下方仍提供比分参考", html)
        self.assertIn('if (!compactFootballView && top.length)', html)
        self.assertIn('if (!compactFootballView && htfProbs.length)', html)

if __name__ == '__main__':
    unittest.main()
