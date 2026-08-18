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
        self.assertIn("🔎 证据审计", html)
        self.assertIn("deriveFootballEvidence", html)

    def test_filter_uses_prediction_reliability_not_information_completeness(self):
        path = os.path.join(ROOT, 'web', 'index.html')
        with open(path, encoding='utf-8') as handle:
            html = handle.read()
        self.assertIn(
            'calculateFootballPredictionReliability(probability, informationCompleteness)',
            html,
        )
        self.assertIn("reliability >= 0.80", html)
        self.assertIn(
            "reliability >= 0.60 && reliability < 0.80",
            html,
        )
        self.assertIn("超强可信≥80%", html)
        self.assertIn("高可信60%–80%", html)
        self.assertIn("📡 信息完整度", html)
        self.assertIn("预测可信度", html)
        self.assertIn("本场专业证据审计", html)
        self.assertIn("胜平负预测", html)
        self.assertIn("让球胜平负", html)
        self.assertIn("赛后比分", html)

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
        self.assertIn("95%区间", html)
        self.assertIn("数据读取失败，不代表样本为0场", html)

    def test_web_requests_and_renders_all_repaired_football_markets(self):
        path = os.path.join(ROOT, 'web', 'index.html')
        with open(path, encoding='utf-8') as handle:
            html = handle.read()

        self.assertIn("types: 'spf,rqspf,zjq'", html)
        self.assertIn("if (m.rqspf && !m.rqspf.error)", html)
        self.assertIn("🎯 让球胜平负", html)
        self.assertIn("已应用统一亚盘水位与大小球变化修正", html)
        self.assertIn("🎯 让球胜平负（主队", html)
        self.assertIn("欧赔 + 亚盘水位综合预测", html)
        self.assertIn("欧赔提供基础概率，亚盘升降盘与水位变化修正方向", html)

    def test_football_list_uses_compact_joint_market_summary(self):
        path = os.path.join(ROOT, 'web', 'index.html')
        with open(path, encoding='utf-8') as handle:
            html = handle.read()

        self.assertIn('const compactFootballView = true;', html)
        self.assertIn('class="football-compact-summary"', html)
        self.assertIn("compactProbLine(standardPrediction.probs, ['胜','平','负'])", html)
        self.assertIn("compactProbLine(handicapCard.prediction.probs, ['让胜','让平','让负'])", html)
        self.assertIn('if (!compactFootballView && top.length)', html)
        self.assertIn('if (!compactFootballView && htfProbs.length)', html)

    def test_lottery3d_uses_single_compact_primary_view(self):
        path = os.path.join(ROOT, 'web', 'index.html')
        with open(path, encoding='utf-8') as handle:
            html = handle.read()

        self.assertIn('function renderLottery3dCompact(r)', html)
        self.assertIn('renderLottery3dCompact(data.result);', html)
        self.assertIn('lottery3d-compact-primary', html)
        self.assertIn('组六唯一主推', html)
        self.assertIn('最近25期数字出现频率', html)


if __name__ == '__main__':
    unittest.main()
