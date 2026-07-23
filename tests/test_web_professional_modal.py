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


if __name__ == '__main__':
    unittest.main()
