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

    def test_confidence_filter_uses_displayed_confidence_score(self):
        path = os.path.join(ROOT, 'web', 'index.html')
        with open(path, encoding='utf-8') as handle:
            html = handle.read()
        self.assertIn(
            'const confidence = Number(item?.result?.confidence?.score);',
            html,
        )
        self.assertIn("confidence >= 0.75", html)
        self.assertIn("confidence >= 0.60 && confidence < 0.75", html)
        self.assertIn("高置信60%–75%", html)

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
        self.assertIn("数据读取失败，不代表样本为0场", html)


if __name__ == '__main__':
    unittest.main()
