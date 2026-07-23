import unittest

from src.football.bayes_report import REPORT_SCHEMA_VERSION, render_html


class BayesReportProfessionalGateTests(unittest.TestCase):
    def test_report_visibly_blocks_unvalidated_betting(self):
        report = {
            'match': {'league': '英超', 'home': 'A', 'away': 'B', 'time': '20:00', 'num': '001'},
            'wdl': {'home': .5, 'draw': .3, 'away': .2},
            'p0': {'home': .48, 'draw': .3, 'away': .22},
            'ts': '2026-07-23 12:00',
            'module_version': 'test',
            'tool_log': [],
            'tactical': {'available': False, 'trap_note': 'missing'},
            'league': {'lines': ['test']},
            'update': {'evidence': []},
            'confidence_label': '中',
            'confidence_score': .6,
            'risk_level': '中',
            'scripts': [],
            'trap_warn': {'available': False, 'note': 'missing'},
            'risks': ['test'],
            'live_context_quality': {'quality_score': .6, 'confidence_multiplier': .6},
            'professional_validation': {
                'available': True,
                'out_of_sample_n': 1804,
                'model': {'logloss': .997},
                'market': {'logloss': .977},
                'strategy': {'roi': -.0192, 'mean_clv': -.0063},
            },
            'decision_gate': {'official_bet_allowed': False},
        }
        html = render_html(report)
        self.assertIn(f"data-report-schema='{REPORT_SCHEMA_VERSION}'", html)
        self.assertIn('研究模式 / 禁止正式投注', html)
        self.assertIn('严格样本外验证', html)
        self.assertIn('ROI -1.92%', html)
        self.assertIn('本场专属结论', html)
        self.assertIn('研究倾向：主胜', html)
        self.assertIn('本场未提供可核验的体彩让球盘', html)
        self.assertNotIn('让球胜平负 · 80%目标精选', html)

    def test_match_specific_verdict_changes_with_probabilities(self):
        base = {
            'match': {'league': '英超', 'home': 'A', 'away': 'B', 'time': '20:00', 'num': '001'},
            'p0': {'home': .4, 'draw': .3, 'away': .3},
            'ts': '2026-07-23 12:00',
            'module_version': 'test',
            'tool_log': [],
            'tactical': {'available': False, 'trap_note': 'missing'},
            'league': {'lines': ['test']},
            'update': {'evidence': []},
            'confidence_label': '中',
            'confidence_score': .6,
            'risk_level': '中',
            'scripts': [],
            'trap_warn': {'available': False, 'note': 'missing'},
            'risks': ['test'],
            'live_context_quality': {},
            'professional_validation': {'available': False},
            'decision_gate': {'official_bet_allowed': False},
        }
        home_report = dict(base, wdl={'home': .72, 'draw': .19, 'away': .09})
        away_report = dict(base, wdl={'home': .18, 'draw': .22, 'away': .60})
        self.assertIn('研究倾向：主胜', render_html(home_report))
        self.assertIn('研究倾向：客胜', render_html(away_report))


if __name__ == '__main__':
    unittest.main()
