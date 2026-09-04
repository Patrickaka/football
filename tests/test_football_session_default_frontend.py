# -*- coding: utf-8 -*-
"""足球预测页打开时期次默认选中当天，用户点过之后不再被自动改写。"""

import unittest
from pathlib import Path


HTML = Path('web/index.html').read_text(encoding='utf-8')


def _between(start, end):
    return HTML.split(start, 1)[1].split(end, 1)[0]


class FootballSessionDefaultTests(unittest.TestCase):

    def test_default_session_is_today_when_present_otherwise_all(self):
        self.assertIn('function defaultFootballSessionFilter(results)', HTML)
        helper = _between('function defaultFootballSessionFilter(results)',
                          'function footballMatchPassesSessionFilter')
        self.assertIn('todaySessionKey()', helper)
        self.assertIn("return sessions.has(today) ? today : ''", helper)

    def test_render_applies_default_until_user_picks_one(self):
        self.assertIn('let footballSessionFilterPinned = false;', HTML)
        renderer = _between('function renderFootballPredictions(results)',
                            'function setFootballQualityFilter')
        self.assertIn(
            'if (!footballSessionFilterPinned) footballSessionFilter = defaultFootballSessionFilter(results);',
            renderer,
        )
        setter = _between('function setFootballSessionFilter(value)',
                          'function renderFootballSessionButtons')
        self.assertIn('footballSessionFilterPinned = true;', setter)
