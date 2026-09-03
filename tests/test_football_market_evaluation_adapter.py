# -*- coding: utf-8 -*-
"""读 football-data CSV 的适配层：把文件变成领域层能吃的行。"""

import pathlib
import unittest

from src.football.market_evaluation import (
    football_data_files,
    load_football_data_rows,
    run_market_evaluation,
)

DATA = pathlib.Path('data')


@unittest.skipUnless((DATA / 'E0_2526.csv').exists(), '本地没有 football-data CSV')
class LoadRowsTests(unittest.TestCase):

    def test_rows_carry_league_season_and_pinnacle_closing_odds(self):
        rows = load_football_data_rows(DATA / 'E0_2526.csv')

        self.assertGreater(len(rows), 300)
        first = rows[0]
        self.assertEqual(first['league'], 'E0')
        self.assertEqual(first['season'], '2526')
        self.assertIn(first['FTR'], ('H', 'D', 'A'))
        self.assertGreater(float(first['PSCH']), 1.0)

    def test_bom_on_the_first_header_is_stripped(self):
        rows = load_football_data_rows(DATA / 'E0_2526.csv')

        self.assertIn('Div', rows[0])
        self.assertNotIn('﻿Div', rows[0])


class FileDiscoveryTests(unittest.TestCase):

    def test_only_league_season_csvs_are_picked_up(self):
        files = football_data_files(DATA)

        names = [path.name for path in files]
        self.assertTrue(all(name.endswith('.csv') for name in names))
        self.assertTrue(all('_' in name for name in names))
        self.assertNotIn('kl8_history.json', names)


@unittest.skipUnless((DATA / 'E0_2526.csv').exists(), '本地没有 football-data CSV')
class RunEvaluationTests(unittest.TestCase):

    def test_report_covers_every_source_and_the_ev_grid(self):
        report = run_market_evaluation([DATA / 'E0_2526.csv'])

        self.assertIn('pinnacle_close', report['sources'])
        # 2526 赛季进行中，Pinnacle 收盘列只填了一部分行；只要求非空且不超过总行数。
        pinnacle_n = report['sources']['pinnacle_close']['n']
        self.assertGreater(pinnacle_n, 100)
        self.assertLessEqual(pinnacle_n, report['n_rows'])
        self.assertIn('b365_close', report['ev'])
        self.assertIn(0.0, report['ev']['b365_close'])
        self.assertEqual(report['by_league']['E0']['sources']['pinnacle_close']['n'],
                         report['sources']['pinnacle_close']['n'])

    def test_devig_method_is_threaded_through_the_report(self):
        proportional = run_market_evaluation([DATA / 'E0_2526.csv'])
        power = run_market_evaluation([DATA / 'E0_2526.csv'], devig='power')

        self.assertEqual(power['devig'], 'power')
        self.assertNotEqual(proportional['sources']['pinnacle_close']['log_loss'],
                            power['sources']['pinnacle_close']['log_loss'])
