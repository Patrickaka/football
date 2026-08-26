import json
import unittest
from datetime import datetime

from src.domain.sports.match import Match


def _match(**kwargs):
    base = dict(
        match_id='bb-1', league='NBA', home='Lakers', away='Celtics',
        start_time=datetime(2026, 8, 26, 10, 30),
        home_score=None, away_score=None,
    )
    base.update(kwargs)
    return Match(**base)


class MatchTests(unittest.TestCase):
    def test_not_settled_without_scores(self):
        self.assertFalse(_match().is_settled())

    def test_not_settled_with_partial_score(self):
        self.assertFalse(_match(home_score=100).is_settled())

    def test_settled_with_both_scores(self):
        self.assertTrue(_match(home_score=100, away_score=98).is_settled())

    def test_zero_zero_counts_as_settled(self):
        """0:0 是合法比分，不能因为假值被判成未结算。"""
        self.assertTrue(_match(home_score=0, away_score=0).is_settled())

    def test_to_dict_is_json_native(self):
        payload = _match(home_score=100, away_score=98).to_dict()
        json.dumps(payload)
        self.assertEqual(payload['start_time'], '2026-08-26T10:30:00')
        self.assertIsInstance(payload['home_score'], int)

    def test_to_dict_keeps_none_scores(self):
        payload = _match().to_dict()
        self.assertIsNone(payload['home_score'])


if __name__ == '__main__':
    unittest.main()
