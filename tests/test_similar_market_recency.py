import unittest
from datetime import datetime

from src.football.similar_market import MatchRecord, _recency_weight


class SimilarMarketRecencyTests(unittest.TestCase):
    def test_recent_sample_has_higher_weight_than_old_sample(self):
        recent = MatchRecord({'date': '01/06/2026'})
        old = MatchRecord({'date': '01/06/2020'})
        now = datetime(2026, 6, 25)

        self.assertGreater(_recency_weight(recent, now), _recency_weight(old, now))

    def test_missing_date_recent_season_uses_soft_weight(self):
        record = MatchRecord({'season': '2025-2026'})

        self.assertGreaterEqual(_recency_weight(record, datetime(2026, 6, 25)), 0.7)


if __name__ == '__main__':
    unittest.main()
