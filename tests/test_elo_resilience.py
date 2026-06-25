import unittest
from unittest.mock import patch

from src.football.elo import ELORatingSystem


class ELOResilienceTests(unittest.TestCase):
    def test_save_ratings_handles_repository_failure_without_temp_file_error(self):
        elo = ELORatingSystem.__new__(ELORatingSystem)
        elo.elo_file = 'unused'
        elo.ratings = {'A': 1500}
        elo.history = {}

        with patch('src.football.elo.repositories.elo_save', side_effect=RuntimeError('db down')):
            elo._save_ratings()


if __name__ == '__main__':
    unittest.main()
