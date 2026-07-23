import os
import tempfile
import unittest
from unittest.mock import patch

from src.football.cache_manager import FootballCacheManager


class FootballCacheMemoryTests(unittest.TestCase):
    def test_second_get_uses_memory_when_disk_read_is_unavailable(self):
        with tempfile.TemporaryDirectory() as temp:
            cache = FootballCacheManager(temp)
            cache.set('analysis', 'match-1', {'value': 7})
            self.assertEqual(cache.get('analysis', 'match-1'), {'value': 7})
            with patch('builtins.open', side_effect=AssertionError('disk should not be read')):
                self.assertEqual(cache.get('analysis', 'match-1'), {'value': 7})

    def test_atomic_write_leaves_no_temp_file(self):
        with tempfile.TemporaryDirectory() as temp:
            cache = FootballCacheManager(temp)
            cache.set('analysis', 'match-2', {'value': 8})
            self.assertFalse(any(name.endswith('.tmp') for name in os.listdir(temp)))


if __name__ == '__main__':
    unittest.main()
