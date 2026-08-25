import tempfile
import unittest

from src.foundation.fetch.snapshot import SnapshotStore


class SnapshotStoreTests(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp(prefix='snap-')
        self.store = SnapshotStore(self.root)

    def test_load_missing_returns_none(self):
        self.assertIsNone(self.store.load('https://a.com/x'))

    def test_save_then_load_roundtrips(self):
        self.store.save('https://a.com/x?q=1', '<html>hi</html>')
        self.assertEqual(self.store.load('https://a.com/x?q=1'), '<html>hi</html>')

    def test_different_urls_do_not_collide(self):
        self.store.save('https://a.com/x', 'one')
        self.store.save('https://a.com/y', 'two')
        self.assertEqual(self.store.load('https://a.com/x'), 'one')
        self.assertEqual(self.store.load('https://a.com/y'), 'two')

    def test_query_string_is_part_of_identity(self):
        self.store.save('https://a.com/x?q=1', 'one')
        self.assertIsNone(self.store.load('https://a.com/x?q=2'))

    def test_unicode_body_roundtrips(self):
        self.store.save('https://a.com/x', '中文内容')
        self.assertEqual(self.store.load('https://a.com/x'), '中文内容')


if __name__ == '__main__':
    unittest.main()
