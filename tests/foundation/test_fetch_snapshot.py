import os
import re
import tempfile
import threading
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


class SnapshotStoreConcurrencyTests(unittest.TestCase):
    """回归测试：钉死"并发 save() 同一 url 不得损坏/残留 tmp 文件"这条契约。

    fix round 1 之前的实现用固定的 `path + '.tmp'` 做中间文件名，
    多线程并发写同一 url 时会共享该路径，稳定复现 FileNotFoundError 或
    内容交织。改为按 pid+线程号生成唯一 tmp 文件名后，本用例应稳定通过。
    """

    def test_concurrent_save_same_url_does_not_corrupt_or_leak_tmp_files(self):
        root = tempfile.mkdtemp(prefix='snap-')
        store = SnapshotStore(root)
        url = 'https://a.com/x'
        errors = []

        def writer(n):
            try:
                for i in range(50):
                    store.save(url, f'thread-{n}-iter-{i}')
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=writer, args=(n,)) for n in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(errors, [])
        content = store.load(url)
        self.assertIsNotNone(content)
        self.assertRegex(content, r'^thread-\d+-iter-\d+$')
        leftover_tmp = [name for name in os.listdir(root) if re.search(r'\.tmp$', name)]
        self.assertEqual(leftover_tmp, [])


if __name__ == '__main__':
    unittest.main()
