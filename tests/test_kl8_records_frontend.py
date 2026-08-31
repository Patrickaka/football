# -*- coding: utf-8 -*-
"""快乐8预测记录页面必须使用服务端分页和短期页面缓存。"""

import unittest
from pathlib import Path


HTML = Path('web/index.html').read_text(encoding='utf-8')


class KL8RecordsFrontend(unittest.TestCase):

    def test_records_request_contains_server_side_pagination(self):
        self.assertIn("fetchJson('/api/kl8/records?page='", HTML)
        self.assertIn("'&page_size=' + KL8_RECORDS_PAGE_SIZE", HTML)

    def test_loaded_pages_are_cached_briefly(self):
        self.assertIn('const KL8_RECORDS_CACHE_TTL = 30000;', HTML)
        self.assertIn('const kl8RecordsPageCache = new Map();', HTML)
        self.assertIn('Date.now() - cached.loadedAt < KL8_RECORDS_CACHE_TTL', HTML)

    def test_record_mutations_invalidate_the_page_cache(self):
        self.assertGreaterEqual(HTML.count('invalidateKL8RecordsCache();'), 4)


if __name__ == '__main__':
    unittest.main()
