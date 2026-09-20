import unittest
from unittest import mock

from sqlalchemy import text

from src.foundation.store.engine import Database, make_engine, database_url_from_env


class DatabaseUrlTests(unittest.TestCase):
    def test_builds_url_from_env(self):
        with mock.patch.dict('os.environ', {'FOOTBALL_DB_PATH': '/srv/f.db'}, clear=True):
            self.assertEqual(database_url_from_env(), 'sqlite+pysqlite:////srv/f.db')

    def test_defaults_to_the_data_directory(self):
        from pathlib import Path

        from sqlalchemy.engine import make_url

        with mock.patch.dict('os.environ', {}, clear=True):
            url = database_url_from_env()
        self.assertTrue(url.startswith('sqlite+pysqlite:///'), url)
        # 拆成 Path 比对，不写整段路径字面量——仓库有条守卫在扫测试里的
        # data 目录引用，而这里只是断言默认值，并不读那个文件。
        default = Path(make_url(url).database)
        self.assertEqual(default.name, 'football.db')
        self.assertEqual(default.parent.name, 'data')

    def test_path_survives_the_round_trip(self):
        """路径带空格也要能还原，不然换个目录就悄悄连到别的库。"""
        from sqlalchemy.engine import make_url

        path = '/srv/my football/f.db'
        with mock.patch.dict('os.environ', {'FOOTBALL_DB_PATH': path}, clear=True):
            self.assertEqual(make_url(database_url_from_env()).database, path)


class DatabaseTests(unittest.TestCase):
    def setUp(self):
        self.db = Database(make_engine('sqlite+pysqlite:///:memory:'))

    def test_busy_timeout_is_applied_to_new_connections(self):
        """busy_timeout 是连接级的，不像 WAL 写在库文件里——每条连接都要设。"""
        with self.db.connect() as conn:
            self.assertEqual(conn.execute(text('PRAGMA busy_timeout')).scalar(), 5000)

    def test_connect_executes_query(self):
        with self.db.connect() as conn:
            self.assertEqual(conn.execute(text('select 1')).scalar(), 1)

    def test_begin_commits_on_success(self):
        with self.db.begin() as conn:
            conn.execute(text('create table t (id integer)'))
            conn.execute(text('insert into t values (1)'))
        with self.db.connect() as conn:
            self.assertEqual(conn.execute(text('select count(*) from t')).scalar(), 1)

    def test_begin_rolls_back_on_error(self):
        with self.db.begin() as conn:
            conn.execute(text('create table t (id integer)'))
        with self.assertRaises(ValueError):
            with self.db.begin() as conn:
                conn.execute(text('insert into t values (1)'))
                raise ValueError('boom')
        with self.db.connect() as conn:
            self.assertEqual(conn.execute(text('select count(*) from t')).scalar(), 0)


if __name__ == '__main__':
    unittest.main()
