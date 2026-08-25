import unittest
from unittest import mock

from sqlalchemy import text

from src.foundation.store.engine import Database, make_engine, mysql_url_from_env


class MysqlUrlTests(unittest.TestCase):
    def test_builds_url_from_env(self):
        env = {
            'MYSQL_HOST': 'db.internal',
            'MYSQL_PORT': '3307',
            'MYSQL_USER': 'football',
            'MYSQL_PASSWORD': 'secret',
            'MYSQL_DB': 'football',
        }
        with mock.patch.dict('os.environ', env, clear=True):
            url = mysql_url_from_env()
        self.assertEqual(
            url, 'mysql+pymysql://football:secret@db.internal:3307/football?charset=utf8mb4'
        )

    def test_applies_defaults_when_env_missing(self):
        with mock.patch.dict('os.environ', {'MYSQL_PASSWORD': 'p'}, clear=True):
            url = mysql_url_from_env()
        self.assertIn('@127.0.0.1:3306/football', url)

    def test_password_is_url_quoted(self):
        with mock.patch.dict('os.environ', {'MYSQL_PASSWORD': 'p@ss/word'}, clear=True):
            url = mysql_url_from_env()
        self.assertIn('p%40ss%2Fword', url)

    def test_password_with_space_survives_round_trip(self):
        from sqlalchemy.engine import make_url

        secret = 'p@ss word/slash:colon'
        with mock.patch.dict('os.environ', {'MYSQL_PASSWORD': secret}, clear=True):
            url = mysql_url_from_env()
        self.assertEqual(make_url(url).password, secret)


class DatabaseTests(unittest.TestCase):
    def setUp(self):
        self.db = Database(make_engine('sqlite+pysqlite:///:memory:'))

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
