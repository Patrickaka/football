# -*- coding: utf-8 -*-
"""SQLite 连接层。

从 MySQL 换过来的动机：MySQL 进程独占 340 MB RSS，而这个库一天只有 384 次
写入（binlog 数出来的），单写者锁绰绰有余。顺带甩掉 binlog——2026-07-20
那次把 40G 盘撑满、导致锁超时卡死的就是它。

上层 28 处 SQL 沿用 `%s` 占位符，由本层统一转成 `?`：全仓 SQL 里没有字面量
`%`，这样上层一行都不用改，也就没有手改 28 处的笔误风险。
"""
import os
import sqlite3
import tempfile
import threading
import unittest
from pathlib import Path

from src.common import db


class _Base(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.path = str(Path(self.dir.name) / 'test.db')
        self._prev = os.environ.get('FOOTBALL_DB_PATH')
        os.environ['FOOTBALL_DB_PATH'] = self.path
        db.close_connection()
        self.addCleanup(self._restore)
        db.execute('CREATE TABLE t (k TEXT PRIMARY KEY, v TEXT, n REAL)')

    def _restore(self):
        db.close_connection()
        if self._prev is None:
            os.environ.pop('FOOTBALL_DB_PATH', None)
        else:
            os.environ['FOOTBALL_DB_PATH'] = self._prev


class PlaceholderTests(_Base):
    """上层写 %s，这层转成 ?。"""

    def test_single_placeholder(self):
        db.execute('INSERT INTO t (k, v) VALUES (%s, %s)', ('a', '1'))
        self.assertEqual(db.query_one('SELECT v FROM t WHERE k=%s', ('a',))['v'], '1')

    def test_many_placeholders(self):
        db.execute_many('INSERT INTO t (k, v) VALUES (%s, %s)', [('a', '1'), ('b', '2')])
        self.assertEqual(len(db.query('SELECT * FROM t')), 2)

    def test_in_clause_built_from_a_join(self):
        db.execute_many('INSERT INTO t (k, v) VALUES (%s, %s)', [('a', '1'), ('b', '2')])
        placeholders = ','.join(['%s'] * 2)
        rows = db.query(f'SELECT k FROM t WHERE k IN ({placeholders})', ('a', 'b'))
        self.assertEqual({r['k'] for r in rows}, {'a', 'b'})


class RowShapeTests(_Base):
    """上层按字典键取值（r['team']），不能退化成元组。"""

    def test_rows_are_dicts(self):
        db.execute('INSERT INTO t (k, v, n) VALUES (%s, %s, %s)', ('a', '1', 2.5))
        row = db.query('SELECT k, v, n FROM t')[0]
        self.assertEqual(row['k'], 'a')
        self.assertEqual(row['n'], 2.5)

    def test_missing_row_is_none(self):
        self.assertIsNone(db.query_one('SELECT v FROM t WHERE k=%s', ('nope',)))


class StreamingTests(_Base):
    """整表读大 JSON 列不能先 fetchall：919 行 185 MB 同时驻留，
    释放后 glibc 也不还给系统。"""

    def test_iter_query_yields_rows_lazily(self):
        db.execute_many('INSERT INTO t (k, v) VALUES (%s, %s)',
                        [(str(i), 'x') for i in range(50)])
        stream = db.iter_query('SELECT k FROM t')
        first = next(stream)
        self.assertIn('k', first)
        self.assertEqual(1 + sum(1 for _ in stream), 50)

    def test_iter_query_returns_a_generator_not_a_list(self):
        import types
        self.assertIsInstance(db.iter_query('SELECT k FROM t'), types.GeneratorType)


class UpsertTests(_Base):
    """SQLite 的 ON CONFLICT 必须写明冲突列，MySQL 的 ON DUPLICATE KEY 不用。"""

    def test_on_conflict_updates_in_place(self):
        sql = ('INSERT INTO t (k, v) VALUES (%s, %s) '
               'ON CONFLICT(k) DO UPDATE SET v=excluded.v')
        db.execute(sql, ('a', '1'))
        db.execute(sql, ('a', '2'))
        self.assertEqual(len(db.query('SELECT * FROM t')), 1)
        self.assertEqual(db.query_one('SELECT v FROM t WHERE k=%s', ('a',))['v'], '2')


class PragmaTests(_Base):
    def test_wal_is_enabled(self):
        """WAL 让读写不互相阻塞；后台任务写的时候接口还要能读。"""
        self.assertEqual(db.query('PRAGMA journal_mode')[0]['journal_mode'].lower(), 'wal')

    def test_busy_timeout_is_set(self):
        self.assertGreater(db.query('PRAGMA busy_timeout')[0]['timeout'], 0)


class ThreadTests(_Base):
    """sqlite3 连接不能跨线程用，沿用每线程一连接。"""

    def test_each_thread_gets_its_own_connection(self):
        # 持有连接对象本身再比 id：线程结束后对象被回收，地址会被下一个
        # 线程复用，只记 id 会得到假的「同一个连接」。
        seen = {}

        def work(name):
            seen[name] = db.get_connection()
            db.execute('INSERT INTO t (k, v) VALUES (%s, %s)', (name, 'x'))

        threads = [threading.Thread(target=work, args=(f't{i}',)) for i in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(len({id(conn) for conn in seen.values()}), 3)
        self.assertEqual(len(db.query('SELECT * FROM t')), 3)


class SchemaTests(_Base):
    def test_init_db_is_idempotent(self):
        db.init_db()
        db.init_db()
        names = {r['name'] for r in
                 db.query("SELECT name FROM sqlite_master WHERE type='table'")}
        self.assertIn('football_prediction', names)
        self.assertIn('kv_store', names)

    def test_init_db_creates_the_indexes(self):
        db.init_db()
        names = {r['name'] for r in
                 db.query("SELECT name FROM sqlite_master WHERE type='index'")}
        self.assertTrue(any('settled' in n for n in names), names)
