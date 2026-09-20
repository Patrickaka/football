"""SQLite 连接层。

配置来自环境变量，本地与服务器共用同一套代码：
    FOOTBALL_DB_PATH  默认 data/football.db

**为什么从 MySQL 换过来**：MySQL 进程独占 340 MB RSS，而这个库一天只有
384 次写入（从 binlog 数出来的），单写者锁绰绰有余；顺带甩掉 binlog——
2026-07-20 把 40G 盘撑满、导致锁等待超时卡死的就是它。

两处兼容垫片，目的是让上层 28 处 SQL 与 4 处事务代码一个字都不用改：
1. 占位符 `%s` 统一转成 `?`。全仓 SQL 里没有字面量 `%`，所以这个替换是
   安全的，也省掉手改 28 处的笔误风险。
2. 连接与游标包一层，补上 PyMySQL 的 `conn.begin()` 与
   `with conn.cursor() as cur` 用法——sqlite3 两样都没有。

sqlite3 的连接不能跨线程使用，沿用原来的「每线程一个连接」。
"""
import os
import sqlite3
import threading
from pathlib import Path

from .paths import data_path

_SCHEMA_FILE = Path(__file__).resolve().parent / 'schema.sql'
_local = threading.local()

#: 写锁被占时等这么久再报 database is locked。后台任务偶尔会写得久一点。
BUSY_TIMEOUT_MS = 5000


def _database_path():
    return os.getenv('FOOTBALL_DB_PATH') or data_path('football.db')


def _dict_row(cursor, row):
    return {column[0]: row[index] for index, column in enumerate(cursor.description)}


def _sqlite_sql(sql):
    """把上层的 `%s` 占位符换成 `?`。"""
    return sql.replace('%s', '?')


class _Cursor:
    """PyMySQL 风格的游标：支持 `with`，并转换占位符。"""

    def __init__(self, cursor):
        self._cursor = cursor

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self._cursor.close()
        return False

    def __iter__(self):
        return iter(self._cursor)

    def execute(self, sql, params=None):
        self._cursor.execute(_sqlite_sql(sql), tuple(params or ()))
        return self._cursor.rowcount

    def executemany(self, sql, seq_params):
        self._cursor.executemany(_sqlite_sql(sql), [tuple(p) for p in seq_params])
        return self._cursor.rowcount

    def fetchall(self):
        return self._cursor.fetchall()

    def fetchone(self):
        return self._cursor.fetchone()

    def close(self):
        self._cursor.close()

    @property
    def rowcount(self):
        return self._cursor.rowcount


class _Connection:
    """PyMySQL 风格的连接：补上 `begin()`，`cursor()` 返回上面那个游标。"""

    def __init__(self, raw):
        self._raw = raw

    def cursor(self, *_args, **_kwargs):
        return _Cursor(self._raw.cursor())

    def begin(self):
        # autocommit 模式下事务要显式开。
        self._raw.execute('BEGIN')

    def commit(self):
        self._raw.commit()

    def rollback(self):
        self._raw.rollback()

    def executescript(self, script):
        self._raw.executescript(script)

    def close(self):
        self._raw.close()


def _connect():
    path = _database_path()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    raw = sqlite3.connect(path, timeout=BUSY_TIMEOUT_MS / 1000,
                          isolation_level=None)
    raw.row_factory = _dict_row
    # WAL：读写互不阻塞——后台任务写库时接口还要能读。
    raw.execute('PRAGMA journal_mode=WAL')
    raw.execute(f'PRAGMA busy_timeout={BUSY_TIMEOUT_MS}')
    # NORMAL 在 WAL 下仍然崩溃安全，只是掉电可能丢最后几个事务；
    # 这些数据都能从上游重新抓，不值得为它每次写都 fsync。
    raw.execute('PRAGMA synchronous=NORMAL')
    return _Connection(raw)


def get_connection():
    """返回当前线程的连接，按需建立。"""
    conn = getattr(_local, 'conn', None)
    if conn is None:
        conn = _connect()
        _local.conn = conn
    return conn


def close_connection():
    """关掉当前线程的连接。切换库文件（测试）与进程退出时用。"""
    conn = getattr(_local, 'conn', None)
    if conn is not None:
        try:
            conn.close()
        finally:
            _local.conn = None


def query(sql, params=None):
    """执行查询，返回字典行列表。"""
    with get_connection().cursor() as cur:
        cur.execute(sql, params)
        return cur.fetchall()


def iter_query(sql, params=None):
    """流式执行查询，逐行产出字典行。

    大 JSON 列的整表读取不能先 fetchall：919 行 185 MB 的字符串同时驻留，
    释放后 glibc 也不还给系统。游标一次只把一行拉进进程。
    """
    with get_connection().cursor() as cur:
        cur.execute(sql, params)
        for row in cur:
            yield row


def query_one(sql, params=None):
    """执行查询，返回首行字典或 None。"""
    with get_connection().cursor() as cur:
        cur.execute(sql, params)
        return cur.fetchone()


def execute(sql, params=None):
    """执行写入语句，返回受影响行数。"""
    with get_connection().cursor() as cur:
        return cur.execute(sql, params)


def execute_many(sql, seq_params):
    """批量写入，返回受影响行数。"""
    seq = list(seq_params)
    if not seq:
        return 0
    with get_connection().cursor() as cur:
        return cur.executemany(sql, seq)


def init_db():
    """执行 schema.sql 建表（幂等）。"""
    get_connection().executescript(_SCHEMA_FILE.read_text(encoding='utf-8'))
