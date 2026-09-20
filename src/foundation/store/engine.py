import os
from contextlib import contextmanager
from pathlib import Path

from sqlalchemy import Engine, create_engine, event

#: 默认库文件。与 `src.common.db` 指向同一个——新旧两套数据访问层共用一个库。
_DEFAULT_DB_PATH = Path(__file__).resolve().parents[3] / 'data' / 'football.db'
#: 写锁被占时等这么久，与 `src.common.db.BUSY_TIMEOUT_MS` 保持一致。
BUSY_TIMEOUT_MS = 5000


def database_url_from_env():
    """从环境变量拼出连接串。

    `FOOTBALL_DB_PATH` 默认 data/football.db。注意：SSH 独立进程读不到
    systemd 的 Environment= 配置，需显式导出后再运行。
    """
    path = os.getenv('FOOTBALL_DB_PATH') or str(_DEFAULT_DB_PATH)
    return f'sqlite+pysqlite:///{path}'


def make_engine(url, **kwargs):
    """建引擎。"""
    options = {
        'pool_pre_ping': True,
        'pool_recycle': 3600,
        'future': True,
    }
    if url.startswith('sqlite'):
        # 连接池会把连接跨线程复用，必须关掉 sqlite3 自带的同线程检查。
        options['connect_args'] = {'check_same_thread': False,
                                   'timeout': BUSY_TIMEOUT_MS / 1000}
    options.update(kwargs)
    engine = create_engine(url, **options)
    if url.startswith('sqlite'):
        _apply_sqlite_pragmas(engine)
    return engine


def _apply_sqlite_pragmas(engine):
    """每条新连接都设一遍 busy_timeout——它是连接级的，不像 WAL 写在库文件里。"""

    @event.listens_for(engine, 'connect')
    def _set_pragmas(dbapi_connection, _record):
        cursor = dbapi_connection.cursor()
        try:
            cursor.execute(f'PRAGMA busy_timeout={BUSY_TIMEOUT_MS}')
            cursor.execute('PRAGMA synchronous=NORMAL')
        finally:
            cursor.close()


class Database:
    """引擎持有者，提供连接与事务两种上下文。"""

    def __init__(self, engine: Engine):
        self.engine = engine

    @contextmanager
    def connect(self):
        with self.engine.connect() as conn:
            yield conn

    @contextmanager
    def begin(self):
        with self.engine.begin() as conn:
            yield conn

    def dispose(self):
        self.engine.dispose()
