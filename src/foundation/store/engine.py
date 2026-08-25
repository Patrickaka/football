import os
from contextlib import contextmanager
from urllib.parse import quote_plus

from sqlalchemy import Engine, create_engine


def mysql_url_from_env():
    """从环境变量拼出 MySQL 连接串。

    凭据只从环境变量读取。注意：SSH 独立进程读不到 systemd 的
    Environment= 配置，需显式 source service 文件后再运行。
    """
    host = os.getenv('MYSQL_HOST', '127.0.0.1')
    port = os.getenv('MYSQL_PORT', '3306')
    user = os.getenv('MYSQL_USER', 'root')
    password = quote_plus(os.getenv('MYSQL_PASSWORD', ''))
    database = os.getenv('MYSQL_DB', 'football')
    charset = os.getenv('MYSQL_CHARSET', 'utf8mb4')
    return f'mysql+pymysql://{user}:{password}@{host}:{port}/{database}?charset={charset}'


def make_engine(url, **kwargs):
    """建引擎。pool_pre_ping 处理 MySQL 空闲断连，pool_recycle 短于 wait_timeout。"""
    options = {
        'pool_pre_ping': True,
        'pool_recycle': 3600,
        'future': True,
    }
    if not url.startswith('sqlite'):
        options['pool_size'] = int(os.getenv('MYSQL_POOL_SIZE', '5'))
        options['max_overflow'] = int(os.getenv('MYSQL_MAX_OVERFLOW', '5'))
    options.update(kwargs)
    return create_engine(url, **options)


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
