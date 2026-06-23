"""MySQL 连接层。

配置全部来自环境变量，本地与服务器共用同一套代码：
    MYSQL_HOST      默认 127.0.0.1
    MYSQL_PORT      默认 3306
    MYSQL_USER      默认 root
    MYSQL_PASSWORD  默认 空
    MYSQL_DB        默认 football
    MYSQL_CHARSET   默认 utf8mb4

server.py 使用 ThreadingHTTPServer，PyMySQL 连接非线程安全，故每线程持有独立连接。
"""
import os
import threading
from pathlib import Path

import pymysql
from pymysql.cursors import DictCursor

_SCHEMA_FILE = Path(__file__).resolve().parent / 'schema.sql'
_local = threading.local()


def _config():
    return {
        'host': os.getenv('MYSQL_HOST', '127.0.0.1'),
        'port': int(os.getenv('MYSQL_PORT', '3306')),
        'user': os.getenv('MYSQL_USER', 'root'),
        'password': os.getenv('MYSQL_PASSWORD', ''),
        'database': os.getenv('MYSQL_DB', 'football'),
        'charset': os.getenv('MYSQL_CHARSET', 'utf8mb4'),
        'connect_timeout': int(os.getenv('MYSQL_CONNECT_TIMEOUT', '10')),
        'autocommit': True,
        'cursorclass': DictCursor,
    }


def get_connection():
    """返回当前线程的 MySQL 连接，断线自动重建。"""
    conn = getattr(_local, 'conn', None)
    if conn is not None:
        try:
            conn.ping(reconnect=False)
            return conn
        except Exception:
            try:
                conn.close()
            except Exception:
                pass
            _local.conn = None
    conn = pymysql.connect(**_config())
    _local.conn = conn
    return conn


def query(sql, params=None):
    """执行查询，返回字典行列表。"""
    with get_connection().cursor() as cur:
        cur.execute(sql, params or ())
        return cur.fetchall()


def query_one(sql, params=None):
    """执行查询，返回首行字典或 None。"""
    with get_connection().cursor() as cur:
        cur.execute(sql, params or ())
        return cur.fetchone()


def execute(sql, params=None):
    """执行写入语句，返回受影响行数。"""
    with get_connection().cursor() as cur:
        return cur.execute(sql, params or ())


def execute_many(sql, seq_params):
    """批量写入，返回受影响行数。"""
    seq = list(seq_params)
    if not seq:
        return 0
    with get_connection().cursor() as cur:
        return cur.executemany(sql, seq)


def init_db():
    """执行 schema.sql 建表（幂等）。"""
    sql_text = _SCHEMA_FILE.read_text(encoding='utf-8')
    statements = [s.strip() for s in sql_text.split(';') if s.strip()]
    with get_connection().cursor() as cur:
        for stmt in statements:
            cur.execute(stmt)
