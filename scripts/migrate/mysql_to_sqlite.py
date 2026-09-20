# -*- coding: utf-8 -*-
"""把 MySQL 整库迁进 SQLite。

换 SQLite 的动机：MySQL 进程独占 340 MB RSS，而这个库一天只有 384 次写入，
单写者锁绰绰有余；顺带甩掉 binlog——2026-07-20 把 40G 盘撑满、导致锁等待
超时卡死的就是它。

**迁完不留 MySQL 备份，所以校验必须全量逐行，不能抽样**：条数对得上而某
一行的 doc 被截断过，抽样是看不出来的，而那时候已经没有地方找回原文了。
校验不通过就不算迁移成功，调用方不该继续往下切。

用法：
    python3 -m scripts.migrate.mysql_to_sqlite --dry-run
    python3 -m scripts.migrate.mysql_to_sqlite
    python3 -m scripts.migrate.mysql_to_sqlite --verify-only
"""
import argparse
import datetime
import decimal
import logging
import os
import sqlite3
from pathlib import Path

log = logging.getLogger('migrate.mysql_to_sqlite')

#: 明确的手工备份表，不迁（迁完 MySQL 就删，备份本来也不留）。
EXCLUDED_TABLES = frozenset({'bb_odds_snapshot_bak'})
#: 一批搬多少行。够大以摊薄往返，又不至于让 19763 行的 matches 一次性进内存。
BATCH_SIZE = 500


def mysql_connect():
    import pymysql

    return pymysql.connect(
        host=os.getenv('MYSQL_HOST', '127.0.0.1'),
        port=int(os.getenv('MYSQL_PORT', '3306')),
        user=os.getenv('MYSQL_USER', 'root'),
        password=os.getenv('MYSQL_PASSWORD', ''),
        database=os.getenv('MYSQL_DB', 'football'),
        charset='utf8mb4',
    )


def sqlite_connect(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, isolation_level=None)
    conn.execute('PRAGMA journal_mode=WAL')
    conn.execute('PRAGMA synchronous=NORMAL')
    return conn


def source_tables(mysql):
    with mysql.cursor() as cur:
        cur.execute("SELECT table_name FROM information_schema.tables "
                    "WHERE table_schema=DATABASE() ORDER BY table_name")
        return [name for (name,) in cur.fetchall() if name not in EXCLUDED_TABLES]


def columns_of(mysql, table):
    with mysql.cursor() as cur:
        cur.execute("SELECT column_name, column_type, is_nullable, column_key "
                    "FROM information_schema.columns "
                    "WHERE table_schema=DATABASE() AND table_name=%s "
                    "ORDER BY ordinal_position", (table,))
        return cur.fetchall()


def sqlite_type(mysql_type):
    """MySQL 列类型映射到 SQLite 的存储类。

    只分三类就够：SQLite 是动态类型，列类型只决定亲和性。要紧的是别把
    数值列建成 TEXT——那样 ORDER BY 会按字符串排，1000 会排在 2 前面。
    """
    lowered = mysql_type.lower()
    if lowered.startswith(('tinyint', 'smallint', 'mediumint', 'int', 'bigint')):
        return 'INTEGER'
    if lowered.startswith(('double', 'float', 'decimal', 'numeric')):
        return 'REAL'
    return 'TEXT'


def create_table_ddl(table, columns):
    """按 MySQL 的列定义生成 SQLite 建表语句。

    只用于代码里没有建表定义的遗留表（dlt_history / pailie5_history）。
    有定义的表一律让应用代码自己建，免得结构与代码的期望对不上——
    upsert 依赖主键/唯一约束，那个对不上是会静默写错的。
    """
    parts = []
    primary = [name for name, _type, _null, key in columns if key == 'PRI']
    for name, column_type, nullable, _key in columns:
        piece = f'"{name}" {sqlite_type(column_type)}'
        if nullable == 'NO' and name not in primary:
            piece += ' NOT NULL'
        parts.append(piece)
    if primary:
        parts.append('PRIMARY KEY (' + ','.join(f'"{c}"' for c in primary) + ')')
    return f'CREATE TABLE IF NOT EXISTS "{table}" (' + ', '.join(parts) + ')'


def build_known_tables(sqlite_path):
    """让应用代码建它自己认得的表，结构与代码期望完全一致。"""
    os.environ['FOOTBALL_DB_PATH'] = sqlite_path
    from src.common import db as app_db
    from src.domain.numeric import repository as numeric_repo
    from src.domain.sports.basketball import repository as basketball_repo
    from src.foundation.store import Database, make_engine, database_url_from_env

    app_db.close_connection()
    app_db.init_db()
    engine_db = Database(make_engine(database_url_from_env()))
    numeric_repo.create_all(engine_db)
    basketball_repo.create_all(engine_db)
    engine_db.dispose()
    app_db.close_connection()


def existing_tables(sqlite):
    return {row[0] for row in
            sqlite.execute("SELECT name FROM sqlite_master WHERE type='table'")}


def normalize(value):
    """把两边读出来的值折算成可比较的形式。

    MySQL 的 DATE/DATETIME/DECIMAL 驱动会还原成 Python 对象，SQLite 那边
    一律是字符串。不折算的话每一行都会被判成不一致——错的是校验不是数据。
    """
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, decimal.Decimal):
        return float(value)
    if isinstance(value, datetime.datetime):
        return value.isoformat(sep=' ')
    if isinstance(value, datetime.date):
        return value.isoformat()
    if isinstance(value, datetime.timedelta):
        total = int(value.total_seconds())
        return f'{total // 3600:02d}:{total % 3600 // 60:02d}:{total % 60:02d}'
    if isinstance(value, (bytes, bytearray)):
        return value.decode('utf-8', 'replace')
    return value


def copy_table(mysql, sqlite, table, columns, batch_size=BATCH_SIZE):
    """逐批把一张表搬过去，返回搬了多少行。"""
    import pymysql.cursors

    names = [name for name, *_ in columns]
    lite_cols = ','.join(f'"{name}"' for name in names)
    my_cols = ','.join(f'`{name}`' for name in names)
    placeholders = ','.join(['?'] * len(names))
    insert = f'INSERT OR REPLACE INTO "{table}" ({lite_cols}) VALUES ({placeholders})'

    sqlite.execute(f'DELETE FROM "{table}"')
    moved = 0
    batch = []
    with mysql.cursor(pymysql.cursors.SSCursor) as cur:
        cur.execute(f'SELECT {my_cols} FROM `{table}`')
        for row in cur:
            batch.append(tuple(normalize(v) for v in row))
            if len(batch) >= batch_size:
                sqlite.executemany(insert, batch)
                moved += len(batch)
                batch = []
    if batch:
        sqlite.executemany(insert, batch)
        moved += len(batch)
    return moved


def verify_table(mysql, sqlite, table, columns):
    """逐行全量比对，返回问题描述列表。

    两边都按全部列排序后并排走，避免依赖主键——有的遗留表压根没有主键。
    """
    import pymysql.cursors

    names = [name for name, *_ in columns]
    my_cols = ','.join(f'`{name}`' for name in names)
    lite_cols = ','.join(f'"{name}"' for name in names)
    order = ','.join(str(i + 1) for i in range(len(names)))

    problems = []
    with mysql.cursor(pymysql.cursors.SSCursor) as cur:
        cur.execute(f'SELECT {my_cols} FROM `{table}` ORDER BY {order}')
        lite = sqlite.execute(f'SELECT {lite_cols} FROM "{table}" ORDER BY {order}')
        index = 0
        while True:
            left = cur.fetchone()
            right = lite.fetchone()
            if left is None and right is None:
                break
            if left is None or right is None:
                problems.append(f'{table}: 行数不一致（在第 {index} 行处一边先结束）')
                break
            expected = tuple(normalize(v) for v in left)
            actual = tuple(normalize(v) for v in right)
            if expected != actual:
                for name, want, got in zip(names, expected, actual):
                    if want != got:
                        problems.append(
                            f'{table} 第 {index} 行 {name} 不一致：'
                            f'MySQL {want!r} SQLite {got!r}')
                if len(problems) >= 20:
                    return problems
            index += 1
    return problems


def migrate(sqlite_path, dry_run=False, verify_only=False):
    mysql = mysql_connect()
    try:
        tables = source_tables(mysql)
        layout = {table: columns_of(mysql, table) for table in tables}
        if dry_run:
            with mysql.cursor() as cur:
                for table in tables:
                    cur.execute(f'SELECT COUNT(*) FROM `{table}`')
                    log.info('%-32s %7d 行，%d 列', table, cur.fetchone()[0],
                             len(layout[table]))
            log.info('共 %d 张表（已排除 %s）', len(tables), ','.join(sorted(EXCLUDED_TABLES)))
            return []

        if not verify_only:
            build_known_tables(sqlite_path)

        sqlite = sqlite_connect(sqlite_path)
        try:
            if not verify_only:
                known = existing_tables(sqlite)
                for table in tables:
                    if table not in known:
                        log.info('代码里没有 %s 的定义，按 MySQL 结构建表', table)
                        sqlite.execute(create_table_ddl(table, layout[table]))
                sqlite.execute('BEGIN')
                for table in tables:
                    moved = copy_table(mysql, sqlite, table, layout[table])
                    log.info('%-32s 搬入 %7d 行', table, moved)
                sqlite.execute('COMMIT')

            problems = []
            for table in tables:
                found = verify_table(mysql, sqlite, table, layout[table])
                if found:
                    problems.extend(found)
                else:
                    log.info('%-32s 逐行校验通过', table)
            return problems
        finally:
            sqlite.close()
    finally:
        mysql.close()


def main():
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')
    parser = argparse.ArgumentParser(description='MySQL 整库迁移到 SQLite')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--verify-only', action='store_true')
    parser.add_argument('--path', default=None, help='SQLite 库文件路径')
    args = parser.parse_args()

    from src.common.paths import data_path
    path = args.path or os.getenv('FOOTBALL_DB_PATH') or data_path('football.db')

    problems = migrate(path, dry_run=args.dry_run, verify_only=args.verify_only)
    if problems:
        for item in problems[:20]:
            log.error(item)
        if len(problems) > 20:
            log.error('……另有 %d 项不一致未列出', len(problems) - 20)
        raise SystemExit(1)
    if not args.dry_run:
        log.info('全部表逐行校验通过，SQLite 与 MySQL 一致')


if __name__ == '__main__':
    main()
