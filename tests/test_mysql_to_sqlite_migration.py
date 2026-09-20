# -*- coding: utf-8 -*-
"""MySQL → SQLite 迁移里的纯函数。

迁完不留 MySQL 备份，校验是最后一道关。而校验最容易错在**类型折算**上：
两边读出来的同一个值，一边是 `datetime.date`，一边是字符串，不折算的话
每一行都会被判成不一致——那时候错的是校验，不是数据，很容易让人误删。
"""
import datetime
import decimal
import sqlite3
import unittest

from scripts.migrate.mysql_to_sqlite import (create_table_ddl, normalize,
                                             mysql_order_term, order_keys,
                                             sqlite_type)


class TypeMappingTests(unittest.TestCase):
    def test_integers_stay_integers(self):
        for column_type in ('tinyint(1)', 'int(11)', 'bigint(20)', 'smallint'):
            self.assertEqual(sqlite_type(column_type), 'INTEGER', column_type)

    def test_reals_stay_reals(self):
        for column_type in ('double', 'float', 'decimal(10,2)'):
            self.assertEqual(sqlite_type(column_type), 'REAL', column_type)

    def test_everything_else_is_text(self):
        for column_type in ('varchar(64)', 'json', 'longtext', 'date', 'time'):
            self.assertEqual(sqlite_type(column_type), 'TEXT', column_type)

    def test_numeric_columns_do_not_become_text(self):
        """建成 TEXT 的话 ORDER BY 会按字符串排，1000 排在 2 前面。"""
        self.assertNotEqual(sqlite_type('bigint'), 'TEXT')


class NormalizeTests(unittest.TestCase):
    def test_date_becomes_iso_text(self):
        self.assertEqual(normalize(datetime.date(2024, 8, 16)), '2024-08-16')

    def test_datetime_becomes_space_separated_text(self):
        self.assertEqual(normalize(datetime.datetime(2024, 8, 16, 20, 0, 0)),
                         '2024-08-16 20:00:00')

    def test_time_delta_becomes_clock_text(self):
        """MySQL 的 TIME 列经驱动还原成 timedelta，SQLite 那边是 'HH:MM:SS'。"""
        self.assertEqual(normalize(datetime.timedelta(hours=20, minutes=30)),
                         '20:30:00')

    def test_decimal_becomes_float(self):
        self.assertEqual(normalize(decimal.Decimal('1.25')), 1.25)

    def test_bool_becomes_int(self):
        self.assertEqual(normalize(True), 1)
        self.assertEqual(normalize(False), 0)

    def test_bytes_become_text(self):
        self.assertEqual(normalize('值'.encode()), '值')

    def test_plain_values_pass_through(self):
        for value in ('x', 3, 2.5, None):
            self.assertEqual(normalize(value), value)

    def test_both_sides_of_a_time_column_compare_equal(self):
        """这正是校验要回答的问题：同一个值两边读出来能不能对上。"""
        from_mysql = normalize(datetime.timedelta(hours=20))
        from_sqlite = normalize('20:00:00')
        self.assertEqual(from_mysql, from_sqlite)


class DdlTests(unittest.TestCase):
    """只用于代码里没有建表定义的遗留表（dlt_history / pailie5_history）。"""

    def test_ddl_is_valid_sqlite_and_keeps_the_primary_key(self):
        columns = [('id', 'bigint(20)', 'NO', 'PRI'),
                   ('issue', 'varchar(32)', 'NO', ''),
                   ('rate', 'double', 'YES', '')]
        ddl = create_table_ddl('legacy', columns)

        conn = sqlite3.connect(':memory:')
        conn.execute(ddl)
        info = {row[1]: row for row in conn.execute('PRAGMA table_info(legacy)')}
        self.assertEqual(info['id'][2], 'INTEGER')
        self.assertEqual(info['rate'][2], 'REAL')
        self.assertEqual(info['id'][5], 1)  # 主键
        self.assertEqual(info['issue'][3], 1)  # NOT NULL

    def test_is_idempotent(self):
        columns = [('k', 'varchar(16)', 'NO', 'PRI')]
        conn = sqlite3.connect(':memory:')
        conn.execute(create_table_ddl('t', columns))
        conn.execute(create_table_ddl('t', columns))


class OrderKeyTests(unittest.TestCase):
    """校验要逐行并排比对，就得让两边按同样的顺序出行。

    但**大文本列绝不能进 ORDER BY**：MySQL 会把整列塞进 sort buffer，
    football_prediction 的 doc 列有 302 MB，直接撞
    `(1038, 'Out of sort memory')`——这正是第一次跑迁移挂掉的地方。
    """

    def test_primary_key_is_preferred(self):
        columns = [('match_id', 'varchar(128)', 'NO', 'PRI'),
                   ('league', 'varchar(64)', 'YES', ''),
                   ('doc', 'json', 'NO', '')]
        self.assertEqual(order_keys(columns), ['match_id'])

    def test_large_text_columns_never_appear(self):
        columns = [('id', 'bigint', 'NO', 'PRI'), ('doc', 'json', 'NO', ''),
                   ('body', 'longtext', 'YES', ''), ('note', 'text', 'YES', '')]
        for name in ('doc', 'body', 'note'):
            self.assertNotIn(name, order_keys(columns))

    def test_falls_back_to_small_columns_when_there_is_no_primary_key(self):
        columns = [('team', 'varchar(128)', 'NO', ''),
                   ('rating', 'double', 'YES', ''),
                   ('doc', 'json', 'YES', '')]
        self.assertEqual(order_keys(columns), ['team', 'rating'])

    def test_a_table_of_only_large_columns_yields_no_keys(self):
        """排不了就别排——顺序由自然顺序决定，总比撞 1038 强。"""
        self.assertEqual(order_keys([('doc', 'json', 'NO', '')]), [])


class CollationTests(unittest.TestCase):
    """两边的字符串排序规则必须一致，否则逐行并排比对会整体错位。

    库默认 utf8mb4_unicode_ci 把 'lottery_dlt…' 排在 'lottery3d' 前面，
    而 SQLite 按字节码（'3'=0x33 < '_'=0x5F）正好相反——第一次校验就是
    在 kv_store 第 9 行被这个绊住的，而数据本身完全没问题。
    """

    def test_text_columns_are_forced_to_binary_collation(self):
        self.assertEqual(mysql_order_term('k', 'varchar(128)'),
                         '`k` COLLATE utf8mb4_bin')

    def test_numeric_columns_take_no_collation(self):
        """数值列加 COLLATE 是语法错误。"""
        self.assertEqual(mysql_order_term('id', 'bigint(20)'), '`id`')
        self.assertEqual(mysql_order_term('rating', 'double'), '`rating`')

    def test_binary_collation_matches_python_byte_order(self):
        """SQLite 的默认排序就是字节序，与 Python 的字符串比较同序。"""
        self.assertLess('lottery3d', 'lottery_dlt_online_predictions')
