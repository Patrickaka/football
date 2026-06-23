"""doc-JSON 表通用读写。

承载开放式 dict 记录（业务会追加键），用 doc 列存完整记录 + promoted 列做查询。
写入沿用原「整文件重写」语义：事务内 DELETE 全量 + 批量 INSERT。
"""
import json

from . import db


def load_all(table, order_by):
    """读取整表，反序列化 doc 列为 dict 列表。"""
    rows = db.query(f"SELECT doc FROM {table} ORDER BY {order_by}")
    return [json.loads(r['doc']) for r in rows]


def replace_all(table, columns, rows_values):
    """事务内清空并批量写入。columns 与 rows_values 的每个元组列序一致。"""
    placeholders = ",".join(["%s"] * len(columns))
    sql = f"INSERT INTO {table} ({','.join(columns)}) VALUES ({placeholders})"
    conn = db.get_connection()
    try:
        conn.begin()
        with conn.cursor() as cur:
            cur.execute(f"DELETE FROM {table}")
            if rows_values:
                cur.executemany(sql, rows_values)
        conn.commit()
    except Exception:
        conn.rollback()
        raise
