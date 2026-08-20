"""doc-JSON 表通用读写。

承载开放式 dict 记录（业务会追加键），用 doc 列存完整记录 + promoted 列做查询。
写入沿用原「整文件重写」语义：事务内 DELETE 全量 + 批量 INSERT。
"""
import json
from pathlib import Path

from . import db
from .paths import data_path


def _fallback_path(table):
    return Path(data_path(f'doc_store_{table}.json'))


def _fallback_load_all(table):
    path = _fallback_path(table)
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding='utf-8'))
    except Exception:
        return []
    if isinstance(data, list):
        return data
    return data.get('records', [])


def _fallback_replace_all(table, columns, rows_values):
    records = []
    try:
        doc_index = columns.index('doc')
    except ValueError:
        doc_index = None

    for row in rows_values:
        if doc_index is not None:
            doc = row[doc_index]
            records.append(json.loads(doc) if isinstance(doc, str) else doc)
        else:
            records.append(dict(zip(columns, row)))

    path = _fallback_path(table)
    tmp_path = path.with_suffix(path.suffix + '.tmp')
    tmp_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding='utf-8')
    tmp_path.replace(path)


def load_all(table, order_by):
    """读取整表，反序列化 doc 列为 dict 列表。"""
    try:
        rows = db.query(f"SELECT doc FROM {table} ORDER BY {order_by}")
        return [json.loads(r['doc']) for r in rows]
    except Exception:
        return _fallback_load_all(table)


def _fallback_upsert_one(table, columns, row_values, key_cols):
    """MySQL 不可用时的单行 UPSERT 降级：按 key_cols 在 fallback JSON 中替换或追加。"""
    try:
        doc_index = columns.index('doc')
    except ValueError:
        doc_index = None

    if doc_index is not None:
        doc = row_values[doc_index]
        new_record = json.loads(doc) if isinstance(doc, str) else doc
    else:
        new_record = dict(zip(columns, row_values))

    col_pos = {c: i for i, c in enumerate(columns)}
    key = tuple(row_values[col_pos[c]] for c in key_cols)

    def record_key(rec):
        # fallback 记录是完整 doc（dict），键列取自 doc 自身
        return tuple(rec.get(c) for c in key_cols)

    records = _fallback_load_all(table)
    replaced = False
    for idx, rec in enumerate(records):
        if record_key(rec) == key:
            records[idx] = new_record
            replaced = True
            break
    if not replaced:
        records.append(new_record)

    path = _fallback_path(table)
    tmp_path = path.with_suffix(path.suffix + '.tmp')
    tmp_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding='utf-8')
    tmp_path.replace(path)


def upsert_one(table, columns, row_values, key_cols):
    """单行 UPSERT，仅写一行，避免整表 DELETE+INSERT 重写。

    表须有以 key_cols 为主键/唯一键的约束来决定冲突。冲突时更新除 key_cols
    外的所有列。用于每请求级的高频写入（如 football_prediction），把写入量
    从 O(表行数) 降到 O(1)。
    """
    placeholders = ",".join(["%s"] * len(columns))
    update_cols = [c for c in columns if c not in key_cols]
    update_clause = ",".join(f"{c}=VALUES({c})" for c in update_cols)
    sql = (
        f"INSERT INTO {table} ({','.join(columns)}) VALUES ({placeholders})"
        f" ON DUPLICATE KEY UPDATE {update_clause}"
    )
    try:
        conn = db.get_connection()
    except Exception:
        _fallback_upsert_one(table, columns, row_values, key_cols)
        return 'fallback'
    try:
        with conn.cursor() as cur:
            cur.execute(sql, row_values)
        return 'mysql'
    except Exception:
        _fallback_upsert_one(table, columns, row_values, key_cols)
        return 'fallback'


def replace_all(table, columns, rows_values):
    """事务内清空并批量写入。columns 与 rows_values 的每个元组列序一致。"""
    placeholders = ",".join(["%s"] * len(columns))
    sql = f"INSERT INTO {table} ({','.join(columns)}) VALUES ({placeholders})"
    try:
        conn = db.get_connection()
    except Exception:
        _fallback_replace_all(table, columns, rows_values)
        return
    try:
        conn.begin()
        with conn.cursor() as cur:
            cur.execute(f"DELETE FROM {table}")
            if rows_values:
                cur.executemany(sql, rows_values)
        conn.commit()
    except Exception:
        conn.rollback()
        _fallback_replace_all(table, columns, rows_values)
