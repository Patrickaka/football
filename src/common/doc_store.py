"""doc-JSON 表通用读写。

承载开放式 dict 记录（业务会追加键），用 doc 列存完整记录 + promoted 列做查询。
写入沿用原「整文件重写」语义：事务内 DELETE 全量 + 批量 INSERT。
"""
import json
import logging
from datetime import datetime
from pathlib import Path

from . import db
from .paths import data_path

log = logging.getLogger(__name__)

# 表 -> 最近一次降级信息。MySQL 不可用时读写会静默回落到本地 JSON 快照，
# 快照可能是几天前的：不把降级暴露出去，页面上「记录凭空消失」就无从解释。
_degradations = {}


def _record_degradation(table, error, operation='load'):
    _degradations[table] = {
        'source': 'fallback',
        'operation': operation,
        'error': str(error),
        'at': datetime.now().isoformat(),
    }
    log.error('doc_store %s 降级到本地快照（表 %s）：%s', operation, table, error)


def clear_degradation(table):
    _degradations.pop(table, None)


def degradation(table):
    """返回该表最近一次降级信息；None 表示当前读写走的是 MySQL。"""
    return _degradations.get(table)


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


def _sort_key(row, columns):
    # NULL 与字符串不可比较，统一压成 (非空, 值)；NULL 排在前面与 MySQL 的
    # `ORDER BY ... ASC` 一致。排序列都是时间戳/自增 ID，collation 差异无影响。
    return tuple((row.get(c) is not None, row.get(c)) for c in columns)


def load_all(table, order_by):
    """读取整表，反序列化 doc 列为 dict 列表。

    **排序放在 Python 侧**：doc 是大 JSON 列（football_prediction 单条约 40KB），
    交给 MySQL `ORDER BY` 会把整列塞进 sort buffer，行数一多就
    `ERROR 1038 Out of sort memory`，整表读取失败后静默回落到过期快照。
    """
    columns = [c.strip() for c in order_by.split(',') if c.strip()]
    select_cols = ', '.join(columns + ['doc'])
    try:
        rows = list(db.query(f"SELECT {select_cols} FROM {table}"))
    except Exception as e:
        _record_degradation(table, e)
        return _fallback_load_all(table)
    clear_degradation(table)
    rows.sort(key=lambda r: _sort_key(r, columns))
    return [json.loads(r['doc']) for r in rows]


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
    except Exception as e:
        _record_degradation(table, e, operation='upsert')
        _fallback_upsert_one(table, columns, row_values, key_cols)
        return 'fallback'
    try:
        with conn.cursor() as cur:
            cur.execute(sql, row_values)
        clear_degradation(table)
        return 'mysql'
    except Exception as e:
        _record_degradation(table, e, operation='upsert')
        _fallback_upsert_one(table, columns, row_values, key_cols)
        return 'fallback'


def sync_append_only(table, columns, rows_values):
    """把「只增不改」的表同步到 rows_values，只写新增的尾部。

    similar_market 这类样本库有近两万行，每次保存都整表 DELETE + INSERT，
    row-based binlog 会把两万行的删除和插入全记一遍（约 12MB/次）。行数只增
    的表没有理由这么写：库里已有 N 行就只插第 N 行之后的部分。

    行数变少意味着调用方重建了内容（导入、清空），此时才回退到整表重写。
    """
    try:
        stored = db.query(f"SELECT COUNT(*) AS c FROM {table}")[0]['c']
    except Exception as e:
        _record_degradation(table, e, operation='append')
        _fallback_replace_all(table, columns, rows_values)
        return

    if len(rows_values) < stored:
        replace_all(table, columns, rows_values)
        return

    pending = rows_values[stored:]
    if not pending:
        clear_degradation(table)
        return

    placeholders = ",".join(["%s"] * len(columns))
    sql = f"INSERT INTO {table} ({','.join(columns)}) VALUES ({placeholders})"
    try:
        conn = db.get_connection()
    except Exception as e:
        _record_degradation(table, e, operation='append')
        _fallback_replace_all(table, columns, rows_values)
        return
    try:
        conn.begin()
        with conn.cursor() as cur:
            cur.executemany(sql, pending)
        conn.commit()
        clear_degradation(table)
    except Exception as e:
        conn.rollback()
        _record_degradation(table, e, operation='append')
        _fallback_replace_all(table, columns, rows_values)


def replace_all(table, columns, rows_values):
    """事务内清空并批量写入。columns 与 rows_values 的每个元组列序一致。"""
    placeholders = ",".join(["%s"] * len(columns))
    sql = f"INSERT INTO {table} ({','.join(columns)}) VALUES ({placeholders})"
    try:
        conn = db.get_connection()
    except Exception as e:
        _record_degradation(table, e, operation='replace')
        _fallback_replace_all(table, columns, rows_values)
        return
    try:
        conn.begin()
        with conn.cursor() as cur:
            cur.execute(f"DELETE FROM {table}")
            if rows_values:
                cur.executemany(sql, rows_values)
        conn.commit()
        clear_degradation(table)
    except Exception as e:
        conn.rollback()
        _record_degradation(table, e, operation='replace')
        _fallback_replace_all(table, columns, rows_values)
