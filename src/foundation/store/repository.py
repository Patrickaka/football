from sqlalchemy import delete, func, insert, select, update


class Repository:
    """表级仓储基类。子类只需声明 table 属性。

    返回值一律为普通 dict，使领域层不依赖 SQLAlchemy 类型。
    """

    table = None

    def __init__(self, db):
        if self.table is None:
            raise ValueError(f'{type(self).__name__} 未声明 table')
        self.db = db

    def insert_many(self, rows):
        rows = list(rows)
        if not rows:
            return 0
        for row in rows:
            self._reject_unknown_columns(row)
        with self.db.begin() as conn:
            conn.execute(insert(self.table), rows)
        return len(rows)

    def upsert(self, row, key_cols):
        """按 key_cols 判断存在与否，决定 update 还是 insert。

        不使用方言特有的 ON DUPLICATE KEY / ON CONFLICT，以保持
        MySQL 与测试用 SQLite 行为一致。

        并发前提：key_cols 必须对应表上真实的主键或唯一约束。若
        key_cols 只是"逻辑主键"而表上没有对应 UNIQUE 约束，并发下
        两次 upsert 可能都判定"不存在"而各自 insert 成功，产生
        静默重复行——先查后写这一权衡本身不提供跨连接的原子性。
        """
        self._reject_unknown_columns(row)
        conditions = [self.table.c[col] == row[col] for col in key_cols]
        with self.db.begin() as conn:
            existing = conn.execute(select(self.table).where(*conditions)).first()
            if existing is None:
                conn.execute(insert(self.table), [row])
                return 'inserted'
            payload = {k: v for k, v in row.items() if k not in key_cols}
            if payload:
                conn.execute(update(self.table).where(*conditions).values(**payload))
            return 'updated'

    def find_all(self, order_by=None):
        stmt = select(self.table)
        if order_by:
            stmt = stmt.order_by(*[self.table.c[c] for c in _as_list(order_by)])
        return self._fetch(stmt)

    def find_by(self, order_by=None, **filters):
        stmt = select(self.table)
        for col, value in filters.items():
            stmt = stmt.where(self.table.c[col] == value)
        if order_by:
            stmt = stmt.order_by(*[self.table.c[c] for c in _as_list(order_by)])
        return self._fetch(stmt)

    def delete_by(self, **filters):
        """按条件删除，返回受影响行数。

        拒绝无条件删除：全表清空须显式调用 delete_all()。
        """
        if not filters:
            raise ValueError(
                'delete_by 需要至少一个过滤条件；确实要清空整表请显式调用 delete_all()'
            )
        stmt = delete(self.table)
        for col, value in filters.items():
            stmt = stmt.where(self.table.c[col] == value)
        with self.db.begin() as conn:
            return conn.execute(stmt).rowcount

    def delete_all(self):
        """清空整表，返回受影响行数。调用点应当显眼。"""
        with self.db.begin() as conn:
            return conn.execute(delete(self.table)).rowcount

    def count(self):
        with self.db.connect() as conn:
            return conn.execute(select(func.count()).select_from(self.table)).scalar()

    def _fetch(self, stmt):
        with self.db.connect() as conn:
            return [dict(r) for r in conn.execute(stmt).mappings()]

    def _reject_unknown_columns(self, row):
        unknown = set(row) - set(self.table.c.keys())
        if unknown:
            raise ValueError(
                f'{self.table.name} 不存在这些列: {sorted(unknown)}'
            )


def _as_list(value):
    return [value] if isinstance(value, str) else list(value)
