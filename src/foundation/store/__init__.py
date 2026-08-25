"""存储层：SQLAlchemy Core 之上的 Repository 实现。

MySQL 为唯一真相源，不提供 JSON 文件兜底 —— 旧 kv_store/doc_store 的
双轨制会在静默降级时给出错误结论，本层不重复该设计。
"""
from .engine import Database, make_engine, mysql_url_from_env

__all__ = ['Database', 'make_engine', 'mysql_url_from_env']
