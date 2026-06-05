"""项目路径锚点。

数据文件统一放在仓库根的 data/ 下，避免依赖运行时工作目录。
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / 'data'
WEB_DIR = PROJECT_ROOT / 'web'

DATA_DIR.mkdir(exist_ok=True)


def data_path(name: str) -> str:
    """返回 data/ 下文件的绝对路径字符串。"""
    return str(DATA_DIR / name)
