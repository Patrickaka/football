#!/usr/bin/env python3
"""跨平台本地服务启动器。

直接运行：

    python start_local.py

首次运行会创建项目内的 ``.venv`` 并安装 ``requirements.txt``；之后直接启动
FastAPI 服务。Windows 使用 ``.venv/Scripts/python.exe``，Linux/macOS 使用
``.venv/bin/python``。
"""

from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path
import platform
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
REQUIREMENTS = ROOT / "requirements.txt"


def _configure_console() -> None:
    """Windows 控制台和重定向输出都统一为 UTF-8。"""
    if os.name == "nt":
        try:
            import ctypes
            ctypes.windll.kernel32.SetConsoleOutputCP(65001)
            ctypes.windll.kernel32.SetConsoleCP(65001)
        except (AttributeError, OSError):
            pass
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure:
            reconfigure(encoding="utf-8", errors="replace")


_configure_console()

# requirements.txt 中发行包名与 import 名并不总是相同。
RUNTIME_MODULES = (
    "pymysql", "requests", "urllib3", "certifi", "charset_normalizer", "idna",
    "apscheduler", "numpy", "pandas", "scipy", "sklearn", "lightgbm",
    "xgboost", "catboost", "fastapi", "uvicorn", "pydantic", "starlette",
    "sqlalchemy", "redis", "click", "h11", "anyio", "typing_extensions",
    "typing_inspection", "annotated_types", "pydantic_core",
)


def _project_python() -> Path:
    if os.name == "nt":
        return ROOT / ".venv" / "Scripts" / "python.exe"
    return ROOT / ".venv" / "bin" / "python"


def _same_executable(left: Path, right: Path) -> bool:
    try:
        return left.resolve() == right.resolve()
    except OSError:
        return os.path.normcase(str(left)) == os.path.normcase(str(right))


def _create_or_enter_venv() -> None:
    """创建项目虚拟环境，并把当前进程替换为虚拟环境 Python。"""
    target = _project_python()
    current = Path(sys.executable)
    if target.exists():
        if not _same_executable(current, target):
            os.execv(str(target), [str(target), str(Path(__file__).resolve()), *sys.argv[1:]])
        return

    print(f"[启动] 未发现虚拟环境，正在创建: {ROOT / '.venv'}", flush=True)
    try:
        subprocess.check_call([sys.executable, "-m", "venv", str(ROOT / ".venv")])
    except (OSError, subprocess.CalledProcessError) as exc:
        raise SystemExit(f"[错误] 创建虚拟环境失败: {exc}") from exc

    if not target.exists():
        raise SystemExit(f"[错误] 虚拟环境创建完成，但未找到 Python: {target}")
    os.execv(str(target), [str(target), str(Path(__file__).resolve()), *sys.argv[1:]])


def _missing_modules() -> list[str]:
    return [name for name in RUNTIME_MODULES if importlib.util.find_spec(name) is None]


def _install_dependencies() -> None:
    missing = _missing_modules()
    if not missing:
        return
    if not REQUIREMENTS.exists():
        raise SystemExit(f"[错误] 缺少依赖且找不到 {REQUIREMENTS}")

    preview = ", ".join(missing[:8]) + ("…" if len(missing) > 8 else "")
    print(f"[启动] 首次运行，正在安装依赖: {preview}", flush=True)
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", str(REQUIREMENTS),
        ], cwd=str(ROOT))
    except (OSError, subprocess.CalledProcessError) as exc:
        command = (
            r".\.venv\Scripts\python.exe -m pip install -r requirements.txt"
            if os.name == "nt"
            else ".venv/bin/python -m pip install -r requirements.txt"
        )
        raise SystemExit(
            f"[错误] 自动安装依赖失败: {exc}\n请在项目目录手动执行：\n{command}"
        ) from exc

    still_missing = _missing_modules()
    if still_missing:
        raise SystemExit("[错误] 安装后仍缺少模块: " + ", ".join(still_missing))


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="启动 Football 本地 FastAPI 服务")
    parser.add_argument("--host", default=os.getenv("FOOTBALL_HOST", "127.0.0.1"),
                        help="监听地址，默认 127.0.0.1；局域网访问可用 0.0.0.0")
    parser.add_argument("--port", type=int,
                        default=int(os.getenv("FOOTBALL_PORT", "9004")),
                        help="监听端口，默认 9004")
    parser.add_argument("--no-install", action="store_true",
                        help="依赖缺失时不自动安装")
    tasks = parser.add_mutually_exclusive_group()
    tasks.add_argument("--startup-tasks", action="store_true", default=None,
                       help="开启缓存预热、定时回填等后台任务")
    tasks.add_argument("--no-startup-tasks", action="store_false",
                       dest="startup_tasks", help="关闭后台任务（本地默认）")
    return parser.parse_args()


def main() -> None:
    args = _arguments()
    os.chdir(ROOT)
    _create_or_enter_venv()

    missing = _missing_modules()
    if missing and args.no_install:
        raise SystemExit("[错误] 缺少运行依赖: " + ", ".join(missing))
    if missing:
        _install_dependencies()

    startup_tasks = args.startup_tasks
    if startup_tasks is None:
        startup_tasks = os.getenv("RUN_STARTUP_TASKS", "0").strip().lower() not in {
            "0", "false", "no", "off",
        }

    os.environ["FOOTBALL_HOST"] = args.host
    os.environ["FOOTBALL_PORT"] = str(args.port)
    os.environ["RUN_STARTUP_TASKS"] = "1" if startup_tasks else "0"

    system = platform.system() or os.name
    display_host = "127.0.0.1" if args.host == "0.0.0.0" else args.host
    print(f"[启动] 系统: {system}", flush=True)
    print(f"[启动] Python: {sys.executable}", flush=True)
    print(f"[启动] 后台任务: {'开启' if startup_tasks else '关闭'}", flush=True)
    print(f"[启动] 访问地址: http://{display_host}:{args.port}", flush=True)
    print("[启动] 按 Ctrl+C 停止服务", flush=True)

    try:
        import main as service_entry
    except ModuleNotFoundError as exc:
        raise SystemExit(f"[错误] 服务模块导入失败，缺少依赖: {exc.name}") from exc
    service_entry.main()


if __name__ == "__main__":
    main()
