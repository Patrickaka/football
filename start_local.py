#!/usr/bin/env python3
"""跨平台本地服务启动器。

直接运行：

    python start_local.py

查看运行状态：``python start_local.py --status``。
指定其他端口：``python start_local.py --port 9005``。

首次运行会创建项目内的 ``.venv`` 并安装 ``requirements.txt``；之后直接启动
FastAPI 服务。Windows 使用 ``.venv/Scripts/python.exe``，Linux/macOS 使用
``.venv/bin/python``。
"""

from __future__ import annotations

import argparse
import errno
from http.client import HTTPException
import importlib.util
import json
import os
from pathlib import Path
import platform
import socket
import subprocess
import sys
from urllib.error import HTTPError
from urllib.request import HTTPRedirectHandler, ProxyHandler, build_opener


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


def _port_number(value: str) -> int:
    try:
        port = int(value)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError("端口必须是 1 到 65535 的整数") from None
    if not 1 <= port <= 65535:
        raise argparse.ArgumentTypeError("端口必须是 1 到 65535 的整数")
    return port


def _display_host(host: str) -> str:
    return {"0.0.0.0": "127.0.0.1", "": "127.0.0.1", "::": "::1"}.get(host, host)


def _display_url(host: str, port: int) -> str:
    host = _display_host(host)
    return f"http://{'[' + host + ']' if ':' in host else host}:{port}"


def _reserve_listeners(host: str, port: int) -> list[socket.socket]:
    """在导入应用前占住端口，并将同一组 socket 交给 Uvicorn。"""
    listeners = []
    seen = set()
    unavailable = {errno.EAFNOSUPPORT, errno.EPROTONOSUPPORT, errno.EADDRNOTAVAIL}
    last_unavailable = None
    try:
        addresses = socket.getaddrinfo(
            host or None, port, socket.AF_UNSPEC, socket.SOCK_STREAM, 0, socket.AI_PASSIVE,
        )
        for family, socktype, protocol, _, address in addresses:
            if family not in (socket.AF_INET, socket.AF_INET6) or (family, address) in seen:
                continue
            seen.add((family, address))
            try:
                listener = socket.socket(family, socktype, protocol)
            except OSError as error:
                if error.errno not in unavailable:
                    raise
                last_unavailable = error
                continue
            listeners.append(listener)
            try:
                if os.name == "nt":
                    listener.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
                else:
                    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                if family == socket.AF_INET6:
                    listener.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 1)
                listener.bind(address)
                listener.listen(socket.SOMAXCONN)
                listener.setblocking(False)
            except OSError as error:
                if error.errno not in unavailable:
                    raise
                listener.close()
                listeners.pop()
                last_unavailable = error
        if not listeners:
            raise last_unavailable or OSError(errno.EADDRNOTAVAIL, "没有可用的本机监听地址")
        return listeners
    except BaseException:
        for listener in listeners:
            listener.close()
        raise


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _existing_football_service(host: str, port: int) -> bool:
    """只读服务元数据，不触发数据库健康检查，也不信任任意 HTTP 200。"""
    try:
        opener = build_opener(ProxyHandler({}), _NoRedirect())
        with opener.open(_display_url(host, port) + "/openapi.json", timeout=2) as response:
            payload = response.read(1024 * 1024 + 1)
        if len(payload) > 1024 * 1024:
            return False
        schema = json.loads(payload)
        return (isinstance(schema, dict)
                and isinstance(schema.get("info"), dict)
                and schema["info"].get("title") == "Football 预测服务"
                and isinstance(schema.get("paths"), dict)
                and {"/api/matches", "/api/predictions", "/healthz"} <= schema["paths"].keys())
    except HTTPError as error:
        error.close()
        return False
    except (OSError, ValueError, HTTPException):
        return False


def _show_running_service(host: str, port: int) -> None:
    print(f"[启动] 该地址已有 Football 服务运行：{_display_url(host, port)}", flush=True)
    print("[启动] 已跳过重复启动，可直接访问上面的地址。", flush=True)
    print("[提示] 如需加载修改后的代码，请先在原服务窗口按 Ctrl+C 停止，再重新启动。", flush=True)


def _handle_bind_error(host: str, port: int, error: OSError) -> None:
    occupied = error.errno == errno.EADDRINUSE or getattr(error, "winerror", None) == 10048
    forbidden = error.errno in (errno.EACCES, errno.EPERM)
    if (occupied or forbidden) and _existing_football_service(host, port):
        _show_running_service(host, port)
        return
    if occupied:
        reason = "端口已被占用，未确认是 Football 服务"
    elif forbidden:
        reason = "系统拒绝监听，端口可能被独占、保留或权限不足"
    elif isinstance(error, socket.gaierror):
        reason = "无法解析监听地址"
    else:
        reason = str(error)
    alternative_port = port + 1 if port < 65535 else 65534
    suggestion = (f"可用 python start_local.py --port {alternative_port} 指定其他端口。"
                  if occupied or forbidden else "请检查 --host 是否为有效的本机地址。")
    raise SystemExit(
        f"[错误] 无法监听 {host}:{port}：{reason}。\n"
        f"[提示] {suggestion}"
    )


def _show_status(host: str, port: int) -> None:
    if _existing_football_service(host, port):
        print(f"[状态] Football 服务正在运行：{_display_url(host, port)}", flush=True)
        return
    try:
        with socket.create_connection((_display_host(host), port), timeout=1):
            pass
    except OSError:
        raise SystemExit(f"[状态] 未检测到可连接的服务：{_display_url(host, port)}") from None
    raise SystemExit(f"[状态] 端口可连接，但未确认是 Football 服务：{_display_url(host, port)}")


def _arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="启动 Football 本地 FastAPI 服务")
    parser.add_argument("--host", default=os.getenv("FOOTBALL_HOST", "127.0.0.1"),
                        help="监听地址，默认 127.0.0.1；局域网访问可用 0.0.0.0")
    parser.add_argument("--port", type=_port_number,
                        default=os.getenv("FOOTBALL_PORT", "9004"),
                        help="监听端口，默认 9004")
    parser.add_argument("--status", action="store_true",
                        help="查看指定地址的服务状态，不启动服务或安装依赖")
    parser.add_argument("--no-install", action="store_true",
                        help="依赖缺失时不自动安装")
    tasks = parser.add_mutually_exclusive_group()
    tasks.add_argument("--startup-tasks", action="store_true", default=None,
                       help="开启缓存预热、定时回填等后台任务")
    tasks.add_argument("--no-startup-tasks", action="store_false",
                       dest="startup_tasks", help="关闭后台任务（本地默认）")
    return parser.parse_args(argv)


def main() -> None:
    _configure_console()
    args = _arguments()
    if args.status:
        _show_status(args.host, args.port)
        return
    os.chdir(ROOT)
    _create_or_enter_venv()

    try:
        listeners = _reserve_listeners(args.host, args.port)
    except OSError as exc:
        _handle_bind_error(args.host, args.port, exc)
        return
    try:
        _start_service(args, listeners)
    except KeyboardInterrupt:
        print("\n[启动] 服务已停止", flush=True)
    finally:
        for listener in listeners:
            listener.close()


def _start_service(args: argparse.Namespace, listeners: list[socket.socket]) -> None:

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
    print(f"[启动] 系统: {system}", flush=True)
    print(f"[启动] Python: {sys.executable}", flush=True)
    print(f"[启动] 后台任务: {'开启' if startup_tasks else '关闭'}", flush=True)
    print(f"[启动] 访问地址: {_display_url(args.host, args.port)}", flush=True)
    print("[启动] 按 Ctrl+C 停止服务", flush=True)

    try:
        import main as service_entry
    except ModuleNotFoundError as exc:
        raise SystemExit(f"[错误] 服务模块导入失败，缺少依赖: {exc.name}") from exc
    service_entry.main(sockets=listeners)


if __name__ == "__main__":
    main()
