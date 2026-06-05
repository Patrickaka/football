import logging
import logging.handlers
import os
import sys
import time
from functools import wraps
from pathlib import Path

LOG_LEVEL = os.environ.get('FOOTBALL_LOG_LEVEL', 'INFO').upper()

LOG_DIR = Path(os.environ.get('FOOTBALL_LOG_DIR', Path(__file__).parent / 'logs'))

_FORMATTER = logging.Formatter(
    '%(asctime)s [%(levelname)-5s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)

# 小时级轮转，保留 24 个文件 = 1 天
_FILE_HANDLER = None


class _SafeTimedRotatingFileHandler(logging.handlers.TimedRotatingFileHandler):
    """安全的TimedRotatingFileHandler，在Windows文件占用时忽略轮转错误"""
    
    def doRollover(self):
        try:
            super().doRollover()
        except PermissionError:
            # 文件被其他进程占用，跳过本轮轮转
            pass

    def rotate(self, source, dest):
        try:
            super().rotate(source, dest)
        except PermissionError:
            # 文件被占用，跳过轮转
            pass


def _ensure_file_handler():
    global _FILE_HANDLER
    if _FILE_HANDLER is not None:
        return _FILE_HANDLER
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = str(LOG_DIR / 'football.log')
    _FILE_HANDLER = _SafeTimedRotatingFileHandler(
        log_file, when='H', interval=1, backupCount=24, encoding='utf-8',
    )
    _FILE_HANDLER.setFormatter(_FORMATTER)
    _FILE_HANDLER.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))
    return _FILE_HANDLER


def setup_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))
    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(_FORMATTER)
        logger.addHandler(handler)
    if not any(isinstance(h, logging.handlers.TimedRotatingFileHandler) for h in logger.handlers):
        logger.addHandler(_ensure_file_handler())
    logger.propagate = False
    return logger


def log_duration(logger=None):
    """装饰器：记录函数耗时。logger 为 None 时从第一个参数取 logger 属性。"""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            log = logger
            if log is None and args:
                log = getattr(args[0], 'logger', None)
            if log is None:
                log = logging.getLogger(func.__module__)
            start = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                elapsed = time.perf_counter() - start
                log.debug('%s 完成 (%.3fs)', func.__name__, elapsed)
                return result
            except Exception:
                elapsed = time.perf_counter() - start
                log.error('%s 异常 (%.3fs)', func.__name__, elapsed, exc_info=True)
                raise

        return wrapper

    if callable(logger):
        return decorator(logger)
    return decorator
