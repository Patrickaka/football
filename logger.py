import logging
import os
import sys
import time
from functools import wraps

LOG_LEVEL = os.environ.get('FOOTBALL_LOG_LEVEL', 'INFO').upper()

_FORMATTER = logging.Formatter(
    '%(asctime)s [%(levelname)-5s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)


def setup_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))
    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(_FORMATTER)
        logger.addHandler(handler)
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
