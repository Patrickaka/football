# -*- coding: utf-8 -*-
"""快乐8兼容缓存状态。

快乐8主读取路径已经使用 foundation/cache 与自身的跨进程预测缓存；这里仅保留
刷新、抓取和手动删号接口仍会同步更新的进程内兼容状态。
"""

from .lazy_modules import KL8_PREDICTOR_VERSION


def _current_kl8_predictor_version():
    """从模块读取当前版本，使缓存键跟随代码热更新。"""
    try:
        import src.kl8 as kl8_module

        return getattr(kl8_module, 'KL8_PREDICTOR_VERSION', KL8_PREDICTOR_VERSION)
    except Exception:
        return KL8_PREDICTOR_VERSION


_CACHE = {
    'kl8': {
        'data': None,
        'timestamp': 0,
        'expire_seconds': 86400,
    },
}
