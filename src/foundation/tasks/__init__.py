"""后台任务调度：并发受限、按优先级执行。

旧实现在启动瞬间并发拉起 4 个预热线程加篮球调度器与维护任务，
在 3.6G 内存的机器上直接把 CPU 打满。
"""
from .scheduler import TaskScheduler

__all__ = ['TaskScheduler']
