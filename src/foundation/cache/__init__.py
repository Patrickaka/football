"""缓存层：L1 进程内存 + L2 Redis，统一单飞与 SWR。

对外只暴露 get / invalidate。禁止任何调用方直接操作缓存结构——
旧实现中 kl8 绕过 _serve_cached 手搓 _CACHE 字典，导致缓存过期时
并发请求同时触发 6 秒重算。
"""
