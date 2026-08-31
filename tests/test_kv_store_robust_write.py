# -*- coding: utf-8 -*-
"""kv_store 原子写入在 Windows WinError 5 (拒绝访问) 下的鲁棒性回归测试。

覆盖三种场景：
  1. os.replace 前两次抛 PermissionError 后成功 -> 数据落盘、重试生效。
  2. os.replace 永远失败 -> 兜底直接写仍保证数据不丢失。
  3. 多线程并发写不同 key -> 锁串行化，两个 key 都完整存在。
"""
import os
import sys
import json
import tempfile
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import src.common.kv_store as kv  # noqa: E402

TMP = tempfile.mkdtemp()
TARGET = os.path.join(TMP, "kv_store_fallback.json")
kv._fallback_path = lambda: TARGET


def _reset():
    if os.path.exists(TARGET):
        os.remove(TARGET)
    if os.path.exists(TARGET + ".tmp"):
        os.remove(TARGET + ".tmp")
    kv._FALLBACK_CACHE = None
    kv._FALLBACK_CACHE_SIG = None


def test_retry_then_success():
    _reset()
    calls = {"n": 0}
    orig = os.replace

    def flaky(src, dst):
        calls["n"] += 1
        if calls["n"] <= 2:
            raise PermissionError("[WinError 5] 拒绝访问。")
        return orig(src, dst)

    kv.os.replace = flaky
    try:
        kv._fallback_save("k1", {"a": 1})
        data = kv._fallback_load("k1")
        assert data == {"a": 1}, f"期望落盘 {{'a':1}}，实际 {data}"
        assert calls["n"] >= 3, f"期望重试 >=3 次，实际 {calls['n']}"
    finally:
        kv.os.replace = orig
    print("[OK] test_retry_then_success: 重试后写入成功")


def test_always_fail_fallback_direct_write():
    _reset()
    # `kv.os` 与 `os` 是同一个模块对象，赋值会改掉**全局**的 os.replace，
    # 所以真函数必须先存下来——写成 `kv.os.replace = os.replace` 等于拿
    # 打过桩的自己还原自己，os.replace 会永久停在"总是失败"上：后面
    # test_concurrent_writes 的 100 次写入每次跑满 6 轮重试与退避，
    # 单这一条就要 100 秒，而且整个进程里所有 os.replace 都被带偏。
    orig = os.replace

    def always_fail(src, dst):
        raise PermissionError("[WinError 5] 拒绝访问。")

    kv.os.replace = always_fail
    try:
        kv._fallback_save("k2", {"b": 2})
        # 直接写兜底：目标文件应已包含数据（即便 .tmp->.json 改名失败）
        assert os.path.exists(TARGET), "兜底直接写后目标文件应存在"
        with open(TARGET, "r", encoding="utf-8") as f:
            raw = json.load(f)
        assert raw["k2"]["json_value"] == json.dumps({"b": 2}, ensure_ascii=False), \
            f"兜底数据缺失: {raw}"
        # 通过正常接口也能读回
        assert kv._fallback_load("k2") == {"b": 2}
    finally:
        kv.os.replace = orig
    print("[OK] test_always_fail_fallback_direct_write: 兜底写入保证不丢数据")


def test_concurrent_writes():
    _reset()
    errors = []

    def worker(key, val):
        try:
            for i in range(20):
                kv._fallback_save(key, {"i": i})
        except Exception as e:  # pragma: no cover
            errors.append(e)

    ts = [threading.Thread(target=worker, args=(f"t{k}", k)) for k in range(5)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()

    assert not errors, f"并发写出现异常: {errors}"
    for k in range(5):
        v = kv._fallback_load(f"t{k}")
        assert v is not None and "i" in v, f"key t{k} 丢失或损坏: {v}"
    print("[OK] test_concurrent_writes: 5 线程各写 20 次均完好")


if __name__ == "__main__":
    test_retry_then_success()
    test_always_fail_fallback_direct_write()
    test_concurrent_writes()
    print("\n全部 kv_store 鲁棒性测试通过 ✅")
