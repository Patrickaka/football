import hashlib
import os
import threading


class SnapshotStore:
    """响应快照。用于离线回归测试与抓取失败时的兜底。"""

    def __init__(self, root):
        self.root = root
        os.makedirs(self.root, exist_ok=True)

    def save(self, url, body):
        path = self._path(url)
        # tmp 文件名必须带上进程号+线程号：两个线程并发 save 同一个 url 时若共用
        # 同一个 tmp 路径，会出现"A 正在写 tmp，B 打开同名 tmp 截断重写，A 的
        # os.replace 把 B 尚未写完的内容搬到正式路径，B 随后 os.replace 时 tmp
        # 已被 A 搬走"这类内容错乱/FileNotFoundError。各写各的 tmp 后，
        # os.replace 本身的原子性才能真正生效：最终落地的是最后一次 replace
        # 的完整内容，不会是两次写入交织出的半成品。
        tmp = f'{path}.{os.getpid()}.{threading.get_ident()}.tmp'
        with open(tmp, 'w', encoding='utf-8') as handle:
            handle.write(body)
        os.replace(tmp, path)

    def load(self, url):
        path = self._path(url)
        if not os.path.exists(path):
            return None
        with open(path, encoding='utf-8') as handle:
            return handle.read()

    def _path(self, url):
        digest = hashlib.sha256(url.encode('utf-8')).hexdigest()[:32]
        return os.path.join(self.root, f'{digest}.snap')
