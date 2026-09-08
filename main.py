"""服务入口。取代原 server.py 的 ThreadingHTTPServer。

内存受限（3.6G，available 不足 1G），固定单 worker + 线程池，
不启多 worker。
"""
import os
import sys

import uvicorn

from src.api.app import create_app

app = create_app()


def server_config():
    return {
        'host': os.getenv('FOOTBALL_HOST', '0.0.0.0'),
        'port': int(os.getenv('FOOTBALL_PORT', '9004')),
        'workers': 1,
        'log_level': os.getenv('LOG_LEVEL', 'info'),
    }


def main(*, sockets=None):
    # 编码设置属于进程入口——库模块顶层做这件事会换掉调用方的 stdout，
    # 在 pytest 下表现为整套测试从某个用例起全红且查不出关联。
    sys.stdout.reconfigure(encoding='utf-8')
    config = server_config()
    if sockets is not None:
        # 本地启动器已预留端口；沿用原 socket，避免初始化完成后再抢端口。
        server = uvicorn.Server(uvicorn.Config('main:app', **config))
        server.run(sockets=sockets)
        if not server.started:
            raise SystemExit(3)
        return
    uvicorn.run(
        'main:app',
        host=config['host'],
        port=config['port'],
        workers=config['workers'],
        log_level=config['log_level'],
    )


if __name__ == '__main__':
    main()
