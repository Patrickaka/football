"""服务入口。取代原 server.py 的 ThreadingHTTPServer。

内存受限（3.6G，available 不足 1G），固定单 worker + 线程池，
不启多 worker。
"""
import os

import uvicorn

from src.api.app import create_app

app = create_app()


def server_config():
    return {
        'host': os.getenv('FOOTBALL_HOST', '0.0.0.0'),
        'port': int(os.getenv('FOOTBALL_PORT', '9000')),
        'workers': 1,
        'log_level': os.getenv('LOG_LEVEL', 'info'),
    }


def main():
    config = server_config()
    uvicorn.run(
        'main:app',
        host=config['host'],
        port=config['port'],
        workers=config['workers'],
        log_level=config['log_level'],
    )


if __name__ == '__main__':
    main()
