import logging
import pathlib
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from starlette.middleware.gzip import GZipMiddleware
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from src.api.auth import AuthSettings, build_session_manager, install_auth
from src.api.deps import Settings, build_cache, build_database, get_executor, shutdown_executor
from src.api.rate_limit import ClientRateLimiters, install_rate_limit
from src.api.routers import auth as auth_routes
from src.api.routers import basketball, beidan, football, health, kl8, lottery, pages
from src.api import startup as startup_orchestration
from src.webapp import background

log = logging.getLogger('api.app')


LOGIN_PAGE = pathlib.Path(__file__).resolve().parents[2] / 'web' / 'login.html'


def create_app(settings=None, auth_settings=None):
    settings = settings or Settings.from_env()
    auth_settings = auth_settings or AuthSettings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.settings = settings
        app.state.cache = build_cache(settings)
        # 会话存 L2：生产态是 Redis，跨进程重启保留，撤销也是真的撤销。
        # 降级为内存时重启即全员登出——是预期行为，不是故障。
        app.state.auth = auth_settings
        app.state.sessions = build_session_manager(app.state.cache.l2, auth_settings)
        if not auth_settings.enabled:
            log.warning('鉴权未启用（未配置 FOOTBALL_USERS）——所有接口对外开放')
        else:
            log.info('鉴权已启用，用户: %s', ', '.join(sorted(auth_settings.credentials)))
        app.state.db = build_database(settings)
        # **用进程级的那一个调度器，不再另建一个**。原来这里建了个空的
        # `TaskScheduler` 只为让健康检查有东西可看——它永远 0 个任务、
        # 永远不 start()，是个纯粹的摆设。真正跑着三族周期任务的是
        # `src.webapp.background` 的单例；健康检查要看的是那一个。
        app.state.tasks = background.scheduler()
        # get_executor 是模块级全局单例，只有首次调用的 workers 参数生效；
        # 必须在任何 run_blocking(...) 之前、在此显式用 settings.executor_workers
        # 完成首次初始化，否则该字段会因为“从未被首次调用消费”而形同虚设。
        get_executor(settings.executor_workers)

        # 磁盘清理、缓存恢复、三族后台任务、三个预热线程、周期维护。
        # **漏掉任何一件都不会让服务起不来**，只会安静地少干活——
        # 后台不再回填赛果、缓存不再跨重启保留、用户重新承担冷计算。
        if settings.run_startup_tasks:
            startup_orchestration.run_all()
        else:
            log.info('启动编排已跳过（RUN_STARTUP_TASKS=0）')
        log.info('API 启动完成')
        yield
        # 关闭顺序：先排空 SWR 后台刷新，再停消费者，最后释放消费者依赖的资源。
        # 1. cache.wait_for_refreshes：SWR 刷新线程是 daemon，进程退出即被杀，
        #    `finally: self.l2.unlock(key)` 得不到执行，会在 Redis 残留一把
        #    TTL 最长 lock_timeout 秒的锁——重启后第一个请求撞上这把锁，
        #    复现 P1 惊群。必须在关闭序列的最前面排空。
        # 2. tasks.shutdown → shutdown_executor：先停消费者（任务调度器与
        #    线程池），再释放它们依赖的资源（db），顺序不能反——旧写法先
        #    dispose db 再关 executor，executor 里仍在跑的任务可能这期间
        #    还在用 db。
        app.state.cache.wait_for_refreshes(timeout=app.state.cache.lock_timeout)
        background.shutdown(wait=True)
        shutdown_executor()
        app.state.db.dispose()
        log.info('API 已停止')

    app = FastAPI(title='Football 预测服务', version='2.0.0', lifespan=lifespan)
    # 登录页在建 app 时读一次。读不到不该让服务起不来——鉴权本身照常工作，
    # 只是登录页显示一句提示（`/auth/login` 仍可直接调用）。
    try:
        app.state.login_page = LOGIN_PAGE.read_text(encoding='utf-8')
    except OSError as exc:
        log.error('登录页读取失败（%s）：%s', LOGIN_PAGE, exc)
        app.state.login_page = '<!doctype html><meta charset="utf-8">登录页缺失'
    app.state.auth = auth_settings
    install_validation_error_shape(app)
    app.include_router(health.router)
    app.include_router(auth_routes.router)
    app.include_router(basketball.router)
    app.include_router(beidan.router)
    app.include_router(lottery.router)
    app.include_router(kl8.router)
    app.include_router(football.router)
    app.include_router(pages.router)

    # 中间件按**注册的逆序**执行：后注册的先跑。限流要排在鉴权前面，
    # 否则未登录的洪水请求会先去查一遍会话（打 Redis）再被 401 挡下——
    # 那正好是最不该在被攻击时做的事。
    # **接口返回的是高度重复的 JSON，压缩比能到数倍以上**（北单整页
    # 332 KB → 45 KB，一次批量预测 456 KB），代价只有几毫秒 CPU。
    # 旧入口一直在压，切过来时漏了——那是手机端能直接感觉到的降级，
    # 而且不会有任何报错。
    #
    # **必须最先注册**（= 最内层，紧贴路由）：`@app.middleware('http')`
    # 加的是 `BaseHTTPMiddleware`，它把响应转成流式、丢掉 `Content-Length`。
    # GZip 拿不到长度就只能一律压缩，`minimum_size` 形同虚设——
    # 8 字节的 `{"ok":1}` 也会被压，白白多出压缩头。
    app.add_middleware(GZipMiddleware, minimum_size=settings.gzip_min_bytes,
                       compresslevel=settings.gzip_level)
    install_auth(app)
    install_rate_limit(app, build_rate_limiters(settings))
    return app


def install_validation_error_shape(app):
    """参数校验失败时，在 FastAPI 的 `detail` 之外补一个 `error` 字段。

    状态码保持 422——那是正确的语义，旧入口把参数错误伪装成 `200` 里的
    一句 error 字符串，不该照搬。但**网页整套是按 `data.error` 判错的**
    （`web/index.html` 里 `if (data.error)`），只给 `detail` 的话前端会
    以为请求成功、然后拿不到数据，页面空白且没有任何提示。

    两边都给：机器读 `detail`，网页读 `error`。
    """

    @app.exception_handler(RequestValidationError)
    async def _shape(request: Request, exc: RequestValidationError):
        first = (exc.errors() or [{}])[0]
        field = '.'.join(str(part) for part in first.get('loc', ())[1:]) or '参数'
        return JSONResponse(
            {'error': f"参数 {field} 无效: {first.get('msg', '校验失败')}",
             'detail': exc.errors()},
            status_code=422,
        )

    return app


def build_rate_limiters(settings):
    """限流器；未配置速率则返回 None（不限流）。"""
    if settings.rate_limit_per_sec <= 0:
        log.info('入站限流未启用（RATE_LIMIT_PER_SEC 未配置）')
        return None
    log.info('入站限流已启用：%.1f 次/秒，突发 %d',
             settings.rate_limit_per_sec, settings.rate_limit_burst)
    return ClientRateLimiters(
        rate_per_sec=settings.rate_limit_per_sec,
        burst=settings.rate_limit_burst,
        maxsize=settings.rate_limit_clients,
    )
