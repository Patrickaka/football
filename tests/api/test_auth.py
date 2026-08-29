# -*- coding: utf-8 -*-
"""新入口的鉴权：会话 Cookie + 登录页。

旧入口用的是 HTTP Basic（浏览器原生弹窗、不能登出、每个请求原样重发密码）。
这一版换成服务端会话：登录页签发 `HttpOnly` Cookie，登出真的撤销。

**凭据来源没变**：仍是 `FOOTBALL_USERS=甲:密码1,乙:密码2` 环境变量，
解析函数与旧的 `src/webapp/settings.py::_load_credentials` 共用同一份
（判据 11），4014 条语料逐条比对过。

## 拦截方式：默认全拦，白名单豁免

不是逐个路由挂 `Depends(...)`。**漏挂一个 `Depends` 不会有任何东西报错**，
那条路由就此裸奔；而漏加一条豁免会立刻以 401 暴露出来。两种错误的代价
不对称，所以选会响的那种。豁免清单由本文件盯着。
"""
import unittest

from fastapi.testclient import TestClient

from src.api.app import create_app
from src.api.auth import PUBLIC_PATHS, SESSION_COOKIE, AuthSettings
from src.foundation.auth import parse_credentials, verify_password

USERS = {'lijl': 'pw-one', 'zhangao': 'pw-two'}


def make_client(**overrides):
    settings = AuthSettings(credentials=dict(USERS), **overrides)
    return TestClient(create_app(auth_settings=settings))


class CredentialParsing(unittest.TestCase):

    def test_the_multi_user_form_is_parsed(self):
        self.assertEqual(parse_credentials({'FOOTBALL_USERS': 'a:1,b:2'}),
                         {'a': '1', 'b': '2'})

    def test_padding_is_trimmed(self):
        self.assertEqual(parse_credentials({'FOOTBALL_USERS': ' a : 1 '}), {'a': '1'})

    def test_half_empty_entries_are_dropped(self):
        """**"配了等于没配"要当成没配**，否则看起来像已启用其实是空的。"""
        for broken in ('a:', ':1', 'a', '', ',,,', ':'):
            with self.subTest(value=broken):
                self.assertEqual(parse_credentials({'FOOTBALL_USERS': broken}), {})

    def test_a_password_may_contain_a_colon(self):
        self.assertEqual(parse_credentials({'FOOTBALL_USERS': 'a:p:q'}), {'a': 'p:q'})

    def test_the_single_user_form_also_works(self):
        self.assertEqual(
            parse_credentials({'FOOTBALL_USER': 'x', 'FOOTBALL_PASS': 'y'}), {'x': 'y'})

    def test_the_single_user_form_does_not_override_the_list(self):
        """**反方向**：两种配置同时存在时，列表里的那份优先。"""
        self.assertEqual(
            parse_credentials({'FOOTBALL_USERS': 'x:1',
                               'FOOTBALL_USER': 'x', 'FOOTBALL_PASS': 'y'}),
            {'x': '1'})

    def test_no_configuration_means_no_credentials(self):
        self.assertEqual(parse_credentials({}), {})


class PasswordVerification(unittest.TestCase):

    def test_the_right_password_passes(self):
        self.assertTrue(verify_password('lijl', 'pw-one', USERS))

    def test_a_wrong_password_fails(self):
        self.assertFalse(verify_password('lijl', 'pw-two', USERS))

    def test_an_unknown_user_fails(self):
        self.assertFalse(verify_password('nobody', 'pw-one', USERS))

    def test_empty_credentials_never_pass(self):
        """**凭据为空一律不放行**——"要不要鉴权"是调用方的判断，不是这里的。"""
        self.assertFalse(verify_password('lijl', 'pw-one', {}))

    def test_non_string_input_is_rejected_rather_than_crashing(self):
        for user, password in ((None, 'x'), ('lijl', None), (1, 2), ([], {})):
            with self.subTest(user=user, password=password):
                self.assertFalse(verify_password(user, password, USERS))


class LoginFlow(unittest.TestCase):

    def test_a_valid_login_sets_an_httponly_cookie(self):
        with make_client() as client:
            response = client.post('/auth/login',
                                   json={'user': 'lijl', 'password': 'pw-one'})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json(), {'user': 'lijl'})
            cookie = response.headers['set-cookie']
            self.assertIn('HttpOnly', cookie)
            self.assertIn('fb_session=', cookie)

    def test_the_cookie_is_not_the_password(self):
        """会话 id 里不能出现账密——那等于把密码存进浏览器。"""
        with make_client() as client:
            response = client.post('/auth/login',
                                   json={'user': 'lijl', 'password': 'pw-one'})
            cookie = response.headers['set-cookie']
            self.assertNotIn('pw-one', cookie)
            self.assertNotIn('lijl', cookie.split(';')[0])

    def test_each_login_gets_its_own_session(self):
        """同一用户多处登录各拿各的 id，踢掉一处不波及其它。"""
        with make_client() as client:
            first = client.post('/auth/login',
                                json={'user': 'lijl', 'password': 'pw-one'})
            second = client.post('/auth/login',
                                 json={'user': 'lijl', 'password': 'pw-one'})
            self.assertNotEqual(first.cookies[SESSION_COOKIE],
                                second.cookies[SESSION_COOKIE])

    def test_a_wrong_password_is_rejected(self):
        with make_client() as client:
            response = client.post('/auth/login',
                                   json={'user': 'lijl', 'password': 'nope'})
            self.assertEqual(response.status_code, 401)
            self.assertNotIn('set-cookie', response.headers)

    def test_failure_does_not_reveal_whether_the_user_exists(self):
        """**用户不存在与密码错误必须同一句话、同一个状态码**，
        否则这就是个免费的用户名枚举接口。
        """
        with make_client() as client:
            unknown = client.post('/auth/login',
                                  json={'user': '查无此人', 'password': 'x'})
            wrong = client.post('/auth/login',
                                json={'user': 'lijl', 'password': 'x'})
            self.assertEqual(unknown.status_code, wrong.status_code)
            self.assertEqual(unknown.json(), wrong.json())

    def test_logout_revokes_the_session_server_side(self):
        """**登出必须是服务端撤销**，不只是让浏览器忘掉 Cookie。"""
        with make_client() as client:
            login = client.post('/auth/login',
                                json={'user': 'lijl', 'password': 'pw-one'})
            session_id = login.cookies[SESSION_COOKIE]
            client.post('/auth/logout')

            client.cookies.set(SESSION_COOKIE, session_id)
            self.assertIsNone(client.get('/auth/me').json()['user'])

    def test_logout_is_idempotent(self):
        with make_client() as client:
            self.assertEqual(client.post('/auth/logout').status_code, 200)

    def test_a_forged_cookie_does_not_authenticate(self):
        with make_client() as client:
            client.cookies.set(SESSION_COOKIE, 'made-up-session-id')
            self.assertIsNone(client.get('/auth/me').json()['user'])

    def test_identity_reflects_the_session_on_public_paths_too(self):
        """`/auth/me` 是豁免路径，但**它也得认领会话**。

        中间件若只在拦截路径上认领，这里就永远回 `user: null`，
        前端会以为自己没登录——看着能用、其实没在工作。
        """
        with make_client() as client:
            self.assertIsNone(client.get('/auth/me').json()['user'])
            client.post('/auth/login', json={'user': 'lijl', 'password': 'pw-one'})
            self.assertEqual(client.get('/auth/me').json()['user'], 'lijl')


class Interception(unittest.TestCase):

    PROTECTED = ('/', '/api/anything')

    def test_protected_paths_reject_anonymous_requests(self):
        with make_client() as client:
            for path in self.PROTECTED:
                with self.subTest(path=path):
                    response = client.get(path, headers={'accept': 'application/json'},
                                          follow_redirects=False)
                    self.assertEqual(response.status_code, 401)

    def test_the_home_page_is_not_public(self):
        """`/` 是业务首页——豁免它等于整个界面裸奔。旧入口同样要求凭据。"""
        self.assertNotIn('/', PUBLIC_PATHS)

    def test_a_browser_is_redirected_to_the_login_page(self):
        """页面请求跳登录页，接口请求回 401 JSON。

        给 XHR 返 302 到 HTML 登录页的话，前端只会拿到一坨 HTML，
        不知道自己该重新登录。
        """
        with make_client() as client:
            response = client.get('/', headers={'accept': 'text/html'},
                                  follow_redirects=False)
            self.assertEqual(response.status_code, 303)
            self.assertEqual(response.headers['location'], '/login')

    def test_the_redirect_carries_the_mount_prefix(self):
        """**线上就是栽在这里**：跳转写死了 `/login`，而反代
        `location /football/ { proxy_pass http://127.0.0.1:9000/; }`
        剥掉了前缀。浏览器被送到 `https://域名/login`——那条路径在反代上
        没有对应 location，用户拿到一个 openresty 的 404，根本进不来。

        本文件此前所有用例都在 `cookie_path='/'` 下跑，那种配置下
        `/login` 恰好是对的，于是**测试全绿、线上打不开**。
        """
        with make_client(cookie_path='/football/') as client:
            response = client.get('/', headers={'accept': 'text/html'},
                                  follow_redirects=False)
            self.assertEqual(response.headers['location'], '/football/login')

    def test_a_deep_path_also_redirects_to_the_prefixed_login(self):
        """**相对地址 `./login` 同样不行**：从 `/football/api/xxx` 被拦下时，
        相对解析的结果是 `/football/api/login`，一样不存在。
        """
        with make_client(cookie_path='/football/') as client:
            response = client.get('/api/deep/thing', headers={'accept': 'text/html'},
                                  follow_redirects=False)
            self.assertEqual(response.headers['location'], '/football/login')

    def test_the_prefix_comes_from_the_cookie_path(self):
        """挂载前缀与 Cookie 的 Path **必须是同一个值**（Cookie 要覆盖整个
        应用），所以不另设配置项让它们有机会不一致。
        """
        from src.api.auth import login_url
        for cookie_path, expected in (('/', '/login'),
                                      ('/football/', '/football/login'),
                                      ('/football', '/football/login'),
                                      ('/a/b/', '/a/b/login')):
            with self.subTest(cookie_path=cookie_path):
                self.assertEqual(login_url(AuthSettings(cookie_path=cookie_path)),
                                 expected)

    def test_a_script_gets_json_not_html(self):
        with make_client() as client:
            response = client.get('/', headers={'accept': 'application/json'},
                                  follow_redirects=False)
            self.assertEqual(response.status_code, 401)
            self.assertIn('detail', response.json())

    def test_the_public_list_is_exactly_these_six_paths(self):
        """豁免清单是安全边界——加路由时别默默扩它，这条会挡下来。"""
        self.assertEqual(
            PUBLIC_PATHS,
            frozenset({'/healthz', '/auth/login', '/auth/logout', '/auth/me',
                       '/login', '/login.html'}))

    def test_nothing_business_facing_is_public(self):
        """**反方向**：豁免的只能是健康探针与鉴权自身，不能有业务路径。"""
        for path in PUBLIC_PATHS:
            with self.subTest(path=path):
                self.assertTrue(path.startswith(('/auth/', '/login', '/healthz')))

    def test_public_paths_work_without_a_session(self):
        with make_client() as client:
            self.assertEqual(client.get('/healthz').status_code, 200)
            self.assertEqual(client.get('/login').status_code, 200)
            self.assertEqual(client.get('/auth/me').status_code, 200)

    def test_cors_preflight_is_let_through(self):
        """浏览器发 OPTIONS 预检时不带 Cookie——拦下来的话所有跨域请求
        都会在预检阶段失败。旧入口同样放行 OPTIONS。
        """
        with make_client() as client:
            self.assertNotEqual(client.options('/api/anything').status_code, 401)

    def test_a_trailing_slash_does_not_slip_past_the_list(self):
        """`/healthz/` 与 `/healthz` 是同一条；反过来别的路径也别想靠斜杠混进来。"""
        with make_client() as client:
            self.assertNotEqual(client.get('/healthz/').status_code, 401)
            response = client.get('/api/x/', headers={'accept': 'application/json'},
                                  follow_redirects=False)
            self.assertEqual(response.status_code, 401)


class DisabledAuth(unittest.TestCase):
    """`FOOTBALL_USERS` 没配 = 不启用鉴权，与旧入口的约定一致。"""

    def setUp(self):
        self.client = TestClient(create_app(auth_settings=AuthSettings(credentials={})))

    def test_everything_is_open(self):
        with self.client as client:
            self.assertEqual(client.get('/healthz').status_code, 200)
            # 404 = 路由本身不存在，说明请求没被鉴权拦住
            self.assertEqual(client.get('/api/anything').status_code, 404)

    def test_identity_says_auth_is_off(self):
        with self.client as client:
            self.assertEqual(client.get('/auth/me').json(),
                             {'user': None, 'auth_enabled': False})

    def test_login_is_refused_rather_than_pretending_to_work(self):
        """未启用时登录要明确拒绝——签发一个谁都能拿到的会话更糟。"""
        with self.client as client:
            response = client.post('/auth/login',
                                   json={'user': 'lijl', 'password': 'pw-one'})
            self.assertEqual(response.status_code, 400)
            self.assertNotIn('set-cookie', response.headers)


class CookieAttributes(unittest.TestCase):
    """线上反代把服务挂在 `/football/` 下并剥掉前缀，且走 https。
    这两件事应用自己都看不见，只能由配置告诉它。
    """

    def _login_cookie(self, **overrides):
        with make_client(**overrides) as client:
            return client.post('/auth/login',
                               json={'user': 'lijl', 'password': 'pw-one'}
                               ).headers['set-cookie']

    def test_the_path_is_configurable(self):
        """默认 `Path=/` 会把 Cookie 发给整个域，同域下的别的应用都收得到。"""
        self.assertIn('Path=/football/', self._login_cookie(cookie_path='/football/'))

    def test_secure_is_off_by_default_and_can_be_turned_on(self):
        """本地开发是 http——写死 `Secure` 浏览器就不发 Cookie 了。"""
        self.assertNotIn('Secure', self._login_cookie())
        self.assertIn('Secure', self._login_cookie(cookie_secure=True))

    def test_samesite_defaults_to_lax(self):
        self.assertIn('SameSite=lax', self._login_cookie().lower().replace('samesite', 'SameSite'))

    def test_settings_read_the_documented_environment_variables(self):
        settings = AuthSettings.from_env({
            'FOOTBALL_USERS': 'a:1',
            'FOOTBALL_SESSION_TTL': '3600',
            'FOOTBALL_COOKIE_PATH': '/football/',
            'FOOTBALL_COOKIE_SECURE': 'true',
        })
        self.assertTrue(settings.enabled)
        self.assertEqual(settings.session_ttl, 3600)
        self.assertEqual(settings.cookie_path, '/football/')
        self.assertTrue(settings.cookie_secure)

    def test_secure_only_accepts_explicit_affirmatives(self):
        for value, expected in (('1', True), ('true', True), ('YES', True),
                                ('0', False), ('false', False), ('', False)):
            with self.subTest(value=value):
                self.assertEqual(
                    AuthSettings.from_env({'FOOTBALL_COOKIE_SECURE': value}).cookie_secure,
                    expected)


if __name__ == '__main__':
    unittest.main()
