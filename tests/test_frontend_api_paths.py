# -*- coding: utf-8 -*-
"""网页里的接口地址必须是后端真有的路由。

**BFF 上线当天就栽在这里**：那一条写成了相对地址 `bff/football/home`，
在 `/football/` 下解析成 `/football/bff/football/home`——少了 `api/`
这一段，反代上没有对应 location，用户看到「接口不存在」。

`web/index.html` 里其余几十条全是绝对路径 `/api/...`，只有新加的那条
是相对的。**一致性本身就是防线**：混着写，出错的那条不会有任何征兆。

登录页 `web/login.html` 是另一回事——它自己就在 `/football/login`，
用相对地址 `auth/login` 正好解析到 `/football/auth/login`。那是对的，
所以这条守卫只管 `index.html`。
"""
import re
import unittest
from pathlib import Path

from src.api.app import create_app
from src.api.auth import AuthSettings

INDEX = Path('web/index.html')


def called_paths():
    """网页里 fetch 出去的接口地址（去掉查询串与模板插值）。"""
    html = INDEX.read_text(encoding='utf-8')
    found = set()
    for match in re.finditer(r"""fetchJson[A-Za-z]*\(\s*['"`]([^'"`]+)""", html):
        raw = match.group(1)
        path = raw.split('?')[0].split('#')[0]
        path = re.sub(r'\$\{[^}]*\}', '{}', path)
        if path:
            found.add(path)
    return found


def served_paths():
    app = create_app(auth_settings=AuthSettings(credentials={}))
    return {re.sub(r'\{[^}]*\}', '{}', p) for p in app.openapi()['paths']}


class FrontendPaths(unittest.TestCase):

    def test_every_call_is_an_absolute_api_path(self):
        """**混着写就是隐患**：相对地址在子路径部署下会解析到别处，
        而且只有那一条会错，看不出征兆。
        """
        for path in sorted(called_paths()):
            with self.subTest(path=path):
                self.assertTrue(path.startswith('/'),
                                f'{path} 是相对地址，子路径部署下会解析错')

    def test_every_call_hits_a_real_route(self):
        """前端调的每条都得在后端存在——pailie5 就是这么坏了好几个月的。

        通配路由（`/reports/{name}`）按前缀匹配：具体文件在不在是运行时的事，
        路由存在就够。`loadProfessionalBacktestFallback` 取那份静态回测报告
        正属此类——**取不到是预期内的**，它自己会退回内置基线，
        函数名就叫 Fallback。
        """
        served = served_paths()
        prefixes = tuple(p.split('{')[0] for p in served if '{}' in p)
        missing = sorted(p for p in called_paths()
                         if p not in served and not p.startswith(prefixes))
        self.assertEqual(missing, [],
                         f'这些地址后端没有对应路由：{missing}')

    def test_the_check_actually_sees_something(self):
        """守卫的守卫：提取器要真的提到东西，否则上面两条是空转。"""
        paths = called_paths()
        self.assertGreater(len(paths), 20)
        self.assertIn('/api/bff/football/home', paths)


if __name__ == '__main__':
    unittest.main()
