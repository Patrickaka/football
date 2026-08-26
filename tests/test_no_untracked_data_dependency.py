"""测试不得依赖未跟踪的本地数据。

**这条守卫来自一次真实事故**：统计层的测试直接读 `data/kl8_history.json`，
本地跑得好好的，CI 上直接 FileNotFoundError——那个文件在 `.gitignore` 里。

同一类问题此前还出现过两次形态：SQLite 与 MySQL 的方言差异、字节码缓存
让变异验证结论失效。共同点都是「本地环境与目标环境不一样，而测试只在本地
验证过」。这条守卫只能挡住其中最容易犯的那一种，但它至少是自动的。

需要真实数据时把样本提交进 `tests/fixtures/`——线上抓来的真实页面与真实
记录本来就该入库，它们是判据 4「迁移前必读真实数据」的载体。
"""
import ast
import pathlib
import unittest

REPO = pathlib.Path(__file__).resolve().parents[1]
TESTS = REPO / 'tests'


def _gitignored_prefixes():
    """从 .gitignore 里取出被忽略的数据目录前缀。"""
    prefixes = []
    for line in (REPO / '.gitignore').read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        if line.startswith('data/'):
            prefixes.append(line)
    return prefixes


def _contains_tests(tree):
    """这个文件里到底有没有测试。

    `tests/` 下混着几个零测试的分析脚本（`test_kl8_backtest.py` 等，
    pytest 一个用例都收集不到）。守卫的对象是**测试**——脚本在本地读真实
    数据是它的本分，不该被这条规则管。
    """
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name.startswith('test_'):
                return True
        elif isinstance(node, ast.ClassDef):
            if node.name.startswith('Test') or node.name.endswith('Tests'):
                return True
    return False


def _data_path_offences(tree):
    """找出「把路径指向仓库 data/ 目录」的写法。

    只认三种真实形态，不认光秃秃的 `'data'`——那在测试里绝大多数时候是
    字典键（`raw.get('results', raw.get('data', []))`）。守卫宽一分，
    误报就多一片，最后只会被加白名单绕过去。
    """
    # 文档里提到 `data/kl8_history.json` 是在说明数据来自哪儿，不是在读它。
    # 判据 4 要求写清真实数据的出处，守卫不该跟这条打架。
    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                             ast.AsyncFunctionDef)):
            body = getattr(node, 'body', None) or []
            if body and isinstance(body[0], ast.Expr) and isinstance(
                    body[0].value, ast.Constant) and isinstance(body[0].value.value, str):
                docstrings.add(id(body[0].value))

    offences = []
    for node in ast.walk(tree):
        if id(node) in docstrings:
            continue
        # 形态一：字面量里直接带 data/
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if 'data/' in node.value:
                offences.append((node.lineno, node.value))
        # 形态二：pathlib 的 `... / 'data' / ...`
        elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
            for side in (node.left, node.right):
                if isinstance(side, ast.Constant) and side.value == 'data':
                    offences.append((node.lineno, "路径拼接 / 'data'"))
        # 形态三：data_path(...)，它解析到的正是那个目录
        elif isinstance(node, ast.Call):
            name = getattr(node.func, 'id', None) or getattr(node.func, 'attr', None)
            if name == 'data_path':
                offences.append((node.lineno, 'data_path(...)'))
    return offences


class NoUntrackedDataTests(unittest.TestCase):
    def test_no_test_file_reads_the_data_directory(self):
        """测试文件不该把路径指向仓库的 data/ 目录。

        查源码而不是运行时行为：真读到了才失败的话，在恰好有那个文件的机器上
        永远发现不了——正是这次事故的成因。
        """
        offenders = []
        for path in sorted(TESTS.rglob('test_*.py')):
            if path.name == pathlib.Path(__file__).name:
                continue
            tree = ast.parse(path.read_text(encoding='utf-8'))
            if not _contains_tests(tree):
                continue
            for lineno, what in _data_path_offences(tree):
                offenders.append(f'{path.relative_to(REPO)}:{lineno} {what}')
        self.assertEqual(offenders, [],
                         '测试把路径指向了 data/；把样本提交到 tests/fixtures/ 下')

    def test_the_guard_catches_the_real_mistake(self):
        """守卫本身也要被验证——它认不出那次事故的写法就是个摆设。"""
        source = ("import pathlib\n"
                  "DATA = pathlib.Path(__file__).parents[3] / 'data' / 'kl8.json'\n")
        self.assertTrue(_data_path_offences(ast.parse(source)))

    def test_the_guard_only_applies_to_files_with_tests(self):
        """`tests/` 下混着几个零测试的分析脚本，它们读真实数据是本分。"""
        script = ast.parse("def load_data():\n    return data_path('x.json')\n")
        self.assertFalse(_contains_tests(script))
        with_test = ast.parse("def test_x():\n    pass\n")
        self.assertTrue(_contains_tests(with_test))

    def test_the_guard_ignores_docstrings(self):
        """文档里提到数据出处是判据 4 的要求，守卫不该跟它打架。"""
        source = '"""结构依据实读线上 `data/kl8_history.json`。"""\n'
        self.assertEqual(_data_path_offences(ast.parse(source)), [])

    def test_the_guard_ignores_dict_keys_named_data(self):
        """`raw.get('data', [])` 是字典键，不是路径。守卫宽一分误报就多一片，
        最后只会被加白名单绕过去。"""
        source = "value = raw.get('results', raw.get('data', []))\n"
        self.assertEqual(_data_path_offences(ast.parse(source)), [])

    def test_fixtures_directory_is_tracked(self):
        """夹具目录必须真的在版本控制里，否则这条守卫等于把问题换个地方藏。"""
        import subprocess

        result = subprocess.run(
            ['git', 'ls-files', 'tests/fixtures'], cwd=REPO,
            capture_output=True, text=True, check=True)
        tracked = [line for line in result.stdout.splitlines() if line.strip()]
        self.assertTrue(tracked, 'tests/fixtures 下没有任何被跟踪的文件')

    def test_data_directory_is_gitignored(self):
        """这条守卫的前提：data/ 确实是被忽略的。前提变了要重新考虑。"""
        self.assertTrue(_gitignored_prefixes(),
                        '.gitignore 里已经没有 data/ 规则，本守卫的前提不再成立')


if __name__ == '__main__':
    unittest.main()
