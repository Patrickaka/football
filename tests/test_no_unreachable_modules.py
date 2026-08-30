# -*- coding: utf-8 -*-
"""从入口出发的可达性检查：不该有谁都到不了的模块。

死代码不会报错，只会越攒越多，并且让人以为某个能力已经具备
（`foundation/ml` 就是这样：建了 `TrainValidSplit` 这类抽象，
看着像底座提供了 ML 能力，实际没有任何业务走它）。

**必须把动态导入算进去。** 第一版只看 `ast.Import`，结论是 38 个模块
不可达——其中 33 个是 `importlib.import_module('src.lottery3d')` 这类
拉起来的活代码（3D tab 的整个后端）。**照那个结论删就是删活的。**

已知的、有意保留的不可达项列在 `ENTRY_SCRIPTS` 里：它们是手动执行的
离线工具，入口是命令行而不是 import。
"""
import ast
import unittest
from collections import deque
from pathlib import Path

SRC = Path('src')

#: 从这些地方出发算可达。scripts/ 里的一次性工具也是入口。
ROOTS = (
    'src.api.app', 'src.api.startup', 'src.api.runtime.jobs',
    'src.football.cli', 'src.common.maintenance',
)

#: 命令行入口，靠 `python3 -m ...` 手动跑，没有 import 它的地方。
#: **不是死代码**——删了就没有重训模型的入口了。
ENTRY_SCRIPTS = {
    'src.football.ml_trainer',
}


def _modules():
    found = {}
    for path in SRC.rglob('*.py'):
        if '__pycache__' in str(path):
            continue
        module = str(path.with_suffix('')).replace('/', '.')
        if module.endswith('.__init__'):
            module = module[:-len('.__init__')]
        found[module] = path
    return found


def _imports_of(path):
    """静态 import **加上**字符串形式的动态导入。"""
    out = set()
    try:
        tree = ast.parse(path.read_text(encoding='utf-8'))
    except SyntaxError:
        return out
    package = str(path.parent).replace('/', '.')
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out.update(a.name for a in node.names)
        elif isinstance(node, ast.Call):
            name = getattr(node.func, 'attr', None) or getattr(node.func, 'id', None)
            if name in ('import_module', '__import__') and node.args:
                arg = node.args[0]
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    out.add(arg.value)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) \
                and node.value.startswith('src.') and '.' in node.value[4:]:
            out.add(node.value)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = package
                for _ in range(node.level - 1):
                    base = base.rsplit('.', 1)[0]
                module = f'{base}.{node.module}' if node.module else base
            else:
                module = node.module or ''
            out.add(module)
            out.update(f'{module}.{a.name}' for a in node.names)
    return out


def reachable_modules():
    modules = _modules()
    roots = [r for r in ROOTS if r in modules]
    roots += [m for m in modules if m.startswith(('src.api.routers', 'src.api.services'))]
    # scripts/ 下的工具也是入口，它们引到的东西不算死
    for path in Path('scripts').rglob('*.py'):
        if '__pycache__' in str(path):
            continue
        for imported in _imports_of(path):
            for candidate in (imported, imported.rsplit('.', 1)[0]):
                if candidate in modules:
                    roots.append(candidate)

    seen, queue = set(), deque(roots)
    while queue:
        module = queue.popleft()
        if module in seen:
            continue
        seen.add(module)
        for imported in _imports_of(modules[module]):
            for candidate in (imported, imported.rsplit('.', 1)[0]):
                if candidate in modules and candidate not in seen:
                    queue.append(candidate)
    return modules, seen


class Reachability(unittest.TestCase):

    def test_no_module_is_orphaned(self):
        modules, seen = reachable_modules()
        orphans = sorted(set(modules) - seen - ENTRY_SCRIPTS)
        self.assertEqual(orphans, [],
                         f'这些模块从任何入口都到不了：{orphans}\n'
                         f'要么接上，要么删掉，要么加进 ENTRY_SCRIPTS 并说明理由')

    def test_dynamic_imports_are_counted(self):
        """**守卫的守卫**：漏掉动态导入，上一条会把活代码报成孤儿。

        `src.lottery3d` 是 3D tab 的整个后端，只被
        `importlib.import_module('src.lottery3d')` 拉起来。
        """
        _, seen = reachable_modules()
        self.assertIn('src.lottery3d', seen,
                      '动态导入没被算进去——这个分析会把活代码判成死的')
        self.assertIn('src.lottery3d.ml', seen)

    def test_the_entry_script_list_is_not_a_dumping_ground(self):
        """豁免清单是给命令行工具的，**它们必须真的有 `__main__` 入口**。

        否则这个清单就成了"报错了就往里加一条"的垃圾桶。
        """
        modules = _modules()
        for module in ENTRY_SCRIPTS:
            with self.subTest(module=module):
                self.assertIn(module, modules, f'{module} 已经不存在了，清单该更新')
                source = modules[module].read_text(encoding='utf-8')
                self.assertIn("if __name__ == '__main__':", source)


if __name__ == '__main__':
    unittest.main()
