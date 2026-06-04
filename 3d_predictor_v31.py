#!/usr/bin/env python3
"""福彩3D预测 CLI 入口（业务逻辑在 3d/ 目录）"""
import importlib.util
import sys
from pathlib import Path


def _load():
    pkg = Path(__file__).parent / '3d'
    spec = importlib.util.spec_from_file_location(
        'lottery_3d', pkg / '__init__.py',
        submodule_search_locations=[str(pkg)],
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


if __name__ == '__main__':
    _load().main()
