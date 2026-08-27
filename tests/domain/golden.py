"""黄金文件：迁移期差分测试的接班人。

迁移期间，这些用例的期望值来自旧实现——对同一组输入同时跑新旧两份、断言
逐字相等。旧实现删掉之后那条路断了，但它当时覆盖到的输入组合不该跟着消失：
变异验证里有十几处逃逸正是被这些组合捕获的。

所以把当时验证通过的输出固化下来。**黄金值是从新实现生成的，而新实现已经
与旧实现逐字比对通过**，所以它记录的仍是迁移前的行为。

期望值变了怎么办：先确认这是**有意**的改动，再用
`scripts/regen_golden.py` 重新生成，并在提交信息里说明为什么。
黄金文件对「无意的改动」很敏感，这正是它的用处。
"""
import gzip
import json
import pathlib

GOLDEN_DIR = pathlib.Path(__file__).resolve().parents[1] / 'fixtures' / 'golden'


def load(name):
    """读黄金文件。文件不存在时返回空字典。

    容忍缺失只为一件事：生成脚本要 import 测试模块才能拿到输入语料，
    而测试模块在 import 时就会 load——首次生成时文件还不存在，
    不容忍就死锁了。缺失时测试会因为取不到键而失败，不会假装通过。
    """
    path = GOLDEN_DIR / f'{name}.json.gz'
    if not path.exists():
        return {}
    with gzip.open(path, 'rt', encoding='utf-8') as fh:
        return json.load(fh)


def as_json(value):
    """把结果规范化成 JSON 可比的形状。

    元组在 JSON 里会变成数组，直接比会因类型不同而失败——必须两边都过一遍
    同样的序列化，比的才是「内容」而不是「Python 类型」。
    """
    return json.loads(json.dumps(value, ensure_ascii=False, sort_keys=True))


def as_comparable(value, ndigits=10):
    """把结果规范化成**能与黄金文件逐条比对**的形状。

    比 `as_json` 多做两件事，都是 3D 的特征层逼出来的：

    - **元组键**：二阶马尔可夫的键是 `(前两期, 前一期)`。JSON 不允许非字符串
      的键，`json.dumps` 会直接抛错，所以统一转成字符串。
    - **浮点取整**：指数加权、频率这类量在不同求和顺序下末位可能差一个 ulp。
      不取整的话，一次纯粹的结构调整也会让黄金比对整片变红，那就分不清
      「结果变了」和「加法顺序变了」。取到第 10 位远严于任何有意义的变化。

    生成黄金文件与比对黄金文件**必须走同一个函数**——两边各写一套规范化，
    比的就不再是同一个东西了。
    """
    if isinstance(value, dict):
        return {str(k): as_comparable(v, ndigits)
                for k, v in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple)):
        return [as_comparable(v, ndigits) for v in value]
    if isinstance(value, float):
        return round(value, ndigits)
    return value
