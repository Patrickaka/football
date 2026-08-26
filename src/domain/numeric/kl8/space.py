"""kl8 的号码空间：80 选 20。

单独一个模块，是因为评分、形态、候选池、多注覆盖四处都要用它，而「一期开
几个号」这种事有两个定义就迟早会对不上——`_adaptive_repeat_target` 拿它
换算重号比例，覆盖模拟拿它抽样，两处写死成不同的数不会报错，只会让推荐
悄悄偏掉。
"""
from src.domain.numeric.statistics import NumberSpace

SPACE = NumberSpace(low=1, high=80)

# 每期开出的号码个数。重号比例、覆盖率模拟都以它为分母。
DRAW_COUNT = 20
