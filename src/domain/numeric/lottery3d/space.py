"""3D 的号码空间：三位，每位 0~9。

单独一个模块的理由与 kl8 的 `space.py` 相同——位数、每位的取值范围、
和值上限这些量在特征、评分、选号里到处要用，有两个定义就迟早对不上。
"""
from src.domain.numeric.statistics import NumberSpace

# 每一位的取值空间。注意与 kl8 的差别：这里描述的是**一位**，
# 不是一注——一注是三个这样的位。
DIGIT_SPACE = NumberSpace(low=0, high=9)

# 位数。百位、十位、个位。
POSITIONS = 3

# 就是这三个字，不带「位」。斜连的关注码字典**以它们为键**，
# 写成「百位」会让下游查不到任何关注码——而查不到不报错，只是加分恒为 0。
POSITION_NAMES = ('百', '十', '个')

# 和值的取值范围：三位全 0 到三位全 9。
SUM_MIN = DIGIT_SPACE.low * POSITIONS
SUM_MAX = DIGIT_SPACE.high * POSITIONS
