"""分项回测：胆码/杀码、形态、和值跨度区间、斜连信号各自的命中率。

主回测（`backtest.py`）回答的是「整套排序好不好」，这里回答的是「其中某一
项自己站不站得住」。分开测是有必要的——整注命中率是十几个特征合起来的
结果，某一项坏掉了往往只让它掉一两个百分点，淹没在噪声里。

**四个都只累加与聚合，怎么预测由调用方注入**，理由与主回测相同。

**每个比率的分母都是实际评估的期数**，不是请求的期数。旧实现一律拿请求值
当分母，历史不够长时命中率会被系统性低估，而且报出来的 `trials` 与实际
跑的期数对不上——两个数字都不假，只是不是一回事。
"""
from src.domain.numeric.lottery3d.space import DIGIT_SPACE

# 胆码算「中了」的两档：至少一个、至少两个。3D 一注三位，中两个胆码
# 基本就锁定了组选。
DAN_HIT_TIERS = (1, 2)
# 和值中心的容差档。三档一起报是为了看衰减有多快——只报一档看不出
# 「差一点」和「差很远」的比例。
SUM_TOLERANCES = (2, 3, 4)
SPAN_TOLERANCES = (1, 2)

POSITION_SLOPE = 'position_slope'
CROSS_PERIOD_SLOPE = 'cross_period_slope'


class _Trials:
    """记了多少期。四个分项共用——分母写错是这类统计最常见的毛病。"""

    def __init__(self):
        self._trials = 0

    @property
    def trials(self):
        return self._trials

    def _count(self):
        self._trials += 1

    def _rate(self, hit, total=None):
        total = self._trials if total is None else total
        return hit / total if total else 0.0


class DanKillBacktest(_Trials):
    """胆码命中与杀码失手。

    **杀码统计的是「失手」而不是「命中」**：杀码的作用是排除，它出现在开奖
    号里就是错了。写成命中率会让好坏方向反过来，而数字本身看不出来。
    """

    def __init__(self, tiers=DAN_HIT_TIERS):
        super().__init__()
        self.tiers = tiers
        self.dan_hits = {tier: 0 for tier in tiers}
        self.kill_fail = 0

    def observe(self, actual, dan, kill):
        self._count()
        digits = set(actual)
        hit_count = len(set(dan) & digits)
        for tier in self.tiers:
            if hit_count >= tier:
                self.dan_hits[tier] += 1
        if set(kill) & digits:
            self.kill_fail += 1

    def summarise(self):
        result = {'trials': self.trials}
        for tier in self.tiers:
            result[f'dan_hit{tier}_rate'] = self._rate(self.dan_hits[tier])
        result['kill_fail_rate'] = self._rate(self.kill_fail)
        return result


class FormBacktest(_Trials):
    """形态预测：整体 Top1 命中，外加组六/组三各自的**精确率**。

    精确率的分母是「预测成这个形态的期数」，不是总期数——组六本来就占七成，
    拿总期数当分母的话，一个永远猜组六的模型看起来也很准。
    """

    def __init__(self, tracked_forms):
        super().__init__()
        self.tracked_forms = tracked_forms
        self.hit = 0
        self.predicted = {form: 0 for form in tracked_forms}
        self.correct = {form: 0 for form in tracked_forms}

    def observe(self, actual_form, predicted_form):
        self._count()
        if predicted_form == actual_form:
            self.hit += 1
        if predicted_form in self.predicted:
            self.predicted[predicted_form] += 1
            if actual_form == predicted_form:
                self.correct[predicted_form] += 1

    def summarise(self):
        result = {'trials': self.trials, 'form_top1_rate': self._rate(self.hit)}
        for form in self.tracked_forms:
            result[f'{form}_precision'] = self._rate(
                self.correct[form], self.predicted[form])
        return result


class SumSpanBacktest(_Trials):
    """和值与跨度落在中心值多少以内。

    这不是命中率而是**离散度**：中心值预测得准，说明和值分布的中心被抓住了；
    抓住中心不代表能中奖，只代表推荐池不会整体偏到一边去。
    """

    def __init__(self, sum_tolerances=SUM_TOLERANCES, span_tolerances=SPAN_TOLERANCES):
        super().__init__()
        self.sum_tolerances = sum_tolerances
        self.span_tolerances = span_tolerances
        self.sum_hits = {tol: 0 for tol in sum_tolerances}
        self.span_hits = {tol: 0 for tol in span_tolerances}

    def observe(self, actual, sum_center, span_center):
        self._count()
        sum_error = abs(sum(actual) - sum_center)
        span_error = abs(max(actual) - min(actual) - span_center)
        for tol in self.sum_tolerances:
            if sum_error <= tol:
                self.sum_hits[tol] += 1
        for tol in self.span_tolerances:
            if span_error <= tol:
                self.span_hits[tol] += 1

    def summarise(self):
        result = {'trials': self.trials}
        for tol in self.sum_tolerances:
            result[f'sum_hit_{tol}_rate'] = self._rate(self.sum_hits[tol])
        for tol in self.span_tolerances:
            result[f'span_hit_{tol}_rate'] = self._rate(self.span_hits[tol])
        return result


class SlopeBacktest(_Trials):
    """斜连信号：预测的那个分位到底开没开出来。

    **同位与跨期分开统计**：两者强度不同，混在一起算命中率会把强的那类摊薄。

    分母是**信号条数**而不是期数——一期可能出好几条信号，也可能一条都没有。
    拿期数当分母，信号少的那段时间命中率会莫名其妙地低。
    """

    def __init__(self, baseline=None):
        super().__init__()
        # 单个分位随机命中的概率：十个数字里猜中一个
        self.baseline = 1 / DIGIT_SPACE.size if baseline is None else baseline
        self.hits = {POSITION_SLOPE: 0, CROSS_PERIOD_SLOPE: 0}
        self.totals = {POSITION_SLOPE: 0, CROSS_PERIOD_SLOPE: 0}

    def observe(self, actual, signals):
        self._count()
        for signal in signals:
            kind = signal.get('type')
            if kind not in self.totals:
                continue
            self.totals[kind] += 1
            # 没带 position 的信号预测不到具体分位，直接记未命中——
            # **不能跳过**：跳过等于把它从分母里也拿掉，命中率会虚高
            position = signal.get('position')
            if position is not None and actual[position] == signal.get('predict_digit'):
                self.hits[kind] += 1

    def summarise(self):
        return {
            'trials': self.trials,
            'position_slope_hit': self.hits[POSITION_SLOPE],
            'position_slope_total': self.totals[POSITION_SLOPE],
            'position_slope_rate': self._round(POSITION_SLOPE),
            'cross_slope_hit': self.hits[CROSS_PERIOD_SLOPE],
            'cross_slope_total': self.totals[CROSS_PERIOD_SLOPE],
            'cross_slope_rate': self._round(CROSS_PERIOD_SLOPE),
            'baseline_single_pos': self.baseline,
        }

    def _round(self, kind):
        total = self.totals[kind]
        return round(self.hits[kind] / total, 4) if total else 0.0
