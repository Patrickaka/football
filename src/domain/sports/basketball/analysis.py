"""三个玩法的赛前分析：胜负 / 让分胜负 / 大小分。

每个玩法的结论都由三份证据合成，权重高低就是对各自信息量的判断：

- **市场价格**是最强的基线。收盘赔率已经吸收了公开信息，模型的角色是
  在它之上做小幅修正，而不是另起炉灶。
- **Elo** 只有在双方都攒够历史后才逐步参与（trust 随较少一方的场次线性
  升到 20 场封顶）。冷启动评分若按满权重进来，会稀释掉信息量大得多的市场。
- **赔率走势**只增强、不反转；唯一能改变方向的是 `apply_market_inference`
  里那条「强信号翻弱模型」的通路，且要求样本充分、未过期、水位与盘口不冲突。

依赖注入而非取全局单例：Elo 与校准器都要读库，写死单例就没法在测试里
替换，也就没法证明迁移前后行为一致。两者都可以为 None——它们是增强项，
缺了要降级成纯市场价格，而不是让整场比赛分析失败。
"""
import logging
import math

from src.domain.sports.basketball import movement as mv

log = logging.getLogger('domain.basketball.analysis')

LEAGUE_PROFILES = {
    'NBA': {'avg_total': 220.0, 'home_win_rate': 0.57},
    'CBA': {'avg_total': 190.0, 'home_win_rate': 0.55},
    'NCAAB': {'avg_total': 150.0, 'home_win_rate': 0.58},
    '欧洲篮球': {'avg_total': 165.0, 'home_win_rate': 0.54},
    'WNBA': {'avg_total': 160.0, 'home_win_rate': 0.56},
    '美职女篮': {'avg_total': 170.0, 'home_win_rate': 0.55},
}

# 双方各自打满这么多场，Elo 才拿到完整权重
ELO_FULL_TRUST_GAMES = 20.0
ELO_WEIGHT = {'spf': 0.30, 'rqspf': 0.25, 'dx': 0.20}
# 各玩法成为「官方推荐」的最低胜率——让分与大小分本就更接近掷硬币，
# 门槛压得比胜负低一点，否则常年无推荐。
OFFICIAL_PICK_THRESHOLDS = {'spf': 0.56, 'rqspf': 0.55, 'dx': 0.54}
PICK_NAMES = {'spf': ('主胜', '客胜'), 'rqspf': ('让胜', '让负'),
              'dx': ('大分', '小分')}
# 校准后的取值区间。下界略高于 0.5：校准可以削弱信心，但不允许它把
# 选中的一侧悄悄翻到另一侧——反转必须来自新的市场或模型证据。
CALIBRATED_MIN = 0.5001
CALIBRATED_MAX = 0.95


def odds_to_prob(odds):
    if odds is None or odds <= 0:
        return 0.0
    return 1.0 / odds


def calc_implied_prob(first_odds, second_odds):
    """去掉庄家抽水后的两路隐含概率。"""
    p_first = odds_to_prob(first_odds)
    p_second = odds_to_prob(second_odds)
    total = p_first + p_second + 1e-9
    return p_first / total, p_second / total


def confidence_from_probs(p_first, p_second):
    gap = abs(float(p_first) - float(p_second))
    return 'high' if gap > 0.15 else ('medium' if gap > 0.08 else 'low')


def official_pick_status(bet_type, pick_prob, confidence):
    """把「模型的看法」与「计入准确率的官方推荐」分开。

    两者混在一起时，低置信度的场次会拖累准确率统计，让人无从判断模型
    在它真正有把握时表现如何。
    """
    if confidence == 'low':
        return {'playable': False, 'official': False,
                'skip_reason': 'low_confidence'}
    if float(pick_prob or 0) < OFFICIAL_PICK_THRESHOLDS.get(bet_type, 0.56):
        return {'playable': False, 'official': False,
                'skip_reason': 'probability_below_threshold'}
    return {'playable': True, 'official': True, 'skip_reason': None}


def movement_accuracy_gate(bet_type, status, movement, confirmation, inference=None):
    """让分与大小分的官方推荐额外要求走势佐证——准确率优先于出票量。

    胜负玩法不设这道闸：它的市场最厚、价格最可信，缺走势也能出票。
    """
    if bet_type not in ('rqspf', 'dx'):
        return status

    reason = _gate_rejection(movement, confirmation, inference)
    if reason is None:
        return status
    return {**status, 'playable': False, 'official': False, 'skip_reason': reason}


def _gate_rejection(movement, confirmation, inference):
    if not movement or not movement.get('available'):
        return 'movement_unavailable'
    if movement.get('stale'):
        return 'movement_stale'
    if int(movement.get('samples', 0) or 0) < 2:
        return 'movement_samples_insufficient'
    if inference is not None and not inference.get('actionable'):
        return inference.get('reason') or 'movement_signal_weak'
    if not confirmation.get('confirmed'):
        return 'movement_conflicts_with_model'
    return None


class BasketballAnalyzer:
    """把一场比赛的赔率、Elo 与赔率走势合成三个玩法的分析结论。

    elo / calibrator 均可为 None：两者都是增强项，缺失时降级为纯市场价格。
    """

    def __init__(self, elo=None, calibrator=None):
        self._elo = elo
        self._calibrator = calibrator

    # ---------- 胜负 ----------

    def analyze_spf(self, match, movement=None):
        home_odds = match.get('spf_home')
        away_odds = match.get('spf_away')
        if home_odds is None or away_odds is None:
            return _unavailable('missing_odds', 'home_prob', 'away_prob')

        p_home, p_away = calc_implied_prob(home_odds, away_odds)
        league = match.get('league', '')
        # 联赛主场先验刻意只占一成：它是个长期平均值，对单场的信息量远低于价格。
        home_bias = _profile(league).get('home_win_rate', 0.55)
        p_home = p_home * 0.9 + home_bias * 0.1
        market_home_prob = p_home

        elo_win, _, _, elo_trust = self._elo_predictions(match)
        if elo_win:
            p_home = _blend(p_home, elo_win['home_prob'], ELO_WEIGHT['spf'] * elo_trust)
        p_away = 1.0 - p_home

        if movement and movement.get('available') and movement.get('side') != 'flat':
            p_home, p_away = mv.apply_movement(p_home, p_away, movement)

        p_home, p_away, meta = self._finalize('spf', p_home, p_away, league, movement)
        return {
            'available': True,
            'home_prob': round(p_home, 4),
            'away_prob': round(p_away, 4),
            'home_odds': home_odds,
            'away_odds': away_odds,
            'market_home_prob': round(market_home_prob, 4),
            'elo_home_prob': elo_win.get('home_prob') if elo_win else None,
            'elo_trust': round(elo_trust, 3),
            **meta,
        }

    # ---------- 让分胜负 ----------

    def analyze_rqspf(self, match, movement=None):
        handicap = match.get('handicap')
        home_odds = match.get('rqspf_home')
        away_odds = match.get('rqspf_away')
        if home_odds is None or away_odds is None:
            return _unavailable('missing_rqspf_odds', 'home_prob', 'away_prob',
                                handicap=handicap)

        p_home, p_away = calc_implied_prob(home_odds, away_odds)
        market_home_prob = p_home

        _, elo_margin, _, elo_trust = self._elo_predictions(match)
        elo_home_prob = None
        if elo_margin:
            # 让分后的净胜分期望过 logistic；8.5 分是篮球单场分差的经验尺度。
            adjusted = float(elo_margin.get('expected_margin', 0)) + _as_float(handicap)
            elo_home_prob = _logistic(adjusted, 8.5)
            p_home = _blend(p_home, elo_home_prob, ELO_WEIGHT['rqspf'] * elo_trust)
            p_away = 1.0 - p_home

        p_home, p_away, inference = mv.apply_market_inference(
            p_home, p_away, movement, 'rqspf')
        p_home, p_away, meta = self._finalize(
            'rqspf', p_home, p_away, match.get('league', ''), movement, inference)
        inference['final_recommendation'] = meta['recommendation']
        return {
            'available': True,
            'handicap': handicap,
            'home_prob': round(p_home, 4),
            'away_prob': round(p_away, 4),
            'home_odds': home_odds,
            'away_odds': away_odds,
            'market_home_prob': round(market_home_prob, 4),
            'elo_margin': elo_margin.get('expected_margin') if elo_margin else None,
            'elo_home_prob': round(elo_home_prob, 4) if elo_home_prob is not None else None,
            'elo_trust': round(elo_trust, 3),
            'water_inference': inference,
            'movement_led': bool(inference.get('reversed_model')),
            **meta,
        }

    # ---------- 大小分 ----------

    def analyze_daxiao(self, match, movement=None):
        total_line = match.get('total_line')
        over_odds = match.get('dx_over')
        under_odds = match.get('dx_under')
        if over_odds is None or under_odds is None:
            return _unavailable('missing_dx_odds', 'over_prob', 'under_prob',
                                total_line=total_line)

        league = match.get('league', '')
        p_over, p_under = self._total_line_prior(
            *calc_implied_prob(over_odds, under_odds), league, total_line)
        market_over_prob = p_over

        _, _, elo_total, elo_trust = self._elo_predictions(match)
        expected_total = elo_total.get('expected_total') if elo_total else None
        elo_over_prob = None
        if expected_total is not None and total_line is not None:
            # 12 分是总分预测的经验误差尺度，比分差尺度大是因为总分方差更大。
            elo_over_prob = _logistic(float(expected_total) - float(total_line), 12.0)
            p_over = _blend(p_over, elo_over_prob, ELO_WEIGHT['dx'] * elo_trust)
            p_under = 1.0 - p_over

        p_over, p_under, inference = mv.apply_market_inference(
            p_over, p_under, movement, 'dx')
        p_over, p_under, meta = self._finalize(
            'dx', p_over, p_under, league, movement, inference)
        inference['final_recommendation'] = meta['recommendation']
        return {
            'available': True,
            'total_line': total_line,
            'over_prob': round(p_over, 4),
            'under_prob': round(p_under, 4),
            'over_odds': over_odds,
            'under_odds': under_odds,
            'market_over_prob': round(market_over_prob, 4),
            'elo_total': expected_total,
            'elo_over_prob': round(elo_over_prob, 4) if elo_over_prob is not None else None,
            'elo_trust': round(elo_trust, 3),
            'water_inference': inference,
            'movement_led': bool(inference.get('reversed_model')),
            **meta,
        }

    @staticmethod
    def _total_line_prior(p_over, p_under, league, total_line):
        """盘口明显偏离联赛均值时向回归方向让一点。

        联赛均分是这三个玩法里唯一有稳定长期锚的量，偏离超过 5 分才动，
        避免把正常波动当信号。
        """
        avg_total = _profile(league).get('avg_total', 200.0)
        if total_line is None or not avg_total:
            return p_over, p_under

        line_diff = total_line - avg_total
        if line_diff > 5:
            p_over, p_under = p_over - 0.05, p_under + 0.05
        elif line_diff < -5:
            p_over, p_under = p_over + 0.05, p_under - 0.05
        total = p_over + p_under + 1e-9
        return p_over / total, p_under / total

    # ---------- 三个玩法共用的收尾 ----------

    def _finalize(self, bet_type, p_first, p_second, league, movement, inference=None):
        """校准 → 定方向 → 定档 → 走势确认 → 准入闸。

        三个玩法这一段原本各抄了一份逐字相同的代码。它们的差异只有玩法名、
        两路的中文名和是否带反推结论，抽出来后阈值调整只会有一处。
        """
        confidence = confidence_from_probs(p_first, p_second)
        (p_first, p_second), raw_pick_prob = self._calibrate_pick(
            bet_type, p_first, p_second, league, confidence)

        first_name, second_name = PICK_NAMES[bet_type]
        recommendation = first_name if p_first > p_second else second_name
        pick_prob = max(p_first, p_second)
        status = official_pick_status(bet_type, pick_prob, confidence)

        confirmation = {'confirmed': False, 'reason': 'movement_unavailable',
                        'boost': 0.0}
        line_movement = None
        sharp_confirmed = False
        if movement and movement.get('available'):
            confirmation = mv.sharp_confirmation(movement, recommendation)
            line_movement = {**movement, 'recommendation': recommendation,
                             'confirmed': confirmation['confirmed'],
                             'reason': confirmation['reason']}
            if confirmation['confirmed'] and confirmation['boost'] > 0:
                sharp_confirmed = True
                confidence = _bump_confidence(confidence, confirmation['boost'])

        status = movement_accuracy_gate(bet_type, status, movement, confirmation,
                                        inference)
        return p_first, p_second, {
            'recommendation': recommendation,
            'confidence': confidence,
            'pick_prob': round(pick_prob, 4),
            'raw_pick_prob': raw_pick_prob,
            'line_movement': line_movement,
            'sharp_confirmed': sharp_confirmed,
            **status,
        }

    def _calibrate_pick(self, bet_type, p_first, p_second, league, confidence):
        """只校准被选中的那一侧，另一侧取补数以保持两路归一。"""
        pick_first = p_first >= p_second
        raw_pick = p_first if pick_first else p_second

        calibrated = raw_pick
        if self._calibrator is not None:
            try:
                calibrated = self._calibrator.calibrate(
                    bet_type, raw_pick, league, confidence)
            except Exception as exc:
                log.warning('篮球历史校准不可用: %s', exc)
                calibrated = raw_pick
        calibrated = max(CALIBRATED_MIN, min(CALIBRATED_MAX, float(calibrated)))

        pair = ((calibrated, 1.0 - calibrated) if pick_first
                else (1.0 - calibrated, calibrated))
        return pair, round(raw_pick, 4)

    def _elo_predictions(self, match):
        """返回 (胜负, 分差, 总分, 信任度)。任何不可用都退成零信任。

        零信任而不是抛错：Elo 是增强项，它不可用时结论应回落到市场价格，
        而不是让整场比赛没有分析结果。
        """
        if self._elo is None:
            return {}, {}, {}, 0.0
        try:
            home, away = match.get('home', ''), match.get('away', '')
            league = match.get('league', '')
            win = self._elo.predict_win_prob(home, away, league)
            margin = self._elo.predict_margin(home, away, league)
            total = self._elo.predict_total_score(home, away, league)
            games = min(int(win.get('home_games', 0)), int(win.get('away_games', 0)))
            return win, margin, total, min(1.0, games / ELO_FULL_TRUST_GAMES)
        except Exception as exc:
            log.warning('篮球 ELO 预测不可用: %s', exc)
            return {}, {}, {}, 0.0


def _unavailable(reason, first_key, second_key, **extra):
    """缺赔率时的统一返回。两路各给 0.5 而不是 None——下游多处直接参与比较。"""
    return {
        'available': False,
        'reason': reason,
        first_key: 0.5,
        second_key: 0.5,
        'recommendation': None,
        'confidence': 'low',
        **extra,
    }


def _profile(league):
    return LEAGUE_PROFILES.get(league, {})


def _blend(base, other, weight):
    return base * (1.0 - weight) + other * weight


def _logistic(value, scale):
    return 1.0 / (1.0 + math.exp(-value / scale))


def _as_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _bump_confidence(confidence, boost):
    """聪明钱确认时上调一档。medium→high 要求更强的确认（steam 级）。"""
    if confidence == 'low':
        return 'medium'
    if confidence == 'medium' and boost >= 1.0:
        return 'high'
    return confidence
