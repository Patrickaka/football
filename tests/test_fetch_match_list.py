"""
回归测试：比赛列表解析
====================
夹具 index_jczq.html 是 500.com 真实页面快照，其布局特征为
shuju-<id> 链接出现在 title="...VS..." 锚点之前。

历史 bug：原正则假设 shuju 在 title 之后，导致
  1) 每场比赛配到下一场的 match_id（错位）
  2) 最后一场被整场丢弃
本测试锁定该场景。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import src.football as football

FIXTURE = Path(__file__).parent / 'fixtures' / 'index_jczq.html'

# 夹具中的真实比赛与正确的 id、竞彩编号配对
EXPECTED = [
    ('1411019', '丹麦', '刚果(金)', '周三201'),
    ('1393310', '荷兰', '阿尔及利亚', '周三202'),
    ('1404861', '波兰', '尼日利亚', '周三203'),
    ('1411009', '卢森堡', '意大利', '周三204'),
]


def _load(monkeypatched_html):
    football.fetch = lambda *a, **k: monkeypatched_html
    return football.fetch_match_list()


def test_all_matches_parsed_with_correct_ids():
    html = FIXTURE.read_text(encoding='utf-8')
    matches = _load(html)

    assert len(matches) == len(EXPECTED), \
        f"应解析 {len(EXPECTED)} 场，实际 {len(matches)} 场"

    got = {(m['match_id'], m['home'], m['away'], m.get('num')) for m in matches}
    for mid, home, away, num in EXPECTED:
        assert (mid, home, away, num) in got, \
            f"缺失或错配: id={mid} {home} vs {away} {num}；实得 {sorted(got)}"


if __name__ == '__main__':
    test_all_matches_parsed_with_correct_ids()
    print("✓ 测试通过：4 场比赛全部解析且 id、竞彩编号配对正确")
