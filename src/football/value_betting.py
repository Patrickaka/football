#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
赔率价值分析模块
=================

功能：
1. 计算比分的期望值和价值
2. 识别存在价值的比分
3. 根据价值调整推荐权重

职业博彩模型基本都这样干。
"""

import math
from typing import Dict, List, Tuple, Optional

from ..domain.sports.football import value as _v

calculate_value = _v.calculate_value
calculate_ev = _v.calculate_ev
adjust_by_value = _v.adjust_by_value
identify_value_bets = _v.identify_value_bets
