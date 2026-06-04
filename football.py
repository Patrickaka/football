"""入口模块 - 足球分析"""
from src.football import (
    fetch_match_list,
    analyze_match,
    FootballAnalyzer,
)

__all__ = ['fetch_match_list', 'analyze_match', 'FootballAnalyzer']
