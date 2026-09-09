from src.kl8.analyzer import KL8Analyzer


def test_requested_500_period_window_does_not_silently_truncate_to_250():
    analyzer = KL8Analyzer.__new__(KL8Analyzer)
    analyzer.history_file = ''
    analyzer._data_mtime = 0
    analyzer.using_simulated_data = False
    analyzer.history_data = [
        {'issue': str(2026501 - i), 'numbers': list(range(1, 21)) if i < 250 else list(range(21, 41))}
        for i in range(501)
    ]
    analyzer.update_statistics()
    assert analyzer.statistics['total_periods'] == 250
    assert analyzer.statistics['frequency'].get(21, 0) == 0
    extended = analyzer._build_window_analyzer(500)
    assert extended.statistics['total_periods'] == 500
    assert extended.statistics['frequency'][21] == 250
    assert analyzer.statistics['total_periods'] == 250  # no mutation of live default
