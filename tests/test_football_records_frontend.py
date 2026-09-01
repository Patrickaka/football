# -*- coding: utf-8 -*-
"""足球赛程为空时，历史预测记录入口与加载链路仍须可用。"""

import unittest
from pathlib import Path


HTML = Path('web/index.html').read_text(encoding='utf-8')


class FootballRecordsWithoutMatches(unittest.TestCase):

    def test_empty_schedule_uses_the_existing_record_toolbar(self):
        load_matches = HTML.split('async function loadMatches()', 1)[1].split(
            'let footballProfessionalStatus', 1,
        )[0]
        self.assertNotIn(
            "if (!allMatches.length) return showError(",
            load_matches,
        )
        self.assertIn('await loadAllPredictions(data.predictions);', load_matches)

        renderer = HTML.split('function renderFootballPredictions(results)', 1)[1].split(
            'function setFootballQualityFilter', 1,
        )[0]
        self.assertIn("onclick=\"switchTab('predictions')\"", renderer)
        self.assertIn('未找到任何比赛数据，请稍后重试', renderer)
        self.assertIn('历史预测记录仍可通过上方按钮查看', renderer)

    def test_fetch_failure_still_offers_the_record_view(self):
        self.assertIn('function showFootballLoadError(text)', HTML)
        error_view = HTML.split('function showFootballLoadError(text)', 1)[1].split(
            'async function loadMatches()', 1,
        )[0]
        self.assertIn('onclick="loadMatches()"', error_view)
        self.assertIn('onclick="switchTab(\'predictions\')"', error_view)
        self.assertIn('历史预测记录独立保存', error_view)

        load_matches = HTML.split('async function loadMatches()', 1)[1].split(
            'let footballProfessionalStatus', 1,
        )[0]
        self.assertIn('return showFootballLoadError(`${data.error}', load_matches)
        self.assertIn("showFootballLoadError('网络请求失败：'", load_matches)

    def test_auxiliary_failures_do_not_hide_saved_records(self):
        loader = HTML.split('async function loadPredictions()', 1)[1].split(
            'async function exportPredictionRecords()', 1,
        )[0]
        self.assertIn("fetchJson('/api/predictions')", loader)
        self.assertIn(
            "optionalResult(fetchJsonWithTimeout('/api/sync/status', 5000))",
            loader,
        )
        self.assertIn(
            "optionalResult(fetchJsonWithTimeout('/api/football/diagnostics', 5000))",
            loader,
        )
        self.assertIn('if (recordsData.error) throw new Error(recordsData.error);', loader)
        self.assertIn('以下预测记录仍可正常查看', loader)

    def test_records_support_date_filter_and_match_number(self):
        loader = HTML.split('async function loadPredictions()', 1)[1].split(
            'async function exportPredictionRecords()', 1,
        )[0]
        self.assertIn('predictionRecordDateKey', loader)
        self.assertIn('prediction-date-trigger', loader)
        self.assertIn('prediction-calendar-popover', loader)
        self.assertIn('togglePredictionCalendar(event)', loader)
        self.assertIn('role="dialog"', loader)
        self.assertNotIn('class="prediction-date-input" type="date"', loader)
        self.assertNotIn('<select id="prediction-date-filter"', loader)
        self.assertIn('choosePredictionCalendarDate', HTML)
        self.assertIn('shiftPredictionCalendar', HTML)
        self.assertIn('setPredictionDateToday()', loader)
        self.assertIn('record.match_num', loader)
        self.assertIn('record.lottery_offer_matched === false', loader)
        self.assertIn('让球胜平负${handicapText}预测', loader)
        self.assertIn('主队 ${Number(handicap)', loader)
        self.assertIn('const hasSpf = spf.available', loader)
        self.assertIn('const hasRqspf = rqspf.available', loader)
        self.assertNotIn("detail: '暂无预测'", loader)


if __name__ == '__main__':
    unittest.main()
