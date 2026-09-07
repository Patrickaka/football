# -*- coding: utf-8 -*-
"""快乐8视觉重构不能破坏既有前端功能钩子。"""
import unittest
from pathlib import Path


HTML = Path('web/index.html').read_text(encoding='utf-8')


class KL8VisualShell(unittest.TestCase):

    def test_primary_plays_come_first_and_the_rest_fold(self):
        renderer = HTML.split('function renderKL8(r)', 1)[1].split(
            'function getKL8ExcludeOptions()', 1,
        )[0]
        self.assertIn('KL8_PRIMARY_SELECTS.filter', renderer)
        self.assertLess(renderer.index('<h3>本期号码</h3>'), renderer.index('<h3>复式</h3>'))
        self.assertLess(renderer.index('<h3>复式</h3>'), renderer.index('<h3>更多玩法</h3>'))
        self.assertLess(renderer.index('<h3>更多玩法</h3>'), renderer.index('<h3>最近开奖</h3>'))
        self.assertIn('<details class="card kl8-panel kl8-fold">', renderer)
        self.assertEqual(HTML.count('KL8_PRIMARY_SELECTS = [5, 6, 10];'), 1)

    def test_copy_is_chinese_only_on_the_kl8_page(self):
        for english in ('Statistical Console', 'Draw history', 'Number selections',
                        'Heuristic ranking', 'strategy /', 'random E='):
            with self.subTest(text=english):
                self.assertNotIn(english, HTML)

    def test_dashboard_is_scoped_to_the_kl8_tab(self):
        self.assertIn(
            "document.body.classList.toggle('kl8-view', tab === 'kl8');",
            HTML,
        )
        self.assertIn('<div class="kl8-dashboard">', HTML)
        self.assertIn('class="kl8-hero kl8-topbar"', HTML)
        self.assertIn('class="kl8-picks-grid kl8-primary-grid"', HTML)
        self.assertIn('class="kl8-tools-row"', HTML)

    def test_all_five_existing_actions_remain_available(self):
        renderer = HTML.split('function renderKL8(r)', 1)[1].split(
            'function getKL8ExcludeOptions()', 1,
        )[0]
        handlers = (
            'onclick="refreshKL8()"',
            'onclick="openKL8ExcludeModal()"',
            'onclick="openKL8KillModal()"',
            'onclick="fetchKL8Data()"',
            'onclick="openKL8RecordsModal()"',
        )
        for handler in handlers:
            with self.subTest(handler=handler):
                self.assertEqual(renderer.count(handler), 1)

        self.assertIn('id="kl8-parameter-search-result"', renderer)
        self.assertIn('id="kl8-exclude-modal"', renderer)
        self.assertIn('id="kl8-records-modal"', renderer)

    def test_recalculation_and_records_hooks_survive_the_new_modals(self):
        required = (
            'name="kl8-exclude-play"',
            'id="kl8-exclude-options"',
            'id="kl8-exclude-result"',
            'name="kl8-kill-play"',
            'id="kl8-kill-input"',
            'id="kl8-kill-options"',
            'id="kl8-kill-result"',
            'id="kl8-records-body"',
            'id="kl8-records-summary"',
            'id="kl8-records-list"',
            'id="kl8-records-pager"',
        )
        for hook in required:
            with self.subTest(hook=hook):
                self.assertIn(hook, HTML)
        self.assertGreaterEqual(HTML.count('class="kl8-modal-backdrop"'), 3)

    def test_mobile_layout_stacks_cards_and_actions(self):
        self.assertIn('@media (max-width: 768px)', HTML)
        self.assertIn('.kl8-picks-grid, .kl8-primary-grid { grid-template-columns: 1fr;', HTML)
        self.assertIn('.kl8-topbar-actions { position: static;', HTML)
        self.assertIn('max-height: min(92dvh, 860px);', HTML)
        self.assertIn('grid-template-areas: "issue date" "balls balls";', HTML)

    def test_fushi7_recalculation_excludes_all_seven_numbers(self):
        self.assertNotIn(
            "recalculationNumbers: r.fu_shi_7?.select_6_numbers",
            HTML,
        )
        self.assertEqual(
            HTML.count(
                'option.currentRecalculationNumbers = newNumbers;'
            ),
            2,
        )
        self.assertEqual(
            HTML.count(
                'option.currentRecalculationNumbers || option.currentNumbers || []'
            ),
            2,
        )
        self.assertIn(
            'removeNow.length < option.requiredRecalculationPick',
            HTML,
        )


if __name__ == '__main__':
    unittest.main()
