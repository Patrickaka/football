# -*- coding: utf-8 -*-
"""快乐8视觉重构不能破坏既有前端功能钩子。"""
import unittest
from pathlib import Path


HTML = Path('web/index.html').read_text(encoding='utf-8')


class KL8VisualShell(unittest.TestCase):

    def test_dashboard_is_scoped_to_the_kl8_tab(self):
        self.assertIn(
            "document.body.classList.toggle('kl8-view', tab === 'kl8');",
            HTML,
        )
        self.assertIn('<div class="kl8-dashboard">', HTML)
        self.assertIn('class="kl8-hero"', HTML)
        self.assertIn('class="kl8-picks-grid"', HTML)
        self.assertIn('class="kl8-action-dock"', HTML)

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

    def test_mobile_layout_keeps_primary_record_action_full_width(self):
        self.assertIn('@media (max-width: 768px)', HTML)
        self.assertIn('.kl8-action-dock .action-success { grid-column: 1 / -1;', HTML)
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
