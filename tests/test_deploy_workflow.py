# -*- coding: utf-8 -*-
"""生产部署必须证明测试、代码版本和运行进程是同一个提交。"""

import pathlib
import unittest


WORKFLOW = pathlib.Path('.github/workflows/deploy.yml').read_text(encoding='utf-8')


class DeployWorkflowTests(unittest.TestCase):

    def test_deploy_waits_for_the_test_workflow(self):
        self.assertIn('workflow_run:', WORKFLOW)
        self.assertIn('workflows: [Test]', WORKFLOW)
        self.assertIn("workflow_run.conclusion == 'success'", WORKFLOW)
        self.assertNotIn('  push:\n', WORKFLOW)

    def test_remote_script_fails_fast_and_deploys_the_tested_sha(self):
        self.assertIn('set -Eeuo pipefail', WORKFLOW)
        self.assertIn('git merge --ff-only "${DEPLOY_SHA}"', WORKFLOW)
        self.assertIn('test "$(git rev-parse HEAD)" = "${DEPLOY_SHA}"', WORKFLOW)

    def test_runtime_health_must_report_the_same_sha(self):
        self.assertIn('deployed_revision', WORKFLOW)
        self.assertIn('http://127.0.0.1:9004/healthz', WORKFLOW)
        self.assertIn('"revision\\\":\\\"${DEPLOY_SHA}', WORKFLOW)


if __name__ == '__main__':
    unittest.main()
