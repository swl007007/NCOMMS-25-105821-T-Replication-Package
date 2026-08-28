"""Release-level checks for the root replication workflow."""

from __future__ import annotations

import subprocess
import sys
import unittest

import run_replication


class ReplicationRunnerTests(unittest.TestCase):
    def test_readiness_contract_passes(self) -> None:
        self.assertEqual(run_replication.check_readiness(), [])

    def test_core_workflow_order_is_dependency_safe(self) -> None:
        scripts = [step[1].name for step in run_replication.WORKFLOW]
        self.assertEqual(
            scripts,
            [
                "generate_all_prediction_temporal_test.py",
                "generate_all_prediction_evaluation.py",
                "generate_all_prediction_temporal_test_evaluation.py",
                "generate_phase_cumulative_scatter_comparison.py",
            ],
        )

    def test_check_only_cli_does_not_fit_models(self) -> None:
        completed = subprocess.run(
            [sys.executable, str(run_replication.REPO_ROOT / "run_replication.py"), "--check-only"],
            cwd=run_replication.REPO_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("Readiness checks passed", completed.stdout)


if __name__ == "__main__":
    unittest.main()
