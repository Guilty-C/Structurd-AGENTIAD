"""Lightweight acceptance check for Prompt 1.3.

This script verifies that the canonical non-tool baseline path exists, that the
baseline run definition is frozen, and that the local fixture-backed smoke
tests pass. It intentionally avoids any heavy model execution or full dataset
evaluation.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))


def require(condition: bool, label: str, failures: list[str]) -> None:
    """Print one acceptance result and collect failures."""

    if condition:
        print(f"PASS: {label}")
    else:
        print(f"FAIL: {label}")
        failures.append(label)


def read_text(path: Path) -> str:
    """Read one UTF-8 text file from the repository."""

    return path.read_text(encoding="utf-8")


def main() -> int:
    """Run file-presence checks plus the Prompt 1.3 baseline smoke tests."""

    failures: list[str] = []
    readme_text = read_text(REPO_ROOT / "README.md")
    worklog_text = read_text(REPO_ROOT / "Working Log/2026-03-10 Prompt 1.3.md").lower()
    report_text = read_text(REPO_ROOT / "audit/reports/prompt_1_3_acceptance_report.md")
    config_text = read_text(REPO_ROOT / "configs/baseline_non_tool_fixture.json")

    for relative in [
        "src/agentiad_recon/backends.py",
        "src/agentiad_recon/baseline.py",
        "src/agentiad_recon/evaluation.py",
        "src/agentiad_recon/contracts/schemas/baseline_run_definition.schema.json",
        "src/agentiad_recon/contracts/schemas/baseline_prediction_record.schema.json",
        "src/agentiad_recon/contracts/schemas/baseline_metrics_report.schema.json",
        "src/agentiad_recon/contracts/schemas/baseline_run_summary.schema.json",
        "tests/test_prompt_1_3_baseline.py",
        "configs/baseline_non_tool_fixture.json",
    ]:
        require((REPO_ROOT / relative).exists(), f"required file exists: {relative}", failures)

    for phrase in ["Non-Tool Baseline", "mock backend", "tool-enabled inference", "baseline_non_tool_fixture.json"]:
        require(phrase in readme_text, f"README phrase present: {phrase}", failures)

    require('"mode": "no_tools"' in config_text, "baseline config disables tools", failures)
    require('"type": "mock"' in config_text, "baseline config uses mock backend for local smoke path", failures)

    for phrase in ["timestamp", "prompt number", "what changed", "why it changed", "effects", "git-diff-style summary"]:
        require(phrase in worklog_text, f"working log field present: {phrase}", failures)

    for phrase in ["self-score", "Heavy compute was NOT run", "Known Remaining Gaps"]:
        require(phrase in report_text, f"acceptance report field present: {phrase}", failures)

    suite = unittest.defaultTestLoader.discover(
        str(REPO_ROOT / "tests"),
        pattern="test_prompt_1_3_baseline.py",
    )
    result = unittest.TextTestRunner(stream=sys.stdout, verbosity=1).run(suite)
    require(result.wasSuccessful(), "Prompt 1.3 baseline smoke tests pass", failures)

    if failures:
        print("\nPrompt 1.3 acceptance check FAILED.")
        for failure in failures:
            print(f"- {failure}")
        return 1

    print("\nPrompt 1.3 acceptance check PASSED.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
