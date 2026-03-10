"""Lightweight acceptance check for Prompt 1.2.

This script validates that the local repository now contains one canonical
runtime waist for MMAD samples, deterministic tools, prompt/answer contracts,
and trace handling. It runs only fixture-backed tests and static/local checks.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))


def require(condition: bool, label: str, failures: list[str]) -> None:
    """Print one acceptance condition and collect failures."""

    if condition:
        print(f"PASS: {label}")
    else:
        print(f"FAIL: {label}")
        failures.append(label)


def read_text(path: Path) -> str:
    """Read one UTF-8 text file from the repository."""

    return path.read_text(encoding="utf-8")


def main() -> int:
    """Run repository checks and the Prompt 1.2 smoke-test suite."""

    failures: list[str] = []
    readme_text = read_text(REPO_ROOT / "README.md")
    contracts_readme = read_text(REPO_ROOT / "src/agentiad_recon/contracts/README.md")
    worklog_text = read_text(REPO_ROOT / "Working Log/2026-03-10 Prompt 1.2.md").lower()
    report_text = read_text(REPO_ROOT / "audit/reports/prompt_1_2_acceptance_report.md")

    for relative in [
        "src/agentiad_recon/mmad.py",
        "src/agentiad_recon/tooling.py",
        "src/agentiad_recon/prompting.py",
        "src/agentiad_recon/traces.py",
        "src/agentiad_recon/contracts/validation.py",
        "src/agentiad_recon/contracts/schemas/trace_record.schema.json",
        "tests/test_prompt_1_2_smoke.py",
        "tests/fixtures/mmad_fixture/fixture_manifest.json",
    ]:
        require((REPO_ROOT / relative).exists(), f"required file exists: {relative}", failures)

    for phrase in ["Canonical Runtime Waist", "Fixture vs Real Dataset Paths", "PZ", "CR"]:
        require(phrase in readme_text, f"README phrase present: {phrase}", failures)

    for phrase in ["trace_record.schema.json", "mask/ROI", "auditable output payloads"]:
        require(phrase in contracts_readme, f"contracts README phrase present: {phrase}", failures)

    for phrase in ["timestamp", "prompt number", "what changed", "why it changed", "effects", "git-diff-style summary"]:
        require(phrase in worklog_text, f"working log field present: {phrase}", failures)

    for phrase in ["self-score", "Heavy compute was NOT run", "Known Remaining Gaps"]:
        require(phrase in report_text, f"acceptance report field present: {phrase}", failures)

    # The unittest suite is the main local proof that D/E/F works end to end on fixtures.
    suite = unittest.defaultTestLoader.discover(str(REPO_ROOT / "tests"), pattern="test_prompt_1_2_smoke.py")
    result = unittest.TextTestRunner(stream=sys.stdout, verbosity=1).run(suite)
    require(result.wasSuccessful(), "Prompt 1.2 smoke tests pass", failures)

    if failures:
        print("\nPrompt 1.2 acceptance check FAILED.")
        for failure in failures:
            print(f"- {failure}")
        return 1

    print("\nPrompt 1.2 acceptance check PASSED.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
