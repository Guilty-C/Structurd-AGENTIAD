"""Lightweight acceptance check for Prompt 2.4 prompt audit and mode-contract fix."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def require(condition: bool, label: str, failures: list[str]) -> None:
    """Print one acceptance result and collect failures."""

    if condition:
        print(f"PASS: {label}")
    else:
        print(f"FAIL: {label}")
        failures.append(label)


def read_text(path: Path) -> str:
    """Read one UTF-8 file."""

    return path.read_text(encoding="utf-8")


def main() -> int:
    """Run Prompt 2.4 acceptance checks without heavy compute."""

    failures: list[str] = []
    baseline_text = read_text(REPO_ROOT / "src/agentiad_recon/baseline.py")
    evaluation_text = read_text(REPO_ROOT / "src/agentiad_recon/evaluation.py")
    prediction_schema = read_text(REPO_ROOT / "src/agentiad_recon/contracts/schemas/baseline_prediction_record.schema.json")
    metrics_schema = read_text(REPO_ROOT / "src/agentiad_recon/contracts/schemas/baseline_metrics_report.schema.json")
    summary_schema = read_text(REPO_ROOT / "src/agentiad_recon/contracts/schemas/baseline_run_summary.schema.json")
    manifest_schema = read_text(REPO_ROOT / "src/agentiad_recon/contracts/schemas/artifact_manifest.schema.json")
    readme_text = read_text(REPO_ROOT / "README.md").lower()
    worklog_text = read_text(
        REPO_ROOT / "Working Log/2026-03-16 Prompt 2.4 pz_cr runtime prompt audit and mode-contract fix.md"
    ).lower()

    for relative in [
        "src/agentiad_recon/baseline.py",
        "src/agentiad_recon/evaluation.py",
        "src/agentiad_recon/contracts/schemas/baseline_prediction_record.schema.json",
        "src/agentiad_recon/contracts/schemas/baseline_metrics_report.schema.json",
        "src/agentiad_recon/contracts/schemas/baseline_run_summary.schema.json",
        "src/agentiad_recon/contracts/schemas/artifact_manifest.schema.json",
        "tests/test_prompt_2_4_prompt_audit_mode_contract.py",
        "audit/check_prompt_2_4.py",
    ]:
        require((REPO_ROOT / relative).exists(), f"required file exists: {relative}", failures)

    for phrase in [
        "def _prompt_audit_payload",
        "def _prompt_audit_summary",
        "def _prompt_audit_sidecar_path",
        "mode_contract_missing_cr_tool",
        "mode_contract_pz_only_leakage",
        "prompt_audit[\"mode_contract_mismatch\"]",
    ]:
        require(phrase in baseline_text, f"baseline prompt-audit hook present: {phrase}", failures)

    for phrase in [
        "failure_reason",
        "prompt_audit_summary",
    ]:
        require(phrase in evaluation_text, f"evaluation contract hook present: {phrase}", failures)
        require(phrase in prediction_schema or phrase in metrics_schema or phrase in summary_schema or phrase in manifest_schema, f"schema hook present: {phrase}", failures)

    require(
        "first-turn prompt audit sidecars" in readme_text and "concrete `failure_reason`" in readme_text,
        "README documents Prompt 2.4 prompt audit and failure reasons",
        failures,
    )

    for phrase in [
        "timestamp",
        "prompt number",
        "what changed",
        "why it changed",
        "effects",
        "git-diff-style summary",
        "toolcall_rate = 0.0",
        "pz_only wording",
        "the blocker is eval/runtime mode-contract mismatch",
    ]:
        require(phrase in worklog_text, f"working log field present: {phrase}", failures)

    sys.path.insert(0, str(REPO_ROOT / "src"))
    suite = unittest.defaultTestLoader.discover(
        str(REPO_ROOT / "tests"),
        pattern="test_prompt_2_*.py",
    )
    result = unittest.TextTestRunner(stream=sys.stdout, verbosity=1).run(suite)
    require(result.wasSuccessful(), "Prompt 2.x lightweight tests pass", failures)

    if failures:
        print("\nPrompt 2.4 acceptance check FAILED.")
        for failure in failures:
            print(f"- {failure}")
        return 1

    print("\nPrompt 2.4 acceptance check PASSED.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
