"""Lightweight acceptance check for Prompt 2.3 mixed-output normalization."""

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
    """Run Prompt 2.3 acceptance checks without heavy compute."""

    failures: list[str] = []
    tooling_text = read_text(REPO_ROOT / "src/agentiad_recon/tooling.py")
    baseline_text = read_text(REPO_ROOT / "src/agentiad_recon/baseline.py")
    readme_text = read_text(REPO_ROOT / "README.md").lower()
    worklog_text = read_text(
        REPO_ROOT / "Working Log/2026-03-14 Prompt 2.3 mixed-output runtime normalization.md"
    ).lower()

    for relative in [
        "src/agentiad_recon/tooling.py",
        "src/agentiad_recon/baseline.py",
        "tests/test_prompt_2_3_mixed_output_runtime_normalization.py",
        "audit/check_prompt_2_3.py",
    ]:
        require((REPO_ROOT / relative).exists(), f"required file exists: {relative}", failures)

    for phrase in [
        "class ProtocolDecision",
        "def normalize_protocol_turn",
        "mixed_tool_call_and_final_answer",
        "additional_valid_tool_calls_discarded",
        'raise ToolContractError("Output cannot contain both a tool call and a final answer block")',
    ]:
        require(phrase in tooling_text, f"tooling normalization hook present: {phrase}", failures)

    for phrase in [
        "normalize_protocol_turn(response.raw_output, tool_path=definition[\"mode\"])",
        "_normalization_sidecar_path",
        "\"normalization_applied\": True",
        "\"normalization_events\": normalization_events",
        "\"normalization_summary\": normalization_summary",
    ]:
        require(phrase in baseline_text, f"baseline audit hook present: {phrase}", failures)

    require(
        "filterwarnings" not in tooling_text and "filterwarnings" not in baseline_text,
        "Prompt 2.3 does not suppress warnings or swallow mixed-output evidence",
        failures,
    )
    require(
        "mixed tool-call/final-answer outputs are normalized" in readme_text
        and "premature final answer" in readme_text,
        "README documents Prompt 2.3 mixed-output normalization",
        failures,
    )

    for phrase in [
        "timestamp",
        "prompt number",
        "what changed",
        "why it changed",
        "effects",
        "git-diff-style summary",
        "toolcontracterror",
        "raw output was not audit-visible enough",
        "this patch does not suppress or filter logs",
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
        print("\nPrompt 2.3 acceptance check FAILED.")
        for failure in failures:
            print(f"- {failure}")
        return 1

    print("\nPrompt 2.3 acceptance check PASSED.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
