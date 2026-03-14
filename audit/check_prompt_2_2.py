"""Lightweight acceptance check for Prompt 2.2 real generate-input device placement."""

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
    """Run Prompt 2.2 acceptance checks without heavy compute."""

    failures: list[str] = []
    backends_text = read_text(REPO_ROOT / "src/agentiad_recon/backends.py")
    readme_text = read_text(REPO_ROOT / "README.md").lower()
    worklog_text = read_text(
        REPO_ROOT / "Working Log/2026-03-14 Prompt 2.2 real generate input device fix.md"
    ).lower()

    for relative in [
        "src/agentiad_recon/backends.py",
        "tests/test_prompt_2_1_runtime_hygiene.py",
        "tests/test_prompt_2_2_runtime_device_fix.py",
        "audit/check_prompt_2_1.py",
        "audit/check_prompt_2_2.py",
    ]:
        require((REPO_ROOT / relative).exists(), f"required file exists: {relative}", failures)

    for phrase in [
        "_embedding_like_device",
        "_candidate_runtime_modules",
        "_module_parameter_device",
        "_module_hf_device",
        "encoded_inputs = self._move_batch_to_device(encoded_inputs, model_device)",
    ]:
        require(phrase in backends_text, f"backend device-fix hook present: {phrase}", failures)

    require(
        "filterwarnings" not in backends_text and "warnings.filterwarnings" not in backends_text,
        "device warning is not being suppressed by log filtering",
        failures,
    )
    require(
        "deterministic generation config is sanitized" in readme_text
        and "prompt 2.2" in readme_text,
        "README documents the Prompt 2.2 runtime-correct fix",
        failures,
    )

    for phrase in [
        "timestamp",
        "prompt number",
        "what changed",
        "why it changed",
        "effects",
        "git-diff-style summary",
        "prompt 2.1's fix was incomplete",
        "instead of suppressing or filtering the warning",
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
        print("\nPrompt 2.2 acceptance check FAILED.")
        for failure in failures:
            print(f"- {failure}")
        return 1

    print("\nPrompt 2.2 acceptance check PASSED.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
