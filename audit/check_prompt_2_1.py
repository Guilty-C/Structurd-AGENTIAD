"""Lightweight acceptance check for Prompt 2.1 runtime hygiene.

This script verifies that the real transformers checkpoint-eval backend now
contains explicit device-placement and deterministic-generation hygiene helpers,
that the Prompt 2.0 checkpoint-eval path still exists, and that the relevant
lightweight unit tests pass without any real model inference.
"""

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
    """Run Prompt 2.1 acceptance checks without heavy compute."""

    failures: list[str] = []
    backends_text = read_text(REPO_ROOT / "src/agentiad_recon/backends.py")
    readme_text = read_text(REPO_ROOT / "README.md").lower()
    worklog_text = read_text(
        REPO_ROOT / "Working Log/2026-03-14 Prompt 2.1 eval runtime device placement and generation hygiene.md"
    ).lower()

    for relative in [
        "src/agentiad_recon/backends.py",
        "tests/test_prompt_2_0_checkpoint_eval.py",
        "tests/test_prompt_2_1_runtime_hygiene.py",
        "audit/check_prompt_2_0.py",
        "audit/check_prompt_2_1.py",
    ]:
        require((REPO_ROOT / relative).exists(), f"required file exists: {relative}", failures)

    for phrase in [
        "_infer_model_device",
        "_move_batch_to_device",
        "_sanitize_generation_kwargs",
        "encoded_inputs = self._move_batch_to_device(encoded_inputs, model_device)",
        'generation_config = self._sanitize_generation_kwargs(self.runtime_config["generation"])',
    ]:
        require(phrase in backends_text, f"backend runtime hygiene hook present: {phrase}", failures)

    require(
        "TransformersVisionLanguageBackend" in backends_text and "PeftModel" in backends_text,
        "Prompt 2.0 real checkpoint-eval backend still exists",
        failures,
    )
    require(
        "device placement is handled inside the backend runtime path" in readme_text,
        "README documents backend-side device placement",
        failures,
    )
    require(
        "deterministic generation config is sanitized" in readme_text,
        "README documents deterministic generation hygiene",
        failures,
    )

    for phrase in [
        "timestamp",
        "prompt number",
        "what changed",
        "why it changed",
        "effects",
        "git-diff-style summary",
    ]:
        require(phrase in worklog_text, f"working log field present: {phrase}", failures)

    sys.path.insert(0, str(REPO_ROOT / "src"))
    suite = unittest.defaultTestLoader.discover(
        str(REPO_ROOT / "tests"),
        pattern="test_prompt_2_*.py",
    )
    result = unittest.TextTestRunner(stream=sys.stdout, verbosity=1).run(suite)
    require(result.wasSuccessful(), "Prompt 2.0 and 2.1 lightweight tests pass", failures)

    if failures:
        print("\nPrompt 2.1 acceptance check FAILED.")
        for failure in failures:
            print(f"- {failure}")
        return 1

    print("\nPrompt 2.1 acceptance check PASSED.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
