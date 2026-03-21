"""Acceptance check for Prompt 2.15 auditable PEFT adapter loading.

This checker validates that the unified `baseline.py` evaluator gained a real
adapter preflight path, explicit adapter-load provenance, and fail-fast
behavior without introducing a second evaluator or any heavy local compute.
"""

from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from agentiad_recon.baseline import dry_run_from_config


FIXTURE_ROOT = REPO_ROOT / "tests" / "fixtures" / "mmad_fixture"
REMOTE_NO_TOOLS_CONFIG = REPO_ROOT / "configs" / "eval_transformers_no_tools_remote_template.json"


def require(condition: bool, label: str, failures: list[str]) -> None:
    if condition:
        print(f"PASS: {label}")
    else:
        print(f"FAIL: {label}")
        failures.append(label)


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def score(failures: list[str]) -> str:
    if not failures:
        return "10/10"
    if len(failures) <= 2:
        return "8/10"
    if len(failures) <= 5:
        return "6/10"
    return "4/10"


def _run_python_module(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "agentiad_recon.baseline", *args],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
        env={"PYTHONPATH": str(REPO_ROOT / "src")},
    )


def main() -> int:
    failures: list[str] = []
    baseline_text = read_text(REPO_ROOT / "src/agentiad_recon/baseline.py")
    backends_text = read_text(REPO_ROOT / "src/agentiad_recon/backends.py")
    readme_text = read_text(REPO_ROOT / "README.md").lower()
    worklog_path = REPO_ROOT / "Working Log/2026-03-21 Prompt 2.15 auditable PEFT adapter loading.md"
    worklog_text = read_text(worklog_path).lower() if worklog_path.exists() else ""

    for relative in [
        "src/agentiad_recon/backends.py",
        "src/agentiad_recon/baseline.py",
        "src/agentiad_recon/contracts/schemas/baseline_run_definition.schema.json",
        "src/agentiad_recon/contracts/schemas/baseline_run_summary.schema.json",
        "src/agentiad_recon/contracts/schemas/baseline_metrics_report.schema.json",
        "src/agentiad_recon/contracts/schemas/artifact_manifest.schema.json",
        "tests/test_prompt_2_15_adapter_loading.py",
        "audit/check_prompt_2_15.py",
        "README.md",
    ]:
        require((REPO_ROOT / relative).exists(), f"required file exists: {relative}", failures)

    require("prepare_runtime" in backends_text, "backend exposes runtime preflight hook", failures)
    require('PeftModel.from_pretrained' in backends_text, "maintained PEFT load path is present", failures)
    require("adapter_load_attempted" in backends_text, "backend records adapter load attempts", failures)
    require("allow-missing-adapter" in baseline_text, "CLI exposes explicit adapter fallback opt-out", failures)
    require(
        "_prepare_backend_runtime_for_eval" in baseline_text,
        "runner preflights adapter-backed runtimes before inference",
        failures,
    )
    require(
        "argparse.ArgumentParser" in baseline_text and "run_from_config" in baseline_text,
        "single evaluator entrypoint remains baseline.py",
        failures,
    )

    help_result = _run_python_module("--help")
    require(help_result.returncode == 0, "baseline.py --help succeeds", failures)
    require("--allow-missing-adapter" in help_result.stdout, "--help exposes adapter fallback flag", failures)

    dry_run = dry_run_from_config(
        config_path=REMOTE_NO_TOOLS_CONFIG,
        dataset_root=FIXTURE_ROOT,
        artifact_root=REPO_ROOT / "dist" / "tmp" / "prompt_2_15_acceptance_dry_run",
        max_samples=1,
        runtime_overrides={
            "base_model_path": "/models/base",
            "adapter_checkpoint_path": "/checkpoints/run_x/checkpoint-553",
        },
    )
    runtime = dry_run["runtime_provenance"]
    require(runtime["adapter_checkpoint_path"] == "/checkpoints/run_x/checkpoint-553", "dry-run keeps adapter path provenance", failures)
    require(runtime["adapter_load_attempted"] is False, "dry-run does not pretend an adapter was loaded", failures)
    require(runtime["adapter_loaded"] is False, "dry-run keeps adapter_loaded false", failures)
    require("adapter_load_error" in runtime, "dry-run runtime provenance includes adapter error field", failures)
    require("allow_missing_adapter" in runtime, "dry-run runtime provenance includes fallback flag", failures)

    for phrase in [
        "prompt 2.15",
        "adapter_loaded",
        "allow-missing-adapter",
        "adapter_load_attempted",
        "tiny base+adapter smoke",
    ]:
        require(phrase in readme_text, f"README phrase present: {phrase}", failures)

    for phrase in [
        "what changed",
        "why it changed",
        "effect of the changes",
        "file-level summary",
    ]:
        require(phrase in worklog_text, f"working log includes: {phrase}", failures)

    suite = unittest.defaultTestLoader.discover(
        str(REPO_ROOT / "tests"),
        pattern="test_prompt_2_15_adapter_loading.py",
    )
    result = unittest.TextTestRunner(stream=sys.stdout, verbosity=1).run(suite)
    require(result.wasSuccessful(), "Prompt 2.15 regression tests pass", failures)

    print(f"Acceptance score: {score(failures)}")
    if failures:
        print("Prompt 2.15 acceptance check FAILED.")
        return 1
    print("Prompt 2.15 acceptance check PASSED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
