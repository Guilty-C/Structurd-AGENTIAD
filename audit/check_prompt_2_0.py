"""Lightweight acceptance check for Prompt 2.0 checkpoint evaluation wiring.

This script verifies that the existing `baseline.py` entrypoint can accept base
model plus optional adapter surfaces, that a real transformers+PEFT backend
exists as the chosen non-skeleton path, and that local fixture/mocked checks
cover provenance-bearing summary and manifest artifacts without running heavy
inference or training.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from agentiad_recon.baseline import dry_run_from_config, run_baseline


FIXTURE_ROOT = REPO_ROOT / "tests" / "fixtures" / "mmad_fixture"
BASELINE_CONFIG = REPO_ROOT / "configs" / "baseline_non_tool_fixture.json"
REMOTE_NO_TOOLS_CONFIG = REPO_ROOT / "configs" / "eval_transformers_no_tools_remote_template.json"
REMOTE_PZ_CR_CONFIG = REPO_ROOT / "configs" / "eval_transformers_pz_cr_remote_template.json"


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
    """Run Prompt 2.0 acceptance checks without heavy local compute."""

    failures: list[str] = []
    baseline_text = read_text(REPO_ROOT / "src/agentiad_recon/baseline.py")
    backends_text = read_text(REPO_ROOT / "src/agentiad_recon/backends.py")
    readme_text = read_text(REPO_ROOT / "README.md")
    worklog_text = read_text(REPO_ROOT / "Working Log/2026-03-13 Prompt 2.0 checkpoint eval path.md").lower()

    for relative in [
        "src/agentiad_recon/baseline.py",
        "src/agentiad_recon/backends.py",
        "src/agentiad_recon/evaluation.py",
        "src/agentiad_recon/contracts/schemas/baseline_run_definition.schema.json",
        "src/agentiad_recon/contracts/schemas/baseline_run_summary.schema.json",
        "src/agentiad_recon/contracts/schemas/baseline_metrics_report.schema.json",
        "src/agentiad_recon/contracts/schemas/artifact_manifest.schema.json",
        "configs/eval_transformers_no_tools_remote_template.json",
        "configs/eval_transformers_pz_cr_remote_template.json",
        "tests/test_prompt_2_0_checkpoint_eval.py",
    ]:
        require((REPO_ROOT / relative).exists(), f"required file exists: {relative}", failures)

    for phrase in [
        "--base-model-path",
        "--adapter-checkpoint-path",
        "--dry-run",
        "dry_run_from_config",
    ]:
        require(phrase in baseline_text, f"baseline entrypoint surface present: {phrase}", failures)

    require(
        "TransformersVisionLanguageBackend" in backends_text,
        "transformers backend exists for real checkpoint evaluation",
        failures,
    )
    require(
        'import_module("peft")' in backends_text and "from_pretrained(" in backends_text,
        "adapter loading is wired into the backend implementation",
        failures,
    )
    require(
        "maintained-runtime skeleton only" in backends_text and "TransformersVisionLanguageBackend" in backends_text,
        "the chosen backend path is not the old skeleton-only route",
        failures,
    )
    require(
        "argparse.ArgumentParser" in baseline_text and "src/agentiad_recon/evaluation.py" not in baseline_text,
        "no second evaluator CLI was introduced",
        failures,
    )

    for phrase in [
        "Prompt 2.0",
        "checkpoint",
        "transformers",
        "adapter_checkpoint_path",
        "baseline.py",
    ]:
        require(phrase.lower() in readme_text.lower(), f"README phrase present: {phrase}", failures)

    for phrase in [
        "timestamp",
        "prompt number",
        "what changed",
        "why it changed",
        "effects",
        "git-diff-style summary",
    ]:
        require(phrase in worklog_text, f"working log field present: {phrase}", failures)

    dry_run = dry_run_from_config(
        config_path=REMOTE_PZ_CR_CONFIG,
        dataset_root=FIXTURE_ROOT,
        artifact_root=REPO_ROOT / "dist" / "tmp" / "prompt_2_0_acceptance_dry_run",
        max_samples=1,
        runtime_overrides={
            "base_model_path": "/models/base",
            "adapter_checkpoint_path": "/checkpoints/run_x/checkpoint-553",
        },
    )
    runtime = dry_run["runtime_provenance"]
    require(runtime["runtime_backend_type"] == "transformers", "dry-run resolves the transformers backend", failures)
    require(runtime["checkpoint_step"] == 553, "checkpoint step provenance is inferred", failures)
    require(
        runtime["adapter_checkpoint_path"] == "/checkpoints/run_x/checkpoint-553",
        "dry-run carries adapter checkpoint provenance",
        failures,
    )

    with tempfile.TemporaryDirectory(prefix="prompt_2_0_acceptance_") as tempdir:
        result = run_baseline(
            config_path=BASELINE_CONFIG,
            dataset_root=FIXTURE_ROOT,
            artifact_root=tempdir,
            max_samples=2,
            runtime_overrides={
                "base_model_path": "/models/base",
                "adapter_checkpoint_path": "/checkpoints/run_x/checkpoint-553",
            },
        )
        summary = json.loads(Path(result["summary_path"]).read_text(encoding="utf-8"))
        manifest = json.loads(Path(result["run_manifest_path"]).read_text(encoding="utf-8"))

    require("runtime_provenance" in summary, "run summary includes runtime provenance", failures)
    require("run_provenance" in manifest, "artifact manifest includes runtime provenance", failures)
    require(
        summary["runtime_provenance"]["adapter_checkpoint_path"] == "/checkpoints/run_x/checkpoint-553",
        "summary records adapter checkpoint path",
        failures,
    )
    require(
        summary["runtime_provenance"]["adapter_loaded"] is False,
        "local acceptance does not pretend a mock run loaded the adapter",
        failures,
    )
    require(
        summary["runtime_provenance"]["tool_mode"] == "no_tools",
        "canonical scientific target remains unchanged for no_tools baseline",
        failures,
    )

    suite = unittest.defaultTestLoader.discover(
        str(REPO_ROOT / "tests"),
        pattern="test_prompt_2_0_checkpoint_eval.py",
    )
    test_result = unittest.TextTestRunner(stream=sys.stdout, verbosity=1).run(suite)
    require(test_result.wasSuccessful(), "Prompt 2.0 checkpoint-eval tests pass", failures)

    if failures:
        print("\nPrompt 2.0 acceptance check FAILED.")
        for failure in failures:
            print(f"- {failure}")
        return 1

    print("\nPrompt 2.0 acceptance check PASSED.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
