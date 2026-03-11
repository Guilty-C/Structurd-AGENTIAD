"""Lightweight acceptance check for Prompt 1.5.

This script verifies that the repository now exports canonical `pz_only` and
`pz_cr` SFT trajectories, freezes a unified dataset contract, projects that
dataset through a thin MS-Swift adapter layer, and only runs local format and
sanity checks. It does not run full SFT.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from agentiad_recon.ms_swift_adapter import swift_runtime_probe
from agentiad_recon.sft import run_prompt_1_5_export


FIXTURE_ROOT = REPO_ROOT / "tests" / "fixtures" / "mmad_fixture"
EXPORT_CONFIG = REPO_ROOT / "configs" / "sft_export_fixture.json"
SWIFT_RECIPE = REPO_ROOT / "configs" / "ms_swift_sft_fixture.json"
IMAGE_PLACEHOLDER_TOKEN = "<image>"


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
    """Run Prompt 1.5 file checks plus fixture-backed export validation."""

    failures: list[str] = []
    readme_text = read_text(REPO_ROOT / "README.md")
    config_readme_text = read_text(REPO_ROOT / "configs/README.md")
    worklog_text = read_text(REPO_ROOT / "Working Log/2026-03-10 Prompt 1.5.md").lower()
    report_text = read_text(REPO_ROOT / "audit/reports/prompt_1_5_acceptance_report.md")

    for relative in [
        "src/agentiad_recon/sft.py",
        "src/agentiad_recon/ms_swift_adapter.py",
        "src/agentiad_recon/contracts/schemas/sft_export_definition.schema.json",
        "src/agentiad_recon/contracts/schemas/sft_dataset_record.schema.json",
        "src/agentiad_recon/contracts/schemas/ms_swift_recipe.schema.json",
        "src/agentiad_recon/contracts/schemas/ms_swift_record.schema.json",
        "configs/sft_export_fixture.json",
        "configs/ms_swift_sft_fixture.json",
        "configs/ms_swift_sft_remote_template.json",
        "tests/test_prompt_1_5_sft_export.py",
    ]:
        require((REPO_ROOT / relative).exists(), f"required file exists: {relative}", failures)

    for phrase in [
        "Trajectory reconstruction for SFT",
        "MS-Swift",
        "decisive-turn loss mask",
        "sft_export_fixture.json",
        "ms_swift_sft_fixture.json",
        "remote-only",
    ]:
        require(phrase in readme_text, f"README phrase present: {phrase}", failures)

    for phrase in [
        "sft_export_fixture.json",
        "ms_swift_sft_fixture.json",
        "ms_swift_sft_remote_template.json",
    ]:
        require(phrase in config_readme_text, f"config README phrase present: {phrase}", failures)

    for phrase in ["timestamp", "prompt number", "what changed", "why it changed", "effects", "git-diff-style summary"]:
        require(phrase in worklog_text, f"working log field present: {phrase}", failures)

    for phrase in ["self-score", "Heavy compute was NOT run", "Known remaining gaps"]:
        require(phrase in report_text, f"acceptance report field present: {phrase}", failures)

    with tempfile.TemporaryDirectory() as tempdir:
        artifacts = run_prompt_1_5_export(
            export_config_path=EXPORT_CONFIG,
            swift_recipe_path=SWIFT_RECIPE,
            dataset_root=FIXTURE_ROOT,
            output_root=tempdir,
            max_samples_per_mode=2,
        )
        canonical_records = [
            json.loads(line)
            for line in Path(artifacts.canonical_dataset_path).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        swift_records = [
            json.loads(line)
            for line in Path(artifacts.swift_dataset_path).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

        require(len(canonical_records) == 4, "canonical SFT export contains both modes across two samples", failures)
        require({record["trajectory_mode"] for record in canonical_records} == {"pz_only", "pz_cr"}, "both trajectory modes are exported", failures)
        require(
            all(record["loss_summary"]["decisive_message_indices"] for record in canonical_records),
            "decisive-turn loss targets are exported",
            failures,
        )
        require(
            any(record["trajectory_mode"] == "pz_cr" and len(record["tool_events"]) == 2 for record in canonical_records),
            "pz_cr records keep both tool events",
            failures,
        )
        require(
            any(record["trajectory_mode"] == "pz_only" and len(record["tool_events"]) == 1 for record in canonical_records),
            "pz_only records keep one visual tool event",
            failures,
        )
        require(len(swift_records) == 4, "MS-Swift adapter dataset record count matches canonical export", failures)
        require(
            all(any(message["loss"] for message in record["messages"]) for record in swift_records),
            "MS-Swift adapter preserves masked assistant targets",
            failures,
        )
        require(
            all(
                isinstance(message["content"], str)
                and not isinstance(message["content"], (list, dict))
                for record in swift_records
                for message in record["messages"]
            ),
            "MS-Swift message content is string-only",
            failures,
        )
        require(
            all(
                sum(message["content"].count(IMAGE_PLACEHOLDER_TOKEN) for message in record["messages"])
                == len(record["images"])
                for record in swift_records
            ),
            "MS-Swift placeholder count matches top-level images length",
            failures,
        )
        require(
            all(
                len(record["images"]) == len(set(record["images"]))
                for record in swift_records
            ),
            "MS-Swift top-level images are first-occurrence ordered without duplicates",
            failures,
        )
        require(
            artifacts.local_validation["exemplar_linkage_checked"],
            "local validation checks exemplar linkage",
            failures,
        )
        require(
            isinstance(artifacts.swift_runtime_check["available"], bool),
            "local run reports MS-Swift runtime availability as a boolean probe",
            failures,
        )

    runtime_probe = swift_runtime_probe()
    require(isinstance(runtime_probe["available"], bool), "runtime probe returns boolean availability", failures)

    suite = unittest.defaultTestLoader.discover(
        str(REPO_ROOT / "tests"),
        pattern="test_prompt_1_5_sft_export.py",
    )
    result = unittest.TextTestRunner(stream=sys.stdout, verbosity=1).run(suite)
    require(result.wasSuccessful(), "Prompt 1.5 SFT export tests pass", failures)

    if failures:
        print("\nPrompt 1.5 acceptance check FAILED.")
        for failure in failures:
            print(f"- {failure}")
        return 1

    print("\nPrompt 1.5 acceptance check PASSED.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
