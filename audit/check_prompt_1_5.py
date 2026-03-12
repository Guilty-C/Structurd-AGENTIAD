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
                isinstance(image, str)
                and bool(image.strip())
                and not isinstance(image, dict)
                for record in swift_records
                for image in record["images"]
            ),
            "MS-Swift images are non-empty string paths (not dict entries)",
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
        require(Path(artifacts.swift_length_audit_path).exists(), "true length-audit sidecar exists", failures)
        require(Path(artifacts.swift_proxy_length_audit_path).exists(), "proxy length-audit sidecar exists", failures)
        length_audit = json.loads(Path(artifacts.swift_length_audit_path).read_text(encoding="utf-8"))
        swift_manifest = json.loads(Path(artifacts.swift_manifest_path).read_text(encoding="utf-8"))
        require(
            all(key in length_audit for key in ["record_count", "p50", "p90", "p95", "p99", "max", "top_offenders"]),
            "length audit includes percentile and offender fields",
            failures,
        )
        print(
            "INFO: true_length_audit_summary",
            json.dumps(
                {
                    "backend": length_audit.get("backend"),
                    "true_multimodal_encode": length_audit.get("true_multimodal_encode"),
                    "record_count": length_audit.get("record_count"),
                    "p95": length_audit.get("p95"),
                    "max": length_audit.get("max"),
                    "count_above_4096": length_audit.get("count_above_4096"),
                    "count_above_8192": length_audit.get("count_above_8192"),
                },
                sort_keys=True,
            ),
        )
        require(
            all(
                "artifact_path" not in message["content"] and "coordinate_convention" not in message["content"]
                for record in swift_records
                for message in record["messages"]
                if message["role"] == "tool"
            ),
            "compact tool responses omit verbose artifact metadata",
            failures,
        )
        require(
            all(
                message["content"].startswith("<tool_call>")
                and "</tool_call>" in message["content"]
                and "<think>" not in message["content"]
                for record in swift_records
                for message in record["messages"]
                if message.get("role") == "assistant" and message.get("tool_name") is not None
            ),
            "assistant tool-request content is compact tool_call text",
            failures,
        )
        require(
            isinstance(artifacts.swift_runtime_check["available"], bool),
            "local run reports MS-Swift runtime availability as a boolean probe",
            failures,
        )
        require(Path(artifacts.resolved_remote_surfaces_path).exists(), "resolved remote surfaces artifact exists", failures)
        require(
            all(
                key in artifacts.resolved_remote_surfaces_summary
                for key in [
                    "length_audit_backend",
                    "true_multimodal_encode",
                    "threshold_clean_basis",
                    "strict_true_length_audit_requested",
                    "strict_true_length_audit_passed",
                ]
            ),
            "resolved remote surfaces summary contains truthful audit fields",
            failures,
        )
        require(
            all(key in artifacts.swift_filtered_manifests for key in ["4096", "8192"]),
            "filtered manifest mapping includes 4096 and 8192",
            failures,
        )
        require(
            "filtered_export_summary_by_threshold" in swift_manifest,
            "main manifest includes filtered-export summary by threshold",
            failures,
        )
        true_lengths = {row["id"]: row["encoded_length"] for row in length_audit["lengths"]}
        for threshold in (4096, 8192):
            threshold_key = str(threshold)
            manifest_path = Path(artifacts.swift_filtered_manifests[threshold_key])
            require(manifest_path.exists(), f"filtered manifest exists for <= {threshold}", failures)
            filtered_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            require(filtered_manifest["threshold"] == threshold, f"filtered manifest threshold matches {threshold}", failures)
            require(
                all(
                    key in filtered_manifest
                    for key in [
                        "threshold_clean_basis",
                        "true_threshold_clean_certified",
                        "true_multimodal_encode",
                        "length_audit_backend",
                        "strict_true_length_audit_requested",
                        "strict_true_length_audit_passed",
                        "source_swift_dataset_path",
                        "source_true_audit_path",
                        "dropped_ratio",
                        "top_dropped_offenders",
                        "max_kept_encoded_length",
                        "min_dropped_encoded_length",
                    ]
                ),
                f"filtered manifest includes truthful fields for <= {threshold}",
                failures,
            )
            filtered_dataset_path = Path(filtered_manifest["kept_dataset_path"])
            require(filtered_dataset_path.exists(), f"filtered dataset exists for <= {threshold}", failures)
            filtered_records = [
                json.loads(line)
                for line in filtered_dataset_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            require(
                all(true_lengths[record["id"]] <= threshold for record in filtered_records),
                f"filtered dataset is threshold-clean for <= {threshold}",
                failures,
            )
            require(
                filtered_manifest["kept_count"] + filtered_manifest["dropped_count"] == len(swift_records),
                f"filtered kept+dropped is consistent for <= {threshold}",
                failures,
            )
            require(
                abs(filtered_manifest["dropped_ratio"] - (filtered_manifest["dropped_count"] / len(swift_records))) < 1e-9,
                f"filtered dropped_ratio is consistent for <= {threshold}",
                failures,
            )
            require(
                Path(filtered_manifest["source_swift_dataset_path"]).resolve() == Path(artifacts.swift_dataset_path).resolve(),
                f"filtered manifest source dataset path is correct for <= {threshold}",
                failures,
            )
            require(
                Path(filtered_manifest["source_true_audit_path"]).resolve() == Path(artifacts.swift_length_audit_path).resolve(),
                f"filtered manifest source true-audit path is correct for <= {threshold}",
                failures,
            )
            require(filtered_manifest["kept_count"] > 0, f"filtered dataset is non-empty for <= {threshold}", failures)
            if filtered_manifest["kept_count"] > 0:
                require(
                    filtered_manifest["max_kept_encoded_length"] <= threshold,
                    f"max kept encoded length obeys threshold for <= {threshold}",
                    failures,
                )
            if filtered_manifest["dropped_count"] > 0:
                require(
                    filtered_manifest["min_dropped_encoded_length"] > threshold,
                    f"min dropped encoded length exceeds threshold for <= {threshold}",
                    failures,
                )
                offender_lengths = [row["encoded_length"] for row in filtered_manifest["top_dropped_offenders"]]
                require(
                    len(offender_lengths) > 0 and all(length > threshold for length in offender_lengths),
                    f"top dropped offenders are over-threshold for <= {threshold}",
                    failures,
                )
                require(
                    offender_lengths == sorted(offender_lengths, reverse=True),
                    f"top dropped offenders are sorted desc for <= {threshold}",
                    failures,
                )
            else:
                require(
                    filtered_manifest["min_dropped_encoded_length"] is None,
                    f"min dropped encoded length is null when no rows dropped for <= {threshold}",
                    failures,
                )
                require(
                    len(filtered_manifest["top_dropped_offenders"]) == 0,
                    f"top dropped offenders is empty when no rows dropped for <= {threshold}",
                    failures,
                )
            summary = swift_manifest["filtered_export_summary_by_threshold"][threshold_key]
            require(summary["threshold"] == threshold, f"main manifest summary threshold matches {threshold}", failures)
            require(summary["kept_count"] == filtered_manifest["kept_count"], f"main manifest kept_count matches for <= {threshold}", failures)
            require(summary["dropped_count"] == filtered_manifest["dropped_count"], f"main manifest dropped_count matches for <= {threshold}", failures)
            require(
                abs(summary["dropped_ratio"] - filtered_manifest["dropped_ratio"]) < 1e-9,
                f"main manifest dropped_ratio matches for <= {threshold}",
                failures,
            )
            if length_audit.get("true_multimodal_encode"):
                require(
                    filtered_manifest["true_threshold_clean_certified"] is True
                    and filtered_manifest["threshold_clean_basis"] == "true_multimodal_encode",
                    f"true audit certification is honest for <= {threshold}",
                    failures,
                )
            else:
                require(
                    filtered_manifest["true_threshold_clean_certified"] is False
                    and filtered_manifest["threshold_clean_basis"] == "fallback_derived_not_true_certified",
                    f"fallback audit cannot masquerade as true for <= {threshold}",
                    failures,
                )
            print(
                f"INFO: filtered_le{threshold}_summary",
                json.dumps(
                    {
                        "kept_count": filtered_manifest["kept_count"],
                        "dropped_count": filtered_manifest["dropped_count"],
                        "threshold": filtered_manifest["threshold"],
                        "threshold_clean_basis": filtered_manifest["threshold_clean_basis"],
                        "true_threshold_clean_certified": filtered_manifest["true_threshold_clean_certified"],
                    },
                    sort_keys=True,
                ),
            )

    with tempfile.TemporaryDirectory() as strict_tempdir:
        strict_failed = False
        strict_success_true_encode = False
        try:
            strict_artifacts = run_prompt_1_5_export(
                export_config_path=EXPORT_CONFIG,
                swift_recipe_path=SWIFT_RECIPE,
                dataset_root=FIXTURE_ROOT,
                output_root=strict_tempdir,
                max_samples_per_mode=1,
                strict_true_length_audit=True,
            )
            require(
                strict_artifacts.swift_length_audit_summary["true_multimodal_encode"] is True
                and strict_artifacts.resolved_remote_surfaces_summary["strict_true_length_audit_passed"] is True,
                "strict mode only succeeds with real true-audit encode",
                failures,
            )
            strict_success_true_encode = bool(strict_artifacts.swift_length_audit_summary["true_multimodal_encode"])
        except RuntimeError:
            strict_failed = True
        require(
            strict_failed or strict_success_true_encode,
            "strict mode fails when true encoder is unavailable",
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
