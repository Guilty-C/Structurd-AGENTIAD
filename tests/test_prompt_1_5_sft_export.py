"""Fixture-backed tests for the Prompt 1.5 SFT export and MS-Swift adapter layer.

These tests keep the Prompt 1.4 waist intact and verify that Prompt 1.5 adds
one canonical SFT exporter, one unified dataset contract, and one thin
MS-Swift-facing projection. They intentionally stop at local format, masking,
and linkage validation rather than running full training.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from agentiad_recon.mmad import MMADIndexer
from agentiad_recon.ms_swift_adapter import load_swift_recipe, swift_runtime_probe
from agentiad_recon.sft import (
    _build_training_trace,
    build_swift_records,
    build_unified_sft_record,
    export_sft_dataset,
    run_prompt_1_5_export,
)


FIXTURE_ROOT = REPO_ROOT / "tests" / "fixtures" / "mmad_fixture"
EXPORT_CONFIG = REPO_ROOT / "configs" / "sft_export_fixture.json"
SWIFT_RECIPE = REPO_ROOT / "configs" / "ms_swift_sft_fixture.json"
IMAGE_PLACEHOLDER_TOKEN = "<image>"


class Prompt15SFTExportTests(unittest.TestCase):
    """Local-only Prompt 1.5 tests for trajectory export and adapter wiring."""

    @classmethod
    def setUpClass(cls) -> None:
        """Load a deterministic training split from the MMAD fixture once."""

        cls.samples = [sample.to_dict() for sample in MMADIndexer(FIXTURE_ROOT, source="mmad_fixture").index_samples()]
        cls.train_samples = [sample for sample in cls.samples if sample["split"] == "train"]
        cls.sample = cls.train_samples[1]

    def test_pz_only_trajectory_has_expected_order_and_loss_targets(self) -> None:
        """The perceptive-only trajectory should keep the expected turn pattern."""

        with tempfile.TemporaryDirectory() as tempdir:
            trace = _build_training_trace(
                sample=self.sample,
                sample_pool=self.samples,
                mode="pz_only",
                artifact_root=Path(tempdir),
            )
            record = build_unified_sft_record(trace, self.sample)
            self.assertEqual(
                [message["message_type"] for message in record["messages"]],
                [
                    "system_instruction",
                    "user_prompt",
                    "reasoning",
                    "tool_request",
                    "tool_result",
                    "reasoning",
                    "final_answer",
                ],
            )
            self.assertEqual(record["loss_summary"]["decisive_message_indices"], [3, 5])
            self.assertEqual(record["loss_summary"]["loss_message_indices"], [3, 5, 6])
            self.assertEqual(record["tool_events"][0]["tool_name"], "PZ")
            self.assertEqual(len(record["tool_events"][0]["image_bindings"]), 1)

    def test_pz_cr_trajectory_links_exemplar_and_masks_last_visual_step(self) -> None:
        """The comparative trajectory should include auditable exemplar linkage."""

        with tempfile.TemporaryDirectory() as tempdir:
            trace = _build_training_trace(
                sample=self.sample,
                sample_pool=self.samples,
                mode="pz_cr",
                artifact_root=Path(tempdir),
            )
            record = build_unified_sft_record(trace, self.sample)
            self.assertEqual(
                [message["message_type"] for message in record["messages"]],
                [
                    "system_instruction",
                    "user_prompt",
                    "reasoning",
                    "tool_request",
                    "tool_result",
                    "reasoning",
                    "tool_request",
                    "tool_result",
                    "reasoning",
                    "final_answer",
                ],
            )
            exemplar = record["tool_events"][1]["output_payload"]["selected_exemplar"]
            self.assertEqual(exemplar["category"], record["sample"]["category"])
            self.assertNotEqual(exemplar["sample_id"], record["sample_id"])
            self.assertEqual(record["loss_summary"]["decisive_message_indices"], [6, 8])
            self.assertEqual(record["loss_summary"]["loss_message_indices"], [6, 8, 9])

    def test_unified_export_writes_both_modes_and_manifest(self) -> None:
        """The canonical exporter should write one dataset containing both modes."""

        with tempfile.TemporaryDirectory() as tempdir:
            records, metadata = export_sft_dataset(
                config_path=EXPORT_CONFIG,
                dataset_root=FIXTURE_ROOT,
                output_root=tempdir,
                max_samples_per_mode=2,
            )
            self.assertEqual(len(records), 4)
            self.assertEqual({record["trajectory_mode"] for record in records}, {"pz_only", "pz_cr"})
            manifest = json.loads(Path(metadata["manifest_path"]).read_text(encoding="utf-8"))
            self.assertEqual(manifest["record_count"], 4)
            self.assertEqual(manifest["sample_count_per_mode"], 2)

    def test_ms_swift_projection_is_thin_and_honest_about_local_runtime(self) -> None:
        """The adapter should validate configs and avoid faking local framework execution."""

        with tempfile.TemporaryDirectory() as tempdir:
            records, _metadata = export_sft_dataset(
                config_path=EXPORT_CONFIG,
                dataset_root=FIXTURE_ROOT,
                output_root=tempdir,
                max_samples_per_mode=1,
            )
            recipe = load_swift_recipe(SWIFT_RECIPE)
            swift_records = build_swift_records(records, recipe)
            self.assertEqual(len(swift_records), 2)
            self.assertEqual({record["metadata"]["trajectory_mode"] for record in swift_records}, {"pz_only", "pz_cr"})
            for canonical_record, swift_record in zip(records, swift_records, strict=True):
                self.assertTrue(all("messages" in record for record in swift_records))
                self.assertTrue(any(message["loss"] for message in swift_record["messages"]))
                self.assertTrue(all(isinstance(message["content"], str) for message in swift_record["messages"]))
                self.assertTrue(
                    all(not isinstance(message["content"], (list, dict)) for message in swift_record["messages"])
                )
                self.assertTrue(all(isinstance(image, str) and bool(image.strip()) for image in swift_record["images"]))
                self.assertTrue(all(not isinstance(image, dict) for image in swift_record["images"]))
                placeholder_count = sum(
                    message["content"].count(IMAGE_PLACEHOLDER_TOKEN) for message in swift_record["messages"]
                )
                self.assertEqual(placeholder_count, len(swift_record["images"]))
                self.assertTrue(
                    all(
                        "artifact_path" not in message["content"] and "coordinate_convention" not in message["content"]
                        for message in swift_record["messages"]
                        if message["role"] == "tool"
                    )
                )
                self.assertTrue(
                    all(
                        message["content"].startswith("<tool_call>")
                        and "</tool_call>" in message["content"]
                        and "<think>" not in message["content"]
                        for message in swift_record["messages"]
                        if message.get("role") == "assistant" and message.get("tool_name") is not None
                    )
                )

                expected_images: list[str] = []
                for message in canonical_record["messages"]:
                    for image_ref in message["image_refs"]:
                        if image_ref not in expected_images:
                            expected_images.append(image_ref)
                self.assertEqual(swift_record["images"], expected_images)

            runtime_probe = swift_runtime_probe()
            self.assertIsInstance(runtime_probe["available"], bool)
            self.assertIn("detail", runtime_probe)

    def test_prompt_1_5_export_runner_writes_both_dataset_layers(self) -> None:
        """The top-level Prompt 1.5 runner should emit canonical and MS-Swift artifacts."""

        with tempfile.TemporaryDirectory() as tempdir:
            artifacts = run_prompt_1_5_export(
                export_config_path=EXPORT_CONFIG,
                swift_recipe_path=SWIFT_RECIPE,
                dataset_root=FIXTURE_ROOT,
                output_root=tempdir,
                max_samples_per_mode=1,
            )
            self.assertTrue(Path(artifacts.canonical_dataset_path).exists())
            self.assertTrue(Path(artifacts.swift_dataset_path).exists())
            self.assertTrue(Path(artifacts.swift_length_audit_path).exists())
            self.assertTrue(Path(artifacts.swift_proxy_length_audit_path).exists())
            self.assertTrue(Path(artifacts.resolved_remote_surfaces_path).exists())
            self.assertEqual(artifacts.local_validation["record_count"], 2)
            self.assertIsInstance(artifacts.swift_runtime_check["available"], bool)
            self.assertIn("p95", artifacts.swift_length_audit_summary)
            self.assertIn("top_offenders", artifacts.swift_length_audit_summary)
            self.assertIn("audit_mode", artifacts.swift_length_audit_summary)
            self.assertIn("length_audit_backend", artifacts.swift_length_audit_summary)
            self.assertIn("4096", artifacts.swift_filtered_manifests)
            self.assertIn("8192", artifacts.swift_filtered_manifests)
            self.assertTrue(Path(artifacts.swift_filtered_manifests["4096"]).exists())
            self.assertTrue(Path(artifacts.swift_filtered_manifests["8192"]).exists())
            self.assertIn("strict_true_length_audit_passed", artifacts.resolved_remote_surfaces_summary)
            self.assertIn("threshold_clean_basis", artifacts.resolved_remote_surfaces_summary)
            swift_manifest = json.loads(Path(artifacts.swift_manifest_path).read_text(encoding="utf-8"))
            self.assertIn("filtered_export_summary_by_threshold", swift_manifest)

            true_audit_rows = {
                row["id"]: row["encoded_length"]
                for row in artifacts.swift_length_audit_summary["lengths"]
            }
            total_count = len(true_audit_rows)
            for threshold in (4096, 8192):
                threshold_key = str(threshold)
                filtered_manifest = json.loads(
                    Path(artifacts.swift_filtered_manifests[threshold_key]).read_text(encoding="utf-8")
                )
                self.assertEqual(filtered_manifest["threshold"], threshold)
                self.assertTrue(
                    all(
                        key in filtered_manifest
                        for key in [
                            "source_swift_dataset_path",
                            "source_true_audit_path",
                            "kept_count",
                            "dropped_count",
                            "dropped_ratio",
                            "threshold_clean_basis",
                            "true_threshold_clean_certified",
                            "true_multimodal_encode",
                            "length_audit_backend",
                            "top_dropped_offenders",
                            "max_kept_encoded_length",
                            "min_dropped_encoded_length",
                        ]
                    )
                )
                filtered_dataset_path = Path(filtered_manifest["kept_dataset_path"])
                self.assertTrue(filtered_dataset_path.exists())
                filtered_records = [
                    json.loads(line)
                    for line in filtered_dataset_path.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
                self.assertTrue(
                    all(true_audit_rows[record["id"]] <= threshold for record in filtered_records)
                )
                self.assertAlmostEqual(
                    filtered_manifest["dropped_ratio"],
                    filtered_manifest["dropped_count"] / total_count,
                )
                if filtered_records:
                    self.assertLessEqual(filtered_manifest["max_kept_encoded_length"], threshold)
                if filtered_manifest["dropped_count"] > 0:
                    self.assertGreater(filtered_manifest["min_dropped_encoded_length"], threshold)
                    offender_lengths = [row["encoded_length"] for row in filtered_manifest["top_dropped_offenders"]]
                    self.assertTrue(all(length > threshold for length in offender_lengths))
                    self.assertEqual(offender_lengths, sorted(offender_lengths, reverse=True))
                else:
                    self.assertIsNone(filtered_manifest["min_dropped_encoded_length"])
                    self.assertEqual(filtered_manifest["top_dropped_offenders"], [])

                main_summary = swift_manifest["filtered_export_summary_by_threshold"][threshold_key]
                self.assertEqual(main_summary["threshold"], threshold)
                self.assertEqual(main_summary["kept_count"], filtered_manifest["kept_count"])
                self.assertEqual(main_summary["dropped_count"], filtered_manifest["dropped_count"])
                self.assertAlmostEqual(main_summary["dropped_ratio"], filtered_manifest["dropped_ratio"])
                if artifacts.swift_length_audit_summary["true_multimodal_encode"]:
                    self.assertTrue(filtered_manifest["true_threshold_clean_certified"])
                    self.assertEqual(filtered_manifest["threshold_clean_basis"], "true_multimodal_encode")
                else:
                    self.assertFalse(filtered_manifest["true_threshold_clean_certified"])
                    self.assertEqual(filtered_manifest["threshold_clean_basis"], "fallback_derived_not_true_certified")

    def test_strict_true_length_audit_requires_real_encoder(self) -> None:
        """Strict mode must fail when true multimodal encode is unavailable."""

        with tempfile.TemporaryDirectory() as tempdir:
            try:
                artifacts = run_prompt_1_5_export(
                    export_config_path=EXPORT_CONFIG,
                    swift_recipe_path=SWIFT_RECIPE,
                    dataset_root=FIXTURE_ROOT,
                    output_root=tempdir,
                    max_samples_per_mode=1,
                    strict_true_length_audit=True,
                )
            except RuntimeError as exc:
                self.assertIn("strict mode", str(exc))
                return

            self.assertTrue(artifacts.swift_length_audit_summary["true_multimodal_encode"])
            self.assertTrue(artifacts.resolved_remote_surfaces_summary["strict_true_length_audit_passed"])


if __name__ == "__main__":
    unittest.main()
