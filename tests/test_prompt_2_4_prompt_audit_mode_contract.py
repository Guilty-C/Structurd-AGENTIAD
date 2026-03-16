"""Prompt 2.4 tests for pz_cr prompt audit and fail-fast mode-contract checks."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from agentiad_recon.backends import BackendRequest, BackendResponse, InferenceBackend, MockToolAwareBackend
from agentiad_recon.baseline import (
    _artifact_dirs,
    _prompt_audit_payload,
    _runtime_config,
    _tool_loop_sample,
    load_run_definition,
    run_baseline,
    run_tool_augmented,
)
from agentiad_recon.mmad import MMADIndexer
from agentiad_recon.prompting import PromptBundle, build_prompt


FIXTURE_ROOT = REPO_ROOT / "tests" / "fixtures" / "mmad_fixture"
PZ_CR_CONFIG = REPO_ROOT / "configs" / "tool_pz_cr_fixture.json"
BASELINE_CONFIG = REPO_ROOT / "configs" / "baseline_non_tool_fixture.json"


class NeverCalledToolBackend(InferenceBackend):
    """Backend stub that must never be reached when prompt audit fail-fast works."""

    backend_name = "never_called_tool_backend_v1"

    def __init__(self) -> None:
        self.called = False

    def describe_runtime(self) -> dict[str, object]:
        return {"adapter_loaded": False}

    def generate(self, request: BackendRequest, *, sample: dict[str, object]) -> BackendResponse:
        self.called = True
        raise AssertionError("backend.generate should not run after prompt-audit fail-fast")


class Prompt24PromptAuditTests(unittest.TestCase):
    """Prompt 2.4 regression tests."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.samples = [sample.to_dict() for sample in MMADIndexer(FIXTURE_ROOT, source="mmad_fixture").index_samples()]
        cls.sample = cls.samples[0]

    def _leaked_pz_only_prompt(self, sample: dict[str, object]) -> PromptBundle:
        """Return a pz_only-rendered prompt while the runtime still thinks it is pz_cr."""

        leaked = build_prompt(sample, tool_path="pz_only")
        return PromptBundle(
            prompt_version=leaked.prompt_version,
            tool_path="pz_cr",
            messages=leaked.messages,
            stop_sequences=leaked.stop_sequences,
        )

    def test_pz_cr_prompt_audit_reports_cr_surface_when_prompt_is_correct(self) -> None:
        """The rendered pz_cr prompt should expose CR and avoid pz_only leakage."""

        prompt_bundle = build_prompt(self.sample, tool_path="pz_cr")
        audit_payload = _prompt_audit_payload(
            prompt_bundle,
            sample_id=self.sample["sample_id"],
            seed=0,
            turn_index=0,
            runtime_tool_mode="pz_cr",
            tool_first_intervention_strategy="baseline",
        )

        self.assertEqual(audit_payload["runtime_tool_mode"], "pz_cr")
        self.assertEqual(audit_payload["declared_available_tools"], ["PZ", "CR"])
        self.assertFalse(audit_payload["prompt_contains_pz_only"])
        self.assertTrue(audit_payload["cr_available_in_prompt_surface"])
        self.assertFalse(audit_payload["mode_contract_mismatch"])

    def test_pz_cr_prompt_audit_flags_pz_only_leakage_and_sets_failure_reason(self) -> None:
        """A leaked pz_only prompt in pz_cr runtime should fail fast with audit evidence."""

        definition = load_run_definition(PZ_CR_CONFIG)
        runtime_config = _runtime_config(definition, dataset_root=FIXTURE_ROOT, runtime_overrides=None)
        backend = NeverCalledToolBackend()

        with tempfile.TemporaryDirectory() as tempdir, mock.patch(
            "agentiad_recon.baseline.build_prompt",
            side_effect=lambda sample, tool_path, **_kwargs: self._leaked_pz_only_prompt(sample),
        ):
            directories = _artifact_dirs(definition, Path(tempdir))
            record, trace_payload = _tool_loop_sample(
                definition=definition,
                backend=backend,
                runtime_config=runtime_config,
                sample=self.sample,
                sample_pool=self.samples,
                seed=0,
                directories=directories,
            )
            prompt_audit_path = Path(record["metadata"]["prompt_audit_path"])
            prompt_audit = json.loads(prompt_audit_path.read_text(encoding="utf-8"))

        self.assertFalse(backend.called)
        self.assertEqual(record["failure_reason"], "mode_contract_pz_only_leakage")
        self.assertFalse(record["parser_valid"])
        self.assertFalse(record["schema_valid"])
        self.assertIsNone(record["prediction"])
        self.assertTrue(record["metadata"]["prompt_audit_mismatch"])
        self.assertIn("mode_contract_pz_only_leakage", record["metadata"]["prompt_audit_mismatch_reasons"])
        self.assertIn("mode_contract_missing_cr_tool", record["metadata"]["prompt_audit_mismatch_reasons"])
        self.assertTrue(prompt_audit["mode_contract_mismatch"])
        self.assertTrue(prompt_audit["prompt_contains_pz_only"])
        self.assertFalse(prompt_audit["cr_available_in_prompt_surface"])
        self.assertEqual(trace_payload["metadata"]["prompt_audit_path"], str(prompt_audit_path.resolve()))

    def test_failed_baseline_predictions_always_have_non_empty_failure_reason(self) -> None:
        """Any failed prediction record should now carry a concrete failure reason."""

        with tempfile.TemporaryDirectory() as tempdir:
            result = run_baseline(
                config_path=BASELINE_CONFIG,
                dataset_root=FIXTURE_ROOT,
                artifact_root=tempdir,
                max_samples=2,
            )

        failed_records = [
            record
            for record in result["prediction_records"]
            if record["prediction"] is None or not record["parser_valid"] or not record["schema_valid"]
        ]
        self.assertTrue(failed_records)
        self.assertTrue(all(record["failure_reason"] for record in failed_records))
        self.assertTrue(all(record["failure_reason"].startswith("parser_invalid:") for record in failed_records))

    def test_run_tool_augmented_aggregates_prompt_audit_summary(self) -> None:
        """Run artifacts should aggregate prompt-audit mismatch counts and zero missing reasons."""

        with tempfile.TemporaryDirectory() as tempdir, mock.patch(
            "agentiad_recon.baseline.build_prompt",
            side_effect=lambda sample, tool_path, **_kwargs: self._leaked_pz_only_prompt(sample),
        ), mock.patch.object(
            MockToolAwareBackend,
            "generate",
            autospec=True,
            side_effect=AssertionError("backend.generate should not run after prompt-audit fail-fast"),
        ):
            result = run_tool_augmented(
                config_path=PZ_CR_CONFIG,
                dataset_root=FIXTURE_ROOT,
                artifact_root=tempdir,
                max_samples=1,
            )
            summary = json.loads(Path(result["summary_path"]).read_text(encoding="utf-8"))
            manifest = json.loads(Path(result["run_manifest_path"]).read_text(encoding="utf-8"))
            metrics = result["metrics_report"]

        for payload in (summary, manifest, metrics):
            self.assertGreater(payload["prompt_audit_summary"]["prompt_audit_event_count"], 0)
            self.assertGreater(payload["prompt_audit_summary"]["samples_with_prompt_audit_mismatch"], 0)
            self.assertGreater(payload["prompt_audit_summary"]["mode_contract_pz_only_leakage_count"], 0)
            self.assertGreater(payload["prompt_audit_summary"]["mode_contract_missing_cr_tool_count"], 0)
            self.assertEqual(payload["prompt_audit_summary"]["failed_count_with_missing_reason_count"], 0)


if __name__ == "__main__":
    unittest.main()
