"""Prompt 2.6 tests for tool-first intervention ablation under valid pz_cr contract."""

from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from agentiad_recon.backends import BackendRequest, BackendResponse, InferenceBackend
from agentiad_recon.baseline import (
    _artifact_dirs,
    _build_parser,
    _prompt_audit_payload,
    _runtime_config,
    _tool_loop_sample,
    load_run_definition,
    run_tool_augmented,
)
from agentiad_recon.mmad import MMADIndexer
from agentiad_recon.prompting import build_prompt, render_answer_block


FIXTURE_ROOT = REPO_ROOT / "tests" / "fixtures" / "mmad_fixture"
PZ_CR_CONFIG = REPO_ROOT / "configs" / "tool_pz_cr_fixture.json"


class DirectFinalAnswerBackend(InferenceBackend):
    """Backend stub that always collapses directly to a final answer."""

    backend_name = "prompt_2_6_direct_final_backend_v1"

    def describe_runtime(self) -> dict[str, object]:
        return {"adapter_loaded": False}

    def generate(self, request: BackendRequest, *, sample: dict[str, object]) -> BackendResponse:
        return BackendResponse(
            backend_name=self.backend_name,
            raw_output=render_answer_block(
                {
                    "anomaly_present": False,
                    "top_anomaly": None,
                    "visual_descriptions": [],
                },
                wrapper_tag="answer",
                think="Prompt 2.6 fixture direct terminal answer.",
            ),
            metadata={},
        )


class Prompt26ToolFirstInterventionTests(unittest.TestCase):
    """Prompt 2.6 regression tests."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.definition = load_run_definition(PZ_CR_CONFIG)
        cls.samples = [sample.to_dict() for sample in MMADIndexer(FIXTURE_ROOT, source="mmad_fixture").index_samples()]
        cls.sample = cls.samples[0]

    def _runtime(self, strategy: str) -> dict[str, object]:
        return _runtime_config(
            self.definition,
            dataset_root=FIXTURE_ROOT,
            runtime_overrides={"tool_first_intervention_strategy": strategy},
        )

    def test_strategy_selection_supports_config_and_runtime_override(self) -> None:
        """Config-provided strategy should load, and runtime override should win."""

        definition = copy.deepcopy(self.definition)
        definition["runtime"] = {"tool_first_intervention_strategy": "tool_first_nudge"}

        with tempfile.TemporaryDirectory() as tempdir:
            config_path = Path(tempdir) / "tool_prompt_2_6_config.json"
            config_path.write_text(json.dumps(definition, indent=2, sort_keys=True), encoding="utf-8")
            loaded = load_run_definition(config_path)

        config_runtime = _runtime_config(loaded, dataset_root=FIXTURE_ROOT, runtime_overrides=None)
        override_runtime = _runtime_config(
            loaded,
            dataset_root=FIXTURE_ROOT,
            runtime_overrides={"tool_first_intervention_strategy": "tool_first_strict"},
        )
        cli_args = _build_parser().parse_args(
            [
                "--config",
                str(PZ_CR_CONFIG),
                "--tool-first-intervention-strategy",
                "tool_first_strict",
                "--dry-run",
            ]
        )

        self.assertEqual(config_runtime["tool_first_intervention_strategy"], "tool_first_nudge")
        self.assertEqual(override_runtime["tool_first_intervention_strategy"], "tool_first_strict")
        self.assertEqual(cli_args.tool_first_intervention_strategy, "tool_first_strict")

    def test_baseline_strategy_keeps_original_prompt_surface(self) -> None:
        """The baseline intervention should remain backward-compatible with the current prompt."""

        legacy = build_prompt(self.sample, tool_path="pz_cr")
        explicit_baseline = build_prompt(
            self.sample,
            tool_path="pz_cr",
            tool_first_intervention_strategy="baseline",
        )

        self.assertEqual(legacy.messages, explicit_baseline.messages)
        self.assertEqual(legacy.stop_sequences, explicit_baseline.stop_sequences)

    def test_pz_cr_first_turn_prompt_surface_differs_across_strategies(self) -> None:
        """baseline/nudge/strict should render observably different first-turn prompts."""

        baseline_bundle = build_prompt(self.sample, tool_path="pz_cr", tool_first_intervention_strategy="baseline")
        nudge_bundle = build_prompt(
            self.sample,
            tool_path="pz_cr",
            tool_first_intervention_strategy="tool_first_nudge",
        )
        strict_bundle = build_prompt(
            self.sample,
            tool_path="pz_cr",
            tool_first_intervention_strategy="tool_first_strict",
        )

        baseline_audit = _prompt_audit_payload(
            baseline_bundle,
            sample_id=self.sample["sample_id"],
            seed=0,
            turn_index=0,
            runtime_tool_mode="pz_cr",
            tool_first_intervention_strategy="baseline",
        )
        nudge_audit = _prompt_audit_payload(
            nudge_bundle,
            sample_id=self.sample["sample_id"],
            seed=0,
            turn_index=0,
            runtime_tool_mode="pz_cr",
            tool_first_intervention_strategy="tool_first_nudge",
        )
        strict_audit = _prompt_audit_payload(
            strict_bundle,
            sample_id=self.sample["sample_id"],
            seed=0,
            turn_index=0,
            runtime_tool_mode="pz_cr",
            tool_first_intervention_strategy="tool_first_strict",
        )

        self.assertFalse(baseline_audit["intervention_text_applied"])
        self.assertTrue(nudge_audit["intervention_text_applied"])
        self.assertTrue(strict_audit["intervention_text_applied"])
        self.assertEqual(baseline_audit["tool_first_contract_strength"], "baseline")
        self.assertEqual(nudge_audit["tool_first_contract_strength"], "nudge")
        self.assertEqual(strict_audit["tool_first_contract_strength"], "strict")
        self.assertIn("tool_first_nudge", nudge_audit["user_text"])
        self.assertIn("tool_first_strict", strict_audit["user_text"])
        self.assertNotEqual(baseline_audit["prompt_surface_digest"], nudge_audit["prompt_surface_digest"])
        self.assertNotEqual(nudge_audit["prompt_surface_digest"], strict_audit["prompt_surface_digest"])

    def test_prompt_audit_sidecar_records_strategy_fields_and_runtime_provenance(self) -> None:
        """The real runtime path should persist strategy fields in prompt-audit sidecars."""

        with tempfile.TemporaryDirectory() as tempdir:
            directories = _artifact_dirs(self.definition, Path(tempdir))
            record, _trace_payload = _tool_loop_sample(
                definition=self.definition,
                backend=DirectFinalAnswerBackend(),
                runtime_config=self._runtime("tool_first_strict"),
                sample=self.sample,
                sample_pool=self.samples,
                seed=0,
                directories=directories,
            )
            prompt_audit_path = Path(record["metadata"]["prompt_audit_path"])
            prompt_audit = json.loads(prompt_audit_path.read_text(encoding="utf-8"))

        self.assertEqual(prompt_audit["tool_first_intervention_strategy"], "tool_first_strict")
        self.assertTrue(prompt_audit["intervention_text_applied"])
        self.assertEqual(prompt_audit["tool_first_contract_strength"], "strict")
        self.assertTrue(prompt_audit["prompt_surface_contains_tool_first_marker"])
        self.assertEqual(
            record["metadata"]["runtime_provenance"]["tool_first_intervention_strategy"],
            "tool_first_strict",
        )

    def test_no_strategy_path_fabricates_tool_calls(self) -> None:
        """Prompt interventions must not fabricate tool calls when the backend emits a final answer."""

        for strategy in ("baseline", "tool_first_nudge", "tool_first_strict"):
            with self.subTest(strategy=strategy), tempfile.TemporaryDirectory() as tempdir:
                directories = _artifact_dirs(self.definition, Path(tempdir))
                record, _trace_payload = _tool_loop_sample(
                    definition=self.definition,
                    backend=DirectFinalAnswerBackend(),
                    runtime_config=self._runtime(strategy),
                    sample=self.sample,
                    sample_pool=self.samples,
                    seed=0,
                    directories=directories,
                )

            self.assertEqual(record["first_protocol_event_type"], "final_answer")
            self.assertEqual(record["tool_call_count"], 0)
            self.assertEqual(record["called_tools"], [])
            self.assertTrue(record["terminal_without_tool_call"])

    def test_run_exports_strategy_provenance_and_strategy_summary_artifact(self) -> None:
        """Unified tool runs should export strategy provenance plus the per-strategy summary artifact."""

        with tempfile.TemporaryDirectory() as tempdir, mock.patch(
            "agentiad_recon.baseline._select_backend",
            return_value=DirectFinalAnswerBackend(),
        ):
            result = run_tool_augmented(
                config_path=PZ_CR_CONFIG,
                dataset_root=FIXTURE_ROOT,
                artifact_root=tempdir,
                max_samples=1,
                runtime_overrides={"tool_first_intervention_strategy": "tool_first_nudge"},
            )
            summary = json.loads(Path(result["summary_path"]).read_text(encoding="utf-8"))
            manifest = json.loads(Path(result["run_manifest_path"]).read_text(encoding="utf-8"))
            metrics = result["metrics_report"]
            strategy_summary_path = Path(summary["artifact_paths"]["tool_first_strategy_summary"])
            self.assertTrue(strategy_summary_path.exists())
            strategy_summary = json.loads(strategy_summary_path.read_text(encoding="utf-8"))

        for payload in (summary, manifest, metrics):
            self.assertEqual(payload["tool_first_intervention_strategy"], "tool_first_nudge")
        self.assertEqual(summary["runtime_provenance"]["tool_first_intervention_strategy"], "tool_first_nudge")
        self.assertEqual(metrics["runtime_provenance"]["tool_first_intervention_strategy"], "tool_first_nudge")
        self.assertEqual(manifest["run_provenance"]["tool_first_intervention_strategy"], "tool_first_nudge")
        self.assertEqual(strategy_summary["tool_first_intervention_strategy"], "tool_first_nudge")
        self.assertEqual(
            strategy_summary["zero_tool_behavior_summary"]["zero_tool_terminal_count"],
            summary["zero_tool_behavior_summary"]["zero_tool_terminal_count"],
        )


if __name__ == "__main__":
    unittest.main()
