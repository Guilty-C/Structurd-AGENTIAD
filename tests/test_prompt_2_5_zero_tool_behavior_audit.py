"""Prompt 2.5 tests for zero-tool behavior auditing."""

from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from agentiad_recon.backends import BackendRequest, BackendResponse, InferenceBackend, MockToolAwareBackend
from agentiad_recon.baseline import _artifact_dirs, _runtime_config, _tool_loop_sample, load_run_definition
from agentiad_recon.behavior_audit import (
    audit_train_side_dataset,
    summarize_zero_tool_behavior,
    write_zero_tool_behavior_sidecars,
)
from agentiad_recon.mmad import MMADIndexer
from agentiad_recon.prompting import render_answer_block
from agentiad_recon.sft import export_sft_dataset


FIXTURE_ROOT = REPO_ROOT / "tests" / "fixtures" / "mmad_fixture"
PZ_CR_CONFIG = REPO_ROOT / "configs" / "tool_pz_cr_fixture.json"
EXPORT_CONFIG = REPO_ROOT / "configs" / "sft_export_fixture.json"


class DirectFinalAnswerBackend(InferenceBackend):
    """Backend that immediately returns a terminal answer without tools."""

    backend_name = "direct_final_answer_backend_v1"

    def describe_runtime(self) -> dict[str, object]:
        return {"adapter_loaded": False}

    def generate(self, request: BackendRequest, *, sample: dict[str, object]) -> BackendResponse:
        raw_output = render_answer_block(
            {
                "anomaly_present": False,
                "top_anomaly": None,
                "visual_descriptions": [],
            },
            wrapper_tag="answer",
            think="Direct final answer without any tool call.",
        )
        return BackendResponse(backend_name=self.backend_name, raw_output=raw_output, metadata={})


class Prompt25ZeroToolBehaviorTests(unittest.TestCase):
    """Prompt 2.5 regression tests."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.samples = [sample.to_dict() for sample in MMADIndexer(FIXTURE_ROOT, source="mmad_fixture").index_samples()]
        cls.sample = cls.samples[0]

    def test_direct_final_answer_record_carries_zero_tool_audit_fields(self) -> None:
        """A direct terminal answer under pz_cr should be marked as zero-tool terminal behavior."""

        definition = load_run_definition(PZ_CR_CONFIG)
        runtime_config = _runtime_config(definition, dataset_root=FIXTURE_ROOT, runtime_overrides=None)

        with tempfile.TemporaryDirectory() as tempdir:
            directories = _artifact_dirs(definition, Path(tempdir))
            record, _trace_payload = _tool_loop_sample(
                definition=definition,
                backend=DirectFinalAnswerBackend(),
                runtime_config=runtime_config,
                sample=self.sample,
                sample_pool=self.samples,
                seed=0,
                directories=directories,
            )

        self.assertEqual(record["first_protocol_event_type"], "final_answer")
        self.assertTrue(record["first_assistant_output_terminal"])
        self.assertEqual(record["tool_call_count"], 0)
        self.assertEqual(record["called_tools"], [])
        self.assertTrue(record["terminal_without_tool_call"])
        self.assertTrue(record["terminal_false_null_without_tool_call"])
        self.assertEqual(record["terminal_answer_turn_index"], 0)

    def test_tool_first_record_carries_non_collapse_audit_fields(self) -> None:
        """A tool-first pz_cr trace should not be marked as zero-tool collapse."""

        definition = load_run_definition(PZ_CR_CONFIG)
        runtime_config = _runtime_config(definition, dataset_root=FIXTURE_ROOT, runtime_overrides=None)
        backend = MockToolAwareBackend(
            backend_name="mock_tool_scripted_pz_cr_v1",
            policy="fixture_scripted_pz_cr_v1",
        )

        with tempfile.TemporaryDirectory() as tempdir:
            directories = _artifact_dirs(definition, Path(tempdir))
            record, _trace_payload = _tool_loop_sample(
                definition=definition,
                backend=backend,
                runtime_config=runtime_config,
                sample=self.sample,
                sample_pool=self.samples,
                seed=0,
                directories=directories,
            )

        self.assertEqual(record["first_protocol_event_type"], "tool_call")
        self.assertFalse(record["first_assistant_output_terminal"])
        self.assertEqual(record["tool_call_count"], 2)
        self.assertEqual(record["called_tools"], ["PZ", "CR"])
        self.assertFalse(record["terminal_without_tool_call"])
        self.assertFalse(record["terminal_false_null_without_tool_call"])
        self.assertEqual(record["terminal_answer_turn_index"], 2)

    def test_zero_tool_behavior_summary_and_sidecars_are_written(self) -> None:
        """Aggregate zero-tool summaries and sidecars should be deterministic."""

        records = [
            {
                "sample_id": "sample-direct",
                "category": "capsule",
                "prediction": {"anomaly_present": False, "top_anomaly": None, "visual_descriptions": []},
                "parser_valid": True,
                "schema_valid": True,
                "failure_reason": None,
                "first_protocol_event_type": "final_answer",
                "first_assistant_output_terminal": True,
                "tool_call_count": 0,
                "called_tools": [],
                "terminal_without_tool_call": True,
                "terminal_false_null_without_tool_call": True,
                "terminal_answer_turn_index": 0,
                "metadata": {"sample_source_kind": "canonical"},
            },
            {
                "sample_id": "sample-tool",
                "category": "capsule",
                "prediction": {"anomaly_present": True, "top_anomaly": "crack", "visual_descriptions": ["x"]},
                "parser_valid": True,
                "schema_valid": True,
                "failure_reason": None,
                "first_protocol_event_type": "tool_call",
                "first_assistant_output_terminal": False,
                "tool_call_count": 2,
                "called_tools": ["PZ", "CR"],
                "terminal_without_tool_call": False,
                "terminal_false_null_without_tool_call": False,
                "terminal_answer_turn_index": 2,
                "metadata": {"sample_source_kind": "canonical"},
            },
        ]

        summary = summarize_zero_tool_behavior(records)
        self.assertEqual(summary["turn0_direct_final_answer_count"], 1)
        self.assertEqual(summary["turn0_tool_call_count"], 1)
        self.assertEqual(summary["samples_with_any_tool_call"], 1)
        self.assertEqual(summary["zero_tool_terminal_count"], 1)
        self.assertEqual(summary["zero_tool_terminal_false_null_count"], 1)
        self.assertEqual(summary["failed_count_with_missing_reason_count"], 0)

        with tempfile.TemporaryDirectory() as tempdir:
            sidecars = write_zero_tool_behavior_sidecars(
                prediction_records=records,
                metrics_dir=Path(tempdir),
            )
            dataset_payload = json.loads(Path(sidecars["per_dataset_zero_tool_behavior"]).read_text(encoding="utf-8"))
            category_payload = json.loads(Path(sidecars["per_category_zero_tool_behavior"]).read_text(encoding="utf-8"))

        self.assertIn("canonical", dataset_payload["groups"])
        self.assertEqual(dataset_payload["groups"]["canonical"]["zero_tool_terminal_count"], 1)
        self.assertIn("capsule", category_payload["groups"])
        self.assertEqual(category_payload["groups"]["capsule"]["turn0_tool_call_count"], 1)

    def test_train_side_audit_distinguishes_tool_first_vs_direct_final_pz_cr_rows(self) -> None:
        """The read-only train-side audit should quantify tool-first supervision strength."""

        with tempfile.TemporaryDirectory() as tempdir:
            records, _metadata = export_sft_dataset(
                config_path=EXPORT_CONFIG,
                dataset_root=FIXTURE_ROOT,
                output_root=tempdir,
                max_samples_per_mode=1,
            )
            pz_cr_record = next(record for record in records if record["trajectory_mode"] == "pz_cr")
            direct_final = copy.deepcopy(pz_cr_record)
            direct_final["trajectory_id"] = direct_final["trajectory_id"] + ":direct"
            direct_final["sample_id"] = direct_final["sample_id"] + ":direct"
            direct_final["tool_events"] = []
            direct_final["final_answer"] = {
                "anomaly_present": False,
                "top_anomaly": None,
                "visual_descriptions": [],
            }
            direct_final["messages"] = [
                message
                for message in direct_final["messages"]
                if message["message_type"] not in {"tool_request", "tool_result"}
            ]
            direct_final["messages"][-1]["content"] = render_answer_block(
                {
                    "anomaly_present": False,
                    "top_anomaly": None,
                    "visual_descriptions": [],
                },
                wrapper_tag="answer",
                think="Collapsed direct terminal answer.",
            )
            dataset_path = Path(tempdir) / "prompt_2_5_train_audit.jsonl"
            dataset_path.write_text(
                "\n".join(json.dumps(record, sort_keys=True) for record in [pz_cr_record, direct_final]) + "\n",
                encoding="utf-8",
            )

            summary = audit_train_side_dataset(dataset_path, dataset_format="canonical")

        self.assertEqual(summary["pz_cr_record_count"], 2)
        self.assertEqual(summary["first_assistant_tool_call_count"], 1)
        self.assertEqual(summary["first_assistant_final_answer_count"], 1)
        self.assertAlmostEqual(summary["tool_first_ratio"], 0.5)
        self.assertAlmostEqual(summary["zero_tool_terminal_ratio"], 0.5)
        self.assertAlmostEqual(summary["terminal_false_null_without_tool_ratio"], 0.5)


if __name__ == "__main__":
    unittest.main()
