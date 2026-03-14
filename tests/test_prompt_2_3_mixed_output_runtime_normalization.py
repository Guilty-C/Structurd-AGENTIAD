"""Prompt 2.3 tests for mixed tool-call/final-answer runtime normalization."""

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
    _runtime_config,
    _tool_loop_sample,
    load_run_definition,
    run_tool_augmented,
)
from agentiad_recon.mmad import MMADIndexer
from agentiad_recon.tooling import normalize_protocol_turn, parse_tool_call, protocol_event


FIXTURE_ROOT = REPO_ROOT / "tests" / "fixtures" / "mmad_fixture"
PZ_CR_CONFIG = REPO_ROOT / "configs" / "tool_pz_cr_fixture.json"


class MixedOutputBackend(InferenceBackend):
    """Scripted backend that emits mixed output once, then a clean final answer."""

    backend_name = "mixed_output_mock_backend_v1"

    def describe_runtime(self) -> dict[str, object]:
        return {"adapter_loaded": False}

    def generate(self, request: BackendRequest, *, sample: dict[str, object]) -> BackendResponse:
        anomaly_label = sample["ground_truth"]["top_anomaly"]
        tool_results = [
            message
            for message in request.messages
            if message.get("message_type") == "tool_result" and message.get("tool_name") is not None
        ]
        if not tool_results:
            raw_output = (
                "<think>\nNeed a crop before answering.\n</think>\n"
                "<tool_call>\n"
                "{\"call_id\":\"mix-pz-1\",\"tool_name\":\"PZ\",\"arguments\":{\"bbox\":{\"x0\":0.2,\"y0\":0.2,\"x1\":0.8,\"y1\":0.8}}}\n"
                "</tool_call>\n"
                "<answer>\n"
                "  <anomaly_present>true</anomaly_present>\n"
                f"  <top_anomaly>{anomaly_label}</top_anomaly>\n"
                "  <visual_descriptions>\n"
                "    <description>premature answer that should be discarded this turn</description>\n"
                "  </visual_descriptions>\n"
                "</answer>"
            )
        else:
            raw_output = (
                "<think>\nThe crop is available; finalize now.\n</think>\n"
                "<answer>\n"
                "  <anomaly_present>true</anomaly_present>\n"
                f"  <top_anomaly>{anomaly_label}</top_anomaly>\n"
                "  <visual_descriptions>\n"
                "    <description>The crop confirms a dent on the bottle.</description>\n"
                "  </visual_descriptions>\n"
                "</answer>"
            )
        return BackendResponse(
            backend_name=self.backend_name,
            raw_output=raw_output,
            metadata={"policy": "mixed_output_fixture"},
        )


class Prompt23MixedOutputNormalizationTests(unittest.TestCase):
    """Regression tests for Prompt 2.3 runtime normalization."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.samples = [sample.to_dict() for sample in MMADIndexer(FIXTURE_ROOT, source="mmad_fixture").index_samples()]
        cls.sample = next(sample for sample in cls.samples if sample["anomaly_present"])

    def test_pure_tool_and_final_paths_keep_old_semantics(self) -> None:
        """Pure tool-call and pure final-answer outputs should remain unchanged."""

        tool_output = (
            "<think>\nNeed a crop.\n</think>\n"
            "<tool_call>\n"
            "{\"tool_name\":\"PZ\",\"arguments\":{\"bbox\":{\"x0\":0.1,\"y0\":0.1,\"x1\":0.9,\"y1\":0.9}}}\n"
            "</tool_call>"
        )
        final_output = (
            "<think>\nReady.\n</think>\n"
            "<answer>\n"
            "  <anomaly_present>false</anomaly_present>\n"
            "  <top_anomaly>null</top_anomaly>\n"
            "  <visual_descriptions>\n"
            "  </visual_descriptions>\n"
            "</answer>"
        )

        self.assertEqual(protocol_event(tool_output), "tool_call")
        self.assertEqual(parse_tool_call(tool_output, tool_path="pz_cr").tool_name, "PZ")
        self.assertEqual(protocol_event(final_output), "final_answer")

    def test_mixed_output_normalizes_to_first_legal_tool_call_with_audit_payload(self) -> None:
        """Mixed output should pick the first legal tool call and retain audit evidence."""

        mixed_output = (
            "<think>\nNeed tools first.\n</think>\n"
            "<tool_call>\n"
            "{\"call_id\":\"first-pz\",\"tool_name\":\"PZ\",\"arguments\":{\"bbox\":{\"x0\":0.2,\"y0\":0.2,\"x1\":0.8,\"y1\":0.8}}}\n"
            "</tool_call>\n"
            "<tool_call>\n"
            "{\"call_id\":\"second-cr\",\"tool_name\":\"CR\",\"arguments\":{\"policy\":\"same_category_normal\"}}\n"
            "</tool_call>\n"
            "<answer>\n"
            "  <anomaly_present>true</anomaly_present>\n"
            "  <top_anomaly>dent</top_anomaly>\n"
            "  <visual_descriptions>\n"
            "    <description>discard me this turn</description>\n"
            "  </visual_descriptions>\n"
            "</answer>"
        )

        decision = normalize_protocol_turn(mixed_output, tool_path="pz_cr")
        audit_payload = decision.to_audit_payload(
            sample_id="sample-001",
            turn_index=0,
            raw_output_path="/tmp/sample-001/turn_0.txt",
        )

        self.assertEqual(decision.event_type, "tool_call")
        self.assertTrue(decision.normalization_applied)
        self.assertEqual(decision.parsed_call.call_id, "first-pz")
        self.assertEqual(decision.parsed_call.tool_name, "PZ")
        self.assertTrue(decision.discarded_final_answer_present)
        self.assertEqual(decision.reason, "mixed_tool_call_and_final_answer")
        self.assertEqual(audit_payload["selected_protocol_event_type"], "tool_call")
        self.assertEqual(audit_payload["selected_tool_name"], "PZ")
        self.assertEqual(audit_payload["additional_valid_tool_calls_discarded"], 1)
        self.assertTrue(audit_payload["discarded_final_answer_present"])
        self.assertEqual(audit_payload["raw_output"], mixed_output)

    def test_tool_loop_records_mixed_output_normalization_sidecar_and_trace(self) -> None:
        """The unified tool loop should keep running and export auditable normalization evidence."""

        definition = load_run_definition(PZ_CR_CONFIG)
        runtime_config = _runtime_config(definition, dataset_root=FIXTURE_ROOT, runtime_overrides=None)
        backend = MixedOutputBackend()

        with tempfile.TemporaryDirectory() as tempdir:
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

            normalization_events = record["metadata"]["normalization_events"]
            self.assertEqual(record["tool_usage"]["per_tool_counts"]["PZ"], 1)
            self.assertEqual(record["tool_usage"]["total_calls"], 1)
            self.assertTrue(record["parser_valid"])
            self.assertTrue(record["schema_valid"])
            self.assertEqual(len(normalization_events), 1)
            self.assertEqual(normalization_events[0]["reason"], "mixed_tool_call_and_final_answer")
            self.assertTrue(normalization_events[0]["discarded_final_answer_present"])

            sidecar_path = Path(record["metadata"]["normalization_event_paths"][0])
            sidecar_payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
            self.assertEqual(sidecar_payload["sample_id"], self.sample["sample_id"])
            self.assertEqual(sidecar_payload["selected_protocol_event_type"], "tool_call")
            self.assertEqual(sidecar_payload["selected_tool_name"], "PZ")
            self.assertTrue(sidecar_payload["discarded_final_answer_present"])
            self.assertIn("<answer>", sidecar_payload["raw_output"])

            self.assertEqual(trace_payload["metadata"]["normalization_event_count"], 1)
            self.assertEqual(trace_payload["metadata"]["normalization_event_paths"], [str(sidecar_path.resolve())])
            tool_request_messages = [
                message
                for message in trace_payload["messages"]
                if message["message_type"] == "tool_request"
            ]
            self.assertEqual(len(tool_request_messages), 1)
            self.assertTrue(tool_request_messages[0]["metadata"]["normalization_applied"])
            self.assertEqual(
                tool_request_messages[0]["metadata"]["normalization_reason"],
                "mixed_tool_call_and_final_answer",
            )

    def test_run_tool_augmented_exports_normalization_summary(self) -> None:
        """The single entrypoint should aggregate normalization counts into run artifacts."""

        original_generate = MockToolAwareBackend.generate

        def mixed_once(self: MockToolAwareBackend, request: BackendRequest, *, sample: dict[str, object]) -> BackendResponse:
            anomaly_label = sample["ground_truth"]["top_anomaly"]
            tool_results = [
                message
                for message in request.messages
                if message.get("message_type") == "tool_result" and message.get("tool_name") is not None
            ]
            if self.policy == "fixture_scripted_pz_cr_v1" and not tool_results:
                return BackendResponse(
                    backend_name=self.backend_name,
                    raw_output=(
                        "<think>\nNeed a crop before answering.\n</think>\n"
                        "<tool_call>\n"
                        "{\"call_id\":\"summary-pz\",\"tool_name\":\"PZ\",\"arguments\":{\"bbox\":{\"x0\":0.2,\"y0\":0.2,\"x1\":0.8,\"y1\":0.8}}}\n"
                        "</tool_call>\n"
                        "<answer>\n"
                        "  <anomaly_present>true</anomaly_present>\n"
                        f"  <top_anomaly>{anomaly_label}</top_anomaly>\n"
                        "  <visual_descriptions>\n"
                        "    <description>premature answer for summary coverage</description>\n"
                        "  </visual_descriptions>\n"
                        "</answer>"
                    ),
                    metadata={"policy": self.policy, "tool_mode": request.tool_mode},
                )
            return original_generate(self, request, sample=sample)

        with tempfile.TemporaryDirectory() as tempdir, mock.patch.object(
            MockToolAwareBackend,
            "generate",
            autospec=True,
            side_effect=mixed_once,
        ):
            result = run_tool_augmented(
                config_path=PZ_CR_CONFIG,
                dataset_root=FIXTURE_ROOT,
                artifact_root=tempdir,
                max_samples=1,
            )
            summary = json.loads(Path(result["summary_path"]).read_text(encoding="utf-8"))
            manifest = json.loads(Path(result["run_manifest_path"]).read_text(encoding="utf-8"))

        self.assertGreater(summary["normalization_summary"]["event_count"], 0)
        self.assertGreater(summary["normalization_summary"]["mixed_tool_call_and_final_answer_count"], 0)
        self.assertGreater(summary["normalization_summary"]["discarded_premature_final_answer_count"], 0)
        self.assertEqual(summary["normalization_summary"], manifest["normalization_summary"])


if __name__ == "__main__":
    unittest.main()
