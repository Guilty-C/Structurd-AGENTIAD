"""Prompt 2.8 tests for retry-only repair of recoverable first-turn gate outputs."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from agentiad_recon.backends import BackendRequest, BackendResponse, InferenceBackend
from agentiad_recon.baseline import _artifact_dirs, _runtime_config, _tool_loop_sample, load_run_definition, run_tool_augmented
from agentiad_recon.mmad import MMADIndexer
from agentiad_recon.prompting import render_answer_block
from agentiad_recon.tooling import ToolContractError, parse_tool_call, repair_retry_tool_call_output


FIXTURE_ROOT = REPO_ROOT / "tests" / "fixtures" / "mmad_fixture"
PZ_CR_CONFIG = REPO_ROOT / "configs" / "tool_pz_cr_fixture.json"


def _direct_final_answer(think: str) -> str:
    return render_answer_block(
        {"anomaly_present": False, "top_anomaly": None, "visual_descriptions": []},
        wrapper_tag="answer",
        think=think,
    )


WRAPPED_SMART_QUOTES = (
    '<tool_call>{“tool_name”:“PZ”,“arguments”:{“bbox”:{“x0”:0.10,“y0:0.10,“x1”:0.80,“y1”:0.80}}}</tool_call>'
)
BARE_PZ_JSON = '{"tool_name":"PZ","arguments":{"bbox":{"x0":0.10,"y0":0.10,"x1":0.80,"y1":0.80}}}'
BARE_ALIAS_JSON = '{"tool_name":"pz_cr","arguments":{"bbox":[0.10,0.10,0.80,0.80]}}'
PSEUDO_TOOL_JSON = '{"tool_name":"pz_cr","arguments":{"crop_result":{"status":"ok"},"query_result":{"status":"none"}}}'


class SequenceBackend(InferenceBackend):
    """Backend that returns a deterministic sequence of raw outputs."""

    backend_name = "prompt_2_8_sequence_backend_v1"

    def __init__(self, outputs: list[str]) -> None:
        self.outputs = outputs
        self.calls = 0

    def describe_runtime(self) -> dict[str, object]:
        return {"adapter_loaded": False}

    def generate(self, request: BackendRequest, *, sample: dict[str, object]) -> BackendResponse:
        index = min(self.calls, len(self.outputs) - 1)
        raw_output = self.outputs[index]
        self.calls += 1
        return BackendResponse(backend_name=self.backend_name, raw_output=raw_output, metadata={})


class Prompt28RetryRepairTests(unittest.TestCase):
    """Prompt 2.8 regression tests."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.definition = load_run_definition(PZ_CR_CONFIG)
        cls.samples = [sample.to_dict() for sample in MMADIndexer(FIXTURE_ROOT, source="mmad_fixture").index_samples()]
        cls.sample = cls.samples[0]

    def _runtime(self) -> dict[str, object]:
        return _runtime_config(
            self.definition,
            dataset_root=FIXTURE_ROOT,
            runtime_overrides={"first_turn_protocol_gate_mode": "retry_once_pz_cr"},
        )

    def test_wrapped_smart_quotes_repair_recovers_valid_pz(self) -> None:
        """Wrapped malformed JSON with smart quotes should repair into strict PZ syntax."""

        decision = repair_retry_tool_call_output(WRAPPED_SMART_QUOTES, tool_path="pz_cr")

        self.assertTrue(decision.attempted)
        self.assertTrue(decision.succeeded)
        self.assertEqual(decision.selected_tool_name, "PZ")
        self.assertTrue(decision.quote_normalization_applied)
        self.assertIn("<tool_call>", decision.repaired_text)
        self.assertEqual(decision.selected_canonical_arguments["bbox"]["y0"], 0.10)

    def test_bare_json_missing_wrapper_is_recovered(self) -> None:
        """A bare JSON object with explicit bbox crop intent should recover via wrapper repair."""

        decision = repair_retry_tool_call_output(BARE_PZ_JSON, tool_path="pz_cr")

        self.assertTrue(decision.succeeded)
        self.assertTrue(decision.wrapper_recovery_applied)
        self.assertIn("wrapper_recovery", decision.repair_categories)
        self.assertEqual(decision.selected_tool_name, "PZ")

    def test_alias_pz_cr_is_canonicalized_to_pz(self) -> None:
        """Mode-like aliases should canonicalize to the actual crop tool when bbox intent is explicit."""

        decision = repair_retry_tool_call_output(BARE_ALIAS_JSON, tool_path="pz_cr")

        self.assertTrue(decision.succeeded)
        self.assertTrue(decision.alias_canonicalization_applied)
        self.assertTrue(decision.bbox_canonicalization_applied)
        self.assertEqual(decision.selected_tool_name, "PZ")
        self.assertEqual(decision.selected_canonical_arguments["bbox"]["x1"], 0.80)

    def test_pseudo_tool_payload_still_fails(self) -> None:
        """Pseudo-tool state without an explicit crop contract must remain unrecoverable."""

        decision = repair_retry_tool_call_output(PSEUDO_TOOL_JSON, tool_path="pz_cr")

        self.assertTrue(decision.attempted)
        self.assertFalse(decision.succeeded)
        self.assertEqual(
            decision.failure_family,
            "retry_repair_unrecoverable_pseudo_observation_payload",
        )

    def test_normal_parser_path_remains_strict(self) -> None:
        """Prompt 2.8 must not relax strict parsing outside the retry-only gate path."""

        with self.assertRaises(ToolContractError):
            parse_tool_call(BARE_PZ_JSON, tool_path="pz_cr")

    def test_retry_gate_uses_repair_and_preserves_evidence(self) -> None:
        """Gate-on retry should recover a malformed crop call without fabricating output."""

        backend = SequenceBackend(
            [
                _direct_final_answer("turn 0 final answer"),
                BARE_ALIAS_JSON,
                _direct_final_answer("post-tool final answer"),
            ]
        )
        with tempfile.TemporaryDirectory() as tempdir:
            directories = _artifact_dirs(self.definition, Path(tempdir))
            record, trace_payload = _tool_loop_sample(
                definition=self.definition,
                backend=backend,
                runtime_config=self._runtime(),
                sample=self.sample,
                sample_pool=self.samples,
                seed=0,
                directories=directories,
            )
            sidecar = json.loads(Path(record["first_turn_gate_sidecar_path"]).read_text(encoding="utf-8"))

        self.assertEqual(record["first_turn_gate_outcome"], "recovered_to_tool_call")
        self.assertTrue(record["first_turn_gate_repair_attempted"])
        self.assertTrue(record["first_turn_gate_repair_succeeded"])
        self.assertEqual(record["first_turn_gate_repair_outcome"], "repaired_to_tool_call")
        self.assertIn("wrapper_recovery", record["first_turn_gate_repair_categories"])
        self.assertIn("alias_canonicalization", record["first_turn_gate_repair_categories"])
        self.assertIn("bbox_canonicalization", record["first_turn_gate_repair_categories"])
        self.assertEqual(record["called_tools"], ["PZ"])
        self.assertEqual(sidecar["retry_repair"]["original_text"], BARE_ALIAS_JSON)
        self.assertIn("<tool_call>", sidecar["retry_repair"]["repaired_text"])
        self.assertEqual(trace_payload["metadata"]["first_turn_gate_repair_outcome"], "repaired_to_tool_call")

    def test_retry_gate_unrecoverable_case_keeps_concrete_failure_reason(self) -> None:
        """Unrecoverable malformed retry output should remain an auditable failure."""

        backend = SequenceBackend(
            [
                _direct_final_answer("turn 0 final answer"),
                PSEUDO_TOOL_JSON,
            ]
        )
        with tempfile.TemporaryDirectory() as tempdir:
            directories = _artifact_dirs(self.definition, Path(tempdir))
            record, _trace_payload = _tool_loop_sample(
                definition=self.definition,
                backend=backend,
                runtime_config=self._runtime(),
                sample=self.sample,
                sample_pool=self.samples,
                seed=0,
                directories=directories,
            )
            sidecar = json.loads(Path(record["first_turn_gate_sidecar_path"]).read_text(encoding="utf-8"))

        self.assertEqual(record["first_turn_gate_outcome"], "retry_parse_failure")
        self.assertTrue(record["first_turn_gate_repair_attempted"])
        self.assertFalse(record["first_turn_gate_repair_succeeded"])
        self.assertEqual(record["first_turn_gate_repair_outcome"], "repair_failed")
        self.assertEqual(record["tool_call_count"], 0)
        self.assertEqual(
            record["failure_reason"],
            "runtime_exception:retry_repair_unrecoverable_pseudo_observation_payload",
        )
        self.assertEqual(
            record["first_turn_gate_repair_failure_family"],
            "retry_repair_unrecoverable_pseudo_observation_payload",
        )
        self.assertEqual(sidecar["retry_repair"]["original_text"], PSEUDO_TOOL_JSON)
        self.assertIsNone(sidecar["retry_repair"]["repaired_text"])
        self.assertEqual(
            sidecar["retry_repair"]["failure_family"],
            "retry_repair_unrecoverable_pseudo_observation_payload",
        )

    def test_run_outputs_include_repair_aggregates(self) -> None:
        """Summary, metrics, and manifest should expose repair aggregate counters."""

        backend = SequenceBackend(
            [
                _direct_final_answer("turn 0 final answer"),
                BARE_ALIAS_JSON,
                _direct_final_answer("post-tool final answer"),
            ]
        )
        with tempfile.TemporaryDirectory() as tempdir, mock.patch(
            "agentiad_recon.baseline._select_backend",
            return_value=backend,
        ):
            result = run_tool_augmented(
                config_path=PZ_CR_CONFIG,
                dataset_root=FIXTURE_ROOT,
                artifact_root=tempdir,
                max_samples=1,
                runtime_overrides={"first_turn_protocol_gate_mode": "retry_once_pz_cr"},
            )
            summary = json.loads(Path(result["summary_path"]).read_text(encoding="utf-8"))
            manifest = json.loads(Path(result["run_manifest_path"]).read_text(encoding="utf-8"))
            metrics = result["metrics_report"]

        for payload in (summary, manifest, metrics):
            repair = payload["first_turn_gate_repair_summary"]
            self.assertEqual(repair["first_turn_gate_repair_attempt_count"], 1)
            self.assertEqual(repair["first_turn_gate_repair_success_count"], 1)
            self.assertEqual(repair["first_turn_gate_repair_failure_count"], 0)
            self.assertEqual(repair["first_turn_gate_repair_wrapper_recovery_count"], 1)
            self.assertEqual(repair["first_turn_gate_repair_duplicate_candidate_deduplication_count"], 0)
            self.assertEqual(repair["first_turn_gate_repair_alias_canonicalization_count"], 1)
            self.assertEqual(repair["first_turn_gate_repair_bbox_canonicalization_count"], 1)
            self.assertEqual(
                repair["first_turn_gate_repair_failure_families"]["retry_repair_no_unique_candidate"],
                0,
            )
            self.assertEqual(repair["failed_count_with_missing_reason_count"], 0)


if __name__ == "__main__":
    unittest.main()
