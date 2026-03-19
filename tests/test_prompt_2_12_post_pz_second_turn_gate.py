"""Prompt 2.12 tests for the bounded post-PZ second-turn CR protocol gate."""

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
from agentiad_recon.baseline import (
    _artifact_dirs,
    _runtime_config,
    _should_trigger_post_pz_second_turn_gate,
    _tool_loop_sample,
    load_run_definition,
    run_tool_augmented,
)
from agentiad_recon.mmad import MMADIndexer
from agentiad_recon.prompting import render_answer_block


FIXTURE_ROOT = REPO_ROOT / "tests" / "fixtures" / "mmad_fixture"
PZ_CR_CONFIG = REPO_ROOT / "configs" / "tool_pz_cr_fixture.json"
TOOL_CALL_PZ = (
    '<tool_call>{"tool_name":"PZ","arguments":{"bbox":{"x0":0.10,"y0":0.10,"x1":0.80,"y1":0.80}}}</tool_call>'
)
TOOL_CALL_CR = (
    '<tool_call>{"tool_name":"CR","arguments":{"policy":"same_category_normal"}}</tool_call>'
)


def _direct_final_answer(think: str) -> str:
    return render_answer_block(
        {"anomaly_present": False, "top_anomaly": None, "visual_descriptions": []},
        wrapper_tag="answer",
        think=think,
    )


def _leaky_first_turn_final_answer() -> str:
    return _direct_final_answer(
        "The crop result is sufficient; no comparative retrieval is allowed in pz_only."
    )


class SequenceBackend(InferenceBackend):
    """Backend that returns a deterministic sequence of raw outputs."""

    backend_name = "prompt_2_12_sequence_backend_v1"

    def __init__(self, outputs: list[str]) -> None:
        self.outputs = outputs
        self.calls = 0

    def describe_runtime(self) -> dict[str, object]:
        return {"adapter_loaded": False}

    def generate(self, request: BackendRequest, *, sample: dict[str, object]) -> BackendResponse:
        index = min(self.calls, len(self.outputs) - 1)
        self.calls += 1
        return BackendResponse(backend_name=self.backend_name, raw_output=self.outputs[index], metadata={})


class Prompt212PostPZSecondTurnGateTests(unittest.TestCase):
    """Prompt 2.12 regression tests."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.definition = load_run_definition(PZ_CR_CONFIG)
        cls.samples = [sample.to_dict() for sample in MMADIndexer(FIXTURE_ROOT, source="mmad_fixture").index_samples()]
        cls.sample = cls.samples[0]

    def _runtime(self, *, gate_mode: str = "retry_once_require_cr_after_pz") -> dict[str, object]:
        return _runtime_config(
            self.definition,
            dataset_root=FIXTURE_ROOT,
            runtime_overrides={
                "first_turn_protocol_gate_mode": "retry_once_pz_cr",
                "post_pz_second_turn_gate_mode": gate_mode,
            },
        )

    def _run_sequence(self, outputs: list[str], *, gate_mode: str = "retry_once_require_cr_after_pz") -> dict[str, object]:
        backend = SequenceBackend(outputs)
        with tempfile.TemporaryDirectory() as tempdir:
            directories = _artifact_dirs(self.definition, Path(tempdir))
            record, trace_payload = _tool_loop_sample(
                definition=self.definition,
                backend=backend,
                runtime_config=self._runtime(gate_mode=gate_mode),
                sample=self.sample,
                sample_pool=self.samples,
                seed=0,
                directories=directories,
            )
            gate_sidecar = None
            if record["post_pz_second_turn_gate_sidecar_path"] is not None:
                gate_sidecar = json.loads(
                    Path(record["post_pz_second_turn_gate_sidecar_path"]).read_text(encoding="utf-8")
                )
            transition_sidecar = None
            if record["post_pz_transition_sidecar_path"] is not None:
                transition_sidecar = json.loads(
                    Path(record["post_pz_transition_sidecar_path"]).read_text(encoding="utf-8")
                )
            return {
                "record": record,
                "trace_payload": trace_payload,
                "gate_sidecar": gate_sidecar,
                "transition_sidecar": transition_sidecar,
                "raw_output_exists": Path(record["raw_output_path"]).exists(),
                "retry_raw_output_exists": (
                    Path(record["post_pz_second_turn_gate_retry_raw_output_path"]).exists()
                    if record["post_pz_second_turn_gate_retry_raw_output_path"] is not None
                    else False
                ),
            }

    def test_gate_trigger_predicate_does_not_trigger_when_mode_is_off(self) -> None:
        self.assertFalse(
            _should_trigger_post_pz_second_turn_gate(
                gate_mode="off",
                tool_mode="pz_cr",
                first_successful_tool_name="PZ",
                post_pz_transition_audited=True,
                post_pz_assistant_turn_index=1,
                current_turn_index=1,
                post_sanitation_contract_valid_for_cr=True,
                event="final_answer",
            )
        )

    def test_gate_trigger_predicate_does_not_trigger_for_pz_only(self) -> None:
        self.assertFalse(
            _should_trigger_post_pz_second_turn_gate(
                gate_mode="retry_once_require_cr_after_pz",
                tool_mode="pz_only",
                first_successful_tool_name="PZ",
                post_pz_transition_audited=True,
                post_pz_assistant_turn_index=1,
                current_turn_index=1,
                post_sanitation_contract_valid_for_cr=True,
                event="final_answer",
            )
        )

    def test_gate_trigger_predicate_does_not_trigger_if_first_tool_is_not_pz(self) -> None:
        self.assertFalse(
            _should_trigger_post_pz_second_turn_gate(
                gate_mode="retry_once_require_cr_after_pz",
                tool_mode="pz_cr",
                first_successful_tool_name="CR",
                post_pz_transition_audited=True,
                post_pz_assistant_turn_index=1,
                current_turn_index=1,
                post_sanitation_contract_valid_for_cr=True,
                event="final_answer",
            )
        )

    def test_gate_trigger_predicate_does_not_trigger_if_contract_is_invalid(self) -> None:
        self.assertFalse(
            _should_trigger_post_pz_second_turn_gate(
                gate_mode="retry_once_require_cr_after_pz",
                tool_mode="pz_cr",
                first_successful_tool_name="PZ",
                post_pz_transition_audited=True,
                post_pz_assistant_turn_index=1,
                current_turn_index=1,
                post_sanitation_contract_valid_for_cr=False,
                event="final_answer",
            )
        )

    def test_gate_does_not_trigger_when_mode_is_off(self) -> None:
        result = self._run_sequence(
            [
                _leaky_first_turn_final_answer(),
                TOOL_CALL_PZ,
                _direct_final_answer("second turn direct final answer"),
            ],
            gate_mode="off",
        )
        record = result["record"]

        self.assertFalse(record["post_pz_second_turn_gate_triggered"])
        self.assertEqual(record["post_pz_second_turn_gate_outcome"], "gate_not_triggered")
        self.assertFalse(record["post_pz_second_turn_gate_retry_attempted"])
        self.assertIsNone(record["post_pz_second_turn_gate_retry_raw_output_path"])

    def test_gate_does_not_trigger_if_first_post_pz_turn_already_calls_cr(self) -> None:
        result = self._run_sequence(
            [
                _leaky_first_turn_final_answer(),
                TOOL_CALL_PZ,
                TOOL_CALL_CR,
                _direct_final_answer("answer after CR"),
            ]
        )
        record = result["record"]

        self.assertFalse(record["post_pz_second_turn_gate_triggered"])
        self.assertEqual(record["post_pz_second_turn_gate_outcome"], "gate_not_triggered")
        self.assertTrue(record["post_pz_second_turn_called_cr"])

    def test_gate_triggers_on_direct_final_after_clean_valid_post_pz_contract(self) -> None:
        result = self._run_sequence(
            [
                _leaky_first_turn_final_answer(),
                TOOL_CALL_PZ,
                _direct_final_answer("second turn direct final answer"),
                _direct_final_answer("retry still direct final answer"),
            ]
        )
        record = result["record"]
        sidecar = result["gate_sidecar"]

        self.assertTrue(record["post_pz_transition_post_sanitation_contract_valid_for_cr"])
        self.assertTrue(record["post_pz_second_turn_gate_triggered"])
        self.assertTrue(record["post_pz_second_turn_gate_retry_attempted"])
        self.assertEqual(sidecar["trigger_reason"], "direct_final_answer_after_valid_clean_post_pz_cr_contract")

    def test_retry_recovers_to_cr_tool_call(self) -> None:
        result = self._run_sequence(
            [
                _leaky_first_turn_final_answer(),
                TOOL_CALL_PZ,
                _direct_final_answer("second turn direct final answer"),
                TOOL_CALL_CR,
                _direct_final_answer("answer after CR"),
            ]
        )
        record = result["record"]
        sidecar = result["gate_sidecar"]

        self.assertTrue(record["post_pz_second_turn_gate_triggered"])
        self.assertEqual(record["post_pz_second_turn_gate_outcome"], "recovered_to_cr_call")
        self.assertEqual(record["post_pz_second_turn_gate_retry_called_tool_name"], "CR")
        self.assertFalse(record["post_pz_second_turn_called_cr"])
        self.assertTrue(result["raw_output_exists"])
        self.assertTrue(result["retry_raw_output_exists"])
        self.assertEqual(sidecar["final_gate_outcome"], "recovered_to_cr_call")

    def test_retry_remains_terminal_after_reminder(self) -> None:
        result = self._run_sequence(
            [
                _leaky_first_turn_final_answer(),
                TOOL_CALL_PZ,
                _direct_final_answer("second turn direct final answer"),
                _direct_final_answer("retry still direct final answer"),
            ]
        )
        record = result["record"]

        self.assertEqual(record["post_pz_second_turn_gate_outcome"], "still_terminal_after_retry")
        self.assertTrue(record["post_pz_second_turn_gate_retry_parser_valid"])
        self.assertTrue(record["post_pz_second_turn_gate_retry_schema_valid"])

    def test_retry_parse_failure_is_classified(self) -> None:
        result = self._run_sequence(
            [
                _leaky_first_turn_final_answer(),
                TOOL_CALL_PZ,
                _direct_final_answer("second turn direct final answer"),
                "I will answer directly without a contract block.",
            ]
        )
        record = result["record"]

        self.assertEqual(record["post_pz_second_turn_gate_outcome"], "retry_parse_failure")
        self.assertFalse(record["post_pz_second_turn_gate_retry_parser_valid"])
        self.assertFalse(record["post_pz_second_turn_gate_retry_schema_valid"])
        self.assertIn("post_pz_second_turn_gate_retry_missing_contract_block", record["post_pz_second_turn_gate_retry_failure_reason"])

    def test_retry_wrong_tool_call_is_classified(self) -> None:
        result = self._run_sequence(
            [
                _leaky_first_turn_final_answer(),
                TOOL_CALL_PZ,
                _direct_final_answer("second turn direct final answer"),
                TOOL_CALL_PZ,
            ]
        )
        record = result["record"]
        sidecar = result["gate_sidecar"]

        self.assertEqual(record["post_pz_second_turn_gate_outcome"], "called_non_cr_tool_after_retry")
        self.assertEqual(record["post_pz_second_turn_gate_retry_called_tool_name"], "PZ")
        self.assertIn("called_non_cr_tool_after_retry", record["post_pz_second_turn_gate_retry_failure_reason"])
        self.assertEqual(sidecar["retry_called_tool_name"], "PZ")

    def test_summary_manifest_metrics_and_consistency_fields_are_exposed(self) -> None:
        backend = SequenceBackend(
            [
                _leaky_first_turn_final_answer(),
                TOOL_CALL_PZ,
                _direct_final_answer("second turn direct final answer"),
                TOOL_CALL_CR,
                _direct_final_answer("answer after CR"),
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
                runtime_overrides={
                    "first_turn_protocol_gate_mode": "retry_once_pz_cr",
                    "post_pz_second_turn_gate_mode": "retry_once_require_cr_after_pz",
                },
            )
            summary = json.loads(Path(result["summary_path"]).read_text(encoding="utf-8"))
            manifest = json.loads(Path(result["run_manifest_path"]).read_text(encoding="utf-8"))
            metrics = result["metrics_report"]
            record = result["prediction_records"][0]
            gate_sidecar = json.loads(Path(record["post_pz_second_turn_gate_sidecar_path"]).read_text(encoding="utf-8"))
            transition_sidecar = json.loads(Path(record["post_pz_transition_sidecar_path"]).read_text(encoding="utf-8"))

        for payload in (summary, manifest, metrics):
            gate_summary = payload["post_pz_second_turn_gate_summary"]
            self.assertEqual(gate_summary["post_pz_second_turn_gate_trigger_count"], 1)
            self.assertEqual(gate_summary["post_pz_second_turn_gate_recovered_to_cr_call_count"], 1)
            self.assertEqual(gate_summary["post_pz_second_turn_gate_retry_parse_failure_count"], 0)
            self.assertEqual(gate_summary["post_pz_second_turn_gate_still_terminal_count"], 0)
            self.assertEqual(gate_summary["post_pz_second_turn_gate_called_non_cr_tool_count"], 0)
            self.assertEqual(gate_summary["failed_count_with_missing_reason_count"], 0)
            self.assertEqual(payload["post_pz_second_turn_gate_mode"], "retry_once_require_cr_after_pz")

        self.assertIn("post_pz_second_turn_gate_summary", summary["artifact_paths"])
        self.assertTrue(record["post_pz_transition_sanitation_applied"])
        self.assertTrue(record["post_pz_transition_post_sanitation_contract_valid_for_cr"])
        self.assertTrue(record["post_pz_second_turn_gate_triggered"])
        self.assertEqual(record["post_pz_second_turn_gate_outcome"], "recovered_to_cr_call")
        self.assertEqual(
            gate_sidecar["post_pz_transition_sidecar_path"],
            record["post_pz_transition_sidecar_path"],
        )
        self.assertTrue(transition_sidecar["post_sanitation_transition_contract_valid_for_cr"])


if __name__ == "__main__":
    unittest.main()
