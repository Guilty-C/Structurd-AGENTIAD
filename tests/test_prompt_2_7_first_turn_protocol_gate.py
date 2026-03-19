"""Prompt 2.7 tests for the pz_cr first-turn protocol gate and single retry."""

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
    _tool_loop_sample,
    load_run_definition,
    run_tool_augmented,
)
from agentiad_recon.mmad import MMADIndexer
from agentiad_recon.prompting import PromptBundle, build_prompt, render_answer_block


FIXTURE_ROOT = REPO_ROOT / "tests" / "fixtures" / "mmad_fixture"
PZ_CR_CONFIG = REPO_ROOT / "configs" / "tool_pz_cr_fixture.json"
PZ_ONLY_CONFIG = REPO_ROOT / "configs" / "tool_pz_only_fixture.json"


def _direct_final_answer(think: str) -> str:
    return render_answer_block(
        {
            "anomaly_present": False,
            "top_anomaly": None,
            "visual_descriptions": [],
        },
        wrapper_tag="answer",
        think=think,
    )


TOOL_CALL_PZ = (
    '<tool_call>{"tool_name":"PZ","arguments":{"bbox":{"x0":0.10,"y0":0.10,"x1":0.80,"y1":0.80}}}</tool_call>'
)


class SequenceBackend(InferenceBackend):
    """Backend that returns a fixed sequence of raw outputs."""

    backend_name = "prompt_2_7_sequence_backend_v1"

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


class Prompt27FirstTurnGateTests(unittest.TestCase):
    """Prompt 2.7 regression tests."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.pz_cr_definition = load_run_definition(PZ_CR_CONFIG)
        cls.pz_only_definition = load_run_definition(PZ_ONLY_CONFIG)
        cls.samples = [sample.to_dict() for sample in MMADIndexer(FIXTURE_ROOT, source="mmad_fixture").index_samples()]
        cls.sample = cls.samples[0]

    def _runtime(self, definition: dict[str, object], gate_mode: str) -> dict[str, object]:
        return _runtime_config(
            definition,
            dataset_root=FIXTURE_ROOT,
            runtime_overrides={"first_turn_protocol_gate_mode": gate_mode},
        )

    def _leaked_pz_only_prompt(self, sample: dict[str, object]) -> PromptBundle:
        leaked = build_prompt(sample, tool_path="pz_only")
        return PromptBundle(
            prompt_version=leaked.prompt_version,
            tool_path="pz_cr",
            messages=leaked.messages,
            stop_sequences=leaked.stop_sequences,
        )

    def test_gate_does_not_fire_in_pz_only(self) -> None:
        """The gate is pz_cr-only and must stay off for pz_only runs."""

        backend = SequenceBackend([_direct_final_answer("pz_only direct final answer")])
        with tempfile.TemporaryDirectory() as tempdir:
            directories = _artifact_dirs(self.pz_only_definition, Path(tempdir))
            record, _trace_payload = _tool_loop_sample(
                definition=self.pz_only_definition,
                backend=backend,
                runtime_config=self._runtime(self.pz_only_definition, "retry_once_pz_cr"),
                sample=self.sample,
                sample_pool=self.samples,
                seed=0,
                directories=directories,
            )

        self.assertFalse(record["first_turn_gate_triggered"])
        self.assertEqual(record["first_turn_gate_outcome"], "not_triggered")
        self.assertEqual(record["first_turn_gate_retry_count"], 0)
        self.assertEqual(backend.calls, 1)
        self.assertIsNone(record["first_turn_gate_sidecar_path"])

    def test_gate_does_not_fire_when_prompt_contract_is_already_broken(self) -> None:
        """A broken prompt audit should fail fast without gate intervention."""

        backend = SequenceBackend([_direct_final_answer("should never be used")])
        with tempfile.TemporaryDirectory() as tempdir, mock.patch(
            "agentiad_recon.baseline.build_prompt",
            side_effect=lambda sample, tool_path, **_kwargs: self._leaked_pz_only_prompt(sample),
        ):
            directories = _artifact_dirs(self.pz_cr_definition, Path(tempdir))
            record, _trace_payload = _tool_loop_sample(
                definition=self.pz_cr_definition,
                backend=backend,
                runtime_config=self._runtime(self.pz_cr_definition, "retry_once_pz_cr"),
                sample=self.sample,
                sample_pool=self.samples,
                seed=0,
                directories=directories,
            )

        self.assertFalse(record["first_turn_gate_triggered"])
        self.assertEqual(record["first_turn_gate_outcome"], "not_triggered")
        self.assertEqual(record["failure_reason"], "mode_contract_pz_only_leakage")
        self.assertEqual(backend.calls, 0)

    def test_gate_fires_once_and_recovers_to_tool_call(self) -> None:
        """The gate should fire on turn-0 direct-final collapse and recover if retry emits a tool call."""

        backend = SequenceBackend(
            [
                _direct_final_answer("turn 0 direct final answer"),
                TOOL_CALL_PZ,
                _direct_final_answer("final answer after tool use"),
            ]
        )
        with tempfile.TemporaryDirectory() as tempdir:
            directories = _artifact_dirs(self.pz_cr_definition, Path(tempdir))
            record, trace_payload = _tool_loop_sample(
                definition=self.pz_cr_definition,
                backend=backend,
                runtime_config=self._runtime(self.pz_cr_definition, "retry_once_pz_cr"),
                sample=self.sample,
                sample_pool=self.samples,
                seed=0,
                directories=directories,
            )
            sidecar = json.loads(Path(record["first_turn_gate_sidecar_path"]).read_text(encoding="utf-8"))

        self.assertTrue(record["first_turn_gate_triggered"])
        self.assertEqual(record["first_turn_gate_retry_count"], 1)
        self.assertTrue(record["first_turn_gate_recovered"])
        self.assertEqual(record["first_turn_gate_outcome"], "recovered_to_tool_call")
        self.assertGreater(record["tool_call_count"], 0)
        self.assertEqual(backend.calls, 3)
        self.assertEqual(sidecar["final_gate_outcome"], "recovered_to_tool_call")
        self.assertEqual(trace_payload["metadata"]["first_turn_gate_outcome"], "recovered_to_tool_call")

    def test_gate_still_terminal_path_is_auditable_and_non_fabricating(self) -> None:
        """If retry still answers directly, the runtime must not invent tool usage."""

        backend = SequenceBackend(
            [
                _direct_final_answer("turn 0 direct final answer"),
                _direct_final_answer("retry still direct final answer"),
            ]
        )
        with tempfile.TemporaryDirectory() as tempdir:
            directories = _artifact_dirs(self.pz_cr_definition, Path(tempdir))
            record, _trace_payload = _tool_loop_sample(
                definition=self.pz_cr_definition,
                backend=backend,
                runtime_config=self._runtime(self.pz_cr_definition, "retry_once_pz_cr"),
                sample=self.sample,
                sample_pool=self.samples,
                seed=0,
                directories=directories,
            )
            sidecar = json.loads(Path(record["first_turn_gate_sidecar_path"]).read_text(encoding="utf-8"))

        self.assertTrue(record["first_turn_gate_triggered"])
        self.assertFalse(record["first_turn_gate_recovered"])
        self.assertEqual(record["first_turn_gate_outcome"], "still_terminal_after_retry")
        self.assertEqual(record["tool_call_count"], 0)
        self.assertEqual(record["called_tools"], [])
        self.assertEqual(sidecar["retry_count"], 1)
        self.assertEqual(sidecar["final_gate_outcome"], "still_terminal_after_retry")
        self.assertEqual(backend.calls, 2)

    def test_gate_retry_parse_failure_is_recorded(self) -> None:
        """Malformed retry outputs should be recorded as retry_parse_failure."""

        backend = SequenceBackend(
            [
                _direct_final_answer("turn 0 direct final answer"),
                "this retry is missing both tool and answer blocks",
            ]
        )
        with tempfile.TemporaryDirectory() as tempdir:
            directories = _artifact_dirs(self.pz_cr_definition, Path(tempdir))
            record, _trace_payload = _tool_loop_sample(
                definition=self.pz_cr_definition,
                backend=backend,
                runtime_config=self._runtime(self.pz_cr_definition, "retry_once_pz_cr"),
                sample=self.sample,
                sample_pool=self.samples,
                seed=0,
                directories=directories,
            )

        self.assertTrue(record["first_turn_gate_triggered"])
        self.assertEqual(record["first_turn_gate_outcome"], "retry_parse_failure")
        self.assertEqual(record["first_turn_gate_retry_count"], 1)
        self.assertEqual(record["failure_reason"], "runtime_exception:retry_repair_no_unique_candidate")
        self.assertEqual(backend.calls, 2)

    def test_run_summary_gate_counts_match_prediction_records(self) -> None:
        """Run-level gate summary counts should equal the prediction-level counts."""

        backend = SequenceBackend(
            [
                _direct_final_answer("turn 0 direct final answer"),
                _direct_final_answer("retry still direct final answer"),
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
            records = result["prediction_records"]

        triggered = sum(1 for record in records if record["first_turn_gate_triggered"])
        still_terminal = sum(
            1 for record in records if record["first_turn_gate_outcome"] == "still_terminal_after_retry"
        )
        recovered = sum(
            1 for record in records if record["first_turn_gate_outcome"] == "recovered_to_tool_call"
        )
        parse_failure = sum(
            1 for record in records if record["first_turn_gate_outcome"] == "retry_parse_failure"
        )
        for payload in (summary, manifest, metrics):
            gate = payload["first_turn_gate_summary"]
            self.assertEqual(gate["first_turn_gate_trigger_count"], triggered)
            self.assertEqual(gate["samples_with_first_turn_gate_events"], triggered)
            self.assertEqual(gate["first_turn_gate_still_terminal_count"], still_terminal)
            self.assertEqual(gate["first_turn_gate_recovered_to_tool_call_count"], recovered)
            self.assertEqual(gate["first_turn_gate_retry_parse_failure_count"], parse_failure)


if __name__ == "__main__":
    unittest.main()
