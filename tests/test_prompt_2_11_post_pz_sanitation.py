"""Prompt 2.11 tests for post-PZ second-turn runtime sanitation."""

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
    _append_final_answer_message,
    _append_runtime_gate_reminder,
    _append_tool_request,
    _artifact_dirs,
    _post_pz_transition_payload,
    _prompt_history,
    _runtime_config,
    _sanitize_post_pz_transition_history,
    _tool_loop_sample,
    _first_turn_protocol_gate_retry_instruction,
    load_run_definition,
    run_tool_augmented,
)
from agentiad_recon.mmad import MMADIndexer
from agentiad_recon.prompting import build_prompt, render_answer_block
from agentiad_recon.tooling import execute_tool_call, parse_tool_call, reinsert_tool_result


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

    backend_name = "prompt_2_11_sequence_backend_v1"

    def __init__(self, outputs: list[str]) -> None:
        self.outputs = outputs
        self.calls = 0

    def describe_runtime(self) -> dict[str, object]:
        return {"adapter_loaded": False}

    def generate(self, request: BackendRequest, *, sample: dict[str, object]) -> BackendResponse:
        index = min(self.calls, len(self.outputs) - 1)
        self.calls += 1
        return BackendResponse(backend_name=self.backend_name, raw_output=self.outputs[index], metadata={})


class Prompt211PostPZSanitationTests(unittest.TestCase):
    """Prompt 2.11 regression tests."""

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

    def _contaminated_history_after_recovered_pz(self) -> list[dict[str, object]]:
        prompt_bundle = build_prompt(self.sample, tool_path="pz_cr")
        history = _prompt_history(prompt_bundle)
        _append_final_answer_message(
            history,
            _leaky_first_turn_final_answer(),
            backend_name="prompt_2_11_test_backend",
            raw_output_path="/tmp/turn_0.txt",
        )
        _append_runtime_gate_reminder(
            history,
            _first_turn_protocol_gate_retry_instruction(),
            gate_mode="retry_once_pz_cr",
        )
        parsed_call = parse_tool_call(TOOL_CALL_PZ, tool_path="pz_cr")
        _append_tool_request(
            history,
            TOOL_CALL_PZ,
            backend_name="prompt_2_11_test_backend",
            raw_output_path="/tmp/turn_0.retry_1.txt",
            tool_name=parsed_call.tool_name,
            call_id=parsed_call.call_id,
        )
        tool_result = execute_tool_call(
            parsed_call,
            sample=self.sample,
            sample_pool=self.samples,
            artifact_dir=Path(tempfile.gettempdir()) / "prompt_2_11_tool_artifacts",
        )
        return reinsert_tool_result(history, tool_result)

    def _sanitized_payload(self) -> dict[str, object]:
        history = self._contaminated_history_after_recovered_pz()
        sanitized_history, sanitation_audit = _sanitize_post_pz_transition_history(history)
        return _post_pz_transition_payload(
            pre_sanitation_history=history,
            post_sanitation_history=sanitized_history,
            sample_id=self.sample["sample_id"],
            tool_mode="pz_cr",
            first_tool_name="PZ",
            first_tool_turn_index=0,
            first_turn_gate_outcome="recovered_to_tool_call",
            first_turn_retry_repair_involved=False,
            post_pz_assistant_turn_index=1,
            prior_tool_trace_count=1,
            sanitation_audit=sanitation_audit,
        )

    def test_sanitation_removes_obsolete_terminal_answer_branch(self) -> None:
        payload = self._sanitized_payload()

        self.assertTrue(payload["sanitation_applied"])
        self.assertGreater(payload["removed_obsolete_terminal_answer_count"], 0)
        self.assertGreater(payload["removed_message_count"], 0)

    def test_sanitation_removes_pz_only_leakage_from_active_surface(self) -> None:
        payload = self._sanitized_payload()

        self.assertTrue(payload["pre_sanitation_pz_only_leakage_present"])
        self.assertFalse(payload["post_sanitation_pz_only_leakage_present"])

    def test_sanitation_preserves_valid_system_user_and_tool_result_messages(self) -> None:
        history = self._contaminated_history_after_recovered_pz()
        sanitized_history, _sanitation_audit = _sanitize_post_pz_transition_history(history)

        self.assertEqual(sanitized_history[0]["role"], "system")
        self.assertEqual(sanitized_history[1]["role"], "user")
        self.assertTrue(
            any(
                message["role"] == "tool"
                and message["message_type"] == "tool_result"
                and message.get("tool_name") == "PZ"
                for message in sanitized_history
            )
        )

    def test_post_sanitized_transition_contract_becomes_valid_for_cr(self) -> None:
        payload = self._sanitized_payload()

        self.assertIn("post_pz_transition_pz_only_leakage", payload["transition_mismatch_reasons"])
        self.assertTrue(payload["post_sanitation_transition_contract_valid_for_cr"])
        self.assertEqual(payload["post_sanitation_transition_mismatch_reasons"], [])

    def test_second_turn_direct_final_after_sanitized_valid_cr_contract(self) -> None:
        backend = SequenceBackend(
            [
                _leaky_first_turn_final_answer(),
                TOOL_CALL_PZ,
                _direct_final_answer("second turn direct final answer"),
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
            sidecar = json.loads(Path(record["post_pz_transition_sidecar_path"]).read_text(encoding="utf-8"))

        self.assertTrue(record["post_pz_transition_sanitation_applied"])
        self.assertTrue(record["post_pz_transition_post_sanitation_contract_valid_for_cr"])
        self.assertTrue(record["post_pz_second_turn_direct_final_without_cr"])
        self.assertFalse(record["post_pz_second_turn_called_cr"])
        self.assertTrue(record["parser_valid"])
        self.assertTrue(record["schema_valid"])
        self.assertFalse(sidecar["post_sanitation_pz_only_leakage_present"])

    def test_second_turn_cr_call_after_sanitized_valid_cr_contract(self) -> None:
        backend = SequenceBackend(
            [
                _leaky_first_turn_final_answer(),
                TOOL_CALL_PZ,
                TOOL_CALL_CR,
                _direct_final_answer("answer after CR"),
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

        self.assertTrue(record["post_pz_transition_post_sanitation_contract_valid_for_cr"])
        self.assertTrue(record["post_pz_second_turn_called_cr"])
        self.assertFalse(record["post_pz_second_turn_direct_final_without_cr"])

    def test_summary_aggregation_and_sidecar_integration(self) -> None:
        backend = SequenceBackend(
            [
                _leaky_first_turn_final_answer(),
                TOOL_CALL_PZ,
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
                runtime_overrides={"first_turn_protocol_gate_mode": "retry_once_pz_cr"},
            )
            summary = json.loads(Path(result["summary_path"]).read_text(encoding="utf-8"))
            manifest = json.loads(Path(result["run_manifest_path"]).read_text(encoding="utf-8"))
            metrics = result["metrics_report"]

        for payload in (summary, manifest, metrics):
            sanitation = payload["post_pz_transition_sanitation_summary"]
            self.assertEqual(sanitation["post_pz_transition_sanitation_event_count"], 1)
            self.assertEqual(sanitation["post_pz_transition_sanitation_applied_count"], 1)
            self.assertEqual(sanitation["post_pz_transition_post_sanitation_contract_valid_count"], 1)
            self.assertEqual(sanitation["post_pz_transition_post_sanitation_pz_only_leakage_count"], 0)
            self.assertEqual(sanitation["post_pz_second_turn_called_cr_count"], 1)

        for key in (
            "post_pz_transition_sanitation_summary",
            "per_dataset_post_pz_transition_sanitation",
            "per_category_post_pz_transition_sanitation",
        ):
            self.assertIn(key, summary["artifact_paths"])


if __name__ == "__main__":
    unittest.main()
