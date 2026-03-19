"""Prompt 2.9 tests for bounded retry repair expansion and unrecoverable taxonomy."""

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
from agentiad_recon.tooling import RETRY_REPAIR_ORIGINAL_FAILURE_FAMILY, repair_retry_tool_call_output


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
DUPLICATE_BARE_PZ_JSON = f"<think>{BARE_PZ_JSON}</think>\n{BARE_PZ_JSON}"
BARE_ALIAS_JSON = '{"tool_name":"pz_cr","arguments":{"bbox":[0.10,0.10,0.80,0.80]}}'
PSEUDO_TOOL_JSON = '{"tool_name":"pz_cr","arguments":{"crop_result":{"status":"ok"},"query_result":{"status":"none"}}}'
AMBIGUOUS_CANDIDATES = (
    '{"tool_name":"PZ","arguments":{"bbox":{"x0":0.10,"y0":0.10,"x1":0.80,"y1":0.80}}}\n'
    '{"tool_name":"PZ","arguments":{"bbox":{"x0":0.20,"y0":0.20,"x1":0.70,"y1":0.70}}}'
)


class SequenceBackend(InferenceBackend):
    """Backend that returns a deterministic sequence of raw outputs."""

    backend_name = "prompt_2_9_sequence_backend_v1"

    def __init__(self, outputs: list[str]) -> None:
        self.outputs = outputs
        self.calls = 0

    def describe_runtime(self) -> dict[str, object]:
        return {"adapter_loaded": False}

    def generate(self, request: BackendRequest, *, sample: dict[str, object]) -> BackendResponse:
        index = min(self.calls, len(self.outputs) - 1)
        self.calls += 1
        return BackendResponse(backend_name=self.backend_name, raw_output=self.outputs[index], metadata={})


class Prompt29RetryRepairTaxonomyTests(unittest.TestCase):
    """Prompt 2.9 regression tests."""

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

    def test_identical_duplicate_candidate_recovers_once(self) -> None:
        """Plain plus <think> duplicate candidates should deduplicate to one recoverable PZ call."""

        decision = repair_retry_tool_call_output(DUPLICATE_BARE_PZ_JSON, tool_path="pz_cr")

        self.assertTrue(decision.succeeded)
        self.assertTrue(decision.wrapper_recovery_applied)
        self.assertTrue(decision.duplicate_candidate_deduplication_applied)
        self.assertEqual(decision.extracted_candidate_count, 2)
        self.assertEqual(decision.unique_candidate_count, 1)
        self.assertIn("duplicate_candidate_deduplication", decision.repair_categories)

    def test_wrapper_quote_corruption_recovers(self) -> None:
        """Smart quotes and minor punctuation corruption around a wrapped bbox dict should recover."""

        decision = repair_retry_tool_call_output(WRAPPED_SMART_QUOTES, tool_path="pz_cr")

        self.assertTrue(decision.succeeded)
        self.assertTrue(decision.quote_normalization_applied)
        self.assertEqual(decision.selected_canonical_arguments["bbox"]["y0"], 0.10)

    def test_alias_and_bbox_list_recover_to_canonical_pz(self) -> None:
        """Allowed aliases plus bbox lists should canonicalize to the strict PZ contract."""

        decision = repair_retry_tool_call_output(BARE_ALIAS_JSON, tool_path="pz_cr")

        self.assertTrue(decision.succeeded)
        self.assertTrue(decision.alias_canonicalization_applied)
        self.assertTrue(decision.bbox_canonicalization_applied)
        self.assertEqual(decision.selected_tool_name, "PZ")
        self.assertEqual(decision.selected_canonical_arguments["bbox"]["x0"], 0.10)

    def test_pseudo_observation_payload_is_not_repaired(self) -> None:
        """Result-shaped payloads must remain unrecoverable and explicitly classified."""

        decision = repair_retry_tool_call_output(PSEUDO_TOOL_JSON, tool_path="pz_cr")

        self.assertTrue(decision.attempted)
        self.assertFalse(decision.succeeded)
        self.assertEqual(
            decision.failure_family,
            "retry_repair_unrecoverable_pseudo_observation_payload",
        )
        self.assertEqual(decision.original_failure_family, RETRY_REPAIR_ORIGINAL_FAILURE_FAMILY)
        self.assertIsNone(decision.repaired_text)

    def test_materially_different_candidates_remain_ambiguous_failure(self) -> None:
        """Multiple different candidate payloads must remain unrecoverable."""

        decision = repair_retry_tool_call_output(AMBIGUOUS_CANDIDATES, tool_path="pz_cr")

        self.assertTrue(decision.attempted)
        self.assertFalse(decision.succeeded)
        self.assertEqual(decision.failure_family, "retry_repair_ambiguous_multiple_candidates")
        self.assertEqual(decision.unique_candidate_count, 2)

    def test_sidecar_and_summary_expose_failure_family_counters(self) -> None:
        """Prediction records, sidecars, and summaries should expose retry failure families."""

        backend = SequenceBackend(
            [
                _direct_final_answer("turn 0 final answer"),
                PSEUDO_TOOL_JSON,
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
            record = result["prediction_records"][0]
            summary = json.loads(Path(result["summary_path"]).read_text(encoding="utf-8"))
            manifest = json.loads(Path(result["run_manifest_path"]).read_text(encoding="utf-8"))
            metrics = result["metrics_report"]
            sidecar = json.loads(Path(record["first_turn_gate_sidecar_path"]).read_text(encoding="utf-8"))
            failure_family_artifact = json.loads(
                (
                    Path(summary["artifact_paths"]["first_turn_gate_repair_failure_families"])
                ).read_text(encoding="utf-8")
            )

        self.assertEqual(
            record["first_turn_gate_repair_original_failure_family"],
            RETRY_REPAIR_ORIGINAL_FAILURE_FAMILY,
        )
        self.assertEqual(
            record["first_turn_gate_repair_failure_family"],
            "retry_repair_unrecoverable_pseudo_observation_payload",
        )
        self.assertEqual(
            record["failure_reason"],
            "runtime_exception:retry_repair_unrecoverable_pseudo_observation_payload",
        )
        self.assertEqual(
            sidecar["retry_repair"]["failure_family"],
            "retry_repair_unrecoverable_pseudo_observation_payload",
        )
        self.assertEqual(sidecar["retry_repair"]["original_text"], PSEUDO_TOOL_JSON)
        self.assertIsNone(sidecar["retry_repair"]["repaired_text"])

        for payload in (summary, manifest, metrics):
            repair_summary = payload["first_turn_gate_repair_summary"]
            self.assertEqual(repair_summary["first_turn_gate_repair_attempt_count"], 1)
            self.assertEqual(repair_summary["first_turn_gate_repair_success_count"], 0)
            self.assertEqual(repair_summary["first_turn_gate_repair_failure_count"], 1)
            self.assertEqual(
                repair_summary["first_turn_gate_repair_failure_families"][
                    "retry_repair_unrecoverable_pseudo_observation_payload"
                ],
                1,
            )
            self.assertEqual(repair_summary["failed_count_with_missing_reason_count"], 0)

        self.assertEqual(
            failure_family_artifact["families"]["retry_repair_unrecoverable_pseudo_observation_payload"]["count"],
            1,
        )
        self.assertEqual(
            failure_family_artifact["families"]["retry_repair_unrecoverable_pseudo_observation_payload"]["sample_ids"],
            [self.sample["sample_id"]],
        )


if __name__ == "__main__":
    unittest.main()
