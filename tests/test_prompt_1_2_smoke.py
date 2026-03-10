"""Fixture-backed smoke tests for the Prompt 1.2 D/E/F plumbing.

These tests exercise the canonical sample layer, deterministic tools, prompt
contract, final-answer parser, and trace handling using a tiny local fixture
dataset. They do not run any model, full dataset job, or remote workflow.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from agentiad_recon.mmad import MMADIndexer, export_canonical_samples
from agentiad_recon.prompting import PROMPT_VERSION, build_prompt, parse_final_answer
from agentiad_recon.tooling import (
    ComparativeRetriever,
    PerceptiveZoomer,
    execute_tool_call,
    parse_tool_call,
    protocol_event,
    reinsert_tool_result,
)
from agentiad_recon.traces import TraceMessage, TraceRecord


FIXTURE_ROOT = Path(__file__).resolve().parent / "fixtures" / "mmad_fixture"


class Prompt12SmokeTests(unittest.TestCase):
    """Local-only contract smoke tests for the clean-room Prompt 1.2 layer."""

    def setUp(self) -> None:
        """Index the fixture dataset once per test for deterministic assertions."""

        self.samples = [sample.to_dict() for sample in MMADIndexer(FIXTURE_ROOT).index_samples()]
        self.anomaly_sample = next(sample for sample in self.samples if sample["anomaly_present"])
        self.normal_sample = next(sample for sample in self.samples if not sample["anomaly_present"])

    def test_mmad_indexing_and_export(self) -> None:
        """Fixture indexing should stay deterministic and schema-valid."""

        self.assertEqual([sample["sample_id"] for sample in self.samples], sorted(sample["sample_id"] for sample in self.samples))
        self.assertEqual(self.anomaly_sample["category"], "capsule")
        self.assertEqual(self.anomaly_sample["anomaly_candidates"], ["crack"])
        self.assertIsNotNone(self.anomaly_sample["mask"])

        with tempfile.TemporaryDirectory() as tempdir:
            output_path = Path(tempdir) / "samples.jsonl"
            manifest = export_canonical_samples(MMADIndexer(FIXTURE_ROOT).index_samples(), output_path, limit=2)
            lines = output_path.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(lines), 2)
            self.assertEqual(manifest["sample_count"], 2)

    def test_pz_contract_is_deterministic(self) -> None:
        """PZ should convert normalized coordinates into stable crop metadata."""

        with tempfile.TemporaryDirectory() as tempdir:
            result = PerceptiveZoomer().run(
                image_path=self.anomaly_sample["image"]["uri"],
                bbox={"x0": 0.25, "y0": 0.25, "x1": 0.75, "y1": 0.75},
                call_id="cropcheck",
                artifact_dir=tempdir,
            )
            payload = result.to_payload()
            self.assertEqual(payload["output_payload"]["pixel_bbox"], {"left": 1, "top": 1, "right": 3, "bottom": 3})
            self.assertTrue(Path(payload["output_payload"]["artifact_path"]).exists())

    def test_cr_contract_is_deterministic_and_logs_fallback(self) -> None:
        """CR should pick the same exemplar every time and log empty cases."""

        result = ComparativeRetriever().run(
            target_sample=self.anomaly_sample,
            sample_pool=self.samples,
            call_id="crcheck",
        )
        payload = result.to_payload()
        self.assertEqual(payload["output_payload"]["selected_exemplar"]["sample_id"], self.normal_sample["sample_id"])

        fallback = ComparativeRetriever().run(
            target_sample=self.anomaly_sample,
            sample_pool=[self.anomaly_sample],
            call_id="crfallback",
        ).to_payload()
        self.assertIsNone(fallback["output_payload"]["selected_exemplar"])

    def test_tool_protocol_and_reinsertion(self) -> None:
        """Tool calls should parse cleanly and produce reinsertion-ready payloads."""

        raw = """
        <tool_call>
        {"tool_name":"PZ","arguments":{"bbox":{"x0":0.1,"y0":0.1,"x1":0.9,"y1":0.9}}}
        </tool_call>
        """
        self.assertEqual(protocol_event(raw), "tool_call")
        parsed = parse_tool_call(raw, tool_path="pz_only")
        with tempfile.TemporaryDirectory() as tempdir:
            result = execute_tool_call(parsed, sample=self.anomaly_sample, sample_pool=self.samples, artifact_dir=tempdir)
        history = reinsert_tool_result([], result)
        self.assertEqual(history[-1]["role"], "tool")
        self.assertEqual(history[-1]["tool_name"], "PZ")

    def test_prompt_and_answer_contracts(self) -> None:
        """Prompts should be versioned and answers should parse strictly."""

        prompt = build_prompt(self.anomaly_sample, tool_path="pz_cr")
        self.assertEqual(prompt.prompt_version, PROMPT_VERSION)
        self.assertIn("PZ and CR", prompt.messages[1]["content"])

        parsed = parse_final_answer(
            """
            <final_answer>
              <anomaly_present>true</anomaly_present>
              <top_anomaly>crack</top_anomaly>
              <visual_descriptions>
                <item>Thin crack on the capsule surface.</item>
              </visual_descriptions>
            </final_answer>
            """
        )
        self.assertTrue(parsed["anomaly_present"])
        self.assertEqual(parsed["top_anomaly"], "crack")

        with self.assertRaisesRegex(Exception, "top_anomaly"):
            parse_final_answer(
                """
                <final_answer>
                  <anomaly_present>false</anomaly_present>
                  <top_anomaly>crack</top_anomaly>
                  <visual_descriptions></visual_descriptions>
                </final_answer>
                """
            )

    def test_trace_serialization_supports_eval_and_training_views(self) -> None:
        """Audit traces and training trajectory projections should stay structured."""

        tool_call = ComparativeRetriever().run(
            target_sample=self.anomaly_sample,
            sample_pool=self.samples,
            call_id="tracecr",
        ).to_payload()
        final_answer = parse_final_answer(
            """
            <final_answer>
              <anomaly_present>true</anomaly_present>
              <top_anomaly>crack</top_anomaly>
              <visual_descriptions>
                <item>Localized crack is visible on the capsule.</item>
              </visual_descriptions>
            </final_answer>
            """
        )
        trace = TraceRecord(
            trace_id="trace-001",
            sample_id=self.anomaly_sample["sample_id"],
            stage="sft",
            tool_path="pz_cr",
            storage_purpose="training",
            messages=(
                TraceMessage(role="system", message_type="system_instruction", content="System", metadata={}),
                TraceMessage(role="user", message_type="user_prompt", content="User", metadata={}),
                TraceMessage(role="assistant", message_type="reasoning", content="Need a reference sample.", metadata={}),
                TraceMessage(role="assistant", message_type="tool_request", content="<tool_call>...</tool_call>", tool_name="CR", call_id="tracecr", metadata={}),
                TraceMessage(role="tool", message_type="tool_result", content=json.dumps(tool_call["output_payload"], sort_keys=True), tool_name="CR", call_id="tracecr", metadata={}),
                TraceMessage(role="assistant", message_type="final_answer", content="<final_answer>...</final_answer>", metadata={}),
            ),
            tool_traces=(tool_call,),
            final_answer=final_answer,
            metadata={"fixture": True},
        )
        audit_payload = trace.to_audit_payload()
        training_payload = trace.to_training_trajectory()
        self.assertEqual(audit_payload["storage_purpose"], "training")
        self.assertEqual(training_payload["tool_path"], "pz_cr")


if __name__ == "__main__":
    unittest.main()
