"""Fixture-backed smoke tests for the Prompt 1.4 tool-enabled inference path.

These tests extend the Prompt 1.3 baseline runner instead of creating a second
stack. They validate `pz_only` and `pz_cr` prompt generation, bounded tool-loop
execution, explicit malformed/disallowed tool handling, auditable trace export,
and structural delta-vs-baseline artifact generation using only local fixtures
and scripted mock backends.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from agentiad_recon.backends import BackendRequest, MockToolAwareBackend
from agentiad_recon.baseline import load_run_definition, run_tool_augmented
from agentiad_recon.mmad import MMADIndexer
from agentiad_recon.prompting import PROMPT_VERSION, build_prompt
from agentiad_recon.tooling import ToolContractError, execute_tool_call, parse_tool_call, reinsert_tool_result


FIXTURE_ROOT = REPO_ROOT / "tests" / "fixtures" / "mmad_fixture"
PZ_ONLY_CONFIG = REPO_ROOT / "configs" / "tool_pz_only_fixture.json"
PZ_CR_CONFIG = REPO_ROOT / "configs" / "tool_pz_cr_fixture.json"
BASELINE_CONFIG = REPO_ROOT / "configs" / "baseline_non_tool_fixture.json"


class Prompt14ToolInferenceSmokeTests(unittest.TestCase):
    """Local-only tool-enabled pipeline tests for Prompt 1.4."""

    @classmethod
    def setUpClass(cls) -> None:
        """Load a deterministic canonical sample pool once for all smoke tests."""

        cls.samples = [sample.to_dict() for sample in MMADIndexer(FIXTURE_ROOT, source="mmad_fixture").index_samples()]
        cls.sample = cls.samples[1]

    def test_tool_configs_are_frozen_and_aligned_with_baseline(self) -> None:
        """Both tool configs should keep seeds and sample source aligned with baseline."""

        baseline = load_run_definition(BASELINE_CONFIG)
        pz_only = load_run_definition(PZ_ONLY_CONFIG)
        pz_cr = load_run_definition(PZ_CR_CONFIG)

        self.assertEqual(pz_only["mode"], "pz_only")
        self.assertEqual(pz_cr["mode"], "pz_cr")
        self.assertEqual(pz_only["prompt_version"], PROMPT_VERSION)
        self.assertEqual(pz_cr["prompt_version"], PROMPT_VERSION)
        self.assertEqual(pz_only["seeds"], baseline["seeds"])
        self.assertEqual(pz_cr["seeds"], baseline["seeds"])
        self.assertEqual(pz_only["sample_source"], baseline["sample_source"])
        self.assertEqual(pz_cr["sample_source"], baseline["sample_source"])

    def test_tool_prompts_separate_pz_only_and_pz_cr(self) -> None:
        """Prompt wording should distinguish allowed tool surfaces by mode."""

        pz_only_prompt = build_prompt(self.sample, tool_path="pz_only")
        pz_cr_prompt = build_prompt(self.sample, tool_path="pz_cr")

        self.assertEqual(pz_only_prompt.prompt_version, PROMPT_VERSION)
        self.assertEqual(pz_cr_prompt.prompt_version, PROMPT_VERSION)
        self.assertIn("Available tools: PZ only", pz_only_prompt.messages[1]["content"])
        self.assertIn("Do not request CR in this mode", pz_only_prompt.messages[1]["content"])
        self.assertIn("Available tools: PZ and CR", pz_cr_prompt.messages[1]["content"])
        self.assertEqual(pz_only_prompt.stop_sequences, ["</tool_call>", "</answer>"])
        self.assertEqual(pz_cr_prompt.stop_sequences, ["</tool_call>", "</answer>"])

    def test_disallowed_cr_is_rejected_in_pz_only_mode(self) -> None:
        """The tool protocol must reject CR requests when the mode is pz_only."""

        raw_output = (
            "<think>\nInvalid pz_only request.\n</think>\n"
            "<tool_call>\n"
            "{\"tool_name\":\"CR\",\"arguments\":{\"policy\":\"same_category_normal\"}}\n"
            "</tool_call>"
        )
        with self.assertRaises(ToolContractError):
            parse_tool_call(raw_output, tool_path="pz_only")

    def test_malformed_tool_call_is_explicit(self) -> None:
        """Malformed tool JSON should fail loudly rather than silently continuing."""

        backend = MockToolAwareBackend(
            backend_name="mock_tool_malformed_v1",
            policy="fixture_scripted_malformed_tool_call_v1",
        )
        prompt = build_prompt(self.sample, tool_path="pz_only")
        request = BackendRequest(
            sample_id=self.sample["sample_id"],
            seed=0,
            prompt_version=prompt.prompt_version,
            messages=prompt.messages,
            stop_sequences=prompt.stop_sequences,
            tool_mode="pz_only",
        )
        response = backend.generate(request, sample=self.sample)
        with self.assertRaises(ToolContractError):
            parse_tool_call(response.raw_output, tool_path="pz_only")

    def test_tool_result_reinsertion_preserves_crop_and_reference_refs(self) -> None:
        """Tool result messages should carry image refs back into the dialogue history."""

        with tempfile.TemporaryDirectory() as tempdir:
            pz_call = parse_tool_call(
                "<tool_call>\n"
                "{\"tool_name\":\"PZ\",\"arguments\":{\"bbox\":{\"x0\":0.1,\"y0\":0.1,\"x1\":0.9,\"y1\":0.9}}}\n"
                "</tool_call>",
                tool_path="pz_only",
            )
            pz_result = execute_tool_call(
                pz_call,
                sample=self.sample,
                sample_pool=self.samples,
                artifact_dir=tempdir,
            )
            pz_history = reinsert_tool_result([], pz_result)
            self.assertEqual(len(pz_history[-1]["image_refs"]), 1)

            cr_call = parse_tool_call(
                "<tool_call>\n"
                "{\"tool_name\":\"CR\",\"arguments\":{\"policy\":\"same_category_normal\"}}\n"
                "</tool_call>",
                tool_path="pz_cr",
            )
            cr_result = execute_tool_call(
                cr_call,
                sample=self.sample,
                sample_pool=self.samples,
                artifact_dir=tempdir,
            )
            cr_history = reinsert_tool_result([], cr_result)
            self.assertEqual(len(cr_history[-1]["image_refs"]), 1)

    def test_pz_only_run_writes_tool_artifacts_and_delta(self) -> None:
        """The pz_only runner should execute PZ, export traces, and compare to baseline."""

        with tempfile.TemporaryDirectory() as tempdir:
            result = run_tool_augmented(
                config_path=PZ_ONLY_CONFIG,
                dataset_root=FIXTURE_ROOT,
                artifact_root=tempdir,
                max_samples=2,
            )
            delta_report = json.loads(Path(result["delta_report_path"]).read_text(encoding="utf-8"))
            self.assertEqual(delta_report["tool_mode"], "pz_only")
            self.assertGreater(delta_report["tool_usage_delta"]["toolcall_rate"], 0.0)
            self.assertGreater(delta_report["tool_usage_delta"]["per_tool_frequency"]["PZ"], 0.0)
            self.assertEqual(delta_report["tool_usage_delta"]["per_tool_frequency"]["CR"], 0.0)

            trace_files = sorted(Path(tempdir).glob("traces/seed_*/*.json"))
            self.assertTrue(trace_files)
            trace_payload = json.loads(trace_files[0].read_text(encoding="utf-8"))
            tool_names = [trace["tool_name"] for trace in trace_payload["tool_traces"]]
            self.assertIn("PZ", tool_names)
            self.assertNotIn("CR", tool_names)

    def test_pz_cr_run_calls_both_tools_and_exports_delta(self) -> None:
        """The pz_cr runner should call both tools and write comparison artifacts."""

        with tempfile.TemporaryDirectory() as tempdir:
            result = run_tool_augmented(
                config_path=PZ_CR_CONFIG,
                dataset_root=FIXTURE_ROOT,
                artifact_root=tempdir,
                max_samples=2,
            )
            summary = json.loads(Path(result["summary_path"]).read_text(encoding="utf-8"))
            delta_report = json.loads(Path(result["delta_report_path"]).read_text(encoding="utf-8"))
            self.assertEqual(summary["tool_mode"], "pz_cr")
            self.assertGreater(summary["tool_usage_summary"]["per_tool_counts"]["PZ"], 0)
            self.assertGreater(summary["tool_usage_summary"]["per_tool_counts"]["CR"], 0)
            self.assertGreater(delta_report["tool_usage_delta"]["per_tool_frequency"]["CR"], 0.0)

            prediction_records = result["prediction_records"]
            self.assertTrue(all(record["tool_mode"] == "pz_cr" for record in prediction_records))
            self.assertTrue(any(record["tool_usage"]["per_tool_counts"]["CR"] > 0 for record in prediction_records))


if __name__ == "__main__":
    unittest.main()
