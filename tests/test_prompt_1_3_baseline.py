"""Fixture-backed smoke tests for the Prompt 1.3 non-tool baseline path.

These tests validate the first canonical inference/evaluation pipeline without
running any real model. They exercise the no-tool prompt builder, the mock
backend round-trip, strict parsing, per-sample artifacts, metrics aggregation,
and malformed-output handling using the tiny MMAD fixture dataset.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from agentiad_recon.baseline import load_run_definition, run_baseline
from agentiad_recon.prompting import (
    BASELINE_PROMPT_VERSION,
    FINAL_ANSWER_PARSER_VERSION,
    build_baseline_prompt,
    parse_final_answer,
)


FIXTURE_ROOT = REPO_ROOT / "tests" / "fixtures" / "mmad_fixture"
CONFIG_PATH = REPO_ROOT / "configs" / "baseline_non_tool_fixture.json"


class Prompt13BaselineSmokeTests(unittest.TestCase):
    """Local-only baseline pipeline tests for Prompt 1.3."""

    def test_run_definition_is_frozen_and_non_tool(self) -> None:
        """The baseline config should stay externalized and tool-free."""

        definition = load_run_definition(CONFIG_PATH)
        self.assertEqual(definition["mode"], "no_tools")
        self.assertEqual(definition["prompt_version"], BASELINE_PROMPT_VERSION)
        self.assertEqual(definition["parser_version"], FINAL_ANSWER_PARSER_VERSION)

    def test_baseline_prompt_omits_tools(self) -> None:
        """Baseline prompts should preserve the answer contract and disable tools."""

        sample = {
            "sample_id": "sample-001",
            "category": "capsule",
            "anomaly_candidates": ["crack"],
            "image": {"uri": "/tmp/fake_image.png"},
        }
        prompt = build_baseline_prompt(sample)
        self.assertEqual(prompt.tool_path, "no_tools")
        self.assertEqual(prompt.prompt_version, BASELINE_PROMPT_VERSION)
        self.assertNotIn("<tool_call>", prompt.messages[0]["content"])
        self.assertIn("<answer>", prompt.messages[0]["content"])
        self.assertEqual(prompt.stop_sequences, ["</answer>"])

    def test_parser_accepts_baseline_answer_wrapper(self) -> None:
        """The strict parser should handle the baseline `<answer>` wrapper."""

        parsed = parse_final_answer(
            """
            <think>
            Short reasoning.
            </think>
            <answer>
              <anomaly_present>true</anomaly_present>
              <top_anomaly>crack</top_anomaly>
              <visual_descriptions>
                <item>Visible crack on the capsule.</item>
              </visual_descriptions>
            </answer>
            """
        )
        self.assertTrue(parsed["anomaly_present"])
        self.assertEqual(parsed["top_anomaly"], "crack")

    def test_baseline_run_writes_artifacts_and_metrics(self) -> None:
        """A local mock-backed run should generate auditable evidence files."""

        with tempfile.TemporaryDirectory() as tempdir:
            result = run_baseline(
                config_path=CONFIG_PATH,
                dataset_root=FIXTURE_ROOT,
                artifact_root=tempdir,
                max_samples=3,
            )
            summary_path = Path(result["summary_path"])
            manifest_path = Path(result["run_manifest_path"])
            self.assertTrue(summary_path.exists())
            self.assertTrue(manifest_path.exists())

            metrics_report = result["metrics_report"]
            self.assertEqual(len(metrics_report["per_seed_metrics"]), 2)
            self.assertGreaterEqual(metrics_report["aggregate_metrics"]["parser_valid_rate"]["std"], 0.0)

            # The per-sample prediction artifacts are written as individual JSON
            # files plus one JSONL index per seed.
            prediction_files = sorted(Path(tempdir).glob("predictions/seed_*/*.json"))
            self.assertEqual(len(prediction_files), 6)
            prediction_record = json.loads(prediction_files[0].read_text(encoding="utf-8"))
            self.assertEqual(prediction_record["tool_mode"], "no_tools")

    def test_malformed_output_is_counted_explicitly(self) -> None:
        """Odd-seed anomaly outputs should fail loudly and be counted in metrics."""

        with tempfile.TemporaryDirectory() as tempdir:
            result = run_baseline(
                config_path=CONFIG_PATH,
                dataset_root=FIXTURE_ROOT,
                artifact_root=tempdir,
                max_samples=3,
            )
            invalid_records = [
                record for record in result["prediction_records"] if not record["parser_valid"]
            ]
            self.assertEqual(len(invalid_records), 1)
            self.assertIn("Invalid anomaly_present value", invalid_records[0]["error_message"])


if __name__ == "__main__":
    unittest.main()
