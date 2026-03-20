"""Prompt 2.14 tests for deterministic evaluator sharding and merge integrity."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from agentiad_recon.baseline import (
    InferenceRunError,
    _load_samples,
    _runtime_config,
    _select_shard_samples,
    load_run_definition,
    run_baseline,
)
from agentiad_recon.merge_shards import DuplicateShardSampleError, merge_prediction_jsonl_files


FIXTURE_ROOT = REPO_ROOT / "tests" / "fixtures" / "mmad_fixture"
NO_TOOLS_CONFIG = REPO_ROOT / "configs" / "baseline_non_tool_fixture.json"


class Prompt214ShardingTests(unittest.TestCase):
    """Prompt 2.14 regression tests."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.definition = load_run_definition(NO_TOOLS_CONFIG)
        cls.all_samples = _load_samples(cls.definition, dataset_root=FIXTURE_ROOT, max_samples=2)

    def test_shard_union_equals_full_set(self) -> None:
        shard_unions: set[str] = set()
        for shard_index in range(3):
            shard_samples, _summary = _select_shard_samples(
                self.all_samples,
                num_shards=3,
                shard_index=shard_index,
            )
            shard_unions.update(sample["sample_id"] for sample in shard_samples)

        self.assertEqual(shard_unions, {sample["sample_id"] for sample in self.all_samples})

    def test_shards_are_pairwise_disjoint(self) -> None:
        shard_sets = []
        for shard_index in range(3):
            shard_samples, _summary = _select_shard_samples(
                self.all_samples,
                num_shards=3,
                shard_index=shard_index,
            )
            shard_sets.append({sample["sample_id"] for sample in shard_samples})

        for left_index, left in enumerate(shard_sets):
            for right in shard_sets[left_index + 1 :]:
                self.assertFalse(left & right)

    def test_default_no_shard_behavior_is_unchanged(self) -> None:
        selected_samples, shard_summary = _select_shard_samples(
            self.all_samples,
            num_shards=1,
            shard_index=0,
        )
        self.assertEqual(
            [sample["sample_id"] for sample in selected_samples],
            [sample["sample_id"] for sample in self.all_samples],
        )
        self.assertEqual(shard_summary["full_sample_count"], len(self.all_samples))
        self.assertEqual(shard_summary["selected_sample_count"], len(self.all_samples))

    def test_invalid_shard_args_are_rejected(self) -> None:
        for overrides, expected_message in [
            ({"num_shards": 0}, "num_shards must be >= 1"),
            ({"shard_index": -1}, "shard_index must be >= 0"),
            ({"num_shards": 2, "shard_index": 2}, "shard_index must be < num_shards"),
        ]:
            with self.subTest(overrides=overrides), self.assertRaisesRegex(
                InferenceRunError,
                expected_message,
            ):
                _runtime_config(
                    self.definition,
                    dataset_root=FIXTURE_ROOT,
                    runtime_overrides=overrides,
                )

    def test_summary_and_provenance_include_shard_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            result = run_baseline(
                config_path=NO_TOOLS_CONFIG,
                dataset_root=FIXTURE_ROOT,
                artifact_root=tempdir,
                max_samples=2,
                runtime_overrides={"num_shards": 2, "shard_index": 1},
            )
            summary = json.loads(Path(result["summary_path"]).read_text(encoding="utf-8"))
            manifest = json.loads(Path(result["run_manifest_path"]).read_text(encoding="utf-8"))

        self.assertEqual(summary["shard_summary"]["num_shards"], 2)
        self.assertEqual(summary["shard_summary"]["shard_index"], 1)
        self.assertEqual(summary["shard_summary"]["full_sample_count"], 2)
        self.assertEqual(summary["shard_summary"]["selected_sample_count"], 1)
        self.assertEqual(summary["runtime_provenance"]["sharding_strategy"], "stable_index_mod")
        self.assertEqual(manifest["shard_summary"]["selected_sample_count"], 1)
        self.assertEqual(result["runtime_provenance"]["full_sample_count"], 2)

    def test_merge_helper_rejects_duplicate_sample_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            shard_a = root / "shard_a.jsonl"
            shard_b = root / "shard_b.jsonl"
            output = root / "merged.jsonl"
            duplicate_record = {"sample_id": "dup-sample", "seed": 0, "prediction": None}
            shard_a.write_text(json.dumps(duplicate_record) + "\n", encoding="utf-8")
            shard_b.write_text(json.dumps(duplicate_record) + "\n", encoding="utf-8")

            with self.assertRaises(DuplicateShardSampleError):
                merge_prediction_jsonl_files(
                    input_paths=[shard_a, shard_b],
                    output_path=output,
                )
