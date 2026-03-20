"""Acceptance check for Prompt 2.14 deterministic evaluator sharding."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from agentiad_recon.baseline import _load_samples, _select_shard_samples, load_run_definition, run_baseline
from agentiad_recon.merge_shards import DuplicateShardSampleError, merge_prediction_jsonl_files


FIXTURE_ROOT = REPO_ROOT / "tests" / "fixtures" / "mmad_fixture"
NO_TOOLS_CONFIG = REPO_ROOT / "configs" / "baseline_non_tool_fixture.json"


def require(condition: bool, label: str, failures: list[str]) -> None:
    if condition:
        print(f"PASS: {label}")
    else:
        print(f"FAIL: {label}")
        failures.append(label)


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def score(failures: list[str]) -> str:
    if not failures:
        return "10/10"
    if len(failures) <= 2:
        return "8/10"
    if len(failures) <= 5:
        return "6/10"
    return "4/10"


def _run_python_module(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "agentiad_recon.baseline", *args],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
        env={"PYTHONPATH": str(REPO_ROOT / "src")},
    )


def main() -> int:
    failures: list[str] = []
    baseline_text = read_text(REPO_ROOT / "src/agentiad_recon/baseline.py")
    merge_text = read_text(REPO_ROOT / "src/agentiad_recon/merge_shards.py")
    readme_text = read_text(REPO_ROOT / "README.md").lower()
    worklog_text = read_text(
        REPO_ROOT
        / "Working Log/2026-03-20 Prompt 2.14 deterministic multi-gpu sharded evaluation patch.md"
    ).lower()

    for relative in [
        "src/agentiad_recon/baseline.py",
        "src/agentiad_recon/merge_shards.py",
        "tests/test_prompt_2_14_sharding.py",
        "audit/check_prompt_2_14.py",
    ]:
        require((REPO_ROOT / relative).exists(), f"required file exists: {relative}", failures)

    help_result = _run_python_module("--help")
    require(help_result.returncode == 0, "baseline.py --help succeeds", failures)
    require("--num-shards" in help_result.stdout, "--help exposes --num-shards", failures)
    require("--shard-index" in help_result.stdout, "--help exposes --shard-index", failures)

    invalid_cases = [
        (("--config", str(NO_TOOLS_CONFIG), "--dry-run", "--num-shards", "0"), "num_shards must be >= 1"),
        (("--config", str(NO_TOOLS_CONFIG), "--dry-run", "--shard-index", "-1"), "shard_index must be >= 0"),
        (
            ("--config", str(NO_TOOLS_CONFIG), "--dry-run", "--num-shards", "2", "--shard-index", "2"),
            "shard_index must be < num_shards",
        ),
    ]
    for args, expected in invalid_cases:
        result = _run_python_module(*args)
        require(result.returncode != 0, f"invalid args fail fast: {' '.join(args)}", failures)
        require(expected in (result.stderr + result.stdout), f"invalid args message: {expected}", failures)

    definition = load_run_definition(NO_TOOLS_CONFIG)
    all_samples = _load_samples(definition, dataset_root=FIXTURE_ROOT, max_samples=2)
    shard_sets = []
    union_set: set[str] = set()
    for shard_index in range(2):
        selected_samples, shard_summary = _select_shard_samples(
            all_samples,
            num_shards=2,
            shard_index=shard_index,
        )
        shard_ids = {sample["sample_id"] for sample in selected_samples}
        shard_sets.append(shard_ids)
        union_set.update(shard_ids)
        require(
            shard_summary["sharding_strategy"] == "stable_index_mod",
            f"deterministic shard strategy recorded for shard {shard_index}",
            failures,
        )
    require(union_set == {sample["sample_id"] for sample in all_samples}, "shard union equals full set", failures)
    require(not (shard_sets[0] & shard_sets[1]), "shards are pairwise disjoint", failures)

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
        require(summary["shard_summary"]["num_shards"] == 2, "summary records num_shards", failures)
        require(summary["shard_summary"]["shard_index"] == 1, "summary records shard_index", failures)
        require(
            summary["shard_summary"]["full_sample_count"] == 2
            and summary["shard_summary"]["selected_sample_count"] == 1,
            "summary records full and selected sample counts",
            failures,
        )
        require(
            manifest["run_provenance"]["sharding_strategy"] == "stable_index_mod",
            "manifest provenance records sharding strategy",
            failures,
        )

    with tempfile.TemporaryDirectory() as tempdir:
        root = Path(tempdir)
        shard_a = root / "a.jsonl"
        shard_b = root / "b.jsonl"
        output = root / "merged.jsonl"
        record = {"sample_id": "dup-sample", "seed": 0}
        shard_a.write_text(json.dumps(record) + "\n", encoding="utf-8")
        shard_b.write_text(json.dumps(record) + "\n", encoding="utf-8")
        try:
            merge_prediction_jsonl_files(input_paths=[shard_a, shard_b], output_path=output)
            duplicate_detected = False
        except DuplicateShardSampleError:
            duplicate_detected = True
        require(duplicate_detected, "merge helper rejects duplicate sample_id", failures)

    require("def run_from_config" in baseline_text, "single evaluator dispatch remains in baseline.py", failures)
    require("merge_prediction_jsonl_files" in merge_text, "merge helper is implemented", failures)
    require("prompt 2.14" in readme_text and "shard" in readme_text, "README documents Prompt 2.14", failures)
    for phrase in [
        "what changed",
        "why it changed",
        "effect of the changes",
        "file-level summary",
    ]:
        require(phrase in worklog_text, f"working log includes: {phrase}", failures)

    suite = unittest.defaultTestLoader.discover(
        str(REPO_ROOT / "tests"),
        pattern="test_prompt_2_14_sharding.py",
    )
    result = unittest.TextTestRunner(stream=sys.stdout, verbosity=1).run(suite)
    require(result.wasSuccessful(), "Prompt 2.14 regression tests pass", failures)

    print(f"Acceptance score: {score(failures)}")
    if failures:
        print("Prompt 2.14 acceptance check FAILED.")
        return 1
    print("Prompt 2.14 acceptance check PASSED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
