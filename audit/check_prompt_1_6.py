"""Lightweight acceptance check for Prompt 1.6 multi-source MMAD indexing.

This script builds a tiny temporary fixture spanning DS-MVTec, MVTec-AD,
MVTec-LOCO, VisA, and GoodsAD, then verifies that `MMADIndexer.index_samples()`
normalizes all of them into the canonical sample schema. It also reruns the
existing Prompt 1.2 smoke tests so the legacy fixture path stays intact.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from agentiad_recon.mmad import MMADIndexer, summarize_samples


LEGACY_FIXTURE_ROOT = REPO_ROOT / "tests" / "fixtures" / "mmad_fixture"


def require(condition: bool, label: str, failures: list[str]) -> None:
    """Print one acceptance result and collect failures."""

    if condition:
        print(f"PASS: {label}")
    else:
        print(f"FAIL: {label}")
        failures.append(label)


def _write_image(path: Path, color: tuple[int, int, int]) -> None:
    """Create one tiny RGB image for a temporary local fixture."""

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (4, 4), color).save(path)


def _build_multisource_fixture(root: Path) -> None:
    """Create a minimal five-source MMAD-style extracted fixture."""

    # DS-MVTec
    _write_image(root / "DS-MVTec" / "train" / "capsule" / "image" / "good" / "capsule_good_0001.png", (10, 10, 10))
    _write_image(root / "DS-MVTec" / "val" / "capsule" / "image" / "good" / "capsule_good_0002.png", (15, 15, 15))
    _write_image(root / "DS-MVTec" / "test" / "capsule" / "image" / "defective" / "capsule_def_0001.png", (200, 20, 20))
    _write_image(root / "DS-MVTec" / "test" / "capsule" / "mask" / "defective" / "capsule_def_0001.png", (255, 255, 255))

    # MVTec-AD
    _write_image(root / "MVTec-AD" / "bottle" / "train" / "good" / "bottle_good_0001.png", (20, 20, 20))
    _write_image(root / "MVTec-AD" / "bottle" / "test" / "good" / "bottle_good_0002.png", (25, 25, 25))
    _write_image(root / "MVTec-AD" / "bottle" / "test" / "crack" / "bottle_crack_0001.png", (220, 30, 30))
    _write_image(root / "MVTec-AD" / "bottle" / "ground_truth" / "crack" / "bottle_crack_0001.png", (255, 255, 255))

    # MVTec-LOCO
    _write_image(root / "MVTec-LOCO" / "breakfast_box" / "train" / "good" / "box_good_0001.png", (30, 30, 30))
    _write_image(root / "MVTec-LOCO" / "breakfast_box" / "validation" / "good" / "box_good_0002.png", (35, 35, 35))
    _write_image(root / "MVTec-LOCO" / "breakfast_box" / "test" / "logical_anomaly" / "box_logic_0001.png", (240, 40, 40))

    # VisA
    _write_image(root / "VisA" / "candle" / "train" / "good" / "candle_good_0001.png", (40, 40, 40))
    _write_image(root / "VisA" / "candle" / "test" / "good" / "candle_good_0002.png", (45, 45, 45))
    _write_image(root / "VisA" / "candle" / "test" / "bad" / "candle_bad_0001.png", (250, 50, 50))
    _write_image(root / "VisA" / "candle" / "ground_truth" / "bad" / "candle_bad_0001.png", (255, 255, 255))

    # GoodsAD
    _write_image(root / "GoodsAD" / "screw" / "train" / "good" / "screw_good_0001.png", (50, 50, 50))
    _write_image(root / "GoodsAD" / "screw" / "test" / "good" / "screw_good_0002.png", (55, 55, 55))
    _write_image(root / "GoodsAD" / "screw" / "test" / "scratch" / "screw_scratch_0001.png", (255, 60, 60))
    _write_image(root / "GoodsAD" / "screw" / "ground_truth" / "scratch" / "screw_scratch_0001.png", (255, 255, 255))

    (root / "fixture_manifest.json").write_text(
        json.dumps({"kind": "prompt_1_6_multisource_fixture"}, indent=2),
        encoding="utf-8",
    )


def main() -> int:
    """Run Prompt 1.6 file checks plus synthetic multi-source fixture validation."""

    failures: list[str] = []
    require((REPO_ROOT / "src/agentiad_recon/mmad.py").exists(), "mmad.py exists", failures)
    require((REPO_ROOT / "Working Log/2026-03-10 Prompt 1.6.md").exists(), "Prompt 1.6 working log exists", failures)

    with tempfile.TemporaryDirectory() as tempdir:
        fixture_root = Path(tempdir) / "extracted"
        _build_multisource_fixture(fixture_root)
        samples = MMADIndexer(fixture_root).index_samples()
        payloads = [sample.to_dict() for sample in samples]
        summary = summarize_samples(samples)
        per_source = Counter(sample["source"] for sample in payloads)

        print(json.dumps(summary, indent=2, sort_keys=True))

        require(set(per_source) == {"DS-MVTec", "MVTec-AD", "MVTec-LOCO", "VisA", "GoodsAD"}, "all five sources are indexed", failures)
        require(summary["DS-MVTec"]["train"] == 1 and summary["DS-MVTec"]["val"] == 1 and summary["DS-MVTec"]["test"] == 1, "DS-MVTec split normalization works", failures)
        require(summary["MVTec-LOCO"]["val"] == 1, "MVTec-LOCO validation maps to val", failures)
        require(any(sample["source"] == "VisA" and sample["anomaly_present"] for sample in payloads), "VisA bad samples become anomaly_present=true", failures)
        require(any(sample["source"] == "GoodsAD" and sample["mask"] is not None for sample in payloads if sample["anomaly_present"]), "GoodsAD anomaly masks are linked", failures)
        require(all(sample["single_agent"] for sample in payloads), "single-agent contract is preserved", failures)
        require(all(sample["allowed_tools"] == ["PZ", "CR"] for sample in payloads), "allowed tool surface is preserved", failures)

    legacy_samples = [sample.to_dict() for sample in MMADIndexer(LEGACY_FIXTURE_ROOT).index_samples()]
    require(len(legacy_samples) == 3, "legacy Prompt 1.2 fixture still indexes", failures)

    suite = unittest.defaultTestLoader.discover(
        str(REPO_ROOT / "tests"),
        pattern="test_prompt_1_2_smoke.py",
    )
    result = unittest.TextTestRunner(stream=sys.stdout, verbosity=1).run(suite)
    require(result.wasSuccessful(), "Prompt 1.2 smoke tests still pass", failures)

    if failures:
        print("\nPrompt 1.6 acceptance check FAILED.")
        for failure in failures:
            print(f"- {failure}")
        return 1

    print("\nPrompt 1.6 acceptance check PASSED.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
