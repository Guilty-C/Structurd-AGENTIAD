"""MMAD canonical sample indexing and export for the AgentIAD rebuild.

This module implements the local data waist for Prompt 1.2. It provides one
deterministic indexing path for real MMAD-style directories when they exist,
plus a tiny fixture-backed path that is explicitly marked as a local smoke-test
dataset rather than a claim about full MMAD availability.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from PIL import Image

from agentiad_recon.contracts import validate_payload
from agentiad_recon.reproducibility import sha256_file


MMAD_ROOT_ENV = "AGENTIAD_MMAD_ROOT"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".ppm", ".pgm"}
NORMAL_LABELS = {"good", "normal", "ok"}
MASK_DIR_NAMES = ("masks", "mask", "ground_truth", "gt", "roi")
SPLIT_ALIASES = {
    "train": "train",
    "training": "train",
    "val": "val",
    "valid": "val",
    "validation": "val",
    "test": "test",
    "testing": "test",
}
SPLIT_ORDER = {"train": 0, "val": 1, "test": 2}


class MMADIndexingError(ValueError):
    """Raised when the dataset root or directory layout cannot be interpreted."""


@dataclass(frozen=True)
class CanonicalSample:
    """A deterministic, schema-aligned sample record for one MMAD image."""

    sample_id: str
    split: str
    source: str
    category: str
    anomaly_present: bool
    anomaly_candidates: tuple[str, ...]
    image: dict[str, Any]
    mask: dict[str, Any] | None
    roi_source: dict[str, Any] | None
    single_agent: bool
    allowed_tools: tuple[str, ...]
    ground_truth: dict[str, Any]
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Convert the immutable dataclass into a JSON-serializable payload."""

        payload = {
            "sample_id": self.sample_id,
            "split": self.split,
            "source": self.source,
            "category": self.category,
            "anomaly_present": self.anomaly_present,
            "anomaly_candidates": list(self.anomaly_candidates),
            "image": self.image,
            "mask": self.mask,
            "roi_source": self.roi_source,
            "single_agent": self.single_agent,
            "allowed_tools": list(self.allowed_tools),
            "ground_truth": self.ground_truth,
            "metadata": self.metadata,
        }
        validate_payload(payload, "sample.schema.json")
        return payload


def discover_dataset_root(dataset_root: str | Path | None = None) -> Path:
    """Resolve the dataset root from an explicit path or environment variable."""

    candidate = dataset_root or os.environ.get(MMAD_ROOT_ENV)
    if not candidate:
        raise MMADIndexingError(
            f"MMAD root not provided. Pass dataset_root or set {MMAD_ROOT_ENV}."
        )

    root = Path(candidate).expanduser().resolve()
    if not root.exists():
        raise MMADIndexingError(f"MMAD root does not exist: {root}")
    if not root.is_dir():
        raise MMADIndexingError(f"MMAD root is not a directory: {root}")
    return root


def normalize_split_name(raw_split: str) -> str:
    """Normalize common split directory aliases to the canonical split names."""

    key = raw_split.strip().lower()
    if key not in SPLIT_ALIASES:
        raise MMADIndexingError(f"Unsupported split directory name: {raw_split}")
    return SPLIT_ALIASES[key]


def _dataset_kind(dataset_root: Path) -> str:
    """Mark fixture roots explicitly so tests do not masquerade as full data."""

    return "fixture" if (dataset_root / "fixture_manifest.json").exists() else "real"


def _sorted_image_files(directory: Path) -> list[Path]:
    """Return image files in stable lexical order from one directory."""

    return sorted(
        path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def _candidate_image_dirs(condition_dir: Path) -> Iterable[Path]:
    """Yield plausible image directories for one condition node."""

    for name in ("images", "imgs", "rgb"):
        candidate = condition_dir / name
        if candidate.is_dir():
            yield candidate
    yield condition_dir


def _resolve_mask_path(condition_dir: Path, image_path: Path) -> Path | None:
    """Look for a same-stem mask/ROI artifact in a small set of known folders."""

    candidate_dirs = [condition_dir / name for name in MASK_DIR_NAMES]
    for mask_dir in candidate_dirs:
        if not mask_dir.is_dir():
            continue
        for suffix in (image_path.suffix, ".png", ".jpg", ".jpeg", ".ppm", ".pgm"):
            candidate = mask_dir / f"{image_path.stem}{suffix}"
            if candidate.exists():
                return candidate
    return None


def _image_metadata(dataset_root: Path, image_path: Path) -> dict[str, Any]:
    """Extract lightweight metadata for one image path without changing content."""

    with Image.open(image_path) as image:
        width, height = image.size
    return {
        "uri": str(image_path.resolve()),
        "relative_path": image_path.relative_to(dataset_root).as_posix(),
        "sha256": sha256_file(image_path),
        "width": width,
        "height": height,
    }


def _mask_metadata(dataset_root: Path, mask_path: Path | None) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Build optional mask and ROI-source metadata from a resolved mask path."""

    if mask_path is None:
        return None, {"kind": "none", "description": "No mask or ROI source discovered."}

    mask_record = {
        "uri": str(mask_path.resolve()),
        "relative_path": mask_path.relative_to(dataset_root).as_posix(),
        "sha256": sha256_file(mask_path),
    }
    roi_source = {
        "kind": "mask",
        "description": "ROI source discovered from a sibling mask directory.",
        "uri": mask_record["uri"],
        "relative_path": mask_record["relative_path"],
    }
    return mask_record, roi_source


def _ground_truth_from_labels(anomaly_present: bool, condition_label: str) -> dict[str, Any]:
    """Derive the narrowest possible answer label without inventing descriptions."""

    if anomaly_present:
        return {
            "anomaly_present": True,
            "top_anomaly": condition_label,
            "visual_descriptions": [],
        }
    return {
        "anomaly_present": False,
        "top_anomaly": None,
        "visual_descriptions": [],
    }


def _sample_sort_key(record: dict[str, Any]) -> tuple[Any, ...]:
    """Keep exported samples deterministic across runs and platforms."""

    return (
        SPLIT_ORDER[record["split"]],
        record["category"],
        record["condition_label"],
        record["image_path"].relative_to(record["dataset_root"]).as_posix(),
    )


class MMADIndexer:
    """Index an MMAD-style directory tree into canonical sample records."""

    def __init__(self, dataset_root: str | Path | None = None, *, source: str = "mmad") -> None:
        self.dataset_root = discover_dataset_root(dataset_root)
        self.source = source
        self.dataset_kind = _dataset_kind(self.dataset_root)

    def index_samples(self) -> list[CanonicalSample]:
        """Scan the dataset root and return a deterministically ordered sample list."""

        raw_records: list[dict[str, Any]] = []
        category_candidates: dict[str, set[str]] = defaultdict(set)

        split_dirs = sorted(
            (
                (normalize_split_name(path.name), path)
                for path in self.dataset_root.iterdir()
                if path.is_dir() and path.name.lower() in SPLIT_ALIASES
            ),
            key=lambda item: SPLIT_ORDER[item[0]],
        )
        if not split_dirs:
            raise MMADIndexingError(f"No recognizable split directories found under {self.dataset_root}")

        for split_name, split_dir in split_dirs:
            for category_dir in sorted(path for path in split_dir.iterdir() if path.is_dir()):
                for condition_dir in sorted(path for path in category_dir.iterdir() if path.is_dir()):
                    condition_label = condition_dir.name
                    anomaly_present = condition_label.lower() not in NORMAL_LABELS
                    image_paths: list[Path] = []
                    for image_dir in _candidate_image_dirs(condition_dir):
                        image_paths = _sorted_image_files(image_dir)
                        if image_paths:
                            break
                    for image_path in image_paths:
                        if anomaly_present:
                            category_candidates[category_dir.name].add(condition_label)
                        raw_records.append(
                            {
                                "dataset_root": self.dataset_root,
                                "split": split_name,
                                "category": category_dir.name,
                                "condition_label": condition_label,
                                "anomaly_present": anomaly_present,
                                "image_path": image_path,
                                "mask_path": _resolve_mask_path(condition_dir, image_path),
                            }
                        )

        if not raw_records:
            raise MMADIndexingError(f"No MMAD-style image files found under {self.dataset_root}")

        ordered_records = sorted(raw_records, key=_sample_sort_key)
        samples: list[CanonicalSample] = []
        for record in ordered_records:
            image = _image_metadata(self.dataset_root, record["image_path"])
            mask, roi_source = _mask_metadata(self.dataset_root, record["mask_path"])
            anomaly_candidates = tuple(sorted(category_candidates.get(record["category"], set())))
            sample_id = (
                f"{self.source}:{record['split']}:{record['category']}:"
                f"{record['condition_label']}:{record['image_path'].stem}"
            )

            samples.append(
                CanonicalSample(
                    sample_id=sample_id,
                    split=record["split"],
                    source=self.source,
                    category=record["category"],
                    anomaly_present=record["anomaly_present"],
                    anomaly_candidates=anomaly_candidates,
                    image=image,
                    mask=mask,
                    roi_source=roi_source,
                    single_agent=True,
                    allowed_tools=("PZ", "CR"),
                    ground_truth=_ground_truth_from_labels(
                        record["anomaly_present"], record["condition_label"]
                    ),
                    metadata={
                        "dataset_kind": self.dataset_kind,
                        "dataset_root": str(self.dataset_root),
                        "condition_label": record["condition_label"],
                        "indexing_version": "mmad_canonical_v1",
                    },
                )
            )
        return samples


def export_canonical_samples(
    samples: Iterable[CanonicalSample],
    output_path: str | Path,
    *,
    limit: int | None = None,
) -> dict[str, Any]:
    """Export a deterministic subset of canonical samples to JSON or JSONL."""

    output = Path(output_path)
    selected = [sample.to_dict() for sample in samples]
    if limit is not None:
        selected = selected[:limit]
    if not selected:
        raise MMADIndexingError("No canonical samples available for export.")

    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix == ".jsonl":
        output.write_text(
            "\n".join(json.dumps(record, sort_keys=True) for record in selected) + "\n",
            encoding="utf-8",
        )
    elif output.suffix == ".json":
        output.write_text(json.dumps(selected, indent=2, sort_keys=True), encoding="utf-8")
    else:
        raise MMADIndexingError(f"Unsupported export suffix for {output}; use .json or .jsonl")

    manifest = {
        "schema": "agentiad/sample.schema.json",
        "sample_count": len(selected),
        "output_path": str(output.resolve()),
        "output_sha256": sha256_file(output),
        "ordering": "split/category/condition/image_path",
    }
    manifest_path = output.with_suffix(output.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest
