"""MMAD canonical sample indexing and export for the AgentIAD rebuild.

This module is the single dataset-indexing waist used by the Prompt 1.2 through
Prompt 1.5 layers. Prompt 1.6 extends it from one generic split/category layout
to a multi-source indexer that can recurse through MMAD `extracted/` trees and
normalize DS-MVTec, MVTec-AD, MVTec-LOCO, VisA, and GoodsAD into the same
canonical sample contract. Use `python -m agentiad_recon.mmad --help` for a
small local smoke CLI that only counts and exports indexed samples.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from PIL import Image

from agentiad_recon.contracts import validate_payload
from agentiad_recon.reproducibility import sha256_file


MMAD_ROOT_ENV = "AGENTIAD_MMAD_ROOT"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".ppm", ".pgm", ".tif", ".tiff"}
NORMAL_LABELS = {"good", "normal", "ok"}
ANOMALY_ALIAS_MAP = {"defective": "defective", "bad": "bad"}
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
KNOWN_SOURCES = {
    "ds-mvtec": "DS-MVTec",
    "mvtec-ad": "MVTec-AD",
    "mvtec-loco": "MVTec-LOCO",
    "visa": "VisA",
    "goodsad": "GoodsAD",
}
SOURCE_FORMAT_VERSION = {
    "DS-MVTec": "ds_mvtec_v1",
    "MVTec-AD": "mvtec_ad_v1",
    "MVTec-LOCO": "mvtec_loco_v1",
    "VisA": "visa_v1",
    "GoodsAD": "goodsad_v1",
    "legacy": "legacy_split_category_condition_v1",
}


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

    return "fixture" if list(dataset_root.rglob("fixture_manifest.json")) else "real"


def _sorted_image_files(directory: Path) -> list[Path]:
    """Return image files in stable lexical order from one directory."""

    if not directory.is_dir():
        return []
    return sorted(
        path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def _normalize_condition_label(raw_label: str) -> str:
    """Normalize anomaly labels while preserving the canonical good/null semantics."""

    lowered = raw_label.strip().lower()
    if lowered in NORMAL_LABELS:
        return "good"
    return ANOMALY_ALIAS_MAP.get(lowered, raw_label.strip())


def _is_normal_label(label: str) -> bool:
    """Return whether one condition label should be treated as the normal class."""

    return _normalize_condition_label(label) == "good"


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


def _mask_metadata(dataset_root: Path, mask_path: Path | None, *, mask_type: str | None) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Build optional mask and ROI-source metadata from a resolved mask path."""

    if mask_path is None:
        return None, {"kind": "none", "description": "No mask or ROI source discovered."}

    mask_record = {
        "uri": str(mask_path.resolve()),
        "relative_path": mask_path.relative_to(dataset_root).as_posix(),
        "sha256": sha256_file(mask_path),
    }
    roi_source = {
        "kind": mask_type or "mask",
        "description": f"ROI source discovered from a {mask_type or 'mask'} directory.",
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
        record["source_name"],
        SPLIT_ORDER[record["split"]],
        record["category"],
        record["condition_label"],
        record["image_path"].relative_to(record["dataset_root"]).as_posix(),
    )


def _find_mask_in_directory(mask_dir: Path, image_path: Path) -> Path | None:
    """Look for a same-stem mask in one explicit directory."""

    if not mask_dir.is_dir():
        return None
    candidates = [image_path.stem, image_path.name]
    for stem_or_name in candidates:
        for suffix in (image_path.suffix, ".png", ".jpg", ".jpeg", ".bmp", ".ppm", ".pgm", ".tif", ".tiff"):
            candidate = mask_dir / f"{Path(stem_or_name).stem}{suffix}"
            if candidate.exists():
                return candidate
    for candidate in sorted(mask_dir.iterdir()):
        if candidate.is_file() and candidate.suffix.lower() in IMAGE_SUFFIXES and candidate.stem.startswith(image_path.stem):
            return candidate
    return None


def _resolve_legacy_mask_path(condition_dir: Path, image_path: Path) -> tuple[Path | None, str | None]:
    """Look for a same-stem mask/ROI artifact in the generic legacy layout."""

    for name in ("masks", "mask", "ground_truth", "gt", "roi"):
        candidate = _find_mask_in_directory(condition_dir / name, image_path)
        if candidate is not None:
            return candidate, "mask"
    return None, None


def _known_source_name(path: Path) -> str | None:
    """Normalize one directory name to a supported benchmark source label."""

    key = path.name.strip().lower()
    return KNOWN_SOURCES.get(key)


def _iter_source_roots(dataset_root: Path) -> list[tuple[str, Path]]:
    """Return supported source roots or an empty list when using the legacy layout."""

    source_roots = [
        (source_name, path)
        for path in sorted(dataset_root.iterdir())
        if path.is_dir() and (source_name := _known_source_name(path)) is not None
    ]
    if source_roots:
        return source_roots

    root_source = _known_source_name(dataset_root)
    if root_source is not None:
        return [(root_source, dataset_root)]
    return []


def _collect_generic_records(
    dataset_root: Path,
    *,
    source_name: str,
    source_format: str,
    split_dir: Path,
) -> list[dict[str, Any]]:
    """Index the generic split/category/condition layout used by the Prompt 1.2 fixture."""

    split_name = normalize_split_name(split_dir.name)
    records: list[dict[str, Any]] = []
    for category_dir in sorted(path for path in split_dir.iterdir() if path.is_dir()):
        for condition_dir in sorted(path for path in category_dir.iterdir() if path.is_dir()):
            condition_label = _normalize_condition_label(condition_dir.name)
            anomaly_present = not _is_normal_label(condition_label)
            image_paths: list[Path] = []
            for image_dir in (condition_dir / "images", condition_dir / "imgs", condition_dir / "rgb", condition_dir):
                image_paths = _sorted_image_files(image_dir)
                if image_paths:
                    break
            mask_type = None
            for image_path in image_paths:
                mask_path, mask_type = _resolve_legacy_mask_path(condition_dir, image_path)
                records.append(
                    {
                        "dataset_root": dataset_root,
                        "source_name": source_name,
                        "source_format": source_format,
                        "split": split_name,
                        "category": category_dir.name,
                        "condition_label": condition_label,
                        "anomaly_present": anomaly_present,
                        "image_path": image_path,
                        "mask_path": mask_path,
                        "mask_type": mask_type,
                        "split_origin": split_dir.name,
                    }
                )
    return records


def _records_from_train_test_category_layout(
    dataset_root: Path,
    *,
    source_name: str,
    source_root: Path,
    validation_like_names: set[str] | None = None,
    test_requires_ground_truth: bool = True,
) -> list[dict[str, Any]]:
    """Index train/validation/test style category layouts used by several sources.

    The supported forms are:
    - `<category>/<split>/<condition>/<image>`
    - `<category>/<split>/<condition>/images/<image>`
    - optional `<category>/ground_truth/<condition>/<mask>`
    """

    validation_like_names = validation_like_names or set()
    records: list[dict[str, Any]] = []
    split_names = {"train", "test", "validation"} | validation_like_names

    for category_dir in sorted(path for path in source_root.iterdir() if path.is_dir() and path.name.lower() not in {"ground_truth", "mask"}):
        ground_truth_root = category_dir / "ground_truth"
        for split_dir in sorted(path for path in category_dir.iterdir() if path.is_dir() and path.name.lower() in split_names):
            split_name = normalize_split_name(split_dir.name)
            for condition_dir in sorted(path for path in split_dir.iterdir() if path.is_dir()):
                condition_label = _normalize_condition_label(condition_dir.name)
                anomaly_present = not _is_normal_label(condition_label)
                image_paths: list[Path] = []
                for image_dir in (condition_dir / "images", condition_dir / "imgs", condition_dir / "rgb", condition_dir):
                    image_paths = _sorted_image_files(image_dir)
                    if image_paths:
                        break

                for image_path in image_paths:
                    mask_path = None
                    mask_type = None
                    if anomaly_present and (split_name != "train" or test_requires_ground_truth):
                        mask_path = _find_mask_in_directory(ground_truth_root / condition_dir.name, image_path)
                        mask_type = "mask" if mask_path is not None else None
                    records.append(
                        {
                            "dataset_root": dataset_root,
                            "source_name": source_name,
                            "source_format": SOURCE_FORMAT_VERSION[source_name],
                            "split": split_name,
                            "category": category_dir.name,
                            "condition_label": condition_label,
                            "anomaly_present": anomaly_present,
                            "image_path": image_path,
                            "mask_path": mask_path,
                            "mask_type": mask_type,
                            "split_origin": split_dir.name,
                        }
                    )
    return records


def _records_from_ds_mvtec(
    dataset_root: Path,
    *,
    source_root: Path,
    source_name: str,
) -> list[dict[str, Any]]:
    """Index DS-MVTec with explicit `image/` and `mask/` branches.

    Supported DS-MVTec forms:
    - `<category>/<split>/image/<condition>/<image>` and `<category>/<split>/mask/<condition>/<mask>`
    - `<split>/<category>/image/<condition>/<image>` and `<split>/<category>/mask/<condition>/<mask>`
    - `<category>/image/<condition>/<image>` as a local smoke fallback when no split dir is present
    """

    records: list[dict[str, Any]] = []

    def collect_category(category_dir: Path, *, split_name: str | None) -> None:
        image_root = category_dir / "image"
        mask_root = category_dir / "mask"
        if not image_root.is_dir():
            return
        effective_split = split_name or "test"
        split_origin = split_name or "implicit_test"
        for condition_dir in sorted(path for path in image_root.iterdir() if path.is_dir()):
            condition_label = _normalize_condition_label(condition_dir.name)
            anomaly_present = not _is_normal_label(condition_label)
            for image_path in _sorted_image_files(condition_dir):
                mask_path = None
                mask_type = None
                if anomaly_present:
                    mask_path = _find_mask_in_directory(mask_root / condition_dir.name, image_path)
                    mask_type = "mask" if mask_path is not None else None
                records.append(
                    {
                        "dataset_root": dataset_root,
                        "source_name": source_name,
                        "source_format": SOURCE_FORMAT_VERSION[source_name],
                        "split": effective_split,
                        "category": category_dir.name,
                        "condition_label": condition_label,
                        "anomaly_present": anomaly_present,
                        "image_path": image_path,
                        "mask_path": mask_path,
                        "mask_type": mask_type,
                        "split_origin": split_origin,
                    }
                )

    split_root_seen = False
    for split_dir in sorted(path for path in source_root.iterdir() if path.is_dir() and path.name.lower() in SPLIT_ALIASES):
        split_root_seen = True
        split_name = normalize_split_name(split_dir.name)
        for category_dir in sorted(path for path in split_dir.iterdir() if path.is_dir()):
            collect_category(category_dir, split_name=split_name)

    if split_root_seen:
        return records

    for category_dir in sorted(path for path in source_root.iterdir() if path.is_dir()):
        collect_category(category_dir, split_name=None)
    return records


def _records_from_source(
    dataset_root: Path,
    *,
    source_root: Path,
    source_name: str,
) -> list[dict[str, Any]]:
    """Dispatch one known source root to its supported directory parser."""

    if source_name == "DS-MVTec":
        return _records_from_ds_mvtec(dataset_root, source_root=source_root, source_name=source_name)
    if source_name == "MVTec-AD":
        return _records_from_train_test_category_layout(
            dataset_root,
            source_name=source_name,
            source_root=source_root,
        )
    if source_name == "MVTec-LOCO":
        return _records_from_train_test_category_layout(
            dataset_root,
            source_name=source_name,
            source_root=source_root,
            validation_like_names={"validation"},
            test_requires_ground_truth=False,
        )
    if source_name == "VisA":
        return _records_from_train_test_category_layout(
            dataset_root,
            source_name=source_name,
            source_root=source_root,
        )
    if source_name == "GoodsAD":
        return _records_from_train_test_category_layout(
            dataset_root,
            source_name=source_name,
            source_root=source_root,
        )
    raise MMADIndexingError(f"Unsupported source name: {source_name}")


def summarize_samples(samples: Iterable[CanonicalSample | dict[str, Any]]) -> dict[str, dict[str, int]]:
    """Summarize indexed samples by source and split for local smoke printing."""

    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for sample in samples:
        payload = sample.to_dict() if isinstance(sample, CanonicalSample) else sample
        counts[payload["source"]][payload["split"]] += 1
    return {
        source: {
            split: split_counts.get(split, 0)
            for split in ("train", "val", "test")
        }
        for source, split_counts in sorted(counts.items())
    }


class MMADIndexer:
    """Index an MMAD or multi-source MMAD-extracted directory into canonical samples.

    Split detection rules:
    - legacy fixture layout: `<split>/<category>/<condition>/images`
    - DS-MVTec: explicit `image/` and `mask/` branches, optionally nested under a split directory
    - MVTec-AD, MVTec-LOCO, VisA, GoodsAD: `<category>/<split>/<condition>` plus optional `ground_truth`
    - all split variants are normalized to `train`, `val`, and `test`
    """

    def __init__(self, dataset_root: str | Path | None = None, *, source: str = "mmad") -> None:
        self.dataset_root = discover_dataset_root(dataset_root)
        self.source = source
        self.dataset_kind = _dataset_kind(self.dataset_root)

    def _collect_raw_records(self) -> list[dict[str, Any]]:
        """Collect raw source-aware records before canonical sample construction."""

        source_roots = _iter_source_roots(self.dataset_root)
        if source_roots:
            raw_records: list[dict[str, Any]] = []
            for source_name, source_root in source_roots:
                raw_records.extend(
                    _records_from_source(
                        self.dataset_root,
                        source_root=source_root,
                        source_name=source_name,
                    )
                )
            return raw_records

        split_dirs = sorted(
            (
                path for path in self.dataset_root.iterdir()
                if path.is_dir() and path.name.lower() in SPLIT_ALIASES
            ),
            key=lambda path: SPLIT_ORDER[normalize_split_name(path.name)],
        )
        if not split_dirs:
            raise MMADIndexingError(
                f"No recognizable split directories or supported source roots found under {self.dataset_root}"
            )

        raw_records: list[dict[str, Any]] = []
        legacy_source_name = self.source
        for split_dir in split_dirs:
            raw_records.extend(
                _collect_generic_records(
                    self.dataset_root,
                    source_name=legacy_source_name,
                    source_format=SOURCE_FORMAT_VERSION["legacy"],
                    split_dir=split_dir,
                )
            )
        return raw_records

    def index_samples(self) -> list[CanonicalSample]:
        """Scan the dataset root and return a deterministically ordered sample list."""

        raw_records = self._collect_raw_records()
        if not raw_records:
            raise MMADIndexingError(f"No MMAD-style image files found under {self.dataset_root}")

        category_candidates: dict[tuple[str, str], set[str]] = defaultdict(set)
        for record in raw_records:
            if record["anomaly_present"]:
                category_candidates[(record["source_name"], record["category"])].add(record["condition_label"])

        ordered_records = sorted(raw_records, key=_sample_sort_key)
        samples: list[CanonicalSample] = []
        for record in ordered_records:
            image = _image_metadata(self.dataset_root, record["image_path"])
            mask, roi_source = _mask_metadata(
                self.dataset_root,
                record["mask_path"],
                mask_type=record["mask_type"],
            )
            anomaly_candidates = tuple(
                sorted(category_candidates.get((record["source_name"], record["category"]), set()))
            )
            sample_id = (
                f"{record['source_name']}:{record['split']}:{record['category']}:"
                f"{record['condition_label']}:{record['image_path'].stem}"
            )

            sample = CanonicalSample(
                sample_id=sample_id,
                split=record["split"],
                source=record["source_name"],
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
                    "indexing_version": "mmad_canonical_v1_6",
                    "source_format": record["source_format"],
                    "split_origin": record["split_origin"],
                    "mask_type": record["mask_type"],
                },
            )
            sample.to_dict()
            samples.append(sample)
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
        "ordering": "source/split/category/condition/image_path",
    }
    manifest_path = output.with_suffix(output.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def _build_parser() -> argparse.ArgumentParser:
    """Build the small local smoke CLI for multi-source indexing checks."""

    parser = argparse.ArgumentParser(
        description="Index canonical MMAD samples from a legacy or multi-source extracted root."
    )
    parser.add_argument("--dataset-root", required=True, help="Dataset root or extracted/ root to index.")
    parser.add_argument("--source", default="mmad", help="Legacy source label override when not using known source roots.")
    parser.add_argument("--limit", type=int, help="Optional sample print limit.")
    parser.add_argument("--export-jsonl", help="Optional output path for canonical JSONL export.")
    return parser


def main() -> int:
    """Run the local smoke CLI and print counts plus a small sample preview."""

    args = _build_parser().parse_args()
    samples = MMADIndexer(args.dataset_root, source=args.source).index_samples()
    payload: dict[str, Any] = {
        "sample_count": len(samples),
        "counts_by_source_and_split": summarize_samples(samples),
        "preview": [sample.to_dict() for sample in samples[: args.limit or 3]],
    }
    if args.export_jsonl:
        payload["export_manifest"] = export_canonical_samples(samples, args.export_jsonl)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
