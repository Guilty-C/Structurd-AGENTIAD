"""Small integrity-first helper for merging sharded prediction JSONL files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


class DuplicateShardSampleError(ValueError):
    """Raised when shard prediction files contain duplicate sample IDs."""

    def __init__(self, summary: dict[str, Any]) -> None:
        super().__init__(
            "Duplicate sample_id values detected while merging shards: "
            + ", ".join(summary["duplicate_sample_ids"])
        )
        self.summary = summary


def merge_prediction_jsonl_files(
    *,
    input_paths: list[str | Path],
    output_path: str | Path,
) -> dict[str, Any]:
    """Merge shard prediction JSONL files and fail fast on duplicate sample IDs."""

    output_path = Path(output_path)
    input_paths = [Path(path) for path in input_paths]
    merged_records: list[dict[str, Any]] = []
    seen_sample_ids: set[str] = set()
    duplicate_sample_ids: list[str] = []

    for path in input_paths:
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            if not raw_line.strip():
                continue
            record = json.loads(raw_line)
            sample_id = str(record["sample_id"])
            if sample_id in seen_sample_ids:
                duplicate_sample_ids.append(sample_id)
                continue
            seen_sample_ids.add(sample_id)
            merged_records.append(record)

    summary = {
        "records": len(merged_records),
        "unique_sample_ids": len(seen_sample_ids),
        "duplicate_count": len(duplicate_sample_ids),
        "duplicate_sample_ids": sorted(set(duplicate_sample_ids)),
        "output_path": str(output_path.resolve()),
    }
    if duplicate_sample_ids:
        raise DuplicateShardSampleError(summary)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in merged_records),
        encoding="utf-8",
    )
    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Merge sharded prediction JSONL files and reject duplicate sample_id values."
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="Prediction JSONL files from individual shards.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Merged output JSONL path.",
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    try:
        summary = merge_prediction_jsonl_files(input_paths=args.inputs, output_path=args.output)
    except DuplicateShardSampleError as exc:
        payload = dict(exc.summary)
        payload["error"] = str(exc)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 1
    except ValueError as exc:
        print(json.dumps({"error": str(exc)}, indent=2, sort_keys=True))
        return 1
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
