"""Baseline artifact writing and metric aggregation helpers for AgentIAD.

This module keeps evaluation output handling small and auditable. Prompt 1.3
uses it to validate per-sample prediction artifacts, per-seed and per-class
metrics, plus mean/std aggregation without inventing a second pipeline stack.
There is no standalone CLI for this module.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

from agentiad_recon.contracts import validate_payload


RATE_KEYS = (
    "parser_valid_rate",
    "schema_valid_rate",
    "anomaly_present_accuracy",
    "top_anomaly_accuracy",
)


def safe_slug(value: str) -> str:
    """Convert a sample identifier into a stable filesystem-friendly slug."""

    return re.sub(r"[^A-Za-z0-9._-]+", "_", value)


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    """Write one deterministic JSON artifact to disk."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def write_jsonl(path: str | Path, payloads: list[dict[str, Any]]) -> None:
    """Write one deterministic JSONL artifact to disk."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(payload, sort_keys=True) for payload in payloads]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_prediction_record(
    *,
    run_id: str,
    sample: dict[str, Any],
    seed: int,
    prompt_version: str,
    parser_version: str,
    backend_name: str,
    prediction: dict[str, Any] | None,
    parser_valid: bool,
    schema_valid: bool,
    error_message: str | None,
    raw_output_path: str,
    raw_output_sha256: str,
    trace_path: str,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """Build and validate one per-sample baseline prediction record."""

    record = {
        "record_id": f"{run_id}:{seed}:{sample['sample_id']}",
        "run_id": run_id,
        "sample_id": sample["sample_id"],
        "split": sample["split"],
        "category": sample["category"],
        "seed": seed,
        "tool_mode": "no_tools",
        "prompt_version": prompt_version,
        "parser_version": parser_version,
        "backend_name": backend_name,
        "parser_valid": parser_valid,
        "schema_valid": schema_valid,
        "prediction": prediction,
        "ground_truth": sample["ground_truth"],
        "error_message": error_message,
        "raw_output_path": raw_output_path,
        "raw_output_sha256": raw_output_sha256,
        "trace_path": trace_path,
        "metadata": metadata,
    }
    validate_payload(record, "baseline_prediction_record.schema.json")
    return record


def _safe_rate(numerator: int, denominator: int) -> float:
    """Return a stable rate and avoid division-by-zero crashes in small tests."""

    if denominator == 0:
        return 0.0
    return numerator / denominator


def _metric_record(
    records: list[dict[str, Any]],
    *,
    scope: str,
    scope_name: str,
    seed: int | None = None,
    category: str | None = None,
) -> dict[str, Any]:
    """Compute one metric record for either a seed scope or a class scope."""

    sample_count = len(records)
    parser_valid_count = sum(1 for record in records if record["parser_valid"])
    schema_valid_count = sum(1 for record in records if record["schema_valid"])
    anomaly_present_correct = 0
    top_anomaly_correct = 0
    for record in records:
        prediction = record["prediction"]
        ground_truth = record["ground_truth"]
        if prediction is None:
            continue
        # Invalid parser outputs count against accuracy while still remaining
        # visible in separate validity counters.
        if prediction["anomaly_present"] == ground_truth["anomaly_present"]:
            anomaly_present_correct += 1
        if prediction["top_anomaly"] == ground_truth["top_anomaly"]:
            top_anomaly_correct += 1

    return {
        "scope": scope,
        "scope_name": scope_name,
        "seed": seed,
        "category": category,
        "sample_count": sample_count,
        "parser_valid_count": parser_valid_count,
        "schema_valid_count": schema_valid_count,
        "parser_valid_rate": _safe_rate(parser_valid_count, sample_count),
        "schema_valid_rate": _safe_rate(schema_valid_count, sample_count),
        "anomaly_present_accuracy": _safe_rate(anomaly_present_correct, sample_count),
        "top_anomaly_accuracy": _safe_rate(top_anomaly_correct, sample_count),
    }


def build_metrics_report(
    *,
    run_id: str,
    prompt_version: str,
    parser_version: str,
    backend_name: str,
    prediction_records: list[dict[str, Any]],
    seeds: list[int],
) -> dict[str, Any]:
    """Aggregate per-seed, per-class, and mean/std baseline metrics."""

    per_seed_metrics = [
        _metric_record(
            [record for record in prediction_records if record["seed"] == seed],
            scope="seed",
            scope_name=f"seed_{seed}",
            seed=seed,
        )
        for seed in seeds
    ]

    categories = sorted({record["category"] for record in prediction_records})
    per_class_metrics = [
        _metric_record(
            [record for record in prediction_records if record["category"] == category],
            scope="class",
            scope_name=category,
            category=category,
        )
        for category in categories
    ]

    aggregate_metrics = {}
    for key in RATE_KEYS:
        values = [metric[key] for metric in per_seed_metrics]
        # Population std keeps the tiny local smoke runs deterministic and
        # avoids inflating variance when only a few seeds are present.
        aggregate_metrics[key] = {
            "mean": mean(values),
            "std": pstdev(values) if len(values) > 1 else 0.0,
        }

    report = {
        "run_id": run_id,
        "tool_mode": "no_tools",
        "prompt_version": prompt_version,
        "parser_version": parser_version,
        "backend_name": backend_name,
        "seeds": seeds,
        "per_seed_metrics": per_seed_metrics,
        "per_class_metrics": per_class_metrics,
        "aggregate_metrics": aggregate_metrics,
    }
    validate_payload(report, "baseline_metrics_report.schema.json")
    return report


def build_run_summary(
    *,
    run_id: str,
    backend_name: str,
    prompt_version: str,
    parser_version: str,
    execution_boundary: str,
    sample_source: dict[str, Any],
    seeds: list[int],
    prediction_records: list[dict[str, Any]],
    artifact_paths: dict[str, str],
    notes: list[str] | None = None,
) -> dict[str, Any]:
    """Create one compact summary manifest for the baseline evidence package."""

    parser_valid = sum(1 for record in prediction_records if record["parser_valid"])
    schema_valid = sum(1 for record in prediction_records if record["schema_valid"])
    sample_count = len(prediction_records)
    summary = {
        "run_id": run_id,
        "tool_mode": "no_tools",
        "backend_name": backend_name,
        "prompt_version": prompt_version,
        "parser_version": parser_version,
        "execution_boundary": execution_boundary,
        "sample_source": sample_source,
        "seeds": seeds,
        "sample_count": sample_count,
        "parser_validity": {
            "valid": parser_valid,
            "invalid": sample_count - parser_valid,
        },
        "schema_validity": {
            "valid": schema_valid,
            "invalid": sample_count - schema_valid,
        },
        "artifact_paths": artifact_paths,
        "notes": notes or [],
    }
    validate_payload(summary, "baseline_run_summary.schema.json")
    return summary
