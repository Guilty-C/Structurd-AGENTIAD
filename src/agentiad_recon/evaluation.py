"""Artifact writing and metric aggregation helpers for AgentIAD inference.

This module keeps evaluation output handling small and auditable across the
baseline and tool-augmented inference paths. Prompt 1.4 extends the same
metrics family with tool-usage statistics and delta-vs-baseline reporting
without inventing a second evaluator stack. There is no standalone CLI here.
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
    "toolcall_rate",
)
TOOL_NAMES = ("PZ", "CR")


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


def empty_tool_usage() -> dict[str, Any]:
    """Return a normalized zero-tool-usage payload for no-tool runs."""

    return {
        "total_calls": 0,
        "samples_with_tool_call": 0,
        "per_tool_counts": {tool: 0 for tool in TOOL_NAMES},
    }


def build_prediction_record(
    *,
    run_id: str,
    sample: dict[str, Any],
    seed: int,
    tool_mode: str,
    tool_usage: dict[str, Any],
    prompt_version: str,
    parser_version: str,
    backend_name: str,
    prediction: dict[str, Any] | None,
    parser_valid: bool,
    schema_valid: bool,
    error_message: str | None,
    failure_reason: str | None,
    raw_output_path: str,
    raw_output_sha256: str,
    trace_path: str,
    first_protocol_event_type: str,
    first_assistant_output_terminal: bool,
    tool_call_count: int,
    called_tools: list[str],
    terminal_without_tool_call: bool,
    terminal_false_null_without_tool_call: bool,
    terminal_answer_turn_index: int | None,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """Build and validate one per-sample prediction record."""

    record = {
        "record_id": f"{run_id}:{seed}:{sample['sample_id']}",
        "run_id": run_id,
        "sample_id": sample["sample_id"],
        "split": sample["split"],
        "category": sample["category"],
        "seed": seed,
        "tool_mode": tool_mode,
        "tool_usage": tool_usage,
        "prompt_version": prompt_version,
        "parser_version": parser_version,
        "backend_name": backend_name,
        "parser_valid": parser_valid,
        "schema_valid": schema_valid,
        "prediction": prediction,
        "ground_truth": sample["ground_truth"],
        "error_message": error_message,
        "failure_reason": failure_reason,
        "raw_output_path": raw_output_path,
        "raw_output_sha256": raw_output_sha256,
        "trace_path": trace_path,
        "first_protocol_event_type": first_protocol_event_type,
        "first_assistant_output_terminal": first_assistant_output_terminal,
        "tool_call_count": tool_call_count,
        "called_tools": called_tools,
        "terminal_without_tool_call": terminal_without_tool_call,
        "terminal_false_null_without_tool_call": terminal_false_null_without_tool_call,
        "terminal_answer_turn_index": terminal_answer_turn_index,
        "metadata": metadata,
    }
    validate_payload(record, "baseline_prediction_record.schema.json")
    return record


def _safe_rate(numerator: int, denominator: int) -> float:
    """Return a stable rate and avoid division-by-zero crashes in small tests."""

    if denominator == 0:
        return 0.0
    return numerator / denominator


def _tool_counts(records: list[dict[str, Any]]) -> dict[str, int]:
    """Sum per-tool call counts across a scope of prediction records."""

    return {
        tool: sum(record["tool_usage"]["per_tool_counts"][tool] for record in records)
        for tool in TOOL_NAMES
    }


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
    samples_with_tool_calls = sum(record["tool_usage"]["samples_with_tool_call"] for record in records)
    per_tool_counts = _tool_counts(records)
    anomaly_present_correct = 0
    top_anomaly_correct = 0
    for record in records:
        prediction = record["prediction"]
        ground_truth = record["ground_truth"]
        if prediction is None:
            continue
        # Invalid parser outputs count against accuracy while remaining visible
        # in the explicit validity counters.
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
        "toolcall_rate": _safe_rate(samples_with_tool_calls, sample_count),
        "per_tool_frequency": {
            tool: _safe_rate(per_tool_counts[tool], sample_count) for tool in TOOL_NAMES
        },
    }


def build_metrics_report(
    *,
    run_id: str,
    tool_mode: str,
    prompt_version: str,
    parser_version: str,
    backend_name: str,
    prediction_records: list[dict[str, Any]],
    seeds: list[int],
    runtime_provenance: dict[str, Any],
    tool_first_intervention_strategy: str | None = None,
    prompt_audit_summary: dict[str, int] | None = None,
    zero_tool_behavior_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Aggregate per-seed, per-class, and mean/std inference metrics."""

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

    aggregate_metrics: dict[str, Any] = {}
    for key in RATE_KEYS:
        values = [metric[key] for metric in per_seed_metrics]
        # Population std keeps the tiny local smoke runs deterministic and
        # avoids inflating variance when only a few seeds are present.
        aggregate_metrics[key] = {
            "mean": mean(values),
            "std": pstdev(values) if len(values) > 1 else 0.0,
        }

    aggregate_metrics["per_tool_frequency"] = {
        tool: {
            "mean": mean([metric["per_tool_frequency"][tool] for metric in per_seed_metrics]),
            "std": (
                pstdev([metric["per_tool_frequency"][tool] for metric in per_seed_metrics])
                if len(per_seed_metrics) > 1
                else 0.0
            ),
        }
        for tool in TOOL_NAMES
    }

    sample_count = len(prediction_records)
    failed_count = sum(
        1
        for record in prediction_records
        if not record["parser_valid"] or not record["schema_valid"] or record["prediction"] is None
    )
    report = {
        "run_id": run_id,
        "tool_mode": tool_mode,
        "prompt_version": prompt_version,
        "parser_version": parser_version,
        "backend_name": backend_name,
        "seeds": seeds,
        "sample_count": sample_count,
        "evaluated_count": sample_count - failed_count,
        "failed_count": failed_count,
        "runtime_provenance": runtime_provenance,
        "per_seed_metrics": per_seed_metrics,
        "per_class_metrics": per_class_metrics,
        "aggregate_metrics": aggregate_metrics,
    }
    if tool_first_intervention_strategy is not None:
        report["tool_first_intervention_strategy"] = tool_first_intervention_strategy
    if prompt_audit_summary is not None:
        report["prompt_audit_summary"] = prompt_audit_summary
    if zero_tool_behavior_summary is not None:
        report["zero_tool_behavior_summary"] = zero_tool_behavior_summary
    validate_payload(report, "baseline_metrics_report.schema.json")
    return report


def summarize_tool_usage(prediction_records: list[dict[str, Any]]) -> dict[str, Any]:
    """Collapse per-sample tool usage into one run-level summary."""

    total_calls = sum(record["tool_usage"]["total_calls"] for record in prediction_records)
    samples_with_tool_calls = sum(
        record["tool_usage"]["samples_with_tool_call"] for record in prediction_records
    )
    return {
        "total_calls": total_calls,
        "samples_with_tool_calls": samples_with_tool_calls,
        "per_tool_counts": _tool_counts(prediction_records),
    }


def build_run_summary(
    *,
    run_id: str,
    tool_mode: str,
    backend_name: str,
    prompt_version: str,
    parser_version: str,
    execution_boundary: str,
    sample_source: dict[str, Any],
    seeds: list[int],
    prediction_records: list[dict[str, Any]],
    runtime_provenance: dict[str, Any],
    artifact_paths: dict[str, str],
    notes: list[str] | None = None,
    tool_first_intervention_strategy: str | None = None,
    normalization_summary: dict[str, int] | None = None,
    prompt_audit_summary: dict[str, int] | None = None,
    zero_tool_behavior_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create one compact summary manifest for the evidence package."""

    parser_valid = sum(1 for record in prediction_records if record["parser_valid"])
    schema_valid = sum(1 for record in prediction_records if record["schema_valid"])
    sample_count = len(prediction_records)
    failed_count = sum(
        1
        for record in prediction_records
        if not record["parser_valid"] or not record["schema_valid"] or record["prediction"] is None
    )
    summary = {
        "run_id": run_id,
        "tool_mode": tool_mode,
        "backend_name": backend_name,
        "prompt_version": prompt_version,
        "parser_version": parser_version,
        "execution_boundary": execution_boundary,
        "sample_source": sample_source,
        "seeds": seeds,
        "sample_count": sample_count,
        "evaluated_count": sample_count - failed_count,
        "failed_count": failed_count,
        "parser_validity": {
            "valid": parser_valid,
            "invalid": sample_count - parser_valid,
        },
        "schema_validity": {
            "valid": schema_valid,
            "invalid": sample_count - schema_valid,
        },
        "runtime_provenance": runtime_provenance,
        "tool_usage_summary": summarize_tool_usage(prediction_records),
        "artifact_paths": artifact_paths,
        "notes": notes or [],
    }
    if tool_first_intervention_strategy is not None:
        summary["tool_first_intervention_strategy"] = tool_first_intervention_strategy
    if normalization_summary is not None:
        summary["normalization_summary"] = normalization_summary
    if prompt_audit_summary is not None:
        summary["prompt_audit_summary"] = prompt_audit_summary
    if zero_tool_behavior_summary is not None:
        summary["zero_tool_behavior_summary"] = zero_tool_behavior_summary
    validate_payload(summary, "baseline_run_summary.schema.json")
    return summary


def build_delta_report(
    *,
    tool_run_id: str,
    tool_mode: str,
    tool_metrics_report: dict[str, Any],
    baseline_run_id: str,
    baseline_metrics_report: dict[str, Any],
    notes: list[str] | None = None,
) -> dict[str, Any]:
    """Compute one structural delta report against the non-tool baseline."""

    metric_deltas = {
        key: (
            tool_metrics_report["aggregate_metrics"][key]["mean"]
            - baseline_metrics_report["aggregate_metrics"][key]["mean"]
        )
        for key in (
            "parser_valid_rate",
            "schema_valid_rate",
            "anomaly_present_accuracy",
            "top_anomaly_accuracy",
        )
    }
    tool_usage_delta = {
        "toolcall_rate": (
            tool_metrics_report["aggregate_metrics"]["toolcall_rate"]["mean"]
            - baseline_metrics_report["aggregate_metrics"]["toolcall_rate"]["mean"]
        ),
        "per_tool_frequency": {
            tool: (
                tool_metrics_report["aggregate_metrics"]["per_tool_frequency"][tool]["mean"]
                - baseline_metrics_report["aggregate_metrics"]["per_tool_frequency"][tool]["mean"]
            )
            for tool in TOOL_NAMES
        },
    }

    report = {
        "tool_run_id": tool_run_id,
        "tool_mode": tool_mode,
        "baseline_run_id": baseline_run_id,
        "baseline_tool_mode": "no_tools",
        "metric_deltas": metric_deltas,
        "tool_usage_delta": tool_usage_delta,
        "notes": notes or [],
    }
    validate_payload(report, "tool_delta_report.schema.json")
    return report
