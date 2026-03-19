"""Thin behavior-audit helpers for eval-side and train-side zero-tool analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

from agentiad_recon.evaluation import write_json
from agentiad_recon.prompting import parse_final_answer


ZERO_TOOL_SUMMARY_KEYS = (
    "sample_count",
    "turn0_direct_final_answer_count",
    "turn0_tool_call_count",
    "samples_with_any_tool_call",
    "zero_tool_terminal_count",
    "zero_tool_terminal_false_null_count",
)
POST_PZ_TRANSITION_MISMATCH_REASONS = (
    "post_pz_transition_missing_reinserted_pz_result",
    "post_pz_transition_missing_cr_tool",
    "post_pz_transition_missing_query_image_instruction",
    "post_pz_transition_pz_only_leakage",
    "post_pz_transition_missing_tool_context",
)


def build_zero_tool_behavior_fields(
    *,
    first_protocol_event_type: str,
    called_tools: list[str],
    terminal_answer_present: bool,
    terminal_answer_turn_index: int | None,
    prediction: dict[str, Any] | None,
) -> dict[str, Any]:
    """Build deterministic per-sample zero-tool behavior fields."""

    tool_call_count = len(called_tools)
    terminal_without_tool_call = terminal_answer_present and tool_call_count == 0
    terminal_false_null_without_tool_call = (
        terminal_without_tool_call
        and prediction is not None
        and prediction.get("anomaly_present") is False
        and prediction.get("top_anomaly") is None
    )
    return {
        "first_protocol_event_type": first_protocol_event_type,
        "first_assistant_output_terminal": first_protocol_event_type == "final_answer",
        "tool_call_count": tool_call_count,
        "called_tools": called_tools,
        "terminal_without_tool_call": terminal_without_tool_call,
        "terminal_false_null_without_tool_call": terminal_false_null_without_tool_call,
        "terminal_answer_turn_index": terminal_answer_turn_index,
    }


def summarize_zero_tool_behavior(prediction_records: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate zero-tool behavior statistics across prediction records."""

    sample_count = len(prediction_records)
    turn0_direct_final_answer_count = sum(
        1 for record in prediction_records if record["first_protocol_event_type"] == "final_answer"
    )
    turn0_tool_call_count = sum(
        1 for record in prediction_records if record["first_protocol_event_type"] == "tool_call"
    )
    samples_with_any_tool_call = sum(1 for record in prediction_records if record["tool_call_count"] > 0)
    zero_tool_terminal_count = sum(
        1 for record in prediction_records if record["terminal_without_tool_call"]
    )
    zero_tool_terminal_false_null_count = sum(
        1 for record in prediction_records if record["terminal_false_null_without_tool_call"]
    )
    failed_count_with_missing_reason_count = sum(
        1
        for record in prediction_records
        if (
            record["prediction"] is None or not record["parser_valid"] or not record["schema_valid"]
        )
        and (record.get("failure_reason") is None or not str(record["failure_reason"]).strip())
    )
    return {
        "turn0_direct_final_answer_count": turn0_direct_final_answer_count,
        "turn0_tool_call_count": turn0_tool_call_count,
        "samples_with_any_tool_call": samples_with_any_tool_call,
        "zero_tool_terminal_count": zero_tool_terminal_count,
        "zero_tool_terminal_false_null_count": zero_tool_terminal_false_null_count,
        "zero_tool_terminal_ratio": (zero_tool_terminal_count / sample_count) if sample_count else 0.0,
        "zero_tool_terminal_false_null_ratio": (
            zero_tool_terminal_false_null_count / sample_count
        )
        if sample_count
        else 0.0,
        "failed_count_with_missing_reason_count": failed_count_with_missing_reason_count,
    }


def grouped_zero_tool_behavior(
    prediction_records: list[dict[str, Any]],
    *,
    key_name: str,
    key_fn: Any,
) -> dict[str, dict[str, Any]]:
    """Group prediction records and summarize zero-tool behavior for each key."""

    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in prediction_records:
        key = str(key_fn(record))
        grouped.setdefault(key, []).append(record)

    payload: dict[str, dict[str, Any]] = {}
    for key in sorted(grouped):
        records = grouped[key]
        summary = summarize_zero_tool_behavior(records)
        payload[key] = {
            "sample_count": len(records),
            "samples_with_any_tool_call": summary["samples_with_any_tool_call"],
            "zero_tool_terminal_count": summary["zero_tool_terminal_count"],
            "zero_tool_terminal_false_null_count": summary["zero_tool_terminal_false_null_count"],
            "turn0_direct_final_answer_count": summary["turn0_direct_final_answer_count"],
            "turn0_tool_call_count": summary["turn0_tool_call_count"],
            key_name: key,
        }
    return payload


def write_zero_tool_behavior_sidecars(
    *,
    prediction_records: list[dict[str, Any]],
    metrics_dir: str | Path,
) -> dict[str, str]:
    """Write deterministic per-dataset and per-category zero-tool behavior sidecars."""

    metrics_dir = Path(metrics_dir)
    per_dataset_path = metrics_dir / "per_dataset_zero_tool_behavior.json"
    per_category_path = metrics_dir / "per_category_zero_tool_behavior.json"

    write_json(
        per_dataset_path,
        {
            "scope": "dataset",
            "groups": grouped_zero_tool_behavior(
                prediction_records,
                key_name="dataset",
                key_fn=lambda record: record["metadata"].get("sample_source_kind", "unknown"),
            ),
        },
    )
    write_json(
        per_category_path,
        {
            "scope": "category",
            "groups": grouped_zero_tool_behavior(
                prediction_records,
                key_name="category",
                key_fn=lambda record: record["category"],
            ),
        },
    )
    return {
        "per_dataset_zero_tool_behavior": str(per_dataset_path.resolve()),
        "per_category_zero_tool_behavior": str(per_category_path.resolve()),
    }


def build_tool_first_strategy_summary(
    *,
    tool_first_intervention_strategy: str,
    runtime_provenance: dict[str, Any],
    prompt_audit_summary: dict[str, Any] | None,
    zero_tool_behavior_summary: dict[str, Any],
    metrics_report: dict[str, Any],
) -> dict[str, Any]:
    """Build one per-run strategy summary that is easy to compare across runs."""

    aggregate_metrics = metrics_report["aggregate_metrics"]
    return {
        "tool_mode": runtime_provenance["tool_mode"],
        "tool_first_intervention_strategy": tool_first_intervention_strategy,
        "sample_count": metrics_report["sample_count"],
        "evaluated_count": metrics_report["evaluated_count"],
        "failed_count": metrics_report["failed_count"],
        "toolcall_rate": aggregate_metrics["toolcall_rate"]["mean"],
        "per_tool_frequency": {
            tool: aggregate_metrics["per_tool_frequency"][tool]["mean"]
            for tool in sorted(aggregate_metrics["per_tool_frequency"])
        },
        "zero_tool_behavior_summary": zero_tool_behavior_summary,
        "prompt_audit_summary": prompt_audit_summary,
        "runtime_provenance": runtime_provenance,
    }


def summarize_post_pz_transition(prediction_records: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate post-PZ second-turn transition audit statistics across prediction records."""

    event_records = [record for record in prediction_records if record.get("post_pz_transition_audited")]
    failed_count_with_missing_reason_count = sum(
        1
        for record in event_records
        if record.get("post_pz_second_turn_failure_reason")
        and (record.get("failure_reason") is None or not str(record["failure_reason"]).strip())
    )
    return {
        "post_pz_transition_event_count": len(event_records),
        "samples_with_post_pz_transition_events": len(event_records),
        "post_pz_transition_contract_valid_count": sum(
            1 for record in event_records if record.get("post_pz_transition_contract_valid_for_cr") is True
        ),
        "post_pz_transition_contract_mismatch_count": sum(
            1 for record in event_records if record.get("post_pz_transition_contract_valid_for_cr") is False
        ),
        "post_pz_transition_missing_reinserted_pz_result_count": sum(
            1
            for record in event_records
            if "post_pz_transition_missing_reinserted_pz_result"
            in record.get("post_pz_transition_mismatch_reasons", [])
        ),
        "post_pz_transition_missing_cr_tool_count": sum(
            1
            for record in event_records
            if "post_pz_transition_missing_cr_tool" in record.get("post_pz_transition_mismatch_reasons", [])
        ),
        "post_pz_transition_missing_query_image_instruction_count": sum(
            1
            for record in event_records
            if "post_pz_transition_missing_query_image_instruction"
            in record.get("post_pz_transition_mismatch_reasons", [])
        ),
        "post_pz_transition_pz_only_leakage_count": sum(
            1
            for record in event_records
            if "post_pz_transition_pz_only_leakage"
            in record.get("post_pz_transition_mismatch_reasons", [])
        ),
        "post_pz_transition_missing_tool_context_count": sum(
            1
            for record in event_records
            if "post_pz_transition_missing_tool_context"
            in record.get("post_pz_transition_mismatch_reasons", [])
        ),
        "post_pz_second_turn_direct_final_without_cr_count": sum(
            1 for record in event_records if record.get("post_pz_second_turn_direct_final_without_cr")
        ),
        "post_pz_second_turn_called_cr_count": sum(
            1 for record in event_records if record.get("post_pz_second_turn_called_cr")
        ),
        "post_pz_second_turn_called_non_cr_tool_count": sum(
            1 for record in event_records if record.get("post_pz_second_turn_called_non_cr_tool")
        ),
        "post_pz_second_turn_parser_failure_count": sum(
            1
            for record in event_records
            if record.get("post_pz_second_turn_parser_valid") is False
            or record.get("post_pz_second_turn_protocol_event_type") == "parse_failure"
        ),
        "failed_count_with_missing_reason_count": failed_count_with_missing_reason_count,
    }


def summarize_post_pz_transition_sanitation(prediction_records: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate post-PZ sanitation audit statistics across prediction records."""

    event_records = [record for record in prediction_records if record.get("post_pz_transition_audited")]
    failed_count_with_missing_reason_count = sum(
        1
        for record in event_records
        if record.get("post_pz_second_turn_failure_reason")
        and (record.get("failure_reason") is None or not str(record["failure_reason"]).strip())
    )
    return {
        "post_pz_transition_sanitation_event_count": len(event_records),
        "samples_with_post_pz_transition_sanitation_events": len(event_records),
        "post_pz_transition_sanitation_applied_count": sum(
            1 for record in event_records if record.get("post_pz_transition_sanitation_applied")
        ),
        "post_pz_transition_removed_message_total": sum(
            int(record.get("post_pz_transition_removed_message_count", 0)) for record in event_records
        ),
        "post_pz_transition_removed_obsolete_terminal_answer_total": sum(
            int(record.get("post_pz_transition_removed_obsolete_terminal_answer_count", 0))
            for record in event_records
        ),
        "post_pz_transition_removed_pz_only_leakage_message_total": sum(
            int(record.get("post_pz_transition_removed_pz_only_leakage_message_count", 0))
            for record in event_records
        ),
        "post_pz_transition_pre_sanitation_pz_only_leakage_count": sum(
            1
            for record in event_records
            if record.get("post_pz_transition_pre_sanitation_pz_only_leakage_present")
        ),
        "post_pz_transition_post_sanitation_pz_only_leakage_count": sum(
            1
            for record in event_records
            if record.get("post_pz_transition_post_sanitation_pz_only_leakage_present")
        ),
        "post_pz_transition_post_sanitation_contract_valid_count": sum(
            1
            for record in event_records
            if record.get("post_pz_transition_post_sanitation_contract_valid_for_cr") is True
        ),
        "post_pz_transition_post_sanitation_contract_mismatch_count": sum(
            1
            for record in event_records
            if record.get("post_pz_transition_post_sanitation_contract_valid_for_cr") is False
        ),
        "post_pz_second_turn_direct_final_without_cr_count": sum(
            1 for record in event_records if record.get("post_pz_second_turn_direct_final_without_cr")
        ),
        "post_pz_second_turn_called_cr_count": sum(
            1 for record in event_records if record.get("post_pz_second_turn_called_cr")
        ),
        "failed_count_with_missing_reason_count": failed_count_with_missing_reason_count,
    }


def grouped_post_pz_transition(
    prediction_records: list[dict[str, Any]],
    *,
    key_name: str,
    key_fn: Any,
) -> dict[str, dict[str, Any]]:
    """Group prediction records and summarize post-PZ transition behavior for each key."""

    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in prediction_records:
        key = str(key_fn(record))
        grouped.setdefault(key, []).append(record)

    payload: dict[str, dict[str, Any]] = {}
    for key in sorted(grouped):
        records = grouped[key]
        summary = summarize_post_pz_transition(records)
        payload[key] = {
            "sample_count": len(records),
            "post_pz_transition_event_count": summary["post_pz_transition_event_count"],
            "post_pz_transition_contract_valid_count": summary["post_pz_transition_contract_valid_count"],
            "post_pz_transition_contract_mismatch_count": summary["post_pz_transition_contract_mismatch_count"],
            "post_pz_second_turn_direct_final_without_cr_count": summary[
                "post_pz_second_turn_direct_final_without_cr_count"
            ],
            "post_pz_second_turn_called_cr_count": summary["post_pz_second_turn_called_cr_count"],
            "post_pz_second_turn_called_non_cr_tool_count": summary[
                "post_pz_second_turn_called_non_cr_tool_count"
            ],
            key_name: key,
        }
    return payload


def grouped_post_pz_transition_sanitation(
    prediction_records: list[dict[str, Any]],
    *,
    key_name: str,
    key_fn: Any,
) -> dict[str, dict[str, Any]]:
    """Group prediction records and summarize post-PZ sanitation behavior for each key."""

    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in prediction_records:
        key = str(key_fn(record))
        grouped.setdefault(key, []).append(record)

    payload: dict[str, dict[str, Any]] = {}
    for key in sorted(grouped):
        records = grouped[key]
        summary = summarize_post_pz_transition_sanitation(records)
        payload[key] = {
            "sample_count": len(records),
            "post_pz_transition_sanitation_event_count": summary[
                "post_pz_transition_sanitation_event_count"
            ],
            "post_pz_transition_sanitation_applied_count": summary[
                "post_pz_transition_sanitation_applied_count"
            ],
            "post_pz_transition_post_sanitation_contract_valid_count": summary[
                "post_pz_transition_post_sanitation_contract_valid_count"
            ],
            "post_pz_transition_post_sanitation_pz_only_leakage_count": summary[
                "post_pz_transition_post_sanitation_pz_only_leakage_count"
            ],
            "post_pz_second_turn_direct_final_without_cr_count": summary[
                "post_pz_second_turn_direct_final_without_cr_count"
            ],
            "post_pz_second_turn_called_cr_count": summary[
                "post_pz_second_turn_called_cr_count"
            ],
            key_name: key,
        }
    return payload


def write_post_pz_transition_sidecars(
    *,
    prediction_records: list[dict[str, Any]],
    metrics_dir: str | Path,
) -> dict[str, str]:
    """Write deterministic post-PZ transition summary artifacts."""

    metrics_dir = Path(metrics_dir)
    summary_path = metrics_dir / "post_pz_transition_summary.json"
    per_dataset_path = metrics_dir / "per_dataset_post_pz_transition.json"
    per_category_path = metrics_dir / "per_category_post_pz_transition.json"

    write_json(summary_path, summarize_post_pz_transition(prediction_records))
    write_json(
        per_dataset_path,
        {
            "scope": "dataset",
            "groups": grouped_post_pz_transition(
                prediction_records,
                key_name="dataset",
                key_fn=lambda record: record["metadata"].get("sample_source_kind", "unknown"),
            ),
        },
    )
    write_json(
        per_category_path,
        {
            "scope": "category",
            "groups": grouped_post_pz_transition(
                prediction_records,
                key_name="category",
                key_fn=lambda record: record["category"],
            ),
        },
    )
    return {
        "post_pz_transition_summary": str(summary_path.resolve()),
        "per_dataset_post_pz_transition": str(per_dataset_path.resolve()),
        "per_category_post_pz_transition": str(per_category_path.resolve()),
    }


def write_post_pz_transition_sanitation_sidecars(
    *,
    prediction_records: list[dict[str, Any]],
    metrics_dir: str | Path,
) -> dict[str, str]:
    """Write deterministic post-PZ sanitation summary artifacts."""

    metrics_dir = Path(metrics_dir)
    summary_path = metrics_dir / "post_pz_transition_sanitation_summary.json"
    per_dataset_path = metrics_dir / "per_dataset_post_pz_transition_sanitation.json"
    per_category_path = metrics_dir / "per_category_post_pz_transition_sanitation.json"

    write_json(summary_path, summarize_post_pz_transition_sanitation(prediction_records))
    write_json(
        per_dataset_path,
        {
            "scope": "dataset",
            "groups": grouped_post_pz_transition_sanitation(
                prediction_records,
                key_name="dataset",
                key_fn=lambda record: record["metadata"].get("sample_source_kind", "unknown"),
            ),
        },
    )
    write_json(
        per_category_path,
        {
            "scope": "category",
            "groups": grouped_post_pz_transition_sanitation(
                prediction_records,
                key_name="category",
                key_fn=lambda record: record["category"],
            ),
        },
    )
    return {
        "post_pz_transition_sanitation_summary": str(summary_path.resolve()),
        "per_dataset_post_pz_transition_sanitation": str(per_dataset_path.resolve()),
        "per_category_post_pz_transition_sanitation": str(per_category_path.resolve()),
    }


def write_tool_first_strategy_summary(
    *,
    metrics_dir: str | Path,
    tool_first_intervention_strategy: str,
    runtime_provenance: dict[str, Any],
    prompt_audit_summary: dict[str, Any] | None,
    zero_tool_behavior_summary: dict[str, Any],
    metrics_report: dict[str, Any],
) -> str:
    """Write one deterministic per-strategy run summary artifact."""

    metrics_dir = Path(metrics_dir)
    path = metrics_dir / "tool_first_strategy_summary.json"
    write_json(
        path,
        build_tool_first_strategy_summary(
            tool_first_intervention_strategy=tool_first_intervention_strategy,
            runtime_provenance=runtime_provenance,
            prompt_audit_summary=prompt_audit_summary,
            zero_tool_behavior_summary=zero_tool_behavior_summary,
            metrics_report=metrics_report,
        ),
    )
    return str(path.resolve())


def _load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read a JSONL dataset into memory for lightweight auditing."""

    rows: list[dict[str, Any]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _canonical_behavior_row(record: dict[str, Any]) -> dict[str, Any]:
    """Extract tool-first behavior signals from one canonical SFT dataset row."""

    assistant_events = [
        message
        for message in record["messages"]
        if message["role"] == "assistant" and message["message_type"] in {"tool_request", "final_answer"}
    ]
    first_event = assistant_events[0]["message_type"] if assistant_events else "unknown"
    final_answer = record.get("final_answer")
    zero_tool_terminal = len(record["tool_events"]) == 0 and final_answer is not None
    false_null = (
        zero_tool_terminal
        and final_answer.get("anomaly_present") is False
        and final_answer.get("top_anomaly") is None
    )
    return {
        "sample_id": record["sample_id"],
        "dataset": record["sample"]["metadata"].get("dataset_kind", record["sample"]["source"]),
        "category": record["sample"]["category"],
        "first_assistant_protocol_event": "tool_call" if first_event == "tool_request" else "final_answer",
        "tool_event_count": len(record["tool_events"]),
        "zero_tool_terminal": zero_tool_terminal,
        "terminal_false_null_without_tool": false_null,
    }


def _swift_behavior_row(record: dict[str, Any]) -> dict[str, Any]:
    """Extract tool-first behavior signals from one MS-Swift-facing row."""

    assistant_messages = [message for message in record["messages"] if message["role"] == "assistant"]
    first_event = "unknown"
    for message in assistant_messages:
        if message.get("tool_name"):
            first_event = "tool_call"
            break
        if "<answer>" in message["content"] or "<final_answer>" in message["content"]:
            first_event = "final_answer"
            break
    tool_call_count = sum(1 for message in assistant_messages if message.get("tool_name"))
    parsed_final_answer: dict[str, Any] | None = None
    for message in reversed(assistant_messages):
        if "<answer>" in message["content"] or "<final_answer>" in message["content"]:
            try:
                parsed_final_answer = parse_final_answer(message["content"])
            except Exception:  # noqa: BLE001 - audit should remain read-only and tolerant.
                parsed_final_answer = None
            break
    zero_tool_terminal = tool_call_count == 0 and parsed_final_answer is not None
    false_null = (
        zero_tool_terminal
        and parsed_final_answer.get("anomaly_present") is False
        and parsed_final_answer.get("top_anomaly") is None
    )
    return {
        "sample_id": record["metadata"]["sample_id"],
        "dataset": "ms_swift_projection",
        "category": record["metadata"].get("category", "unknown"),
        "first_assistant_protocol_event": first_event,
        "tool_event_count": tool_call_count,
        "zero_tool_terminal": zero_tool_terminal,
        "terminal_false_null_without_tool": false_null,
    }


def audit_train_side_dataset(
    dataset_path: str | Path,
    *,
    dataset_format: str = "auto",
) -> dict[str, Any]:
    """Audit pz_cr supervision strength from an exported SFT dataset JSONL."""

    rows = _load_jsonl(dataset_path)
    if not rows:
        raise ValueError("Train-side audit cannot run on an empty dataset.")

    if dataset_format == "auto":
        if "trajectory_mode" in rows[0]:
            dataset_format = "canonical"
        elif "metadata" in rows[0] and "trajectory_mode" in rows[0]["metadata"]:
            dataset_format = "ms_swift"
        else:
            raise ValueError("Could not infer dataset format for train-side behavior audit.")

    behavior_rows: list[dict[str, Any]] = []
    for row in rows:
        row_mode = row["trajectory_mode"] if dataset_format == "canonical" else row["metadata"]["trajectory_mode"]
        if row_mode != "pz_cr":
            continue
        behavior_rows.append(
            _canonical_behavior_row(row) if dataset_format == "canonical" else _swift_behavior_row(row)
        )

    if not behavior_rows:
        raise ValueError("Train-side audit found zero pz_cr rows in the dataset.")

    pz_cr_record_count = len(behavior_rows)
    first_assistant_tool_call_count = sum(
        1 for row in behavior_rows if row["first_assistant_protocol_event"] == "tool_call"
    )
    first_assistant_final_answer_count = sum(
        1 for row in behavior_rows if row["first_assistant_protocol_event"] == "final_answer"
    )
    zero_tool_terminal_count = sum(1 for row in behavior_rows if row["zero_tool_terminal"])
    false_null_count = sum(1 for row in behavior_rows if row["terminal_false_null_without_tool"])

    def _group(rows_for_group: Iterable[dict[str, Any]], *, key: str) -> dict[str, Any]:
        grouped: dict[str, list[dict[str, Any]]] = {}
        for row in rows_for_group:
            grouped.setdefault(str(row[key]), []).append(row)
        payload: dict[str, Any] = {}
        for group_key in sorted(grouped):
            subset = grouped[group_key]
            subset_count = len(subset)
            payload[group_key] = {
                "pz_cr_record_count": subset_count,
                "first_assistant_tool_call_count": sum(
                    1 for row in subset if row["first_assistant_protocol_event"] == "tool_call"
                ),
                "first_assistant_final_answer_count": sum(
                    1 for row in subset if row["first_assistant_protocol_event"] == "final_answer"
                ),
                "tool_first_ratio": (
                    sum(1 for row in subset if row["first_assistant_protocol_event"] == "tool_call") / subset_count
                ),
                "zero_tool_terminal_ratio": (
                    sum(1 for row in subset if row["zero_tool_terminal"]) / subset_count
                ),
                "terminal_false_null_without_tool_ratio": (
                    sum(1 for row in subset if row["terminal_false_null_without_tool"]) / subset_count
                ),
            }
        return payload

    return {
        "dataset_path": str(Path(dataset_path).resolve()),
        "dataset_format": dataset_format,
        "pz_cr_record_count": pz_cr_record_count,
        "first_assistant_tool_call_count": first_assistant_tool_call_count,
        "first_assistant_final_answer_count": first_assistant_final_answer_count,
        "tool_first_ratio": first_assistant_tool_call_count / pz_cr_record_count,
        "zero_tool_terminal_ratio": zero_tool_terminal_count / pz_cr_record_count,
        "terminal_false_null_without_tool_ratio": false_null_count / pz_cr_record_count,
        "per_dataset": _group(behavior_rows, key="dataset"),
        "per_category": _group(behavior_rows, key="category"),
    }


def main() -> int:
    """Run the lightweight train-side behavior audit from the CLI."""

    parser = argparse.ArgumentParser(description="Audit tool-first vs zero-tool behavior in exported SFT datasets.")
    parser.add_argument("--dataset", required=True, help="Path to a canonical or MS-Swift JSONL dataset.")
    parser.add_argument("--output", required=True, help="Where to write the JSON audit summary.")
    parser.add_argument(
        "--format",
        default="auto",
        choices=["auto", "canonical", "ms_swift"],
        help="Dataset row format. Defaults to auto-detect.",
    )
    args = parser.parse_args()

    summary = audit_train_side_dataset(args.dataset, dataset_format=args.format)
    write_json(args.output, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
