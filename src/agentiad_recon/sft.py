"""Canonical SFT trajectory export and local sanity checks for AgentIAD.

This module adds the first SFT-facing layer on top of the existing Prompt 1.4
waist. It reuses the canonical MMAD samples, prompt family, tool contracts, and
final-answer parser to export auditable `pz_only` and `pz_cr` training
trajectories without introducing a second training stack. Use
`python -m agentiad_recon.sft --help` for the local export entrypoint; it only
runs lightweight fixture-backed export and validation, never full SFT.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agentiad_recon.backends import BackendRequest, MockToolAwareBackend
from agentiad_recon.contracts import validate_payload
from agentiad_recon.mmad import MMADIndexer
from agentiad_recon.ms_swift_adapter import (
    compute_true_length_audit,
    load_swift_recipe,
    swift_runtime_probe,
    validate_swift_record,
)
from agentiad_recon.prompting import (
    FINAL_ANSWER_PARSER_VERSION,
    PROMPT_VERSION,
    build_prompt,
    extract_think_block,
    parse_final_answer,
)
from agentiad_recon.reproducibility import canonical_json_bytes, sha256_bytes, sha256_file, sha256_mapping
from agentiad_recon.tooling import execute_tool_call, parse_tool_call, protocol_event, reinsert_tool_result
from agentiad_recon.traces import TraceMessage, TraceRecord


REPO_ROOT = Path(__file__).resolve().parents[2]
TRAJECTORY_VERSION = "agentiad_sft_trajectory_v1_5"
UNIFIED_DATASET_VERSION = "agentiad_sft_dataset_v1_5"
SCRIPTED_SFT_POLICIES = {
    "pz_only": "fixture_scripted_pz_only_v1",
    "pz_cr": "fixture_scripted_pz_cr_v1",
}
TOOL_ORDER = {"PZ": 0, "CR": 1}
IMAGE_PLACEHOLDER_TOKEN = "<image>"


class SFTExportError(RuntimeError):
    """Raised when SFT trajectory construction or validation breaks the contract."""


@dataclass(frozen=True)
class SFTArtifacts:
    """Returned paths for the exported canonical and framework-facing SFT artifacts."""

    canonical_dataset_path: str
    canonical_manifest_path: str
    swift_dataset_path: str
    swift_manifest_path: str
    swift_recipe_path: str
    swift_runtime_check: dict[str, Any]
    swift_length_audit_path: str
    swift_length_audit_summary: dict[str, Any]
    swift_proxy_length_audit_path: str
    swift_proxy_length_audit_summary: dict[str, Any]
    swift_filtered_manifests: dict[str, str]
    resolved_remote_surfaces_path: str
    resolved_remote_surfaces_summary: dict[str, Any]
    local_validation: dict[str, Any]


def _resolve_path(value: str | Path) -> Path:
    """Resolve repository-relative paths so config files stay portable."""

    path = Path(value)
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def load_sft_definition(path: str | Path) -> dict[str, Any]:
    """Load and validate the Prompt 1.5 export definition."""

    definition = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_payload(definition, "sft_export_definition.schema.json")
    return definition


def _prompt_history(prompt_bundle: Any) -> list[dict[str, Any]]:
    """Copy prompt messages into a mutable dialogue history."""

    return [
        {
            "role": message["role"],
            "message_type": message["message_type"],
            "content": message["content"],
            "image_refs": list(message["image_refs"]),
            "metadata": dict(message["metadata"]),
            "tool_name": message.get("tool_name"),
            "call_id": message.get("call_id"),
        }
        for message in prompt_bundle.messages
    ]


def _append_reasoning(history: list[dict[str, Any]], raw_output: str, *, backend_name: str) -> None:
    """Append an assistant reasoning message when the response has a think block."""

    think_block = extract_think_block(raw_output)
    if think_block is None:
        return
    history.append(
        {
            "role": "assistant",
            "message_type": "reasoning",
            "content": think_block,
            "image_refs": [],
            "metadata": {"backend_name": backend_name},
            "tool_name": None,
            "call_id": None,
        }
    )


def _append_tool_request(
    history: list[dict[str, Any]],
    raw_output: str,
    *,
    backend_name: str,
    tool_name: str,
    call_id: str,
) -> None:
    """Append the raw assistant tool request so the trajectory stays auditable."""

    history.append(
        {
            "role": "assistant",
            "message_type": "tool_request",
            "content": raw_output,
            "image_refs": [],
            "metadata": {"backend_name": backend_name},
            "tool_name": tool_name,
            "call_id": call_id,
        }
    )


def _append_final_answer_message(
    history: list[dict[str, Any]],
    raw_output: str,
    *,
    backend_name: str,
) -> None:
    """Append the assistant final-answer turn to the training trace."""

    history.append(
        {
            "role": "assistant",
            "message_type": "final_answer",
            "content": raw_output,
            "image_refs": [],
            "metadata": {"backend_name": backend_name},
            "tool_name": None,
            "call_id": None,
        }
    )


def _history_to_trace_messages(history: list[dict[str, Any]]) -> tuple[TraceMessage, ...]:
    """Project mutable history items into frozen trace messages."""

    return tuple(
        TraceMessage(
            role=message["role"],
            message_type=message["message_type"],
            content=message["content"],
            image_refs=tuple(message.get("image_refs", [])),
            tool_name=message.get("tool_name"),
            call_id=message.get("call_id"),
            metadata=message.get("metadata", {}),
        )
        for message in history
    )


def _build_training_trace(
    *,
    sample: dict[str, Any],
    sample_pool: list[dict[str, Any]],
    mode: str,
    artifact_root: Path,
) -> TraceRecord:
    """Generate one deterministic training trace by reusing the Prompt 1.4 tool loop."""

    backend = MockToolAwareBackend(
        backend_name=f"mock_sft_{mode}_builder_v1",
        policy=SCRIPTED_SFT_POLICIES[mode],
    )
    prompt_bundle = build_prompt(sample, tool_path=mode)
    history = _prompt_history(prompt_bundle)
    tool_traces: list[dict[str, Any]] = []
    final_answer: dict[str, Any] | None = None

    for turn_index in range(4):
        request = BackendRequest(
            sample_id=sample["sample_id"],
            seed=0,
            prompt_version=prompt_bundle.prompt_version,
            messages=history,
            stop_sequences=prompt_bundle.stop_sequences,
            tool_mode=mode,
            metadata={"tool_mode": mode, "trajectory_stage": "sft", "turn_index": turn_index},
        )
        response = backend.generate(request, sample=sample)
        _append_reasoning(history, response.raw_output, backend_name=response.backend_name)
        event = protocol_event(response.raw_output)
        if event == "tool_call":
            parsed_call = parse_tool_call(response.raw_output, tool_path=mode)
            _append_tool_request(
                history,
                response.raw_output,
                backend_name=response.backend_name,
                tool_name=parsed_call.tool_name,
                call_id=parsed_call.call_id,
            )
            tool_result = execute_tool_call(
                parsed_call,
                sample=sample,
                sample_pool=sample_pool,
                artifact_dir=artifact_root / sample["sample_id"].replace(":", "_"),
            )
            tool_payload = tool_result.to_payload()
            tool_traces.append(tool_payload)
            history = reinsert_tool_result(history, tool_result)
            continue

        if event != "final_answer":
            raise SFTExportError(
                f"Training trajectory for {sample['sample_id']} in {mode} ended without a final answer block"
            )
        final_answer = parse_final_answer(response.raw_output)
        _append_final_answer_message(history, response.raw_output, backend_name=response.backend_name)
        break

    if final_answer is None:
        raise SFTExportError(f"No final answer produced for {sample['sample_id']} in {mode}")

    return TraceRecord(
        trace_id=f"sft:{mode}:{sample['sample_id']}",
        sample_id=sample["sample_id"],
        stage="sft",
        tool_path=mode,
        storage_purpose="training",
        messages=_history_to_trace_messages(history),
        tool_traces=tuple(tool_traces),
        final_answer=final_answer,
        metadata={
            "sample_category": sample["category"],
            "trajectory_version": TRAJECTORY_VERSION,
            "prompt_version": prompt_bundle.prompt_version,
            "parser_version": FINAL_ANSWER_PARSER_VERSION,
        },
    )


def _validate_final_answer_consistency(sample: dict[str, Any], final_answer: dict[str, Any]) -> None:
    """Check that scripted final answers stay aligned with the canonical labels."""

    expected = sample["ground_truth"]
    if final_answer["anomaly_present"] != expected["anomaly_present"]:
        raise SFTExportError(
            f"anomaly_present mismatch for {sample['sample_id']}: {final_answer['anomaly_present']} vs {expected['anomaly_present']}"
        )
    if final_answer["top_anomaly"] != expected["top_anomaly"]:
        raise SFTExportError(
            f"top_anomaly mismatch for {sample['sample_id']}: {final_answer['top_anomaly']} vs {expected['top_anomaly']}"
        )


def _expected_message_order(mode: str) -> list[str]:
    """Return the exact ordered message-type pattern required for each trajectory mode."""

    if mode == "pz_only":
        return [
            "system_instruction",
            "user_prompt",
            "reasoning",
            "tool_request",
            "tool_result",
            "reasoning",
            "final_answer",
        ]
    if mode == "pz_cr":
        return [
            "system_instruction",
            "user_prompt",
            "reasoning",
            "tool_request",
            "tool_result",
            "reasoning",
            "tool_request",
            "tool_result",
            "reasoning",
            "final_answer",
        ]
    raise SFTExportError(f"Unsupported trajectory mode: {mode}")


def validate_trace_contract(trace: TraceRecord, sample: dict[str, Any]) -> None:
    """Run the Prompt 1.5 trajectory integrity checks before export."""

    trace.to_audit_payload()
    trace.to_training_trajectory()
    _validate_final_answer_consistency(sample, trace.final_answer or {})

    observed_order = [message.message_type for message in trace.messages]
    expected_order = _expected_message_order(trace.tool_path)
    if observed_order != expected_order:
        raise SFTExportError(
            f"Message order mismatch for {trace.sample_id} in {trace.tool_path}: {observed_order} != {expected_order}"
        )

    primary_image = sample["image"]["uri"]
    user_message = trace.messages[1]
    if tuple(user_message.image_refs) != (primary_image,):
        raise SFTExportError(f"Primary image binding mismatch for {trace.sample_id}")

    if trace.tool_path == "pz_only":
        if len(trace.tool_traces) != 1 or trace.tool_traces[0]["tool_name"] != "PZ":
            raise SFTExportError(f"pz_only trajectory for {trace.sample_id} must contain exactly one PZ call")
        tool_result_message = trace.messages[4]
        if len(tool_result_message.image_refs) != 1:
            raise SFTExportError(f"PZ tool result for {trace.sample_id} must carry one crop image ref")

    if trace.tool_path == "pz_cr":
        tool_names = [tool_trace["tool_name"] for tool_trace in trace.tool_traces]
        if tool_names != ["PZ", "CR"]:
            raise SFTExportError(f"pz_cr trajectory for {trace.sample_id} must contain PZ then CR")
        cr_trace = trace.tool_traces[1]
        exemplar = cr_trace["output_payload"].get("selected_exemplar")
        if not isinstance(exemplar, dict):
            raise SFTExportError(f"CR exemplar linkage missing for {trace.sample_id}")
        if exemplar["category"] != sample["category"]:
            raise SFTExportError(f"CR exemplar category mismatch for {trace.sample_id}")
        if exemplar["sample_id"] == sample["sample_id"]:
            raise SFTExportError(f"CR exemplar must differ from the target sample for {trace.sample_id}")
        cr_result_message = trace.messages[7]
        if tuple(cr_result_message.image_refs) != (exemplar["image_uri"],):
            raise SFTExportError(f"CR image-ref insertion mismatch for {trace.sample_id}")


def _loss_reason(message_type: str, message_index: int, mode: str) -> tuple[bool, bool, str]:
    """Assign the Prompt 1.5 decisive-turn and answer-alignment supervision labels."""

    decisive_reasoning_index = 5 if mode == "pz_only" else 8
    decisive_tool_index = 3 if mode == "pz_only" else 6
    final_answer_index = 6 if mode == "pz_only" else 9

    if message_type == "tool_request" and message_index == decisive_tool_index:
        return True, True, "last_visual_operation"
    if message_type == "reasoning" and message_index == decisive_reasoning_index:
        return True, True, "final_reasoning"
    if message_type == "final_answer" and message_index == final_answer_index:
        return True, False, "final_answer_alignment"
    return False, False, "none"


def _tool_events(trace: TraceRecord) -> list[dict[str, Any]]:
    """Project raw tool traces into a stable, sorted event list for the unified dataset."""

    events: list[dict[str, Any]] = []
    for tool_trace in sorted(trace.tool_traces, key=lambda item: (TOOL_ORDER[item["tool_name"]], item["call_id"])):
        image_bindings: list[str] = []
        if tool_trace["tool_name"] == "PZ":
            artifact_path = tool_trace["output_payload"].get("artifact_path")
            if isinstance(artifact_path, str):
                image_bindings.append(artifact_path)
        if tool_trace["tool_name"] == "CR":
            exemplar = tool_trace["output_payload"].get("selected_exemplar")
            if isinstance(exemplar, dict):
                image_uri = exemplar.get("image_uri")
                if isinstance(image_uri, str):
                    image_bindings.append(image_uri)

        events.append(
            {
                "call_id": tool_trace["call_id"],
                "tool_name": tool_trace["tool_name"],
                "arguments": tool_trace["arguments"],
                "output_ref": tool_trace["output_ref"],
                "output_payload": tool_trace["output_payload"],
                "image_bindings": image_bindings,
                "status": tool_trace["status"],
            }
        )
    return events


def build_unified_sft_record(trace: TraceRecord, sample: dict[str, Any]) -> dict[str, Any]:
    """Convert one validated training trace into the Prompt 1.5 unified dataset record."""

    validate_trace_contract(trace, sample)

    tool_events = _tool_events(trace)
    messages: list[dict[str, Any]] = []
    loss_message_indices: list[int] = []
    decisive_message_indices: list[int] = []
    for index, message in enumerate(trace.messages):
        loss_mask, decisive_turn, loss_reason = _loss_reason(message.message_type, index, trace.tool_path)
        if loss_mask:
            loss_message_indices.append(index)
        if decisive_turn:
            decisive_message_indices.append(index)
        messages.append(
            {
                "message_index": index,
                "role": message.role,
                "message_type": message.message_type,
                "content": message.content,
                "image_refs": list(message.image_refs),
                "tool_name": message.tool_name,
                "call_id": message.call_id,
                "loss_mask": loss_mask,
                "decisive_turn": decisive_turn,
                "loss_reason": loss_reason,
                "metadata": message.metadata,
            }
        )

    payload = {
        "trajectory_id": trace.trace_id,
        "sample_id": trace.sample_id,
        "trajectory_mode": trace.tool_path,
        "sample": {
            "source": sample["source"],
            "split": sample["split"],
            "category": sample["category"],
            "anomaly_present": sample["anomaly_present"],
            "top_anomaly": sample["ground_truth"]["top_anomaly"],
            "anomaly_candidates": sample["anomaly_candidates"],
            "primary_image_uri": sample["image"]["uri"],
            "primary_image_sha256": sample["image"]["sha256"],
            "metadata": sample["metadata"],
        },
        "versions": {
            "prompt_version": PROMPT_VERSION,
            "parser_version": FINAL_ANSWER_PARSER_VERSION,
            "trajectory_version": TRAJECTORY_VERSION,
            "dataset_version": UNIFIED_DATASET_VERSION,
        },
        "messages": messages,
        "tool_events": tool_events,
        "final_answer": trace.final_answer,
        "loss_summary": {
            "loss_message_indices": loss_message_indices,
            "decisive_message_indices": decisive_message_indices,
            "loss_rule": "supervise last visual operation, final reasoning, and final answer alignment",
        },
        "metadata": {
            "agent_mode": "single_agent",
            "tool_path": trace.tool_path,
            "hash_basis": sha256_mapping(
                {
                    "trajectory_id": trace.trace_id,
                    "sample_id": trace.sample_id,
                    "mode": trace.tool_path,
                    "messages": messages,
                    "tool_events": tool_events,
                    "final_answer": trace.final_answer,
                }
            ),
        },
    }
    validate_payload(payload, "sft_dataset_record.schema.json")
    return payload


def export_sft_dataset(
    *,
    config_path: str | Path,
    dataset_root: str | Path | None = None,
    output_root: str | Path | None = None,
    max_samples_per_mode: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Export the canonical Prompt 1.5 SFT dataset for the requested trajectory modes."""

    definition = load_sft_definition(config_path)
    resolved_dataset_root = _resolve_path(dataset_root or definition["sample_source"]["path"])
    output_directory = _resolve_path(output_root or definition["output"]["root"])
    output_directory.mkdir(parents=True, exist_ok=True)

    indexer = MMADIndexer(resolved_dataset_root, source=definition["sample_source"]["source_name"])
    samples = [sample.to_dict() for sample in indexer.index_samples()]
    split_filter = set(definition["sample_source"]["splits"])
    selected_samples = [sample for sample in samples if sample["split"] in split_filter]
    per_mode_limit = max_samples_per_mode if max_samples_per_mode is not None else definition["sample_source"]["max_samples_per_mode"]
    if per_mode_limit is not None:
        selected_samples = selected_samples[:per_mode_limit]
    if not selected_samples:
        raise SFTExportError("No canonical samples were selected for SFT export.")

    artifact_root = output_directory / "trajectory_artifacts"
    records: list[dict[str, Any]] = []
    for mode in definition["trajectory_modes"]:
        for sample in selected_samples:
            trace = _build_training_trace(
                sample=sample,
                sample_pool=samples,
                mode=mode,
                artifact_root=artifact_root / mode,
            )
            records.append(build_unified_sft_record(trace, sample))

    canonical_dataset_path = output_directory / definition["output"]["canonical_dataset"]
    canonical_dataset_path.write_text(
        "\n".join(json.dumps(record, sort_keys=True) for record in records) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "dataset_version": UNIFIED_DATASET_VERSION,
        "record_count": len(records),
        "trajectory_modes": definition["trajectory_modes"],
        "sample_count_per_mode": len(selected_samples),
        "canonical_dataset_path": str(canonical_dataset_path.resolve()),
        "canonical_dataset_sha256": sha256_file(canonical_dataset_path),
        "ordering": "trajectory_mode/sample_id/message_index",
        "boundary": definition["execution_boundary"],
        "config_hash": sha256_file(config_path),
    }
    validate_payload(manifest, "sft_dataset_manifest.schema.json")
    manifest_path = output_directory / definition["output"]["manifest"]
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return records, {"definition": definition, "manifest": manifest, "manifest_path": str(manifest_path.resolve())}


def local_dataset_sanity(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Run lightweight Prompt 1.5 checks over the unified canonical dataset."""

    per_mode_counts = {mode: 0 for mode in SCRIPTED_SFT_POLICIES}
    for record in records:
        validate_payload(record, "sft_dataset_record.schema.json")
        per_mode_counts[record["trajectory_mode"]] += 1
        message_types = [message["message_type"] for message in record["messages"]]
        if message_types != _expected_message_order(record["trajectory_mode"]):
            raise SFTExportError(f"Unexpected message order inside unified record {record['trajectory_id']}")

        decisive_reasons = [
            message["loss_reason"]
            for message in record["messages"]
            if message["decisive_turn"]
        ]
        if sorted(decisive_reasons) != ["final_reasoning", "last_visual_operation"]:
            raise SFTExportError(f"Decisive-turn loss targets are wrong for {record['trajectory_id']}")

        if "final_answer_alignment" not in {message["loss_reason"] for message in record["messages"]}:
            raise SFTExportError(f"Final answer alignment loss target is missing for {record['trajectory_id']}")

        if record["trajectory_mode"] == "pz_cr":
            exemplar = record["tool_events"][1]["output_payload"]["selected_exemplar"]
            if exemplar["category"] != record["sample"]["category"]:
                raise SFTExportError(f"CR exemplar linkage failed in {record['trajectory_id']}")

    return {
        "record_count": len(records),
        "per_mode_counts": per_mode_counts,
        "message_order_checked": True,
        "image_bindings_checked": True,
        "decisive_turn_loss_checked": True,
        "exemplar_linkage_checked": True,
    }


_THINK_BLOCK_PATTERN = re.compile(r"(?s)(?P<prefix>.*?)<think>\s*(?P<think>.*?)\s*</think>(?P<suffix>.*)")
_TOOL_CALL_BLOCK_PATTERN = re.compile(r"(?s)<tool_call>\s*(\{.*?\})\s*</tool_call>")


def _normalized_text(text: str) -> str:
    """Collapse whitespace so duplicate-think detection is robust to formatting."""

    return " ".join(text.split())


def _dedupe_prefix_against_think(content: str) -> str:
    """Drop assistant prose duplicated both outside and inside one `<think>` block."""

    match = _THINK_BLOCK_PATTERN.fullmatch(content)
    if match is None:
        return content

    prefix = match.group("prefix")
    think = match.group("think")
    suffix = match.group("suffix")
    if prefix.strip() and _normalized_text(prefix) == _normalized_text(think):
        compact = f"<think>\n{think.strip()}\n</think>"
        if suffix.strip():
            compact = f"{compact}\n{suffix.strip()}"
        return compact
    return content


def _compact_tool_call_block(content: str) -> str:
    """Keep only a compact `<tool_call>` JSON block for tool-request supervision."""

    compact_source = _dedupe_prefix_against_think(content)
    match = _TOOL_CALL_BLOCK_PATTERN.search(compact_source)
    if match is None:
        return compact_source
    try:
        payload = json.loads(match.group(1))
    except json.JSONDecodeError:
        return compact_source
    return f"<tool_call>{json.dumps(payload, separators=(',', ':'), sort_keys=True)}</tool_call>"


def _placeholder_value(image_count: int) -> str | list[str]:
    """Return one placeholder string or a repeated placeholder list."""

    if image_count <= 0:
        return ""
    if image_count == 1:
        return IMAGE_PLACEHOLDER_TOKEN
    return [IMAGE_PLACEHOLDER_TOKEN for _ in range(image_count)]


def _compact_tool_result_payload(
    *,
    tool_name: str | None,
    content: str,
    image_refs: list[str],
) -> str:
    """Project verbose tool payloads into compact training-oriented JSON text."""

    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        payload = {}
    if not isinstance(payload, dict):
        payload = {}

    compact: dict[str, Any] = {"tool": tool_name or "UNKNOWN"}
    placeholder = _placeholder_value(len(image_refs))

    if tool_name == "PZ":
        if placeholder:
            compact["image"] = placeholder
        bbox = payload.get("normalized_bbox")
        if isinstance(bbox, dict):
            compact_bbox = {}
            for key in ("x0", "y0", "x1", "y1"):
                value = bbox.get(key)
                if isinstance(value, (int, float)):
                    compact_bbox[key] = value
            if compact_bbox:
                compact["bbox"] = compact_bbox
        compact["result"] = "crop_ready"
        return json.dumps(compact, separators=(",", ":"), sort_keys=True)

    if tool_name == "CR":
        exemplar = payload.get("selected_exemplar")
        if isinstance(exemplar, dict):
            compact_exemplar: dict[str, Any] = {}
            for key in ("sample_id", "category", "split"):
                value = exemplar.get(key)
                if isinstance(value, str) and value:
                    compact_exemplar[key] = value
            if placeholder:
                compact_exemplar["image"] = placeholder
            compact["selected_exemplar"] = compact_exemplar
        else:
            compact["selected_exemplar"] = None
        compact["result"] = "reference_ready"
        return json.dumps(compact, separators=(",", ":"), sort_keys=True)

    if placeholder:
        compact["image"] = placeholder
    compact["result"] = "tool_output"
    return json.dumps(compact, separators=(",", ":"), sort_keys=True)


def _render_swift_message_content(
    message: dict[str, Any],
    *,
    image_refs_for_placeholders: list[str] | None = None,
) -> str:
    """Render one canonical message into the MS-Swift string-content shape."""

    content = message["content"]
    if not isinstance(content, str):
        raise SFTExportError(f"MS-Swift projection requires string message content, got {type(content)!r}")

    image_refs = (
        list(message["image_refs"])
        if image_refs_for_placeholders is None
        else list(image_refs_for_placeholders)
    )
    if not image_refs:
        if message["message_type"] == "tool_request":
            return _compact_tool_call_block(content)
        return _dedupe_prefix_against_think(content)

    if message["message_type"] == "tool_result":
        return _compact_tool_result_payload(
            tool_name=message.get("tool_name"),
            content=content,
            image_refs=image_refs,
        )

    placeholder_prefix = IMAGE_PLACEHOLDER_TOKEN * len(image_refs)
    compact_content = _dedupe_prefix_against_think(content)
    return f"{placeholder_prefix}\n{compact_content}" if compact_content else placeholder_prefix


def _nearest_rank(sorted_values: list[int], percentile: int) -> int:
    """Compute nearest-rank percentile from a pre-sorted non-empty integer list."""

    rank = max(1, math.ceil((percentile / 100.0) * len(sorted_values)))
    return sorted_values[rank - 1]


def _build_proxy_length_audit(swift_records: list[dict[str, Any]]) -> dict[str, Any]:
    """Build proxy length statistics for compact MS-Swift-facing records."""

    if not swift_records:
        raise SFTExportError("Cannot build length audit for an empty MS-Swift dataset.")

    rows: list[dict[str, Any]] = []
    for record in swift_records:
        char_length = sum(len(message["content"]) for message in record["messages"])
        token_estimate = sum(len(message["content"].split()) for message in record["messages"])
        rows.append(
            {
                "id": record["id"],
                "sample_id": record["metadata"]["sample_id"],
                "trajectory_mode": record["metadata"]["trajectory_mode"],
                "char_length": char_length,
                "token_estimate": token_estimate,
            }
        )

    token_lengths = sorted(row["token_estimate"] for row in rows)
    top_rows = sorted(rows, key=lambda row: row["token_estimate"], reverse=True)[:5]
    return {
        "length_unit": "whitespace_token_estimate_from_message_content",
        "true_multimodal_encode": False,
        "backend": "proxy_whitespace_split",
        "backend_detail": "does_not_include_true_template_or_visual_expansion",
        "record_count": len(rows),
        "p50": _nearest_rank(token_lengths, 50),
        "p90": _nearest_rank(token_lengths, 90),
        "p95": _nearest_rank(token_lengths, 95),
        "p99": _nearest_rank(token_lengths, 99),
        "max": token_lengths[-1],
        "count_above_4096": sum(1 for value in token_lengths if value > 4096),
        "count_above_8192": sum(1 for value in token_lengths if value > 8192),
        "top_offenders": [
            {
                "id": row["id"],
                "sample_id": row["sample_id"],
                "trajectory_mode": row["trajectory_mode"],
                "token_estimate": row["token_estimate"],
                "char_length": row["char_length"],
            }
            for row in top_rows
        ],
    }


def build_swift_records(records: list[dict[str, Any]], recipe: dict[str, Any]) -> list[dict[str, Any]]:
    """Project canonical Prompt 1.5 records into a thin MS-Swift-facing dataset shape."""

    swift_records: list[dict[str, Any]] = []
    for record in records:
        dataset_messages: list[dict[str, Any]] = []
        images: list[str] = []
        seen_images: set[str] = set()
        placeholder_ref_traversal: list[str] = []
        for message in record["messages"]:
            swift_role = message["role"]
            if message["message_type"] == "tool_result":
                swift_role = "tool"

            new_image_refs: list[str] = []
            for image_ref in message["image_refs"]:
                if image_ref not in seen_images:
                    seen_images.add(image_ref)
                    images.append(image_ref)
                    new_image_refs.append(image_ref)
                    placeholder_ref_traversal.append(image_ref)

            dataset_message = {
                "role": swift_role,
                "content": _render_swift_message_content(
                    message,
                    image_refs_for_placeholders=new_image_refs,
                ),
                "loss": message["loss_mask"] if swift_role == "assistant" else False,
            }
            if not isinstance(dataset_message["content"], str):
                raise SFTExportError(
                    f"MS-Swift projection produced non-string content in {record['trajectory_id']}"
                )
            if message["tool_name"] is not None:
                dataset_message["tool_name"] = message["tool_name"]
            if message["call_id"] is not None:
                dataset_message["call_id"] = message["call_id"]
            dataset_messages.append(dataset_message)

        placeholder_count = sum(
            dataset_message["content"].count(IMAGE_PLACEHOLDER_TOKEN) for dataset_message in dataset_messages
        )
        if placeholder_count != len(images):
            raise SFTExportError(
                f"Placeholder/image mismatch in {record['trajectory_id']}: "
                f"{placeholder_count} placeholders vs {len(images)} images"
            )
        if placeholder_ref_traversal != images:
            raise SFTExportError(
                f"Placeholder traversal order diverged from image order in {record['trajectory_id']}"
            )

        swift_record = {
            "id": record["trajectory_id"],
            "messages": dataset_messages,
            "images": images,
            "tools": recipe["dataset"].get("tools", []),
            "metadata": {
                "sample_id": record["sample_id"],
                "trajectory_mode": record["trajectory_mode"],
                "top_anomaly": record["final_answer"]["top_anomaly"],
                "loss_message_indices": record["loss_summary"]["loss_message_indices"],
            },
        }
        validate_swift_record(swift_record)
        swift_records.append(swift_record)
    return swift_records


def export_swift_dataset(
    *,
    canonical_records: list[dict[str, Any]],
    recipe_path: str | Path,
    output_root: str | Path,
    strict_true_length_audit: bool = False,
) -> dict[str, Any]:
    """Write the thin MS-Swift dataset projection and a recipe manifest."""

    recipe = load_swift_recipe(recipe_path)
    output_directory = _resolve_path(output_root)
    output_directory.mkdir(parents=True, exist_ok=True)

    swift_records = build_swift_records(canonical_records, recipe)
    swift_dataset_path = output_directory / recipe["dataset"]["output_jsonl"]
    swift_dataset_path.write_text(
        "\n".join(json.dumps(record, sort_keys=True) for record in swift_records) + "\n",
        encoding="utf-8",
    )
    stem = Path(recipe["dataset"]["output_jsonl"]).stem
    proxy_length_audit = _build_proxy_length_audit(swift_records)
    proxy_length_audit_path = output_directory / f"{stem}.length_audit.proxy.json"
    proxy_length_audit_path.write_text(json.dumps(proxy_length_audit, indent=2, sort_keys=True), encoding="utf-8")

    true_length_audit = compute_true_length_audit(swift_records, recipe, strict=strict_true_length_audit)
    true_length_audit_path = output_directory / f"{stem}.length_audit.true.json"
    true_length_audit_path.write_text(json.dumps(true_length_audit, indent=2, sort_keys=True), encoding="utf-8")
    true_multimodal_encode = bool(true_length_audit["true_multimodal_encode"])
    length_audit_backend = str(true_length_audit["backend"])
    threshold_clean_basis = (
        "true_multimodal_encode"
        if true_multimodal_encode
        else "fallback_derived_not_true_certified"
    )
    strict_passed = strict_true_length_audit and true_multimodal_encode

    true_lengths = {row["id"]: int(row["encoded_length"]) for row in true_length_audit["lengths"]}
    filtered_exports: list[dict[str, Any]] = []
    filtered_manifest_paths: dict[str, str] = {}
    filtered_export_summary_by_threshold: dict[str, dict[str, Any]] = {}
    total_record_count = len(swift_records)
    for threshold in (4096, 8192):
        kept_records = [record for record in swift_records if true_lengths[record["id"]] <= threshold]
        dropped_rows = sorted(
            [row for row in true_length_audit["lengths"] if row["encoded_length"] > threshold],
            key=lambda row: row["encoded_length"],
            reverse=True,
        )
        kept_lengths = [true_lengths[record["id"]] for record in kept_records]
        dropped_count = len(dropped_rows)
        dropped_ratio = (dropped_count / total_record_count) if total_record_count else 0.0

        filtered_dataset_path = output_directory / f"{stem}_le{threshold}.jsonl"
        filtered_dataset_path.write_text(
            "\n".join(json.dumps(record, sort_keys=True) for record in kept_records) + ("\n" if kept_records else ""),
            encoding="utf-8",
        )
        filtered_manifest = {
            "record_count": len(kept_records),
            "threshold": threshold,
            "parent_dataset_path": str(swift_dataset_path.resolve()),
            "parent_dataset_sha256": sha256_file(swift_dataset_path),
            "source_swift_dataset_path": str(swift_dataset_path.resolve()),
            "source_true_audit_path": str(true_length_audit_path.resolve()),
            "kept_dataset_path": str(filtered_dataset_path.resolve()),
            "kept_count": len(kept_records),
            "dropped_count": dropped_count,
            "dropped_ratio": dropped_ratio,
            "true_length_audit_path": str(true_length_audit_path.resolve()),
            "true_multimodal_encode": true_multimodal_encode,
            "length_audit_backend": length_audit_backend,
            "threshold_clean_basis": threshold_clean_basis,
            "strict_true_length_audit_requested": strict_true_length_audit,
            "strict_true_length_audit_passed": strict_passed,
            "true_threshold_clean_certified": true_multimodal_encode,
            "top_dropped_offenders": dropped_rows[:10],
            "max_kept_encoded_length": max(kept_lengths) if kept_lengths else None,
            "min_dropped_encoded_length": min(row["encoded_length"] for row in dropped_rows) if dropped_rows else None,
        }
        filtered_manifest_path = output_directory / f"{stem}_le{threshold}.manifest.json"
        filtered_manifest_path.write_text(json.dumps(filtered_manifest, indent=2, sort_keys=True), encoding="utf-8")
        filtered_summary = {
            "threshold": threshold,
            "dataset_path": str(filtered_dataset_path.resolve()),
            "manifest_path": str(filtered_manifest_path.resolve()),
            "kept_count": len(kept_records),
            "dropped_count": dropped_count,
            "dropped_ratio": dropped_ratio,
            "true_multimodal_encode": true_multimodal_encode,
            "length_audit_backend": length_audit_backend,
            "threshold_clean_basis": threshold_clean_basis,
            "true_threshold_clean_certified": true_multimodal_encode,
        }
        filtered_exports.append(filtered_summary)
        filtered_export_summary_by_threshold[str(threshold)] = filtered_summary
        filtered_manifest_paths[str(threshold)] = str(filtered_manifest_path.resolve())

    runtime_probe = swift_runtime_probe()
    manifest = {
        "recipe_name": recipe["recipe_name"],
        "record_count": len(swift_records),
        "swift_dataset_path": str(swift_dataset_path.resolve()),
        "swift_dataset_sha256": sha256_file(swift_dataset_path),
        "length_audit_proxy_path": str(proxy_length_audit_path.resolve()),
        "length_audit_proxy_summary": {
            "record_count": proxy_length_audit["record_count"],
            "p50": proxy_length_audit["p50"],
            "p90": proxy_length_audit["p90"],
            "p95": proxy_length_audit["p95"],
            "p99": proxy_length_audit["p99"],
            "max": proxy_length_audit["max"],
            "count_above_4096": proxy_length_audit["count_above_4096"],
            "count_above_8192": proxy_length_audit["count_above_8192"],
        },
        "length_audit_true_path": str(true_length_audit_path.resolve()),
        "length_audit_true_summary": {
            "record_count": true_length_audit["record_count"],
            "p50": true_length_audit["p50"],
            "p90": true_length_audit["p90"],
            "p95": true_length_audit["p95"],
            "p99": true_length_audit["p99"],
            "max": true_length_audit["max"],
            "count_above_4096": true_length_audit["count_above_4096"],
            "count_above_8192": true_length_audit["count_above_8192"],
            "true_multimodal_encode": true_multimodal_encode,
            "length_audit_backend": length_audit_backend,
            "threshold_clean_basis": threshold_clean_basis,
            "strict_true_length_audit_requested": strict_true_length_audit,
            "strict_true_length_audit_passed": strict_passed,
        },
        "filtered_exports": filtered_exports,
        "filtered_export_summary_by_threshold": filtered_export_summary_by_threshold,
        "runtime_probe": runtime_probe,
        "notes": [
            "This dataset is adapter-generated for later MS-Swift ownership.",
            "Local Prompt 1.9 keeps threshold-clean export driven by true multimodal length audit when available.",
        ],
    }
    validate_payload(manifest, "sft_dataset_manifest.schema.json")
    manifest_path = output_directory / recipe["dataset"]["manifest_json"]
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return {
        "swift_dataset_path": swift_dataset_path,
        "swift_manifest_path": manifest_path,
        "proxy_length_audit_path": proxy_length_audit_path,
        "proxy_length_audit": proxy_length_audit,
        "true_length_audit_path": true_length_audit_path,
        "true_length_audit": true_length_audit,
        "filtered_manifest_paths": filtered_manifest_paths,
        "true_multimodal_encode": true_multimodal_encode,
        "length_audit_backend": length_audit_backend,
        "threshold_clean_basis": threshold_clean_basis,
        "strict_true_length_audit_requested": strict_true_length_audit,
        "strict_true_length_audit_passed": strict_passed,
        "recipe": recipe,
        "runtime_probe": runtime_probe,
    }


def _write_resolved_remote_surfaces(
    *,
    output_root: Path,
    export_config_path: str | Path,
    swift_recipe_path: str | Path,
    dataset_root: Path,
    strict_true_length_audit: bool,
    swift_export: dict[str, Any],
) -> tuple[Path, dict[str, Any]]:
    """Write one resolved-surface artifact so remote commands avoid template ambiguity."""

    resolved = {
        "export_config_path": str(_resolve_path(export_config_path)),
        "swift_recipe_path": str(_resolve_path(swift_recipe_path)),
        "dataset_root": str(dataset_root.resolve()),
        "output_root": str(output_root.resolve()),
        "model_id_or_path": swift_export["recipe"]["training"]["model_id_or_path"],
        "swift_dataset_path": str(swift_export["swift_dataset_path"].resolve()),
        "swift_manifest_path": str(swift_export["swift_manifest_path"].resolve()),
        "length_audit_true_path": str(swift_export["true_length_audit_path"].resolve()),
        "length_audit_proxy_path": str(swift_export["proxy_length_audit_path"].resolve()),
        "length_audit_backend": swift_export["length_audit_backend"],
        "true_multimodal_encode": swift_export["true_multimodal_encode"],
        "threshold_clean_basis": swift_export["threshold_clean_basis"],
        "strict_true_length_audit_requested": strict_true_length_audit,
        "strict_true_length_audit_passed": swift_export["strict_true_length_audit_passed"],
        "filtered_manifests": swift_export["filtered_manifest_paths"],
    }
    path = output_root / "prompt_1_8_resolved_remote_surfaces.json"
    path.write_text(json.dumps(resolved, indent=2, sort_keys=True), encoding="utf-8")
    summary = {
        "dataset_root": resolved["dataset_root"],
        "output_root": resolved["output_root"],
        "length_audit_backend": resolved["length_audit_backend"],
        "true_multimodal_encode": resolved["true_multimodal_encode"],
        "threshold_clean_basis": resolved["threshold_clean_basis"],
        "strict_true_length_audit_requested": resolved["strict_true_length_audit_requested"],
        "strict_true_length_audit_passed": resolved["strict_true_length_audit_passed"],
    }
    return path, summary


def run_prompt_1_5_export(
    *,
    export_config_path: str | Path,
    swift_recipe_path: str | Path,
    dataset_root: str | Path | None = None,
    output_root: str | Path | None = None,
    max_samples_per_mode: int | None = None,
    strict_true_length_audit: bool = False,
) -> SFTArtifacts:
    """Run the full local Prompt 1.5 export, validation, and adapter projection."""

    records, export_metadata = export_sft_dataset(
        config_path=export_config_path,
        dataset_root=dataset_root,
        output_root=output_root,
        max_samples_per_mode=max_samples_per_mode,
    )
    definition = export_metadata["definition"]
    resolved_dataset_root = _resolve_path(dataset_root or definition["sample_source"]["path"])
    resolved_output_root = _resolve_path(output_root or definition["output"]["root"])
    local_validation = local_dataset_sanity(records)
    local_validation["canonical_record_digest"] = sha256_mapping(
        {"records": [sha256_bytes(canonical_json_bytes(record)) for record in records]}
    )

    swift_export = export_swift_dataset(
        canonical_records=records,
        recipe_path=swift_recipe_path,
        output_root=resolved_output_root,
        strict_true_length_audit=strict_true_length_audit,
    )
    resolved_surfaces_path, resolved_surfaces_summary = _write_resolved_remote_surfaces(
        output_root=resolved_output_root,
        export_config_path=export_config_path,
        swift_recipe_path=swift_recipe_path,
        dataset_root=resolved_dataset_root,
        strict_true_length_audit=strict_true_length_audit,
        swift_export=swift_export,
    )
    swift_manifest_payload = json.loads(swift_export["swift_manifest_path"].read_text(encoding="utf-8"))
    swift_manifest_payload["resolved_remote_surfaces_path"] = str(resolved_surfaces_path.resolve())
    validate_payload(swift_manifest_payload, "sft_dataset_manifest.schema.json")
    swift_export["swift_manifest_path"].write_text(
        json.dumps(swift_manifest_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return SFTArtifacts(
        canonical_dataset_path=str((resolved_output_root / definition["output"]["canonical_dataset"]).resolve()),
        canonical_manifest_path=export_metadata["manifest_path"],
        swift_dataset_path=str(swift_export["swift_dataset_path"].resolve()),
        swift_manifest_path=str(swift_export["swift_manifest_path"].resolve()),
        swift_recipe_path=str(_resolve_path(swift_recipe_path)),
        swift_runtime_check=swift_export["runtime_probe"],
        swift_length_audit_path=str(swift_export["true_length_audit_path"].resolve()),
        swift_length_audit_summary=swift_export["true_length_audit"],
        swift_proxy_length_audit_path=str(swift_export["proxy_length_audit_path"].resolve()),
        swift_proxy_length_audit_summary=swift_export["proxy_length_audit"],
        swift_filtered_manifests=swift_export["filtered_manifest_paths"],
        resolved_remote_surfaces_path=str(resolved_surfaces_path.resolve()),
        resolved_remote_surfaces_summary=resolved_surfaces_summary,
        local_validation=local_validation,
    )


def _build_parser() -> argparse.ArgumentParser:
    """Build the local-only SFT export CLI for Prompt 1.5."""

    parser = argparse.ArgumentParser(
        description="Export AgentIAD Prompt 1.5 canonical SFT trajectories and an MS-Swift adapter dataset."
    )
    parser.add_argument("--config", required=True, help="Path to the Prompt 1.5 SFT export definition JSON.")
    parser.add_argument("--swift-recipe", required=True, help="Path to the MS-Swift adapter recipe JSON.")
    parser.add_argument("--dataset-root", help="Optional dataset root override.")
    parser.add_argument("--output-root", help="Optional artifact root override.")
    parser.add_argument("--max-samples-per-mode", type=int, help="Optional sample cap per trajectory mode.")
    parser.add_argument(
        "--strict-true-length-audit",
        action="store_true",
        help="Fail export if true multimodal length audit cannot use a real local processor/tokenizer encode path.",
    )
    return parser


def main() -> int:
    """Run the local Prompt 1.5 export CLI and print the generated artifact paths."""

    args = _build_parser().parse_args()
    artifacts = run_prompt_1_5_export(
        export_config_path=args.config,
        swift_recipe_path=args.swift_recipe,
        dataset_root=args.dataset_root,
        output_root=args.output_root,
        max_samples_per_mode=args.max_samples_per_mode,
        strict_true_length_audit=args.strict_true_length_audit,
    )
    print(
        json.dumps(
            {
                "canonical_dataset_path": artifacts.canonical_dataset_path,
                "canonical_manifest_path": artifacts.canonical_manifest_path,
                "swift_dataset_path": artifacts.swift_dataset_path,
                "swift_manifest_path": artifacts.swift_manifest_path,
                "swift_recipe_path": artifacts.swift_recipe_path,
                "swift_runtime_check": artifacts.swift_runtime_check,
                "swift_length_audit_path": artifacts.swift_length_audit_path,
                "swift_length_audit_summary": artifacts.swift_length_audit_summary,
                "swift_proxy_length_audit_path": artifacts.swift_proxy_length_audit_path,
                "swift_proxy_length_audit_summary": artifacts.swift_proxy_length_audit_summary,
                "swift_filtered_manifests": artifacts.swift_filtered_manifests,
                "resolved_remote_surfaces_path": artifacts.resolved_remote_surfaces_path,
                "resolved_remote_surfaces_summary": artifacts.resolved_remote_surfaces_summary,
                "local_validation": artifacts.local_validation,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
