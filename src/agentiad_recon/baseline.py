"""Canonical inference runner for AgentIAD baseline and tool smoke paths.

This module remains the single inference entrypoint introduced in Prompt 1.3.
Prompt 1.4 extends it to dispatch `no_tools`, `pz_only`, and `pz_cr` runs from
external configs while preserving the same sample layer, prompt/parser helpers,
trace storage, evaluator family, and artifact grammar. Use
`python -m agentiad_recon.baseline --help` for local smoke execution.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agentiad_recon.backends import (
    BackendRequest,
    InferenceBackend,
    MockInferenceBackend,
    MockToolAwareBackend,
    TransformersVisionLanguageBackend,
    VLLMBackendAdapter,
)
from agentiad_recon.behavior_audit import (
    build_zero_tool_behavior_fields,
    summarize_post_pz_transition,
    summarize_post_pz_second_turn_gate,
    summarize_post_pz_transition_sanitation,
    summarize_zero_tool_behavior,
    write_post_pz_second_turn_gate_summary,
    write_post_pz_transition_sidecars,
    write_post_pz_transition_sanitation_sidecars,
    write_tool_first_strategy_summary,
    write_zero_tool_behavior_sidecars,
)
from agentiad_recon.contracts import validate_payload
from agentiad_recon.evaluation import (
    build_delta_report,
    build_metrics_report,
    build_prediction_record,
    build_run_summary,
    empty_tool_usage,
    safe_slug,
    write_json,
    write_jsonl,
)
from agentiad_recon.mmad import MMADIndexer
from agentiad_recon.prompting import (
    BASELINE_PROMPT_VERSION,
    FINAL_ANSWER_PARSER_VERSION,
    PROMPT_VERSION,
    TOOL_FIRST_CONTRACT_STRENGTH,
    TOOL_FIRST_INTERVENTION_STRATEGIES,
    build_baseline_prompt,
    build_prompt,
    extract_think_block,
    parse_final_answer,
)
from agentiad_recon.reproducibility import build_run_metadata, sha256_file
from agentiad_recon.tooling import (
    RETRY_REPAIR_FAILURE_FAMILIES,
    RETRY_REPAIR_ORIGINAL_FAILURE_FAMILY,
    execute_tool_call,
    normalize_protocol_turn,
    parse_tool_call,
    repair_retry_tool_call_output,
    reinsert_tool_result,
)
from agentiad_recon.traces import TraceMessage, TraceRecord


REPO_ROOT = Path(__file__).resolve().parents[2]
FIRST_TURN_PROTOCOL_GATE_MODES = ("off", "retry_once_pz_cr")
POST_PZ_SECOND_TURN_GATE_MODES = ("off", "retry_once_require_cr_after_pz")
ARTIFACT_LEVELS = ("forensic", "throughput")
PROGRESS_MODES = ("off", "auto", "bar", "log")
GENERATION_OVERRIDE_STAGES = (
    "turn0_initial",
    "turn0_retry",
    "post_pz_second_turn",
    "post_pz_second_turn_retry",
    "final_answer",
)


class InferenceRunError(RuntimeError):
    """Raised when a run definition or inference loop violates the contract."""


def _resolve_path(value: str | Path) -> Path:
    """Resolve repository-relative paths deterministically."""

    path = Path(value)
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def load_run_definition(path: str | Path) -> dict[str, Any]:
    """Load and validate either the baseline or tool-enabled run definition."""

    definition = json.loads(Path(path).read_text(encoding="utf-8"))
    mode = definition.get("mode")
    if mode == "no_tools":
        validate_payload(definition, "baseline_run_definition.schema.json")
        return definition
    if mode in {"pz_only", "pz_cr"}:
        validate_payload(definition, "tool_run_definition.schema.json")
        return definition
    raise InferenceRunError(f"Unsupported run mode in config: {mode!r}")


def _infer_checkpoint_provenance(adapter_checkpoint_path: str | None) -> dict[str, Any]:
    """Infer checkpoint step/run-dir lineage from an adapter path when possible."""

    if not adapter_checkpoint_path:
        return {"checkpoint_step": None, "checkpoint_run_dir": None}

    path = _resolve_path(adapter_checkpoint_path)
    checkpoint_step = None
    if path.name.startswith("checkpoint-"):
        try:
            checkpoint_step = int(path.name.split("-", 1)[1])
        except ValueError:
            checkpoint_step = None
    checkpoint_run_dir = str(path.parent.resolve()) if path.parent != path else None
    return {
        "checkpoint_step": checkpoint_step,
        "checkpoint_run_dir": checkpoint_run_dir,
    }


def _runtime_defaults() -> dict[str, Any]:
    """Return deterministic runtime defaults for auditable eval runs."""

    return {
        "base_model_path": None,
        "adapter_checkpoint_path": None,
        "checkpoint_step": None,
        "checkpoint_run_dir": None,
        "local_files_only": True,
        "trust_remote_code": True,
        "dtype": "auto",
        "device": "auto",
        "generation": {
            "max_new_tokens": 512,
            "do_sample": False,
            "temperature": 0.0,
            "top_p": 1.0,
        },
        "generation_stage_overrides": {},
        "tool_first_intervention_strategy": "baseline",
        "first_turn_protocol_gate_mode": "off",
        "post_pz_second_turn_gate_mode": "off",
        "emit_baseline_compare": True,
        "emit_delta_report": True,
        "artifact_level": "forensic",
        "timing_enabled": False,
        "progress_mode": "off",
        "progress_update_every_n_samples": 1,
        "progress_snapshot_path": None,
    }


def _normalize_generation_stage_overrides(overrides: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    """Validate and normalize stage-specific generation overrides."""

    normalized: dict[str, dict[str, Any]] = {}
    for stage, values in dict(overrides or {}).items():
        if stage not in GENERATION_OVERRIDE_STAGES:
            raise InferenceRunError(f"Unsupported generation override stage: {stage!r}")
        if not isinstance(values, dict):
            raise InferenceRunError(f"Generation override for stage {stage!r} must be an object")
        normalized[stage] = {
            key: value
            for key, value in values.items()
            if key in {"max_new_tokens", "do_sample", "temperature", "top_p"} and value is not None
        }
    return normalized


def _generation_config_for_stage(runtime_config: dict[str, Any], stage: str) -> dict[str, Any]:
    """Resolve one generation config for a specific runtime stage."""

    config = dict(runtime_config["generation"])
    config.update(runtime_config["generation_stage_overrides"].get(stage, {}))
    return config


def _new_timing_counters() -> dict[str, float | int]:
    """Return one zero-initialized per-sample timing bucket payload."""

    return {
        "prompt_render_ms": 0.0,
        "request_build_or_processor_ms": 0.0,
        "generate_ms": 0.0,
        "tool_exec_ms": 0.0,
        "parse_validate_ms": 0.0,
        "file_write_ms": 0.0,
        "tail_compare_delta_ms": 0.0,
        "generation_call_count": 0,
        "retry_count": 0,
    }


def _elapsed_ms(start_time: float) -> float:
    """Convert one perf-counter start point into elapsed milliseconds."""

    return (time.perf_counter() - start_time) * 1000.0


def _sha256_text(value: str) -> str:
    """Hash one UTF-8 text payload without requiring file retention."""

    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _utc_now_iso() -> str:
    """Return one deterministic UTC timestamp string."""

    return datetime.now(timezone.utc).isoformat()


class ProgressReporter:
    """Small progress helper for interactive and redirected unified eval runs."""

    def __init__(
        self,
        *,
        run_id: str,
        total_samples: int,
        mode: str,
        update_every_n_samples: int,
        snapshot_path: Path | None,
    ) -> None:
        self.run_id = run_id
        self.total_samples = max(total_samples, 0)
        self.mode = mode
        self.update_every_n_samples = max(update_every_n_samples, 1)
        self.snapshot_path = snapshot_path
        self.start_time = time.perf_counter()
        self.last_emitted_processed = 0
        self._bar_active = False
        self._resolved_mode = self._resolve_mode()

    def _resolve_mode(self) -> str:
        if self.mode == "off":
            return "off"
        if self.mode == "log":
            return "log"
        if self.mode == "bar":
            return "bar"
        return "bar" if getattr(sys.stderr, "isatty", lambda: False)() else "log"

    def _snapshot_payload(
        self,
        *,
        processed_samples: int,
        current_sample_id: str | None,
        timing_summary: dict[str, Any],
    ) -> dict[str, Any]:
        elapsed_seconds = max(time.perf_counter() - self.start_time, 0.0)
        avg_seconds_per_sample = (
            elapsed_seconds / processed_samples if processed_samples > 0 else 0.0
        )
        remaining = max(self.total_samples - processed_samples, 0)
        eta_seconds = avg_seconds_per_sample * remaining if processed_samples > 0 else None
        percent_complete = (
            (processed_samples / self.total_samples) * 100.0 if self.total_samples > 0 else 100.0
        )
        return {
            "run_id": self.run_id,
            "processed_samples": processed_samples,
            "total_samples": self.total_samples,
            "percent_complete": percent_complete,
            "elapsed_seconds": elapsed_seconds,
            "avg_seconds_per_sample": avg_seconds_per_sample,
            "eta_seconds": eta_seconds,
            "current_sample_id": current_sample_id,
            "generation_call_count_total": timing_summary["generation_call_count_total"],
            "retry_count_total": timing_summary["retry_count_total"],
            "last_update_utc": _utc_now_iso(),
        }

    def _write_snapshot(
        self,
        *,
        processed_samples: int,
        current_sample_id: str | None,
        timing_summary: dict[str, Any],
    ) -> None:
        if self.snapshot_path is None:
            return
        write_json(
            self.snapshot_path,
            self._snapshot_payload(
                processed_samples=processed_samples,
                current_sample_id=current_sample_id,
                timing_summary=timing_summary,
            ),
        )

    def update(
        self,
        *,
        processed_samples: int,
        current_sample_id: str | None,
        timing_summary: dict[str, Any],
        force: bool = False,
    ) -> None:
        if self._resolved_mode == "off":
            return
        if (
            not force
            and processed_samples < self.total_samples
            and processed_samples - self.last_emitted_processed < self.update_every_n_samples
        ):
            return

        snapshot = self._snapshot_payload(
            processed_samples=processed_samples,
            current_sample_id=current_sample_id,
            timing_summary=timing_summary,
        )
        self._write_snapshot(
            processed_samples=processed_samples,
            current_sample_id=current_sample_id,
            timing_summary=timing_summary,
        )

        elapsed_seconds = snapshot["elapsed_seconds"]
        eta_seconds = snapshot["eta_seconds"]
        avg_seconds_per_sample = snapshot["avg_seconds_per_sample"]
        current = current_sample_id or "-"
        if self._resolved_mode == "bar":
            bar_width = 24
            filled = (
                int((processed_samples / self.total_samples) * bar_width)
                if self.total_samples > 0
                else bar_width
            )
            bar = "#" * filled + "-" * max(bar_width - filled, 0)
            line = (
                f"\r[{bar}] {processed_samples}/{self.total_samples} "
                f"({snapshot['percent_complete']:.1f}%) elapsed={elapsed_seconds:.1f}s "
                f"avg={avg_seconds_per_sample:.2f}s/sample eta={eta_seconds or 0.0:.1f}s "
                f"sample={current}"
            )
            sys.stderr.write(line)
            sys.stderr.flush()
            self._bar_active = True
            if processed_samples >= self.total_samples:
                sys.stderr.write("\n")
                sys.stderr.flush()
                self._bar_active = False
        else:
            sys.stderr.write(
                "[progress] "
                f"processed={processed_samples}/{self.total_samples} "
                f"percent={snapshot['percent_complete']:.1f} "
                f"elapsed_s={elapsed_seconds:.1f} "
                f"avg_s_per_sample={avg_seconds_per_sample:.2f} "
                f"eta_s={(eta_seconds or 0.0):.1f} "
                f"sample_id={current}\n"
            )
            sys.stderr.flush()

        self.last_emitted_processed = processed_samples


def _timing_summary_from_prediction_records(
    prediction_records: list[dict[str, Any]],
    *,
    tail_compare_delta_ms_total: float = 0.0,
) -> dict[str, Any]:
    """Aggregate per-sample timing metadata into one run-level timing summary."""

    totals = _new_timing_counters()
    for record in prediction_records:
        timing = record.get("metadata", {}).get("timing", {})
        for key in (
            "prompt_render_ms",
            "request_build_or_processor_ms",
            "generate_ms",
            "tool_exec_ms",
            "parse_validate_ms",
            "file_write_ms",
            "tail_compare_delta_ms",
        ):
            totals[key] += float(timing.get(key, 0.0))
        for key in ("generation_call_count", "retry_count"):
            totals[key] += int(timing.get(key, 0))

    totals["tail_compare_delta_ms"] += tail_compare_delta_ms_total
    sample_count = len(prediction_records)
    return {
        "prompt_render_ms_total": totals["prompt_render_ms"],
        "request_build_or_processor_ms_total": totals["request_build_or_processor_ms"],
        "generate_ms_total": totals["generate_ms"],
        "tool_exec_ms_total": totals["tool_exec_ms"],
        "parse_validate_ms_total": totals["parse_validate_ms"],
        "file_write_ms_total": totals["file_write_ms"],
        "tail_compare_delta_ms_total": totals["tail_compare_delta_ms"],
        "generation_call_count_total": totals["generation_call_count"],
        "retry_count_total": totals["retry_count"],
        "avg_generate_ms_per_sample": (
            totals["generate_ms"] / sample_count if sample_count > 0 else 0.0
        ),
    }


def _runtime_config(
    definition: dict[str, Any],
    *,
    dataset_root: str | Path | None,
    runtime_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Resolve runtime settings from config plus optional CLI overrides."""

    config = _runtime_defaults()
    config.update(definition.get("runtime", {}))
    generation_overrides = dict(runtime_overrides.get("generation", {})) if runtime_overrides else {}
    config["generation"] = dict(config["generation"])
    config["generation"].update(generation_overrides)
    config["generation_stage_overrides"] = _normalize_generation_stage_overrides(
        config.get("generation_stage_overrides")
    )

    for key in (
        "base_model_path",
        "adapter_checkpoint_path",
        "checkpoint_step",
        "checkpoint_run_dir",
        "local_files_only",
        "trust_remote_code",
        "dtype",
        "device",
        "tool_first_intervention_strategy",
        "first_turn_protocol_gate_mode",
        "post_pz_second_turn_gate_mode",
        "emit_baseline_compare",
        "emit_delta_report",
        "artifact_level",
        "timing_enabled",
        "progress_mode",
        "progress_update_every_n_samples",
        "progress_snapshot_path",
    ):
        if runtime_overrides and key in runtime_overrides and runtime_overrides[key] is not None:
            config[key] = runtime_overrides[key]
    if runtime_overrides and runtime_overrides.get("generation_stage_overrides") is not None:
        config["generation_stage_overrides"] = _normalize_generation_stage_overrides(
            runtime_overrides["generation_stage_overrides"]
        )

    inferred = _infer_checkpoint_provenance(config["adapter_checkpoint_path"])
    if config["checkpoint_step"] is None:
        config["checkpoint_step"] = inferred["checkpoint_step"]
    if config["checkpoint_run_dir"] is None:
        config["checkpoint_run_dir"] = inferred["checkpoint_run_dir"]
    if config["tool_first_intervention_strategy"] not in TOOL_FIRST_INTERVENTION_STRATEGIES:
        raise InferenceRunError(
            "Unsupported tool_first_intervention_strategy: "
            f"{config['tool_first_intervention_strategy']!r}"
        )
    if config["first_turn_protocol_gate_mode"] not in FIRST_TURN_PROTOCOL_GATE_MODES:
        raise InferenceRunError(
            "Unsupported first_turn_protocol_gate_mode: "
            f"{config['first_turn_protocol_gate_mode']!r}"
        )
    if config["post_pz_second_turn_gate_mode"] not in POST_PZ_SECOND_TURN_GATE_MODES:
        raise InferenceRunError(
            "Unsupported post_pz_second_turn_gate_mode: "
            f"{config['post_pz_second_turn_gate_mode']!r}"
        )
    if config["artifact_level"] not in ARTIFACT_LEVELS:
        raise InferenceRunError(f"Unsupported artifact_level: {config['artifact_level']!r}")
    if config["progress_mode"] not in PROGRESS_MODES:
        raise InferenceRunError(f"Unsupported progress_mode: {config['progress_mode']!r}")
    if int(config["progress_update_every_n_samples"]) < 1:
        raise InferenceRunError("progress_update_every_n_samples must be >= 1")

    resolved_dataset_root = _resolve_path(dataset_root or definition["sample_source"]["path"])
    config["dataset_root"] = str(resolved_dataset_root)
    config["tool_mode"] = definition["mode"]
    config["inference_mode"] = definition["mode"]
    config["runtime_backend_name"] = definition["backend"]["name"]
    config["runtime_backend_type"] = definition["backend"]["type"]
    config["runtime_owner"] = definition["backend"]["runtime_owner"]
    if config["progress_snapshot_path"] is not None:
        config["progress_snapshot_path"] = str(_resolve_path(config["progress_snapshot_path"]))
    return config


def _resolved_runtime_provenance(
    definition: dict[str, Any],
    runtime_config: dict[str, Any],
    backend: InferenceBackend,
) -> dict[str, Any]:
    """Merge requested runtime settings with backend-reported effective state."""

    backend_runtime = backend.describe_runtime()
    return {
        "runtime_backend_name": definition["backend"]["name"],
        "runtime_backend_type": definition["backend"]["type"],
        "runtime_owner": definition["backend"]["runtime_owner"],
        "policy": definition["backend"]["policy"],
        "base_model_path": runtime_config["base_model_path"],
        "adapter_checkpoint_path": runtime_config["adapter_checkpoint_path"],
        "adapter_loaded": backend_runtime.get("adapter_loaded", False),
        "checkpoint_step": runtime_config["checkpoint_step"],
        "checkpoint_run_dir": runtime_config["checkpoint_run_dir"],
        "dataset_root": runtime_config["dataset_root"],
        "tool_mode": runtime_config["tool_mode"],
        "inference_mode": runtime_config["inference_mode"],
        "generation_config": dict(runtime_config["generation"]),
        "generation_stage_overrides": runtime_config["generation_stage_overrides"],
        "local_files_only": runtime_config["local_files_only"],
        "trust_remote_code": runtime_config["trust_remote_code"],
        "dtype": runtime_config["dtype"],
        "device": runtime_config["device"],
        "tool_first_intervention_strategy": runtime_config["tool_first_intervention_strategy"],
        "first_turn_protocol_gate_mode": runtime_config["first_turn_protocol_gate_mode"],
        "post_pz_second_turn_gate_mode": runtime_config["post_pz_second_turn_gate_mode"],
        "emit_baseline_compare": runtime_config["emit_baseline_compare"],
        "emit_delta_report": runtime_config["emit_delta_report"],
        "artifact_level": runtime_config["artifact_level"],
        "timing_enabled": runtime_config["timing_enabled"],
        "progress_mode": runtime_config["progress_mode"],
        "progress_update_every_n_samples": runtime_config["progress_update_every_n_samples"],
        "progress_snapshot_path": runtime_config["progress_snapshot_path"],
    }


def _select_backend(config: dict[str, Any], *, runtime_config: dict[str, Any] | None = None) -> InferenceBackend:
    """Instantiate the requested backend while keeping runtime ownership thin."""

    backend_type = config["type"]
    if backend_type == "mock":
        if config["policy"] == "fixture_scripted_non_tool_v1":
            return MockInferenceBackend(
                backend_name=config["name"],
                policy=config["policy"],
                runtime_config=runtime_config,
            )
        return MockToolAwareBackend(
            backend_name=config["name"],
            policy=config["policy"],
            runtime_config=runtime_config,
        )
    if backend_type == "transformers":
        return TransformersVisionLanguageBackend(
            backend_name=config["name"],
            policy=config["policy"],
            runtime_config=runtime_config,
        )
    if backend_type == "vllm":
        return VLLMBackendAdapter(
            backend_name=config["name"],
            runtime_config=runtime_config,
        )
    raise InferenceRunError(f"Unsupported backend type: {backend_type}")


def _load_samples(
    definition: dict[str, Any],
    *,
    dataset_root: str | Path | None = None,
    max_samples: int | None = None,
) -> list[dict[str, Any]]:
    """Load canonical samples from the Prompt 1.2 MMAD indexer."""

    sample_source = definition["sample_source"]
    resolved_root = _resolve_path(dataset_root or sample_source["path"])
    source_name = sample_source["source_name"]
    samples = [sample.to_dict() for sample in MMADIndexer(resolved_root, source=source_name).index_samples()]
    split_filter = set(sample_source["splits"])
    samples = [sample for sample in samples if sample["split"] in split_filter]
    limit = max_samples if max_samples is not None else sample_source["max_samples"]
    if limit is not None:
        samples = samples[:limit]
    if not samples:
        raise InferenceRunError("No canonical samples were selected for the inference run.")
    return samples


def _write_text(path: str | Path, payload: str) -> None:
    """Write one UTF-8 text artifact to disk."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")


def _parse_bool_flag(value: str) -> bool:
    """Parse one explicit CLI boolean flag value."""

    normalized = value.strip().lower()
    if normalized in {"true", "1", "yes", "on"}:
        return True
    if normalized in {"false", "0", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got: {value!r}")


def _default_progress_snapshot_path(root: Path) -> Path:
    """Return the default progress snapshot path for one run root."""

    return root / "progress" / "progress_snapshot.json"


def _should_write_sample_artifacts(
    *,
    artifact_level: str,
    failure_reason: str | None,
    first_turn_gate_triggered: bool,
    first_turn_gate_outcome: str,
    first_turn_gate_repair_attempted: bool,
    first_turn_gate_repair_succeeded: bool,
    post_pz_second_turn_gate_triggered: bool,
    post_pz_second_turn_gate_outcome: str,
) -> bool:
    """Return whether optional per-sample raw outputs and sidecars should be retained."""

    if artifact_level == "forensic":
        return True
    if failure_reason:
        return True
    if first_turn_gate_triggered or post_pz_second_turn_gate_triggered:
        return True
    if first_turn_gate_outcome == "recovered_to_tool_call":
        return True
    if post_pz_second_turn_gate_outcome == "recovered_to_cr_call":
        return True
    if first_turn_gate_repair_attempted or first_turn_gate_repair_succeeded:
        return True
    return False


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
    """Append one reasoning trace message when a `<think>` block is present."""

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
    raw_output_path: str,
    tool_name: str | None,
    call_id: str | None,
    error_message: str | None = None,
    extra_metadata: dict[str, Any] | None = None,
) -> None:
    """Append one assistant tool request, including invalid attempts for audit."""

    metadata = {"backend_name": backend_name, "raw_output_path": raw_output_path}
    if error_message is not None:
        metadata["error_message"] = error_message
    if extra_metadata:
        metadata.update(extra_metadata)
    history.append(
        {
            "role": "assistant",
            "message_type": "tool_request",
            "content": raw_output,
            "image_refs": [],
            "metadata": metadata,
            "tool_name": tool_name,
            "call_id": call_id,
        }
    )


def _append_final_answer_message(
    history: list[dict[str, Any]],
    raw_output: str,
    *,
    backend_name: str,
    raw_output_path: str,
    error_message: str | None = None,
) -> None:
    """Append the assistant final-answer turn to the audit history."""

    metadata = {"backend_name": backend_name, "raw_output_path": raw_output_path}
    if error_message is not None:
        metadata["error_message"] = error_message
    history.append(
        {
            "role": "assistant",
            "message_type": "final_answer",
            "content": raw_output,
            "image_refs": [],
            "metadata": metadata,
            "tool_name": None,
            "call_id": None,
        }
    )


def _append_runtime_gate_reminder(
    history: list[dict[str, Any]],
    reminder_text: str,
    *,
    gate_mode: str,
    runtime_intervention: str = "first_turn_protocol_gate",
    gate_mode_metadata_key: str = "first_turn_protocol_gate_mode",
) -> None:
    """Append the auditable user-side runtime reminder used by the protocol gate."""

    history.append(
        {
            "role": "user",
            "message_type": "user_prompt",
            "content": reminder_text,
            "image_refs": [],
            "metadata": {
                "runtime_intervention": runtime_intervention,
                gate_mode_metadata_key: gate_mode,
            },
            "tool_name": None,
            "call_id": None,
        }
    )


def _history_to_trace_messages(history: list[dict[str, Any]]) -> tuple[TraceMessage, ...]:
    """Project the mutable history into the canonical trace dataclasses."""

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


def _tool_usage_from_traces(tool_traces: list[dict[str, Any]]) -> dict[str, Any]:
    """Collapse executed tool traces into the normalized per-sample usage shape."""

    per_tool_counts = {tool: 0 for tool in ("PZ", "CR")}
    for trace in tool_traces:
        per_tool_counts[trace["tool_name"]] += 1
    total_calls = len(tool_traces)
    return {
        "total_calls": total_calls,
        "samples_with_tool_call": 1 if total_calls else 0,
        "per_tool_counts": per_tool_counts,
    }


def _raw_output_path(raw_root: Path, *, seed: int, sample_slug: str, turn_index: int) -> Path:
    """Build a deterministic raw-output path for one assistant turn."""

    return raw_root / f"seed_{seed}" / sample_slug / f"turn_{turn_index}.txt"


def _normalization_sidecar_path(raw_output_path: Path) -> Path:
    """Build the sidecar path for one auditable normalization event."""

    return raw_output_path.with_suffix(".normalization.json")


def _first_turn_gate_sidecar_path(raw_root: Path, *, seed: int, sample_slug: str) -> Path:
    """Build the sidecar path for one first-turn protocol gate event."""

    return raw_root / f"seed_{seed}" / sample_slug / "first_turn_gate.turn_0.json"


def _retry_raw_output_path(raw_root: Path, *, seed: int, sample_slug: str, turn_index: int, retry_count: int) -> Path:
    """Build the raw-output path for one auditable retry attempt."""

    return raw_root / f"seed_{seed}" / sample_slug / f"turn_{turn_index}.retry_{retry_count}.txt"


def _prompt_audit_sidecar_path(raw_root: Path, *, seed: int, sample_slug: str) -> Path:
    """Build the prompt-audit sidecar path for one sample's first turn."""

    return raw_root / f"seed_{seed}" / sample_slug / "prompt_audit.turn_0.json"


def _post_pz_transition_sidecar_path(
    raw_root: Path,
    *,
    seed: int,
    sample_slug: str,
    turn_index: int,
) -> Path:
    """Build the sidecar path for one post-PZ transition audit event."""

    return raw_root / f"seed_{seed}" / sample_slug / f"post_pz_transition.turn_{turn_index}.json"


def _post_pz_second_turn_gate_sidecar_path(
    raw_root: Path,
    *,
    seed: int,
    sample_slug: str,
    turn_index: int,
) -> Path:
    """Build the sidecar path for one bounded post-PZ second-turn gate event."""

    return raw_root / f"seed_{seed}" / sample_slug / f"post_pz_second_turn_gate.turn_{turn_index}.json"


def _sanitize_failure_detail(text: str) -> str:
    """Convert a free-form error detail into a stable machine-readable token."""

    token = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return token[:96] or "unspecified"


def _failure_reason(prefix: str, detail: str | None = None) -> str:
    """Render one structured failure reason with an optional normalized detail."""

    if detail is None or not detail.strip():
        return prefix
    return f"{prefix}:{_sanitize_failure_detail(detail)}"


def _resolve_failure_reason(
    *,
    prediction: dict[str, Any] | None,
    parser_valid: bool,
    schema_valid: bool,
    failure_reason: str | None,
    error_message: str | None,
) -> str | None:
    """Guarantee that every failed sample carries a concrete machine-readable reason."""

    failed = prediction is None or not parser_valid or not schema_valid
    if not failed:
        return None
    if failure_reason is not None and failure_reason.strip():
        return failure_reason
    if error_message is not None and error_message.strip():
        return _failure_reason("runtime_exception", error_message)
    return "runtime_exception:missing_failure_reason_context"


def _prompt_audit_payload(
    prompt_bundle: Any,
    *,
    sample_id: str,
    seed: int,
    turn_index: int,
    runtime_tool_mode: str,
    tool_first_intervention_strategy: str,
) -> dict[str, Any]:
    """Build an auditable snapshot of the rendered prompt/tool surface."""

    system_text = "\n\n".join(
        message["content"] for message in prompt_bundle.messages if message["role"] == "system"
    )
    user_text = "\n\n".join(
        message["content"] for message in prompt_bundle.messages if message["role"] == "user"
    )
    rendered_text = "\n\n".join(
        message["content"] for message in prompt_bundle.messages if isinstance(message.get("content"), str)
    )
    rendered_text_lower = rendered_text.lower()
    if "available tools: pz and cr" in rendered_text_lower:
        declared_available_tools = ["PZ", "CR"]
    elif "available tools: pz only" in rendered_text_lower:
        declared_available_tools = ["PZ"]
    else:
        declared_available_tools = []
        if re.search(r"\bPZ\b", rendered_text) is not None:
            declared_available_tools.append("PZ")
        if re.search(r"\bCR\b", rendered_text) is not None and "do not request cr in this mode" not in rendered_text_lower:
            declared_available_tools.append("CR")

    prompt_contains_pz_only = (
        "pz_only" in rendered_text_lower or "available tools: pz only" in rendered_text_lower
    )
    prompt_contains_pz_cr = "pz_cr" in rendered_text_lower
    cr_available_in_prompt_surface = "available tools: pz and cr" in rendered_text_lower
    prompt_surface_digest = hashlib.sha256(rendered_text.encode("utf-8")).hexdigest()
    tool_first_marker_present = "tool-first intervention strategy:" in rendered_text_lower
    intervention_text_applied = tool_first_marker_present
    mismatch_reasons: list[str] = []
    if runtime_tool_mode == "pz_cr":
        if prompt_contains_pz_only:
            mismatch_reasons.append("mode_contract_pz_only_leakage")
        if not cr_available_in_prompt_surface or "CR" not in declared_available_tools:
            mismatch_reasons.append("mode_contract_missing_cr_tool")

    return {
        "sample_id": sample_id,
        "seed": seed,
        "turn_index": turn_index,
        "runtime_tool_mode": runtime_tool_mode,
        "tool_first_intervention_strategy": tool_first_intervention_strategy,
        "tool_first_contract_strength": TOOL_FIRST_CONTRACT_STRENGTH[tool_first_intervention_strategy],
        "intervention_text_applied": intervention_text_applied,
        "prompt_surface_digest": prompt_surface_digest,
        "system_text": system_text,
        "user_text": user_text,
        "rendered_text": rendered_text,
        "declared_available_tools": declared_available_tools,
        "prompt_contains_pz_only": prompt_contains_pz_only,
        "prompt_contains_pz_cr": prompt_contains_pz_cr,
        "prompt_surface_contains_tool_first_marker": tool_first_marker_present,
        "tool_surface_contains_pz": re.search(r"\bPZ\b", rendered_text) is not None,
        "tool_surface_contains_cr": re.search(r"\bCR\b", rendered_text) is not None,
        "cr_available_in_prompt_surface": cr_available_in_prompt_surface,
        "mode_contract_mismatch": bool(mismatch_reasons),
        "mismatch_reasons": mismatch_reasons,
    }


def _normalization_summary(prediction_records: list[dict[str, Any]]) -> dict[str, int]:
    """Aggregate mixed-output normalization counts across prediction records."""

    summary = {
        "event_count": 0,
        "samples_with_events": 0,
        "mixed_tool_call_and_final_answer_count": 0,
        "multiple_tool_calls_in_single_output_count": 0,
        "discarded_premature_final_answer_count": 0,
        "additional_valid_tool_calls_discarded_count": 0,
    }
    for record in prediction_records:
        events = record.get("metadata", {}).get("normalization_events", [])
        if events:
            summary["samples_with_events"] += 1
        for event in events:
            summary["event_count"] += 1
            if event.get("reason") == "mixed_tool_call_and_final_answer":
                summary["mixed_tool_call_and_final_answer_count"] += 1
            if event.get("reason") == "multiple_tool_calls_in_single_output":
                summary["multiple_tool_calls_in_single_output_count"] += 1
            if event.get("discarded_final_answer_present"):
                summary["discarded_premature_final_answer_count"] += 1
            summary["additional_valid_tool_calls_discarded_count"] += int(
                event.get("additional_valid_tool_calls_discarded", 0)
            )
    return summary


def _prompt_audit_summary(prediction_records: list[dict[str, Any]]) -> dict[str, int]:
    """Aggregate prompt-audit coverage and mismatch counts across prediction records."""

    summary = {
        "prompt_audit_event_count": 0,
        "samples_with_prompt_audit_mismatch": 0,
        "mode_contract_missing_cr_tool_count": 0,
        "mode_contract_pz_only_leakage_count": 0,
        "failed_count_with_missing_reason_count": 0,
    }
    for record in prediction_records:
        prompt_audit = record.get("metadata", {}).get("prompt_audit")
        if prompt_audit is not None:
            summary["prompt_audit_event_count"] += 1
            mismatch_reasons = set(prompt_audit.get("mismatch_reasons", []))
            if prompt_audit.get("mode_contract_mismatch"):
                summary["samples_with_prompt_audit_mismatch"] += 1
            if "mode_contract_missing_cr_tool" in mismatch_reasons:
                summary["mode_contract_missing_cr_tool_count"] += 1
            if "mode_contract_pz_only_leakage" in mismatch_reasons:
                summary["mode_contract_pz_only_leakage_count"] += 1
        failed = (record["prediction"] is None) or (not record["parser_valid"]) or (not record["schema_valid"])
        if failed and (record.get("failure_reason") is None or not str(record["failure_reason"]).strip()):
            summary["failed_count_with_missing_reason_count"] += 1
    return summary


def _render_history_prompt_surface(history: list[dict[str, Any]]) -> str:
    """Render the current multi-message history into one inspectable prompt surface string."""

    rendered_messages: list[str] = []
    for index, message in enumerate(history):
        header = f"[{index}] role={message['role']} message_type={message['message_type']}"
        if message.get("tool_name") is not None:
            header += f" tool_name={message['tool_name']}"
        if message.get("call_id") is not None:
            header += f" call_id={message['call_id']}"
        lines = [header, str(message.get("content", ""))]
        if message.get("image_refs"):
            lines.append(f"image_refs={json.dumps(message['image_refs'], ensure_ascii=False)}")
        rendered_messages.append("\n".join(lines))
    return "\n\n".join(rendered_messages)


def _clone_history(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Copy one mutable dialogue history without changing any message semantics."""

    return [
        {
            "role": message["role"],
            "message_type": message["message_type"],
            "content": message["content"],
            "image_refs": list(message.get("image_refs", [])),
            "metadata": dict(message.get("metadata", {})),
            "tool_name": message.get("tool_name"),
            "call_id": message.get("call_id"),
        }
        for message in history
    ]


def _message_digest(message: dict[str, Any]) -> str:
    """Build one stable digest for a removed or audited runtime message."""

    return hashlib.sha256(
        json.dumps(
            {
                "role": message["role"],
                "message_type": message["message_type"],
                "content": message.get("content"),
                "image_refs": message.get("image_refs", []),
                "metadata": message.get("metadata", {}),
                "tool_name": message.get("tool_name"),
                "call_id": message.get("call_id"),
            },
            ensure_ascii=False,
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def _message_contains_pz_only_leakage(message: dict[str, Any]) -> bool:
    """Detect the narrow stale pz_only-style leakage targeted by Prompt 2.11."""

    content = str(message.get("content", "")).lower()
    return any(
        phrase in content
        for phrase in (
            "pz_only",
            "available tools: pz only",
            "do not request cr in this mode",
            "no comparative retrieval is allowed in pz_only",
        )
    )


def _remove_assistant_messages_before_runtime_intervention(
    history: list[dict[str, Any]],
    *,
    runtime_intervention: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Remove superseded assistant messages immediately before one runtime reminder."""

    sanitized_history = _clone_history(history)
    reminder_index: int | None = None
    for index, message in enumerate(sanitized_history):
        if (
            message["role"] == "user"
            and message.get("metadata", {}).get("runtime_intervention") == runtime_intervention
        ):
            reminder_index = index

    if reminder_index is None:
        return sanitized_history, []

    prior_user_index = None
    for index in range(reminder_index - 1, -1, -1):
        if sanitized_history[index]["role"] == "user":
            prior_user_index = index
            break

    removal_indices = [
        index
        for index in range((prior_user_index or -1) + 1, reminder_index)
        if sanitized_history[index]["role"] == "assistant"
    ]
    if not removal_indices:
        return sanitized_history, []

    removal_index_set = set(removal_indices)
    removed_messages = [sanitized_history[index] for index in removal_indices]
    sanitized_history = [
        message for index, message in enumerate(sanitized_history) if index not in removal_index_set
    ]
    return sanitized_history, removed_messages


def _post_pz_declared_available_tools(rendered_prompt_surface: str) -> list[str]:
    """Infer which tools are still explicitly exposed on the post-PZ prompt surface."""

    rendered_lower = rendered_prompt_surface.lower()
    if "available tools: pz and cr" in rendered_lower:
        return ["PZ", "CR"]
    if "available tools: pz only" in rendered_lower:
        return ["PZ"]

    declared_available_tools: list[str] = []
    if "crop_image_normalized" in rendered_lower or re.search(r"\bPZ\b", rendered_prompt_surface) is not None:
        declared_available_tools.append("PZ")
    if (
        "query_image" in rendered_lower
        or "same-category normal reference exemplar" in rendered_lower
        or re.search(r"\bCR\b", rendered_prompt_surface) is not None
    ) and "do not request cr in this mode" not in rendered_lower:
        declared_available_tools.append("CR")
    return declared_available_tools


def _post_pz_transition_protocol_event_type(event: str) -> str:
    """Normalize the post-PZ second-turn protocol event for audit surfaces."""

    return event if event in {"tool_call", "final_answer"} else "parse_failure"


def _post_pz_transition_contract_fields(history: list[dict[str, Any]]) -> dict[str, Any]:
    """Inspect one post-PZ rendered surface and classify its CR-transition validity."""

    rendered_prompt_surface = _render_history_prompt_surface(history)
    rendered_prompt_surface_digest = hashlib.sha256(
        rendered_prompt_surface.encode("utf-8")
    ).hexdigest()
    rendered_prompt_surface_lower = rendered_prompt_surface.lower()
    declared_available_tools = _post_pz_declared_available_tools(rendered_prompt_surface)
    prompt_contains_pz_only = (
        "pz_only" in rendered_prompt_surface_lower
        or "available tools: pz only" in rendered_prompt_surface_lower
        or "do not request cr in this mode" in rendered_prompt_surface_lower
    )

    pz_tool_result_messages = [
        message
        for message in history
        if message["role"] == "tool"
        and message["message_type"] == "tool_result"
        and message.get("tool_name") == "PZ"
    ]
    latest_pz_tool_result = pz_tool_result_messages[-1] if pz_tool_result_messages else None
    pz_result_reinserted_present = latest_pz_tool_result is not None
    pz_result_reinserted_digest = None
    if latest_pz_tool_result is not None:
        pz_result_reinserted_digest = hashlib.sha256(
            json.dumps(
                {
                    "tool_name": latest_pz_tool_result.get("tool_name"),
                    "call_id": latest_pz_tool_result.get("call_id"),
                    "content": latest_pz_tool_result.get("content"),
                    "image_refs": latest_pz_tool_result.get("image_refs", []),
                    "metadata": latest_pz_tool_result.get("metadata", {}),
                },
                ensure_ascii=False,
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()

    cr_available_in_prompt_surface = "CR" in declared_available_tools
    query_image_instruction_present = any(
        phrase in rendered_prompt_surface_lower
        for phrase in (
            "query_image",
            "same-category normal reference exemplar",
            "cr only when",
            "available tools: pz and cr",
        )
    )
    crop_image_normalized_reference_present = any(
        phrase in rendered_prompt_surface_lower
        for phrase in (
            "crop_image_normalized",
            "localized crop/zoom",
            "tool_name=pz",
            '"tool_name":"pz"',
        )
    )
    tool_context_present = any(
        message["role"] == "assistant"
        and message["message_type"] == "tool_request"
        and message.get("tool_name") == "PZ"
        for message in history
    ) and pz_result_reinserted_present

    transition_mismatch_reasons: list[str] = []
    if not pz_result_reinserted_present:
        transition_mismatch_reasons.append("post_pz_transition_missing_reinserted_pz_result")
    if not cr_available_in_prompt_surface:
        transition_mismatch_reasons.append("post_pz_transition_missing_cr_tool")
    if not query_image_instruction_present:
        transition_mismatch_reasons.append("post_pz_transition_missing_query_image_instruction")
    if prompt_contains_pz_only:
        transition_mismatch_reasons.append("post_pz_transition_pz_only_leakage")
    if not tool_context_present:
        transition_mismatch_reasons.append("post_pz_transition_missing_tool_context")

    return {
        "rendered_prompt_surface": rendered_prompt_surface,
        "rendered_prompt_surface_digest": rendered_prompt_surface_digest,
        "declared_available_tools": declared_available_tools,
        "cr_available_in_prompt_surface": cr_available_in_prompt_surface,
        "query_image_instruction_present": query_image_instruction_present,
        "crop_image_normalized_reference_present": crop_image_normalized_reference_present,
        "pz_result_reinserted_present": pz_result_reinserted_present,
        "pz_result_reinserted_digest": pz_result_reinserted_digest,
        "transition_contract_valid_for_cr": not transition_mismatch_reasons,
        "transition_mismatch_reasons": transition_mismatch_reasons,
        "pz_only_leakage_present": prompt_contains_pz_only,
    }


def _sanitize_post_pz_transition_history(
    history: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Remove only superseded first-turn assistant branch messages from the active post-PZ history."""

    sanitized_history, removed_messages = _remove_assistant_messages_before_runtime_intervention(
        history,
        runtime_intervention="first_turn_protocol_gate",
    )
    sanitation_applied = bool(removed_messages)
    sanitation_reason = (
        "removed_superseded_first_turn_gate_assistant_branch" if sanitation_applied else "not_needed"
    )

    removed_obsolete_terminal_answer_count = sum(
        1 for message in removed_messages if message["message_type"] == "final_answer"
    )
    removed_pz_only_leakage_message_count = sum(
        1 for message in removed_messages if _message_contains_pz_only_leakage(message)
    )
    return sanitized_history, {
        "sanitation_applied": sanitation_applied,
        "sanitation_reason": sanitation_reason,
        "removed_message_count": len(removed_messages),
        "removed_obsolete_terminal_answer_count": removed_obsolete_terminal_answer_count,
        "removed_pz_only_leakage_message_count": removed_pz_only_leakage_message_count,
        "removed_message_role_sequence": [
            f"{message['role']}:{message['message_type']}" for message in removed_messages
        ],
        "removed_message_digests": [_message_digest(message) for message in removed_messages],
    }


def _sanitize_post_pz_second_turn_gate_history(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Remove the superseded direct-final branch once the second-turn gate recovers to CR."""

    sanitized_history, _removed_messages = _remove_assistant_messages_before_runtime_intervention(
        history,
        runtime_intervention="post_pz_second_turn_protocol_gate",
    )
    return sanitized_history


def _post_pz_transition_payload(
    *,
    pre_sanitation_history: list[dict[str, Any]],
    post_sanitation_history: list[dict[str, Any]],
    sample_id: str,
    tool_mode: str,
    first_tool_name: str,
    first_tool_turn_index: int,
    first_turn_gate_outcome: str,
    first_turn_retry_repair_involved: bool,
    post_pz_assistant_turn_index: int,
    prior_tool_trace_count: int,
    sanitation_audit: dict[str, Any],
) -> dict[str, Any]:
    """Build one auditable snapshot of the pre/post-sanitation post-PZ prompt surface."""

    pre_sanitation_contract = _post_pz_transition_contract_fields(pre_sanitation_history)
    post_sanitation_contract = _post_pz_transition_contract_fields(post_sanitation_history)
    return {
        "sample_id": sample_id,
        "tool_mode": tool_mode,
        "first_tool_name": first_tool_name,
        "first_tool_turn_index": first_tool_turn_index,
        "first_turn_gate_outcome": first_turn_gate_outcome,
        "first_turn_retry_repair_involved": first_turn_retry_repair_involved,
        "post_pz_assistant_turn_index": post_pz_assistant_turn_index,
        "prior_tool_trace_count": prior_tool_trace_count,
        "pz_result_reinserted_present": pre_sanitation_contract["pz_result_reinserted_present"],
        "pz_result_reinserted_digest": pre_sanitation_contract["pz_result_reinserted_digest"],
        "rendered_prompt_surface": pre_sanitation_contract["rendered_prompt_surface"],
        "rendered_prompt_surface_digest": pre_sanitation_contract["rendered_prompt_surface_digest"],
        "declared_available_tools": pre_sanitation_contract["declared_available_tools"],
        "cr_available_in_prompt_surface": pre_sanitation_contract["cr_available_in_prompt_surface"],
        "query_image_instruction_present": pre_sanitation_contract["query_image_instruction_present"],
        "crop_image_normalized_reference_present": pre_sanitation_contract[
            "crop_image_normalized_reference_present"
        ],
        "transition_contract_valid_for_cr": pre_sanitation_contract["transition_contract_valid_for_cr"],
        "transition_mismatch_reasons": pre_sanitation_contract["transition_mismatch_reasons"],
        "sanitation_applied": sanitation_audit["sanitation_applied"],
        "sanitation_reason": sanitation_audit["sanitation_reason"],
        "removed_message_count": sanitation_audit["removed_message_count"],
        "removed_obsolete_terminal_answer_count": sanitation_audit[
            "removed_obsolete_terminal_answer_count"
        ],
        "removed_pz_only_leakage_message_count": sanitation_audit[
            "removed_pz_only_leakage_message_count"
        ],
        "removed_message_role_sequence": sanitation_audit["removed_message_role_sequence"],
        "removed_message_digests": sanitation_audit["removed_message_digests"],
        "pre_sanitation_rendered_prompt_surface": pre_sanitation_contract["rendered_prompt_surface"],
        "pre_sanitation_rendered_prompt_surface_digest": pre_sanitation_contract[
            "rendered_prompt_surface_digest"
        ],
        "post_sanitation_rendered_prompt_surface": post_sanitation_contract["rendered_prompt_surface"],
        "post_sanitation_rendered_prompt_surface_digest": post_sanitation_contract[
            "rendered_prompt_surface_digest"
        ],
        "pre_sanitation_pz_only_leakage_present": pre_sanitation_contract["pz_only_leakage_present"],
        "post_sanitation_pz_only_leakage_present": post_sanitation_contract["pz_only_leakage_present"],
        "post_sanitation_transition_contract_valid_for_cr": post_sanitation_contract[
            "transition_contract_valid_for_cr"
        ],
        "post_sanitation_transition_mismatch_reasons": post_sanitation_contract[
            "transition_mismatch_reasons"
        ],
        "second_turn_raw_output_path": None,
        "second_turn_protocol_event_type": None,
        "second_turn_terminal_without_cr": False,
        "second_turn_direct_final_without_cr": False,
        "second_turn_called_cr": False,
        "second_turn_called_non_cr_tool": False,
        "second_turn_parser_valid": None,
        "second_turn_schema_valid": None,
        "second_turn_failure_reason": None,
    }


def _first_turn_protocol_gate_retry_instruction() -> str:
    """Return the deterministic runtime reminder used by the first-turn gate."""

    return (
        "Runtime protocol reminder: this is a pz_cr run. On the first step, do not return a final answer yet. "
        "First inspect with crop_image_normalized. If local evidence is still insufficient after crop, query_image "
        "is allowed. Reply now with exactly one valid <tool_call> and no <answer>."
    )


def _post_pz_second_turn_gate_retry_instruction() -> str:
    """Return the deterministic runtime reminder used by the bounded post-PZ CR gate."""

    return (
        "Runtime protocol reminder: this is a pz_cr run and PZ has already been used. "
        "Before giving a final answer, compare against a same-category normal reference. "
        "Call CR now and reply with exactly one valid <tool_call> and no <answer>."
    )


def _should_trigger_first_turn_gate(
    *,
    gate_mode: str,
    tool_mode: str,
    prompt_audit: dict[str, Any],
    turn_index: int,
    event: str,
    tool_traces: list[dict[str, Any]],
) -> bool:
    """Return whether the opt-in first-turn protocol gate should fire."""

    return (
        gate_mode == "retry_once_pz_cr"
        and tool_mode == "pz_cr"
        and not prompt_audit["mode_contract_mismatch"]
        and prompt_audit["cr_available_in_prompt_surface"]
        and not prompt_audit["prompt_contains_pz_only"]
        and turn_index == 0
        and event == "final_answer"
        and not tool_traces
    )


def _should_trigger_post_pz_second_turn_gate(
    *,
    gate_mode: str,
    tool_mode: str,
    first_successful_tool_name: str | None,
    post_pz_transition_audited: bool,
    post_pz_assistant_turn_index: int | None,
    current_turn_index: int,
    post_sanitation_contract_valid_for_cr: bool | None,
    event: str,
) -> bool:
    """Return whether the bounded post-PZ second-turn CR gate should fire."""

    return (
        gate_mode == "retry_once_require_cr_after_pz"
        and tool_mode == "pz_cr"
        and first_successful_tool_name == "PZ"
        and post_pz_transition_audited
        and post_pz_assistant_turn_index == current_turn_index
        and post_sanitation_contract_valid_for_cr is True
        and event == "final_answer"
    )


def _first_turn_gate_summary(prediction_records: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate first-turn gate outcomes across prediction records."""

    trigger_count = sum(1 for record in prediction_records if record["first_turn_gate_triggered"])
    recovered_count = sum(
        1 for record in prediction_records if record["first_turn_gate_outcome"] == "recovered_to_tool_call"
    )
    still_terminal_count = sum(
        1 for record in prediction_records if record["first_turn_gate_outcome"] == "still_terminal_after_retry"
    )
    retry_parse_failure_count = sum(
        1 for record in prediction_records if record["first_turn_gate_outcome"] == "retry_parse_failure"
    )
    return {
        "first_turn_gate_trigger_count": trigger_count,
        "samples_with_first_turn_gate_events": trigger_count,
        "first_turn_gate_recovered_to_tool_call_count": recovered_count,
        "first_turn_gate_still_terminal_count": still_terminal_count,
        "first_turn_gate_retry_parse_failure_count": retry_parse_failure_count,
        "first_turn_gate_recovery_rate": (recovered_count / trigger_count) if trigger_count else 0.0,
    }


def _first_turn_gate_repair_summary(prediction_records: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate retry-only repair outcomes for first-turn gate parse failures."""

    attempt_count = sum(1 for record in prediction_records if record["first_turn_gate_repair_attempted"])
    success_count = sum(1 for record in prediction_records if record["first_turn_gate_repair_succeeded"])
    failure_count = attempt_count - success_count
    failure_family_counts = {
        family: sum(
            1
            for record in prediction_records
            if record["first_turn_gate_repair_failure_family"] == family
        )
        for family in RETRY_REPAIR_FAILURE_FAMILIES
    }
    return {
        "first_turn_gate_repair_attempt_count": attempt_count,
        "first_turn_gate_repair_success_count": success_count,
        "first_turn_gate_repair_failure_count": failure_count,
        "first_turn_gate_repair_wrapper_recovery_count": sum(
            1
            for record in prediction_records
            if "wrapper_recovery" in record["first_turn_gate_repair_categories"]
        ),
        "first_turn_gate_repair_quote_normalization_count": sum(
            1
            for record in prediction_records
            if "quote_normalization" in record["first_turn_gate_repair_categories"]
        ),
        "first_turn_gate_repair_duplicate_candidate_deduplication_count": sum(
            1
            for record in prediction_records
            if "duplicate_candidate_deduplication" in record["first_turn_gate_repair_categories"]
        ),
        "first_turn_gate_repair_alias_canonicalization_count": sum(
            1
            for record in prediction_records
            if "alias_canonicalization" in record["first_turn_gate_repair_categories"]
        ),
        "first_turn_gate_repair_bbox_canonicalization_count": sum(
            1
            for record in prediction_records
            if "bbox_canonicalization" in record["first_turn_gate_repair_categories"]
        ),
        "first_turn_gate_repair_failure_families": failure_family_counts,
        "failed_count_with_missing_reason_count": sum(
            1
            for record in prediction_records
            if record["first_turn_gate_repair_attempted"]
            and not record["first_turn_gate_repair_succeeded"]
            and (record.get("failure_reason") is None or not str(record["failure_reason"]).strip())
        ),
    }


def _first_turn_gate_repair_failure_family_artifact(
    prediction_records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build one compact artifact for retry-repair failure-family counts and sample IDs."""

    by_family: dict[str, Any] = {}
    for family in RETRY_REPAIR_FAILURE_FAMILIES:
        family_records = [
            record for record in prediction_records if record["first_turn_gate_repair_failure_family"] == family
        ]
        by_family[family] = {
            "count": len(family_records),
            "sample_ids": [record["sample_id"] for record in family_records[:5]],
            "failure_reasons": sorted({record["failure_reason"] for record in family_records if record["failure_reason"]}),
        }
    return {
        "families": by_family,
        "repair_attempt_count": sum(
            1 for record in prediction_records if record["first_turn_gate_repair_attempted"]
        ),
        "repair_failure_count": sum(
            1
            for record in prediction_records
            if record["first_turn_gate_repair_attempted"] and not record["first_turn_gate_repair_succeeded"]
        ),
    }


def _artifact_dirs(definition: dict[str, Any], root: Path) -> dict[str, Path]:
    """Resolve and create the standard artifact directories for one run."""

    directories = {
        "root": root,
        "raw_outputs": root / definition["artifacts"]["raw_outputs"],
        "traces": root / definition["artifacts"]["traces"],
        "predictions": root / definition["artifacts"]["predictions"],
        "metrics": root / definition["artifacts"]["metrics"],
    }
    if "delta" in definition["artifacts"]:
        directories["delta"] = root / definition["artifacts"]["delta"]
    for key, directory in directories.items():
        if key != "root":
            directory.mkdir(parents=True, exist_ok=True)
    return directories


def _baseline_metadata(
    *,
    definition: dict[str, Any],
    config_path: str | Path,
    sample_source_path: Path,
    runtime_provenance: dict[str, Any],
) -> dict[str, Any]:
    """Build reproducibility metadata for the baseline run family."""

    fixture_manifest = sample_source_path / "fixture_manifest.json"
    dataset_manifest_hash = sha256_file(fixture_manifest) if fixture_manifest.exists() else None
    return build_run_metadata(
        run_id=definition["run_id"],
        phase="eval",
        boundary=definition["execution_boundary"],
        config_hashes={"baseline_run_definition": sha256_file(config_path)},
        script_hashes={
            "baseline.py": sha256_file(Path(__file__)),
            "backends.py": sha256_file(Path(__file__).with_name("backends.py")),
            "evaluation.py": sha256_file(Path(__file__).with_name("evaluation.py")),
            "prompting.py": sha256_file(Path(__file__).with_name("prompting.py")),
        },
        dataset_manifest_hash=dataset_manifest_hash,
        notes=[
            f"Canonical baseline run for mode={definition['mode']}.",
            f"Backend type={runtime_provenance['runtime_backend_type']} policy={definition['backend']['policy']}.",
            (
                f"Adapter checkpoint loaded from {runtime_provenance['adapter_checkpoint_path']}."
                if runtime_provenance["adapter_loaded"]
                else "No adapter checkpoint was loaded for this run."
            ),
        ],
    )


def _tool_metadata(
    *,
    definition: dict[str, Any],
    config_path: str | Path,
    sample_source_path: Path,
    compare_config_path: str | Path,
    runtime_provenance: dict[str, Any],
) -> dict[str, Any]:
    """Build reproducibility metadata for the tool-augmented run family."""

    fixture_manifest = sample_source_path / "fixture_manifest.json"
    dataset_manifest_hash = sha256_file(fixture_manifest) if fixture_manifest.exists() else None
    return build_run_metadata(
        run_id=definition["run_id"],
        phase="eval",
        boundary=definition["execution_boundary"],
        config_hashes={
            "tool_run_definition": sha256_file(config_path),
            "baseline_compare_definition": sha256_file(compare_config_path),
        },
        script_hashes={
            "baseline.py": sha256_file(Path(__file__)),
            "backends.py": sha256_file(Path(__file__).with_name("backends.py")),
            "evaluation.py": sha256_file(Path(__file__).with_name("evaluation.py")),
            "prompting.py": sha256_file(Path(__file__).with_name("prompting.py")),
            "tooling.py": sha256_file(Path(__file__).with_name("tooling.py")),
        },
        dataset_manifest_hash=dataset_manifest_hash,
        notes=[
            f"Canonical tool-augmented run for mode={definition['mode']}.",
            f"Backend type={runtime_provenance['runtime_backend_type']} policy={definition['backend']['policy']}.",
            (
                f"Adapter checkpoint loaded from {runtime_provenance['adapter_checkpoint_path']}."
                if runtime_provenance["adapter_loaded"]
                else "No adapter checkpoint was loaded for this run."
            ),
        ],
    )


def run_baseline(
    *,
    config_path: str | Path,
    dataset_root: str | Path | None = None,
    artifact_root: str | Path | None = None,
    max_samples: int | None = None,
    runtime_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run the canonical non-tool baseline and write auditable artifacts."""

    definition = load_run_definition(config_path)
    if definition["mode"] != "no_tools":
        raise InferenceRunError("run_baseline requires a `mode=no_tools` config")

    runtime_config = _runtime_config(
        definition,
        dataset_root=dataset_root,
        runtime_overrides=runtime_overrides,
    )
    backend = _select_backend(definition["backend"], runtime_config=runtime_config)
    samples = _load_samples(definition, dataset_root=dataset_root, max_samples=max_samples)
    root = _resolve_path(artifact_root or definition["artifacts"]["root"])
    directories = _artifact_dirs(definition, root)
    progress_snapshot_path = None
    if runtime_config["progress_mode"] != "off":
        progress_snapshot_path = Path(
            runtime_config["progress_snapshot_path"] or _default_progress_snapshot_path(root)
        ).resolve()
        runtime_config["progress_snapshot_path"] = str(progress_snapshot_path)
    progress_reporter = ProgressReporter(
        run_id=definition["run_id"],
        total_samples=len(definition["seeds"]) * len(samples),
        mode=runtime_config["progress_mode"],
        update_every_n_samples=runtime_config["progress_update_every_n_samples"],
        snapshot_path=progress_snapshot_path,
    )
    processed_samples = 0
    progress_counts = {
        "generation_call_count_total": 0,
        "retry_count_total": 0,
    }

    prediction_records: list[dict[str, Any]] = []
    for seed in definition["seeds"]:
        seed_records: list[dict[str, Any]] = []
        for sample in samples:
            sample_timing = _new_timing_counters()
            prompt_stage_start = time.perf_counter()
            prompt_bundle = build_baseline_prompt(sample)
            sample_timing["prompt_render_ms"] += _elapsed_ms(prompt_stage_start)
            history = _prompt_history(prompt_bundle)
            request_stage_start = time.perf_counter()
            request = BackendRequest(
                sample_id=sample["sample_id"],
                seed=seed,
                prompt_version=prompt_bundle.prompt_version,
                messages=history,
                stop_sequences=prompt_bundle.stop_sequences,
                tool_mode="no_tools",
                generation_config=_generation_config_for_stage(runtime_config, "final_answer"),
                metadata={
                    "tool_mode": "no_tools",
                    "generation_stage": "final_answer",
                    "sample_kind": sample["metadata"].get("dataset_kind", "unknown"),
                },
            )
            sample_timing["request_build_or_processor_ms"] += _elapsed_ms(request_stage_start)
            generate_stage_start = time.perf_counter()
            response = backend.generate(request, sample=sample)
            sample_timing["generate_ms"] += _elapsed_ms(generate_stage_start)
            sample_timing["generation_call_count"] += 1

            sample_slug = safe_slug(sample["sample_id"])
            raw_output_path = _raw_output_path(directories["raw_outputs"], seed=seed, sample_slug=sample_slug, turn_index=0)
            raw_output_text = response.raw_output

            prediction: dict[str, Any] | None = None
            parser_valid = False
            schema_valid = False
            error_message: str | None = None
            failure_reason: str | None = None
            first_protocol_event_type = "unknown"
            terminal_answer_present = False
            terminal_answer_turn_index: int | None = None
            _append_reasoning(history, response.raw_output, backend_name=response.backend_name)
            try:
                parse_stage_start = time.perf_counter()
                prediction = parse_final_answer(response.raw_output)
                sample_timing["parse_validate_ms"] += _elapsed_ms(parse_stage_start)
                parser_valid = True
                schema_valid = True
                first_protocol_event_type = "final_answer"
                terminal_answer_present = True
                terminal_answer_turn_index = 0
            except Exception as exc:  # noqa: BLE001 - gate needs explicit failures.
                sample_timing["parse_validate_ms"] += _elapsed_ms(parse_stage_start)
                error_message = str(exc)
                failure_reason = _failure_reason("parser_invalid", str(exc))
                if "<answer>" in response.raw_output or "<final_answer>" in response.raw_output:
                    first_protocol_event_type = "final_answer"
                    terminal_answer_present = True
                    terminal_answer_turn_index = 0
                else:
                    first_protocol_event_type = "parse_failure"
            failure_reason = _resolve_failure_reason(
                prediction=prediction,
                parser_valid=parser_valid,
                schema_valid=schema_valid,
                failure_reason=failure_reason,
                error_message=error_message,
            )
            keep_optional_artifacts = (
                runtime_config["artifact_level"] == "forensic" or bool(failure_reason)
            )
            if keep_optional_artifacts:
                write_stage_start = time.perf_counter()
                _write_text(raw_output_path, raw_output_text)
                sample_timing["file_write_ms"] += _elapsed_ms(write_stage_start)
            _append_final_answer_message(
                history,
                response.raw_output,
                backend_name=response.backend_name,
                raw_output_path=str(raw_output_path.resolve()),
                error_message=error_message,
            )

            trace = TraceRecord(
                trace_id=f"{definition['run_id']}:{seed}:{sample['sample_id']}",
                sample_id=sample["sample_id"],
                stage="eval",
                tool_path="no_tools",
                storage_purpose="eval",
                messages=_history_to_trace_messages(history),
                tool_traces=(),
                final_answer=prediction,
                metadata={
                    "seed": seed,
                    "backend_metadata": response.metadata,
                    "sample_category": sample["category"],
                },
            )
            trace_payload = trace.to_audit_payload()
            trace_path = directories["traces"] / f"seed_{seed}" / f"{sample_slug}.json"
            write_stage_start = time.perf_counter()
            write_json(trace_path, trace_payload)
            sample_timing["file_write_ms"] += _elapsed_ms(write_stage_start)

            record = build_prediction_record(
                run_id=definition["run_id"],
                sample=sample,
                seed=seed,
                tool_mode="no_tools",
                tool_usage=empty_tool_usage(),
                prompt_version=prompt_bundle.prompt_version,
                parser_version=definition["parser_version"],
                backend_name=response.backend_name,
                prediction=prediction,
                parser_valid=parser_valid,
                schema_valid=schema_valid,
                error_message=error_message,
                failure_reason=failure_reason,
                raw_output_path=str(raw_output_path.resolve()),
                raw_output_sha256=_sha256_text(raw_output_text),
                trace_path=str(trace_path.resolve()),
                first_turn_protocol_gate_mode="off",
                post_pz_second_turn_gate_mode="off",
                first_turn_gate_triggered=False,
                first_turn_gate_retry_count=0,
                first_turn_gate_recovered=False,
                first_turn_gate_outcome="not_triggered",
                first_turn_gate_repair_attempted=False,
                first_turn_gate_repair_succeeded=False,
                first_turn_gate_repair_outcome="not_attempted",
                first_turn_gate_repair_categories=[],
                first_turn_gate_repair_original_failure_family=None,
                first_turn_gate_repair_failure_family=None,
                first_turn_gate_sidecar_path=None,
                first_successful_tool_name=None,
                first_successful_tool_turn_index=None,
                post_pz_transition_audited=False,
                post_pz_transition_sidecar_path=None,
                post_pz_transition_contract_valid_for_cr=None,
                post_pz_transition_mismatch_reasons=[],
                post_pz_transition_sanitation_applied=False,
                post_pz_transition_sanitation_reason="not_applicable",
                post_pz_transition_removed_message_count=0,
                post_pz_transition_removed_obsolete_terminal_answer_count=0,
                post_pz_transition_removed_pz_only_leakage_message_count=0,
                post_pz_transition_pre_sanitation_pz_only_leakage_present=False,
                post_pz_transition_post_sanitation_pz_only_leakage_present=False,
                post_pz_transition_post_sanitation_contract_valid_for_cr=None,
                post_pz_transition_post_sanitation_mismatch_reasons=[],
                post_pz_second_turn_protocol_event_type=None,
                post_pz_second_turn_direct_final_without_cr=False,
                post_pz_second_turn_called_cr=False,
                post_pz_second_turn_called_non_cr_tool=False,
                post_pz_second_turn_parser_valid=None,
                post_pz_second_turn_schema_valid=None,
                post_pz_second_turn_failure_reason=None,
                post_pz_second_turn_gate_triggered=False,
                post_pz_second_turn_gate_outcome="gate_not_triggered",
                post_pz_second_turn_gate_retry_attempted=False,
                post_pz_second_turn_gate_retry_called_tool_name=None,
                post_pz_second_turn_gate_retry_parser_valid=None,
                post_pz_second_turn_gate_retry_schema_valid=None,
                post_pz_second_turn_gate_retry_failure_reason=None,
                post_pz_second_turn_gate_retry_raw_output_path=None,
                post_pz_second_turn_gate_sidecar_path=None,
                metadata={
                    "backend_metadata": response.metadata,
                    "sample_source_kind": sample["metadata"].get("dataset_kind"),
                    "timing": dict(sample_timing),
                    "runtime_provenance": _resolved_runtime_provenance(
                        definition,
                        runtime_config,
                        backend,
                    ),
                },
                **build_zero_tool_behavior_fields(
                    first_protocol_event_type=first_protocol_event_type,
                    called_tools=[],
                    terminal_answer_present=terminal_answer_present,
                    terminal_answer_turn_index=terminal_answer_turn_index,
                    prediction=prediction,
                ),
            )
            prediction_path = directories["predictions"] / f"seed_{seed}" / f"{sample_slug}.json"
            write_stage_start = time.perf_counter()
            write_json(prediction_path, record)
            sample_timing["file_write_ms"] += _elapsed_ms(write_stage_start)
            record["metadata"]["timing"] = dict(sample_timing)
            seed_records.append(record)
            prediction_records.append(record)
            processed_samples += 1
            progress_counts["generation_call_count_total"] += int(sample_timing["generation_call_count"])
            progress_counts["retry_count_total"] += int(sample_timing["retry_count"])
            progress_reporter.update(
                processed_samples=processed_samples,
                current_sample_id=sample["sample_id"],
                timing_summary=progress_counts,
            )

        write_stage_start = time.perf_counter()
        write_jsonl(directories["predictions"] / f"seed_{seed}.jsonl", seed_records)
        write_elapsed_ms = _elapsed_ms(write_stage_start)
        for record in seed_records:
            record["metadata"]["timing"]["file_write_ms"] += write_elapsed_ms / max(len(seed_records), 1)

    progress_reporter.update(
        processed_samples=processed_samples,
        current_sample_id=None,
        timing_summary=progress_counts,
        force=True,
    )

    runtime_provenance = _resolved_runtime_provenance(definition, runtime_config, backend)
    timing_summary = (
        _timing_summary_from_prediction_records(prediction_records)
        if runtime_config["timing_enabled"]
        else None
    )
    metrics_report = build_metrics_report(
        run_id=definition["run_id"],
        tool_mode="no_tools",
        prompt_version=definition["prompt_version"],
        parser_version=definition["parser_version"],
        backend_name=definition["backend"]["name"],
        prediction_records=prediction_records,
        seeds=definition["seeds"],
        runtime_provenance=runtime_provenance,
        artifact_level=runtime_provenance["artifact_level"],
        emit_baseline_compare=runtime_provenance["emit_baseline_compare"],
        emit_delta_report=runtime_provenance["emit_delta_report"],
        timing_enabled=runtime_provenance["timing_enabled"],
        progress_mode=runtime_provenance["progress_mode"],
        timing_summary=timing_summary,
    )
    metrics_report_path = directories["metrics"] / "metrics_report.json"
    per_seed_metrics_path = directories["metrics"] / "per_seed_metrics.json"
    per_class_metrics_path = directories["metrics"] / "per_class_metrics.json"
    aggregate_metrics_path = directories["metrics"] / "aggregate_metrics.json"
    write_json(metrics_report_path, metrics_report)
    write_json(per_seed_metrics_path, {"per_seed_metrics": metrics_report["per_seed_metrics"]})
    write_json(per_class_metrics_path, {"per_class_metrics": metrics_report["per_class_metrics"]})
    write_json(aggregate_metrics_path, {"aggregate_metrics": metrics_report["aggregate_metrics"]})

    sample_source_path = _resolve_path(dataset_root or definition["sample_source"]["path"])
    run_metadata = _baseline_metadata(
        definition=definition,
        config_path=config_path,
        sample_source_path=sample_source_path,
        runtime_provenance=runtime_provenance,
    )
    run_metadata_path = directories["root"] / "run_metadata.json"
    write_json(run_metadata_path, run_metadata)

    summary = build_run_summary(
        run_id=definition["run_id"],
        tool_mode="no_tools",
        backend_name=definition["backend"]["name"],
        prompt_version=definition["prompt_version"],
        parser_version=definition["parser_version"],
        execution_boundary=definition["execution_boundary"],
        sample_source={
            "kind": definition["sample_source"]["kind"],
            "path": str(sample_source_path),
            "source_name": definition["sample_source"]["source_name"],
        },
        seeds=definition["seeds"],
        prediction_records=prediction_records,
        runtime_provenance=runtime_provenance,
        artifact_level=runtime_provenance["artifact_level"],
        emit_baseline_compare=runtime_provenance["emit_baseline_compare"],
        emit_delta_report=runtime_provenance["emit_delta_report"],
        timing_enabled=runtime_provenance["timing_enabled"],
        progress_mode=runtime_provenance["progress_mode"],
        artifact_paths={
            "predictions_dir": str(directories["predictions"].resolve()),
            "traces_dir": str(directories["traces"].resolve()),
            "raw_outputs_dir": str(directories["raw_outputs"].resolve()),
            "metrics_report": str(metrics_report_path.resolve()),
            "per_seed_metrics": str(per_seed_metrics_path.resolve()),
            "per_class_metrics": str(per_class_metrics_path.resolve()),
            "aggregate_metrics": str(aggregate_metrics_path.resolve()),
            "run_metadata": str(run_metadata_path.resolve()),
            "run_manifest": str((directories["root"] / "run_manifest.json").resolve()),
            **(
                {"progress_snapshot": str(progress_snapshot_path)}
                if progress_snapshot_path is not None
                else {}
            ),
        },
        notes=definition.get("notes", []),
        timing_summary=timing_summary,
    )
    summary_path = directories["root"] / "run_summary.json"
    write_json(summary_path, summary)

    run_manifest = {
        "artifact_id": definition["run_id"],
        "artifact_type": "report",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "producer": definition["backend"]["name"],
        "content_hash": sha256_file(summary_path),
        "input_hashes": {
            "baseline_run_definition": sha256_file(config_path),
            "metrics_report": sha256_file(metrics_report_path),
            "run_metadata": sha256_file(run_metadata_path),
        },
        "execution_boundary": definition["execution_boundary"],
        "artifact_level": runtime_provenance["artifact_level"],
        "emit_baseline_compare": runtime_provenance["emit_baseline_compare"],
        "emit_delta_report": runtime_provenance["emit_delta_report"],
        "timing_enabled": runtime_provenance["timing_enabled"],
        "progress_mode": runtime_provenance["progress_mode"],
        "run_provenance": runtime_provenance,
        **({"timing_summary": timing_summary} if timing_summary is not None else {}),
        "notes": [
            "Non-tool baseline summary manifest.",
            (
                "This run used the scripted mock backend for local smoke validation."
                if definition["backend"]["type"] == "mock"
                else "This manifest records a real evaluation-capable backend configuration."
            ),
        ],
    }
    validate_payload(run_manifest, "artifact_manifest.schema.json")
    run_manifest_path = directories["root"] / "run_manifest.json"
    write_json(run_manifest_path, run_manifest)

    return {
        "run_definition": definition,
        "prediction_records": prediction_records,
        "metrics_report": metrics_report,
        "run_metadata_path": str(run_metadata_path.resolve()),
        "run_manifest_path": str(run_manifest_path.resolve()),
        "summary_path": str(summary_path.resolve()),
        "runtime_provenance": runtime_provenance,
    }


def _comparison_artifact_root(
    definition: dict[str, Any],
    artifact_root: str | Path | None,
) -> Path:
    """Resolve a local baseline-comparison artifact root for tool runs."""

    if artifact_root is None:
        return _resolve_path(definition["compare_to"]["artifact_root"])
    return _resolve_path(Path(artifact_root) / "_baseline_compare")


def _tool_loop_sample(
    *,
    definition: dict[str, Any],
    backend: InferenceBackend,
    runtime_config: dict[str, Any],
    sample: dict[str, Any],
    sample_pool: list[dict[str, Any]],
    seed: int,
    directories: dict[str, Path],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Execute the bounded tool loop for one sample and return its artifacts."""

    sample_timing = _new_timing_counters()
    prompt_stage_start = time.perf_counter()
    prompt_bundle = build_prompt(
        sample,
        tool_path=definition["mode"],
        tool_first_intervention_strategy=runtime_config["tool_first_intervention_strategy"],
    )
    sample_timing["prompt_render_ms"] += _elapsed_ms(prompt_stage_start)
    history = _prompt_history(prompt_bundle)
    sample_slug = safe_slug(sample["sample_id"])
    artifact_level = runtime_config["artifact_level"]
    pending_text_artifacts: dict[Path, str] = {}
    pending_json_artifacts: dict[Path, dict[str, Any]] = {}
    tool_traces: list[dict[str, Any]] = []
    prediction: dict[str, Any] | None = None
    parser_valid = False
    schema_valid = False
    error_message: str | None = None
    failure_reason: str | None = None
    last_raw_output_path: Path | None = None
    last_raw_output_text: str | None = None
    normalization_events: list[dict[str, Any]] = []
    normalization_event_paths: list[str] = []
    first_protocol_event_type = "unknown"
    terminal_answer_present = False
    terminal_answer_turn_index: int | None = None
    first_turn_protocol_gate_mode = runtime_config["first_turn_protocol_gate_mode"]
    post_pz_second_turn_gate_mode = runtime_config["post_pz_second_turn_gate_mode"]
    first_turn_gate_triggered = False
    first_turn_gate_retry_count = 0
    first_turn_gate_recovered = False
    first_turn_gate_outcome = "not_triggered"
    first_turn_gate_repair_attempted = False
    first_turn_gate_repair_succeeded = False
    first_turn_gate_repair_outcome = "not_attempted"
    first_turn_gate_repair_categories: list[str] = []
    first_turn_gate_repair_original_failure_family: str | None = None
    first_turn_gate_repair_failure_family: str | None = None
    first_turn_gate_sidecar: dict[str, Any] | None = None
    first_turn_gate_sidecar_path: Path | None = None
    first_successful_tool_name: str | None = None
    first_successful_tool_turn_index: int | None = None
    post_pz_transition_audited = False
    post_pz_transition_sidecar: dict[str, Any] | None = None
    post_pz_transition_sidecar_path: Path | None = None
    post_pz_transition_contract_valid_for_cr: bool | None = None
    post_pz_transition_mismatch_reasons: list[str] = []
    post_pz_transition_sanitation_applied = False
    post_pz_transition_sanitation_reason = "not_applicable"
    post_pz_transition_removed_message_count = 0
    post_pz_transition_removed_obsolete_terminal_answer_count = 0
    post_pz_transition_removed_pz_only_leakage_message_count = 0
    post_pz_transition_pre_sanitation_pz_only_leakage_present = False
    post_pz_transition_post_sanitation_pz_only_leakage_present = False
    post_pz_transition_post_sanitation_contract_valid_for_cr: bool | None = None
    post_pz_transition_post_sanitation_mismatch_reasons: list[str] = []
    post_pz_second_turn_protocol_event_type: str | None = None
    post_pz_second_turn_direct_final_without_cr = False
    post_pz_second_turn_called_cr = False
    post_pz_second_turn_called_non_cr_tool = False
    post_pz_second_turn_parser_valid: bool | None = None
    post_pz_second_turn_schema_valid: bool | None = None
    post_pz_second_turn_failure_reason: str | None = None
    post_pz_second_turn_gate_triggered = False
    post_pz_second_turn_gate_outcome = "gate_not_triggered"
    post_pz_second_turn_gate_retry_attempted = False
    post_pz_second_turn_gate_retry_called_tool_name: str | None = None
    post_pz_second_turn_gate_retry_parser_valid: bool | None = None
    post_pz_second_turn_gate_retry_schema_valid: bool | None = None
    post_pz_second_turn_gate_retry_failure_reason: str | None = None
    post_pz_second_turn_gate_retry_raw_output_path: Path | None = None
    post_pz_second_turn_gate_sidecar_path: Path | None = None

    def _write_required_text(path: Path, payload: str) -> None:
        start_time = time.perf_counter()
        _write_text(path, payload)
        sample_timing["file_write_ms"] += _elapsed_ms(start_time)

    def _write_required_json(path: Path, payload: dict[str, Any]) -> None:
        start_time = time.perf_counter()
        write_json(path, payload)
        sample_timing["file_write_ms"] += _elapsed_ms(start_time)

    def _queue_optional_text(path: Path, payload: str) -> None:
        pending_text_artifacts[path] = payload

    def _queue_optional_json(path: Path, payload: dict[str, Any]) -> None:
        pending_json_artifacts[path] = payload

    def _flush_optional_sample_artifacts(keep_artifacts: bool) -> None:
        if not keep_artifacts:
            return
        for path, payload in pending_text_artifacts.items():
            _write_required_text(path, payload)
        for path, payload in pending_json_artifacts.items():
            _write_required_json(path, payload)
    prompt_audit_path = _prompt_audit_sidecar_path(directories["raw_outputs"], seed=seed, sample_slug=sample_slug)
    prompt_audit = _prompt_audit_payload(
        prompt_bundle,
        sample_id=sample["sample_id"],
        seed=seed,
        turn_index=0,
        runtime_tool_mode=definition["mode"],
        tool_first_intervention_strategy=runtime_config["tool_first_intervention_strategy"],
    )
    _queue_optional_json(prompt_audit_path, prompt_audit)

    if prompt_audit["mode_contract_mismatch"]:
        failure_reason = prompt_audit["mismatch_reasons"][0]
        first_protocol_event_type = "runtime_fail_before_generation"
        error_message = (
            "Prompt audit mismatch before backend.generate(): "
            + ", ".join(prompt_audit["mismatch_reasons"])
        )
        last_raw_output_path = _raw_output_path(
            directories["raw_outputs"],
            seed=seed,
            sample_slug=sample_slug,
            turn_index=0,
        )
        last_raw_output_text = f"NO_GENERATION: {error_message}"
        _queue_optional_text(last_raw_output_path, last_raw_output_text)
        _append_final_answer_message(
            history,
            f"NO_GENERATION: {error_message}",
            backend_name=definition["backend"]["name"],
            raw_output_path=str(last_raw_output_path.resolve()),
            error_message=error_message,
        )

    def _record_response_and_normalize(
        response: Any,
        *,
        turn_index: int,
        raw_output_path: Path,
    ) -> tuple[Any, str, dict[str, Any] | None]:
        """Persist one raw output and return the normalized protocol decision."""

        nonlocal last_raw_output_path
        nonlocal last_raw_output_text
        nonlocal first_protocol_event_type
        last_raw_output_path = raw_output_path
        last_raw_output_text = response.raw_output
        _queue_optional_text(raw_output_path, response.raw_output)
        _append_reasoning(history, response.raw_output, backend_name=response.backend_name)
        parse_stage_start = time.perf_counter()
        decision = normalize_protocol_turn(response.raw_output, tool_path=definition["mode"])
        event = decision.event_type
        if turn_index == 0 and first_protocol_event_type == "unknown":
            if event == "tool_call":
                first_protocol_event_type = "tool_call"
            elif event == "final_answer":
                first_protocol_event_type = "final_answer"
            else:
                first_protocol_event_type = "parse_failure"
        normalization_metadata: dict[str, Any] | None = None
        if decision.normalization_applied:
            sidecar_path = _normalization_sidecar_path(raw_output_path)
            audit_payload = decision.to_audit_payload(
                sample_id=sample["sample_id"],
                turn_index=turn_index,
                raw_output_path=str(raw_output_path.resolve()),
            )
            _queue_optional_json(sidecar_path, audit_payload)
            normalization_events.append(audit_payload)
            normalization_event_paths.append(str(sidecar_path.resolve()))
            normalization_metadata = {
                "normalization_applied": True,
                "normalization_reason": audit_payload["reason"],
                "selected_protocol_event_type": audit_payload["selected_protocol_event_type"],
                "selected_tool_name": audit_payload["selected_tool_name"],
                "discarded_final_answer_present": audit_payload["discarded_final_answer_present"],
                "valid_tool_call_count": audit_payload["valid_tool_call_count"],
                "additional_valid_tool_calls_discarded": audit_payload[
                    "additional_valid_tool_calls_discarded"
                ],
                "normalization_sidecar_path": str(sidecar_path.resolve()),
            }
        sample_timing["parse_validate_ms"] += _elapsed_ms(parse_stage_start)
        return decision, event, normalization_metadata

    def _mark_first_successful_tool(*, tool_name: str, turn_index: int) -> None:
        """Capture the first successfully executed tool without changing runtime behavior."""

        nonlocal first_successful_tool_name
        nonlocal first_successful_tool_turn_index
        if first_successful_tool_name is None:
            first_successful_tool_name = tool_name
            first_successful_tool_turn_index = turn_index

    def _timed_parse_final_answer(raw_output: str) -> dict[str, Any]:
        """Parse one final answer while accumulating timing."""

        parse_stage_start = time.perf_counter()
        try:
            return parse_final_answer(raw_output)
        finally:
            sample_timing["parse_validate_ms"] += _elapsed_ms(parse_stage_start)

    def _timed_parse_tool_call(raw_output: str) -> Any:
        """Parse one tool call while accumulating timing."""

        parse_stage_start = time.perf_counter()
        try:
            return parse_tool_call(raw_output, tool_path=definition["mode"])
        finally:
            sample_timing["parse_validate_ms"] += _elapsed_ms(parse_stage_start)

    def _start_post_pz_transition_audit(*, turn_index: int) -> None:
        """Sanitize and snapshot the first assistant turn shown after PZ reinsertion."""

        nonlocal history
        nonlocal post_pz_transition_audited
        nonlocal post_pz_transition_sidecar
        nonlocal post_pz_transition_sidecar_path
        nonlocal post_pz_transition_contract_valid_for_cr
        nonlocal post_pz_transition_mismatch_reasons
        nonlocal post_pz_transition_sanitation_applied
        nonlocal post_pz_transition_sanitation_reason
        nonlocal post_pz_transition_removed_message_count
        nonlocal post_pz_transition_removed_obsolete_terminal_answer_count
        nonlocal post_pz_transition_removed_pz_only_leakage_message_count
        nonlocal post_pz_transition_pre_sanitation_pz_only_leakage_present
        nonlocal post_pz_transition_post_sanitation_pz_only_leakage_present
        nonlocal post_pz_transition_post_sanitation_contract_valid_for_cr
        nonlocal post_pz_transition_post_sanitation_mismatch_reasons
        if (
            definition["mode"] != "pz_cr"
            or post_pz_transition_audited
            or first_successful_tool_name != "PZ"
            or first_successful_tool_turn_index is None
            or not tool_traces
        ):
            return

        post_pz_transition_audited = True
        post_pz_transition_sidecar_path = _post_pz_transition_sidecar_path(
            directories["raw_outputs"],
            seed=seed,
            sample_slug=sample_slug,
            turn_index=turn_index,
        )
        pre_sanitation_history = _clone_history(history)
        sanitized_history, sanitation_audit = _sanitize_post_pz_transition_history(history)
        post_pz_transition_sidecar = _post_pz_transition_payload(
            pre_sanitation_history=pre_sanitation_history,
            post_sanitation_history=sanitized_history,
            sample_id=sample["sample_id"],
            tool_mode=definition["mode"],
            first_tool_name=first_successful_tool_name,
            first_tool_turn_index=first_successful_tool_turn_index,
            first_turn_gate_outcome=first_turn_gate_outcome,
            first_turn_retry_repair_involved=first_turn_gate_repair_succeeded,
            post_pz_assistant_turn_index=turn_index,
            prior_tool_trace_count=len(tool_traces),
            sanitation_audit=sanitation_audit,
        )
        post_pz_transition_contract_valid_for_cr = post_pz_transition_sidecar[
            "transition_contract_valid_for_cr"
        ]
        post_pz_transition_mismatch_reasons = list(
            post_pz_transition_sidecar["transition_mismatch_reasons"]
        )
        post_pz_transition_sanitation_applied = post_pz_transition_sidecar["sanitation_applied"]
        post_pz_transition_sanitation_reason = post_pz_transition_sidecar["sanitation_reason"]
        post_pz_transition_removed_message_count = post_pz_transition_sidecar["removed_message_count"]
        post_pz_transition_removed_obsolete_terminal_answer_count = post_pz_transition_sidecar[
            "removed_obsolete_terminal_answer_count"
        ]
        post_pz_transition_removed_pz_only_leakage_message_count = post_pz_transition_sidecar[
            "removed_pz_only_leakage_message_count"
        ]
        post_pz_transition_pre_sanitation_pz_only_leakage_present = post_pz_transition_sidecar[
            "pre_sanitation_pz_only_leakage_present"
        ]
        post_pz_transition_post_sanitation_pz_only_leakage_present = post_pz_transition_sidecar[
            "post_sanitation_pz_only_leakage_present"
        ]
        post_pz_transition_post_sanitation_contract_valid_for_cr = post_pz_transition_sidecar[
            "post_sanitation_transition_contract_valid_for_cr"
        ]
        post_pz_transition_post_sanitation_mismatch_reasons = list(
            post_pz_transition_sidecar["post_sanitation_transition_mismatch_reasons"]
        )
        history = sanitized_history

    def _finalize_post_pz_transition_audit(
        *,
        turn_index: int,
        raw_output_path: Path,
        event: str,
        second_turn_called_cr: bool,
        second_turn_called_non_cr_tool: bool,
        second_turn_parser_valid_value: bool | None,
        second_turn_schema_valid_value: bool | None,
        second_turn_failure_reason_value: str | None,
    ) -> None:
        """Write the post-PZ sidecar once the audited second turn has been classified."""

        nonlocal post_pz_second_turn_protocol_event_type
        nonlocal post_pz_second_turn_direct_final_without_cr
        nonlocal post_pz_second_turn_called_cr
        nonlocal post_pz_second_turn_called_non_cr_tool
        nonlocal post_pz_second_turn_parser_valid
        nonlocal post_pz_second_turn_schema_valid
        nonlocal post_pz_second_turn_failure_reason
        if (
            post_pz_transition_sidecar is None
            or post_pz_transition_sidecar_path is None
            or post_pz_transition_sidecar["post_pz_assistant_turn_index"] != turn_index
        ):
            return

        protocol_event_type = _post_pz_transition_protocol_event_type(event)
        post_pz_second_turn_protocol_event_type = protocol_event_type
        post_pz_second_turn_called_cr = second_turn_called_cr
        post_pz_second_turn_called_non_cr_tool = second_turn_called_non_cr_tool
        post_pz_second_turn_direct_final_without_cr = protocol_event_type == "final_answer" and not second_turn_called_cr
        post_pz_second_turn_parser_valid = second_turn_parser_valid_value
        post_pz_second_turn_schema_valid = second_turn_schema_valid_value
        post_pz_second_turn_failure_reason = second_turn_failure_reason_value

        post_pz_transition_sidecar["second_turn_raw_output_path"] = str(raw_output_path.resolve())
        post_pz_transition_sidecar["second_turn_protocol_event_type"] = protocol_event_type
        post_pz_transition_sidecar["second_turn_terminal_without_cr"] = (
            post_pz_second_turn_direct_final_without_cr
        )
        post_pz_transition_sidecar["second_turn_direct_final_without_cr"] = (
            post_pz_second_turn_direct_final_without_cr
        )
        post_pz_transition_sidecar["second_turn_called_cr"] = second_turn_called_cr
        post_pz_transition_sidecar["second_turn_called_non_cr_tool"] = second_turn_called_non_cr_tool
        post_pz_transition_sidecar["second_turn_parser_valid"] = second_turn_parser_valid_value
        post_pz_transition_sidecar["second_turn_schema_valid"] = second_turn_schema_valid_value
        post_pz_transition_sidecar["second_turn_failure_reason"] = second_turn_failure_reason_value
        _queue_optional_json(post_pz_transition_sidecar_path, post_pz_transition_sidecar)

    def _write_post_pz_second_turn_gate_sidecar(
        *,
        turn_index: int,
        first_attempt_raw_output_path: Path,
        retry_instruction_text: str,
        retry_raw_output_path: Path | None,
        final_gate_outcome: str,
        retry_called_tool_name: str | None,
        retry_parser_valid_value: bool | None,
        retry_schema_valid_value: bool | None,
        retry_failure_reason_value: str | None,
    ) -> None:
        """Write one bounded post-PZ second-turn gate sidecar when the retry path is used."""

        nonlocal post_pz_second_turn_gate_sidecar_path
        post_pz_second_turn_gate_sidecar_path = _post_pz_second_turn_gate_sidecar_path(
            directories["raw_outputs"],
            seed=seed,
            sample_slug=sample_slug,
            turn_index=turn_index,
        )
        _queue_optional_json(
            post_pz_second_turn_gate_sidecar_path,
            {
                "sample_id": sample["sample_id"],
                "tool_mode": definition["mode"],
                "gate_mode": post_pz_second_turn_gate_mode,
                "gate_triggered": True,
                "trigger_reason": "direct_final_answer_after_valid_clean_post_pz_cr_contract",
                "first_tool_name": first_successful_tool_name,
                "first_tool_turn_index": first_successful_tool_turn_index,
                "post_pz_assistant_turn_index": turn_index,
                "post_sanitation_transition_contract_valid_for_cr": (
                    post_pz_transition_post_sanitation_contract_valid_for_cr
                ),
                "first_attempt_raw_output_path": str(first_attempt_raw_output_path.resolve()),
                "retry_instruction_text": retry_instruction_text,
                "retry_raw_output_path": (
                    str(retry_raw_output_path.resolve()) if retry_raw_output_path is not None else None
                ),
                "final_gate_outcome": final_gate_outcome,
                "retry_called_tool_name": retry_called_tool_name,
                "retry_parser_valid": retry_parser_valid_value,
                "retry_schema_valid": retry_schema_valid_value,
                "retry_failure_reason": retry_failure_reason_value,
                "prompt_audit_path": str(prompt_audit_path.resolve()),
                "post_pz_transition_sidecar_path": (
                    str(post_pz_transition_sidecar_path.resolve())
                    if post_pz_transition_sidecar_path is not None
                    else None
                ),
            },
        )

    def _normal_generation_stage(turn_index: int) -> str:
        """Resolve the stage label for one normal tool-loop generation call."""

        if turn_index == 0 and not tool_traces:
            return "turn0_initial"
        if (
            definition["mode"] == "pz_cr"
            and first_successful_tool_name == "PZ"
            and post_pz_transition_audited
            and post_pz_transition_sidecar is not None
            and post_pz_transition_sidecar["post_pz_assistant_turn_index"] == turn_index
        ):
            return "post_pz_second_turn"
        return "final_answer"

    for turn_index in range(definition["max_tool_turns"] + 1):
        if prompt_audit["mode_contract_mismatch"]:
            break
        _start_post_pz_transition_audit(turn_index=turn_index)
        stage_label = _normal_generation_stage(turn_index)
        request_stage_start = time.perf_counter()
        request = BackendRequest(
            sample_id=sample["sample_id"],
            seed=seed,
            prompt_version=prompt_bundle.prompt_version,
            messages=history,
            stop_sequences=prompt_bundle.stop_sequences,
            tool_mode=definition["mode"],
            generation_config=_generation_config_for_stage(runtime_config, stage_label),
            metadata={
                "tool_mode": definition["mode"],
                "turn_index": turn_index,
                "generation_stage": stage_label,
                "sample_kind": sample["metadata"].get("dataset_kind", "unknown"),
            },
        )
        sample_timing["request_build_or_processor_ms"] += _elapsed_ms(request_stage_start)
        generate_stage_start = time.perf_counter()
        response = backend.generate(request, sample=sample)
        sample_timing["generate_ms"] += _elapsed_ms(generate_stage_start)
        sample_timing["generation_call_count"] += 1
        raw_output_path = _raw_output_path(
            directories["raw_outputs"],
            seed=seed,
            sample_slug=sample_slug,
            turn_index=turn_index,
        )
        decision, event, normalization_metadata = _record_response_and_normalize(
            response,
            turn_index=turn_index,
            raw_output_path=raw_output_path,
        )
        if _should_trigger_first_turn_gate(
            gate_mode=first_turn_protocol_gate_mode,
            tool_mode=definition["mode"],
            prompt_audit=prompt_audit,
            turn_index=turn_index,
            event=event,
            tool_traces=tool_traces,
        ):
            first_turn_gate_triggered = True
            first_turn_gate_retry_count = 1
            first_turn_gate_sidecar_path = _first_turn_gate_sidecar_path(
                directories["raw_outputs"],
                seed=seed,
                sample_slug=sample_slug,
            )
            retry_instruction_text = _first_turn_protocol_gate_retry_instruction()
            trigger_reason = "turn0_direct_final_answer_under_valid_pz_cr_contract"
            _append_final_answer_message(
                history,
                response.raw_output,
                backend_name=response.backend_name,
                raw_output_path=str(raw_output_path.resolve()),
            )
            _append_runtime_gate_reminder(
                history,
                retry_instruction_text,
                gate_mode=first_turn_protocol_gate_mode,
            )
            sample_timing["retry_count"] += 1
            retry_request_stage_start = time.perf_counter()
            retry_request = BackendRequest(
                sample_id=sample["sample_id"],
                seed=seed,
                prompt_version=prompt_bundle.prompt_version,
                messages=history,
                stop_sequences=prompt_bundle.stop_sequences,
                tool_mode=definition["mode"],
                generation_config=_generation_config_for_stage(runtime_config, "turn0_retry"),
                metadata={
                    "tool_mode": definition["mode"],
                    "turn_index": turn_index,
                    "retry_count": 1,
                    "runtime_intervention": "first_turn_protocol_gate",
                    "generation_stage": "turn0_retry",
                    "sample_kind": sample["metadata"].get("dataset_kind", "unknown"),
                },
            )
            sample_timing["request_build_or_processor_ms"] += _elapsed_ms(retry_request_stage_start)
            retry_generate_start = time.perf_counter()
            retry_response = backend.generate(retry_request, sample=sample)
            sample_timing["generate_ms"] += _elapsed_ms(retry_generate_start)
            sample_timing["generation_call_count"] += 1
            retry_raw_output_path = _retry_raw_output_path(
                directories["raw_outputs"],
                seed=seed,
                sample_slug=sample_slug,
                turn_index=turn_index,
                retry_count=1,
            )
            retry_decision, retry_event, retry_normalization_metadata = _record_response_and_normalize(
                retry_response,
                turn_index=turn_index,
                raw_output_path=retry_raw_output_path,
            )
            first_turn_gate_sidecar = {
                "sample_id": sample["sample_id"],
                "tool_mode": definition["mode"],
                "gate_mode": first_turn_protocol_gate_mode,
                "turn_index": turn_index,
                "prompt_audit_path": str(prompt_audit_path.resolve()),
                "prompt_contract_valid": not prompt_audit["mode_contract_mismatch"],
                "gate_triggered": True,
                "trigger_reason": trigger_reason,
                "first_attempt_raw_output_path": str(raw_output_path.resolve()),
                "retry_instruction_text": retry_instruction_text,
                "retry_count": 1,
                "retry_raw_output_path": str(retry_raw_output_path.resolve()),
                "final_gate_outcome": "retry_parse_failure",
                "retry_repair": {
                    "repair_attempted": False,
                    "repair_succeeded": False,
                    "original_failure_family": RETRY_REPAIR_ORIGINAL_FAILURE_FAMILY,
                    "original_text": retry_response.raw_output,
                    "repaired_text": None,
                    "repair_categories": [],
                    "failure_family": None,
                    "wrapper_recovery_applied": False,
                    "quote_normalization_applied": False,
                    "duplicate_candidate_deduplication_applied": False,
                    "alias_canonicalization_applied": False,
                    "bbox_canonicalization_applied": False,
                    "selected_tool_name": None,
                    "selected_canonical_arguments": None,
                    "extracted_candidate_count": 0,
                    "unique_candidate_count": 0,
                    "error": None,
                },
            }
            if retry_event == "tool_call":
                first_turn_gate_recovered = True
                first_turn_gate_outcome = "recovered_to_tool_call"
                first_turn_gate_sidecar["final_gate_outcome"] = first_turn_gate_outcome
                _queue_optional_json(first_turn_gate_sidecar_path, first_turn_gate_sidecar)
                response = retry_response
                raw_output_path = retry_raw_output_path
                decision = retry_decision
                event = retry_event
                normalization_metadata = retry_normalization_metadata
            elif retry_event == "final_answer":
                terminal_answer_present = True
                terminal_answer_turn_index = turn_index
                try:
                    prediction = _timed_parse_final_answer(retry_response.raw_output)
                    parser_valid = True
                    schema_valid = True
                    first_turn_gate_outcome = "still_terminal_after_retry"
                except Exception as exc:  # noqa: BLE001 - explicit gate failure path.
                    error_message = str(exc)
                    failure_reason = _failure_reason("parser_invalid", str(exc))
                    first_turn_gate_outcome = "retry_parse_failure"
                first_turn_gate_sidecar["final_gate_outcome"] = first_turn_gate_outcome
                _queue_optional_json(first_turn_gate_sidecar_path, first_turn_gate_sidecar)
                _append_final_answer_message(
                    history,
                    retry_response.raw_output,
                    backend_name=retry_response.backend_name,
                    raw_output_path=str(retry_raw_output_path.resolve()),
                    error_message=error_message,
                )
                break
            else:
                repair_decision = repair_retry_tool_call_output(
                    retry_response.raw_output,
                    tool_path=definition["mode"],
                )
                first_turn_gate_repair_attempted = repair_decision.attempted
                first_turn_gate_repair_succeeded = repair_decision.succeeded
                first_turn_gate_repair_outcome = (
                    "repaired_to_tool_call" if repair_decision.succeeded else "repair_failed"
                )
                first_turn_gate_repair_categories = list(repair_decision.repair_categories)
                first_turn_gate_repair_original_failure_family = repair_decision.original_failure_family
                first_turn_gate_repair_failure_family = repair_decision.failure_family
                first_turn_gate_sidecar["retry_repair"] = repair_decision.to_audit_payload()
                if repair_decision.succeeded and repair_decision.parsed_call is not None:
                    first_turn_gate_recovered = True
                    first_turn_gate_outcome = "recovered_to_tool_call"
                    first_turn_gate_sidecar["final_gate_outcome"] = first_turn_gate_outcome
                    _queue_optional_json(first_turn_gate_sidecar_path, first_turn_gate_sidecar)
                    _append_tool_request(
                        history,
                        retry_response.raw_output,
                        backend_name=retry_response.backend_name,
                        raw_output_path=str(retry_raw_output_path.resolve()),
                        tool_name=repair_decision.parsed_call.tool_name,
                        call_id=repair_decision.parsed_call.call_id,
                        extra_metadata={
                            **(retry_normalization_metadata or {}),
                            "retry_repair_applied": True,
                            "retry_repair_succeeded": True,
                            "retry_repair_categories": list(repair_decision.repair_categories),
                            "retry_repair_original_failure_family": repair_decision.original_failure_family,
                        },
                    )
                    tool_exec_start = time.perf_counter()
                    tool_result = execute_tool_call(
                        repair_decision.parsed_call,
                        sample=sample,
                        sample_pool=sample_pool,
                        artifact_dir=directories["raw_outputs"] / f"seed_{seed}" / sample_slug / "tool_artifacts",
                    )
                    sample_timing["tool_exec_ms"] += _elapsed_ms(tool_exec_start)
                    tool_payload = tool_result.to_payload()
                    tool_traces.append(tool_payload)
                    _mark_first_successful_tool(
                        tool_name=tool_result.tool_name,
                        turn_index=turn_index,
                    )
                    history = reinsert_tool_result(history, tool_result)
                    continue

                error_message = repair_decision.error or (
                    "First-turn gate retry did not contain a valid tool call or final answer block"
                )
                failure_reason = (
                    f"runtime_exception:{repair_decision.failure_family}"
                    if repair_decision.failure_family is not None
                    else "runtime_exception:first_turn_gate_retry_repair_failed"
                )
                first_turn_gate_outcome = "retry_parse_failure"
                first_turn_gate_sidecar["final_gate_outcome"] = first_turn_gate_outcome
                _queue_optional_json(first_turn_gate_sidecar_path, first_turn_gate_sidecar)
                _append_final_answer_message(
                    history,
                    retry_response.raw_output,
                    backend_name=retry_response.backend_name,
                    raw_output_path=str(retry_raw_output_path.resolve()),
                    error_message=error_message,
                )
                break
        if event == "tool_call":
            if len(tool_traces) >= definition["max_tool_turns"]:
                error_message = f"Exceeded max_tool_turns={definition['max_tool_turns']} before final answer"
                failure_reason = "runtime_exception:exceeded_max_tool_turns"
                parsed_call = decision.parsed_call
                _finalize_post_pz_transition_audit(
                    turn_index=turn_index,
                    raw_output_path=raw_output_path,
                    event=event,
                    second_turn_called_cr=bool(parsed_call is not None and parsed_call.tool_name == "CR"),
                    second_turn_called_non_cr_tool=bool(
                        parsed_call is not None and parsed_call.tool_name != "CR"
                    ),
                    second_turn_parser_valid_value=None,
                    second_turn_schema_valid_value=None,
                    second_turn_failure_reason_value=failure_reason,
                )
                _append_tool_request(
                    history,
                    response.raw_output,
                    backend_name=response.backend_name,
                    raw_output_path=str(raw_output_path.resolve()),
                    tool_name=None,
                    call_id=None,
                    error_message=error_message,
                    extra_metadata=normalization_metadata,
                )
                break
            try:
                parsed_call = decision.parsed_call or _timed_parse_tool_call(response.raw_output)
            except Exception as exc:  # noqa: BLE001 - explicit gate failure path.
                error_message = str(exc)
                failure_reason = _failure_reason("runtime_exception", str(exc))
                _finalize_post_pz_transition_audit(
                    turn_index=turn_index,
                    raw_output_path=raw_output_path,
                    event="continue",
                    second_turn_called_cr=False,
                    second_turn_called_non_cr_tool=False,
                    second_turn_parser_valid_value=False,
                    second_turn_schema_valid_value=False,
                    second_turn_failure_reason_value=failure_reason,
                )
                _append_tool_request(
                    history,
                    response.raw_output,
                    backend_name=response.backend_name,
                    raw_output_path=str(raw_output_path.resolve()),
                    tool_name=None,
                    call_id=None,
                    error_message=error_message,
                    extra_metadata=normalization_metadata,
                )
                break

            _append_tool_request(
                history,
                response.raw_output,
                backend_name=response.backend_name,
                raw_output_path=str(raw_output_path.resolve()),
                tool_name=parsed_call.tool_name,
                call_id=parsed_call.call_id,
                extra_metadata=normalization_metadata,
            )
            tool_exec_start = time.perf_counter()
            tool_result = execute_tool_call(
                parsed_call,
                sample=sample,
                sample_pool=sample_pool,
                artifact_dir=directories["raw_outputs"] / f"seed_{seed}" / sample_slug / "tool_artifacts",
            )
            sample_timing["tool_exec_ms"] += _elapsed_ms(tool_exec_start)
            tool_payload = tool_result.to_payload()
            tool_traces.append(tool_payload)
            _mark_first_successful_tool(
                tool_name=tool_result.tool_name,
                turn_index=turn_index,
            )
            _finalize_post_pz_transition_audit(
                turn_index=turn_index,
                raw_output_path=raw_output_path,
                event=event,
                second_turn_called_cr=tool_result.tool_name == "CR",
                second_turn_called_non_cr_tool=tool_result.tool_name != "CR",
                second_turn_parser_valid_value=None,
                second_turn_schema_valid_value=None,
                second_turn_failure_reason_value=None,
            )
            history = reinsert_tool_result(history, tool_result)
            continue

        if event == "final_answer":
            if _should_trigger_post_pz_second_turn_gate(
                gate_mode=post_pz_second_turn_gate_mode,
                tool_mode=definition["mode"],
                first_successful_tool_name=first_successful_tool_name,
                post_pz_transition_audited=post_pz_transition_audited,
                post_pz_assistant_turn_index=(
                    post_pz_transition_sidecar["post_pz_assistant_turn_index"]
                    if post_pz_transition_sidecar is not None
                    else None
                ),
                current_turn_index=turn_index,
                post_sanitation_contract_valid_for_cr=(
                    post_pz_transition_post_sanitation_contract_valid_for_cr
                ),
                event=event,
            ):
                post_pz_second_turn_gate_triggered = True
                post_pz_second_turn_gate_retry_attempted = True
                _finalize_post_pz_transition_audit(
                    turn_index=turn_index,
                    raw_output_path=raw_output_path,
                    event=event,
                    second_turn_called_cr=False,
                    second_turn_called_non_cr_tool=False,
                    second_turn_parser_valid_value=None,
                    second_turn_schema_valid_value=None,
                    second_turn_failure_reason_value=None,
                )
                retry_instruction_text = _post_pz_second_turn_gate_retry_instruction()
                _append_final_answer_message(
                    history,
                    response.raw_output,
                    backend_name=response.backend_name,
                    raw_output_path=str(raw_output_path.resolve()),
                )
                _append_runtime_gate_reminder(
                    history,
                    retry_instruction_text,
                    gate_mode=post_pz_second_turn_gate_mode,
                    runtime_intervention="post_pz_second_turn_protocol_gate",
                    gate_mode_metadata_key="post_pz_second_turn_gate_mode",
                )
                sample_timing["retry_count"] += 1
                retry_request_stage_start = time.perf_counter()
                retry_request = BackendRequest(
                    sample_id=sample["sample_id"],
                    seed=seed,
                    prompt_version=prompt_bundle.prompt_version,
                    messages=history,
                    stop_sequences=prompt_bundle.stop_sequences,
                    tool_mode=definition["mode"],
                    generation_config=_generation_config_for_stage(
                        runtime_config,
                        "post_pz_second_turn_retry",
                    ),
                    metadata={
                        "tool_mode": definition["mode"],
                        "turn_index": turn_index,
                        "retry_count": 1,
                        "runtime_intervention": "post_pz_second_turn_protocol_gate",
                        "generation_stage": "post_pz_second_turn_retry",
                        "sample_kind": sample["metadata"].get("dataset_kind", "unknown"),
                    },
                )
                sample_timing["request_build_or_processor_ms"] += _elapsed_ms(retry_request_stage_start)
                retry_generate_start = time.perf_counter()
                retry_response = backend.generate(retry_request, sample=sample)
                sample_timing["generate_ms"] += _elapsed_ms(retry_generate_start)
                sample_timing["generation_call_count"] += 1
                retry_raw_output_path = _retry_raw_output_path(
                    directories["raw_outputs"],
                    seed=seed,
                    sample_slug=sample_slug,
                    turn_index=turn_index,
                    retry_count=1,
                )
                post_pz_second_turn_gate_retry_raw_output_path = retry_raw_output_path
                retry_decision, retry_event, retry_normalization_metadata = _record_response_and_normalize(
                    retry_response,
                    turn_index=turn_index,
                    raw_output_path=retry_raw_output_path,
                )

                if retry_event == "tool_call":
                    try:
                        retry_parsed_call = retry_decision.parsed_call or _timed_parse_tool_call(
                            retry_response.raw_output
                        )
                    except Exception as exc:  # noqa: BLE001 - bounded strict-retry failure path.
                        error_message = str(exc)
                        failure_reason = _failure_reason("runtime_exception", str(exc))
                        post_pz_second_turn_gate_outcome = "retry_parse_failure"
                        post_pz_second_turn_gate_retry_parser_valid = False
                        post_pz_second_turn_gate_retry_schema_valid = False
                        post_pz_second_turn_gate_retry_failure_reason = failure_reason
                        _write_post_pz_second_turn_gate_sidecar(
                            turn_index=turn_index,
                            first_attempt_raw_output_path=raw_output_path,
                            retry_instruction_text=retry_instruction_text,
                            retry_raw_output_path=retry_raw_output_path,
                            final_gate_outcome=post_pz_second_turn_gate_outcome,
                            retry_called_tool_name=None,
                            retry_parser_valid_value=False,
                            retry_schema_valid_value=False,
                            retry_failure_reason_value=failure_reason,
                        )
                        _append_tool_request(
                            history,
                            retry_response.raw_output,
                            backend_name=retry_response.backend_name,
                            raw_output_path=str(retry_raw_output_path.resolve()),
                            tool_name=None,
                            call_id=None,
                            error_message=error_message,
                            extra_metadata=retry_normalization_metadata,
                        )
                        break

                    post_pz_second_turn_gate_retry_called_tool_name = retry_parsed_call.tool_name
                    post_pz_second_turn_gate_retry_parser_valid = True
                    post_pz_second_turn_gate_retry_schema_valid = None
                    if retry_parsed_call.tool_name != "CR":
                        error_message = (
                            "Post-PZ second-turn gate retry called a non-CR tool after the CR reminder"
                        )
                        failure_reason = "runtime_exception:post_pz_second_turn_gate_called_non_cr_tool_after_retry"
                        post_pz_second_turn_gate_outcome = "called_non_cr_tool_after_retry"
                        post_pz_second_turn_gate_retry_failure_reason = failure_reason
                        _write_post_pz_second_turn_gate_sidecar(
                            turn_index=turn_index,
                            first_attempt_raw_output_path=raw_output_path,
                            retry_instruction_text=retry_instruction_text,
                            retry_raw_output_path=retry_raw_output_path,
                            final_gate_outcome=post_pz_second_turn_gate_outcome,
                            retry_called_tool_name=retry_parsed_call.tool_name,
                            retry_parser_valid_value=True,
                            retry_schema_valid_value=None,
                            retry_failure_reason_value=failure_reason,
                        )
                        _append_tool_request(
                            history,
                            retry_response.raw_output,
                            backend_name=retry_response.backend_name,
                            raw_output_path=str(retry_raw_output_path.resolve()),
                            tool_name=retry_parsed_call.tool_name,
                            call_id=retry_parsed_call.call_id,
                            error_message=error_message,
                            extra_metadata=retry_normalization_metadata,
                        )
                        break

                    history = _sanitize_post_pz_second_turn_gate_history(history)
                    _append_tool_request(
                        history,
                        retry_response.raw_output,
                        backend_name=retry_response.backend_name,
                        raw_output_path=str(retry_raw_output_path.resolve()),
                        tool_name=retry_parsed_call.tool_name,
                        call_id=retry_parsed_call.call_id,
                        extra_metadata=retry_normalization_metadata,
                    )
                    try:
                        tool_exec_start = time.perf_counter()
                        retry_tool_result = execute_tool_call(
                            retry_parsed_call,
                            sample=sample,
                            sample_pool=sample_pool,
                            artifact_dir=directories["raw_outputs"] / f"seed_{seed}" / sample_slug / "tool_artifacts",
                        )
                        sample_timing["tool_exec_ms"] += _elapsed_ms(tool_exec_start)
                    except Exception as exc:  # noqa: BLE001 - explicit gate-execution failure path.
                        error_message = str(exc)
                        failure_reason = _failure_reason("runtime_exception", str(exc))
                        post_pz_second_turn_gate_outcome = (
                            "called_cr_but_later_failed_execution_or_contract"
                        )
                        post_pz_second_turn_gate_retry_failure_reason = failure_reason
                        _write_post_pz_second_turn_gate_sidecar(
                            turn_index=turn_index,
                            first_attempt_raw_output_path=raw_output_path,
                            retry_instruction_text=retry_instruction_text,
                            retry_raw_output_path=retry_raw_output_path,
                            final_gate_outcome=post_pz_second_turn_gate_outcome,
                            retry_called_tool_name="CR",
                            retry_parser_valid_value=True,
                            retry_schema_valid_value=None,
                            retry_failure_reason_value=failure_reason,
                        )
                        break

                    post_pz_second_turn_gate_outcome = "recovered_to_cr_call"
                    _write_post_pz_second_turn_gate_sidecar(
                        turn_index=turn_index,
                        first_attempt_raw_output_path=raw_output_path,
                        retry_instruction_text=retry_instruction_text,
                        retry_raw_output_path=retry_raw_output_path,
                        final_gate_outcome=post_pz_second_turn_gate_outcome,
                        retry_called_tool_name="CR",
                        retry_parser_valid_value=True,
                        retry_schema_valid_value=None,
                        retry_failure_reason_value=None,
                    )
                    retry_tool_payload = retry_tool_result.to_payload()
                    tool_traces.append(retry_tool_payload)
                    history = reinsert_tool_result(history, retry_tool_result)
                    continue

                if retry_event == "final_answer":
                    terminal_answer_present = True
                    terminal_answer_turn_index = turn_index
                    try:
                        prediction = _timed_parse_final_answer(retry_response.raw_output)
                        parser_valid = True
                        schema_valid = True
                        post_pz_second_turn_gate_outcome = "still_terminal_after_retry"
                        post_pz_second_turn_gate_retry_parser_valid = True
                        post_pz_second_turn_gate_retry_schema_valid = True
                    except Exception as exc:  # noqa: BLE001 - bounded strict-retry failure path.
                        error_message = str(exc)
                        failure_reason = _failure_reason("parser_invalid", str(exc))
                        post_pz_second_turn_gate_outcome = "retry_parse_failure"
                        post_pz_second_turn_gate_retry_parser_valid = False
                        post_pz_second_turn_gate_retry_schema_valid = False
                        post_pz_second_turn_gate_retry_failure_reason = failure_reason
                    _write_post_pz_second_turn_gate_sidecar(
                        turn_index=turn_index,
                        first_attempt_raw_output_path=raw_output_path,
                        retry_instruction_text=retry_instruction_text,
                        retry_raw_output_path=retry_raw_output_path,
                        final_gate_outcome=post_pz_second_turn_gate_outcome,
                        retry_called_tool_name=None,
                        retry_parser_valid_value=post_pz_second_turn_gate_retry_parser_valid,
                        retry_schema_valid_value=post_pz_second_turn_gate_retry_schema_valid,
                        retry_failure_reason_value=post_pz_second_turn_gate_retry_failure_reason,
                    )
                    _append_final_answer_message(
                        history,
                        retry_response.raw_output,
                        backend_name=retry_response.backend_name,
                        raw_output_path=str(retry_raw_output_path.resolve()),
                        error_message=error_message,
                    )
                    break

                error_message = "Post-PZ second-turn gate retry did not contain a valid CR tool call or final answer block"
                failure_reason = "runtime_exception:post_pz_second_turn_gate_retry_missing_contract_block"
                post_pz_second_turn_gate_outcome = "retry_parse_failure"
                post_pz_second_turn_gate_retry_parser_valid = False
                post_pz_second_turn_gate_retry_schema_valid = False
                post_pz_second_turn_gate_retry_failure_reason = failure_reason
                _write_post_pz_second_turn_gate_sidecar(
                    turn_index=turn_index,
                    first_attempt_raw_output_path=raw_output_path,
                    retry_instruction_text=retry_instruction_text,
                    retry_raw_output_path=retry_raw_output_path,
                    final_gate_outcome=post_pz_second_turn_gate_outcome,
                    retry_called_tool_name=None,
                    retry_parser_valid_value=False,
                    retry_schema_valid_value=False,
                    retry_failure_reason_value=failure_reason,
                )
                _append_final_answer_message(
                    history,
                    retry_response.raw_output,
                    backend_name=retry_response.backend_name,
                    raw_output_path=str(retry_raw_output_path.resolve()),
                    error_message=error_message,
                )
                break

            terminal_answer_present = True
            terminal_answer_turn_index = turn_index
            try:
                prediction = _timed_parse_final_answer(response.raw_output)
                parser_valid = True
                schema_valid = True
            except Exception as exc:  # noqa: BLE001 - explicit gate failure path.
                error_message = str(exc)
                failure_reason = _failure_reason("parser_invalid", str(exc))
            _finalize_post_pz_transition_audit(
                turn_index=turn_index,
                raw_output_path=raw_output_path,
                event=event,
                second_turn_called_cr=False,
                second_turn_called_non_cr_tool=False,
                second_turn_parser_valid_value=parser_valid,
                second_turn_schema_valid_value=schema_valid,
                second_turn_failure_reason_value=failure_reason,
            )
            _append_final_answer_message(
                history,
                response.raw_output,
                backend_name=response.backend_name,
                raw_output_path=str(raw_output_path.resolve()),
                error_message=error_message,
            )
            break

        error_message = "Assistant output did not contain a valid tool call or final answer block"
        failure_reason = "runtime_exception:assistant_output_missing_contract_block"
        _finalize_post_pz_transition_audit(
            turn_index=turn_index,
            raw_output_path=raw_output_path,
            event=event,
            second_turn_called_cr=False,
            second_turn_called_non_cr_tool=False,
            second_turn_parser_valid_value=False,
            second_turn_schema_valid_value=False,
            second_turn_failure_reason_value=failure_reason,
        )
        _append_final_answer_message(
            history,
            response.raw_output,
            backend_name=response.backend_name,
            raw_output_path=str(raw_output_path.resolve()),
            error_message=error_message,
        )
        break

    if last_raw_output_path is None:
        raise InferenceRunError("Tool loop exited without producing any raw output")
    if last_raw_output_text is None:
        raise InferenceRunError("Tool loop exited without retaining the last raw output text")

    failure_reason = _resolve_failure_reason(
        prediction=prediction,
        parser_valid=parser_valid,
        schema_valid=schema_valid,
        failure_reason=failure_reason,
        error_message=error_message,
    )

    trace = TraceRecord(
        trace_id=f"{definition['run_id']}:{seed}:{sample['sample_id']}",
        sample_id=sample["sample_id"],
        stage="eval",
        tool_path=definition["mode"],
        storage_purpose="eval",
        messages=_history_to_trace_messages(history),
        tool_traces=tuple(tool_traces),
        final_answer=prediction,
        metadata={
            "seed": seed,
            "sample_category": sample["category"],
            "tool_mode": definition["mode"],
            "prompt_audit_path": str(prompt_audit_path.resolve()),
            "prompt_audit_mismatch": prompt_audit["mode_contract_mismatch"],
            "prompt_audit_mismatch_reasons": prompt_audit["mismatch_reasons"],
            "prompt_audit": prompt_audit,
            "normalization_event_count": len(normalization_events),
            "normalization_event_paths": normalization_event_paths,
            "normalization_event_reasons": [
                event["reason"] for event in normalization_events if event.get("reason") is not None
            ],
            "normalization_events": normalization_events,
            "first_turn_protocol_gate_mode": first_turn_protocol_gate_mode,
            "first_turn_gate_triggered": first_turn_gate_triggered,
            "first_turn_gate_retry_count": first_turn_gate_retry_count,
            "first_turn_gate_recovered": first_turn_gate_recovered,
            "first_turn_gate_outcome": first_turn_gate_outcome,
            "first_turn_gate_repair_attempted": first_turn_gate_repair_attempted,
            "first_turn_gate_repair_succeeded": first_turn_gate_repair_succeeded,
            "first_turn_gate_repair_outcome": first_turn_gate_repair_outcome,
            "first_turn_gate_repair_categories": first_turn_gate_repair_categories,
            "first_turn_gate_repair_original_failure_family": first_turn_gate_repair_original_failure_family,
            "first_turn_gate_repair_failure_family": first_turn_gate_repair_failure_family,
            "first_turn_gate_sidecar_path": (
                str(first_turn_gate_sidecar_path.resolve()) if first_turn_gate_sidecar_path is not None else None
            ),
            "first_turn_gate": first_turn_gate_sidecar,
            "first_successful_tool_name": first_successful_tool_name,
            "first_successful_tool_turn_index": first_successful_tool_turn_index,
            "post_pz_transition_audited": post_pz_transition_audited,
            "post_pz_transition_sidecar_path": (
                str(post_pz_transition_sidecar_path.resolve())
                if post_pz_transition_sidecar_path is not None
                else None
            ),
            "post_pz_transition_contract_valid_for_cr": post_pz_transition_contract_valid_for_cr,
            "post_pz_transition_mismatch_reasons": post_pz_transition_mismatch_reasons,
            "post_pz_transition_sanitation_applied": post_pz_transition_sanitation_applied,
            "post_pz_transition_sanitation_reason": post_pz_transition_sanitation_reason,
            "post_pz_transition_removed_message_count": post_pz_transition_removed_message_count,
            "post_pz_transition_removed_obsolete_terminal_answer_count": (
                post_pz_transition_removed_obsolete_terminal_answer_count
            ),
            "post_pz_transition_removed_pz_only_leakage_message_count": (
                post_pz_transition_removed_pz_only_leakage_message_count
            ),
            "post_pz_transition_pre_sanitation_pz_only_leakage_present": (
                post_pz_transition_pre_sanitation_pz_only_leakage_present
            ),
            "post_pz_transition_post_sanitation_pz_only_leakage_present": (
                post_pz_transition_post_sanitation_pz_only_leakage_present
            ),
            "post_pz_transition_post_sanitation_contract_valid_for_cr": (
                post_pz_transition_post_sanitation_contract_valid_for_cr
            ),
            "post_pz_transition_post_sanitation_mismatch_reasons": (
                post_pz_transition_post_sanitation_mismatch_reasons
            ),
            "post_pz_second_turn_protocol_event_type": post_pz_second_turn_protocol_event_type,
            "post_pz_second_turn_direct_final_without_cr": post_pz_second_turn_direct_final_without_cr,
            "post_pz_second_turn_called_cr": post_pz_second_turn_called_cr,
            "post_pz_second_turn_called_non_cr_tool": post_pz_second_turn_called_non_cr_tool,
            "post_pz_second_turn_parser_valid": post_pz_second_turn_parser_valid,
            "post_pz_second_turn_schema_valid": post_pz_second_turn_schema_valid,
            "post_pz_second_turn_failure_reason": post_pz_second_turn_failure_reason,
            "post_pz_second_turn_gate_mode": post_pz_second_turn_gate_mode,
            "post_pz_second_turn_gate_triggered": post_pz_second_turn_gate_triggered,
            "post_pz_second_turn_gate_outcome": post_pz_second_turn_gate_outcome,
            "post_pz_second_turn_gate_retry_attempted": post_pz_second_turn_gate_retry_attempted,
            "post_pz_second_turn_gate_retry_called_tool_name": (
                post_pz_second_turn_gate_retry_called_tool_name
            ),
            "post_pz_second_turn_gate_retry_parser_valid": post_pz_second_turn_gate_retry_parser_valid,
            "post_pz_second_turn_gate_retry_schema_valid": post_pz_second_turn_gate_retry_schema_valid,
            "post_pz_second_turn_gate_retry_failure_reason": post_pz_second_turn_gate_retry_failure_reason,
            "post_pz_second_turn_gate_retry_raw_output_path": (
                str(post_pz_second_turn_gate_retry_raw_output_path.resolve())
                if post_pz_second_turn_gate_retry_raw_output_path is not None
                else None
            ),
            "post_pz_second_turn_gate_sidecar_path": (
                str(post_pz_second_turn_gate_sidecar_path.resolve())
                if post_pz_second_turn_gate_sidecar_path is not None
                else None
            ),
            "post_pz_transition": post_pz_transition_sidecar,
            "timing": dict(sample_timing),
        },
    )
    trace_payload = trace.to_audit_payload()
    trace_path = directories["traces"] / f"seed_{seed}" / f"{sample_slug}.json"
    _write_required_json(trace_path, trace_payload)

    tool_usage = _tool_usage_from_traces(tool_traces)
    record = build_prediction_record(
        run_id=definition["run_id"],
        sample=sample,
        seed=seed,
        tool_mode=definition["mode"],
        tool_usage=tool_usage,
        prompt_version=prompt_bundle.prompt_version,
        parser_version=definition["parser_version"],
        backend_name=definition["backend"]["name"],
        prediction=prediction,
        parser_valid=parser_valid,
        schema_valid=schema_valid,
        error_message=error_message,
        failure_reason=failure_reason,
        raw_output_path=str(last_raw_output_path.resolve()),
        raw_output_sha256=_sha256_text(last_raw_output_text),
        trace_path=str(trace_path.resolve()),
        first_turn_protocol_gate_mode=first_turn_protocol_gate_mode,
        post_pz_second_turn_gate_mode=post_pz_second_turn_gate_mode,
        first_turn_gate_triggered=first_turn_gate_triggered,
        first_turn_gate_retry_count=first_turn_gate_retry_count,
        first_turn_gate_recovered=first_turn_gate_recovered,
        first_turn_gate_outcome=first_turn_gate_outcome,
        first_turn_gate_repair_attempted=first_turn_gate_repair_attempted,
        first_turn_gate_repair_succeeded=first_turn_gate_repair_succeeded,
        first_turn_gate_repair_outcome=first_turn_gate_repair_outcome,
        first_turn_gate_repair_categories=first_turn_gate_repair_categories,
        first_turn_gate_repair_original_failure_family=first_turn_gate_repair_original_failure_family,
        first_turn_gate_repair_failure_family=first_turn_gate_repair_failure_family,
        first_turn_gate_sidecar_path=(
            str(first_turn_gate_sidecar_path.resolve()) if first_turn_gate_sidecar_path is not None else None
        ),
        first_successful_tool_name=first_successful_tool_name,
        first_successful_tool_turn_index=first_successful_tool_turn_index,
        post_pz_transition_audited=post_pz_transition_audited,
        post_pz_transition_sidecar_path=(
            str(post_pz_transition_sidecar_path.resolve()) if post_pz_transition_sidecar_path is not None else None
        ),
        post_pz_transition_contract_valid_for_cr=post_pz_transition_contract_valid_for_cr,
        post_pz_transition_mismatch_reasons=post_pz_transition_mismatch_reasons,
        post_pz_transition_sanitation_applied=post_pz_transition_sanitation_applied,
        post_pz_transition_sanitation_reason=post_pz_transition_sanitation_reason,
        post_pz_transition_removed_message_count=post_pz_transition_removed_message_count,
        post_pz_transition_removed_obsolete_terminal_answer_count=(
            post_pz_transition_removed_obsolete_terminal_answer_count
        ),
        post_pz_transition_removed_pz_only_leakage_message_count=(
            post_pz_transition_removed_pz_only_leakage_message_count
        ),
        post_pz_transition_pre_sanitation_pz_only_leakage_present=(
            post_pz_transition_pre_sanitation_pz_only_leakage_present
        ),
        post_pz_transition_post_sanitation_pz_only_leakage_present=(
            post_pz_transition_post_sanitation_pz_only_leakage_present
        ),
        post_pz_transition_post_sanitation_contract_valid_for_cr=(
            post_pz_transition_post_sanitation_contract_valid_for_cr
        ),
        post_pz_transition_post_sanitation_mismatch_reasons=(
            post_pz_transition_post_sanitation_mismatch_reasons
        ),
        post_pz_second_turn_protocol_event_type=post_pz_second_turn_protocol_event_type,
        post_pz_second_turn_direct_final_without_cr=post_pz_second_turn_direct_final_without_cr,
        post_pz_second_turn_called_cr=post_pz_second_turn_called_cr,
        post_pz_second_turn_called_non_cr_tool=post_pz_second_turn_called_non_cr_tool,
        post_pz_second_turn_parser_valid=post_pz_second_turn_parser_valid,
        post_pz_second_turn_schema_valid=post_pz_second_turn_schema_valid,
        post_pz_second_turn_failure_reason=post_pz_second_turn_failure_reason,
        post_pz_second_turn_gate_triggered=post_pz_second_turn_gate_triggered,
        post_pz_second_turn_gate_outcome=post_pz_second_turn_gate_outcome,
        post_pz_second_turn_gate_retry_attempted=post_pz_second_turn_gate_retry_attempted,
        post_pz_second_turn_gate_retry_called_tool_name=post_pz_second_turn_gate_retry_called_tool_name,
        post_pz_second_turn_gate_retry_parser_valid=post_pz_second_turn_gate_retry_parser_valid,
        post_pz_second_turn_gate_retry_schema_valid=post_pz_second_turn_gate_retry_schema_valid,
        post_pz_second_turn_gate_retry_failure_reason=post_pz_second_turn_gate_retry_failure_reason,
        post_pz_second_turn_gate_retry_raw_output_path=(
            str(post_pz_second_turn_gate_retry_raw_output_path.resolve())
            if post_pz_second_turn_gate_retry_raw_output_path is not None
            else None
        ),
        post_pz_second_turn_gate_sidecar_path=(
            str(post_pz_second_turn_gate_sidecar_path.resolve())
            if post_pz_second_turn_gate_sidecar_path is not None
            else None
        ),
        metadata={
            "sample_source_kind": sample["metadata"].get("dataset_kind"),
            "tool_trace_count": len(tool_traces),
            "tool_names": [trace["tool_name"] for trace in tool_traces],
            "prompt_audit_path": str(prompt_audit_path.resolve()),
            "prompt_audit_mismatch": prompt_audit["mode_contract_mismatch"],
            "prompt_audit_mismatch_reasons": prompt_audit["mismatch_reasons"],
            "prompt_audit": prompt_audit,
            "normalization_event_count": len(normalization_events),
            "normalization_event_paths": normalization_event_paths,
            "normalization_event_reasons": [
                event["reason"] for event in normalization_events if event.get("reason") is not None
            ],
            "normalization_events": normalization_events,
            "first_turn_protocol_gate_mode": first_turn_protocol_gate_mode,
            "first_turn_gate_triggered": first_turn_gate_triggered,
            "first_turn_gate_retry_count": first_turn_gate_retry_count,
            "first_turn_gate_recovered": first_turn_gate_recovered,
            "first_turn_gate_outcome": first_turn_gate_outcome,
            "first_turn_gate_repair_attempted": first_turn_gate_repair_attempted,
            "first_turn_gate_repair_succeeded": first_turn_gate_repair_succeeded,
            "first_turn_gate_repair_outcome": first_turn_gate_repair_outcome,
            "first_turn_gate_repair_categories": first_turn_gate_repair_categories,
            "first_turn_gate_repair_original_failure_family": first_turn_gate_repair_original_failure_family,
            "first_turn_gate_repair_failure_family": first_turn_gate_repair_failure_family,
            "first_turn_gate_sidecar_path": (
                str(first_turn_gate_sidecar_path.resolve()) if first_turn_gate_sidecar_path is not None else None
            ),
            "first_turn_gate": first_turn_gate_sidecar,
            "first_successful_tool_name": first_successful_tool_name,
            "first_successful_tool_turn_index": first_successful_tool_turn_index,
            "post_pz_transition_audited": post_pz_transition_audited,
            "post_pz_transition_sidecar_path": (
                str(post_pz_transition_sidecar_path.resolve())
                if post_pz_transition_sidecar_path is not None
                else None
            ),
            "post_pz_transition_contract_valid_for_cr": post_pz_transition_contract_valid_for_cr,
            "post_pz_transition_mismatch_reasons": post_pz_transition_mismatch_reasons,
            "post_pz_transition_sanitation_applied": post_pz_transition_sanitation_applied,
            "post_pz_transition_sanitation_reason": post_pz_transition_sanitation_reason,
            "post_pz_transition_removed_message_count": post_pz_transition_removed_message_count,
            "post_pz_transition_removed_obsolete_terminal_answer_count": (
                post_pz_transition_removed_obsolete_terminal_answer_count
            ),
            "post_pz_transition_removed_pz_only_leakage_message_count": (
                post_pz_transition_removed_pz_only_leakage_message_count
            ),
            "post_pz_transition_pre_sanitation_pz_only_leakage_present": (
                post_pz_transition_pre_sanitation_pz_only_leakage_present
            ),
            "post_pz_transition_post_sanitation_pz_only_leakage_present": (
                post_pz_transition_post_sanitation_pz_only_leakage_present
            ),
            "post_pz_transition_post_sanitation_contract_valid_for_cr": (
                post_pz_transition_post_sanitation_contract_valid_for_cr
            ),
            "post_pz_transition_post_sanitation_mismatch_reasons": (
                post_pz_transition_post_sanitation_mismatch_reasons
            ),
            "post_pz_second_turn_protocol_event_type": post_pz_second_turn_protocol_event_type,
            "post_pz_second_turn_direct_final_without_cr": post_pz_second_turn_direct_final_without_cr,
            "post_pz_second_turn_called_cr": post_pz_second_turn_called_cr,
            "post_pz_second_turn_called_non_cr_tool": post_pz_second_turn_called_non_cr_tool,
            "post_pz_second_turn_parser_valid": post_pz_second_turn_parser_valid,
            "post_pz_second_turn_schema_valid": post_pz_second_turn_schema_valid,
            "post_pz_second_turn_failure_reason": post_pz_second_turn_failure_reason,
            "post_pz_second_turn_gate_mode": post_pz_second_turn_gate_mode,
            "post_pz_second_turn_gate_triggered": post_pz_second_turn_gate_triggered,
            "post_pz_second_turn_gate_outcome": post_pz_second_turn_gate_outcome,
            "post_pz_second_turn_gate_retry_attempted": post_pz_second_turn_gate_retry_attempted,
            "post_pz_second_turn_gate_retry_called_tool_name": (
                post_pz_second_turn_gate_retry_called_tool_name
            ),
            "post_pz_second_turn_gate_retry_parser_valid": post_pz_second_turn_gate_retry_parser_valid,
            "post_pz_second_turn_gate_retry_schema_valid": post_pz_second_turn_gate_retry_schema_valid,
            "post_pz_second_turn_gate_retry_failure_reason": post_pz_second_turn_gate_retry_failure_reason,
            "post_pz_second_turn_gate_retry_raw_output_path": (
                str(post_pz_second_turn_gate_retry_raw_output_path.resolve())
                if post_pz_second_turn_gate_retry_raw_output_path is not None
                else None
            ),
            "post_pz_second_turn_gate_sidecar_path": (
                str(post_pz_second_turn_gate_sidecar_path.resolve())
                if post_pz_second_turn_gate_sidecar_path is not None
                else None
            ),
            "post_pz_transition": post_pz_transition_sidecar,
            "timing": dict(sample_timing),
            "runtime_provenance": _resolved_runtime_provenance(
                definition,
                runtime_config,
                backend,
            ),
        },
        **build_zero_tool_behavior_fields(
            first_protocol_event_type=first_protocol_event_type,
            called_tools=[trace["tool_name"] for trace in tool_traces],
            terminal_answer_present=terminal_answer_present,
            terminal_answer_turn_index=terminal_answer_turn_index,
            prediction=prediction,
        ),
    )
    _flush_optional_sample_artifacts(
        _should_write_sample_artifacts(
            artifact_level=artifact_level,
            failure_reason=failure_reason,
            first_turn_gate_triggered=first_turn_gate_triggered,
            first_turn_gate_outcome=first_turn_gate_outcome,
            first_turn_gate_repair_attempted=first_turn_gate_repair_attempted,
            first_turn_gate_repair_succeeded=first_turn_gate_repair_succeeded,
            post_pz_second_turn_gate_triggered=post_pz_second_turn_gate_triggered,
            post_pz_second_turn_gate_outcome=post_pz_second_turn_gate_outcome,
        )
    )
    record["metadata"]["timing"] = dict(sample_timing)
    prediction_path = directories["predictions"] / f"seed_{seed}" / f"{sample_slug}.json"
    _write_required_json(prediction_path, record)
    record["metadata"]["timing"] = dict(sample_timing)
    return record, trace_payload


def run_tool_augmented(
    *,
    config_path: str | Path,
    dataset_root: str | Path | None = None,
    artifact_root: str | Path | None = None,
    max_samples: int | None = None,
    runtime_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run the canonical tool-enabled inference path for `pz_only` or `pz_cr`."""

    definition = load_run_definition(config_path)
    if definition["mode"] not in {"pz_only", "pz_cr"}:
        raise InferenceRunError("run_tool_augmented requires a `pz_only` or `pz_cr` config")

    runtime_config = _runtime_config(
        definition,
        dataset_root=dataset_root,
        runtime_overrides=runtime_overrides,
    )
    backend = _select_backend(definition["backend"], runtime_config=runtime_config)
    samples = _load_samples(definition, dataset_root=dataset_root, max_samples=max_samples)
    root = _resolve_path(artifact_root or definition["artifacts"]["root"])
    directories = _artifact_dirs(definition, root)
    progress_snapshot_path = None
    if runtime_config["progress_mode"] != "off":
        progress_snapshot_path = Path(
            runtime_config["progress_snapshot_path"] or _default_progress_snapshot_path(root)
        ).resolve()
        runtime_config["progress_snapshot_path"] = str(progress_snapshot_path)
    progress_reporter = ProgressReporter(
        run_id=definition["run_id"],
        total_samples=len(definition["seeds"]) * len(samples),
        mode=runtime_config["progress_mode"],
        update_every_n_samples=runtime_config["progress_update_every_n_samples"],
        snapshot_path=progress_snapshot_path,
    )
    processed_samples = 0
    progress_counts = {
        "generation_call_count_total": 0,
        "retry_count_total": 0,
    }

    prediction_records: list[dict[str, Any]] = []
    for seed in definition["seeds"]:
        seed_records: list[dict[str, Any]] = []
        for sample in samples:
            record, _trace_payload = _tool_loop_sample(
                definition=definition,
                backend=backend,
                runtime_config=runtime_config,
                sample=sample,
                sample_pool=samples,
                seed=seed,
                directories=directories,
            )
            seed_records.append(record)
            prediction_records.append(record)
            processed_samples += 1
            record_timing = record["metadata"].get("timing", {})
            progress_counts["generation_call_count_total"] += int(record_timing.get("generation_call_count", 0))
            progress_counts["retry_count_total"] += int(record_timing.get("retry_count", 0))
            progress_reporter.update(
                processed_samples=processed_samples,
                current_sample_id=sample["sample_id"],
                timing_summary=progress_counts,
            )
        write_stage_start = time.perf_counter()
        write_jsonl(directories["predictions"] / f"seed_{seed}.jsonl", seed_records)
        write_elapsed_ms = _elapsed_ms(write_stage_start)
        for record in seed_records:
            record["metadata"]["timing"]["file_write_ms"] += write_elapsed_ms / max(len(seed_records), 1)

    progress_reporter.update(
        processed_samples=processed_samples,
        current_sample_id=None,
        timing_summary=progress_counts,
        force=True,
    )

    runtime_provenance = _resolved_runtime_provenance(definition, runtime_config, backend)
    normalization_summary = _normalization_summary(prediction_records)
    prompt_audit_summary = _prompt_audit_summary(prediction_records)
    zero_tool_behavior_summary = summarize_zero_tool_behavior(prediction_records)
    post_pz_transition_summary = summarize_post_pz_transition(prediction_records)
    post_pz_transition_sanitation_summary = summarize_post_pz_transition_sanitation(
        prediction_records
    )
    post_pz_second_turn_gate_summary = summarize_post_pz_second_turn_gate(prediction_records)
    first_turn_gate_summary = _first_turn_gate_summary(prediction_records)
    first_turn_gate_repair_summary = _first_turn_gate_repair_summary(prediction_records)
    first_turn_gate_repair_failure_families = _first_turn_gate_repair_failure_family_artifact(
        prediction_records
    )
    zero_tool_sidecars = write_zero_tool_behavior_sidecars(
        prediction_records=prediction_records,
        metrics_dir=directories["metrics"],
    )
    post_pz_transition_sidecars = write_post_pz_transition_sidecars(
        prediction_records=prediction_records,
        metrics_dir=directories["metrics"],
    )
    post_pz_transition_sanitation_sidecars = write_post_pz_transition_sanitation_sidecars(
        prediction_records=prediction_records,
        metrics_dir=directories["metrics"],
    )
    post_pz_second_turn_gate_summary_path = write_post_pz_second_turn_gate_summary(
        prediction_records=prediction_records,
        metrics_dir=directories["metrics"],
    )
    first_turn_gate_repair_failure_families_path = (
        directories["metrics"] / "first_turn_gate_repair_failure_families.json"
    )
    write_json(
        first_turn_gate_repair_failure_families_path,
        first_turn_gate_repair_failure_families,
    )
    timing_summary = (
        _timing_summary_from_prediction_records(prediction_records)
        if runtime_config["timing_enabled"]
        else None
    )
    metrics_report = build_metrics_report(
        run_id=definition["run_id"],
        tool_mode=definition["mode"],
        prompt_version=definition["prompt_version"],
        parser_version=definition["parser_version"],
        backend_name=definition["backend"]["name"],
        prediction_records=prediction_records,
        seeds=definition["seeds"],
        runtime_provenance=runtime_provenance,
        artifact_level=runtime_provenance["artifact_level"],
        emit_baseline_compare=runtime_provenance["emit_baseline_compare"],
        emit_delta_report=runtime_provenance["emit_delta_report"],
        tool_first_intervention_strategy=runtime_provenance["tool_first_intervention_strategy"],
        first_turn_protocol_gate_mode=runtime_provenance["first_turn_protocol_gate_mode"],
        post_pz_second_turn_gate_mode=runtime_provenance["post_pz_second_turn_gate_mode"],
        timing_enabled=runtime_provenance["timing_enabled"],
        progress_mode=runtime_provenance["progress_mode"],
        prompt_audit_summary=prompt_audit_summary,
        zero_tool_behavior_summary=zero_tool_behavior_summary,
        post_pz_transition_summary=post_pz_transition_summary,
        post_pz_transition_sanitation_summary=post_pz_transition_sanitation_summary,
        post_pz_second_turn_gate_summary=post_pz_second_turn_gate_summary,
        first_turn_gate_summary=first_turn_gate_summary,
        first_turn_gate_repair_summary=first_turn_gate_repair_summary,
        timing_summary=timing_summary,
    )
    metrics_report_path = directories["metrics"] / "metrics_report.json"
    per_seed_metrics_path = directories["metrics"] / "per_seed_metrics.json"
    per_class_metrics_path = directories["metrics"] / "per_class_metrics.json"
    aggregate_metrics_path = directories["metrics"] / "aggregate_metrics.json"
    write_json(metrics_report_path, metrics_report)
    write_json(per_seed_metrics_path, {"per_seed_metrics": metrics_report["per_seed_metrics"]})
    write_json(per_class_metrics_path, {"per_class_metrics": metrics_report["per_class_metrics"]})
    write_json(aggregate_metrics_path, {"aggregate_metrics": metrics_report["aggregate_metrics"]})
    strategy_summary_path = write_tool_first_strategy_summary(
        metrics_dir=directories["metrics"],
        tool_first_intervention_strategy=runtime_provenance["tool_first_intervention_strategy"],
        runtime_provenance=runtime_provenance,
        prompt_audit_summary=prompt_audit_summary,
        zero_tool_behavior_summary=zero_tool_behavior_summary,
        metrics_report=metrics_report,
    )

    delta_report_path = directories["delta"] / "delta_vs_baseline.json"

    sample_source_path = _resolve_path(dataset_root or definition["sample_source"]["path"])
    run_metadata = _tool_metadata(
        definition=definition,
        config_path=config_path,
        sample_source_path=sample_source_path,
        compare_config_path=_resolve_path(definition["compare_to"]["config_path"]),
        runtime_provenance=runtime_provenance,
    )
    run_metadata_path = directories["root"] / "run_metadata.json"
    write_json(run_metadata_path, run_metadata)

    core_artifacts_written_before_optional_tail_work = True
    summary = build_run_summary(
        run_id=definition["run_id"],
        tool_mode=definition["mode"],
        backend_name=definition["backend"]["name"],
        prompt_version=definition["prompt_version"],
        parser_version=definition["parser_version"],
        execution_boundary=definition["execution_boundary"],
        sample_source={
            "kind": definition["sample_source"]["kind"],
            "path": str(sample_source_path),
            "source_name": definition["sample_source"]["source_name"],
        },
        seeds=definition["seeds"],
        prediction_records=prediction_records,
        runtime_provenance=runtime_provenance,
        artifact_level=runtime_provenance["artifact_level"],
        emit_baseline_compare=runtime_provenance["emit_baseline_compare"],
        emit_delta_report=runtime_provenance["emit_delta_report"],
        artifact_paths={
            "predictions_dir": str(directories["predictions"].resolve()),
            "traces_dir": str(directories["traces"].resolve()),
            "raw_outputs_dir": str(directories["raw_outputs"].resolve()),
            "metrics_report": str(metrics_report_path.resolve()),
            "per_seed_metrics": str(per_seed_metrics_path.resolve()),
            "per_class_metrics": str(per_class_metrics_path.resolve()),
            "aggregate_metrics": str(aggregate_metrics_path.resolve()),
            "run_metadata": str(run_metadata_path.resolve()),
            "run_manifest": str((directories["root"] / "run_manifest.json").resolve()),
            "per_dataset_zero_tool_behavior": zero_tool_sidecars["per_dataset_zero_tool_behavior"],
            "per_category_zero_tool_behavior": zero_tool_sidecars["per_category_zero_tool_behavior"],
            "post_pz_transition_summary": post_pz_transition_sidecars["post_pz_transition_summary"],
            "per_dataset_post_pz_transition": post_pz_transition_sidecars["per_dataset_post_pz_transition"],
            "per_category_post_pz_transition": post_pz_transition_sidecars["per_category_post_pz_transition"],
            "post_pz_transition_sanitation_summary": post_pz_transition_sanitation_sidecars[
                "post_pz_transition_sanitation_summary"
            ],
            "per_dataset_post_pz_transition_sanitation": post_pz_transition_sanitation_sidecars[
                "per_dataset_post_pz_transition_sanitation"
            ],
            "per_category_post_pz_transition_sanitation": post_pz_transition_sanitation_sidecars[
                "per_category_post_pz_transition_sanitation"
            ],
            "post_pz_second_turn_gate_summary": post_pz_second_turn_gate_summary_path,
            "first_turn_gate_repair_failure_families": str(
                first_turn_gate_repair_failure_families_path.resolve()
            ),
            "tool_first_strategy_summary": strategy_summary_path,
            **(
                {"progress_snapshot": str(progress_snapshot_path)}
                if progress_snapshot_path is not None
                else {}
            ),
            **(
                {"delta_report": str(delta_report_path.resolve())}
                if runtime_config["emit_delta_report"]
                else {}
            ),
        },
        notes=definition.get("notes", []),
        tool_first_intervention_strategy=runtime_provenance["tool_first_intervention_strategy"],
        first_turn_protocol_gate_mode=runtime_provenance["first_turn_protocol_gate_mode"],
        post_pz_second_turn_gate_mode=runtime_provenance["post_pz_second_turn_gate_mode"],
        timing_enabled=runtime_provenance["timing_enabled"],
        progress_mode=runtime_provenance["progress_mode"],
        normalization_summary=normalization_summary,
        prompt_audit_summary=prompt_audit_summary,
        zero_tool_behavior_summary=zero_tool_behavior_summary,
        post_pz_transition_summary=post_pz_transition_summary,
        post_pz_transition_sanitation_summary=post_pz_transition_sanitation_summary,
        post_pz_second_turn_gate_summary=post_pz_second_turn_gate_summary,
        first_turn_gate_summary=first_turn_gate_summary,
        first_turn_gate_repair_summary=first_turn_gate_repair_summary,
        timing_summary=timing_summary,
        core_artifacts_written_before_optional_tail_work=core_artifacts_written_before_optional_tail_work,
    )
    summary_path = directories["root"] / "run_summary.json"
    write_json(summary_path, summary)

    run_manifest = {
        "artifact_id": definition["run_id"],
        "artifact_type": "report",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "producer": definition["backend"]["name"],
        "content_hash": sha256_file(summary_path),
        "input_hashes": {
            "tool_run_definition": sha256_file(config_path),
            "metrics_report": sha256_file(metrics_report_path),
            "run_metadata": sha256_file(run_metadata_path),
        },
        "execution_boundary": definition["execution_boundary"],
        "artifact_level": runtime_provenance["artifact_level"],
        "emit_baseline_compare": runtime_provenance["emit_baseline_compare"],
        "emit_delta_report": runtime_provenance["emit_delta_report"],
        "tool_first_intervention_strategy": runtime_provenance["tool_first_intervention_strategy"],
        "first_turn_protocol_gate_mode": runtime_provenance["first_turn_protocol_gate_mode"],
        "post_pz_second_turn_gate_mode": runtime_provenance["post_pz_second_turn_gate_mode"],
        "timing_enabled": runtime_provenance["timing_enabled"],
        "progress_mode": runtime_provenance["progress_mode"],
        "run_provenance": runtime_provenance,
        "normalization_summary": normalization_summary,
        "prompt_audit_summary": prompt_audit_summary,
        "zero_tool_behavior_summary": zero_tool_behavior_summary,
        "post_pz_transition_summary": post_pz_transition_summary,
        "post_pz_transition_sanitation_summary": post_pz_transition_sanitation_summary,
        "post_pz_second_turn_gate_summary": post_pz_second_turn_gate_summary,
        "first_turn_gate_summary": first_turn_gate_summary,
        "first_turn_gate_repair_summary": first_turn_gate_repair_summary,
        **({"timing_summary": timing_summary} if timing_summary is not None else {}),
        "core_artifacts_written_before_optional_tail_work": core_artifacts_written_before_optional_tail_work,
        "notes": [
            "Tool-augmented inference summary manifest.",
            (
                "This run used the scripted mock backend for local smoke validation."
                if definition["backend"]["type"] == "mock"
                else "This manifest records a real evaluation-capable backend configuration."
            ),
        ],
    }
    validate_payload(run_manifest, "artifact_manifest.schema.json")
    run_manifest_path = directories["root"] / "run_manifest.json"
    write_json(run_manifest_path, run_manifest)

    baseline_result: dict[str, Any] | None = None
    baseline_compare_summary: str | None = None
    if runtime_config["emit_baseline_compare"] or runtime_config["emit_delta_report"]:
        tail_stage_start = time.perf_counter()
        compare_artifact_root = _comparison_artifact_root(definition, artifact_root)
        baseline_result = run_baseline(
            config_path=_resolve_path(definition["compare_to"]["config_path"]),
            dataset_root=dataset_root,
            artifact_root=compare_artifact_root,
            max_samples=max_samples,
            runtime_overrides=runtime_overrides,
        )
        if runtime_config["emit_baseline_compare"]:
            baseline_compare_summary = baseline_result["summary_path"]
        if runtime_config["emit_delta_report"] and baseline_result is not None:
            delta_report = build_delta_report(
                tool_run_id=definition["run_id"],
                tool_mode=definition["mode"],
                tool_metrics_report=metrics_report,
                baseline_run_id=baseline_result["run_definition"]["run_id"],
                baseline_metrics_report=baseline_result["metrics_report"],
                notes=[
                    "Local structural comparison against the Prompt 1.3 non-tool baseline.",
                    "This is smoke-validation evidence only and not a paper-quality model comparison.",
                ],
            )
            write_json(delta_report_path, delta_report)
        tail_compare_delta_ms_total = _elapsed_ms(tail_stage_start)
        if runtime_config["emit_delta_report"]:
            run_manifest["input_hashes"]["delta_report"] = sha256_file(delta_report_path)
        if timing_summary is not None:
            timing_summary = _timing_summary_from_prediction_records(
                prediction_records,
                tail_compare_delta_ms_total=tail_compare_delta_ms_total,
            )
            metrics_report["timing_summary"] = timing_summary
            summary["timing_summary"] = timing_summary
            run_manifest["timing_summary"] = timing_summary
            write_json(metrics_report_path, metrics_report)
            write_json(summary_path, summary)
        run_manifest["content_hash"] = sha256_file(summary_path)
        write_json(run_manifest_path, run_manifest)

    return {
        "run_definition": definition,
        "prediction_records": prediction_records,
        "metrics_report": metrics_report,
        "run_metadata_path": str(run_metadata_path.resolve()),
        "run_manifest_path": str(run_manifest_path.resolve()),
        "summary_path": str(summary_path.resolve()),
        "runtime_provenance": runtime_provenance,
        **(
            {"delta_report_path": str(delta_report_path.resolve())}
            if runtime_config["emit_delta_report"]
            else {}
        ),
        **(
            {"baseline_compare_summary": baseline_compare_summary}
            if baseline_compare_summary is not None
            else {}
        ),
    }


def run_from_config(
    *,
    config_path: str | Path,
    dataset_root: str | Path | None = None,
    artifact_root: str | Path | None = None,
    max_samples: int | None = None,
    runtime_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Dispatch the canonical inference runner from one externalized config."""

    definition = load_run_definition(config_path)
    if definition["mode"] == "no_tools":
        return run_baseline(
            config_path=config_path,
            dataset_root=dataset_root,
            artifact_root=artifact_root,
            max_samples=max_samples,
            runtime_overrides=runtime_overrides,
        )
    return run_tool_augmented(
        config_path=config_path,
        dataset_root=dataset_root,
        artifact_root=artifact_root,
        max_samples=max_samples,
        runtime_overrides=runtime_overrides,
    )


def dry_run_from_config(
    *,
    config_path: str | Path,
    dataset_root: str | Path | None = None,
    artifact_root: str | Path | None = None,
    max_samples: int | None = None,
    runtime_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate the unified baseline surface without loading a model."""

    definition = load_run_definition(config_path)
    runtime_config = _runtime_config(
        definition,
        dataset_root=dataset_root,
        runtime_overrides=runtime_overrides,
    )
    backend = _select_backend(definition["backend"], runtime_config=runtime_config)
    try:
        sample_count = len(_load_samples(definition, dataset_root=dataset_root, max_samples=max_samples))
    except InferenceRunError as exc:
        if "No canonical samples were selected" not in str(exc):
            raise
        sample_count = 0
    artifact_root_path = _resolve_path(artifact_root or definition["artifacts"]["root"])
    return {
        "run_id": definition["run_id"],
        "mode": definition["mode"],
        "sample_count": sample_count,
        "artifact_root": str(artifact_root_path),
        "runtime_provenance": _resolved_runtime_provenance(definition, runtime_config, backend),
    }


def _build_parser() -> argparse.ArgumentParser:
    """Build the small CLI used for local smoke runs and future remote handoff."""

    parser = argparse.ArgumentParser(
        description="Run the AgentIAD canonical inference pipeline for baseline or tool modes."
    )
    parser.add_argument("--config", required=True, help="Path to the run definition JSON.")
    parser.add_argument("--dataset-root", help="Optional dataset root override.")
    parser.add_argument("--artifact-root", help="Optional artifact root override.")
    parser.add_argument("--max-samples", type=int, help="Optional sample limit override.")
    parser.add_argument("--base-model-path", help="Optional base model path override.")
    parser.add_argument("--adapter-checkpoint-path", help="Optional LoRA adapter checkpoint override.")
    parser.add_argument("--checkpoint-step", type=int, help="Optional checkpoint step override.")
    parser.add_argument("--checkpoint-run-dir", help="Optional checkpoint run directory override.")
    parser.add_argument("--device", help="Optional device override, for example `auto`, `cuda`, or `cpu`.")
    parser.add_argument("--dtype", help="Optional dtype override, for example `auto`, `bfloat16`, or `float16`.")
    parser.add_argument("--local-files-only", dest="local_files_only", action="store_true")
    parser.add_argument("--no-local-files-only", dest="local_files_only", action="store_false")
    parser.set_defaults(local_files_only=None)
    parser.add_argument("--trust-remote-code", dest="trust_remote_code", action="store_true")
    parser.add_argument("--no-trust-remote-code", dest="trust_remote_code", action="store_false")
    parser.set_defaults(trust_remote_code=None)
    parser.add_argument("--max-new-tokens", type=int, help="Optional generation max_new_tokens override.")
    parser.add_argument("--temperature", type=float, help="Optional generation temperature override.")
    parser.add_argument("--top-p", type=float, help="Optional generation top_p override.")
    parser.add_argument("--do-sample", dest="do_sample", action="store_true")
    parser.add_argument("--no-do-sample", dest="do_sample", action="store_false")
    parser.set_defaults(do_sample=None)
    parser.add_argument(
        "--generation-stage-overrides-json",
        help="Optional JSON object overriding generation config by runtime stage.",
    )
    parser.add_argument(
        "--tool-first-intervention-strategy",
        choices=TOOL_FIRST_INTERVENTION_STRATEGIES,
        help="Optional first-turn prompt intervention strategy override for tool-enabled pz_cr eval.",
    )
    parser.add_argument(
        "--first-turn-protocol-gate-mode",
        choices=FIRST_TURN_PROTOCOL_GATE_MODES,
        help="Optional first-turn protocol gate override. Defaults to off.",
    )
    parser.add_argument(
        "--post-pz-second-turn-gate-mode",
        choices=POST_PZ_SECOND_TURN_GATE_MODES,
        help="Optional bounded post-PZ second-turn CR gate override. Defaults to off.",
    )
    parser.add_argument(
        "--emit-baseline-compare",
        type=_parse_bool_flag,
        help="Optional bool override for baseline-compare tail work.",
    )
    parser.add_argument(
        "--emit-delta-report",
        type=_parse_bool_flag,
        help="Optional bool override for delta-report tail work.",
    )
    parser.add_argument(
        "--artifact-level",
        choices=ARTIFACT_LEVELS,
        help="Optional artifact retention level override.",
    )
    parser.add_argument(
        "--timing-enabled",
        type=_parse_bool_flag,
        help="Optional bool override enabling timing instrumentation.",
    )
    parser.add_argument(
        "--progress-mode",
        choices=PROGRESS_MODES,
        help="Optional progress reporting mode override.",
    )
    parser.add_argument(
        "--progress-update-every-n-samples",
        type=int,
        help="Optional progress update interval override.",
    )
    parser.add_argument(
        "--progress-snapshot-path",
        help="Optional explicit machine-readable progress snapshot path override.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate config/runtime surfaces without inference.")
    return parser


def main() -> int:
    """Execute the inference CLI and print a compact success summary."""

    args = _build_parser().parse_args()
    generation_stage_overrides = None
    if args.generation_stage_overrides_json:
        generation_stage_overrides = json.loads(args.generation_stage_overrides_json)
    runtime_overrides = {
        "base_model_path": args.base_model_path,
        "adapter_checkpoint_path": args.adapter_checkpoint_path,
        "checkpoint_step": args.checkpoint_step,
        "checkpoint_run_dir": args.checkpoint_run_dir,
        "device": args.device,
        "dtype": args.dtype,
        "local_files_only": args.local_files_only,
        "trust_remote_code": args.trust_remote_code,
        "tool_first_intervention_strategy": args.tool_first_intervention_strategy,
        "first_turn_protocol_gate_mode": args.first_turn_protocol_gate_mode,
        "post_pz_second_turn_gate_mode": args.post_pz_second_turn_gate_mode,
        "generation_stage_overrides": generation_stage_overrides,
        "emit_baseline_compare": args.emit_baseline_compare,
        "emit_delta_report": args.emit_delta_report,
        "artifact_level": args.artifact_level,
        "timing_enabled": args.timing_enabled,
        "progress_mode": args.progress_mode,
        "progress_update_every_n_samples": args.progress_update_every_n_samples,
        "progress_snapshot_path": args.progress_snapshot_path,
        "generation": {
            key: value
            for key, value in {
                "max_new_tokens": args.max_new_tokens,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "do_sample": args.do_sample,
            }.items()
            if value is not None
        },
    }
    if args.dry_run:
        result = dry_run_from_config(
            config_path=args.config,
            dataset_root=args.dataset_root,
            artifact_root=args.artifact_root,
            max_samples=args.max_samples,
            runtime_overrides=runtime_overrides,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0

    result = run_from_config(
        config_path=args.config,
        dataset_root=args.dataset_root,
        artifact_root=args.artifact_root,
        max_samples=args.max_samples,
        runtime_overrides=runtime_overrides,
    )
    payload = {
        "run_manifest_path": result["run_manifest_path"],
        "summary_path": result["summary_path"],
        "runtime_provenance": result["runtime_provenance"],
    }
    if "delta_report_path" in result:
        payload["delta_report_path"] = result["delta_report_path"]
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
