"""Canonical inference runner for AgentIAD baseline and tool smoke paths.

This module remains the single inference entrypoint introduced in Prompt 1.3.
Prompt 1.4 extends it to dispatch `no_tools`, `pz_only`, and `pz_cr` runs from
external configs while preserving the same sample layer, prompt/parser helpers,
trace storage, evaluator family, and artifact grammar. Use
`python -m agentiad_recon.baseline --help` for local smoke execution.
"""

from __future__ import annotations

import argparse
import json
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
    build_baseline_prompt,
    build_prompt,
    extract_think_block,
    parse_final_answer,
)
from agentiad_recon.reproducibility import build_run_metadata, sha256_file
from agentiad_recon.tooling import (
    execute_tool_call,
    parse_tool_call,
    protocol_event,
    reinsert_tool_result,
)
from agentiad_recon.traces import TraceMessage, TraceRecord


REPO_ROOT = Path(__file__).resolve().parents[2]


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

    for key in (
        "base_model_path",
        "adapter_checkpoint_path",
        "checkpoint_step",
        "checkpoint_run_dir",
        "local_files_only",
        "trust_remote_code",
        "dtype",
        "device",
    ):
        if runtime_overrides and key in runtime_overrides and runtime_overrides[key] is not None:
            config[key] = runtime_overrides[key]

    inferred = _infer_checkpoint_provenance(config["adapter_checkpoint_path"])
    if config["checkpoint_step"] is None:
        config["checkpoint_step"] = inferred["checkpoint_step"]
    if config["checkpoint_run_dir"] is None:
        config["checkpoint_run_dir"] = inferred["checkpoint_run_dir"]

    resolved_dataset_root = _resolve_path(dataset_root or definition["sample_source"]["path"])
    config["dataset_root"] = str(resolved_dataset_root)
    config["tool_mode"] = definition["mode"]
    config["inference_mode"] = definition["mode"]
    config["runtime_backend_name"] = definition["backend"]["name"]
    config["runtime_backend_type"] = definition["backend"]["type"]
    config["runtime_owner"] = definition["backend"]["runtime_owner"]
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
        "local_files_only": runtime_config["local_files_only"],
        "trust_remote_code": runtime_config["trust_remote_code"],
        "dtype": runtime_config["dtype"],
        "device": runtime_config["device"],
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
) -> None:
    """Append one assistant tool request, including invalid attempts for audit."""

    metadata = {"backend_name": backend_name, "raw_output_path": raw_output_path}
    if error_message is not None:
        metadata["error_message"] = error_message
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

    prediction_records: list[dict[str, Any]] = []
    for seed in definition["seeds"]:
        seed_records: list[dict[str, Any]] = []
        for sample in samples:
            prompt_bundle = build_baseline_prompt(sample)
            history = _prompt_history(prompt_bundle)
            request = BackendRequest(
                sample_id=sample["sample_id"],
                seed=seed,
                prompt_version=prompt_bundle.prompt_version,
                messages=history,
                stop_sequences=prompt_bundle.stop_sequences,
                tool_mode="no_tools",
                metadata={
                    "tool_mode": "no_tools",
                    "sample_kind": sample["metadata"].get("dataset_kind", "unknown"),
                },
            )
            response = backend.generate(request, sample=sample)

            sample_slug = safe_slug(sample["sample_id"])
            raw_output_path = _raw_output_path(directories["raw_outputs"], seed=seed, sample_slug=sample_slug, turn_index=0)
            _write_text(raw_output_path, response.raw_output)

            prediction: dict[str, Any] | None = None
            parser_valid = False
            schema_valid = False
            error_message: str | None = None
            _append_reasoning(history, response.raw_output, backend_name=response.backend_name)
            try:
                prediction = parse_final_answer(response.raw_output)
                parser_valid = True
                schema_valid = True
            except Exception as exc:  # noqa: BLE001 - gate needs explicit failures.
                error_message = str(exc)
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
            write_json(trace_path, trace_payload)

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
                raw_output_path=str(raw_output_path.resolve()),
                raw_output_sha256=sha256_file(raw_output_path),
                trace_path=str(trace_path.resolve()),
                metadata={
                    "backend_metadata": response.metadata,
                    "sample_source_kind": sample["metadata"].get("dataset_kind"),
                    "runtime_provenance": _resolved_runtime_provenance(
                        definition,
                        runtime_config,
                        backend,
                    ),
                },
            )
            prediction_path = directories["predictions"] / f"seed_{seed}" / f"{sample_slug}.json"
            write_json(prediction_path, record)
            seed_records.append(record)
            prediction_records.append(record)

        write_jsonl(directories["predictions"] / f"seed_{seed}.jsonl", seed_records)

    runtime_provenance = _resolved_runtime_provenance(definition, runtime_config, backend)
    metrics_report = build_metrics_report(
        run_id=definition["run_id"],
        tool_mode="no_tools",
        prompt_version=definition["prompt_version"],
        parser_version=definition["parser_version"],
        backend_name=definition["backend"]["name"],
        prediction_records=prediction_records,
        seeds=definition["seeds"],
        runtime_provenance=runtime_provenance,
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
        },
        notes=definition.get("notes", []),
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
        "run_provenance": runtime_provenance,
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

    prompt_bundle = build_prompt(sample, tool_path=definition["mode"])
    history = _prompt_history(prompt_bundle)
    sample_slug = safe_slug(sample["sample_id"])
    tool_traces: list[dict[str, Any]] = []
    prediction: dict[str, Any] | None = None
    parser_valid = False
    schema_valid = False
    error_message: str | None = None
    last_raw_output_path: Path | None = None

    for turn_index in range(definition["max_tool_turns"] + 1):
        request = BackendRequest(
            sample_id=sample["sample_id"],
            seed=seed,
            prompt_version=prompt_bundle.prompt_version,
            messages=history,
            stop_sequences=prompt_bundle.stop_sequences,
            tool_mode=definition["mode"],
            metadata={
                "tool_mode": definition["mode"],
                "turn_index": turn_index,
                "sample_kind": sample["metadata"].get("dataset_kind", "unknown"),
            },
        )
        response = backend.generate(request, sample=sample)
        raw_output_path = _raw_output_path(
            directories["raw_outputs"],
            seed=seed,
            sample_slug=sample_slug,
            turn_index=turn_index,
        )
        _write_text(raw_output_path, response.raw_output)
        last_raw_output_path = raw_output_path

        _append_reasoning(history, response.raw_output, backend_name=response.backend_name)
        event = protocol_event(response.raw_output)
        if event == "tool_call":
            if len(tool_traces) >= definition["max_tool_turns"]:
                error_message = f"Exceeded max_tool_turns={definition['max_tool_turns']} before final answer"
                _append_tool_request(
                    history,
                    response.raw_output,
                    backend_name=response.backend_name,
                    raw_output_path=str(raw_output_path.resolve()),
                    tool_name=None,
                    call_id=None,
                    error_message=error_message,
                )
                break
            try:
                parsed_call = parse_tool_call(response.raw_output, tool_path=definition["mode"])
            except Exception as exc:  # noqa: BLE001 - explicit gate failure path.
                error_message = str(exc)
                _append_tool_request(
                    history,
                    response.raw_output,
                    backend_name=response.backend_name,
                    raw_output_path=str(raw_output_path.resolve()),
                    tool_name=None,
                    call_id=None,
                    error_message=error_message,
                )
                break

            _append_tool_request(
                history,
                response.raw_output,
                backend_name=response.backend_name,
                raw_output_path=str(raw_output_path.resolve()),
                tool_name=parsed_call.tool_name,
                call_id=parsed_call.call_id,
            )
            tool_result = execute_tool_call(
                parsed_call,
                sample=sample,
                sample_pool=sample_pool,
                artifact_dir=directories["raw_outputs"] / f"seed_{seed}" / sample_slug / "tool_artifacts",
            )
            tool_payload = tool_result.to_payload()
            tool_traces.append(tool_payload)
            history = reinsert_tool_result(history, tool_result)
            continue

        if event == "final_answer":
            try:
                prediction = parse_final_answer(response.raw_output)
                parser_valid = True
                schema_valid = True
            except Exception as exc:  # noqa: BLE001 - explicit gate failure path.
                error_message = str(exc)
            _append_final_answer_message(
                history,
                response.raw_output,
                backend_name=response.backend_name,
                raw_output_path=str(raw_output_path.resolve()),
                error_message=error_message,
            )
            break

        error_message = "Assistant output did not contain a valid tool call or final answer block"
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
        },
    )
    trace_payload = trace.to_audit_payload()
    trace_path = directories["traces"] / f"seed_{seed}" / f"{sample_slug}.json"
    write_json(trace_path, trace_payload)

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
        raw_output_path=str(last_raw_output_path.resolve()),
        raw_output_sha256=sha256_file(last_raw_output_path),
        trace_path=str(trace_path.resolve()),
        metadata={
            "sample_source_kind": sample["metadata"].get("dataset_kind"),
            "tool_trace_count": len(tool_traces),
            "tool_names": [trace["tool_name"] for trace in tool_traces],
            "runtime_provenance": _resolved_runtime_provenance(
                definition,
                runtime_config,
                backend,
            ),
        },
    )
    prediction_path = directories["predictions"] / f"seed_{seed}" / f"{sample_slug}.json"
    write_json(prediction_path, record)
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
        write_jsonl(directories["predictions"] / f"seed_{seed}.jsonl", seed_records)

    runtime_provenance = _resolved_runtime_provenance(definition, runtime_config, backend)
    metrics_report = build_metrics_report(
        run_id=definition["run_id"],
        tool_mode=definition["mode"],
        prompt_version=definition["prompt_version"],
        parser_version=definition["parser_version"],
        backend_name=definition["backend"]["name"],
        prediction_records=prediction_records,
        seeds=definition["seeds"],
        runtime_provenance=runtime_provenance,
    )
    metrics_report_path = directories["metrics"] / "metrics_report.json"
    per_seed_metrics_path = directories["metrics"] / "per_seed_metrics.json"
    per_class_metrics_path = directories["metrics"] / "per_class_metrics.json"
    aggregate_metrics_path = directories["metrics"] / "aggregate_metrics.json"
    write_json(metrics_report_path, metrics_report)
    write_json(per_seed_metrics_path, {"per_seed_metrics": metrics_report["per_seed_metrics"]})
    write_json(per_class_metrics_path, {"per_class_metrics": metrics_report["per_class_metrics"]})
    write_json(aggregate_metrics_path, {"aggregate_metrics": metrics_report["aggregate_metrics"]})

    compare_artifact_root = _comparison_artifact_root(definition, artifact_root)
    baseline_result = run_baseline(
        config_path=_resolve_path(definition["compare_to"]["config_path"]),
        dataset_root=dataset_root,
        artifact_root=compare_artifact_root,
        max_samples=max_samples,
        runtime_overrides=runtime_overrides,
    )
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
    delta_report_path = directories["delta"] / "delta_vs_baseline.json"
    write_json(delta_report_path, delta_report)

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
            "delta_report": str(delta_report_path.resolve()),
        },
        notes=definition.get("notes", []),
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
            "delta_report": sha256_file(delta_report_path),
        },
        "execution_boundary": definition["execution_boundary"],
        "run_provenance": runtime_provenance,
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

    return {
        "run_definition": definition,
        "prediction_records": prediction_records,
        "metrics_report": metrics_report,
        "delta_report_path": str(delta_report_path.resolve()),
        "baseline_compare_summary": baseline_result["summary_path"],
        "run_metadata_path": str(run_metadata_path.resolve()),
        "run_manifest_path": str(run_manifest_path.resolve()),
        "summary_path": str(summary_path.resolve()),
        "runtime_provenance": runtime_provenance,
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
    parser.add_argument("--dry-run", action="store_true", help="Validate config/runtime surfaces without inference.")
    return parser


def main() -> int:
    """Execute the inference CLI and print a compact success summary."""

    args = _build_parser().parse_args()
    runtime_overrides = {
        "base_model_path": args.base_model_path,
        "adapter_checkpoint_path": args.adapter_checkpoint_path,
        "checkpoint_step": args.checkpoint_step,
        "checkpoint_run_dir": args.checkpoint_run_dir,
        "device": args.device,
        "dtype": args.dtype,
        "local_files_only": args.local_files_only,
        "trust_remote_code": args.trust_remote_code,
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
