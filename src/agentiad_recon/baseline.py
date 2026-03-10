"""Canonical non-tool baseline runner for AgentIAD.

This module is the Prompt 1.3 baseline entrypoint. It reuses the canonical
MMAD sample layer, the strict final-answer parser, and the audit trace schema
to run non-tool baseline inference with a thin backend interface. Use
`python -m agentiad_recon.baseline --help` for CLI usage; the local smoke path
should keep `backend.type=mock` and a tiny fixture dataset.
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
    VLLMBackendAdapter,
)
from agentiad_recon.contracts import validate_payload
from agentiad_recon.evaluation import (
    build_metrics_report,
    build_prediction_record,
    build_run_summary,
    safe_slug,
    write_json,
    write_jsonl,
)
from agentiad_recon.mmad import MMADIndexer
from agentiad_recon.prompting import (
    BASELINE_PROMPT_VERSION,
    FINAL_ANSWER_PARSER_VERSION,
    build_baseline_prompt,
    extract_think_block,
    parse_final_answer,
)
from agentiad_recon.reproducibility import build_run_metadata, sha256_file
from agentiad_recon.traces import TraceMessage, TraceRecord


REPO_ROOT = Path(__file__).resolve().parents[2]


class BaselineRunError(RuntimeError):
    """Raised when the baseline runner configuration or artifacts are invalid."""


def _resolve_path(value: str | Path) -> Path:
    """Resolve repository-relative paths deterministically."""

    path = Path(value)
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def load_run_definition(path: str | Path) -> dict[str, Any]:
    """Load and validate the canonical baseline run definition."""

    definition = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_payload(definition, "baseline_run_definition.schema.json")
    return definition


def _select_backend(config: dict[str, Any]) -> InferenceBackend:
    """Instantiate the requested backend while keeping runtime ownership thin."""

    backend_type = config["type"]
    if backend_type == "mock":
        return MockInferenceBackend(backend_name=config["name"], policy=config["policy"])
    if backend_type == "vllm":
        return VLLMBackendAdapter(backend_name=config["name"])
    raise BaselineRunError(f"Unsupported backend type: {backend_type}")


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
        raise BaselineRunError("No canonical samples were selected for the baseline run.")
    return samples


def _write_text(path: str | Path, payload: str) -> None:
    """Write one UTF-8 text artifact to disk."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")


def _trace_messages(prompt_bundle: Any, raw_output: str, *, backend_name: str, raw_output_path: str) -> tuple[TraceMessage, ...]:
    """Convert the baseline prompt and raw model output into trace messages."""

    messages = [
        TraceMessage(
            role=message["role"],
            message_type=message["message_type"],
            content=message["content"],
            image_refs=tuple(message["image_refs"]),
            metadata=message["metadata"],
        )
        for message in prompt_bundle.messages
    ]
    think_block = extract_think_block(raw_output)
    if think_block is not None:
        # Keeping the reasoning block separate makes later audit review easier.
        messages.append(
            TraceMessage(
                role="assistant",
                message_type="reasoning",
                content=think_block,
                metadata={"backend_name": backend_name},
            )
        )
    messages.append(
        TraceMessage(
            role="assistant",
            message_type="final_answer",
            content=raw_output,
            metadata={"backend_name": backend_name, "raw_output_path": raw_output_path},
        )
    )
    return tuple(messages)


def run_baseline(
    *,
    config_path: str | Path,
    dataset_root: str | Path | None = None,
    artifact_root: str | Path | None = None,
    max_samples: int | None = None,
) -> dict[str, Any]:
    """Run the canonical non-tool baseline and write auditable artifacts."""

    definition = load_run_definition(config_path)
    backend = _select_backend(definition["backend"])
    samples = _load_samples(definition, dataset_root=dataset_root, max_samples=max_samples)

    root = _resolve_path(artifact_root or definition["artifacts"]["root"])
    raw_root = root / definition["artifacts"]["raw_outputs"]
    trace_root = root / definition["artifacts"]["traces"]
    prediction_root = root / definition["artifacts"]["predictions"]
    metrics_root = root / definition["artifacts"]["metrics"]
    for directory in (raw_root, trace_root, prediction_root, metrics_root):
        directory.mkdir(parents=True, exist_ok=True)

    prediction_records: list[dict[str, Any]] = []
    for seed in definition["seeds"]:
        seed_records: list[dict[str, Any]] = []
        for sample in samples:
            prompt_bundle = build_baseline_prompt(sample)
            request = BackendRequest(
                sample_id=sample["sample_id"],
                seed=seed,
                prompt_version=prompt_bundle.prompt_version,
                messages=prompt_bundle.messages,
                stop_sequences=prompt_bundle.stop_sequences,
                metadata={"tool_mode": "no_tools"},
            )
            response = backend.generate(request, sample=sample)

            sample_slug = safe_slug(sample["sample_id"])
            raw_output_path = raw_root / f"seed_{seed}" / f"{sample_slug}.txt"
            _write_text(raw_output_path, response.raw_output)

            prediction: dict[str, Any] | None = None
            parser_valid = False
            schema_valid = False
            error_message: str | None = None
            try:
                prediction = parse_final_answer(response.raw_output)
                parser_valid = True
                schema_valid = True
            except Exception as exc:  # noqa: BLE001 - baseline gate needs explicit failures.
                error_message = str(exc)

            trace = TraceRecord(
                trace_id=f"{definition['run_id']}:{seed}:{sample['sample_id']}",
                sample_id=sample["sample_id"],
                stage="eval",
                tool_path="no_tools",
                storage_purpose="eval",
                messages=_trace_messages(
                    prompt_bundle,
                    response.raw_output,
                    backend_name=response.backend_name,
                    raw_output_path=str(raw_output_path.resolve()),
                ),
                tool_traces=(),
                final_answer=prediction,
                metadata={
                    "seed": seed,
                    "backend_metadata": response.metadata,
                    "sample_category": sample["category"],
                },
            )
            trace_payload = trace.to_audit_payload()
            trace_path = trace_root / f"seed_{seed}" / f"{sample_slug}.json"
            write_json(trace_path, trace_payload)

            record = build_prediction_record(
                run_id=definition["run_id"],
                sample=sample,
                seed=seed,
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
                },
            )
            prediction_path = prediction_root / f"seed_{seed}" / f"{sample_slug}.json"
            write_json(prediction_path, record)
            seed_records.append(record)
            prediction_records.append(record)

        write_jsonl(
            prediction_root / f"seed_{seed}.jsonl",
            seed_records,
        )

    metrics_report = build_metrics_report(
        run_id=definition["run_id"],
        prompt_version=definition["prompt_version"],
        parser_version=definition["parser_version"],
        backend_name=definition["backend"]["name"],
        prediction_records=prediction_records,
        seeds=definition["seeds"],
    )
    metrics_report_path = metrics_root / "metrics_report.json"
    per_seed_metrics_path = metrics_root / "per_seed_metrics.json"
    per_class_metrics_path = metrics_root / "per_class_metrics.json"
    aggregate_metrics_path = metrics_root / "aggregate_metrics.json"
    write_json(metrics_report_path, metrics_report)
    write_json(per_seed_metrics_path, {"per_seed_metrics": metrics_report["per_seed_metrics"]})
    write_json(per_class_metrics_path, {"per_class_metrics": metrics_report["per_class_metrics"]})
    write_json(aggregate_metrics_path, {"aggregate_metrics": metrics_report["aggregate_metrics"]})

    sample_source_path = _resolve_path(dataset_root or definition["sample_source"]["path"])
    fixture_manifest = sample_source_path / "fixture_manifest.json"
    dataset_manifest_hash = sha256_file(fixture_manifest) if fixture_manifest.exists() else None
    run_metadata = build_run_metadata(
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
            "Prompt 1.3 local-only baseline run.",
            "Mock backend outputs are deterministic smoke-validation artifacts, not real model inference.",
        ],
    )
    run_metadata_path = root / "run_metadata.json"
    write_json(run_metadata_path, run_metadata)

    summary = build_run_summary(
        run_id=definition["run_id"],
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
        artifact_paths={
            "predictions_dir": str(prediction_root.resolve()),
            "traces_dir": str(trace_root.resolve()),
            "raw_outputs_dir": str(raw_root.resolve()),
            "metrics_report": str(metrics_report_path.resolve()),
            "per_seed_metrics": str(per_seed_metrics_path.resolve()),
            "per_class_metrics": str(per_class_metrics_path.resolve()),
            "aggregate_metrics": str(aggregate_metrics_path.resolve()),
            "run_metadata": str(run_metadata_path.resolve()),
            "run_manifest": str((root / "run_manifest.json").resolve()),
        },
        notes=definition.get("notes", []),
    )
    summary_path = root / "run_summary.json"
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
        "notes": [
            "Non-tool baseline summary manifest.",
            "Tools remained disabled for the entire run.",
        ],
    }
    validate_payload(run_manifest, "artifact_manifest.schema.json")
    run_manifest_path = root / "run_manifest.json"
    write_json(run_manifest_path, run_manifest)

    return {
        "run_definition": definition,
        "prediction_records": prediction_records,
        "metrics_report": metrics_report,
        "run_metadata_path": str(run_metadata_path.resolve()),
        "run_manifest_path": str(run_manifest_path.resolve()),
        "summary_path": str(summary_path.resolve()),
    }


def _build_parser() -> argparse.ArgumentParser:
    """Build the small CLI used for local smoke runs and future remote handoff."""

    parser = argparse.ArgumentParser(description="Run the AgentIAD non-tool baseline pipeline.")
    parser.add_argument("--config", required=True, help="Path to the baseline run definition JSON.")
    parser.add_argument("--dataset-root", help="Optional dataset root override.")
    parser.add_argument("--artifact-root", help="Optional artifact root override.")
    parser.add_argument("--max-samples", type=int, help="Optional sample limit override.")
    return parser


def main() -> int:
    """Execute the baseline CLI and print a compact success summary."""

    args = _build_parser().parse_args()
    result = run_baseline(
        config_path=args.config,
        dataset_root=args.dataset_root,
        artifact_root=args.artifact_root,
        max_samples=args.max_samples,
    )
    print(json.dumps({"summary_path": result["summary_path"], "run_manifest_path": result["run_manifest_path"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
