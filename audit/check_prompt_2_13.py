"""Acceptance check for Prompt 2.13 throughput controls, timing, and progress."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from agentiad_recon.backends import BackendRequest, BackendResponse, InferenceBackend
from agentiad_recon.baseline import run_tool_augmented


FIXTURE_ROOT = REPO_ROOT / "tests" / "fixtures" / "mmad_fixture"
PZ_CR_CONFIG = REPO_ROOT / "configs" / "tool_pz_cr_fixture.json"
TOOL_CALL_PZ = (
    '<tool_call>{"tool_name":"PZ","arguments":{"bbox":{"x0":0.10,"y0":0.10,"x1":0.80,"y1":0.80}}}</tool_call>'
)
TOOL_CALL_CR = (
    '<tool_call>{"tool_name":"CR","arguments":{"policy":"same_category_normal"}}</tool_call>'
)
FINAL_ANSWER = (
    "<answer>\n<think>final answer after tool use</think>\n"
    '{"anomaly_present": false, "top_anomaly": null, "visual_descriptions": []}\n</answer>'
)


class SequenceBackend(InferenceBackend):
    """Acceptance backend with deterministic scripted outputs."""

    backend_name = "prompt_2_13_acceptance_backend_v1"

    def __init__(self, outputs: list[str]) -> None:
        self.outputs = outputs
        self.calls = 0

    def describe_runtime(self) -> dict[str, object]:
        return {"adapter_loaded": False}

    def generate(self, request: BackendRequest, *, sample: dict[str, object]) -> BackendResponse:
        index = min(self.calls, len(self.outputs) - 1)
        self.calls += 1
        return BackendResponse(
            backend_name=self.backend_name,
            raw_output=self.outputs[index],
            metadata={"generation_stage": request.metadata.get("generation_stage")},
        )


def require(condition: bool, label: str, failures: list[str]) -> None:
    if condition:
        print(f"PASS: {label}")
    else:
        print(f"FAIL: {label}")
        failures.append(label)


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def score(failures: list[str]) -> str:
    if not failures:
        return "10/10"
    if len(failures) <= 2:
        return "8/10"
    if len(failures) <= 5:
        return "6/10"
    return "4/10"


def main() -> int:
    failures: list[str] = []
    baseline_text = read_text(REPO_ROOT / "src/agentiad_recon/baseline.py")
    backends_text = read_text(REPO_ROOT / "src/agentiad_recon/backends.py")
    evaluation_text = read_text(REPO_ROOT / "src/agentiad_recon/evaluation.py")
    run_definition_schema = read_text(
        REPO_ROOT / "src/agentiad_recon/contracts/schemas/baseline_run_definition.schema.json"
    )
    summary_schema = read_text(
        REPO_ROOT / "src/agentiad_recon/contracts/schemas/baseline_run_summary.schema.json"
    )
    metrics_schema = read_text(
        REPO_ROOT / "src/agentiad_recon/contracts/schemas/baseline_metrics_report.schema.json"
    )
    manifest_schema = read_text(
        REPO_ROOT / "src/agentiad_recon/contracts/schemas/artifact_manifest.schema.json"
    )
    readme_text = read_text(REPO_ROOT / "README.md").lower()
    worklog_text = read_text(
        REPO_ROOT
        / "Working Log/2026-03-20 Prompt 2.13 opt-in throughput controls timing instrumentation and progress monitoring.md"
    ).lower()

    for relative in [
        "src/agentiad_recon/baseline.py",
        "src/agentiad_recon/evaluation.py",
        "src/agentiad_recon/backends.py",
        "tests/test_prompt_2_13_throughput_and_progress.py",
        "audit/check_prompt_2_13.py",
    ]:
        require((REPO_ROOT / relative).exists(), f"required file exists: {relative}", failures)

    for phrase in [
        "generation_stage_overrides",
        "emit_baseline_compare",
        "emit_delta_report",
        "artifact_level",
        "timing_enabled",
        "progress_mode",
        "ProgressReporter",
        "core_artifacts_written_before_optional_tail_work",
    ]:
        require(phrase in baseline_text, f"baseline includes Prompt 2.13 surface: {phrase}", failures)

    require(
        "_effective_generation_config" in backends_text,
        "backends plumb effective generation config overrides",
        failures,
    )
    require(
        "timing_summary" in evaluation_text,
        "evaluation surfaces include timing_summary",
        failures,
    )

    for schema_name, schema_text in {
        "run_definition": run_definition_schema,
        "summary": summary_schema,
        "metrics": metrics_schema,
        "manifest": manifest_schema,
    }.items():
        for phrase in [
            "generation_stage_overrides",
            "emit_baseline_compare",
            "emit_delta_report",
            "artifact_level",
            "timing_enabled",
            "progress_mode",
        ]:
            require(
                phrase in schema_text,
                f"{schema_name} schema includes {phrase}",
                failures,
            )

    require("prompt 2.13" in readme_text and "progress" in readme_text, "README documents Prompt 2.13", failures)
    for phrase in [
        "what changed",
        "why it changed",
        "what effects the changes have",
        "exact modified files",
        "local validation commands actually run",
        "heavy compute statement",
        "suggested commit message",
        "manager-style summary",
    ]:
        require(phrase in worklog_text, f"working log includes: {phrase}", failures)

    suite = unittest.defaultTestLoader.discover(
        str(REPO_ROOT / "tests"),
        pattern="test_prompt_2_13_throughput_and_progress.py",
    )
    result = unittest.TextTestRunner(stream=sys.stdout, verbosity=1).run(suite)
    require(result.wasSuccessful(), "Prompt 2.13 regression tests pass", failures)

    backend = SequenceBackend([TOOL_CALL_PZ, TOOL_CALL_CR, FINAL_ANSWER])
    with tempfile.TemporaryDirectory() as tempdir, mock.patch(
        "agentiad_recon.baseline._select_backend",
        return_value=backend,
    ), mock.patch(
        "agentiad_recon.baseline.run_baseline",
        side_effect=AssertionError("tail compare work should be disabled in acceptance smoke"),
    ):
        result_payload = run_tool_augmented(
            config_path=PZ_CR_CONFIG,
            dataset_root=FIXTURE_ROOT,
            artifact_root=tempdir,
            max_samples=1,
            runtime_overrides={
                "first_turn_protocol_gate_mode": "off",
                "post_pz_second_turn_gate_mode": "off",
                "artifact_level": "throughput",
                "emit_baseline_compare": False,
                "emit_delta_report": False,
                "timing_enabled": True,
                "progress_mode": "log",
                "progress_update_every_n_samples": 1,
            },
        )
        root = Path(tempdir)
        summary = json.loads(Path(result_payload["summary_path"]).read_text(encoding="utf-8"))
        metrics = json.loads((root / "metrics" / "metrics_report.json").read_text(encoding="utf-8"))
        manifest = json.loads(Path(result_payload["run_manifest_path"]).read_text(encoding="utf-8"))
        progress_snapshot_path = Path(summary["artifact_paths"]["progress_snapshot"])
        progress_snapshot_exists = progress_snapshot_path.exists()
        prediction_path = next((root / "predictions" / "seed_0").glob("*.json"))
        prediction_record = json.loads(prediction_path.read_text(encoding="utf-8"))

    require(progress_snapshot_exists, "progress snapshot exists when progress is enabled", failures)
    require("timing_summary" in summary, "run summary exposes timing_summary", failures)
    require("timing_summary" in metrics, "metrics report exposes timing_summary", failures)
    require("timing_summary" in manifest, "manifest exposes timing_summary", failures)
    require(
        summary["core_artifacts_written_before_optional_tail_work"] is True,
        "core summary/manifest ordering flag is true",
        failures,
    )
    require(
        not Path(prediction_record["raw_output_path"]).exists(),
        "throughput mode can suppress ordinary raw-output retention",
        failures,
    )
    require(
        "delta_report" not in summary["artifact_paths"],
        "delta artifact path is skipped when emit_delta_report=false",
        failures,
    )

    print(f"Acceptance score: {score(failures)}")
    if failures:
        print("Prompt 2.13 acceptance check FAILED.")
        return 1
    print("Prompt 2.13 acceptance check PASSED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
