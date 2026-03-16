"""Lightweight acceptance check for Prompt 2.5 zero-tool behavior auditing."""

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
from agentiad_recon.behavior_audit import audit_train_side_dataset
from agentiad_recon.prompting import render_answer_block
from agentiad_recon.sft import export_sft_dataset


FIXTURE_ROOT = REPO_ROOT / "tests" / "fixtures" / "mmad_fixture"
PZ_CR_CONFIG = REPO_ROOT / "configs" / "tool_pz_cr_fixture.json"
EXPORT_CONFIG = REPO_ROOT / "configs" / "sft_export_fixture.json"


class DirectFinalAnswerBackend(InferenceBackend):
    """Tiny backend that collapses directly to a final answer for fixture auditing."""

    backend_name = "acceptance_direct_final_answer_backend_v1"

    def describe_runtime(self) -> dict[str, object]:
        return {"adapter_loaded": False}

    def generate(self, request: BackendRequest, *, sample: dict[str, object]) -> BackendResponse:
        return BackendResponse(
            backend_name=self.backend_name,
            raw_output=render_answer_block(
                {
                    "anomaly_present": False,
                    "top_anomaly": None,
                    "visual_descriptions": [],
                },
                wrapper_tag="answer",
                think="Acceptance zero-tool collapse fixture.",
            ),
            metadata={},
        )


def require(condition: bool, label: str, failures: list[str]) -> None:
    """Print one acceptance result and collect failures."""

    if condition:
        print(f"PASS: {label}")
    else:
        print(f"FAIL: {label}")
        failures.append(label)


def read_text(path: Path) -> str:
    """Read one UTF-8 file."""

    return path.read_text(encoding="utf-8")


def main() -> int:
    """Run Prompt 2.5 acceptance checks without heavy compute."""

    failures: list[str] = []
    baseline_text = read_text(REPO_ROOT / "src/agentiad_recon/baseline.py")
    behavior_text = read_text(REPO_ROOT / "src/agentiad_recon/behavior_audit.py")
    prediction_schema = read_text(REPO_ROOT / "src/agentiad_recon/contracts/schemas/baseline_prediction_record.schema.json")
    metrics_schema = read_text(REPO_ROOT / "src/agentiad_recon/contracts/schemas/baseline_metrics_report.schema.json")
    summary_schema = read_text(REPO_ROOT / "src/agentiad_recon/contracts/schemas/baseline_run_summary.schema.json")
    manifest_schema = read_text(REPO_ROOT / "src/agentiad_recon/contracts/schemas/artifact_manifest.schema.json")
    readme_text = read_text(REPO_ROOT / "README.md").lower()
    worklog_text = read_text(
        REPO_ROOT / "Working Log/2026-03-16 Prompt 2.5 zero-tool behavior audit under valid pz_cr contract.md"
    ).lower()

    for relative in [
        "src/agentiad_recon/baseline.py",
        "src/agentiad_recon/behavior_audit.py",
        "src/agentiad_recon/contracts/schemas/baseline_prediction_record.schema.json",
        "src/agentiad_recon/contracts/schemas/baseline_metrics_report.schema.json",
        "src/agentiad_recon/contracts/schemas/baseline_run_summary.schema.json",
        "src/agentiad_recon/contracts/schemas/artifact_manifest.schema.json",
        "tests/test_prompt_2_5_zero_tool_behavior_audit.py",
        "audit/check_prompt_2_5.py",
    ]:
        require((REPO_ROOT / relative).exists(), f"required file exists: {relative}", failures)

    for phrase in [
        "build_zero_tool_behavior_fields",
        "summarize_zero_tool_behavior",
        "write_zero_tool_behavior_sidecars",
        "audit_train_side_dataset",
    ]:
        require(phrase in behavior_text or phrase in baseline_text, f"Prompt 2.5 hook present: {phrase}", failures)

    for phrase in [
        "first_protocol_event_type",
        "terminal_without_tool_call",
        "terminal_false_null_without_tool_call",
        "zero_tool_behavior_summary",
    ]:
        require(
            phrase in prediction_schema or phrase in metrics_schema or phrase in summary_schema or phrase in manifest_schema,
            f"schema field present: {phrase}",
            failures,
        )

    require(
        "zero-tool behavior audit fields" in readme_text and "train-side pz_cr supervision audit" in readme_text,
        "README documents Prompt 2.5 behavior audit surfaces",
        failures,
    )

    for phrase in [
        "timestamp",
        "prompt number",
        "what changed",
        "why it changed",
        "effects",
        "git-diff-style summary",
        "prompt 2.4 smoke already proved the `pz_cr` prompt surface is valid".lower(),
        "toolcall_rate` is still `0.0".lower(),
        "this patch does not claim to fix the model".lower(),
    ]:
        require(phrase in worklog_text, f"working log field present: {phrase}", failures)

    suite = unittest.defaultTestLoader.discover(
        str(REPO_ROOT / "tests"),
        pattern="test_prompt_2_*.py",
    )
    result = unittest.TextTestRunner(stream=sys.stdout, verbosity=1).run(suite)
    require(result.wasSuccessful(), "Prompt 2.x lightweight tests pass", failures)

    with tempfile.TemporaryDirectory() as tempdir, mock.patch(
        "agentiad_recon.baseline._select_backend",
        return_value=DirectFinalAnswerBackend(),
    ):
        result = run_tool_augmented(
            config_path=PZ_CR_CONFIG,
            dataset_root=FIXTURE_ROOT,
            artifact_root=tempdir,
            max_samples=1,
        )
        summary = json.loads(Path(result["summary_path"]).read_text(encoding="utf-8"))
        manifest = json.loads(Path(result["run_manifest_path"]).read_text(encoding="utf-8"))
        metrics = result["metrics_report"]

        per_dataset = Path(tempdir) / "metrics" / "per_dataset_zero_tool_behavior.json"
        per_category = Path(tempdir) / "metrics" / "per_category_zero_tool_behavior.json"
        require(per_dataset.exists(), "per_dataset_zero_tool_behavior.json exists", failures)
        require(per_category.exists(), "per_category_zero_tool_behavior.json exists", failures)
        for payload in (summary, manifest, metrics):
            require("zero_tool_behavior_summary" in payload, "zero_tool_behavior_summary exported", failures)
            require(
                payload["zero_tool_behavior_summary"]["failed_count_with_missing_reason_count"] == 0,
                "failed_count_with_missing_reason_count is zero",
                failures,
            )

    with tempfile.TemporaryDirectory() as tempdir:
        records, _metadata = export_sft_dataset(
            config_path=EXPORT_CONFIG,
            dataset_root=FIXTURE_ROOT,
            output_root=tempdir,
            max_samples_per_mode=1,
        )
        dataset_path = Path(tempdir) / "train_side_audit_fixture.jsonl"
        dataset_path.write_text(
            "\n".join(json.dumps(record, sort_keys=True) for record in records) + "\n",
            encoding="utf-8",
        )
        train_audit = audit_train_side_dataset(dataset_path, dataset_format="canonical")
        require(train_audit["pz_cr_record_count"] > 0, "train-side audit output exists", failures)

    if failures:
        print("\nPrompt 2.5 acceptance check FAILED.")
        for failure in failures:
            print(f"- {failure}")
        return 1

    print("\nPrompt 2.5 acceptance check PASSED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
