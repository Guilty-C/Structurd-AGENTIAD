"""Acceptance check for Prompt 2.12 bounded post-PZ second-turn CR gate."""

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
from agentiad_recon.baseline import _artifact_dirs, _runtime_config, _tool_loop_sample, load_run_definition, run_tool_augmented
from agentiad_recon.mmad import MMADIndexer
from agentiad_recon.prompting import render_answer_block


FIXTURE_ROOT = REPO_ROOT / "tests" / "fixtures" / "mmad_fixture"
PZ_CR_CONFIG = REPO_ROOT / "configs" / "tool_pz_cr_fixture.json"
TOOL_CALL_PZ = (
    '<tool_call>{"tool_name":"PZ","arguments":{"bbox":{"x0":0.10,"y0":0.10,"x1":0.80,"y1":0.80}}}</tool_call>'
)
TOOL_CALL_CR = (
    '<tool_call>{"tool_name":"CR","arguments":{"policy":"same_category_normal"}}</tool_call>'
)


def _direct_final_answer(think: str) -> str:
    return render_answer_block(
        {"anomaly_present": False, "top_anomaly": None, "visual_descriptions": []},
        wrapper_tag="answer",
        think=think,
    )


def _leaky_first_turn_final_answer() -> str:
    return _direct_final_answer(
        "The crop result is sufficient; no comparative retrieval is allowed in pz_only."
    )


class SequenceBackend(InferenceBackend):
    """Acceptance backend with deterministic scripted outputs."""

    backend_name = "prompt_2_12_acceptance_sequence_backend_v1"

    def __init__(self, outputs: list[str]) -> None:
        self.outputs = outputs
        self.calls = 0

    def describe_runtime(self) -> dict[str, object]:
        return {"adapter_loaded": False}

    def generate(self, request: BackendRequest, *, sample: dict[str, object]) -> BackendResponse:
        index = min(self.calls, len(self.outputs) - 1)
        self.calls += 1
        return BackendResponse(backend_name=self.backend_name, raw_output=self.outputs[index], metadata={})


def require(condition: bool, label: str, failures: list[str]) -> None:
    if condition:
        print(f"PASS: {label}")
    else:
        print(f"FAIL: {label}")
        failures.append(label)


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _score(failures: list[str]) -> str:
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
    behavior_audit_text = read_text(REPO_ROOT / "src/agentiad_recon/behavior_audit.py")
    evaluation_text = read_text(REPO_ROOT / "src/agentiad_recon/evaluation.py")
    prediction_schema = read_text(REPO_ROOT / "src/agentiad_recon/contracts/schemas/baseline_prediction_record.schema.json")
    metrics_schema = read_text(REPO_ROOT / "src/agentiad_recon/contracts/schemas/baseline_metrics_report.schema.json")
    summary_schema = read_text(REPO_ROOT / "src/agentiad_recon/contracts/schemas/baseline_run_summary.schema.json")
    manifest_schema = read_text(REPO_ROOT / "src/agentiad_recon/contracts/schemas/artifact_manifest.schema.json")
    readme_text = read_text(REPO_ROOT / "README.md").lower()
    worklog_text = read_text(
        REPO_ROOT
        / "Working Log/2026-03-19 Prompt 2.12 bounded post-PZ second-turn CR protocol gate under sanitized valid contract.md"
    ).lower()

    for relative in [
        "src/agentiad_recon/baseline.py",
        "src/agentiad_recon/behavior_audit.py",
        "src/agentiad_recon/evaluation.py",
        "tests/test_prompt_2_12_post_pz_second_turn_gate.py",
        "audit/check_prompt_2_12.py",
    ]:
        require((REPO_ROOT / relative).exists(), f"required file exists: {relative}", failures)

    for phrase in [
        "post_pz_second_turn_gate_mode",
        "post_pz_second_turn_protocol_gate",
        "_post_pz_second_turn_gate_retry_instruction",
        "recovered_to_cr_call",
        "called_non_cr_tool_after_retry",
        "post_pz_second_turn_gate_retry_missing_contract_block",
    ]:
        require(phrase in baseline_text, f"baseline includes Prompt 2.12 gate surface: {phrase}", failures)

    require(
        "summarize_post_pz_second_turn_gate" in behavior_audit_text,
        "behavior audit includes Prompt 2.12 gate summary helper",
        failures,
    )
    require(
        "post_pz_second_turn_gate_summary" in evaluation_text,
        "evaluation surfaces accept Prompt 2.12 gate summary",
        failures,
    )

    for schema_name, schema_text in {
        "prediction": prediction_schema,
        "metrics": metrics_schema,
        "summary": summary_schema,
        "manifest": manifest_schema,
    }.items():
        require(
            "post_pz_second_turn_gate" in schema_text,
            f"{schema_name} schema includes Prompt 2.12 fields",
            failures,
        )

    require("prompt 2.12" in readme_text and "second-turn" in readme_text, "README documents Prompt 2.12", failures)
    for phrase in [
        "what changed",
        "why",
        "effects",
    ]:
        require(phrase in worklog_text, f"working log field present: {phrase}", failures)

    suite = unittest.defaultTestLoader.discover(str(REPO_ROOT / "tests"), pattern="test_prompt_2_*.py")
    result = unittest.TextTestRunner(stream=sys.stdout, verbosity=1).run(suite)
    require(result.wasSuccessful(), "Prompt 2.x lightweight tests pass", failures)

    samples = [sample.to_dict() for sample in MMADIndexer(FIXTURE_ROOT, source="mmad_fixture").index_samples()]
    sample = samples[0]
    definition = load_run_definition(PZ_CR_CONFIG)
    runtime = _runtime_config(
        definition,
        dataset_root=FIXTURE_ROOT,
        runtime_overrides={
            "first_turn_protocol_gate_mode": "retry_once_pz_cr",
            "post_pz_second_turn_gate_mode": "retry_once_require_cr_after_pz",
        },
    )

    recovered_backend = SequenceBackend(
        [
            _leaky_first_turn_final_answer(),
            TOOL_CALL_PZ,
            _direct_final_answer("second turn direct final answer"),
            TOOL_CALL_CR,
            _direct_final_answer("answer after cr"),
        ]
    )
    with tempfile.TemporaryDirectory() as tempdir:
        directories = _artifact_dirs(definition, Path(tempdir))
        record, _trace_payload = _tool_loop_sample(
            definition=definition,
            backend=recovered_backend,
            runtime_config=runtime,
            sample=sample,
            sample_pool=samples,
            seed=0,
            directories=directories,
        )
        gate_sidecar = json.loads(Path(record["post_pz_second_turn_gate_sidecar_path"]).read_text(encoding="utf-8"))
        require(record["post_pz_second_turn_gate_triggered"], "second-turn gate triggers on direct final answer", failures)
        require(
            record["post_pz_second_turn_gate_outcome"] == "recovered_to_cr_call",
            "gate can recover to a real CR call",
            failures,
        )
        require(
            "CR" in record["called_tools"] and record["tool_call_count"] >= 2,
            "recovered gate path reaches canonical CR execution",
            failures,
        )
        require(
            Path(record["raw_output_path"]).exists() and Path(record["post_pz_second_turn_gate_retry_raw_output_path"]).exists(),
            "first-attempt and retry raw outputs are both preserved",
            failures,
        )
        require(
            gate_sidecar["final_gate_outcome"] == "recovered_to_cr_call",
            "gate sidecar records recovered_to_cr_call outcome",
            failures,
        )

    parse_failure_backend = SequenceBackend(
        [
            _leaky_first_turn_final_answer(),
            TOOL_CALL_PZ,
            _direct_final_answer("second turn direct final answer"),
            "I will answer directly without a contract block.",
        ]
    )
    with tempfile.TemporaryDirectory() as tempdir:
        directories = _artifact_dirs(definition, Path(tempdir))
        record, _trace_payload = _tool_loop_sample(
            definition=definition,
            backend=parse_failure_backend,
            runtime_config=runtime,
            sample=sample,
            sample_pool=samples,
            seed=0,
            directories=directories,
        )
        require(
            record["post_pz_second_turn_gate_outcome"] == "retry_parse_failure",
            "retry parse failure is classified explicitly",
            failures,
        )
        require(
            bool(record["post_pz_second_turn_gate_retry_failure_reason"]),
            "retry parse failure records explicit failure reason",
            failures,
        )

    summary_backend = SequenceBackend(
        [
            _leaky_first_turn_final_answer(),
            TOOL_CALL_PZ,
            _direct_final_answer("second turn direct final answer"),
            TOOL_CALL_CR,
            _direct_final_answer("answer after cr"),
        ]
    )
    with tempfile.TemporaryDirectory() as tempdir, mock.patch(
        "agentiad_recon.baseline._select_backend",
        return_value=summary_backend,
    ):
        result = run_tool_augmented(
            config_path=PZ_CR_CONFIG,
            dataset_root=FIXTURE_ROOT,
            artifact_root=tempdir,
            max_samples=1,
            runtime_overrides={
                "first_turn_protocol_gate_mode": "retry_once_pz_cr",
                "post_pz_second_turn_gate_mode": "retry_once_require_cr_after_pz",
            },
        )
        summary = json.loads(Path(result["summary_path"]).read_text(encoding="utf-8"))
        manifest = json.loads(Path(result["run_manifest_path"]).read_text(encoding="utf-8"))
        metrics = result["metrics_report"]
        for payload in (summary, manifest, metrics):
            gate_summary = payload["post_pz_second_turn_gate_summary"]
            require(
                gate_summary["post_pz_second_turn_gate_trigger_count"] == 1,
                "gate trigger count is aggregated",
                failures,
            )
            require(
                gate_summary["post_pz_second_turn_gate_recovered_to_cr_call_count"] == 1,
                "gate recovered-to-CR count is aggregated",
                failures,
            )
            require(
                gate_summary["failed_count_with_missing_reason_count"] == 0,
                "gate summary preserves missing-reason count at zero",
                failures,
            )

    score = _score(failures)
    print(f"\nAcceptance score: {score}")
    if failures:
        print("Prompt 2.12 acceptance check FAILED.")
        for failure in failures:
            print(f"- {failure}")
        return 1

    print("Prompt 2.12 acceptance check PASSED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
