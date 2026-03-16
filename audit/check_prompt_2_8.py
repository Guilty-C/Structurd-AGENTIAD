"""Acceptance check for Prompt 2.8 retry-only repair of malformed PZ-intent outputs."""

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
from agentiad_recon.tooling import ToolContractError, parse_tool_call, repair_retry_tool_call_output


FIXTURE_ROOT = REPO_ROOT / "tests" / "fixtures" / "mmad_fixture"
PZ_CR_CONFIG = REPO_ROOT / "configs" / "tool_pz_cr_fixture.json"


def _direct_final_answer(think: str) -> str:
    return render_answer_block(
        {"anomaly_present": False, "top_anomaly": None, "visual_descriptions": []},
        wrapper_tag="answer",
        think=think,
    )


BARE_ALIAS_JSON = '{"tool_name":"pz_cr","arguments":{"bbox":[0.10,0.10,0.80,0.80]}}'
PSEUDO_TOOL_JSON = '{"tool_name":"pz_cr","arguments":{"crop_result":{"status":"ok"},"query_result":{"status":"none"}}}'


class SequenceBackend(InferenceBackend):
    """Acceptance backend with deterministic scripted outputs."""

    backend_name = "prompt_2_8_acceptance_sequence_backend_v1"

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
    tooling_text = read_text(REPO_ROOT / "src/agentiad_recon/tooling.py")
    evaluation_text = read_text(REPO_ROOT / "src/agentiad_recon/evaluation.py")
    prediction_schema = read_text(REPO_ROOT / "src/agentiad_recon/contracts/schemas/baseline_prediction_record.schema.json")
    metrics_schema = read_text(REPO_ROOT / "src/agentiad_recon/contracts/schemas/baseline_metrics_report.schema.json")
    summary_schema = read_text(REPO_ROOT / "src/agentiad_recon/contracts/schemas/baseline_run_summary.schema.json")
    manifest_schema = read_text(REPO_ROOT / "src/agentiad_recon/contracts/schemas/artifact_manifest.schema.json")
    readme_text = read_text(REPO_ROOT / "README.md").lower()
    worklog_text = read_text(
        REPO_ROOT / "Working Log/2026-03-16 Prompt 2.8 first-turn gate retry tool-call repair for recoverable PZ-intent malformed outputs.md"
    ).lower()

    for relative in [
        "src/agentiad_recon/baseline.py",
        "src/agentiad_recon/tooling.py",
        "src/agentiad_recon/evaluation.py",
        "tests/test_prompt_2_8_retry_repair.py",
        "audit/check_prompt_2_8.py",
    ]:
        require((REPO_ROOT / relative).exists(), f"required file exists: {relative}", failures)

    for phrase in [
        "repair_retry_tool_call_output",
        "first_turn_gate_repair_summary",
        "first_turn_gate_repair_attempted",
    ]:
        require(phrase in baseline_text or phrase in tooling_text or phrase in evaluation_text, f"repair hook present: {phrase}", failures)

    require("repair_retry_tool_call_output" in tooling_text, "repair helper is implemented in tooling.py", failures)
    require("first_turn_gate_retry_missing_contract_block" in tooling_text, "repair path targets the observed retry failure family", failures)
    require("parse_tool_call(" in tooling_text, "strict parser still exists", failures)
    require("Retry repair must not fabricate CR calls" in tooling_text, "repair explicitly forbids fabricated CR", failures)

    for schema_name, schema_text in {
        "prediction": prediction_schema,
        "metrics": metrics_schema,
        "summary": summary_schema,
        "manifest": manifest_schema,
    }.items():
        require("first_turn_gate_repair" in schema_text, f"{schema_name} schema includes repair contract fields", failures)

    require("retry-only repair" in readme_text and "first-turn gate" in readme_text, "README documents Prompt 2.8 repair lane", failures)
    for phrase in [
        "what changed",
        "why it changed",
        "effects",
        "git-diff-style summary",
        "recoverable malformed pz-intent",
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
        runtime_overrides={"first_turn_protocol_gate_mode": "retry_once_pz_cr"},
    )

    decision = repair_retry_tool_call_output(BARE_ALIAS_JSON, tool_path="pz_cr")
    require(decision.succeeded, "recoverable malformed PZ-intent sample repairs successfully", failures)
    require("wrapper_recovery" in decision.repair_categories, "wrapper recovery is tracked", failures)
    require("alias_canonicalization" in decision.repair_categories, "alias canonicalization is tracked", failures)
    require("bbox_canonicalization" in decision.repair_categories, "bbox canonicalization is tracked", failures)

    failed_decision = repair_retry_tool_call_output(PSEUDO_TOOL_JSON, tool_path="pz_cr")
    require(not failed_decision.succeeded, "unrecoverable pseudo-tool payload still fails", failures)
    require("explicit bbox payload" in str(failed_decision.error), "unrecoverable case keeps concrete error detail", failures)

    try:
        parse_tool_call(BARE_ALIAS_JSON, tool_path="pz_cr")
    except ToolContractError:
        print("PASS: strict non-retry parser behavior is preserved")
    else:
        print("FAIL: strict non-retry parser behavior is preserved")
        failures.append("strict non-retry parser behavior is preserved")

    recovery_backend = SequenceBackend(
        [
            _direct_final_answer("turn 0 direct final answer"),
            BARE_ALIAS_JSON,
            _direct_final_answer("post-tool final answer"),
        ]
    )
    with tempfile.TemporaryDirectory() as tempdir:
        directories = _artifact_dirs(definition, Path(tempdir))
        record, _trace_payload = _tool_loop_sample(
            definition=definition,
            backend=recovery_backend,
            runtime_config=runtime,
            sample=sample,
            sample_pool=samples,
            seed=0,
            directories=directories,
        )
        sidecar = json.loads(Path(record["first_turn_gate_sidecar_path"]).read_text(encoding="utf-8"))
        require(record["first_turn_gate_repair_attempted"], "repair attempt is recorded in prediction record", failures)
        require(record["first_turn_gate_repair_succeeded"], "repair success is recorded in prediction record", failures)
        require(sidecar["retry_repair"]["original_text"] == BARE_ALIAS_JSON, "sidecar preserves original retry text", failures)
        require("<tool_call>" in sidecar["retry_repair"]["repaired_text"], "sidecar preserves repaired retry text", failures)
        require(record["called_tools"] == ["PZ"], "repair path recovers a real PZ call", failures)

    unrecoverable_backend = SequenceBackend(
        [
            _direct_final_answer("turn 0 direct final answer"),
            PSEUDO_TOOL_JSON,
        ]
    )
    with tempfile.TemporaryDirectory() as tempdir:
        directories = _artifact_dirs(definition, Path(tempdir))
        record, _trace_payload = _tool_loop_sample(
            definition=definition,
            backend=unrecoverable_backend,
            runtime_config=runtime,
            sample=sample,
            sample_pool=samples,
            seed=0,
            directories=directories,
        )
        require(record["failure_reason"] == "runtime_exception:first_turn_gate_retry_repair_failed", "unrecoverable repair failure keeps concrete failure_reason", failures)
        require(record["tool_call_count"] == 0, "unrecoverable repair path does not fabricate tool calls", failures)

    run_backend = SequenceBackend(
        [
            _direct_final_answer("turn 0 direct final answer"),
            BARE_ALIAS_JSON,
            _direct_final_answer("post-tool final answer"),
        ]
    )
    with tempfile.TemporaryDirectory() as tempdir, mock.patch(
        "agentiad_recon.baseline._select_backend",
        return_value=run_backend,
    ):
        result = run_tool_augmented(
            config_path=PZ_CR_CONFIG,
            dataset_root=FIXTURE_ROOT,
            artifact_root=tempdir,
            max_samples=1,
            runtime_overrides={"first_turn_protocol_gate_mode": "retry_once_pz_cr"},
        )
        summary = json.loads(Path(result["summary_path"]).read_text(encoding="utf-8"))
        manifest = json.loads(Path(result["run_manifest_path"]).read_text(encoding="utf-8"))
        metrics = result["metrics_report"]
        for payload in (summary, manifest, metrics):
            repair = payload["first_turn_gate_repair_summary"]
            require(repair["first_turn_gate_repair_attempt_count"] == 1, "repair attempt count is aggregated", failures)
            require(repair["first_turn_gate_repair_success_count"] == 1, "repair success count is aggregated", failures)
            require(repair["first_turn_gate_repair_failure_count"] == 0, "repair failure count is aggregated", failures)

    score = _score(failures)
    print(f"\nAcceptance score: {score}")
    if failures:
        print("Prompt 2.8 acceptance check FAILED.")
        for failure in failures:
            print(f"- {failure}")
        return 1

    print("Prompt 2.8 acceptance check PASSED.")
    return 0
if __name__ == "__main__":
    raise SystemExit(main())
