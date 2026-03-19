"""Acceptance check for Prompt 2.10 post-PZ second-turn CR transition audit."""

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


class SequenceBackend(InferenceBackend):
    """Acceptance backend with deterministic scripted outputs."""

    backend_name = "prompt_2_10_acceptance_sequence_backend_v1"

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
        REPO_ROOT / "Working Log/2026-03-19 Prompt 2.10 post-PZ second-turn CR transition audit under valid gate-on runtime.md"
    ).lower()

    for relative in [
        "src/agentiad_recon/baseline.py",
        "src/agentiad_recon/behavior_audit.py",
        "src/agentiad_recon/evaluation.py",
        "tests/test_prompt_2_10_post_pz_transition_audit.py",
        "audit/check_prompt_2_10.py",
    ]:
        require((REPO_ROOT / relative).exists(), f"required file exists: {relative}", failures)

    for phrase in [
        "post_pz_transition_missing_reinserted_pz_result",
        "post_pz_transition_missing_cr_tool",
        "post_pz_transition_missing_query_image_instruction",
        "post_pz_transition_pz_only_leakage",
        "post_pz_transition_missing_tool_context",
    ]:
        require(
            phrase in baseline_text or phrase in behavior_audit_text,
            f"post-PZ mismatch taxonomy present: {phrase}",
            failures,
        )

    require("post_pz_transition.turn_" in baseline_text, "post-PZ transition sidecar path is implemented", failures)
    require("post_pz_transition_summary" in behavior_audit_text, "post-PZ transition summary helper is implemented", failures)
    require("post_pz_transition_summary" in evaluation_text, "evaluation surfaces accept post-PZ transition summary", failures)

    for schema_name, schema_text in {
        "prediction": prediction_schema,
        "metrics": metrics_schema,
        "summary": summary_schema,
        "manifest": manifest_schema,
    }.items():
        require("post_pz_transition" in schema_text, f"{schema_name} schema includes Prompt 2.10 fields", failures)

    require("prompt 2.10" in readme_text and "post-pz" in readme_text, "README documents Prompt 2.10 audit", failures)
    for phrase in [
        "what changed",
        "why it changed",
        "effects",
        "git-diff-style summary",
        "does not force cr",
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

    direct_final_backend = SequenceBackend([TOOL_CALL_PZ, _direct_final_answer("second turn direct final answer")])
    with tempfile.TemporaryDirectory() as tempdir:
        directories = _artifact_dirs(definition, Path(tempdir))
        record, _trace_payload = _tool_loop_sample(
            definition=definition,
            backend=direct_final_backend,
            runtime_config=runtime,
            sample=sample,
            sample_pool=samples,
            seed=0,
            directories=directories,
        )
        sidecar = json.loads(Path(record["post_pz_transition_sidecar_path"]).read_text(encoding="utf-8"))
        require(record["first_successful_tool_name"] == "PZ", "first successful tool is recorded as PZ", failures)
        require(record["post_pz_transition_audited"], "post-PZ transition is audited", failures)
        require(record["post_pz_transition_contract_valid_for_cr"], "valid post-PZ CR contract is recognized", failures)
        require(record["post_pz_second_turn_direct_final_without_cr"], "direct final answer without CR is recorded", failures)
        require(sidecar["cr_available_in_prompt_surface"], "sidecar preserves CR availability audit", failures)
        require(sidecar["pz_result_reinserted_present"], "sidecar preserves reinserted PZ evidence", failures)

    cr_backend = SequenceBackend([TOOL_CALL_PZ, TOOL_CALL_CR, _direct_final_answer("answer after cr")])
    with tempfile.TemporaryDirectory() as tempdir, mock.patch(
        "agentiad_recon.baseline._select_backend",
        return_value=cr_backend,
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
        require(
            result["prediction_records"][0]["post_pz_second_turn_called_cr"],
            "second-turn CR call is recorded at prediction level",
            failures,
        )
        for payload in (summary, manifest, metrics):
            transition = payload["post_pz_transition_summary"]
            require(transition["post_pz_transition_event_count"] == 1, "post-PZ event count is aggregated", failures)
            require(transition["post_pz_transition_contract_valid_count"] == 1, "valid contract count is aggregated", failures)
            require(transition["post_pz_second_turn_called_cr_count"] == 1, "called-CR count is aggregated", failures)
            require(
                transition["post_pz_second_turn_direct_final_without_cr_count"] == 0,
                "direct-final-without-CR count stays zero for CR path",
                failures,
            )

    score = _score(failures)
    print(f"\nAcceptance score: {score}")
    if failures:
        print("Prompt 2.10 acceptance check FAILED.")
        for failure in failures:
            print(f"- {failure}")
        return 1

    print("Prompt 2.10 acceptance check PASSED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
