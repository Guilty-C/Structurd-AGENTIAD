"""Acceptance check for Prompt 2.9 bounded retry repair expansion and taxonomy."""

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
from agentiad_recon.tooling import RETRY_REPAIR_ORIGINAL_FAILURE_FAMILY, ToolContractError, parse_tool_call, repair_retry_tool_call_output


FIXTURE_ROOT = REPO_ROOT / "tests" / "fixtures" / "mmad_fixture"
PZ_CR_CONFIG = REPO_ROOT / "configs" / "tool_pz_cr_fixture.json"


def _direct_final_answer(think: str) -> str:
    return render_answer_block(
        {"anomaly_present": False, "top_anomaly": None, "visual_descriptions": []},
        wrapper_tag="answer",
        think=think,
    )


WRAPPED_SMART_QUOTES = (
    '<tool_call>{“tool_name”:“PZ”,“arguments”:{“bbox”:{“x0”:0.10,“y0:0.10,“x1”:0.80,“y1”:0.80}}}</tool_call>'
)
BARE_ALIAS_JSON = '{"tool_name":"pz_cr","arguments":{"bbox":[0.10,0.10,0.80,0.80]}}'
DUPLICATE_BARE_PZ_JSON = (
    '<think>{"tool_name":"PZ","arguments":{"bbox":{"x0":0.10,"y0":0.10,"x1":0.80,"y1":0.80}}}</think>\n'
    '{"tool_name":"PZ","arguments":{"bbox":{"x0":0.10,"y0":0.10,"x1":0.80,"y1":0.80}}}'
)
PSEUDO_TOOL_JSON = '{"tool_name":"pz_cr","arguments":{"crop_result":{"status":"ok"},"query_result":{"status":"none"}}}'
AMBIGUOUS_CANDIDATES = (
    '{"tool_name":"PZ","arguments":{"bbox":{"x0":0.10,"y0":0.10,"x1":0.80,"y1":0.80}}}\n'
    '{"tool_name":"PZ","arguments":{"bbox":{"x0":0.20,"y0":0.20,"x1":0.70,"y1":0.70}}}'
)


class SequenceBackend(InferenceBackend):
    """Acceptance backend with deterministic scripted outputs."""

    backend_name = "prompt_2_9_acceptance_sequence_backend_v1"

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
        REPO_ROOT / "Working Log/2026-03-19 Prompt 2.9 bounded retry-family expansion and unrecoverable taxonomy.md"
    ).lower()

    for relative in [
        "src/agentiad_recon/baseline.py",
        "src/agentiad_recon/tooling.py",
        "src/agentiad_recon/evaluation.py",
        "tests/test_prompt_2_9_retry_repair_taxonomy.py",
        "audit/check_prompt_2_9.py",
    ]:
        require((REPO_ROOT / relative).exists(), f"required file exists: {relative}", failures)

    for phrase in [
        "retry_repair_unrecoverable_pseudo_observation_payload",
        "retry_repair_ambiguous_multiple_candidates",
        "retry_repair_bbox_not_losslessly_recoverable",
        "retry_repair_unsupported_payload_shape",
        "retry_repair_no_unique_candidate",
    ]:
        require(phrase in tooling_text or phrase in baseline_text, f"taxonomy surface present: {phrase}", failures)

    require("duplicate_candidate_deduplication" in tooling_text, "duplicate-candidate repair category is implemented", failures)
    require("first_turn_gate_repair_failure_families" in baseline_text, "baseline aggregates repair failure families", failures)
    require("first_turn_gate_repair_failure_family" in evaluation_text, "prediction record carries repair failure family", failures)
    require("Retry repair must not fabricate CR calls" in tooling_text, "repair still forbids fabricated CR", failures)

    for schema_name, schema_text in {
        "prediction": prediction_schema,
        "metrics": metrics_schema,
        "summary": summary_schema,
        "manifest": manifest_schema,
    }.items():
        require("first_turn_gate_repair_failure_family" in schema_text or "first_turn_gate_repair_failure_families" in schema_text, f"{schema_name} schema includes Prompt 2.9 repair-family fields", failures)

    require("prompt 2.9" in readme_text and "pseudo-observation" in readme_text, "README documents Prompt 2.9 taxonomy", failures)
    for phrase in [
        "what changed",
        "why it changed",
        "effects",
        "git-diff-style summary",
        "pseudo-observation",
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

    duplicate_decision = repair_retry_tool_call_output(DUPLICATE_BARE_PZ_JSON, tool_path="pz_cr")
    require(duplicate_decision.succeeded, "duplicate candidate recovery succeeds", failures)
    require(
        "duplicate_candidate_deduplication" in duplicate_decision.repair_categories,
        "duplicate candidate recovery is categorized",
        failures,
    )

    wrapped_decision = repair_retry_tool_call_output(WRAPPED_SMART_QUOTES, tool_path="pz_cr")
    require(wrapped_decision.succeeded, "quote-corrupted wrapped bbox candidate repairs successfully", failures)
    require(wrapped_decision.quote_normalization_applied, "quote normalization is tracked", failures)

    alias_decision = repair_retry_tool_call_output(BARE_ALIAS_JSON, tool_path="pz_cr")
    require(alias_decision.succeeded, "alias plus bbox-list candidate repairs successfully", failures)
    require(alias_decision.alias_canonicalization_applied, "alias canonicalization is tracked", failures)
    require(alias_decision.bbox_canonicalization_applied, "bbox canonicalization is tracked", failures)

    pseudo_decision = repair_retry_tool_call_output(PSEUDO_TOOL_JSON, tool_path="pz_cr")
    require(not pseudo_decision.succeeded, "pseudo-observation payload remains unrecoverable", failures)
    require(
        pseudo_decision.failure_family == "retry_repair_unrecoverable_pseudo_observation_payload",
        "pseudo-observation payload receives explicit failure family",
        failures,
    )

    ambiguous_decision = repair_retry_tool_call_output(AMBIGUOUS_CANDIDATES, tool_path="pz_cr")
    require(not ambiguous_decision.succeeded, "ambiguous multiple candidates remain unrecoverable", failures)
    require(
        ambiguous_decision.failure_family == "retry_repair_ambiguous_multiple_candidates",
        "ambiguous multiple candidates receive explicit failure family",
        failures,
    )

    try:
        parse_tool_call(BARE_ALIAS_JSON, tool_path="pz_cr")
    except ToolContractError:
        print("PASS: strict non-retry parser behavior is preserved")
    else:
        print("FAIL: strict non-retry parser behavior is preserved")
        failures.append("strict non-retry parser behavior is preserved")

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
        sidecar = json.loads(Path(record["first_turn_gate_sidecar_path"]).read_text(encoding="utf-8"))
        require(
            record["first_turn_gate_repair_original_failure_family"] == RETRY_REPAIR_ORIGINAL_FAILURE_FAMILY,
            "prediction record preserves original retry parse failure family",
            failures,
        )
        require(
            record["first_turn_gate_repair_failure_family"] == "retry_repair_unrecoverable_pseudo_observation_payload",
            "prediction record preserves explicit repair failure family",
            failures,
        )
        require(
            record["failure_reason"] == "runtime_exception:retry_repair_unrecoverable_pseudo_observation_payload",
            "failure_reason uses explicit Prompt 2.9 repair family",
            failures,
        )
        require(
            sidecar["retry_repair"]["original_text"] == PSEUDO_TOOL_JSON,
            "sidecar preserves original retry text",
            failures,
        )
        require(
            sidecar["retry_repair"]["repaired_text"] is None,
            "sidecar only stores repaired canonical text on success",
            failures,
        )

    run_backend = SequenceBackend(
        [
            _direct_final_answer("turn 0 direct final answer"),
            PSEUDO_TOOL_JSON,
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
        family_artifact = json.loads(
            Path(summary["artifact_paths"]["first_turn_gate_repair_failure_families"]).read_text(
                encoding="utf-8"
            )
        )

        for payload in (summary, manifest, metrics):
            repair = payload["first_turn_gate_repair_summary"]
            require(repair["first_turn_gate_repair_attempt_count"] == 1, "repair attempt count is aggregated", failures)
            require(repair["first_turn_gate_repair_success_count"] == 0, "repair success count is aggregated", failures)
            require(repair["first_turn_gate_repair_failure_count"] == 1, "repair failure count is aggregated", failures)
            require(
                repair["first_turn_gate_repair_failure_families"]["retry_repair_unrecoverable_pseudo_observation_payload"] == 1,
                "repair failure family count is aggregated",
                failures,
            )
            require(repair["failed_count_with_missing_reason_count"] == 0, "repair summary keeps zero missing failure reasons", failures)

        require(
            family_artifact["families"]["retry_repair_unrecoverable_pseudo_observation_payload"]["count"] == 1,
            "repair failure family artifact records counts",
            failures,
        )

    score = _score(failures)
    print(f"\nAcceptance score: {score}")
    if failures:
        print("Prompt 2.9 acceptance check FAILED.")
        for failure in failures:
            print(f"- {failure}")
        return 1

    print("Prompt 2.9 acceptance check PASSED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
