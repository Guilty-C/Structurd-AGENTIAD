"""Acceptance check for Prompt 2.11 post-PZ second-turn sanitation."""

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

    backend_name = "prompt_2_11_acceptance_sequence_backend_v1"

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
        REPO_ROOT / "Working Log/2026-03-19 Prompt 2.11 post-PZ second-turn context sanitation and leakage removal.md"
    ).lower()

    for relative in [
        "src/agentiad_recon/baseline.py",
        "src/agentiad_recon/behavior_audit.py",
        "src/agentiad_recon/evaluation.py",
        "tests/test_prompt_2_11_post_pz_sanitation.py",
        "audit/check_prompt_2_11.py",
    ]:
        require((REPO_ROOT / relative).exists(), f"required file exists: {relative}", failures)

    for phrase in [
        "sanitation_applied",
        "removed_obsolete_terminal_answer_count",
        "removed_pz_only_leakage_message_count",
        "pre_sanitation_rendered_prompt_surface_digest",
        "post_sanitation_rendered_prompt_surface_digest",
        "post_sanitation_transition_contract_valid_for_cr",
    ]:
        require(phrase in baseline_text, f"baseline includes Prompt 2.11 audit field: {phrase}", failures)

    require(
        "summarize_post_pz_transition_sanitation" in behavior_audit_text,
        "behavior audit includes Prompt 2.11 sanitation summary helper",
        failures,
    )
    require(
        "post_pz_transition_sanitation_summary" in evaluation_text,
        "evaluation surfaces accept Prompt 2.11 sanitation summary",
        failures,
    )

    for schema_name, schema_text in {
        "prediction": prediction_schema,
        "metrics": metrics_schema,
        "summary": summary_schema,
        "manifest": manifest_schema,
    }.items():
        require(
            "post_pz_transition_sanitation" in schema_text,
            f"{schema_name} schema includes Prompt 2.11 fields",
            failures,
        )

    require("prompt 2.11" in readme_text and "sanitation" in readme_text, "README documents Prompt 2.11", failures)
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
        runtime_overrides={"first_turn_protocol_gate_mode": "retry_once_pz_cr"},
    )

    direct_final_backend = SequenceBackend(
        [_leaky_first_turn_final_answer(), TOOL_CALL_PZ, _direct_final_answer("second turn direct final answer")]
    )
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
        require(record["post_pz_transition_sanitation_applied"], "post-PZ sanitation is applied", failures)
        require(
            record["post_pz_transition_removed_obsolete_terminal_answer_count"] > 0,
            "obsolete terminal answer count is recorded",
            failures,
        )
        require(
            record["post_pz_transition_post_sanitation_contract_valid_for_cr"],
            "post-sanitized CR contract becomes valid",
            failures,
        )
        require(
            not record["post_pz_transition_post_sanitation_pz_only_leakage_present"],
            "post-sanitized surface removes pz_only leakage",
            failures,
        )
        require(sidecar["sanitation_applied"], "sidecar records sanitation_applied", failures)
        require(
            sidecar["post_sanitation_transition_contract_valid_for_cr"],
            "sidecar records valid post-sanitized CR contract",
            failures,
        )

    cr_backend = SequenceBackend(
        [_leaky_first_turn_final_answer(), TOOL_CALL_PZ, TOOL_CALL_CR, _direct_final_answer("answer after cr")]
    )
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
            "second-turn CR call remains visible after sanitation",
            failures,
        )
        for payload in (summary, manifest, metrics):
            sanitation = payload["post_pz_transition_sanitation_summary"]
            require(
                sanitation["post_pz_transition_sanitation_event_count"] == 1,
                "sanitation event count is aggregated",
                failures,
            )
            require(
                sanitation["post_pz_transition_sanitation_applied_count"] == 1,
                "sanitation applied count is aggregated",
                failures,
            )
            require(
                sanitation["post_pz_transition_post_sanitation_contract_valid_count"] == 1,
                "post-sanitized valid-contract count is aggregated",
                failures,
            )
            require(
                sanitation["post_pz_transition_post_sanitation_pz_only_leakage_count"] == 0,
                "post-sanitized pz_only leakage count is zero",
                failures,
            )

    score = _score(failures)
    print(f"\nAcceptance score: {score}")
    if failures:
        print("Prompt 2.11 acceptance check FAILED.")
        for failure in failures:
            print(f"- {failure}")
        return 1

    print("Prompt 2.11 acceptance check PASSED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
