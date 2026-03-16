"""Acceptance check for Prompt 2.7 first-turn protocol gate and single retry."""

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
from agentiad_recon.prompting import PromptBundle, build_prompt, render_answer_block


FIXTURE_ROOT = REPO_ROOT / "tests" / "fixtures" / "mmad_fixture"
PZ_CR_CONFIG = REPO_ROOT / "configs" / "tool_pz_cr_fixture.json"
PZ_ONLY_CONFIG = REPO_ROOT / "configs" / "tool_pz_only_fixture.json"


def _direct_final_answer(think: str) -> str:
    return render_answer_block(
        {"anomaly_present": False, "top_anomaly": None, "visual_descriptions": []},
        wrapper_tag="answer",
        think=think,
    )


TOOL_CALL_PZ = (
    '<tool_call>{"tool_name":"PZ","arguments":{"bbox":{"x0":0.10,"y0":0.10,"x1":0.80,"y1":0.80}}}</tool_call>'
)


class SequenceBackend(InferenceBackend):
    """Acceptance backend with a deterministic response sequence."""

    backend_name = "prompt_2_7_acceptance_sequence_backend_v1"

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
    """Print one acceptance result and collect failures."""

    if condition:
        print(f"PASS: {label}")
    else:
        print(f"FAIL: {label}")
        failures.append(label)


def read_text(path: Path) -> str:
    """Read one UTF-8 file."""

    return path.read_text(encoding="utf-8")


def _leaked_pz_only_prompt(sample: dict[str, object]) -> PromptBundle:
    leaked = build_prompt(sample, tool_path="pz_only")
    return PromptBundle(
        prompt_version=leaked.prompt_version,
        tool_path="pz_cr",
        messages=leaked.messages,
        stop_sequences=leaked.stop_sequences,
    )


def _score(failures: list[str]) -> str:
    if not failures:
        return "10/10"
    if len(failures) <= 2:
        return "8/10"
    if len(failures) <= 5:
        return "6/10"
    return "4/10"


def main() -> int:
    """Run Prompt 2.7 acceptance without heavy compute."""

    failures: list[str] = []
    baseline_text = read_text(REPO_ROOT / "src/agentiad_recon/baseline.py")
    prediction_schema = read_text(REPO_ROOT / "src/agentiad_recon/contracts/schemas/baseline_prediction_record.schema.json")
    metrics_schema = read_text(REPO_ROOT / "src/agentiad_recon/contracts/schemas/baseline_metrics_report.schema.json")
    summary_schema = read_text(REPO_ROOT / "src/agentiad_recon/contracts/schemas/baseline_run_summary.schema.json")
    manifest_schema = read_text(REPO_ROOT / "src/agentiad_recon/contracts/schemas/artifact_manifest.schema.json")
    run_definition_schema = read_text(REPO_ROOT / "src/agentiad_recon/contracts/schemas/baseline_run_definition.schema.json")
    readme_text = read_text(REPO_ROOT / "README.md").lower()
    worklog_text = read_text(
        REPO_ROOT / "Working Log/2026-03-16 Prompt 2.7 pz_cr first-turn protocol gate and single-retry audit under a valid tool contract.md"
    ).lower()

    for relative in [
        "src/agentiad_recon/baseline.py",
        "src/agentiad_recon/evaluation.py",
        "src/agentiad_recon/contracts/schemas/baseline_prediction_record.schema.json",
        "src/agentiad_recon/contracts/schemas/baseline_metrics_report.schema.json",
        "src/agentiad_recon/contracts/schemas/baseline_run_summary.schema.json",
        "src/agentiad_recon/contracts/schemas/artifact_manifest.schema.json",
        "tests/test_prompt_2_7_first_turn_protocol_gate.py",
        "audit/check_prompt_2_7.py",
    ]:
        require((REPO_ROOT / relative).exists(), f"required file exists: {relative}", failures)

    for phrase in [
        "--first-turn-protocol-gate-mode",
        "first_turn_protocol_gate_mode",
        "retry_once_pz_cr",
        "_should_trigger_first_turn_gate",
    ]:
        require(phrase in baseline_text, f"gate hook present: {phrase}", failures)

    for schema_name, schema_text in {
        "run_definition": run_definition_schema,
        "prediction": prediction_schema,
        "metrics": metrics_schema,
        "summary": summary_schema,
        "manifest": manifest_schema,
    }.items():
        require(
            "first_turn_protocol_gate_mode" in schema_text or "first_turn_gate_summary" in schema_text,
            f"{schema_name} schema includes Prompt 2.7 contract fields",
            failures,
        )

    require(
        "first-turn protocol gate" in readme_text and "retry_once_pz_cr" in readme_text,
        "README documents Prompt 2.7 gate surface",
        failures,
    )
    for phrase in [
        "what changed",
        "why it changed",
        "effects",
        "git-diff-style summary",
        "2.6 results show wording changes had zero behavioral effect",
        "this patch does not claim to fix the model",
    ]:
        require(phrase in worklog_text, f"working log field present: {phrase}", failures)

    suite = unittest.defaultTestLoader.discover(str(REPO_ROOT / "tests"), pattern="test_prompt_2_*.py")
    result = unittest.TextTestRunner(stream=sys.stdout, verbosity=1).run(suite)
    require(result.wasSuccessful(), "Prompt 2.x lightweight tests pass", failures)

    samples = [sample.to_dict() for sample in MMADIndexer(FIXTURE_ROOT, source="mmad_fixture").index_samples()]
    sample = samples[0]
    pz_cr_definition = load_run_definition(PZ_CR_CONFIG)
    pz_only_definition = load_run_definition(PZ_ONLY_CONFIG)

    gate_off_backend = SequenceBackend([_direct_final_answer("gate off final answer")])
    with tempfile.TemporaryDirectory() as tempdir:
        directories = _artifact_dirs(pz_cr_definition, Path(tempdir))
        record, _trace_payload = _tool_loop_sample(
            definition=pz_cr_definition,
            backend=gate_off_backend,
            runtime_config=_runtime_config(
                pz_cr_definition,
                dataset_root=FIXTURE_ROOT,
                runtime_overrides={"first_turn_protocol_gate_mode": "off"},
            ),
            sample=sample,
            sample_pool=samples,
            seed=0,
            directories=directories,
        )
        require(record["first_turn_gate_triggered"] is False, "gate default off does not trigger", failures)
        require(record["first_turn_gate_retry_count"] == 0, "gate-off retry count stays zero", failures)

    pz_only_backend = SequenceBackend([_direct_final_answer("pz_only final answer")])
    with tempfile.TemporaryDirectory() as tempdir:
        directories = _artifact_dirs(pz_only_definition, Path(tempdir))
        record, _trace_payload = _tool_loop_sample(
            definition=pz_only_definition,
            backend=pz_only_backend,
            runtime_config=_runtime_config(
                pz_only_definition,
                dataset_root=FIXTURE_ROOT,
                runtime_overrides={"first_turn_protocol_gate_mode": "retry_once_pz_cr"},
            ),
            sample=sample,
            sample_pool=samples,
            seed=0,
            directories=directories,
        )
        require(record["first_turn_gate_triggered"] is False, "gate does not trigger in pz_only", failures)
        require(record["first_turn_gate_sidecar_path"] is None, "pz_only gate path writes no sidecar", failures)

    broken_prompt_backend = SequenceBackend([_direct_final_answer("should not run")])
    with tempfile.TemporaryDirectory() as tempdir, mock.patch(
        "agentiad_recon.baseline.build_prompt",
        side_effect=lambda sample, tool_path, **_kwargs: _leaked_pz_only_prompt(sample),
    ):
        directories = _artifact_dirs(pz_cr_definition, Path(tempdir))
        record, _trace_payload = _tool_loop_sample(
            definition=pz_cr_definition,
            backend=broken_prompt_backend,
            runtime_config=_runtime_config(
                pz_cr_definition,
                dataset_root=FIXTURE_ROOT,
                runtime_overrides={"first_turn_protocol_gate_mode": "retry_once_pz_cr"},
            ),
            sample=sample,
            sample_pool=samples,
            seed=0,
            directories=directories,
        )
        require(record["first_turn_gate_triggered"] is False, "gate does not trigger on broken prompt contract", failures)
        require(broken_prompt_backend.calls == 0, "broken prompt contract still fails fast before backend.generate", failures)

    recovery_backend = SequenceBackend(
        [_direct_final_answer("turn 0 final"), TOOL_CALL_PZ, _direct_final_answer("post-tool final")]
    )
    with tempfile.TemporaryDirectory() as tempdir:
        directories = _artifact_dirs(pz_cr_definition, Path(tempdir))
        record, _trace_payload = _tool_loop_sample(
            definition=pz_cr_definition,
            backend=recovery_backend,
            runtime_config=_runtime_config(
                pz_cr_definition,
                dataset_root=FIXTURE_ROOT,
                runtime_overrides={"first_turn_protocol_gate_mode": "retry_once_pz_cr"},
            ),
            sample=sample,
            sample_pool=samples,
            seed=0,
            directories=directories,
        )
        require(record["first_turn_gate_triggered"], "gate triggers on valid turn-0 direct final answer", failures)
        require(record["first_turn_gate_retry_count"] == 1, "retry count never exceeds one", failures)
        require(record["first_turn_gate_outcome"] == "recovered_to_tool_call", "recovery path is recorded", failures)
        require(record["tool_call_count"] > 0, "recovery path uses a real model-emitted tool call", failures)

    still_terminal_backend = SequenceBackend(
        [_direct_final_answer("turn 0 final"), _direct_final_answer("retry final")]
    )
    with tempfile.TemporaryDirectory() as tempdir, mock.patch(
        "agentiad_recon.baseline._select_backend",
        return_value=still_terminal_backend,
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
        sidecars = sorted(Path(tempdir).glob("raw_outputs/seed_*/*/first_turn_gate.turn_0.json"))
        require(sidecars != [], "sidecar exists for each triggered sample", failures)
        require(
            all(record["first_turn_gate_retry_count"] <= 1 for record in result["prediction_records"]),
            "retry count never exceeds one across prediction records",
            failures,
        )
        require(
            all(
                (record["tool_call_count"] == 0 and record["called_tools"] == [])
                or record["first_turn_gate_outcome"] == "recovered_to_tool_call"
                for record in result["prediction_records"]
            ),
            "no fabricated tool call path exists",
            failures,
        )
        for payload in (summary, manifest, metrics):
            gate = payload["first_turn_gate_summary"]
            require(
                gate["first_turn_gate_trigger_count"] == gate["samples_with_first_turn_gate_events"],
                "summary gate counts are internally consistent",
                failures,
            )

    score = _score(failures)
    print(f"\nAcceptance score: {score}")
    if failures:
        print("Prompt 2.7 acceptance check FAILED.")
        for failure in failures:
            print(f"- {failure}")
        return 1

    print("Prompt 2.7 acceptance check PASSED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
