"""Acceptance check for Prompt 2.6 tool-first intervention ablation."""

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
from agentiad_recon.prompting import render_answer_block


FIXTURE_ROOT = REPO_ROOT / "tests" / "fixtures" / "mmad_fixture"
PZ_CR_CONFIG = REPO_ROOT / "configs" / "tool_pz_cr_fixture.json"
STRATEGIES = ("baseline", "tool_first_nudge", "tool_first_strict")


class DirectFinalAnswerBackend(InferenceBackend):
    """Acceptance backend that always emits a direct final answer."""

    backend_name = "prompt_2_6_acceptance_direct_final_backend_v1"

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
                think="Prompt 2.6 acceptance direct terminal answer.",
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


def _run_strategy(temp_root: Path, strategy: str) -> dict[str, object]:
    """Execute one lightweight strategy run through the unified evaluator."""

    artifact_root = temp_root / strategy
    with mock.patch("agentiad_recon.baseline._select_backend", return_value=DirectFinalAnswerBackend()):
        result = run_tool_augmented(
            config_path=PZ_CR_CONFIG,
            dataset_root=FIXTURE_ROOT,
            artifact_root=artifact_root,
            max_samples=1,
            runtime_overrides={"tool_first_intervention_strategy": strategy},
        )
    summary = json.loads(Path(result["summary_path"]).read_text(encoding="utf-8"))
    manifest = json.loads(Path(result["run_manifest_path"]).read_text(encoding="utf-8"))
    metrics = result["metrics_report"]
    prediction_path = next((artifact_root / "predictions" / "seed_0").glob("*.json"))
    prompt_audit_path = next(artifact_root.glob("raw_outputs/seed_0/*/prompt_audit.turn_0.json"))
    strategy_summary_path = artifact_root / "metrics" / "tool_first_strategy_summary.json"
    return {
        "root": artifact_root,
        "summary": summary,
        "manifest": manifest,
        "metrics": metrics,
        "prediction": json.loads(prediction_path.read_text(encoding="utf-8")),
        "prompt_audit": json.loads(prompt_audit_path.read_text(encoding="utf-8")),
        "strategy_summary_path": strategy_summary_path,
        "strategy_summary": json.loads(strategy_summary_path.read_text(encoding="utf-8")),
    }


def _score(failures: list[str]) -> str:
    """Collapse acceptance failures into a simple score band."""

    if not failures:
        return "10/10"
    if len(failures) <= 2:
        return "8/10"
    if len(failures) <= 5:
        return "6/10"
    return "4/10"


def main() -> int:
    """Run Prompt 2.6 acceptance without heavy compute."""

    failures: list[str] = []
    baseline_text = read_text(REPO_ROOT / "src/agentiad_recon/baseline.py")
    behavior_text = read_text(REPO_ROOT / "src/agentiad_recon/behavior_audit.py")
    prompting_text = read_text(REPO_ROOT / "src/agentiad_recon/prompting.py")
    run_definition_schema = read_text(REPO_ROOT / "src/agentiad_recon/contracts/schemas/baseline_run_definition.schema.json")
    metrics_schema = read_text(REPO_ROOT / "src/agentiad_recon/contracts/schemas/baseline_metrics_report.schema.json")
    summary_schema = read_text(REPO_ROOT / "src/agentiad_recon/contracts/schemas/baseline_run_summary.schema.json")
    manifest_schema = read_text(REPO_ROOT / "src/agentiad_recon/contracts/schemas/artifact_manifest.schema.json")
    readme_text = read_text(REPO_ROOT / "README.md").lower()
    worklog_text = read_text(
        REPO_ROOT / "Working Log/2026-03-16 Prompt 2.6 tool-first intervention ablation under valid pz_cr contract.md"
    ).lower()

    for relative in [
        "src/agentiad_recon/baseline.py",
        "src/agentiad_recon/prompting.py",
        "src/agentiad_recon/behavior_audit.py",
        "src/agentiad_recon/contracts/schemas/baseline_run_definition.schema.json",
        "src/agentiad_recon/contracts/schemas/baseline_metrics_report.schema.json",
        "src/agentiad_recon/contracts/schemas/baseline_run_summary.schema.json",
        "src/agentiad_recon/contracts/schemas/artifact_manifest.schema.json",
        "tests/test_prompt_2_6_tool_first_intervention_ablation.py",
        "audit/check_prompt_2_6.py",
    ]:
        require((REPO_ROOT / relative).exists(), f"required file exists: {relative}", failures)

    for phrase in [
        "tool_first_nudge",
        "tool_first_strict",
        "--tool-first-intervention-strategy",
        "tool_first_intervention_strategy",
    ]:
        require(
            phrase in baseline_text or phrase in prompting_text or phrase in behavior_text,
            f"strategy hook present: {phrase}",
            failures,
        )

    require(
        "tool_first_intervention_strategy" in run_definition_schema,
        "runtime config schema includes tool_first_intervention_strategy",
        failures,
    )
    for schema_name, schema_text in {
        "metrics": metrics_schema,
        "summary": summary_schema,
        "manifest": manifest_schema,
    }.items():
        require(
            "tool_first_intervention_strategy" in schema_text,
            f"{schema_name} schema includes strategy provenance",
            failures,
        )

    require(
        "tool-first intervention" in readme_text and "tool_first_nudge" in readme_text,
        "README documents Prompt 2.6 intervention ablation",
        failures,
    )
    for phrase in [
        "what changed",
        "why it changed",
        "effects",
        "git-diff-style summary",
        "toolcall_rate` is still `0.0",
        "this patch does not claim that the model is fixed",
    ]:
        require(phrase in worklog_text, f"working log field present: {phrase}", failures)

    suite = unittest.defaultTestLoader.discover(
        str(REPO_ROOT / "tests"),
        pattern="test_prompt_2_*.py",
    )
    result = unittest.TextTestRunner(stream=sys.stdout, verbosity=1).run(suite)
    require(result.wasSuccessful(), "Prompt 2.x lightweight tests pass", failures)

    with tempfile.TemporaryDirectory() as tempdir:
        temp_root = Path(tempdir)
        strategy_runs = {strategy: _run_strategy(temp_root, strategy) for strategy in STRATEGIES}

        prompt_audit_paths = sorted(temp_root.glob("**/prompt_audit.turn_0.json"))
        strategy_summary_paths = sorted(temp_root.glob("**/tool_first_strategy_summary.json"))
        require(
            len(prompt_audit_paths) >= 3,
            "prompt-audit sidecars auto-discovered for all strategies",
            failures,
        )
        require(len(strategy_summary_paths) == 3, "strategy summary artifacts auto-discovered for all strategies", failures)

        baseline_run = strategy_runs["baseline"]
        nudge_run = strategy_runs["tool_first_nudge"]
        strict_run = strategy_runs["tool_first_strict"]

        require(
            baseline_run["prompt_audit"]["tool_first_intervention_strategy"] == "baseline",
            "baseline prompt audit records baseline strategy",
            failures,
        )
        require(
            nudge_run["prompt_audit"]["tool_first_intervention_strategy"] == "tool_first_nudge",
            "nudge prompt audit records nudge strategy",
            failures,
        )
        require(
            strict_run["prompt_audit"]["tool_first_intervention_strategy"] == "tool_first_strict",
            "strict prompt audit records strict strategy",
            failures,
        )
        require(
            baseline_run["prompt_audit"]["intervention_text_applied"] is False,
            "baseline prompt surface keeps intervention_text_applied=false",
            failures,
        )
        require(
            nudge_run["prompt_audit"]["intervention_text_applied"] is True
            and strict_run["prompt_audit"]["intervention_text_applied"] is True,
            "nudge/strict prompt surfaces apply intervention text",
            failures,
        )
        require(
            baseline_run["prompt_audit"]["prompt_surface_digest"] != nudge_run["prompt_audit"]["prompt_surface_digest"]
            and nudge_run["prompt_audit"]["prompt_surface_digest"] != strict_run["prompt_audit"]["prompt_surface_digest"],
            "prompt-audit digests differ across baseline/nudge/strict",
            failures,
        )
        require(
            "tool_first_nudge" in nudge_run["prompt_audit"]["user_text"]
            and "tool_first_strict" in strict_run["prompt_audit"]["user_text"],
            "prompt surfaces differ in the intended intervention direction",
            failures,
        )

        for strategy, payload in strategy_runs.items():
            for exported in (payload["summary"], payload["manifest"], payload["metrics"]):
                require(
                    exported["tool_first_intervention_strategy"] == strategy,
                    f"{strategy}: exported payload records selected strategy",
                    failures,
                )
            require(
                payload["summary"]["runtime_provenance"]["tool_first_intervention_strategy"] == strategy,
                f"{strategy}: summary runtime provenance records selected strategy",
                failures,
            )
            require(
                payload["metrics"]["runtime_provenance"]["tool_first_intervention_strategy"] == strategy,
                f"{strategy}: metrics runtime provenance records selected strategy",
                failures,
            )
            require(
                payload["manifest"]["run_provenance"]["tool_first_intervention_strategy"] == strategy,
                f"{strategy}: manifest run provenance records selected strategy",
                failures,
            )
            require(
                payload["prediction"]["tool_call_count"] == 0 and payload["prediction"]["called_tools"] == [],
                f"{strategy}: no fabricated tool calls under direct-final backend",
                failures,
            )
            require(
                payload["strategy_summary"]["tool_first_intervention_strategy"] == strategy,
                f"{strategy}: per-strategy comparison artifact records strategy",
                failures,
            )
            require(
                payload["summary"]["zero_tool_behavior_summary"]["zero_tool_terminal_count"] > 0,
                f"{strategy}: zero-tool summary still computed through unified evaluator",
                failures,
            )

    score = _score(failures)
    print(f"\nAcceptance score: {score}")
    if failures:
        print("Prompt 2.6 acceptance check FAILED.")
        for failure in failures:
            print(f"- {failure}")
        return 1

    print("Prompt 2.6 acceptance check PASSED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
