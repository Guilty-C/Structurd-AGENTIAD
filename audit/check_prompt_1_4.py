"""Lightweight acceptance check for Prompt 1.4.

This script verifies that the canonical Prompt 1.3 runner was extended into a
single tool-enabled path for `pz_only` and `pz_cr`, that the frozen configs and
docs exist, and that local-only fixture-backed smoke validation passes. It does
not run any real model inference or full-MMAD evaluation.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from agentiad_recon.baseline import run_tool_augmented
from agentiad_recon.tooling import ToolContractError, parse_tool_call


FIXTURE_ROOT = REPO_ROOT / "tests" / "fixtures" / "mmad_fixture"
PZ_ONLY_CONFIG = REPO_ROOT / "configs" / "tool_pz_only_fixture.json"
PZ_CR_CONFIG = REPO_ROOT / "configs" / "tool_pz_cr_fixture.json"


def require(condition: bool, label: str, failures: list[str]) -> None:
    """Print one acceptance result and collect failures."""

    if condition:
        print(f"PASS: {label}")
    else:
        print(f"FAIL: {label}")
        failures.append(label)


def read_text(path: Path) -> str:
    """Read one UTF-8 text file from the repository."""

    return path.read_text(encoding="utf-8")


def _run_smoke(config_path: Path) -> tuple[dict[str, object], Path]:
    """Execute one local tool-enabled smoke run inside a temporary artifact root."""

    artifact_root = Path(tempfile.mkdtemp(prefix=f"{config_path.stem}_"))
    result = run_tool_augmented(
        config_path=config_path,
        dataset_root=FIXTURE_ROOT,
        artifact_root=artifact_root,
        max_samples=2,
    )
    return result, artifact_root


def main() -> int:
    """Run file-presence checks plus Prompt 1.4 smoke validations."""

    failures: list[str] = []
    readme_text = read_text(REPO_ROOT / "README.md")
    worklog_text = read_text(REPO_ROOT / "Working Log/2026-03-10 Prompt 1.4.md").lower()
    report_text = read_text(REPO_ROOT / "audit/reports/prompt_1_4_acceptance_report.md")
    config_readme_text = read_text(REPO_ROOT / "configs/README.md")
    pz_only_config_text = read_text(PZ_ONLY_CONFIG)
    pz_cr_config_text = read_text(PZ_CR_CONFIG)

    for relative in [
        "src/agentiad_recon/backends.py",
        "src/agentiad_recon/baseline.py",
        "src/agentiad_recon/evaluation.py",
        "src/agentiad_recon/tooling.py",
        "src/agentiad_recon/contracts/schemas/tool_run_definition.schema.json",
        "src/agentiad_recon/contracts/schemas/tool_delta_report.schema.json",
        "configs/tool_pz_only_fixture.json",
        "configs/tool_pz_cr_fixture.json",
        "tests/test_prompt_1_4_tool_inference.py",
    ]:
        require((REPO_ROOT / relative).exists(), f"required file exists: {relative}", failures)

    for phrase in [
        "Tool-Augmented Inference",
        "pz_only",
        "pz_cr",
        "delta-vs-baseline",
        "tool_pz_only_fixture.json",
        "tool_pz_cr_fixture.json",
    ]:
        require(phrase in readme_text, f"README phrase present: {phrase}", failures)

    for phrase in ["tool_pz_only_fixture.json", "tool_pz_cr_fixture.json", "Prompt 1.4"]:
        require(phrase in config_readme_text, f"config README phrase present: {phrase}", failures)

    require('"mode": "pz_only"' in pz_only_config_text, "pz_only config mode is frozen", failures)
    require('"mode": "pz_cr"' in pz_cr_config_text, "pz_cr config mode is frozen", failures)
    require('"policy": "fixture_scripted_pz_only_v1"' in pz_only_config_text, "pz_only config uses scripted mock backend", failures)
    require('"policy": "fixture_scripted_pz_cr_v1"' in pz_cr_config_text, "pz_cr config uses scripted mock backend", failures)

    for phrase in ["timestamp", "prompt number", "what changed", "why it changed", "effects", "git-diff-style summary"]:
        require(phrase in worklog_text, f"working log field present: {phrase}", failures)

    for phrase in ["self-score", "Heavy compute was NOT run", "Known remaining gaps"]:
        require(phrase in report_text, f"acceptance report field present: {phrase}", failures)

    try:
        parse_tool_call(
            "<tool_call>\n"
            "{\"tool_name\":\"CR\",\"arguments\":{\"policy\":\"same_category_normal\"}}\n"
            "</tool_call>",
            tool_path="pz_only",
        )
    except ToolContractError:
        require(True, "CR is rejected in pz_only mode", failures)
    else:
        require(False, "CR is rejected in pz_only mode", failures)

    try:
        parse_tool_call(
            "<tool_call>\n"
            "{\"tool_name\":\"PZ\",\"arguments\":{\"bbox\":{\"x0\":0.1,\"y0\":0.1,\"x1\":oops}}}\n"
            "</tool_call>",
            tool_path="pz_only",
        )
    except ToolContractError:
        require(True, "malformed tool call handling is explicit", failures)
    else:
        require(False, "malformed tool call handling is explicit", failures)

    pz_only_result, pz_only_root = _run_smoke(PZ_ONLY_CONFIG)
    pz_cr_result, pz_cr_root = _run_smoke(PZ_CR_CONFIG)

    pz_only_summary = json.loads(Path(pz_only_result["summary_path"]).read_text(encoding="utf-8"))
    pz_cr_summary = json.loads(Path(pz_cr_result["summary_path"]).read_text(encoding="utf-8"))
    pz_only_delta = json.loads(Path(pz_only_result["delta_report_path"]).read_text(encoding="utf-8"))
    pz_cr_delta = json.loads(Path(pz_cr_result["delta_report_path"]).read_text(encoding="utf-8"))

    require(
        pz_only_summary["tool_usage_summary"]["per_tool_counts"]["PZ"] > 0,
        "PZ is actually called in pz_only smoke runs",
        failures,
    )
    require(
        pz_only_summary["tool_usage_summary"]["per_tool_counts"]["CR"] == 0,
        "CR stays unused in pz_only smoke runs",
        failures,
    )
    require(
        pz_cr_summary["tool_usage_summary"]["per_tool_counts"]["PZ"] > 0,
        "PZ is actually called in pz_cr smoke runs",
        failures,
    )
    require(
        pz_cr_summary["tool_usage_summary"]["per_tool_counts"]["CR"] > 0,
        "CR is actually called in pz_cr smoke runs",
        failures,
    )
    require(
        pz_only_delta["tool_usage_delta"]["toolcall_rate"] > 0.0,
        "tool usage is non-zero and auditable for pz_only",
        failures,
    )
    require(
        pz_cr_delta["tool_usage_delta"]["per_tool_frequency"]["CR"] > 0.0,
        "tool usage is non-zero and auditable for pz_cr",
        failures,
    )
    require(
        Path(pz_only_result["delta_report_path"]).exists() and Path(pz_cr_result["delta_report_path"]).exists(),
        "delta-vs-baseline artifacts exist",
        failures,
    )
    require(
        list(pz_only_root.glob("traces/seed_*/*.json")) != [] and list(pz_cr_root.glob("traces/seed_*/*.json")) != [],
        "trace artifacts are written",
        failures,
    )

    suite = unittest.defaultTestLoader.discover(
        str(REPO_ROOT / "tests"),
        pattern="test_prompt_1_4_tool_inference.py",
    )
    result = unittest.TextTestRunner(stream=sys.stdout, verbosity=1).run(suite)
    require(result.wasSuccessful(), "Prompt 1.4 tool smoke tests pass", failures)

    if failures:
        print("\nPrompt 1.4 acceptance check FAILED.")
        for failure in failures:
            print(f"- {failure}")
        return 1

    print("\nPrompt 1.4 acceptance check PASSED.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
