"""Lightweight acceptance check for Prompt 1.1.

This script verifies the local clean-room scaffold, architecture lock,
scientific targets, and canonical schema inventory without running any heavy
compute. It is intended to be a repeatable local gate for this checkpoint.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def require(condition: bool, label: str, failures: list[str]) -> None:
    """Record a pass or fail condition for the final acceptance summary."""

    if condition:
        print(f"PASS: {label}")
    else:
        print(f"FAIL: {label}")
        failures.append(label)


def read_text(path: Path) -> str:
    """Read a UTF-8 text file from the repository."""

    return path.read_text(encoding="utf-8")


def load_json(path: Path) -> dict:
    """Load a JSON document used by the local scaffold."""

    return json.loads(read_text(path))


def main() -> int:
    """Run the checkpoint checks and return a process exit code."""

    failures: list[str] = []

    # These directories are the minimum clean-room scaffold required by Prompt 1.1.
    required_dirs = [
        "configs",
        "src",
        "adapters",
        "eval",
        "audit",
        "dist/outputs",
        "dist/paper_artifacts",
        "Working Log",
    ]
    for relative in required_dirs:
        require((REPO_ROOT / relative).is_dir(), f"directory exists: {relative}", failures)

    readme_text = read_text(REPO_ROOT / "README.md")
    adr_text = read_text(REPO_ROOT / "audit/adr/ADR-0001-clean-room-architecture.md")
    science_text = read_text(REPO_ROOT / "audit/scientific_target_lock.md")
    worklog_text = read_text(REPO_ROOT / "Working Log/2026-03-10 Prompt 1.1.md")
    report_text = read_text(REPO_ROOT / "audit/reports/prompt_1_1_acceptance_report.md")

    for phrase in ["SFT owner: MS-Swift", "GRPO/RL owner: VERL", "inference/serving owner: vLLM"]:
        require(phrase in adr_text, f"architecture owner present: {phrase}", failures)

    for phrase in ["Keep:", "Quarantine:", "Ignore:", "Framework-owned:", "Adapter-owned:", "Fully custom:"]:
        require(phrase in adr_text, f"ADR section present: {phrase}", failures)

    for phrase in ["single-agent", "PZ", "CR", "pz_only", "pz_cr", "SFT", "GRPO", "perception", "behavior"]:
        require(phrase in science_text or phrase in adr_text, f"scientific target present: {phrase}", failures)

    require("Architecture Ownership" in readme_text, "README architecture section", failures)
    require("Phase Flow" in readme_text, "README phase flow section", failures)
    require("Local vs Remote Boundary" in readme_text, "README local/remote section", failures)
    require("Working Log" in readme_text and "audit/reports" in readme_text, "README audit visibility", failures)

    required_schemas = {
        "sample.schema.json": ["sample_id", "allowed_tools", "ground_truth"],
        "trajectory.schema.json": ["trajectory_id", "tool_path", "final_answer"],
        "tool_call.schema.json": ["call_id", "tool_name", "output_ref"],
        "final_answer.schema.json": ["anomaly_present", "top_anomaly", "visual_descriptions"],
        "reward_input.schema.json": ["perception_signals", "behavior_signals"],
        "artifact_manifest.schema.json": ["artifact_id", "artifact_type", "execution_boundary"],
    }
    schema_root = REPO_ROOT / "src/agentiad_recon/contracts/schemas"
    for schema_name, required_keys in required_schemas.items():
        schema = load_json(schema_root / schema_name)
        require(schema.get("type") == "object", f"schema object type: {schema_name}", failures)
        # The check stays shallow on purpose: it verifies the thin-waist keys
        # without pretending to be a full JSON Schema validator.
        properties = schema.get("properties", {})
        for required_key in required_keys:
            require(required_key in properties or required_key in schema.get("required", []), f"{schema_name} contains {required_key}", failures)

    run_metadata = load_json(REPO_ROOT / "audit/reproducibility/run_metadata.schema.json")
    require("config_hashes" in run_metadata["properties"], "run metadata config hashes", failures)
    require("script_hashes" in run_metadata["properties"], "run metadata script hashes", failures)
    require("dataset_manifest_hash" in run_metadata["properties"], "run metadata dataset manifest hash", failures)

    for phrase in ["timestamp", "prompt number", "what changed", "why it changed", "effects", "git-diff-style summary"]:
        require(phrase in worklog_text.lower(), f"working log field present: {phrase}", failures)

    for phrase in ["self-score", "Heavy compute was NOT run", "Known Remaining Gaps"]:
        require(phrase in report_text, f"acceptance report field present: {phrase}", failures)

    if failures:
        print("\nPrompt 1.1 acceptance check FAILED.")
        print("Failed items:")
        for failure in failures:
            print(f"- {failure}")
        return 1

    print("\nPrompt 1.1 acceptance check PASSED.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
