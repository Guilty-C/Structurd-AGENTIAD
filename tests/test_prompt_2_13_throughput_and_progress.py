"""Prompt 2.13 tests for throughput controls, timing, and progress monitoring."""

from __future__ import annotations

import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from agentiad_recon.backends import BackendRequest, BackendResponse, InferenceBackend
from agentiad_recon.baseline import (
    ProgressReporter,
    _artifact_dirs,
    _runtime_config,
    _tool_loop_sample,
    load_run_definition,
    run_baseline,
    run_from_config,
    run_tool_augmented,
)
from agentiad_recon.mmad import MMADIndexer
from agentiad_recon.prompting import render_answer_block


FIXTURE_ROOT = REPO_ROOT / "tests" / "fixtures" / "mmad_fixture"
NO_TOOLS_CONFIG = REPO_ROOT / "configs" / "baseline_non_tool_fixture.json"
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
    """Backend that returns one scripted sequence of raw outputs."""

    backend_name = "prompt_2_13_sequence_backend_v1"

    def __init__(self, outputs: list[str]) -> None:
        self.outputs = outputs
        self.calls = 0

    def describe_runtime(self) -> dict[str, object]:
        return {"adapter_loaded": False}

    def generate(self, request: BackendRequest, *, sample: dict[str, object]) -> BackendResponse:
        index = min(self.calls, len(self.outputs) - 1)
        self.calls += 1
        return BackendResponse(
            backend_name=self.backend_name,
            raw_output=self.outputs[index],
            metadata={"generation_stage": request.metadata.get("generation_stage")},
        )


class RecordingSequenceBackend(SequenceBackend):
    """Scripted backend that records stage-specific generation configs."""

    backend_name = "prompt_2_13_recording_backend_v1"

    def __init__(self, outputs: list[str]) -> None:
        super().__init__(outputs)
        self.stage_records: list[dict[str, object]] = []

    def generate(self, request: BackendRequest, *, sample: dict[str, object]) -> BackendResponse:
        self.stage_records.append(
            {
                "stage": request.metadata.get("generation_stage"),
                "generation_config": dict(request.generation_config or {}),
                "sample_id": request.sample_id,
            }
        )
        return super().generate(request, sample=sample)


class Prompt213ThroughputAndProgressTests(unittest.TestCase):
    """Prompt 2.13 regression tests."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.no_tools_definition = load_run_definition(NO_TOOLS_CONFIG)
        cls.pz_cr_definition = load_run_definition(PZ_CR_CONFIG)
        cls.samples = [
            sample.to_dict()
            for sample in MMADIndexer(FIXTURE_ROOT, source="mmad_fixture").index_samples()
        ]
        cls.sample = cls.samples[0]

    def _tool_runtime(self, **overrides: object) -> dict[str, object]:
        runtime_overrides = {
            "first_turn_protocol_gate_mode": "retry_once_pz_cr",
            "post_pz_second_turn_gate_mode": "retry_once_require_cr_after_pz",
        }
        runtime_overrides.update(overrides)
        return _runtime_config(
            self.pz_cr_definition,
            dataset_root=FIXTURE_ROOT,
            runtime_overrides=runtime_overrides,
        )

    def _tool_loop_record(
        self,
        outputs: list[str],
        *,
        runtime_overrides: dict[str, object] | None = None,
        backend: InferenceBackend | None = None,
    ) -> tuple[dict[str, object], Path]:
        backend = backend or SequenceBackend(outputs)
        with tempfile.TemporaryDirectory() as tempdir:
            directories = _artifact_dirs(self.pz_cr_definition, Path(tempdir))
            record, _trace_payload = _tool_loop_sample(
                definition=self.pz_cr_definition,
                backend=backend,
                runtime_config=self._tool_runtime(**(runtime_overrides or {})),
                sample=self.sample,
                sample_pool=self.samples,
                seed=0,
                directories=directories,
            )
            return record, Path(tempdir)

    def test_default_behavior_with_no_new_flags_remains_compatible(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            result = run_baseline(
                config_path=NO_TOOLS_CONFIG,
                dataset_root=FIXTURE_ROOT,
                artifact_root=tempdir,
                max_samples=1,
            )
            summary = json.loads(Path(result["summary_path"]).read_text(encoding="utf-8"))
            manifest = json.loads(Path(result["run_manifest_path"]).read_text(encoding="utf-8"))
            summary_exists = Path(result["summary_path"]).exists()
            manifest_exists = Path(result["run_manifest_path"]).exists()

        self.assertEqual(summary["tool_mode"], "no_tools")
        self.assertEqual(summary["runtime_provenance"]["artifact_level"], "forensic")
        self.assertTrue(summary_exists)
        self.assertTrue(manifest_exists)
        self.assertNotIn("timing_summary", summary)
        self.assertEqual(manifest["progress_mode"], "off")

    def test_stage_specific_generation_override_chooses_correct_budget_per_stage(self) -> None:
        backend = RecordingSequenceBackend(
            [
                _leaky_first_turn_final_answer(),
                TOOL_CALL_PZ,
                _direct_final_answer("second turn direct final answer"),
                TOOL_CALL_CR,
                _direct_final_answer("final answer after CR"),
            ]
        )
        generation_stage_overrides = {
            "turn0_initial": {"max_new_tokens": 96},
            "turn0_retry": {"max_new_tokens": 64},
            "post_pz_second_turn": {"max_new_tokens": 72},
            "post_pz_second_turn_retry": {"max_new_tokens": 48},
            "final_answer": {"max_new_tokens": 160},
        }
        with tempfile.TemporaryDirectory() as tempdir:
            directories = _artifact_dirs(self.pz_cr_definition, Path(tempdir))
            _tool_loop_sample(
                definition=self.pz_cr_definition,
                backend=backend,
                runtime_config=self._tool_runtime(
                    generation_stage_overrides=generation_stage_overrides
                ),
                sample=self.sample,
                sample_pool=self.samples,
                seed=0,
                directories=directories,
            )

        self.assertEqual(
            [entry["stage"] for entry in backend.stage_records],
            [
                "turn0_initial",
                "turn0_retry",
                "post_pz_second_turn",
                "post_pz_second_turn_retry",
                "final_answer",
            ],
        )
        self.assertEqual(
            [entry["generation_config"]["max_new_tokens"] for entry in backend.stage_records],
            [96, 64, 72, 48, 160],
        )

    def test_emit_baseline_compare_false_skips_compare_tail_but_keeps_core_artifacts(self) -> None:
        backend = SequenceBackend([TOOL_CALL_PZ, TOOL_CALL_CR, _direct_final_answer("final answer")])
        with tempfile.TemporaryDirectory() as tempdir, mock.patch(
            "agentiad_recon.baseline._select_backend",
            return_value=backend,
        ), mock.patch(
            "agentiad_recon.baseline.run_baseline",
            side_effect=AssertionError("baseline compare tail should not run"),
        ):
            result = run_tool_augmented(
                config_path=PZ_CR_CONFIG,
                dataset_root=FIXTURE_ROOT,
                artifact_root=tempdir,
                max_samples=1,
                runtime_overrides={
                    "first_turn_protocol_gate_mode": "off",
                    "post_pz_second_turn_gate_mode": "off",
                    "emit_baseline_compare": False,
                    "emit_delta_report": False,
                },
            )
            summary = json.loads(Path(result["summary_path"]).read_text(encoding="utf-8"))
            manifest = json.loads(Path(result["run_manifest_path"]).read_text(encoding="utf-8"))
            summary_exists = Path(result["summary_path"]).exists()
            manifest_exists = Path(result["run_manifest_path"]).exists()

        self.assertTrue(summary_exists)
        self.assertTrue(manifest_exists)
        self.assertFalse(summary["emit_baseline_compare"])
        self.assertFalse(summary["emit_delta_report"])
        self.assertTrue(summary["core_artifacts_written_before_optional_tail_work"])
        self.assertNotIn("delta_report", summary["artifact_paths"])
        self.assertNotIn("baseline_compare_summary", result)
        self.assertNotIn("delta_report_path", result)
        self.assertFalse(manifest["emit_baseline_compare"])

    def test_emit_delta_report_false_skips_delta_but_keeps_compare_summary(self) -> None:
        backend = SequenceBackend([TOOL_CALL_PZ, TOOL_CALL_CR, _direct_final_answer("final answer")])
        fake_baseline_result = {
            "summary_path": "/tmp/fake_baseline_summary.json",
            "run_definition": {"run_id": "fake_baseline"},
            "metrics_report": {
                "aggregate_metrics": {
                    "toolcall_rate": {"mean": 0.0, "std": 0.0},
                    "per_tool_frequency": {
                        "PZ": {"mean": 0.0, "std": 0.0},
                        "CR": {"mean": 0.0, "std": 0.0},
                    },
                },
                "per_seed_metrics": {},
                "per_class_metrics": {},
                "sample_count": 1,
                "evaluated_count": 1,
                "failed_count": 0,
            },
        }
        with tempfile.TemporaryDirectory() as tempdir, mock.patch(
            "agentiad_recon.baseline._select_backend",
            return_value=backend,
        ), mock.patch(
            "agentiad_recon.baseline.run_baseline",
            return_value=fake_baseline_result,
        ) as run_baseline_mock:
            result = run_tool_augmented(
                config_path=PZ_CR_CONFIG,
                dataset_root=FIXTURE_ROOT,
                artifact_root=tempdir,
                max_samples=1,
                runtime_overrides={
                    "first_turn_protocol_gate_mode": "off",
                    "post_pz_second_turn_gate_mode": "off",
                    "emit_baseline_compare": True,
                    "emit_delta_report": False,
                },
            )
            summary = json.loads(Path(result["summary_path"]).read_text(encoding="utf-8"))

        run_baseline_mock.assert_called_once()
        self.assertEqual(result["baseline_compare_summary"], fake_baseline_result["summary_path"])
        self.assertNotIn("delta_report_path", result)
        self.assertNotIn("delta_report", summary["artifact_paths"])

    def test_artifact_level_throughput_suppresses_ordinary_sidecars_but_preserves_gate_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            directories = _artifact_dirs(self.pz_cr_definition, Path(tempdir))
            clean_record, _ = _tool_loop_sample(
                definition=self.pz_cr_definition,
                backend=SequenceBackend([TOOL_CALL_PZ, TOOL_CALL_CR, _direct_final_answer("final answer")]),
                runtime_config=self._tool_runtime(
                    artifact_level="throughput",
                    first_turn_protocol_gate_mode="off",
                    post_pz_second_turn_gate_mode="off",
                ),
                sample=self.sample,
                sample_pool=self.samples,
                seed=0,
                directories=directories,
            )
            gated_record, _ = _tool_loop_sample(
                definition=self.pz_cr_definition,
                backend=SequenceBackend(
                    [
                        _leaky_first_turn_final_answer(),
                        TOOL_CALL_PZ,
                        _direct_final_answer("second turn direct final answer"),
                        TOOL_CALL_CR,
                        _direct_final_answer("final answer after CR"),
                    ]
                ),
                runtime_config=self._tool_runtime(artifact_level="throughput"),
                sample=self.sample,
                sample_pool=self.samples,
                seed=1,
                directories=directories,
            )

            self.assertFalse(Path(clean_record["raw_output_path"]).exists())
            self.assertFalse(Path(clean_record["post_pz_transition_sidecar_path"]).exists())
            self.assertTrue(Path(gated_record["raw_output_path"]).exists())
            self.assertTrue(Path(gated_record["first_turn_gate_sidecar_path"]).exists())
            self.assertTrue(Path(gated_record["post_pz_second_turn_gate_sidecar_path"]).exists())

    def test_timing_enabled_emits_timing_summary_in_summary_report_and_manifest(self) -> None:
        backend = SequenceBackend([TOOL_CALL_PZ, TOOL_CALL_CR, _direct_final_answer("final answer")])
        with tempfile.TemporaryDirectory() as tempdir, mock.patch(
            "agentiad_recon.baseline._select_backend",
            return_value=backend,
        ), mock.patch(
            "agentiad_recon.baseline.run_baseline",
            side_effect=AssertionError("baseline compare tail should not run"),
        ):
            result = run_tool_augmented(
                config_path=PZ_CR_CONFIG,
                dataset_root=FIXTURE_ROOT,
                artifact_root=tempdir,
                max_samples=1,
                runtime_overrides={
                    "first_turn_protocol_gate_mode": "off",
                    "post_pz_second_turn_gate_mode": "off",
                    "emit_baseline_compare": False,
                    "emit_delta_report": False,
                    "timing_enabled": True,
                },
            )
            summary = json.loads(Path(result["summary_path"]).read_text(encoding="utf-8"))
            metrics = json.loads((Path(tempdir) / "metrics" / "metrics_report.json").read_text(encoding="utf-8"))
            manifest = json.loads(Path(result["run_manifest_path"]).read_text(encoding="utf-8"))

        self.assertIn("timing_summary", summary)
        self.assertIn("timing_summary", metrics)
        self.assertIn("timing_summary", manifest)
        self.assertGreater(summary["timing_summary"]["generation_call_count_total"], 0)

    def test_progress_reporter_updates_snapshot_json(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            snapshot_path = Path(tempdir) / "progress" / "progress_snapshot.json"
            reporter = ProgressReporter(
                run_id="prompt_2_13_progress",
                total_samples=5,
                mode="log",
                update_every_n_samples=1,
                snapshot_path=snapshot_path,
            )
            reporter.update(
                processed_samples=2,
                current_sample_id="sample-2",
                timing_summary={"generation_call_count_total": 3, "retry_count_total": 1},
            )

            payload = json.loads(snapshot_path.read_text(encoding="utf-8"))

        self.assertEqual(payload["processed_samples"], 2)
        self.assertEqual(payload["current_sample_id"], "sample-2")
        self.assertEqual(payload["generation_call_count_total"], 3)
        self.assertIn("last_update_utc", payload)

    def test_progress_auto_mode_falls_back_to_log_in_non_tty(self) -> None:
        fake_stderr = io.StringIO()
        fake_stderr.isatty = lambda: False  # type: ignore[attr-defined]
        with mock.patch("agentiad_recon.baseline.sys.stderr", fake_stderr):
            reporter = ProgressReporter(
                run_id="prompt_2_13_progress_auto",
                total_samples=3,
                mode="auto",
                update_every_n_samples=1,
                snapshot_path=None,
            )
            reporter.update(
                processed_samples=1,
                current_sample_id="sample-1",
                timing_summary={"generation_call_count_total": 1, "retry_count_total": 0},
            )

        self.assertEqual(reporter._resolved_mode, "log")
        self.assertIn("[progress]", fake_stderr.getvalue())

    def test_progress_log_mode_emits_parseable_progress_lines(self) -> None:
        fake_stderr = io.StringIO()
        with mock.patch("agentiad_recon.baseline.sys.stderr", fake_stderr):
            reporter = ProgressReporter(
                run_id="prompt_2_13_progress_log",
                total_samples=4,
                mode="log",
                update_every_n_samples=1,
                snapshot_path=None,
            )
            reporter.update(
                processed_samples=1,
                current_sample_id="sample-1",
                timing_summary={"generation_call_count_total": 2, "retry_count_total": 1},
            )

        line = fake_stderr.getvalue()
        self.assertIn("[progress]", line)
        self.assertIn("processed=1/4", line)
        self.assertIn("sample_id=sample-1", line)

    def test_no_new_branch_creates_second_evaluator_path(self) -> None:
        with mock.patch("agentiad_recon.baseline.run_baseline", return_value={"path": "baseline"}) as run_baseline_mock:
            result = run_from_config(
                config_path=NO_TOOLS_CONFIG,
                dataset_root=FIXTURE_ROOT,
                artifact_root="/tmp/unused",
                max_samples=1,
            )
        self.assertEqual(result["path"], "baseline")
        run_baseline_mock.assert_called_once()

        with mock.patch(
            "agentiad_recon.baseline.run_tool_augmented",
            return_value={"path": "tool_augmented"},
        ) as run_tool_augmented_mock:
            result = run_from_config(
                config_path=PZ_CR_CONFIG,
                dataset_root=FIXTURE_ROOT,
                artifact_root="/tmp/unused",
                max_samples=1,
            )
        self.assertEqual(result["path"], "tool_augmented")
        run_tool_augmented_mock.assert_called_once()
