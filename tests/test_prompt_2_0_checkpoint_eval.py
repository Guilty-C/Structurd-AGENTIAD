"""Checkpoint-eval smoke tests for Prompt 2.0.

These tests keep `baseline.py` as the single entrypoint while validating the
new runtime surface for real transformers+PEFT evaluation. They do not load any
real model weights locally; the transformers path is exercised via mocked
runtime modules and the fixture-backed smoke path still uses the tiny MMAD
fixture dataset.
"""

from __future__ import annotations

import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from agentiad_recon.backends import BackendRequest, TransformersVisionLanguageBackend
from agentiad_recon.baseline import dry_run_from_config, run_baseline


FIXTURE_ROOT = REPO_ROOT / "tests" / "fixtures" / "mmad_fixture"
BASELINE_CONFIG = REPO_ROOT / "configs" / "baseline_non_tool_fixture.json"
REMOTE_PZ_CR_CONFIG = REPO_ROOT / "configs" / "eval_transformers_pz_cr_remote_template.json"
FIXTURE_IMAGE = REPO_ROOT / "tests" / "fixtures" / "mmad_fixture" / "train" / "capsule" / "good" / "images" / "capsule_good_0001.ppm"


class FakeTensor:
    """Tiny tensor stand-in that satisfies the backend slicing path."""

    def __init__(self, shape: tuple[int, ...]) -> None:
        self.shape = shape

    def __getitem__(self, item: object) -> "FakeTensor":
        return self


class FakeBatch(dict):
    """Dict-like batch object with a `.to(...)` method."""

    def to(self, device: str) -> "FakeBatch":
        self["moved_to"] = device
        return self


class FakeProcessor:
    """Minimal processor stub for the transformers backend test."""

    last_from_pretrained: tuple[str, dict[str, object]] | None = None

    @classmethod
    def from_pretrained(cls, model_path: str, **kwargs: object) -> "FakeProcessor":
        cls.last_from_pretrained = (model_path, kwargs)
        return cls()

    def apply_chat_template(
        self,
        chat_messages: list[dict[str, object]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str:
        self.chat_messages = chat_messages
        self.template_options = {
            "tokenize": tokenize,
            "add_generation_prompt": add_generation_prompt,
        }
        return "rendered_prompt"

    def __call__(self, **kwargs: object) -> FakeBatch:
        self.processor_kwargs = kwargs
        return FakeBatch({"input_ids": FakeTensor((1, 8))})

    def batch_decode(self, generated_ids: object, **kwargs: object) -> list[str]:
        self.decode_kwargs = kwargs
        return [
            "<think>\nmocked runtime\n</think>\n"
            "<answer>\n"
            "  <anomaly_present>false</anomaly_present>\n"
            "  <top_anomaly>null</top_anomaly>\n"
            "  <visual_descriptions>\n"
            "  </visual_descriptions>\n"
            "</answer>"
        ]


class FakeModel:
    """Minimal causal generation stub."""

    last_from_pretrained: tuple[str, dict[str, object]] | None = None

    @classmethod
    def from_pretrained(cls, model_path: str, **kwargs: object) -> "FakeModel":
        cls.last_from_pretrained = (model_path, kwargs)
        return cls()

    def eval(self) -> "FakeModel":
        return self

    def to(self, device: str) -> "FakeModel":
        self.device = device
        return self

    def generate(self, **kwargs: object) -> FakeTensor:
        self.generate_kwargs = kwargs
        return FakeTensor((1, 16))


class FakePeftModel:
    """Minimal PEFT stub that records adapter loading."""

    last_call: tuple[object, str, dict[str, object]] | None = None

    def __init__(self, base_model: object) -> None:
        self.base_model = base_model
        self.model = base_model
        self.peft_config = {"default": types.SimpleNamespace(target_modules=["q_proj", "v_proj"])}

    @classmethod
    def from_pretrained(cls, model: object, adapter_path: str, **kwargs: object) -> object:
        cls.last_call = (model, adapter_path, kwargs)
        return cls(model)

    def active_adapters(self) -> list[str]:
        return ["default"]

    def eval(self) -> "FakePeftModel":
        if hasattr(self.base_model, "eval"):
            self.base_model.eval()
        return self

    def to(self, device: str) -> "FakePeftModel":
        if hasattr(self.base_model, "to"):
            self.base_model.to(device)
        return self

    def generate(self, **kwargs: object) -> FakeTensor:
        return self.base_model.generate(**kwargs)


class Prompt20CheckpointEvalTests(unittest.TestCase):
    """Prompt 2.0 acceptance tests without real local inference."""

    def test_dry_run_accepts_adapter_checkpoint_surface(self) -> None:
        """The baseline entrypoint should accept runtime adapter overrides in dry-run mode."""

        result = dry_run_from_config(
            config_path=REMOTE_PZ_CR_CONFIG,
            dataset_root=FIXTURE_ROOT,
            artifact_root=REPO_ROOT / "dist" / "tmp" / "prompt_2_0_dry_run",
            max_samples=1,
            runtime_overrides={
                "base_model_path": "/models/base",
                "adapter_checkpoint_path": "/checkpoints/run_x/checkpoint-553",
            },
        )
        runtime = result["runtime_provenance"]
        self.assertEqual(runtime["base_model_path"], "/models/base")
        self.assertEqual(runtime["adapter_checkpoint_path"], "/checkpoints/run_x/checkpoint-553")
        self.assertEqual(runtime["checkpoint_step"], 553)
        self.assertEqual(runtime["runtime_backend_type"], "transformers")
        self.assertFalse(runtime["adapter_loaded"])

    def test_transformers_backend_loads_adapter_via_real_main_path(self) -> None:
        """The real transformers backend path should call processor/model/PEFT loaders."""

        fake_transformers = types.SimpleNamespace(
            AutoProcessor=FakeProcessor,
            AutoModelForImageTextToText=FakeModel,
        )
        fake_torch = types.SimpleNamespace(float16="float16", bfloat16="bfloat16", float32="float32")
        fake_peft = types.SimpleNamespace(PeftModel=FakePeftModel)

        backend = TransformersVisionLanguageBackend(
            backend_name="transformers_qwen25_vl_eval_v1",
            policy="hf_chat_vl_local_paths_v1",
            runtime_config={
                "base_model_path": "/models/base",
                "adapter_checkpoint_path": "/checkpoints/run_x/checkpoint-553",
                "local_files_only": True,
                "trust_remote_code": True,
                "dtype": "auto",
                "device": "auto",
                "generation": {
                    "max_new_tokens": 64,
                    "do_sample": False,
                    "temperature": 0.0,
                    "top_p": 1.0
                }
            },
        )
        request = BackendRequest(
            sample_id="sample-001",
            seed=0,
            prompt_version="agentiad_baseline_prompt_v1_3",
            messages=[
                {
                    "role": "system",
                    "message_type": "system_instruction",
                    "content": "System prompt.",
                    "image_refs": [],
                    "metadata": {},
                },
                {
                    "role": "user",
                    "message_type": "user_prompt",
                    "content": "Look at the image.",
                    "image_refs": [str(FIXTURE_IMAGE.resolve())],
                    "metadata": {},
                },
            ],
            stop_sequences=["</answer>"],
            tool_mode="no_tools",
        )

        with mock.patch.dict(
            sys.modules,
            {
                "transformers": fake_transformers,
                "torch": fake_torch,
                "peft": fake_peft,
            },
        ):
            response = backend.generate(request, sample={})

        self.assertIn("<answer>", response.raw_output)
        self.assertEqual(FakeProcessor.last_from_pretrained[0], "/models/base")
        self.assertEqual(FakeModel.last_from_pretrained[0], "/models/base")
        self.assertEqual(FakePeftModel.last_call[1], "/checkpoints/run_x/checkpoint-553")
        runtime = backend.describe_runtime()
        self.assertTrue(runtime["adapter_load_attempted"])
        self.assertTrue(runtime["adapter_loaded"])
        self.assertEqual(runtime["adapter_backend"], "peft.PeftModel.from_pretrained")
        self.assertEqual(runtime["adapter_target_modules"], ["q_proj", "v_proj"])

    def test_fixture_baseline_summary_and_manifest_include_runtime_provenance(self) -> None:
        """Fixture-backed runs should export provenance fields without loading a real model."""

        with tempfile.TemporaryDirectory() as tempdir:
            result = run_baseline(
                config_path=BASELINE_CONFIG,
                dataset_root=FIXTURE_ROOT,
                artifact_root=tempdir,
                max_samples=2,
                runtime_overrides={
                    "base_model_path": "/models/base",
                    "adapter_checkpoint_path": "/checkpoints/run_x/checkpoint-553",
                },
            )
            summary = json.loads(Path(result["summary_path"]).read_text(encoding="utf-8"))
            manifest = json.loads(Path(result["run_manifest_path"]).read_text(encoding="utf-8"))
            metrics = result["metrics_report"]

        self.assertEqual(summary["runtime_provenance"]["base_model_path"], "/models/base")
        self.assertEqual(summary["runtime_provenance"]["adapter_checkpoint_path"], "/checkpoints/run_x/checkpoint-553")
        self.assertFalse(summary["runtime_provenance"]["adapter_loaded"])
        self.assertIn("run_provenance", manifest)
        self.assertIn("runtime_provenance", metrics)
        self.assertEqual(summary["failed_count"], 1)
        self.assertEqual(metrics["sample_count"], 4)


if __name__ == "__main__":
    unittest.main()
