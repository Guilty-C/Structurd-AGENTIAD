"""Prompt 2.15 tests for auditable PEFT adapter loading in the unified evaluator.

These tests keep `baseline.py` as the only inference/evaluation entrypoint.
They use mocked `transformers` and `peft` modules so local validation proves
adapter-load semantics and provenance without running heavy inference.
"""

from __future__ import annotations

import copy
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from agentiad_recon.baseline import InferenceRunError, run_baseline


FIXTURE_ROOT = REPO_ROOT / "tests" / "fixtures" / "mmad_fixture"
REMOTE_NO_TOOLS_CONFIG = REPO_ROOT / "configs" / "eval_transformers_no_tools_remote_template.json"


class FakeTensor:
    """Tiny tensor stand-in that satisfies the backend generation path."""

    def __init__(self, shape: tuple[int, ...], *, device: str = "cpu") -> None:
        self.shape = shape
        self.device = device

    def to(self, device: object) -> "FakeTensor":
        self.device = str(device)
        return self

    def __getitem__(self, item: object) -> "FakeTensor":
        return self


class FakeBatch(dict):
    """Dict-like batch object with a `.to(...)` helper."""

    def to(self, device: object) -> "FakeBatch":
        rendered = str(device)
        for value in self.values():
            if hasattr(value, "to"):
                value.to(rendered)
        return self


class FakeProcessor:
    """Minimal processor stub for end-to-end evaluator tests."""

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
        return "rendered_prompt"

    def __call__(self, **kwargs: object) -> FakeBatch:
        return FakeBatch({"input_ids": FakeTensor((1, 8)), "attention_mask": FakeTensor((1, 8))})

    def batch_decode(self, generated_ids: object, **kwargs: object) -> list[str]:
        return [
            "<answer>\n"
            "  <anomaly_present>false</anomaly_present>\n"
            "  <top_anomaly>null</top_anomaly>\n"
            "  <visual_descriptions>\n"
            "  </visual_descriptions>\n"
            "</answer>"
        ]


class FakeModel:
    """Minimal generation-capable base model."""

    last_from_pretrained: tuple[str, dict[str, object]] | None = None

    @classmethod
    def from_pretrained(cls, model_path: str, **kwargs: object) -> "FakeModel":
        cls.last_from_pretrained = (model_path, kwargs)
        return cls()

    def __init__(self) -> None:
        self.device = "cpu"
        self.generate_kwargs: dict[str, object] | None = None

    def parameters(self):
        yield types.SimpleNamespace(device=self.device)

    def eval(self) -> "FakeModel":
        return self

    def to(self, device: str) -> "FakeModel":
        self.device = device
        return self

    def generate(self, **kwargs: object) -> FakeTensor:
        self.generate_kwargs = kwargs
        return FakeTensor((1, 16), device=self.device)


class FakePeftModel:
    """PEFT loader stub that can either attach or fail loudly."""

    last_call: tuple[object, str, dict[str, object]] | None = None

    def __init__(self, base_model: FakeModel) -> None:
        self.base_model = base_model
        self.model = base_model
        self.peft_config = {"default": types.SimpleNamespace(target_modules=["q_proj", "v_proj"])}

    @classmethod
    def from_pretrained(cls, model: FakeModel, adapter_path: str, **kwargs: object) -> "FakePeftModel":
        cls.last_call = (model, adapter_path, kwargs)
        if "missing" in adapter_path:
            raise FileNotFoundError(f"adapter checkpoint not found: {adapter_path}")
        return cls(model)

    def active_adapters(self) -> list[str]:
        return ["default"]

    def parameters(self):
        yield from self.base_model.parameters()

    def eval(self) -> "FakePeftModel":
        self.base_model.eval()
        return self

    def to(self, device: str) -> "FakePeftModel":
        self.base_model.to(device)
        return self

    def generate(self, **kwargs: object) -> FakeTensor:
        return self.base_model.generate(**kwargs)


class Prompt215AdapterLoadingTests(unittest.TestCase):
    """Prompt 2.15 regression coverage."""

    def _fixture_transformers_config(self, root: Path) -> Path:
        """Project the remote transformers config onto the local tiny fixture split."""

        definition = json.loads(REMOTE_NO_TOOLS_CONFIG.read_text(encoding="utf-8"))
        definition = copy.deepcopy(definition)
        definition["sample_source"]["kind"] = "fixture"
        definition["sample_source"]["path"] = str(FIXTURE_ROOT)
        definition["sample_source"]["splits"] = ["train"]
        definition["artifacts"]["root"] = str(root / "artifacts")
        config_path = root / "fixture_transformers_no_tools.json"
        config_path.write_text(json.dumps(definition, indent=2), encoding="utf-8")
        return config_path

    def _patched_runtime_modules(self) -> dict[str, object]:
        fake_transformers = types.SimpleNamespace(
            AutoProcessor=FakeProcessor,
            AutoModelForImageTextToText=FakeModel,
        )
        fake_torch = types.SimpleNamespace(
            float16="float16",
            bfloat16="bfloat16",
            float32="float32",
            device=lambda value: value,
            is_tensor=lambda value: isinstance(value, FakeTensor),
        )
        fake_peft = types.SimpleNamespace(PeftModel=FakePeftModel)
        return {"transformers": fake_transformers, "torch": fake_torch, "peft": fake_peft}

    def test_valid_adapter_path_marks_runtime_provenance_loaded(self) -> None:
        """A successful PEFT attach should be reflected by the unified evaluator artifacts."""

        with tempfile.TemporaryDirectory() as tempdir, mock.patch.dict(
            sys.modules,
            self._patched_runtime_modules(),
        ):
            config_path = self._fixture_transformers_config(Path(tempdir))
            result = run_baseline(
                config_path=config_path,
                dataset_root=FIXTURE_ROOT,
                artifact_root=tempdir,
                max_samples=2,
                runtime_overrides={
                    "base_model_path": "/models/base",
                    "adapter_checkpoint_path": "/checkpoints/run_x/checkpoint-553",
                    "num_shards": 2,
                    "shard_index": 1,
                    "emit_baseline_compare": False,
                    "emit_delta_report": False,
                },
            )
            summary = json.loads(Path(result["summary_path"]).read_text(encoding="utf-8"))
            manifest = json.loads(Path(result["run_manifest_path"]).read_text(encoding="utf-8"))

        runtime = summary["runtime_provenance"]
        self.assertTrue(runtime["adapter_load_attempted"])
        self.assertTrue(runtime["adapter_loaded"])
        self.assertEqual(runtime["adapter_backend"], "peft.PeftModel.from_pretrained")
        self.assertIsNone(runtime["adapter_load_error"])
        self.assertEqual(runtime["adapter_target_modules"], ["q_proj", "v_proj"])
        self.assertEqual(runtime["sharding_strategy"], "stable_index_mod")
        self.assertEqual(runtime["selected_sample_count"], 1)
        self.assertTrue(result["runtime_provenance"]["adapter_loaded"])
        self.assertTrue(manifest["run_provenance"]["adapter_loaded"])

    def test_invalid_adapter_path_fails_fast_by_default(self) -> None:
        """Explicit adapter requests should fail non-zero instead of silently degrading."""

        with mock.patch.dict(sys.modules, self._patched_runtime_modules()):
            with tempfile.TemporaryDirectory() as tempdir:
                config_path = self._fixture_transformers_config(Path(tempdir))
                with self.assertRaisesRegex(
                    InferenceRunError,
                    "Failed to load PEFT adapter from /missing/adapter",
                ):
                    run_baseline(
                        config_path=config_path,
                        dataset_root=FIXTURE_ROOT,
                        artifact_root=REPO_ROOT / "dist" / "tmp" / "prompt_2_15_missing_adapter",
                        max_samples=1,
                        runtime_overrides={
                            "base_model_path": "/models/base",
                            "adapter_checkpoint_path": "/missing/adapter",
                            "emit_baseline_compare": False,
                            "emit_delta_report": False,
                        },
                    )

    def test_allow_missing_adapter_records_explicit_error_provenance(self) -> None:
        """The optional fallback mode should stay explicit in written provenance."""

        with tempfile.TemporaryDirectory() as tempdir, mock.patch.dict(
            sys.modules,
            self._patched_runtime_modules(),
        ):
            config_path = self._fixture_transformers_config(Path(tempdir))
            result = run_baseline(
                config_path=config_path,
                dataset_root=FIXTURE_ROOT,
                artifact_root=tempdir,
                max_samples=1,
                runtime_overrides={
                    "base_model_path": "/models/base",
                    "adapter_checkpoint_path": "/missing/adapter",
                    "allow_missing_adapter": True,
                    "emit_baseline_compare": False,
                    "emit_delta_report": False,
                },
            )
            summary = json.loads(Path(result["summary_path"]).read_text(encoding="utf-8"))

        runtime = summary["runtime_provenance"]
        self.assertTrue(runtime["allow_missing_adapter"])
        self.assertTrue(runtime["adapter_load_attempted"])
        self.assertFalse(runtime["adapter_loaded"])
        self.assertEqual(runtime["adapter_load_error_type"], "FileNotFoundError")
        self.assertIn("/missing/adapter", runtime["adapter_load_error"])
        self.assertGreater(summary["sample_count"], 0)


if __name__ == "__main__":
    unittest.main()
