"""Prompt 2.2 tests for the remaining real generate-input device mismatch fix."""

from __future__ import annotations

import sys
import types
import unittest
from collections.abc import Mapping
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from agentiad_recon.backends import BackendRequest, TransformersVisionLanguageBackend


FIXTURE_IMAGE = REPO_ROOT / "tests" / "fixtures" / "mmad_fixture" / "train" / "capsule" / "good" / "images" / "capsule_good_0001.ppm"


class FakeTensor:
    """Tensor stand-in with a moveable device."""

    def __init__(self, shape: tuple[int, ...], *, device: str = "cpu") -> None:
        self.shape = shape
        self.device = device
        self.moved_to: str | None = None

    def to(self, device: object) -> "FakeTensor":
        rendered = str(device)
        self.device = rendered
        self.moved_to = rendered
        return self

    def __getitem__(self, item: object) -> "FakeTensor":
        return self


class FakeParameter:
    """Parameter stand-in with a fixed device."""

    def __init__(self, device: str) -> None:
        self.device = device


class FakeEmbeddingModule:
    """Embedding-like module exposing `.weight.device`."""

    def __init__(self, device: str) -> None:
        self.weight = types.SimpleNamespace(device=device)

    def parameters(self):
        yield FakeParameter(self.weight.device)


class FakeLanguageModel:
    """Language model carrying the actual text-input embedding device."""

    def __init__(self, device: str) -> None:
        self.embed_tokens = FakeEmbeddingModule(device)

    def get_input_embeddings(self) -> FakeEmbeddingModule:
        return self.embed_tokens


class FakeBaseModel:
    """Base model whose top-level device is misleading for text input."""

    @classmethod
    def from_pretrained(cls, model_path: str, **kwargs: object) -> "FakeBaseModel":
        return cls()

    def __init__(self) -> None:
        self.device = "cpu"
        self.language_model = FakeLanguageModel("cuda:0")
        self.hf_device_map = {"": "cpu"}
        self.generate_kwargs: dict[str, object] | None = None

    def parameters(self):
        yield FakeParameter("cpu")

    def eval(self) -> "FakeBaseModel":
        return self

    def generate(self, **kwargs: object) -> FakeTensor:
        self.generate_kwargs = kwargs
        return FakeTensor((1, 12), device="cuda:0")


class FakePeftWrappedModel:
    """PEFT-style wrapper that still exposes a misleading top-level device."""

    def __init__(self, base_model: FakeBaseModel) -> None:
        self.base_model = base_model
        self.model = base_model
        self.language_model = base_model.language_model
        self.device = "cpu"
        self.hf_device_map = {"": "cpu"}
        self.peft_config = {"default": types.SimpleNamespace(target_modules=["q_proj", "v_proj"])}
        self.generate_kwargs: dict[str, object] | None = None

    def parameters(self):
        yield FakeParameter("cpu")

    def active_adapters(self) -> list[str]:
        return ["default"]

    def eval(self) -> "FakePeftWrappedModel":
        return self

    def generate(self, **kwargs: object) -> FakeTensor:
        self.generate_kwargs = kwargs
        return FakeTensor((1, 14), device="cuda:0")


class FakePeftModel:
    """PEFT adapter loader stub returning the wrapped model."""

    @classmethod
    def from_pretrained(cls, model: FakeBaseModel, adapter_path: str, **kwargs: object) -> FakePeftWrappedModel:
        return FakePeftWrappedModel(model)


class FakeBatchEncoding(Mapping):
    """BatchEncoding-like mapping with nested tensor payloads."""

    def __init__(self) -> None:
        self.payload = {
            "input_ids": FakeTensor((1, 8), device="cpu"),
            "pixel_values": FakeTensor((1, 3, 32, 32), device="cpu"),
            "nested": {"attention_mask": FakeTensor((1, 8), device="cpu")},
            "meta": "keep_me",
        }

    def __getitem__(self, key: str):
        return self.payload[key]

    def __iter__(self):
        return iter(self.payload)

    def __len__(self) -> int:
        return len(self.payload)

    def items(self):
        return self.payload.items()


class FakeProcessor:
    """Processor stub returning a BatchEncoding-like mapping."""

    @classmethod
    def from_pretrained(cls, model_path: str, **kwargs: object) -> "FakeProcessor":
        return cls()

    def apply_chat_template(
        self,
        chat_messages: list[dict[str, object]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str:
        return "rendered_prompt"

    def __call__(self, **kwargs: object) -> FakeBatchEncoding:
        return FakeBatchEncoding()

    def batch_decode(self, generated_ids: object, **kwargs: object) -> list[str]:
        return [
            "<answer>\n"
            "  <anomaly_present>false</anomaly_present>\n"
            "  <top_anomaly>null</top_anomaly>\n"
            "  <visual_descriptions>\n"
            "  </visual_descriptions>\n"
            "</answer>"
        ]


class Prompt22RuntimeDeviceFixTests(unittest.TestCase):
    """Prompt 2.2 runtime-device tests."""

    def setUp(self) -> None:
        self.backend = TransformersVisionLanguageBackend(
            backend_name="transformers_qwen25_vl_eval_v1",
            policy="hf_chat_vl_local_paths_v1",
            runtime_config={
                "base_model_path": "/models/base",
                "adapter_checkpoint_path": "/checkpoints/run_x/checkpoint-553",
                "device": "auto",
                "dtype": "auto",
                "generation": {
                    "max_new_tokens": 64,
                    "do_sample": False,
                    "temperature": 0.0,
                    "top_p": 1.0,
                },
            },
        )
        self.backend._torch = types.SimpleNamespace(
            is_tensor=lambda value: isinstance(value, FakeTensor),
            device=lambda value: value,
            float16="float16",
            bfloat16="bfloat16",
            float32="float32",
        )

    def test_infer_model_device_prefers_language_input_path_over_wrapper_device(self) -> None:
        """PEFT-style wrappers should resolve the language embedding device first."""

        wrapped = FakePeftWrappedModel(FakeBaseModel())
        inferred = self.backend._infer_model_device(wrapped)
        self.assertEqual(inferred, "cuda:0")

    def test_move_batch_to_device_handles_batchencoding_like_mapping(self) -> None:
        """BatchEncoding-like mappings should recursively move all tensor payloads."""

        moved = self.backend._move_batch_to_device(FakeBatchEncoding(), "cuda:0")
        self.assertEqual(moved["input_ids"].device, "cuda:0")
        self.assertEqual(moved["pixel_values"].device, "cuda:0")
        self.assertEqual(moved["nested"]["attention_mask"].device, "cuda:0")
        self.assertEqual(moved["meta"], "keep_me")

    def test_generate_passes_cuda_input_ids_into_real_generate_call(self) -> None:
        """The actual generate() call should receive input_ids on the inferred target device."""

        fake_transformers = types.SimpleNamespace(
            AutoProcessor=FakeProcessor,
            AutoModelForImageTextToText=FakeBaseModel,
        )
        fake_peft = types.SimpleNamespace(PeftModel=FakePeftModel)
        request = BackendRequest(
            sample_id="sample-001",
            seed=0,
            prompt_version="agentiad_baseline_prompt_v1_3",
            messages=[
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
            {"transformers": fake_transformers, "torch": self.backend._torch, "peft": fake_peft},
        ):
            response = self.backend.generate(request, sample={})

        generate_kwargs = self.backend._model.generate_kwargs
        self.assertIn("<answer>", response.raw_output)
        self.assertEqual(generate_kwargs["input_ids"].device, "cuda:0")
        self.assertEqual(generate_kwargs["pixel_values"].device, "cuda:0")
        self.assertEqual(generate_kwargs["nested"]["attention_mask"].device, "cuda:0")
        self.assertFalse(generate_kwargs["do_sample"])
        self.assertNotIn("temperature", generate_kwargs)
        self.assertNotIn("top_p", generate_kwargs)


if __name__ == "__main__":
    unittest.main()
