"""Runtime hygiene tests for Prompt 2.1.

These tests verify that the real transformers checkpoint-eval backend moves all
tensor inputs onto the model device and drops ignored sampling kwargs when
deterministic generation is requested. They remain fully local and do not load
real model weights.
"""

from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from agentiad_recon.backends import BackendRequest, TransformersVisionLanguageBackend


FIXTURE_IMAGE = REPO_ROOT / "tests" / "fixtures" / "mmad_fixture" / "train" / "capsule" / "good" / "images" / "capsule_good_0001.ppm"


class FakeTensor:
    """Small tensor stand-in that records device moves."""

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


class FakeProcessor:
    """Minimal processor stub for generate-path verification."""

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

    def __call__(self, **kwargs: object) -> dict[str, object]:
        return {
            "input_ids": FakeTensor((1, 8)),
            "pixel_values": FakeTensor((1, 3, 32, 32)),
        }

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
    """Minimal model stub with an explicit CUDA device."""

    @classmethod
    def from_pretrained(cls, model_path: str, **kwargs: object) -> "FakeModel":
        return cls()

    def __init__(self) -> None:
        self.device = "cuda:0"
        self.generate_kwargs: dict[str, object] | None = None

    def eval(self) -> "FakeModel":
        return self

    def generate(self, **kwargs: object) -> FakeTensor:
        self.generate_kwargs = kwargs
        return FakeTensor((1, 12), device="cuda:0")


class Prompt21RuntimeHygieneTests(unittest.TestCase):
    """Prompt 2.1 lightweight runtime-hygiene tests."""

    def setUp(self) -> None:
        self.backend = TransformersVisionLanguageBackend(
            backend_name="transformers_qwen25_vl_eval_v1",
            policy="hf_chat_vl_local_paths_v1",
            runtime_config={
                "base_model_path": "/models/base",
                "adapter_checkpoint_path": None,
                "device": "auto",
                "dtype": "auto",
                "generation": {
                    "max_new_tokens": 64,
                    "do_sample": False,
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "top_k": 50,
                },
            },
        )
        self.backend._torch = types.SimpleNamespace(
            is_tensor=lambda value: isinstance(value, FakeTensor),
            device=lambda value: value,
        )

    def test_sanitize_generation_kwargs_drops_sampling_only_fields(self) -> None:
        """Deterministic generation should not pass ignored sampling kwargs."""

        sanitized = self.backend._sanitize_generation_kwargs(
            {
                "max_new_tokens": 64,
                "do_sample": False,
                "temperature": 0.0,
                "top_p": 1.0,
                "top_k": 50,
            }
        )
        self.assertEqual(sanitized["max_new_tokens"], 64)
        self.assertFalse(sanitized["do_sample"])
        self.assertNotIn("temperature", sanitized)
        self.assertNotIn("top_p", sanitized)
        self.assertNotIn("top_k", sanitized)

    def test_move_batch_to_device_moves_tensors_and_preserves_non_tensors(self) -> None:
        """Nested tensor inputs should move while metadata stays untouched."""

        batch = {
            "input_ids": FakeTensor((1, 8)),
            "pixel_values": FakeTensor((1, 3, 32, 32)),
            "nested": {
                "list": [FakeTensor((1, 2)), "keep_me"],
            },
            "metadata": {"source": "fixture"},
        }
        moved = self.backend._move_batch_to_device(batch, "cuda:0")

        self.assertEqual(moved["input_ids"].moved_to, "cuda:0")
        self.assertEqual(moved["pixel_values"].moved_to, "cuda:0")
        self.assertEqual(moved["nested"]["list"][0].moved_to, "cuda:0")
        self.assertEqual(moved["nested"]["list"][1], "keep_me")
        self.assertEqual(moved["metadata"]["source"], "fixture")

    def test_generate_moves_inputs_to_model_device_and_uses_sanitized_kwargs(self) -> None:
        """The live generate path should move tensors and avoid ignored kwargs."""

        fake_transformers = types.SimpleNamespace(
            AutoProcessor=FakeProcessor,
            AutoModelForImageTextToText=FakeModel,
        )
        fake_torch = types.SimpleNamespace(
            is_tensor=lambda value: isinstance(value, FakeTensor),
            device=lambda value: value,
            float16="float16",
            bfloat16="bfloat16",
            float32="float32",
        )
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

        with mock.patch.dict(sys.modules, {"transformers": fake_transformers, "torch": fake_torch}):
            response = self.backend.generate(request, sample={})

        generate_kwargs = self.backend._model.generate_kwargs
        self.assertIn("<answer>", response.raw_output)
        self.assertEqual(generate_kwargs["input_ids"].moved_to, "cuda:0")
        self.assertEqual(generate_kwargs["pixel_values"].moved_to, "cuda:0")
        self.assertEqual(generate_kwargs["max_new_tokens"], 64)
        self.assertFalse(generate_kwargs["do_sample"])
        self.assertNotIn("temperature", generate_kwargs)
        self.assertNotIn("top_p", generate_kwargs)
        self.assertNotIn("top_k", generate_kwargs)


if __name__ == "__main__":
    unittest.main()
