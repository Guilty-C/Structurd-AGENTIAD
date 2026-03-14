"""Thin inference backend interfaces for AgentIAD inference runs.

This module keeps runtime ownership thin and explicit across the non-tool
baseline and tool-augmented paths. Prompt 1.4 extends the local smoke layer
with scripted tool-aware backend policies while preserving a future maintained
runtime adapter skeleton for remote execution. There is no standalone CLI for
this module.
"""

from __future__ import annotations

import importlib
import json
from collections.abc import Mapping
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image

from agentiad_recon.prompting import render_answer_block


class BackendError(RuntimeError):
    """Raised when a backend cannot satisfy one inference request."""


@dataclass(frozen=True)
class BackendRequest:
    """Canonical backend request shared by mock and maintained-runtime adapters."""

    sample_id: str
    seed: int
    prompt_version: str
    messages: list[dict[str, Any]]
    stop_sequences: list[str]
    tool_mode: str = "no_tools"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BackendResponse:
    """Captured raw backend response plus audit metadata."""

    backend_name: str
    raw_output: str
    metadata: dict[str, Any]


class InferenceBackend(ABC):
    """Abstract interface for inference backends used by the canonical runner."""

    backend_name: str

    def describe_runtime(self) -> dict[str, Any]:
        """Return audit metadata about the backend runtime configuration."""

        return {}

    @abstractmethod
    def generate(self, request: BackendRequest, *, sample: dict[str, Any]) -> BackendResponse:
        """Generate one raw model response for a canonical sample."""


def _default_generation_config() -> dict[str, Any]:
    """Return deterministic generation defaults for auditable eval runs."""

    return {
        "max_new_tokens": 512,
        "do_sample": False,
        "temperature": 0.0,
        "top_p": 1.0,
    }


def _normalize_runtime_config(runtime_config: dict[str, Any] | None) -> dict[str, Any]:
    """Merge optional runtime config onto stable backend defaults."""

    config = dict(runtime_config or {})
    generation = dict(_default_generation_config())
    generation.update(config.get("generation", {}))
    config["generation"] = generation
    config.setdefault("base_model_path", None)
    config.setdefault("adapter_checkpoint_path", None)
    config.setdefault("checkpoint_step", None)
    config.setdefault("checkpoint_run_dir", None)
    config.setdefault("local_files_only", True)
    config.setdefault("trust_remote_code", True)
    config.setdefault("dtype", "auto")
    config.setdefault("device", "auto")
    return config


class MockInferenceBackend(InferenceBackend):
    """Deterministic fixture-only backend for local non-tool smoke validation.

    The mock backend is intentionally explicit about its policy so no caller can
    mistake it for real model inference. It uses canonical sample metadata and
    ground truth only to exercise the parsing and evaluation plumbing.
    """

    def __init__(self, *, backend_name: str, policy: str, runtime_config: dict[str, Any] | None = None) -> None:
        self.backend_name = backend_name
        self.policy = policy
        self.runtime_config = _normalize_runtime_config(runtime_config)

    def describe_runtime(self) -> dict[str, Any]:
        """Expose the requested runtime surface without claiming real loading."""

        return {
            "backend_type": "mock",
            "runtime_owner": "mock_fixture_only",
            "policy": self.policy,
            "base_model_path": self.runtime_config["base_model_path"],
            "adapter_checkpoint_path": self.runtime_config["adapter_checkpoint_path"],
            "adapter_loaded": False,
            "checkpoint_step": self.runtime_config["checkpoint_step"],
            "checkpoint_run_dir": self.runtime_config["checkpoint_run_dir"],
            "generation_config": dict(self.runtime_config["generation"]),
            "local_files_only": self.runtime_config["local_files_only"],
            "trust_remote_code": self.runtime_config["trust_remote_code"],
            "dtype": self.runtime_config["dtype"],
            "device": self.runtime_config["device"],
        }

    def _fixture_scripted_output(self, request: BackendRequest, sample: dict[str, Any]) -> str:
        """Create deterministic baseline outputs, including one malformed branch."""

        anomaly_label = sample["ground_truth"]["top_anomaly"] or "none"
        if sample["anomaly_present"]:
            answer_payload = {
                "anomaly_present": True,
                "top_anomaly": anomaly_label,
                "visual_descriptions": [
                    f"Visible evidence is consistent with {anomaly_label} on the {sample['category']} sample."
                ],
            }
        else:
            answer_payload = {
                "anomaly_present": False,
                "top_anomaly": None,
                "visual_descriptions": [],
            }

        # Odd seeds intentionally emit one malformed anomaly answer so the
        # baseline gate can verify failure handling and validity counts.
        if request.seed % 2 == 1 and sample["anomaly_present"]:
            return (
                "<think>\nMock backend emits one malformed anomaly output for gate testing.\n</think>\n"
                "<answer>\n"
                "  <anomaly_present>maybe</anomaly_present>\n"
                f"  <top_anomaly>{anomaly_label}</top_anomaly>\n"
                "  <visual_descriptions></visual_descriptions>\n"
                "</answer>"
            )

        return render_answer_block(
            answer_payload,
            wrapper_tag="answer",
            think=(
                "Mock backend response for local-only baseline plumbing validation. "
                "This is not real model inference."
            ),
        )

    def generate(self, request: BackendRequest, *, sample: dict[str, Any]) -> BackendResponse:
        """Return a deterministic raw output for fixture-backed baseline runs."""

        if self.policy != "fixture_scripted_non_tool_v1":
            raise BackendError(f"Unsupported mock backend policy: {self.policy}")
        raw_output = self._fixture_scripted_output(request, sample)
        return BackendResponse(
            backend_name=self.backend_name,
            raw_output=raw_output,
            metadata={
                "policy": self.policy,
                "runtime_owner": "mock_fixture_only",
                "sample_kind": sample["metadata"].get("dataset_kind", "unknown"),
            },
        )


class MockToolAwareBackend(InferenceBackend):
    """Deterministic scripted backend for local tool-loop smoke validation.

    This backend is not a model. It emits a reproducible sequence of tool calls
    and final answers so the runner can validate loop control, tool execution,
    trace storage, and delta-vs-baseline artifacts without any heavy inference.
    """

    def __init__(self, *, backend_name: str, policy: str, runtime_config: dict[str, Any] | None = None) -> None:
        self.backend_name = backend_name
        self.policy = policy
        self.runtime_config = _normalize_runtime_config(runtime_config)

    def describe_runtime(self) -> dict[str, Any]:
        """Expose the requested runtime surface without claiming real loading."""

        return {
            "backend_type": "mock",
            "runtime_owner": "mock_fixture_only",
            "policy": self.policy,
            "base_model_path": self.runtime_config["base_model_path"],
            "adapter_checkpoint_path": self.runtime_config["adapter_checkpoint_path"],
            "adapter_loaded": False,
            "checkpoint_step": self.runtime_config["checkpoint_step"],
            "checkpoint_run_dir": self.runtime_config["checkpoint_run_dir"],
            "generation_config": dict(self.runtime_config["generation"]),
            "local_files_only": self.runtime_config["local_files_only"],
            "trust_remote_code": self.runtime_config["trust_remote_code"],
            "dtype": self.runtime_config["dtype"],
            "device": self.runtime_config["device"],
        }

    def _tool_result_names(self, request: BackendRequest) -> list[str]:
        """Inspect the history and recover the ordered tool-result names."""

        return [
            message["tool_name"]
            for message in request.messages
            if message.get("message_type") == "tool_result" and message.get("tool_name") is not None
        ]

    def _tool_call_block(self, tool_name: str, arguments: dict[str, Any], *, think: str) -> str:
        """Render one assistant tool-call block with optional reasoning text."""

        payload = json.dumps({"tool_name": tool_name, "arguments": arguments}, sort_keys=True)
        return f"<think>\n{think}\n</think>\n<tool_call>\n{payload}\n</tool_call>"

    def _final_tool_answer(self, sample: dict[str, Any], *, think: str) -> str:
        """Render the final answer after the scripted tool sequence completes."""

        anomaly_label = sample["ground_truth"]["top_anomaly"]
        if sample["anomaly_present"]:
            answer_payload = {
                "anomaly_present": True,
                "top_anomaly": anomaly_label,
                "visual_descriptions": [
                    f"Structured mock answer after tool use confirms {anomaly_label} on the {sample['category']} sample."
                ],
            }
        else:
            answer_payload = {
                "anomaly_present": False,
                "top_anomaly": None,
                "visual_descriptions": [],
            }
        return render_answer_block(answer_payload, wrapper_tag="answer", think=think)

    def _scripted_pz_only(self, request: BackendRequest, sample: dict[str, Any]) -> str:
        """Emit PZ then final answer for the `pz_only` smoke path."""

        tool_results = self._tool_result_names(request)
        if not tool_results:
            return self._tool_call_block(
                "PZ",
                {"bbox": {"x0": 0.25, "y0": 0.25, "x1": 0.75, "y1": 0.75}},
                think="Need one localized crop before deciding.",
            )
        return self._final_tool_answer(
            sample,
            think="The crop result is sufficient; no comparative retrieval is allowed in pz_only.",
        )

    def _scripted_pz_cr(self, request: BackendRequest, sample: dict[str, Any]) -> str:
        """Emit PZ, then CR, then the final answer for comparative mode."""

        tool_results = self._tool_result_names(request)
        if not tool_results:
            return self._tool_call_block(
                "PZ",
                {"bbox": {"x0": 0.20, "y0": 0.20, "x1": 0.80, "y1": 0.80}},
                think="Need a localized crop before comparing against a normal reference.",
            )
        if tool_results == ["PZ"]:
            return self._tool_call_block(
                "CR",
                {"policy": "same_category_normal"},
                think="The crop is available; now retrieve a same-category normal exemplar.",
            )
        return self._final_tool_answer(
            sample,
            think="The crop and the normal reference are both available; finalize the structured answer.",
        )

    def generate(self, request: BackendRequest, *, sample: dict[str, Any]) -> BackendResponse:
        """Return deterministic scripted tool-aware outputs for smoke runs."""

        if self.policy == "fixture_scripted_pz_only_v1":
            raw_output = self._scripted_pz_only(request, sample)
        elif self.policy == "fixture_scripted_pz_cr_v1":
            raw_output = self._scripted_pz_cr(request, sample)
        elif self.policy == "fixture_scripted_pz_only_invalid_cr_v1":
            raw_output = self._tool_call_block(
                "CR",
                {"policy": "same_category_normal"},
                think="This intentionally violates the pz_only policy for gate testing.",
            )
        elif self.policy == "fixture_scripted_malformed_tool_call_v1":
            raw_output = (
                "<think>\nMalformed tool payload for gate testing.\n</think>\n"
                "<tool_call>\n"
                "{\"tool_name\":\"PZ\",\"arguments\":{\"bbox\":{\"x0\":0.1,\"y0\":0.1,\"x1\":oops}}}\n"
                "</tool_call>"
            )
        else:
            raise BackendError(f"Unsupported scripted tool backend policy: {self.policy}")

        return BackendResponse(
            backend_name=self.backend_name,
            raw_output=raw_output,
            metadata={
                "policy": self.policy,
                "runtime_owner": "mock_fixture_only",
                "sample_kind": sample["metadata"].get("dataset_kind", "unknown"),
                "tool_mode": request.tool_mode,
            },
        )


class VLLMBackendAdapter(InferenceBackend):
    """Thin maintained-runtime adapter skeleton for future remote execution.

    This adapter exists to keep ownership with a maintained inference runtime
    instead of growing a custom serving framework in-repo. Prompt 1.4 still
    does not execute it locally.
    """

    def __init__(
        self,
        *,
        backend_name: str,
        model: str | None = None,
        runtime_config: dict[str, Any] | None = None,
    ) -> None:
        self.backend_name = backend_name
        self.model = model
        self.runtime_config = _normalize_runtime_config(runtime_config)

    def describe_runtime(self) -> dict[str, Any]:
        """Report the requested vLLM runtime surface honestly."""

        return {
            "backend_type": "vllm",
            "runtime_owner": "maintained_runtime_skeleton",
            "policy": "skeleton",
            "base_model_path": self.runtime_config["base_model_path"],
            "adapter_checkpoint_path": self.runtime_config["adapter_checkpoint_path"],
            "adapter_loaded": False,
            "checkpoint_step": self.runtime_config["checkpoint_step"],
            "checkpoint_run_dir": self.runtime_config["checkpoint_run_dir"],
            "generation_config": dict(self.runtime_config["generation"]),
            "local_files_only": self.runtime_config["local_files_only"],
            "trust_remote_code": self.runtime_config["trust_remote_code"],
            "dtype": self.runtime_config["dtype"],
            "device": self.runtime_config["device"],
        }

    def prepare_payload(self, request: BackendRequest) -> dict[str, Any]:
        """Translate the canonical request into a future vLLM request payload."""

        return {
            "model": self.model,
            "messages": request.messages,
            "stop": request.stop_sequences,
            "seed": request.seed,
            "metadata": request.metadata,
        }

    def generate(self, request: BackendRequest, *, sample: dict[str, Any]) -> BackendResponse:
        """Block local use until a remote/server prompt owns real execution."""

        raise BackendError(
            "The vLLM backend adapter is a maintained-runtime skeleton only. "
            "Prompt 1.4 remains local-only and should use the scripted mock backends."
        )


class TransformersVisionLanguageBackend(InferenceBackend):
    """Minimal maintained-runtime adapter for local-path transformers + PEFT eval."""

    def __init__(
        self,
        *,
        backend_name: str,
        policy: str,
        runtime_config: dict[str, Any] | None = None,
    ) -> None:
        self.backend_name = backend_name
        self.policy = policy
        self.runtime_config = _normalize_runtime_config(runtime_config)
        self._processor: Any | None = None
        self._model: Any | None = None
        self._torch: Any | None = None
        self._adapter_loaded = False
        self._model_class_name: str | None = None

    def describe_runtime(self) -> dict[str, Any]:
        """Return the effective runtime state for audit artifacts."""

        return {
            "backend_type": "transformers",
            "runtime_owner": "transformers_peft_runtime",
            "policy": self.policy,
            "base_model_path": self.runtime_config["base_model_path"],
            "adapter_checkpoint_path": self.runtime_config["adapter_checkpoint_path"],
            "adapter_loaded": self._adapter_loaded,
            "checkpoint_step": self.runtime_config["checkpoint_step"],
            "checkpoint_run_dir": self.runtime_config["checkpoint_run_dir"],
            "generation_config": dict(self.runtime_config["generation"]),
            "local_files_only": self.runtime_config["local_files_only"],
            "trust_remote_code": self.runtime_config["trust_remote_code"],
            "dtype": self.runtime_config["dtype"],
            "device": self.runtime_config["device"],
            "model_class_name": self._model_class_name,
        }

    def _resolve_torch_dtype(self, torch_module: Any) -> Any:
        """Translate string dtype settings into torch dtypes when possible."""

        dtype_name = self.runtime_config["dtype"]
        if dtype_name in {None, "", "auto"}:
            return getattr(torch_module, "float16", None) if self.runtime_config["device"] == "cuda" else "auto"
        return getattr(torch_module, dtype_name)

    def _candidate_model_classes(self, transformers_module: Any) -> list[Any]:
        """Return plausible HF model classes for multimodal chat generation."""

        candidates = []
        for name in (
            "Qwen2_5_VLForConditionalGeneration",
            "AutoModelForImageTextToText",
            "AutoModelForVision2Seq",
            "AutoModelForCausalLM",
        ):
            model_class = getattr(transformers_module, name, None)
            if model_class is not None:
                candidates.append(model_class)
        return candidates

    def _load_model(self) -> None:
        """Load the processor and model, then optionally attach a PEFT adapter."""

        if self._model is not None and self._processor is not None:
            return

        base_model_path = self.runtime_config["base_model_path"]
        if not base_model_path:
            raise BackendError("transformers backend requires a base_model_path")

        try:
            transformers_module = importlib.import_module("transformers")
            torch_module = importlib.import_module("torch")
        except Exception as exc:  # noqa: BLE001
            raise BackendError(f"Failed to import transformers runtime dependencies: {exc}") from exc

        self._torch = torch_module
        processor_class = getattr(transformers_module, "AutoProcessor", None)
        if processor_class is None:
            raise BackendError("transformers.AutoProcessor is unavailable in the local runtime")

        processor_kwargs = {
            "trust_remote_code": self.runtime_config["trust_remote_code"],
            "local_files_only": self.runtime_config["local_files_only"],
        }
        self._processor = processor_class.from_pretrained(base_model_path, **processor_kwargs)

        model_kwargs = dict(processor_kwargs)
        torch_dtype = self._resolve_torch_dtype(torch_module)
        if torch_dtype != "auto":
            model_kwargs["torch_dtype"] = torch_dtype
        device = self.runtime_config["device"]
        if device == "auto":
            model_kwargs["device_map"] = "auto"

        last_error: Exception | None = None
        for model_class in self._candidate_model_classes(transformers_module):
            try:
                self._model = model_class.from_pretrained(base_model_path, **model_kwargs)
                self._model_class_name = model_class.__name__
                break
            except Exception as exc:  # noqa: BLE001
                last_error = exc
        if self._model is None:
            raise BackendError(f"Failed to load model from {base_model_path}: {last_error}") from last_error

        adapter_path = self.runtime_config["adapter_checkpoint_path"]
        if adapter_path:
            try:
                peft_module = importlib.import_module("peft")
                peft_model_class = getattr(peft_module, "PeftModel")
                self._model = peft_model_class.from_pretrained(
                    self._model,
                    adapter_path,
                    local_files_only=self.runtime_config["local_files_only"],
                    is_trainable=False,
                )
                self._adapter_loaded = True
            except Exception as exc:  # noqa: BLE001
                raise BackendError(f"Failed to load PEFT adapter from {adapter_path}: {exc}") from exc

        if device != "auto" and hasattr(self._model, "to"):
            self._model = self._model.to(device)
        if hasattr(self._model, "eval"):
            self._model.eval()

    def _torch_device(self, value: Any) -> Any:
        """Create a torch.device when the runtime exposes that constructor."""

        if self._torch is not None and hasattr(self._torch, "device"):
            return self._torch.device(value)
        return value

    def _module_parameter_device(self, module: Any) -> Any | None:
        """Infer one module device from parameters or buffers when available."""

        parameters = getattr(module, "parameters", None)
        if callable(parameters):
            try:
                first_parameter = next(parameters())
                return first_parameter.device
            except StopIteration:
                pass

        buffers = getattr(module, "buffers", None)
        if callable(buffers):
            try:
                first_buffer = next(buffers())
                return first_buffer.device
            except StopIteration:
                pass
        return None

    def _module_hf_device(self, module: Any) -> Any | None:
        """Recover one device from a Hugging Face device map when present."""

        hf_device_map = getattr(module, "hf_device_map", None)
        if isinstance(hf_device_map, dict) and hf_device_map:
            for mapped_device in hf_device_map.values():
                if mapped_device in {"disk", None}:
                    continue
                if isinstance(mapped_device, int):
                    return self._torch_device(f"cuda:{mapped_device}")
                return self._torch_device(mapped_device)
        return None

    def _module_direct_device(self, module: Any) -> Any | None:
        """Return a direct `.device` attribute when the module exposes one."""

        device_attr = getattr(module, "device", None)
        if device_attr is not None:
            return device_attr
        return None

    def _embedding_like_device(self, module: Any) -> Any | None:
        """Prefer the language-input device over wrapper-level device shortcuts."""

        if module is None:
            return None

        get_input_embeddings = getattr(module, "get_input_embeddings", None)
        if callable(get_input_embeddings):
            try:
                input_embeddings = get_input_embeddings()
            except Exception:  # noqa: BLE001
                input_embeddings = None
            if input_embeddings is not None:
                if hasattr(input_embeddings, "weight") and hasattr(input_embeddings.weight, "device"):
                    return input_embeddings.weight.device
                device = self._module_parameter_device(input_embeddings)
                if device is not None:
                    return device

        embed_tokens = getattr(module, "embed_tokens", None)
        if embed_tokens is not None:
            if hasattr(embed_tokens, "weight") and hasattr(embed_tokens.weight, "device"):
                return embed_tokens.weight.device
            device = self._module_parameter_device(embed_tokens)
            if device is not None:
                return device
        return None

    def _candidate_runtime_modules(self, model: Any) -> list[Any]:
        """Collect likely PEFT/base/language submodules in preference order."""

        candidates: list[Any] = []
        queue = [model]
        seen: set[int] = set()
        attribute_names = (
            "language_model",
            "base_model",
            "model",
            "module",
            "backbone",
        )
        while queue:
            current = queue.pop(0)
            if current is None or id(current) in seen:
                continue
            seen.add(id(current))
            candidates.append(current)
            for name in attribute_names:
                child = getattr(current, name, None)
                if child is not None:
                    queue.append(child)
        return candidates

    def _infer_model_device(self, model: Any) -> Any:
        """Infer the effective device that should receive model inputs."""

        candidates = self._candidate_runtime_modules(model)

        for candidate in candidates:
            device = self._embedding_like_device(candidate)
            if device is not None:
                return device

        for candidate in candidates:
            device = self._module_hf_device(candidate)
            if device is not None:
                return device

        for candidate in candidates:
            device = self._module_direct_device(candidate)
            if device is not None:
                return device

        for candidate in candidates:
            device = self._module_parameter_device(candidate)
            if device is not None:
                return device

        configured_device = self.runtime_config["device"]
        if configured_device not in {None, "", "auto"}:
            return self._torch_device(configured_device)
        return self._torch_device("cpu")

    def _move_batch_to_device(self, batch: Any, device: Any) -> Any:
        """Recursively move tensor-like model inputs onto one target device."""

        if self._torch is not None and hasattr(self._torch, "is_tensor") and self._torch.is_tensor(batch):
            return batch.to(device)
        if isinstance(batch, Mapping):
            return {key: self._move_batch_to_device(value, device) for key, value in batch.items()}
        if isinstance(batch, list):
            return [self._move_batch_to_device(value, device) for value in batch]
        if isinstance(batch, tuple):
            return tuple(self._move_batch_to_device(value, device) for value in batch)
        if hasattr(batch, "to") and callable(batch.to):
            try:
                return batch.to(device)
            except TypeError:
                return batch
        return batch

    def _sanitize_generation_kwargs(self, generation_config: dict[str, Any]) -> dict[str, Any]:
        """Drop sampling-only generation kwargs when deterministic generation is requested."""

        sanitized = dict(generation_config)
        if sanitized.get("do_sample", False):
            return sanitized
        for key in (
            "temperature",
            "top_p",
            "top_k",
            "min_p",
            "typical_p",
            "epsilon_cutoff",
            "eta_cutoff",
        ):
            sanitized.pop(key, None)
        return sanitized

    def _message_text(self, message: dict[str, Any]) -> str:
        """Map canonical history messages to chat-template text payloads."""

        content = message.get("content", "")
        if message.get("role") != "tool":
            return content

        tool_name = message.get("tool_name") or "tool"
        call_id = message.get("call_id") or "unknown_call"
        return f"Tool result from {tool_name} ({call_id}):\n{content}"

    def _render_chat_messages(self, request: BackendRequest) -> tuple[list[dict[str, Any]], list[Any]]:
        """Convert canonical messages plus image refs into processor chat blocks."""

        chat_messages: list[dict[str, Any]] = []
        opened_images: list[Any] = []
        for message in request.messages:
            role = message["role"]
            if role == "tool":
                role = "user"

            blocks: list[dict[str, Any]] = []
            text = self._message_text(message)
            if text:
                blocks.append({"type": "text", "text": text})

            for image_ref in message.get("image_refs", []):
                image = Image.open(Path(image_ref)).convert("RGB")
                opened_images.append(image)
                blocks.append({"type": "image", "image": image})

            if not blocks:
                blocks.append({"type": "text", "text": ""})
            chat_messages.append({"role": role, "content": blocks})
        return chat_messages, opened_images

    def _prepare_inputs(self, request: BackendRequest) -> tuple[Any, list[Any]]:
        """Apply the chat template and prepare one generation batch."""

        if self._processor is None:
            raise BackendError("Processor is unavailable before input preparation")

        chat_messages, opened_images = self._render_chat_messages(request)
        rendered = self._processor.apply_chat_template(
            chat_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        processor_kwargs = {
            "text": [rendered],
            "return_tensors": "pt",
        }
        if opened_images:
            processor_kwargs["images"] = opened_images
        encoded = self._processor(**processor_kwargs)
        return encoded, opened_images

    def _truncate_at_stop_sequences(self, text: str, stop_sequences: list[str]) -> str:
        """Trim decoded text at the earliest requested stop sequence when present."""

        stop_positions = [
            (text.find(sequence), len(sequence))
            for sequence in stop_sequences
            if sequence and text.find(sequence) >= 0
        ]
        if not stop_positions:
            return text.strip()
        end_index = min(position + length for position, length in stop_positions)
        return text[:end_index].strip()

    def generate(self, request: BackendRequest, *, sample: dict[str, Any]) -> BackendResponse:
        """Generate one real response from a local-path transformers runtime."""

        del sample  # The runtime consumes canonical request messages only.
        self._load_model()
        encoded_inputs, opened_images = self._prepare_inputs(request)
        try:
            model_device = self._infer_model_device(self._model)
            encoded_inputs = self._move_batch_to_device(encoded_inputs, model_device)
            generation_config = self._sanitize_generation_kwargs(self.runtime_config["generation"])
            output_ids = self._model.generate(**encoded_inputs, **generation_config)
            prompt_length = encoded_inputs["input_ids"].shape[-1]
            generated_ids = output_ids[:, prompt_length:]
            decoded = self._processor.batch_decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]
        except Exception as exc:  # noqa: BLE001
            raise BackendError(f"transformers generation failed: {exc}") from exc
        finally:
            for image in opened_images:
                image.close()

        return BackendResponse(
            backend_name=self.backend_name,
            raw_output=self._truncate_at_stop_sequences(decoded, request.stop_sequences),
            metadata={
                "policy": self.policy,
                "runtime_owner": "transformers_peft_runtime",
                "sample_kind": request.metadata.get("sample_kind", "unknown"),
                "tool_mode": request.tool_mode,
                "base_model_path": self.runtime_config["base_model_path"],
                "adapter_checkpoint_path": self.runtime_config["adapter_checkpoint_path"],
                "adapter_loaded": self._adapter_loaded,
                "generation_config": dict(self.runtime_config["generation"]),
            },
        )
