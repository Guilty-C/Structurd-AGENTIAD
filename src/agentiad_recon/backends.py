"""Thin inference backend interfaces for AgentIAD inference runs.

This module keeps runtime ownership thin and explicit across the non-tool
baseline and tool-augmented paths. Prompt 1.4 extends the local smoke layer
with scripted tool-aware backend policies while preserving a future maintained
runtime adapter skeleton for remote execution. There is no standalone CLI for
this module.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

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

    @abstractmethod
    def generate(self, request: BackendRequest, *, sample: dict[str, Any]) -> BackendResponse:
        """Generate one raw model response for a canonical sample."""


class MockInferenceBackend(InferenceBackend):
    """Deterministic fixture-only backend for local non-tool smoke validation.

    The mock backend is intentionally explicit about its policy so no caller can
    mistake it for real model inference. It uses canonical sample metadata and
    ground truth only to exercise the parsing and evaluation plumbing.
    """

    def __init__(self, *, backend_name: str, policy: str) -> None:
        self.backend_name = backend_name
        self.policy = policy

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

    def __init__(self, *, backend_name: str, policy: str) -> None:
        self.backend_name = backend_name
        self.policy = policy

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

    def __init__(self, *, backend_name: str, model: str | None = None) -> None:
        self.backend_name = backend_name
        self.model = model

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
