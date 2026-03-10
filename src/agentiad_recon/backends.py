"""Thin inference backend interfaces for AgentIAD baseline execution.

Prompt 1.3 keeps inference ownership with maintained runtime layers by using a
small backend interface plus a future vLLM adapter skeleton. Local smoke checks
use the deterministic mock backend here; real heavy execution remains deferred.
There is no standalone CLI for this module.
"""

from __future__ import annotations

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
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BackendResponse:
    """Captured raw backend response plus audit metadata."""

    backend_name: str
    raw_output: str
    metadata: dict[str, Any]


class InferenceBackend(ABC):
    """Abstract interface for non-tool baseline inference backends."""

    backend_name: str

    @abstractmethod
    def generate(self, request: BackendRequest, *, sample: dict[str, Any]) -> BackendResponse:
        """Generate one raw model response for a canonical sample."""


class MockInferenceBackend(InferenceBackend):
    """Deterministic fixture-only backend for local smoke validation.

    The mock backend is intentionally explicit about its policy so no caller can
    mistake it for real model inference. It uses canonical sample metadata and
    ground truth only to exercise the parsing and evaluation plumbing.
    """

    def __init__(self, *, backend_name: str, policy: str) -> None:
        self.backend_name = backend_name
        self.policy = policy

    def _fixture_scripted_output(self, request: BackendRequest, sample: dict[str, Any]) -> str:
        """Create deterministic fixture outputs, including one malformed branch."""

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
        """Return a deterministic raw output for fixture-backed smoke runs."""

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


class VLLMBackendAdapter(InferenceBackend):
    """Thin maintained-runtime adapter skeleton for future remote execution.

    This adapter exists to keep ownership with a maintained inference runtime
    instead of growing a custom serving framework in-repo. Prompt 1.3 does not
    execute it locally.
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
            "Prompt 1.3 remains local-only and should use the mock backend."
        )
