"""Trace serialization helpers for AgentIAD prompt, tool, and answer flows.

This module provides one auditable trace format for local evaluation storage and
one projection back into the Prompt 1.1 training trajectory schema. The point
is not to overengineer a runtime bus; it is to keep message order, tool usage,
and final answers explicit enough that later SFT and evaluation code can share
the same thin-waist records. The non-tool baseline path introduced in Prompt
1.3 uses the same trace record with `tool_path="no_tools"` during eval.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agentiad_recon.contracts import validate_payload


@dataclass(frozen=True)
class TraceMessage:
    """One ordered prompt, reasoning, tool, or final-answer message."""

    role: str
    message_type: str
    content: str
    image_refs: tuple[str, ...] = ()
    tool_name: str | None = None
    call_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert the message to the audit-trace schema shape."""

        return {
            "role": self.role,
            "message_type": self.message_type,
            "content": self.content,
            "image_refs": list(self.image_refs),
            "tool_name": self.tool_name,
            "call_id": self.call_id,
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class TraceRecord:
    """Auditable record for either training trajectory preparation or eval logs."""

    trace_id: str
    sample_id: str
    stage: str
    tool_path: str
    storage_purpose: str
    messages: tuple[TraceMessage, ...]
    tool_traces: tuple[dict[str, Any], ...]
    final_answer: dict[str, Any] | None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_audit_payload(self) -> dict[str, Any]:
        """Serialize the full audit trace and validate it against the schema."""

        payload = {
            "trace_id": self.trace_id,
            "sample_id": self.sample_id,
            "stage": self.stage,
            "agent_mode": "single_agent",
            "tool_path": self.tool_path,
            "storage_purpose": self.storage_purpose,
            "messages": [message.to_dict() for message in self.messages],
            "tool_traces": list(self.tool_traces),
            "final_answer": self.final_answer,
            "metadata": self.metadata,
        }
        validate_payload(payload, "trace_record.schema.json")
        return payload

    def to_training_trajectory(self) -> dict[str, Any]:
        """Project a trace into the existing training trajectory schema."""

        if self.storage_purpose != "training":
            raise ValueError("Only training traces can be exported as training trajectories")
        if self.stage not in {"sft", "grpo"}:
            raise ValueError("Training trajectory export requires stage 'sft' or 'grpo'")
        if self.tool_path == "no_tools":
            raise ValueError("No-tool baseline traces are eval-only and cannot become training trajectories")
        if self.final_answer is None:
            raise ValueError("Training trajectory export requires a final_answer payload")

        steps: list[dict[str, Any]] = []
        tool_trace_by_call = {
            trace["call_id"]: trace for trace in self.tool_traces if "call_id" in trace
        }
        for index, message in enumerate(self.messages):
            if message.message_type == "reasoning":
                steps.append(
                    {
                        "step_index": index,
                        "step_type": "reasoning",
                        "content": message.content,
                    }
                )
            elif message.message_type == "tool_request" and message.call_id:
                steps.append(
                    {
                        "step_index": index,
                        "step_type": "tool_call",
                        "tool_call": tool_trace_by_call[message.call_id],
                        "content": message.content,
                    }
                )
            elif message.message_type == "tool_result":
                steps.append(
                    {
                        "step_index": index,
                        "step_type": "observation",
                        "content": message.content,
                    }
                )

        payload = {
            "trajectory_id": self.trace_id,
            "sample_id": self.sample_id,
            "stage": self.stage,
            "agent_mode": "single_agent",
            "tool_path": self.tool_path,
            "steps": steps or [{"step_index": 0, "step_type": "reasoning", "content": ""}],
            "final_answer": self.final_answer,
        }
        validate_payload(payload, "trajectory.schema.json")
        return payload
