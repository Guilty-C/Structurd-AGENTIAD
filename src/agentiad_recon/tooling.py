"""Deterministic PZ, CR, and tool-call protocol helpers for AgentIAD.

This module implements the local-only tool environment contracts required by
Prompt 1.2. The tools are deliberately narrow: PZ performs a deterministic
crop from a normalized bounding box, CR chooses a same-category normal exemplar
with an auditable policy, and the tool-call protocol parses and reinserts these
results without introducing model or trainer machinery.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from PIL import Image

from agentiad_recon.contracts import validate_payload
from agentiad_recon.reproducibility import sha256_file, sha256_mapping


TOOL_CALL_PATTERN = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
PROTOCOL_TOOL_PATHS = {
    "pz_only": ("PZ",),
    "pz_cr": ("PZ", "CR"),
}
SPLIT_ORDER = {"train": 0, "val": 1, "test": 2}


class ToolContractError(ValueError):
    """Raised when a tool call or tool execution violates the contract."""


@dataclass(frozen=True)
class NormalizedBBox:
    """Normalized bounding box with explicit `[x0, y0, x1, y1]` semantics."""

    x0: float
    y0: float
    x1: float
    y1: float

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> "NormalizedBBox":
        """Validate and build a box from a JSON-compatible mapping."""

        try:
            bbox = cls(
                x0=float(payload["x0"]),
                y0=float(payload["y0"]),
                x1=float(payload["x1"]),
                y1=float(payload["y1"]),
            )
        except KeyError as exc:
            raise ToolContractError(f"Missing bbox coordinate: {exc.args[0]}") from exc
        except (TypeError, ValueError) as exc:
            raise ToolContractError(f"Invalid bbox coordinate payload: {payload}") from exc

        bbox.validate()
        return bbox

    def validate(self) -> None:
        """Enforce normalized coordinate bounds before any crop is attempted."""

        for name, value in (("x0", self.x0), ("y0", self.y0), ("x1", self.x1), ("y1", self.y1)):
            if value < 0.0 or value > 1.0:
                raise ToolContractError(f"{name} must be within [0, 1], got {value}")
        if self.x1 <= self.x0 or self.y1 <= self.y0:
            raise ToolContractError("Bounding box must satisfy x1 > x0 and y1 > y0")

    def to_mapping(self) -> dict[str, float]:
        """Return the normalized box as a JSON-safe mapping."""

        return {"x0": self.x0, "y0": self.y0, "x1": self.x1, "y1": self.y1}

    def to_pixel_bounds(self, width: int, height: int) -> tuple[int, int, int, int]:
        """Convert normalized coordinates to deterministic pixel bounds."""

        left = math.floor(self.x0 * width)
        top = math.floor(self.y0 * height)
        right = math.ceil(self.x1 * width)
        bottom = math.ceil(self.y1 * height)

        left = max(0, min(left, width - 1))
        top = max(0, min(top, height - 1))
        right = max(left + 1, min(right, width))
        bottom = max(top + 1, min(bottom, height))
        if right <= left or bottom <= top:
            raise ToolContractError("Bounding box collapses to an empty crop after pixel conversion")
        return left, top, right, bottom


@dataclass(frozen=True)
class ParsedToolCall:
    """Structured tool call parsed from one assistant output block."""

    call_id: str
    tool_name: str
    arguments: dict[str, Any]
    raw_text: str


@dataclass(frozen=True)
class ToolResult:
    """Canonicalized tool execution result aligned with `tool_call.schema.json`."""

    call_id: str
    tool_name: str
    arguments: dict[str, Any]
    output_ref: dict[str, Any]
    output_payload: dict[str, Any]
    status: str = "completed"
    raw_text: str | None = None
    audit_log: tuple[str, ...] = field(default_factory=tuple)

    def to_payload(self) -> dict[str, Any]:
        """Convert the result into the canonical tool-call payload."""

        payload = {
            "call_id": self.call_id,
            "tool_name": self.tool_name,
            "status": self.status,
            "arguments": self.arguments,
            "output_ref": self.output_ref,
            "output_payload": self.output_payload,
            "audit_log": list(self.audit_log),
        }
        if self.raw_text is not None:
            payload["raw_text"] = self.raw_text
        validate_payload(payload, "tool_call.schema.json")
        return payload


def protocol_event(text: str) -> str:
    """Classify one model output as a tool call, final answer, or continuation."""

    has_tool_call = "<tool_call>" in text
    has_final_answer = "<final_answer>" in text or "<answer>" in text
    if has_tool_call and has_final_answer:
        raise ToolContractError("Output cannot contain both a tool call and a final answer block")
    if has_tool_call:
        return "tool_call"
    if has_final_answer:
        return "final_answer"
    return "continue"


def parse_tool_call(text: str, *, tool_path: str) -> ParsedToolCall:
    """Parse and validate one tool-call block from assistant text."""

    if tool_path not in PROTOCOL_TOOL_PATHS:
        raise ToolContractError(f"Unsupported tool_path: {tool_path}")

    match = TOOL_CALL_PATTERN.search(text)
    if not match:
        raise ToolContractError("No <tool_call> JSON block found in assistant output")

    raw_block = match.group(0)
    try:
        payload = json.loads(match.group(1))
    except json.JSONDecodeError as exc:
        raise ToolContractError(f"Tool call JSON is malformed: {exc}") from exc

    tool_name = payload.get("tool_name")
    if tool_name not in PROTOCOL_TOOL_PATHS[tool_path]:
        raise ToolContractError(
            f"Tool {tool_name!r} is not allowed for tool_path={tool_path}"
        )

    arguments = payload.get("arguments", {})
    if not isinstance(arguments, dict):
        raise ToolContractError("Tool call arguments must be a JSON object")

    if tool_name == "PZ":
        bbox_payload = arguments.get("bbox")
        if not isinstance(bbox_payload, dict):
            raise ToolContractError("PZ arguments must include a bbox object")
        NormalizedBBox.from_mapping(bbox_payload)
    elif tool_name == "CR":
        policy = arguments.get("policy", "same_category_normal")
        if policy != "same_category_normal":
            raise ToolContractError("CR policy must be 'same_category_normal'")
        arguments = {"policy": "same_category_normal"}

    call_id = payload.get("call_id") or sha256_mapping(
        {"raw_text": raw_block, "tool_name": tool_name, "arguments": arguments}
    )[:12]
    return ParsedToolCall(call_id=call_id, tool_name=tool_name, arguments=arguments, raw_text=raw_block)


class PerceptiveZoomer:
    """Deterministic crop adapter for the PZ tool contract."""

    coordinate_convention = (
        "normalized [x0, y0, x1, y1], top-left origin, inclusive left/top and exclusive right/bottom"
    )

    def run(
        self,
        *,
        image_path: str | Path,
        bbox: dict[str, Any] | NormalizedBBox,
        call_id: str,
        artifact_dir: str | Path | None = None,
        raw_text: str | None = None,
    ) -> ToolResult:
        """Execute one deterministic crop and return its auditable metadata."""

        normalized_bbox = bbox if isinstance(bbox, NormalizedBBox) else NormalizedBBox.from_mapping(bbox)
        image_path = Path(image_path)
        with Image.open(image_path) as image:
            width, height = image.size
            pixel_bbox = normalized_bbox.to_pixel_bounds(width, height)
            crop = image.crop(pixel_bbox)

            output_ref = {"artifact_id": f"{call_id}:pz_crop"}
            payload = {
                "image_path": str(image_path.resolve()),
                "image_size": {"width": width, "height": height},
                "normalized_bbox": normalized_bbox.to_mapping(),
                "pixel_bbox": {
                    "left": pixel_bbox[0],
                    "top": pixel_bbox[1],
                    "right": pixel_bbox[2],
                    "bottom": pixel_bbox[3],
                },
                "crop_size": {"width": crop.size[0], "height": crop.size[1]},
                "coordinate_convention": self.coordinate_convention,
            }

            # Saving the crop is optional so tests can verify real pixels while
            # later runtime loops can choose where to persist artifacts.
            if artifact_dir is not None:
                artifact_dir = Path(artifact_dir)
                artifact_dir.mkdir(parents=True, exist_ok=True)
                crop_path = artifact_dir / f"{image_path.stem}.{call_id}.crop.png"
                crop.save(crop_path)
                output_ref["artifact_id"] = crop_path.stem
                output_ref["sha256"] = sha256_file(crop_path)
                payload["artifact_path"] = str(crop_path.resolve())

        result = ToolResult(
            call_id=call_id,
            tool_name="PZ",
            arguments={"bbox": normalized_bbox.to_mapping()},
            output_ref=output_ref,
            output_payload=payload,
            raw_text=raw_text,
            audit_log=("deterministic_crop",),
        )
        return result


class ComparativeRetriever:
    """Deterministic same-category normal exemplar selection for the CR tool."""

    policy_name = "same_category_normal_v1"

    def run(
        self,
        *,
        target_sample: dict[str, Any],
        sample_pool: Iterable[dict[str, Any]],
        call_id: str,
        raw_text: str | None = None,
    ) -> ToolResult:
        """Select one same-category normal exemplar or return a logged fallback."""

        candidates = [
            sample
            for sample in sample_pool
            if sample["category"] == target_sample["category"]
            and not sample["anomaly_present"]
            and sample["sample_id"] != target_sample["sample_id"]
        ]

        # Ranking is explicit and auditable: same split first, then canonical id.
        candidates.sort(
            key=lambda sample: (
                0 if sample["split"] == target_sample["split"] else 1,
                SPLIT_ORDER[sample["split"]],
                sample["sample_id"],
            )
        )

        audit_log: list[str] = [f"policy={self.policy_name}", f"candidate_count={len(candidates)}"]
        payload: dict[str, Any] = {
            "policy": self.policy_name,
            "target_sample_id": target_sample["sample_id"],
        }
        output_ref = {"artifact_id": f"{call_id}:cr_selection"}

        if candidates:
            exemplar = candidates[0]
            payload["selected_exemplar"] = {
                "sample_id": exemplar["sample_id"],
                "image_uri": exemplar["image"]["uri"],
                "split": exemplar["split"],
                "category": exemplar["category"],
            }
            payload["selection_reason"] = "first deterministic same-category normal exemplar"
            audit_log.append(f"selected={exemplar['sample_id']}")
        else:
            payload["selected_exemplar"] = None
            payload["selection_reason"] = "no_same_category_normal_exemplar_available"
            audit_log.append("selected=None")

        result = ToolResult(
            call_id=call_id,
            tool_name="CR",
            arguments={"policy": "same_category_normal"},
            output_ref=output_ref,
            output_payload=payload,
            raw_text=raw_text,
            audit_log=tuple(audit_log),
        )
        return result


def execute_tool_call(
    parsed_call: ParsedToolCall,
    *,
    sample: dict[str, Any],
    sample_pool: Iterable[dict[str, Any]],
    artifact_dir: str | Path | None = None,
) -> ToolResult:
    """Dispatch one parsed tool call to the deterministic PZ or CR adapter."""

    if parsed_call.tool_name == "PZ":
        return PerceptiveZoomer().run(
            image_path=sample["image"]["uri"],
            bbox=parsed_call.arguments["bbox"],
            call_id=parsed_call.call_id,
            artifact_dir=artifact_dir,
            raw_text=parsed_call.raw_text,
        )
    if parsed_call.tool_name == "CR":
        return ComparativeRetriever().run(
            target_sample=sample,
            sample_pool=sample_pool,
            call_id=parsed_call.call_id,
            raw_text=parsed_call.raw_text,
        )
    raise ToolContractError(f"Unsupported tool_name: {parsed_call.tool_name}")


def reinsert_tool_result(history: list[dict[str, Any]], result: ToolResult) -> list[dict[str, Any]]:
    """Append a tool result message to a dialogue history in canonical form."""

    payload = result.to_payload()
    image_refs: list[str] = []
    if result.tool_name == "PZ":
        artifact_path = payload["output_payload"].get("artifact_path")
        if isinstance(artifact_path, str):
            image_refs.append(artifact_path)
    elif result.tool_name == "CR":
        exemplar = payload["output_payload"].get("selected_exemplar")
        if isinstance(exemplar, dict):
            image_uri = exemplar.get("image_uri")
            if isinstance(image_uri, str):
                image_refs.append(image_uri)

    # Tool outputs stay machine-readable in content while explicit image refs
    # preserve the crop/reference insertion contract for later runtimes.
    next_history = list(history)
    next_history.append(
        {
            "role": "tool",
            "message_type": "tool_result",
            "tool_name": result.tool_name,
            "call_id": result.call_id,
            "content": json.dumps(payload["output_payload"], sort_keys=True),
            "metadata": {"output_ref": payload["output_ref"]},
            "image_refs": image_refs,
        }
    )
    return next_history
