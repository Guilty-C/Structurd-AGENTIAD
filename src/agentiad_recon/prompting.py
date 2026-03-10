"""Prompt and final-answer contract helpers for AgentIAD.

This module freezes the local prompt contract used by later inference and tool
loops. It keeps task wording versioned, makes the two tool settings explicit,
and parses the final XML-like answer block back into the canonical final-answer
schema with loud, informative failures when the contract is broken.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Any

from agentiad_recon.contracts import validate_payload


PROMPT_VERSION = "agentiad_prompt_v1_2"
FINAL_ANSWER_TEMPLATE = """<final_answer>
  <anomaly_present>true|false</anomaly_present>
  <top_anomaly>anomaly_label_or_null</top_anomaly>
  <visual_descriptions>
    <item>short visual fact</item>
  </visual_descriptions>
</final_answer>"""
TOOL_CALL_TEMPLATE = """<tool_call>
{"tool_name":"PZ","arguments":{"bbox":{"x0":0.10,"y0":0.10,"x1":0.80,"y1":0.80}}}
</tool_call>"""


class FinalAnswerContractError(ValueError):
    """Raised when the final-answer block is missing or semantically invalid."""


@dataclass(frozen=True)
class PromptBundle:
    """Versioned prompt bundle passed into later inference frameworks."""

    prompt_version: str
    tool_path: str
    messages: list[dict[str, Any]]
    stop_sequences: list[str]


def _candidate_line(candidates: list[str]) -> str:
    """Render anomaly candidates deterministically for the user prompt."""

    if not candidates:
        return "Locked anomaly candidates: none discovered yet."
    return "Locked anomaly candidates: " + ", ".join(sorted(candidates))


def _tooling_instruction(tool_path: str) -> str:
    """Describe the allowed tool surface for one trajectory mode."""

    if tool_path == "pz_only":
        return (
            "Available tools: PZ only. If you need a crop, emit exactly one <tool_call> block "
            "for PZ. Do not request CR in this mode."
        )
    if tool_path == "pz_cr":
        return (
            "Available tools: PZ and CR. Use PZ for localized crop/zoom, and CR only when "
            "you need a same-category normal reference exemplar."
        )
    raise ValueError(f"Unsupported tool_path: {tool_path}")


def build_prompt(sample: dict[str, Any], *, tool_path: str) -> PromptBundle:
    """Build the canonical local prompt contract for one sample and tool mode."""

    image_rule = (
        f"Primary image: {sample['image']['uri']}. "
        "Reference images may only appear later as tool outputs; do not invent them."
    )
    system_message = (
        f"You are a single-agent anomaly inspector. Prompt version: {PROMPT_VERSION}. "
        "You may either emit one <tool_call> block or one <final_answer> block at a time. "
        f"Tool call format:\n{TOOL_CALL_TEMPLATE}\n"
        f"Final answer format:\n{FINAL_ANSWER_TEMPLATE}"
    )
    user_message = (
        f"Sample id: {sample['sample_id']}\n"
        f"Category: {sample['category']}\n"
        f"{_candidate_line(list(sample['anomaly_candidates']))}\n"
        f"{image_rule}\n"
        f"{_tooling_instruction(tool_path)}"
    )

    # Stop sequences are explicit so later framework integrations can preserve
    # the same contract without reconstructing it from prompt prose.
    return PromptBundle(
        prompt_version=PROMPT_VERSION,
        tool_path=tool_path,
        messages=[
            {"role": "system", "message_type": "system_instruction", "content": system_message, "image_refs": [], "metadata": {}},
            {
                "role": "user",
                "message_type": "user_prompt",
                "content": user_message,
                "image_refs": [sample["image"]["uri"]],
                "metadata": {"prompt_version": PROMPT_VERSION},
            },
        ],
        stop_sequences=["</tool_call>", "</final_answer>"],
    )


def _extract_block(text: str) -> str:
    """Extract the XML-like final-answer block from a model response."""

    match = re.search(r"<final_answer>.*?</final_answer>", text, flags=re.DOTALL)
    if not match:
        raise FinalAnswerContractError("No <final_answer> block found in model output")
    return match.group(0)


def _parse_bool(text: str) -> bool:
    """Normalize common boolean spellings used by LLM outputs."""

    normalized = text.strip().lower()
    if normalized in {"true", "yes", "1"}:
        return True
    if normalized in {"false", "no", "0"}:
        return False
    raise FinalAnswerContractError(f"Invalid anomaly_present value: {text!r}")


def _normalize_top_anomaly(text: str | None) -> str | None:
    """Normalize null-like anomaly labels into `None`."""

    if text is None:
        return None
    cleaned = text.strip()
    if cleaned.lower() in {"", "none", "null", "n/a"}:
        return None
    return cleaned


def _parse_visual_descriptions(root: ET.Element) -> list[str]:
    """Support both `<item>` children and newline-delimited text fallbacks."""

    visual_root = root.find("visual_descriptions")
    if visual_root is None:
        raise FinalAnswerContractError("Missing <visual_descriptions> element")

    items = [item.text.strip() for item in visual_root.findall("item") if item.text and item.text.strip()]
    if items:
        return items

    fallback_text = "".join(visual_root.itertext()).strip()
    if not fallback_text:
        return []
    descriptions = []
    for line in fallback_text.splitlines():
        candidate = line.strip().lstrip("-").strip()
        if candidate:
            descriptions.append(candidate)
    return descriptions


def parse_final_answer(text: str) -> dict[str, Any]:
    """Parse, validate, and semantically check the final answer contract."""

    block = _extract_block(text)
    try:
        root = ET.fromstring(block)
    except ET.ParseError as exc:
        raise FinalAnswerContractError(f"Malformed <final_answer> XML block: {exc}") from exc

    anomaly_node = root.findtext("anomaly_present")
    top_anomaly = _normalize_top_anomaly(root.findtext("top_anomaly"))
    if anomaly_node is None:
        raise FinalAnswerContractError("Missing <anomaly_present> element")
    anomaly_present = _parse_bool(anomaly_node)
    visual_descriptions = _parse_visual_descriptions(root)

    if anomaly_present and not top_anomaly:
        raise FinalAnswerContractError(
            "top_anomaly must be a non-empty label when anomaly_present is true"
        )
    if not anomaly_present and top_anomaly is not None:
        raise FinalAnswerContractError(
            "top_anomaly must be null/empty when anomaly_present is false"
        )
    if anomaly_present and not visual_descriptions:
        raise FinalAnswerContractError(
            "visual_descriptions must contain at least one item when anomaly_present is true"
        )

    payload = {
        "anomaly_present": anomaly_present,
        "top_anomaly": top_anomaly,
        "visual_descriptions": visual_descriptions,
    }
    validate_payload(payload, "final_answer.schema.json")
    return payload
