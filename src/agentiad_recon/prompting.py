"""Prompt and final-answer contract helpers for AgentIAD.

This module keeps one versioned prompt family for the non-tool baseline and the
tool-augmented inference paths. Prompt 1.4 adds explicit tool-enabled prompt
builders for `pz_only` and `pz_cr` while preserving the same `<think>` plus
`<answer>` final contract and the same strict parser. There is no standalone
CLI here; import the builders or run `python -m agentiad_recon.baseline --help`
for the inference entrypoint that consumes these helpers.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Any

from agentiad_recon.contracts import validate_payload


PROMPT_VERSION = "agentiad_tool_prompt_v1_4"
BASELINE_PROMPT_VERSION = "agentiad_baseline_prompt_v1_3"
FINAL_ANSWER_PARSER_VERSION = "agentiad_final_answer_parser_v1_3"
BASELINE_ANSWER_TEMPLATE = """<think>
short reasoning notes
</think>
<answer>
  <anomaly_present>true|false</anomaly_present>
  <top_anomaly>anomaly_label_or_null</top_anomaly>
  <visual_descriptions>
    <item>short visual fact</item>
  </visual_descriptions>
</answer>"""
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
    """Build the canonical tool-enabled prompt contract for one sample and mode."""

    image_rule = (
        f"Primary image: {sample['image']['uri']}. "
        "Reference images may only appear later as tool outputs; do not invent them."
    )
    system_message = (
        f"You are a single-agent anomaly inspector. Prompt version: {PROMPT_VERSION}. "
        "You may either emit one `<tool_call>` block or one `<think>` plus `<answer>` pair at a time. "
        f"Tool call format:\n{TOOL_CALL_TEMPLATE}\n"
        f"Final answer format:\n{BASELINE_ANSWER_TEMPLATE}"
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
        stop_sequences=["</tool_call>", "</answer>"],
    )


def build_baseline_prompt(sample: dict[str, Any]) -> PromptBundle:
    """Build the versioned non-tool baseline prompt used in Prompt 1.3.

    The baseline contract omits tools entirely, preserves the same anomaly
    field semantics, and requests the paper-aligned `<think>` plus `<answer>`
    wrapper format expected by the baseline evaluator.
    """

    image_rule = (
        f"Primary image: {sample['image']['uri']}. "
        "No tool use is available in this baseline mode, so reason from the primary image only."
    )
    system_message = (
        f"You are a single-agent anomaly inspector. Prompt version: {BASELINE_PROMPT_VERSION}. "
        "Tools are disabled in this run. Use only the primary image context in the prompt. "
        "Return exactly one `<think>` block followed by one `<answer>` block in this format:\n"
        f"{BASELINE_ANSWER_TEMPLATE}"
    )
    user_message = (
        f"Sample id: {sample['sample_id']}\n"
        f"Category: {sample['category']}\n"
        f"{_candidate_line(list(sample['anomaly_candidates']))}\n"
        f"{image_rule}"
    )
    return PromptBundle(
        prompt_version=BASELINE_PROMPT_VERSION,
        tool_path="no_tools",
        messages=[
            {
                "role": "system",
                "message_type": "system_instruction",
                "content": system_message,
                "image_refs": [],
                "metadata": {},
            },
            {
                "role": "user",
                "message_type": "user_prompt",
                "content": user_message,
                "image_refs": [sample["image"]["uri"]],
                "metadata": {"prompt_version": BASELINE_PROMPT_VERSION},
            },
        ],
        stop_sequences=["</answer>"],
    )


def extract_think_block(text: str) -> str | None:
    """Extract an optional `<think>` block for audit traces."""

    match = re.search(r"<think>(.*?)</think>", text, flags=re.DOTALL)
    if not match:
        return None
    content = match.group(1).strip()
    return content or None


def render_answer_block(
    answer: dict[str, Any],
    *,
    wrapper_tag: str = "answer",
    think: str | None = None,
) -> str:
    """Render a canonical answer payload as XML-like assistant output text."""

    visual_items = "\n".join(
        f"    <item>{description}</item>" for description in answer["visual_descriptions"]
    )
    if not visual_items:
        visual_items = "    "
    top_anomaly = "null" if answer["top_anomaly"] is None else answer["top_anomaly"]
    answer_block = (
        f"<{wrapper_tag}>\n"
        f"  <anomaly_present>{str(answer['anomaly_present']).lower()}</anomaly_present>\n"
        f"  <top_anomaly>{top_anomaly}</top_anomaly>\n"
        f"  <visual_descriptions>\n"
        f"{visual_items}\n"
        f"  </visual_descriptions>\n"
        f"</{wrapper_tag}>"
    )
    if think is None:
        return answer_block
    return f"<think>\n{think}\n</think>\n{answer_block}"


def _extract_block(text: str) -> str:
    """Extract either the legacy or baseline final-answer block from a response."""

    for tag in ("final_answer", "answer"):
        match = re.search(fr"<{tag}>.*?</{tag}>", text, flags=re.DOTALL)
        if match:
            return match.group(0)
    raise FinalAnswerContractError("No <final_answer> or <answer> block found in model output")


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

    if root.tag not in {"final_answer", "answer"}:
        raise FinalAnswerContractError(f"Unsupported final-answer root tag: {root.tag}")

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
