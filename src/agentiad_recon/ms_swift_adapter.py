"""Thin MS-Swift adapter helpers for Prompt 1.5.

This module keeps framework ownership explicit by limiting itself to dataset and
recipe projection for MS-Swift. It does not implement trainer internals or run
full SFT. Import these helpers from the canonical Prompt 1.5 exporter, or run
`python -m agentiad_recon.ms_swift_adapter --help` for a small local format
check that only validates recipes and runtime availability.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from agentiad_recon.contracts import validate_payload

IMAGE_PLACEHOLDER_TOKEN = "<image>"
FALLBACK_IMAGE_TOKEN_BUDGET = 256
FALLBACK_MESSAGE_OVERHEAD = 16


def _validate_swift_record_semantics(record: dict[str, Any]) -> None:
    """Enforce MS-Swift multimodal message/content invariants beyond schema checks."""

    record_id = record.get("id", "<unknown_record>")
    messages = record.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError(f"{record_id}: `messages` must be a non-empty list")

    images = record.get("images")
    if not isinstance(images, list):
        raise ValueError(f"{record_id}: `images` must be a list")
    if len(set(images)) != len(images):
        raise ValueError(f"{record_id}: `images` must be first-occurrence ordered with no duplicates")

    placeholder_count = 0
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise ValueError(f"{record_id}: message[{index}] must be an object")
        content = message.get("content")
        if isinstance(content, (list, dict)):
            raise ValueError(f"{record_id}: message[{index}].content must be a string, not {type(content)!r}")
        if not isinstance(content, str):
            raise ValueError(f"{record_id}: message[{index}].content must be a string")
        placeholder_count += content.count(IMAGE_PLACEHOLDER_TOKEN)

        role = message.get("role")
        if role != "assistant" and message.get("loss") is True:
            raise ValueError(f"{record_id}: only assistant messages may set loss=true (message[{index}])")
        if role == "tool":
            tool_name = message.get("tool_name")
            call_id = message.get("call_id")
            if not isinstance(tool_name, str) or not tool_name:
                raise ValueError(f"{record_id}: tool message[{index}] must include non-empty tool_name")
            if not isinstance(call_id, str) or not call_id:
                raise ValueError(f"{record_id}: tool message[{index}] must include non-empty call_id")

    for image_index, image_path in enumerate(images):
        if not isinstance(image_path, str) or not image_path.strip():
            raise ValueError(f"{record_id}: images[{image_index}] must be a non-empty image path string")

    if placeholder_count != len(images):
        raise ValueError(
            f"{record_id}: placeholder/image mismatch: {placeholder_count} `<image>` tokens vs {len(images)} images"
        )


def load_swift_recipe(path: str | Path) -> dict[str, Any]:
    """Load and validate the thin MS-Swift recipe/config surface."""

    recipe = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_payload(recipe, "ms_swift_recipe.schema.json")
    return recipe


def swift_runtime_probe() -> dict[str, Any]:
    """Report local MS-Swift availability without pretending the framework ran."""

    try:
        import importlib.util

        spec = importlib.util.find_spec("swift")
    except Exception as exc:  # noqa: BLE001 - runtime probe should never mask the error.
        return {"available": False, "detail": f"probe_failed:{exc}"}

    if spec is None:
        return {"available": False, "detail": "python package `swift` was not found locally"}
    return {"available": True, "detail": f"swift package found at {spec.origin}"}


def validate_swift_record(record: dict[str, Any]) -> None:
    """Validate one projected MS-Swift dataset record."""

    validate_payload(record, "ms_swift_record.schema.json")
    _validate_swift_record_semantics(record)


def _nearest_rank(sorted_values: list[int], percentile: int) -> int:
    """Compute nearest-rank percentile from a pre-sorted non-empty integer list."""

    rank = max(1, math.ceil((percentile / 100.0) * len(sorted_values)))
    return sorted_values[rank - 1]


def _fallback_encoded_length(record: dict[str, Any]) -> int:
    """Fallback estimate when no local Swift-compatible processor is available."""

    text_tokens = sum(len(message["content"].split()) for message in record["messages"])
    image_tokens = FALLBACK_IMAGE_TOKEN_BUDGET * len(record["images"])
    wrapper_tokens = FALLBACK_MESSAGE_OVERHEAD * len(record["messages"])
    return int(text_tokens + image_tokens + wrapper_tokens)


def _transformers_processor_encoder(model_id_or_path: str):
    """Build an encoded-length callable using local HF processor/tokenizer assets."""

    try:
        from PIL import Image
        from transformers import AutoProcessor
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"processor_import_failed:{exc}") from exc

    try:
        processor = AutoProcessor.from_pretrained(
            model_id_or_path,
            trust_remote_code=True,
            local_files_only=True,
        )
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"processor_load_failed:{exc}") from exc

    def encode(record: dict[str, Any]) -> int:
        images = record["images"]
        image_cursor = 0
        chat_messages: list[dict[str, Any]] = []
        opened_images = []
        try:
            for message in record["messages"]:
                role = message["role"] if message["role"] in {"system", "user", "assistant"} else "assistant"
                text = message["content"]
                parts = text.split(IMAGE_PLACEHOLDER_TOKEN)
                blocks: list[dict[str, Any]] = []
                for index, part in enumerate(parts):
                    if part:
                        blocks.append({"type": "text", "text": part})
                    if index < len(parts) - 1:
                        if image_cursor >= len(images):
                            raise RuntimeError("placeholder_image_cursor_overflow")
                        image_path = images[image_cursor]
                        opened_images.append(Image.open(image_path).convert("RGB"))
                        blocks.append({"type": "image", "image": opened_images[-1]})
                        image_cursor += 1
                if not blocks:
                    blocks.append({"type": "text", "text": ""})
                chat_messages.append({"role": role, "content": blocks})

            rendered = processor.apply_chat_template(
                chat_messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            encoded = processor(text=[rendered], images=opened_images, return_tensors="pt")
            input_ids = encoded["input_ids"]
            return int(input_ids.shape[-1])
        finally:
            for image in opened_images:
                image.close()

    return encode


def compute_true_length_audit(
    records: list[dict[str, Any]],
    recipe: dict[str, Any],
    *,
    strict: bool = False,
) -> dict[str, Any]:
    """Compute a true-or-annotated-fallback multimodal length audit for Swift records."""

    if not records:
        raise ValueError("Cannot compute length audit for an empty record list")

    model_id_or_path = recipe.get("training", {}).get("model_id_or_path", "")
    backend = "fallback_proxy_with_visual_and_template_budget"
    backend_detail = "processor_unavailable"
    encoder = _fallback_encoded_length
    true_multimodal = False

    try:
        encoder = _transformers_processor_encoder(model_id_or_path)
        backend = "transformers_processor_local_encode"
        backend_detail = f"loaded:{model_id_or_path}"
        true_multimodal = True
    except Exception as exc:  # noqa: BLE001
        backend_detail = str(exc)
        if strict:
            raise RuntimeError(
                f"True multimodal length audit failed in strict mode: {backend_detail}"
            ) from exc

    rows: list[dict[str, Any]] = []
    for record in records:
        length = int(encoder(record))
        rows.append(
            {
                "id": record["id"],
                "sample_id": record["metadata"]["sample_id"],
                "trajectory_mode": record["metadata"]["trajectory_mode"],
                "encoded_length": length,
            }
        )

    sorted_lengths = sorted(row["encoded_length"] for row in rows)
    top_rows = sorted(rows, key=lambda row: row["encoded_length"], reverse=True)[:10]
    return {
        "audit_type": "multimodal_length_audit",
        "true_multimodal_encode": true_multimodal,
        "backend": backend,
        "backend_detail": backend_detail,
        "record_count": len(rows),
        "p50": _nearest_rank(sorted_lengths, 50),
        "p90": _nearest_rank(sorted_lengths, 90),
        "p95": _nearest_rank(sorted_lengths, 95),
        "p99": _nearest_rank(sorted_lengths, 99),
        "max": sorted_lengths[-1],
        "count_above_4096": sum(1 for value in sorted_lengths if value > 4096),
        "count_above_8192": sum(1 for value in sorted_lengths if value > 8192),
        "top_offenders": top_rows,
        "lengths": rows,
    }


def _build_parser() -> argparse.ArgumentParser:
    """Build the tiny local-only MS-Swift recipe checker CLI."""

    parser = argparse.ArgumentParser(
        description="Validate the Prompt 1.5 MS-Swift recipe and report local runtime availability."
    )
    parser.add_argument("--recipe", required=True, help="Path to the MS-Swift recipe JSON.")
    return parser


def main() -> int:
    """Run the small recipe checker and print the result as JSON."""

    args = _build_parser().parse_args()
    recipe = load_swift_recipe(args.recipe)
    print(
        json.dumps(
            {
                "recipe_name": recipe["recipe_name"],
                "framework_owner": recipe["framework_owner"],
                "runtime_probe": swift_runtime_probe(),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
