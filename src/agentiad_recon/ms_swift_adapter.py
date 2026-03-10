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
from pathlib import Path
from typing import Any

from agentiad_recon.contracts import validate_payload


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
