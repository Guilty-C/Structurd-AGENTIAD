"""JSON Schema validation helpers for the AgentIAD contract layer.

This module centralizes loading and validating the canonical schemas created in
Prompt 1.1 and extended in Prompt 1.2. The goal is to reuse a maintained
generic validator (`jsonschema`) while keeping AgentIAD-specific rules in the
schema files and thin semantic checks in the surrounding modules.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator
from jsonschema.validators import RefResolver


SCHEMA_ROOT = Path(__file__).resolve().parent / "schemas"


class ContractValidationError(ValueError):
    """Raised when a payload fails canonical contract validation."""


@lru_cache(maxsize=1)
def _schema_store() -> dict[str, dict[str, Any]]:
    """Load all local schemas once so `$ref` resolution stays deterministic."""

    store: dict[str, dict[str, Any]] = {}
    for path in sorted(SCHEMA_ROOT.glob("*.schema.json")):
        schema = json.loads(path.read_text(encoding="utf-8"))
        store[path.name] = schema
        store[path.as_uri()] = schema
        schema_id = schema.get("$id")
        if schema_id:
            store[schema_id] = schema
    return store


def load_schema(schema_name: str) -> dict[str, Any]:
    """Load one schema document by filename from the canonical schema root."""

    path = SCHEMA_ROOT / schema_name
    return json.loads(path.read_text(encoding="utf-8"))


@lru_cache(maxsize=None)
def _validator_for(schema_name: str) -> Draft202012Validator:
    """Create a cached validator with local reference resolution enabled."""

    schema = load_schema(schema_name)
    resolver = RefResolver(base_uri=f"{SCHEMA_ROOT.as_uri()}/", referrer=schema, store=_schema_store())
    return Draft202012Validator(schema, resolver=resolver)


def _format_error(error: Any) -> str:
    """Format a jsonschema error into a concise, audit-friendly message."""

    path = ".".join(str(part) for part in error.path)
    location = path or "<root>"
    return f"{location}: {error.message}"


def validate_payload(payload: dict[str, Any], schema_name: str) -> None:
    """Validate a payload against one canonical schema and raise on failure."""

    validator = _validator_for(schema_name)
    errors = sorted(validator.iter_errors(payload), key=lambda item: list(item.path))
    if errors:
        message = "; ".join(_format_error(error) for error in errors)
        raise ContractValidationError(f"{schema_name} validation failed: {message}")
