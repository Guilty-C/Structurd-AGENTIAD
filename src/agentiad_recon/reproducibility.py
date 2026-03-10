"""Reproducibility skeleton for the AgentIAD clean-room rebuild.

This module provides lightweight hashing helpers and a run-metadata constructor
that can be reused by later local scripts without pulling in training
frameworks. The goal is to lock auditable interfaces now, while deferring any
heavy execution to later remote-only prompts.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def sha256_bytes(payload: bytes) -> str:
    """Return a stable SHA256 hex digest for raw bytes."""

    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: str | Path) -> str:
    """Hash a file from disk in chunks to avoid loading large files at once."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        # Chunked hashing keeps this helper safe for later larger manifests.
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(payload: dict[str, Any]) -> bytes:
    """Serialize JSON deterministically for hash-friendly metadata storage."""

    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256_mapping(payload: dict[str, Any]) -> str:
    """Hash a JSON-compatible mapping after canonical serialization."""

    return sha256_bytes(canonical_json_bytes(payload))


def build_run_metadata(
    *,
    run_id: str,
    phase: str,
    boundary: str,
    config_hashes: dict[str, str],
    script_hashes: dict[str, str],
    dataset_manifest_hash: str | None,
    notes: list[str] | None = None,
) -> dict[str, Any]:
    """Build a minimal run-metadata record that matches the locked schema.

    Later prompts can enrich this structure with framework-specific fields, but
    they should not rename the thin-waist keys introduced here.
    """

    # The boundary flag is intentionally first-class so local and remote runs
    # remain easy to separate in later audit reviews.
    return {
        "run_id": run_id,
        "phase": phase,
        "boundary": boundary,
        "config_hashes": config_hashes,
        "script_hashes": script_hashes,
        "dataset_manifest_hash": dataset_manifest_hash,
        "notes": notes or [],
    }
