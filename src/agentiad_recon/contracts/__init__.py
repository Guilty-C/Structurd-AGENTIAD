"""Contract helpers for the AgentIAD clean-room reconstruction.

These helpers keep JSON-schema-backed validation in one place so the custom
MMAD, tool, prompt, answer, and trace code can reuse the same contract surface
without inventing a second validation stack.
"""

from .validation import ContractValidationError, validate_payload

__all__ = ["ContractValidationError", "validate_payload"]
