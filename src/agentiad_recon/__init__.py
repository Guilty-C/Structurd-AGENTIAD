"""AgentIAD clean-room reconstruction package.

This package is intentionally small at Prompt 1.1. It only exposes thin,
auditable utilities that support architecture locking, canonical contracts,
and reproducibility metadata without introducing bespoke trainer machinery.
"""

__all__ = [
    "backends",
    "baseline",
    "evaluation",
    "mmad",
    "ms_swift_adapter",
    "prompting",
    "reproducibility",
    "sft",
    "tooling",
    "traces",
]
