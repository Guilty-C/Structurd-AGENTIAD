# Adapter Scope

Adapters are the only place where AgentIAD-specific environment logic should touch generic frameworks. The intended adapters are:

- `PZ`: crop/zoom interaction surface
- `CR`: normal-reference retrieval surface
- MMAD-to-contract mapping helpers

This prompt does not implement those adapters yet. It only freezes the ownership boundary so later prompts can keep them thin.

Prompt 1.5 keeps that boundary but adds a first framework-facing adapter layer under `src/agentiad_recon/ms_swift_adapter.py`. The adapter only validates the externalized MS-Swift recipe, projects the canonical Prompt 1.5 dataset into an MS-Swift-facing JSONL shape, and reports whether the local runtime is actually available. It still does not own trainer internals.
