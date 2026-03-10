# Adapter Scope

Adapters are the only place where AgentIAD-specific environment logic should touch generic frameworks. The intended adapters are:

- `PZ`: crop/zoom interaction surface
- `CR`: normal-reference retrieval surface
- MMAD-to-contract mapping helpers

This prompt does not implement those adapters yet. It only freezes the ownership boundary so later prompts can keep them thin.
