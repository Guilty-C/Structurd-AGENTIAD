# Scientific Target Lock

Prompt 1.1 freezes the scientific targets that later implementation work must respect.

- Agent formulation remains single-agent.
- The tool surface includes `PZ` crop/zoom and `CR` normal-reference retrieval.
- Canonical trajectories must support both `pz_only` and `pz_cr`.
- Training order is fixed as SFT first, then GRPO.
- Reward inputs must preserve both perception and behavior logic so that answer quality and tool-use quality remain separately auditable.
