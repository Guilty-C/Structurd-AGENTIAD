# ADR-0001: AgentIAD Clean-Room Architecture Lock

## Status

Accepted on 2026-03-10 for Prompt 1.1.

## Context

The repository started as an empty local git repository with no authoritative project files, no prior commits, and no trusted legacy pipeline. This checkpoint therefore establishes a clean-room baseline instead of continuing any ad hoc scripts.

## Repository Authority Audit

Keep:
- `.git/` as the repository root history container
- the new clean-room scaffold introduced in Prompt 1.1

Quarantine:
- none, because no legacy project material exists locally yet

Ignore:
- generated outputs under `dist/outputs/`
- generated paper-facing artifacts under `dist/paper_artifacts/`
- Python bytecode caches

Danger notes:
- there is no maintained local trainer/runtime stack in this repository yet
- any future one-off script proliferation should be rejected unless it clearly wraps a maintained framework entrypoint

## Decision

The project stack is locked as follows:

- SFT owner: MS-Swift
- GRPO/RL owner: VERL
- inference/serving owner: vLLM, with a minimal local `transformers` wrapper permitted only for lightweight smoke paths where full serving is unnecessary

## Ownership Map

Framework-owned:
- model loading, tokenization, checkpointing, optimization, scheduling
- distributed training and rollout orchestration
- serving/runtime execution for standard inference paths

Adapter-owned:
- MMAD canonical data mapping
- PZ adapter
- CR adapter
- prompt, answer, and trajectory binding into framework IO formats

Fully custom:
- canonical schemas
- reward decomposition bundle
- audit manifests and reproducibility metadata

## Scientific Constraints

The architecture must preserve:

- single-agent formulation
- PZ crop/zoom tool
- CR normal-reference retrieval tool
- `pz_only` trajectories
- `pz_cr` trajectories
- SFT before GRPO
- perception reward plus behavior reward

## Rationale

This satisfies the maintained-framework rule because all generic trainer, RL, and serving machinery stays owned by mature external frameworks. AgentIAD-specific code is limited to thin adapters and auditable contracts, which reduces duplicated infrastructure and keeps the reconstruction reviewable.
