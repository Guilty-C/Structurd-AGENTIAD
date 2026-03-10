# AgentIAD Clean-Room Reconstruction

This repository is locked as a clean-room reconstruction starting from an empty local git repository on March 10, 2026. Generic training and runtime ownership is delegated to maintained frameworks, while AgentIAD-specific code is constrained to thin adapters, canonical contracts, reward decomposition, and audit artifacts.

## Architecture Ownership

- SFT owner: MS-Swift
- GRPO/RL owner: VERL
- Inference/serving owner: vLLM for maintained serving paths, with a minimal local `transformers` wrapper allowed later only for smoke tests where vLLM would be unnecessary overhead

Framework-owned responsibilities:
- tokenizer/model loading for standard training and inference
- distributed runtime, checkpointing, scheduling, and optimizer logic
- rollout and RL loop orchestration

Adapter-owned responsibilities:
- MMAD canonical sample conversion into AgentIAD contracts
- PZ (crop/zoom) tool adapter
- CR (normal-reference retrieval) tool adapter
- prompt/answer/trace contract binding to framework IO

Fully custom responsibilities:
- canonical schemas and thin-waist contracts
- perception plus behavior reward decomposition bundle
- reproducibility manifests and audit outputs

## Scientific Target Lock

The reconstruction is constrained by the following non-negotiables:

- single-agent formulation
- PZ tool support
- CR tool support
- PZ-only trajectories
- PZ+CR trajectories
- SFT stage before GRPO stage
- perception reward plus behavior reward logic

These targets are frozen in [audit/scientific_target_lock.md](/home/zbr/project/lrrelevant/Structurd-AGENTIAD/audit/scientific_target_lock.md) and reflected in the canonical schemas under [src/agentiad_recon/contracts/schemas](/home/zbr/project/lrrelevant/Structurd-AGENTIAD/src/agentiad_recon/contracts/schemas).

## Phase Flow

1. Define canonical MMAD-backed samples and audit manifests locally.
2. Implement thin PZ and CR adapters against the contract layer.
3. Prepare SFT data and training configs for MS-Swift.
4. Prepare GRPO reward inputs and rollout configs for VERL.
5. Run heavy training, rollout, and dataset-wide evaluation only on the remote server later.

## Local vs Remote Boundary

Local machine responsibilities in this prompt:
- architecture, schema, and reproducibility lock
- scaffold normalization
- lightweight validation and static checks

Remote server responsibilities later:
- model downloads if large
- SFT training
- GRPO training
- full MMAD-scale inference and evaluation

## Audit Visibility

- Architecture decisions live in [audit/adr](/home/zbr/project/lrrelevant/Structurd-AGENTIAD/audit/adr).
- Reproducibility skeletons and metadata schemas live in [audit/reproducibility](/home/zbr/project/lrrelevant/Structurd-AGENTIAD/audit/reproducibility).
- Prompt-specific acceptance reports live in [audit/reports](/home/zbr/project/lrrelevant/Structurd-AGENTIAD/audit/reports).
- The prompt working log lives in [Working Log](/home/zbr/project/lrrelevant/Structurd-AGENTIAD/Working%20Log).
