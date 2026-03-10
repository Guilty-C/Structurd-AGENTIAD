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

## Canonical Runtime Waist

Prompt 1.2 extends the scaffold with one canonical local runtime waist:

1. MMAD sample in through [src/agentiad_recon/mmad.py](/home/zbr/project/lrrelevant/Structurd-AGENTIAD/src/agentiad_recon/mmad.py)
2. prompt contract out through [src/agentiad_recon/prompting.py](/home/zbr/project/lrrelevant/Structurd-AGENTIAD/src/agentiad_recon/prompting.py)
3. deterministic tool execution through [src/agentiad_recon/tooling.py](/home/zbr/project/lrrelevant/Structurd-AGENTIAD/src/agentiad_recon/tooling.py)
4. strict final answer parsing back into the canonical final-answer schema
5. audit-ready traces through [src/agentiad_recon/traces.py](/home/zbr/project/lrrelevant/Structurd-AGENTIAD/src/agentiad_recon/traces.py)

The local plumbing is intentionally narrow:

- the MMAD layer indexes real dataset paths when available, but Prompt 1.2 only validates against a tiny fixture dataset
- PZ is a deterministic crop adapter, not a vision model
- CR is a deterministic same-category normal exemplar selector, not an ANN retrieval system
- prompt and answer handling are contract-first and versioned for later framework integration

## Phase Flow

1. Define canonical MMAD-backed samples and audit manifests locally.
2. Implement thin PZ and CR adapters against the contract layer.
3. Prepare SFT data and training configs for MS-Swift.
4. Prepare GRPO reward inputs and rollout configs for VERL.
5. Run heavy training, rollout, and dataset-wide evaluation only on the remote server later.

## Local vs Remote Boundary

Local machine responsibilities in this prompt:
- architecture and contract lock
- MMAD canonical sample indexing and export
- deterministic PZ and CR tool adapters
- prompt, answer, and trace contract implementation
- lightweight fixture validation and static checks

Remote server responsibilities later:
- model downloads if large
- SFT training
- GRPO training
- full MMAD-scale inference and evaluation

## Fixture vs Real Dataset Paths

- Real dataset path: pass an explicit MMAD root or set `AGENTIAD_MMAD_ROOT`; the indexer will only report what it can actually discover on disk.
- Fixture validation path: use the tiny MMAD-style fixture under [tests/fixtures/mmad_fixture](/home/zbr/project/lrrelevant/Structurd-AGENTIAD/tests/fixtures/mmad_fixture) to smoke-test indexing, export, PZ, CR, prompt contracts, answer parsing, and trace serialization locally.
- Deferred work: baseline inference, tool-enabled inference loops, SFT export at scale, GRPO rollout, and dataset-wide evaluation remain remote/server-phase work.

## Audit Visibility

- Architecture decisions live in [audit/adr](/home/zbr/project/lrrelevant/Structurd-AGENTIAD/audit/adr).
- Reproducibility skeletons and metadata schemas live in [audit/reproducibility](/home/zbr/project/lrrelevant/Structurd-AGENTIAD/audit/reproducibility).
- Prompt-specific acceptance reports live in [audit/reports](/home/zbr/project/lrrelevant/Structurd-AGENTIAD/audit/reports).
- The prompt working log lives in [Working Log](/home/zbr/project/lrrelevant/Structurd-AGENTIAD/Working%20Log).
