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

Prompt 1.3 extends the same waist with a single canonical non-tool baseline path:

- baseline prompts are built from the same sample records with tools disabled by omission
- backend execution flows through a thin interface in [src/agentiad_recon/backends.py](/home/zbr/project/lrrelevant/Structurd-AGENTIAD/src/agentiad_recon/backends.py)
- local smoke validation uses a clearly labeled mock backend, not real model inference
- parsed answers, audit traces, prediction records, metrics, and manifests are written through the same contract layer

Prompt 1.4 extends that exact runner into one canonical tool-enabled path:

- `pz_only` enables localized crop/zoom through `PZ` and rejects `CR`
- `pz_cr` enables the same `PZ` crop loop plus same-category normal-reference retrieval through `CR`
- tool calls are parsed, executed, and reinserted through [src/agentiad_recon/tooling.py](/home/zbr/project/lrrelevant/Structurd-AGENTIAD/src/agentiad_recon/tooling.py), not hardcoded shortcuts
- tool-enabled metrics, traces, and delta-vs-baseline artifacts are written through the same evaluator family in [src/agentiad_recon/evaluation.py](/home/zbr/project/lrrelevant/Structurd-AGENTIAD/src/agentiad_recon/evaluation.py)

## Non-Tool Baseline

The Prompt 1.3 baseline is the first canonical inference/evaluation path:

1. load canonical samples from [src/agentiad_recon/mmad.py](/home/zbr/project/lrrelevant/Structurd-AGENTIAD/src/agentiad_recon/mmad.py)
2. build a no-tool prompt with [src/agentiad_recon/prompting.py](/home/zbr/project/lrrelevant/Structurd-AGENTIAD/src/agentiad_recon/prompting.py)
3. submit requests through the thin backend interface in [src/agentiad_recon/backends.py](/home/zbr/project/lrrelevant/Structurd-AGENTIAD/src/agentiad_recon/backends.py)
4. parse the final answer with the same strict parser used elsewhere
5. store traces and evaluator artifacts through [src/agentiad_recon/baseline.py](/home/zbr/project/lrrelevant/Structurd-AGENTIAD/src/agentiad_recon/baseline.py) and [src/agentiad_recon/evaluation.py](/home/zbr/project/lrrelevant/Structurd-AGENTIAD/src/agentiad_recon/evaluation.py)

The baseline keeps tools fully disabled. `PZ` and `CR` are not available in this mode and will only be reintroduced in the later tool-augmented inference milestone.

## Tool-Augmented Inference

The Prompt 1.4 tool-enabled path reuses the same sample records, parser, traces, and evaluator outputs as the baseline path. The only change is the allowed tool surface and the bounded multi-turn loop:

1. build a versioned tool-enabled prompt with [src/agentiad_recon/prompting.py](/home/zbr/project/lrrelevant/Structurd-AGENTIAD/src/agentiad_recon/prompting.py)
2. send the canonical message history through the same thin backend interface in [src/agentiad_recon/backends.py](/home/zbr/project/lrrelevant/Structurd-AGENTIAD/src/agentiad_recon/backends.py)
3. parse assistant `<tool_call>` blocks with [src/agentiad_recon/tooling.py](/home/zbr/project/lrrelevant/Structurd-AGENTIAD/src/agentiad_recon/tooling.py)
4. execute `PZ` or `CR`, reinsert auditable tool outputs into history, and continue until a strict `<answer>` block is produced or the turn bound is hit
5. export per-sample prediction records, tool traces, per-seed metrics, per-class metrics, mean/std aggregation, and delta-vs-baseline artifacts through [src/agentiad_recon/baseline.py](/home/zbr/project/lrrelevant/Structurd-AGENTIAD/src/agentiad_recon/baseline.py)

Mode split:

- `pz_only`: `PZ` is allowed, `CR` is rejected by contract, and tool traces should show non-zero `PZ` usage with zero `CR` usage
- `pz_cr`: `PZ` and `CR` are both allowed, and the same evaluator additionally exports comparative tool-usage statistics and structural delta-vs-baseline artifacts

Prompt 1.4 remains local-only and mock-backed. The scripted backend is only there to validate loop structure, stop conditions, tool usage accounting, and artifact generation. It is not real VLM inference.

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
- non-tool baseline inference and evaluator plumbing
- tool-enabled `pz_only` and `pz_cr` smoke-validation plumbing
- lightweight fixture validation and static checks

Remote server responsibilities later:
- model downloads if large
- maintained-runtime baseline execution on real models
- maintained-runtime tool-enabled inference runs
- SFT training
- GRPO training
- full MMAD-scale inference and evaluation

## Fixture vs Real Dataset Paths

- Real dataset path: pass an explicit MMAD root or set `AGENTIAD_MMAD_ROOT`; the indexer will only report what it can actually discover on disk.
- Fixture validation path: use the tiny MMAD-style fixture under [tests/fixtures/mmad_fixture](/home/zbr/project/lrrelevant/Structurd-AGENTIAD/tests/fixtures/mmad_fixture) to smoke-test indexing, export, PZ, CR, prompt contracts, answer parsing, and trace serialization locally.
- Baseline fixture definition: [configs/baseline_non_tool_fixture.json](/home/zbr/project/lrrelevant/Structurd-AGENTIAD/configs/baseline_non_tool_fixture.json) freezes the local-only mock-backed baseline run definition.
- Tool-enabled fixture definitions: [configs/tool_pz_only_fixture.json](/home/zbr/project/lrrelevant/Structurd-AGENTIAD/configs/tool_pz_only_fixture.json) and [configs/tool_pz_cr_fixture.json](/home/zbr/project/lrrelevant/Structurd-AGENTIAD/configs/tool_pz_cr_fixture.json) freeze the local-only scripted smoke runs for `pz_only` and `pz_cr`.
- Delta-vs-baseline artifacts from Prompt 1.4 are structural local smoke evidence only; they do not claim tool quality gains on real models.
- Deferred work: real maintained-runtime baseline execution, real maintained-runtime tool-enabled inference, SFT export at scale, GRPO rollout, and dataset-wide evaluation remain remote/server-phase work.

## Audit Visibility

- Architecture decisions live in [audit/adr](/home/zbr/project/lrrelevant/Structurd-AGENTIAD/audit/adr).
- Reproducibility skeletons and metadata schemas live in [audit/reproducibility](/home/zbr/project/lrrelevant/Structurd-AGENTIAD/audit/reproducibility).
- Prompt-specific acceptance reports live in [audit/reports](/home/zbr/project/lrrelevant/Structurd-AGENTIAD/audit/reports).
- The prompt working log lives in [Working Log](/home/zbr/project/lrrelevant/Structurd-AGENTIAD/Working%20Log).
