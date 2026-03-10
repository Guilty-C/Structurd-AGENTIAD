# Reproducibility Skeleton

This directory locks the minimal audit surface needed before any heavy execution.

## Hashing Skeleton

- Config hashing: hash canonical config files such as [configs/framework_stack.json](/home/zbr/project/lrrelevant/Structurd-AGENTIAD/configs/framework_stack.json).
- Script hashing: hash local entrypoints and adapter scripts before execution.
- Dataset manifest hashing: hash a canonical dataset manifest JSON before training or large-scale inference.

The lightweight implementation hooks live in [src/agentiad_recon/reproducibility.py](/home/zbr/project/lrrelevant/Structurd-AGENTIAD/src/agentiad_recon/reproducibility.py).

## Run Metadata Schema

Run metadata is frozen in [audit/reproducibility/run_metadata.schema.json](/home/zbr/project/lrrelevant/Structurd-AGENTIAD/audit/reproducibility/run_metadata.schema.json). Later scripts may add fields, but they should preserve the locked keys for config hashes, script hashes, dataset manifest hash, and boundary markers.

## Local vs Remote Boundary

Local:
- scaffold checks
- schema validation helpers
- small sample smoke tests
- metadata and manifest generation

Remote:
- large downloads
- SFT
- GRPO
- dataset-wide inference/evaluation
