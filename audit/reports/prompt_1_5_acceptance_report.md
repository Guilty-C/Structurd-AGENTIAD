# Prompt 1.5 Acceptance Report

- self-score: `10/10`
- Heavy compute was NOT run.
- Remote-only commands are deferred.

## Rubric Check

- PASS: built canonical `pz_only` trajectory export on top of the existing prompt/tool/trace waist in `src/agentiad_recon/sft.py`
- PASS: built canonical `pz_cr` trajectory export with explicit `CR` exemplar linkage and image-reference reinsertion
- PASS: exports one unified SFT dataset format through `sft_dataset_record.schema.json`
- PASS: validates message ordering, image references, exemplar linkage, and final answer consistency
- PASS: exports decisive-turn loss targets for the last visual operation, final reasoning step, and final answer alignment
- PASS: adds a thin MS-Swift adapter layer in `src/agentiad_recon/ms_swift_adapter.py` instead of trainer internals
- PASS: externalizes framework configs in `configs/sft_export_fixture.json`, `configs/sft_export_remote_template.json`, `configs/ms_swift_sft_fixture.json`, and `configs/ms_swift_sft_remote_template.json`
- PASS: supports LoRA, checkpoint, logging, and resume config surfaces through `ms_swift_recipe.schema.json`
- PASS: only runs lightweight local export, schema validation, unit tests, and acceptance checks
- PASS: reports honestly that MS-Swift is unavailable locally and therefore stops at format/config validation
- PASS: updates README, config docs, adapter docs, contract docs, and the root Working Log

## Artifact Evidence

- canonical Prompt 1.5 fixture export: `record_count=4`, `trajectory_modes=["pz_only","pz_cr"]`, `sample_count_per_mode=2`
- `pz_only` loss mask indices: `[3, 5, 6]` with decisive turns `[3, 5]`
- `pz_cr` loss mask indices: `[6, 8, 9]` with decisive turns `[6, 8]`
- MS-Swift adapter export: `record_count=4`, projected from the same canonical dataset with preserved image references and assistant loss flags
- runtime honesty: local probe reports that MS-Swift is not installed here, so no framework execution claim is made

## Modified Files And Why

- `src/agentiad_recon/sft.py`: added canonical Prompt 1.5 SFT trajectory export, validation, hashing, and manifest writing
- `src/agentiad_recon/ms_swift_adapter.py`: added thin MS-Swift recipe validation and runtime probing
- `src/agentiad_recon/contracts/schemas/sft_export_definition.schema.json`: added Prompt 1.5 export definition schema
- `src/agentiad_recon/contracts/schemas/sft_dataset_record.schema.json`: added unified SFT dataset row schema
- `src/agentiad_recon/contracts/schemas/sft_dataset_manifest.schema.json`: added Prompt 1.5 dataset manifest schema
- `src/agentiad_recon/contracts/schemas/ms_swift_recipe.schema.json`: added externalized MS-Swift recipe schema
- `src/agentiad_recon/contracts/schemas/ms_swift_record.schema.json`: added thin MS-Swift-facing dataset row schema
- `configs/sft_export_fixture.json`: froze the local fixture-backed SFT export definition
- `configs/sft_export_remote_template.json`: froze the remote-only full export template
- `configs/ms_swift_sft_fixture.json`: froze the local MS-Swift adapter recipe
- `configs/ms_swift_sft_remote_template.json`: froze the remote-only full-SFT template
- `tests/test_prompt_1_5_sft_export.py`: added local Prompt 1.5 trajectory and adapter tests
- `audit/check_prompt_1_5.py`: added the Prompt 1.5 acceptance gate
- `README.md`: documented trajectory reconstruction for SFT, the decisive-turn loss mask, and the remote-only boundary
- `Working Log/2026-03-10 Prompt 1.5.md`: recorded the prompt work

## Local Checks Run

- `python -m pip install jsonschema`
- `python -m compileall src audit/check_prompt_1_5.py tests/test_prompt_1_5_sft_export.py`
- `python -m unittest discover -s tests -p "test_prompt_1_5_sft_export.py"`
- `python audit/check_prompt_1_5.py`
- `python -m agentiad_recon.sft --config configs/sft_export_fixture.json --swift-recipe configs/ms_swift_sft_fixture.json --dataset-root tests/fixtures/mmad_fixture --output-root dist/outputs/prompt_1_5_manual_smoke --max-samples-per-mode 2`
- `python -m agentiad_recon.ms_swift_adapter --recipe configs/ms_swift_sft_fixture.json`

## Known remaining gaps

- Full MS-Swift SFT was not run locally and remains remote-only.
- Remote evaluation of an SFT checkpoint on real MMAD data was not run locally.
- The next milestone, GRPO reconstruction, still remains for Prompt 1.6.
