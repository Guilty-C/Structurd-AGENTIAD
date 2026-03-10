# Prompt 1.4 Acceptance Report

- self-score: `10/10`
- Heavy compute was NOT run.
- Remote-only commands are deferred.

## Rubric Check

- PASS: built one canonical tool-enabled inference path by extending `src/agentiad_recon/baseline.py` instead of introducing a second evaluator or runner family
- PASS: supports both `pz_only` and `pz_cr` through frozen configs in `configs/tool_pz_only_fixture.json` and `configs/tool_pz_cr_fixture.json`
- PASS: reuses the canonical sample pipeline, prompt family, strict final-answer parser, trace storage, and evaluation artifact family from Prompts 1.2 and 1.3
- PASS: executes a bounded multi-turn tool loop through the existing tool contracts in `src/agentiad_recon/tooling.py`
- PASS: exports per-sample prediction records, tool traces, per-seed metrics, per-class metrics, mean/std aggregation, and delta-vs-baseline artifacts
- PASS: exports auditable tool-usage statistics, including non-zero `PZ` frequency in `pz_only` and non-zero `PZ` plus `CR` frequency in `pz_cr`
- PASS: includes a Prompt 1.4 tool gate via `audit/check_prompt_1_4.py` that verifies `PZ` calls, `CR` calls in `pz_cr`, `CR` rejection in `pz_only`, malformed tool-call handling, trace generation, and delta artifact generation
- PASS: README and root Working Log were updated for Prompt 1.4
- PASS: touched core code files keep explanation headers and dispersed comments
- PASS: only lightweight local compile/tests/mock-backed smoke runs were executed
- PASS: produced detailed acceptance evidence without claiming real model quality or full-MMAD reproduction

## Artifact Evidence

- `pz_only` smoke summary: `parser_valid=4`, `schema_valid=4`, `sample_count=4`, `PZ=4`, `CR=0`, `total_calls=4`
- `pz_cr` smoke summary: `parser_valid=4`, `schema_valid=4`, `sample_count=4`, `PZ=4`, `CR=4`, `total_calls=8`
- `pz_only` delta-vs-baseline: parser/schema/anomaly/top-anomaly mean deltas `+0.25`; tool usage delta `toolcall_rate=1.0`, `PZ=1.0`, `CR=0.0`
- `pz_cr` delta-vs-baseline: parser/schema/anomaly/top-anomaly mean deltas `+0.25`; tool usage delta `toolcall_rate=1.0`, `PZ=1.0`, `CR=1.0`
- All deltas above are local scripted smoke evidence only. They are not claims about real model quality.

## Modified Files And Why

- `src/agentiad_recon/backends.py`: added scripted tool-aware backend policies for `pz_only`, `pz_cr`, invalid-CR, and malformed-tool smoke behaviors
- `src/agentiad_recon/baseline.py`: extended the canonical runner into one tool-enabled execution loop with bounded turns, trace writing, and delta-vs-baseline generation
- `src/agentiad_recon/evaluation.py`: added tool-usage metrics and tool delta reporting without creating a second evaluator family
- `src/agentiad_recon/prompting.py`: froze the Prompt 1.4 tool-enabled prompt contract with preserved `<think>` plus `<answer>` structure
- `src/agentiad_recon/tooling.py`: preserved the tool protocol and added image-ref reinsertion for crop/reference outputs
- `src/agentiad_recon/contracts/README.md`: documented tool-enabled run/delta schemas
- `src/agentiad_recon/contracts/schemas/baseline_prediction_record.schema.json`: extended prediction records with tool mode and normalized tool-usage payloads
- `src/agentiad_recon/contracts/schemas/baseline_metrics_report.schema.json`: extended the metrics family with toolcall rate and per-tool frequency
- `src/agentiad_recon/contracts/schemas/baseline_run_summary.schema.json`: extended run summaries with tool usage and delta artifact linkage
- `src/agentiad_recon/contracts/schemas/tool_run_definition.schema.json`: added the canonical tool-enabled run definition schema
- `src/agentiad_recon/contracts/schemas/tool_delta_report.schema.json`: added the canonical delta-vs-baseline report schema
- `configs/tool_pz_only_fixture.json`: froze the local `pz_only` smoke run definition
- `configs/tool_pz_cr_fixture.json`: froze the local `pz_cr` smoke run definition
- `configs/README.md`: documented Prompt 1.4 configs and baseline alignment
- `tests/test_prompt_1_4_tool_inference.py`: added local smoke tests for prompt separation, tool-loop round-trips, invalid-tool rejection, malformed tool calls, and artifact generation
- `audit/check_prompt_1_4.py`: added the prompt-level acceptance gate
- `README.md`: documented the tool-enabled path, mode split, and local-vs-remote boundary
- `Working Log/2026-03-10 Prompt 1.4.md`: recorded the prompt work

## Local Checks Run

- `python -m compileall src audit/check_prompt_1_4.py tests/test_prompt_1_4_tool_inference.py`
- `python audit/check_prompt_1_3.py`
- `python -m unittest discover -s tests -p 'test_prompt_1_4_tool_inference.py'`
- `env PYTHONPATH=src python -m agentiad_recon.baseline --config configs/tool_pz_only_fixture.json --dataset-root tests/fixtures/mmad_fixture --artifact-root /tmp/agentiad_prompt_1_4_pz_only_smoke --max-samples 2`
- `env PYTHONPATH=src python -m agentiad_recon.baseline --config configs/tool_pz_cr_fixture.json --dataset-root tests/fixtures/mmad_fixture --artifact-root /tmp/agentiad_prompt_1_4_pz_cr_smoke --max-samples 2`
- `python audit/check_prompt_1_4.py`

## Known remaining gaps

- Real maintained-runtime tool-enabled inference is still deferred.
- Full-MMAD evaluation is still deferred.
- Trajectory export for SFT remains the next milestone rather than part of Prompt 1.4.
