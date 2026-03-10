# Canonical Contracts

This directory freezes the thin-waist schemas for AgentIAD reconstruction. The schemas are intentionally narrow so that maintained frameworks can own generic execution while custom adapters only translate between AgentIAD concepts and framework IO.

Locked schemas:

- `sample.schema.json`: one canonical MMAD-derived sample presented to a single agent, including category, anomaly candidates, mask/ROI hints, and audit metadata
- `trajectory.schema.json`: one canonical trajectory covering either `pz_only` or `pz_cr`
- `tool_call.schema.json`: one canonical tool invocation for `PZ` or `CR`, including auditable output payloads
- `final_answer.schema.json`: one canonical answer preserving `anomaly_present`, `top_anomaly`, and `visual_descriptions`
- `reward_input.schema.json`: one canonical reward bundle input with perception and behavior signals
- `artifact_manifest.schema.json`: one canonical artifact and manifest record for auditability
- `trace_record.schema.json`: one canonical audit trace for prompt messages, tool exchanges, and final answers
- `baseline_run_definition.schema.json`: one canonical non-tool baseline run config
- `baseline_prediction_record.schema.json`: one canonical per-sample baseline prediction artifact
- `baseline_metrics_report.schema.json`: one canonical metrics bundle containing per-seed, per-class, and aggregate baseline metrics
- `baseline_run_summary.schema.json`: one canonical baseline evidence summary manifest
- `tool_run_definition.schema.json`: one canonical tool-enabled run config for `pz_only` and `pz_cr`
- `tool_delta_report.schema.json`: one canonical delta-vs-baseline comparison artifact
- `sft_export_definition.schema.json`: one canonical Prompt 1.5 SFT export definition
- `sft_dataset_record.schema.json`: one unified SFT dataset row covering message order, tool events, image bindings, and loss masks
- `sft_dataset_manifest.schema.json`: one hashable manifest for either canonical or MS-Swift-facing SFT dataset exports
- `ms_swift_recipe.schema.json`: one thin MS-Swift config surface covering dataset plumbing, LoRA knobs, checkpoints, logging, and resume
- `ms_swift_record.schema.json`: one thin MS-Swift-facing dataset row projected from the canonical Prompt 1.5 record

Scientific constraints expressed by these schemas:

- the agent mode is always `single_agent`
- the tool surface is limited to `PZ` and `CR`
- eval traces may explicitly declare `no_tools` for the baseline path
- trajectories must declare either `pz_only` or `pz_cr`
- SFT exports must preserve both `pz_only` and `pz_cr` trajectories in one unified contract
- decisive-turn supervision must stay explicit for the last visual operation and final reasoning step
- reward input is decomposed into perception and behavior channels
