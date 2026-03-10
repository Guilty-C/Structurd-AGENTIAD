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

Scientific constraints expressed by these schemas:

- the agent mode is always `single_agent`
- the tool surface is limited to `PZ` and `CR`
- eval traces may explicitly declare `no_tools` for the baseline path
- trajectories must declare either `pz_only` or `pz_cr`
- reward input is decomposed into perception and behavior channels
