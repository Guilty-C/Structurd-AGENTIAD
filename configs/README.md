# Config Scope

This directory stores small, auditable configuration artifacts that define project ownership and later phase entrypoints. Prompt 1.1 only freezes the framework stack selection and leaves heavy runtime configs to later prompts after the MMAD, PZ, CR, and trace contracts are implemented.

Prompt 1.3 adds [baseline_non_tool_fixture.json](/home/zbr/project/lrrelevant/Structurd-AGENTIAD/configs/baseline_non_tool_fixture.json), which freezes the canonical non-tool baseline smoke-run definition. It keeps tool mode disabled, fixes prompt/parser versions, records seed handling, and documents that the mock backend is local-only while a maintained runtime adapter remains the future remote execution owner.

Prompt 1.4 adds [tool_pz_only_fixture.json](/home/zbr/project/lrrelevant/Structurd-AGENTIAD/configs/tool_pz_only_fixture.json) and [tool_pz_cr_fixture.json](/home/zbr/project/lrrelevant/Structurd-AGENTIAD/configs/tool_pz_cr_fixture.json). These freeze the canonical tool-enabled smoke-run definitions for `pz_only` and `pz_cr`, keep seeds and sample selection aligned with the Prompt 1.3 baseline config, wire the local scripted mock backend policies, and record the baseline comparison config used to produce delta-vs-baseline artifacts without changing the artifact contract family.
