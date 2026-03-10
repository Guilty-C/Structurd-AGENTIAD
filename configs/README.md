# Config Scope

This directory stores small, auditable configuration artifacts that define project ownership and later phase entrypoints. Prompt 1.1 only freezes the framework stack selection and leaves heavy runtime configs to later prompts after the MMAD, PZ, CR, and trace contracts are implemented.

Prompt 1.3 adds [baseline_non_tool_fixture.json](/home/zbr/project/lrrelevant/Structurd-AGENTIAD/configs/baseline_non_tool_fixture.json), which freezes the canonical non-tool baseline smoke-run definition. It keeps tool mode disabled, fixes prompt/parser versions, records seed handling, and documents that the mock backend is local-only while a maintained runtime adapter remains the future remote execution owner.
