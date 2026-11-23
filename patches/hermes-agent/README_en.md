# Hermes host patches

[简体中文](README.md) | **English**

## Honor the task's requested output format

[`delegate-output-format.patch`](delegate-output-format.patch) resolves conflicting completion instructions: when a task requests JSON or another strict format, the child returns only that format. Tasks without a requested format retain the concise completion report. Result parsing, permissions, scoring thresholds, and model-call counts are unchanged.

This patch targets Hermes checkouts containing `tools/delegate_tool_progress.py` and `_COMPLETION_INSTRUCTIONS`. Run from the **Hermes repository root**, replacing the patch path with your checkout of this repository:

```bash
git apply --check /path/to/WeftPersona/patches/hermes-agent/delegate-output-format.patch
git apply /path/to/WeftPersona/patches/hermes-agent/delegate-output-format.patch
```

Restart the running Hermes process after applying the patch so new children load the updated instructions. A host with equivalent behavior needs no patch; inspect host differences if the application check fails.

The patch was checked against the pre-fix host source and reproduces the version validated with real worker replays and an independent review. See [NOTICE](../../NOTICE) for upstream attribution and licensing.

## Shared Atomic Persistence

Copy [`host/hermes_atomic.py`](../../host/hermes_atomic.py) into the Hermes source root, then apply [`atomic-file-io.patch`](atomic-file-io.patch) there:

```bash
git apply --check /path/to/WeftPersona/patches/hermes-agent/atomic-file-io.patch
git apply /path/to/WeftPersona/patches/hermes-agent/atomic-file-io.patch
```

The existing public `utils` interfaces re-export the same implementation, including the primitives used by host YAML writers. Permissions, symlinks, syncing and replacement behavior remain intact. Standalone scripts can load the lightweight module without model dependencies. If the check fails, inspect the host version instead of replacing its entire `utils.py`. Skip installation when the host already provides the same module and re-exports.
