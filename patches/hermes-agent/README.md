# Hermes 宿主补丁

**简体中文** | [English](README_en.md)

## 按任务指定的格式返回结果

[`delegate-output-format.patch`](delegate-output-format.patch) 修复子助手输出格式与宿主默认汇报样式不一致的问题：任务明确要求 JSON 等严格格式时，只返回该格式内容；任务没有指定格式时，仍给简明工作汇报。它不修改结果解析器、权限、评分门槛或模型调用次数。

适用于包含 `tools/delegate_tool_progress.py` 和 `_COMPLETION_INSTRUCTIONS` 的 Hermes 源码版本。在 **Hermes 源码仓库根目录**执行，将下方补丁路径替换为本仓的实际位置：

```bash
git apply --check /path/to/WeftPersona/patches/hermes-agent/delegate-output-format.patch
git apply /path/to/WeftPersona/patches/hermes-agent/delegate-output-format.patch
```

应用后重启正在运行的 Hermes 进程，使后续子助手加载新提示。若宿主已有同等修复，无需重复应用；检查不通过时先核对宿主差异。

补丁已对修复前的宿主文件验证可应用，结果与通过真实 worker 回放及独立复核的版本一致。上游来源与许可见 [NOTICE](../../NOTICE)。

## 共用原子落盘底座

将 [`host/hermes_atomic.py`](../../host/hermes_atomic.py) 复制到 Hermes 源码根目录，然后在该目录应用 [`atomic-file-io.patch`](atomic-file-io.patch)：

```bash
git apply --check /path/to/WeftPersona/patches/hermes-agent/atomic-file-io.patch
git apply /path/to/WeftPersona/patches/hermes-agent/atomic-file-io.patch
```

补丁让既有 `utils` 公开接口重导出同一实现，YAML 写入等宿主调用者也共用该底座。它保留现有权限、符号链接、同步和替换行为；独立脚本可直接加载轻量模块，无须模型依赖。检查不通过时先核对宿主版本，不覆盖整份 `utils.py`；宿主已有相同底座和重导出时无需重复应用。
