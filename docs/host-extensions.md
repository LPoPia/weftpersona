# 宿主侧能力依赖与改造成本说明

> **参考运行基线**：早苗当前运行的 `hermes-agent` 定制分支基于上游 [NousResearch/hermes-agent](https://github.com/NousResearch/hermes-agent) `main` 的 **`e864515d2`**（2026-10-30，在 `0.22.2` / `ab44d8ac` 之后合入上游主干；下文涉及的宿主文件路径均以此基线为准）。

WeftPersona（`sanae_mem`）在设计上把**记忆代谢、心境动力学与审核规则**放在插件内，把**可信身份、送达确认、发包隔离与群聊控场**交给宿主运行时。

上游原版 [NousResearch/hermes-agent](https://github.com/NousResearch/hermes-agent) 的标准 `MemoryProvider` / `PluginContext` 接口只覆盖基础读写和通用钩子，**不包含**上述执行边界。本文客观说明两件事：
1. 直接装在上游原版 Hermes 上时，哪些功能能跑、哪些会失效或降级；
2. 如果要让插件的全部机制生效，需要在宿主侧具体改哪些位置、接什么接口。

---

## 一、直接装在上游原版 Hermes 上会缺什么

插件内部通过 [`plugins/sanae_mem/infra/host_bridge.py`](../plugins/sanae_mem/infra/host_bridge.py) 探测宿主是否提供扩展契约。装在未改造的上游 Hermes 上时，插件可以启动，但表现如下：

| 机制 | 上游原版 Hermes 上的实际行为 | 原因 |
|---|---|---|
| **SQLite / Chroma 事件记忆、遗忘曲线、梗图库、基础检索** | 正常运行 | 仅依赖标准 `MemoryProvider` 接口与本地存储。 |
| **送达后才结算（防「未发先记」）** | **不会触发**（日志打印 `delivery receipt hook skipped`） | 上游宿主没有 `register_delivery_receipt_handler`，无法向插件回传平台确认送达的 `DeliveryReceipt`，也不能跳过发送前 `sync_turn`。 |
| **四层提示词排布与当轮瞬态块隔离** | **无法生效** | 上游宿主没有 `CognitivePromptPolicy` 排布接口，也不识别 `pre_llm_call` 返回的 `ephemeral: True`（当轮 `[我眼中的世界]` / `[我此刻心里]` 无法做到只进当轮请求报文而不落进历史记录）。 |
| **常驻记忆与技能的写前拦截分流** | **无法由宿主硬拦** | 上游 `memory_tool` / `skill_manager_tool` 写前不会查询 `MemoryMutationGate`，模型直接写 `MEMORY.md` / `USER.md` 或改技能时无法统一交给插件审核分流。 |
| **群聊旁听、发言节奏控制与过时回复丢弃** | **不可用** | 上游网关缺少 `GroupContextPolicy`、`GroupReplyPolicy` 与 `InboundMergePolicy`，无群内旁听上下文采集、无发前作废过时回复、无连发合并委派。 |
| **定时任务热同步与慢通道送达确认** | **不可用** | 上游 `cron` 不提供 `sync_plugin_cron_job` 及已确认发送账本，夜记与自由放空无法随 `shadow` 配置动态启停，也无法核实慢通道是否真发出。 |
| **同机多 HOME 隔离与 `T:` 测试隔离** | **部分受限** | 上游多处状态路径默认绑定单一根目录，且没有 `hermes_test_isolation` 隔离标记。 |

---

## 二、如果要改宿主，具体改哪里、怎么改

插件对宿主扩展能力的全部依赖都收口在 [`plugins/sanae_mem/infra/host_bridge.py`](../plugins/sanae_mem/infra/host_bridge.py)。不需要在宿主里写任何角色名、人设或记忆业务规则，只需按需在原版 `hermes-agent` 补齐以下 **6 组通用基础设施切点**（参考实现中总改动约 50+ 个宿主文件，按需裁剪时私聊场景只需前 3 组即可跑通核心闭环）：

### 1. 建立公共扩展契约注册表（新增 `hermes_host_contracts.py`）
在宿主根目录提供一个领域中立的契约模块，用 `ContextVar` + 进程级注册表暴露以下只读数据对象与策略插座：
- **可信身份**：`TurnEnvelope`（记录平台、会话、发送者、聊天类型、是否群聊/定时/子代理/测试隔离，由网关层填入，禁止从正文推断）与 `CronExecutionEnvelope`；
- **策略注册/查询入口**：`register_cognitive_prompt_policy`、`register_memory_mutation_gate`、`register_inbound_merge_policy`、`register_group_context_policy`、`register_group_reply_policy`、`register_delivery_receipt_handler`、`register_cron_completion_handler`。

### 2. 在网关入站与送达链接入可信身份与送达回执
- **涉及位置**：`gateway/session_envelope.py`、`gateway/run_turn.py`、`gateway/platforms/base.py`、`gateway/delivery_ledger.py`、`agent/memory_manager.py`
- **改法**：
  1. 收到平台消息进入回合前，由传输层用真实平台字段构造 `TurnEnvelope` 并绑定当前上下文；
  2. 在 `MemoryManager` 中支持 `sync_after_delivery` 标记——命中时本轮不在发送前调 `sync_turn`；
  3. 平台适配器确认回复真正发出后，构造 `DeliveryReceipt`（含 `turn_id`、`incoming_message_id`、`delivered_text`、真实对话记录 `transcript`、`transport_confirmed=True`），回调 `register_delivery_receipt_handler` 注册的处理器，由插件完成落库、心境续写与火花消耗。

### 3. 在提示词组装与发包链接入三层排布和瞬态隔离
- **涉及位置**：`agent/system_prompt.py`、`agent/turn_context.py`、`agent/conversation_loop.py`
- **改法**：
  1. 组装系统提示词前查询 `CognitivePromptPolicy`，按插件返回的分层顺序排布 `MEMORY` / `USER` / `BEHAVIOR.md` / `SOUL.md`；
  2. 确保 `turn_start` 钩子早于 `pre_llm_call` 执行；
  3. 处理 `pre_llm_call` 返回的 `{"context": ..., "ephemeral": True}` 时，**只拷贝一份当轮发往模型的请求消息（wire copy）并把瞬态块拼进去**，绝不写回持久化的会话历史列表或状态数据库。

### 4. 在记忆与技能工具写入前接入硬门禁
- **涉及位置**：`tools/memory_tool.py`、`tools/skill_manager_tool.py`、`tools/write_approval.py`
- **改法**：
  - 模型调用 `memory` 写入常驻文件、或调用 `skill_manage` 新建/修改技能前，先调用已注册的 `MemoryMutationGate`；若门禁拦截、改写或分流到短期事件库，则按门禁结果返回，不再直接落盘原请求。

### 5. 在群聊与入站队列接入旁听、控场与连发合并（仅群聊场景需要）
- **涉及位置**：`gateway/session_ingress_burst.py`、`gateway/group_reply_control.py`、`gateway/group_reply_history.py`、`gateway/passive_group_context.py`、`gateway/platforms/base_group_delivery.py`
- **改法**：
  1. 连续多条入站消息交由 `InboundMergePolicy` 决定合并窗口（注意会话忙碌时避免自死锁）；
  2. 群聊入站时查询 `GroupContextPolicy`，允许将授权群内未 `@` 的消息写入旁听缓冲区并随下轮喂给模型；
  3. 群聊生成回复前与发送前查询 `GroupReplyPolicy`，控制非点名参与冷却、在发送前丢弃已过时的回复，并在各平台适配器中保留准确的发言人归属与引用关系。

### 6. 定时任务同步、模型容错、多 HOME 与原子落盘（运维与稳定性增强）
- **定时任务**（`cron/jobs.py`、`cron/scheduler.py`、`cron/scheduler_delivery.py`）：暴露 `sync_plugin_cron_job` / `list_plugin_cron_jobs` / `pause_plugin_cron_job`，并在慢通道等待真实送达确认后回写账本与触发 `CronCompletion` 回调；
- **模型与视觉容错**（`agent/auxiliary_client.py`、`agent/provider_failover_state.py`）：支持辅助槽位 `allow_fallback` / `no_main_fallback`（防止后台整理失败时静默回退消耗主对话模型）及端点级异常隔离；
- **多 HOME 与测试隔离**（`hermes_test_isolation.py`、各 `gateway/` 路径读取点）：将默认路径改为动态调用 `get_hermes_home()`，并识别 `T:` 测试隔离标记；
- **原子落盘与子助手格式**：仓库已直接附带可直接 `git apply` 的补丁，见 [`patches/hermes-agent/README.md`](../patches/hermes-agent/README.md)。

---

## 三、最小可行裁剪建议

如果你只跑**一对一私聊**、暂时不做多平台群聊和复杂定时任务，不需要一次性全改完，只补 **第 1 步（最小 `TurnEnvelope` + 3 个注册口）+ 第 2 步（送达回执）+ 第 3 步（`ephemeral` 瞬态隔离）+ 第 4 步（写前门禁）**，就能把核心记忆防偷写、三层提示词排布和心境连续性完整跑起来；具体每个回调的入参与字段以 [`plugins/sanae_mem/infra/host_bridge.py`](../plugins/sanae_mem/infra/host_bridge.py) 为准。
