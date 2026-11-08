# WeftPersona 系统架构与对话记忆链路

> 🌐 **Language**: **简体中文** | [**English**](en/architecture.md)

本文档说明 **WeftPersona（`sanae_mem`）** 在 Hermes 宿主上的组件契约、数据真源分工与单轮对话的完整处理流程。背后的认知与心理学设计理念详见 [`how-to-simulate-a-living-person.md`](how-to-simulate-a-living-person.md)。

---

## 一、核心组件与宿主钩子契约

| 组件 / 模块 | 入口文件 | 暴露工具 | 宿主钩子与职责契约 |
|---|---|---|---|
| **记忆与心境主入口** | [`plugins/sanae_mem/plugin_entry.py`](../plugins/sanae_mem/plugin_entry.py) | `sanae_mem_add`、`sanae_mem_search`、`sanae_mem_forget`、`sanae_mem_compact`、`sanae_mem_marked_promoted`、`sanae_mem_distill`、`sanae_mem_report_false_trigger` | 注册 `MemoryProvider`（`sync_after_delivery = True`）；`system_prompt_block` 注入 `BEHAVIOR.md` 渲染区；`pre_llm_call` 并发执行按需联想与梗图识别；`pre_tool_call` 执行常驻记忆与受控文件写护栏 |
| **遗忘曲线与三阶坍缩** | [`plugins/sanae_mem/curve.py`](../plugins/sanae_mem/curve.py)、[`event_maintenance.py`](../plugins/sanae_mem/event_maintenance.py) | 经 `sanae_mem_compact` 触发 | 跨日（≥ 12h）召回强化半衰期（幂律增长，上限 120 天）；按激活度自动执行「鲜活原话 → `gist_faded`（删前归档原话只留摘要） → `dormant`（深层休眠）或自然归档」 |
| **常驻记忆代谢与分流** | [`plugins/sanae_mem/resident.py`](../plugins/sanae_mem/resident.py)、[`llm_guard.py`](../plugins/sanae_mem/llm_guard.py) | 拦截宿主 `memory` 工具写入 | 追踪常驻条目对话触达时间，容量 ≥ 85% 时检出 45 天未触达冷条目供退居事件库；写入口用快模型拦截系统元规则、单次琐事与自我检讨，分流至短期心里话或短期事件 |
| **心境动力学（快变量）** | [`plugins/sanae_mem/inner_state.py`](../plugins/sanae_mem/inner_state.py)、[`inner_wiring.py`](../plugins/sanae_mem/inner_wiring.py) | `sanae_inner_state` | 私聊回复确认送达后或受信任个人 `cron` 完成后异步更新 1～3 句第一人称 `[我此刻心里]`、慢变量（`tension`/`closeness`/`energy`）、待办原话摘录（`recent_words`）、未点名联想搭桥（`unspoken_link`）与 DMN 远距碰撞（`spark`）；按真实时间与跨长睡眠 `24h` 方向余温自然衰减 |
| **慢变量印象（慢变量）** | [`plugins/sanae_mem/impression.py`](../plugins/sanae_mem/impression.py) | `sanae_impression` | 复用送达后同一次 Flash 调用，仅在出现长期近况或相处方式变化且有原话引文证据时慢速修订 `[我最近眼里的他]`；每轮在 `Tier 2` 动态上下文的 `[我眼中的世界]` 中实时读取最新版，不破坏 `Tier 0` 系统提示前缀缓存 |
| **每日上下文与好奇心** | [`plugins/sanae_mem/daily_context.py`](../plugins/sanae_mem/daily_context.py)、[`daily_semantics.py`](../plugins/sanae_mem/daily_semantics.py) | 内部钩子自动维护 | 抽取当日动态（`DAILY_FACT`）、客观受限（`ACTIVE_CONSTRAINT`）、好奇心小尾巴（`OPEN_THREAD`）及角色亲口答应的待办承诺（`self_statements`） |
| **梗与表情包记忆区** | [`plugins/sanae_mem/memes.py`](../plugins/sanae_mem/memes.py) | `sanae_meme_save/search/update/delete` | 独立存储文字梗与表情包原图（SHA-256 去重与入站直认免调视觉），聊天发送自动准备 300px 缩略副本（保留 GIF 帧率与透明度） |

---

## 二、数据真源分工（相对于 `$HERMES_HOME`）

| 数据层 | 默认路径 | 职责与生命周期 |
|---|---|---|
| **事件、心境、印象与梗库** | `sanae-mem/sanae.db` | SQLite 唯一权威真源：存情景事件（`events` / `event_occurrences`）、第一人称心里话版本链（`inner_state_revisions`）、慢变量印象版本链与会话视图（`impression_revisions` / `impression_session_views`）、每日动态（`daily_entries`）与梗表（`meme_records`） |
| **向量与全文检索索引** | `sanae-mem/chroma/` ＋ SQLite FTS5 | 检索派生层（本地 `bge-small-zh-v1.5` 嵌入），随时可从 `sanae.db` 重建 |
| **常驻长期记忆** | `memories/MEMORY.md`、`memories/USER.md` | 会话级快照注入：只存跨越数月仍决定如何相处的概括事实与长期稳定认知 |
| **程序性行为层与主体内核** | `BEHAVIOR.md`、`SOUL.md` | [`BEHAVIOR.md`](../prompts/BEHAVIOR.md) 置于系统提示前段；[`SOUL.md`](../prompts/SOUL.md) 与角色设定置于系统提示最末尾 |
| **删前归档冷库** | `memory-archive/` | 任何事件遗忘、原话坍缩或常驻退居前强制先落盘归档（归档失败则中止删除） |

---

## 三、单轮对话完整走向

1. **入站合并与上下文绑定**：
   - 接入层合并同一发送者短时间内的连发图文，计算距离上次互动的真实时间差；已知表情包通过原图字节 SHA-256 直接命中释义，省去多模态视觉调用开销。
2. **回合前动态读取（`pre_llm_call` & `current_turn_context`）**：
   - 读取当前会话首轮或长间隔后的 `[当日上下文]`、跨场合自身未消费发言 `[我刚才在别处说过]`；
   - 按语境门控触发经历联想（普通闲聊遵循省力先验不翻旧账；确需联想时在 3 秒窗口内执行查询规划与混合检索，超时则直接跳过，不退化为机械关键词硬搜）；
   - 等待最多 3 秒获取上一轮送达后正在整理的最新 `[我此刻心里]`（含自上次回复以来刚完成的自主活动事实），拼入当轮用户消息动态上下文，且不写入历史消息的持久化 `api_content`。
3. **系统提示词「工程前置、灵魂尾置」装配（`wire_system_prompt`）**：
   - 前段固定：技能索引 → `MEMORY.md` / `USER.md` → `BEHAVIOR.md` 渲染区 → 环境时间；
   - 末尾紧贴对话：`SOUL.md` → 角色专属设定 → 会话级冻结的 `[我最近眼里的他]`。
4. **工具调用护栏（`pre_tool_call`）**：
   - 若主脑调用 `memory` 写常驻记忆，先经 `llm_guard.py` 旁路判断：将一时气话、主观揣测或自我立规矩静默分流为短期心里话或短期中性事件；
   - 对 `write_file` / `patch` 按真实路径校验受控记忆文件（`MEMORY.md`、`USER.md`、`BEHAVIOR.md`），防止绕过 `memory` 工具直接覆写受控文件。
5. **送达回执确认与后台异步整理（`sync_turn` & `update_inner_state`）**：
   - 仅当回复气泡在聊天端**完整确认送达**后，后台线程才启动每日语义抽取（`daily_semantics`）与内心/印象整理（`inner_state` + `impression` + `spark`）；若发送中断或失败，绝不将未送达内容写入记忆。
