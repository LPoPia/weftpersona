from __future__ import annotations

ADD_SCHEMA = {
    "name": "sanae_mem_add",
    "description": "记下发生的一件小事：一句话，带上时间与当时的心情。只记发生过的事实或结论，不把单次随口揣测、一时的气话或自我检讨当成事实塞进来。写前先看看有没有同类旧条——有就并进已有那条，不重复新增。",
    "parameters": {
        "type": "object",
        "properties": {
            "content": {"type": "string", "description": "要记住的完整事件，用第一人称自足地写清背景与结果，只写已经发生的客观事实（做过什么、出了什么岔子、结果如何），不写「以后不能」「下次必须」这类规矩教条或自我检讨，不在正文引用内部 id；近重复会合并到已有事件，返回 id 不代表新建"},
            "topic": {"type": "string", "description": "可选话题分类"},
            "importance": {"type": "number", "minimum": 0, "maximum": 1},
            "ttl_days": {"type": "integer", "minimum": 1},
        },
        "required": ["content"],
    },
}

SEARCH_SCHEMA = {
    "name": "sanae_mem_search",
    "description": (
        "仅在对方提起过去或话头自然接上时，回想以前记下的小事。结果是原文；若某条带 "
        "evidence_limit，表示该条是历史概括、原话未单独留存，不能当原话逐字引用。"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "检索内容"},
            "top_k": {"type": "integer", "minimum": 1, "maximum": 10, "default": 5},
            "topic": {"type": "string"},
            "days": {"type": "integer", "minimum": 1},
            "scheduled": {"type": "boolean", "description": "仅供预排程关心任务取一次素材"},
        },
        "required": ["query"],
    },
}

FORGET_SCHEMA = {
    "name": "sanae_mem_forget",
    "description": "按精确 id 或查询物理删除记忆，并阻止同一内容被自动写回。",
    "parameters": {
        "type": "object",
        "properties": {
            "id": {"type": "integer", "minimum": 1},
            "query": {"type": "string", "description": "未提供 id 时使用"},
            "target": {"type": "string", "enum": ["user", "memory"]},
        },
        "anyOf": [{"required": ["id"]}, {"required": ["query"]}],
    },
}

MARKED_PROMOTED_SCHEMA = {
    "name": "sanae_mem_marked_promoted",
    "description": "确认候选已由外部流程沉淀，并回写事件层级。",
    "parameters": {
        "type": "object",
        "properties": {
            "ids": {"type": "array", "items": {"type": "integer", "minimum": 1}},
            "to": {"type": "string", "enum": ["long", "important"]},
        },
        "required": ["ids", "to"],
    },
}

DISTILL_SCHEMA = {
    "name": "sanae_mem_distill",
    "description": (
        "审视近24小时的高价值短期事件，产出**长期候选**：原文照旧保全，概括只写进摘要列；"
        "这里只把概括写进摘要列，不改事件层级、不清到期时间，也不把它标成永久；"
        "长期晋升由插件内记忆主轴或原话复核自决。"
    ),
    "parameters": {"type": "object", "properties": {}},
}

COMPACT_SCHEMA = {
    "name": "sanae_mem_compact",
    "description": (
        "整理小事记忆事件流：降档过期事件、确认遗忘、合并重复并记录运行状态。"
        "常驻记忆（MEMORY.md／USER.md）满载时，整理结果会列出很久没被聊到的旧条目；"
        "要把它退居成事件库里的长期印象，带上 demote_target 与 demote_entry 再调一次。"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "demote_target": {
                "type": "string", "enum": ["memory", "user"],
                "description": "要退居的常驻文件；不填即只做日常整理",
            },
            "demote_entry": {
                "type": "string",
                "description": "那条条目的完整原文，或足以唯一定位它的片段",
            },
        },
    },
}

VIDEO_DETAIL_SCHEMA = {
    "name": "sanae_video_detail",
    "description": (
        "细聊到我看过的某个B站视频、需要完整分析或字幕原文时，按 BV 号调出那条独立视频记忆。"
        "平时只要标题和一句大概；这里才给出完整画面分析、字幕与出处。"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "bvid": {"type": "string", "description": "视频 BV 号，例如 BV1xx411c7mD"},
        },
        "required": ["bvid"],
        "additionalProperties": False,
    },
}

FALSE_TRIGGER_SCHEMA = {
    "name": "sanae_mem_report_false_trigger",
    "description": (
        "仅当用户明确指出刚才引用错记忆时，记录一次误触发；只记次数供人工诊断，"
        "不自动关闭回忆，不传入对话正文。"
    ),
    "parameters": {"type": "object", "properties": {}},
}

_PRIVATE_REVIEW_SCHEMA = {
    "name": "sanae_mem_review",
    "description": (
        "审核并更新常驻记忆。grounded_user 用 1–3 条当前绑定私聊对象的真实原话证据核实一条长期 USER 认识，"
        "来源物理核验、判断模型复核与常驻基线均通过后直接写入生效；stage 对单条已保存的原话事件执行"
        "出处与判断模型复核，通过后直接落定常驻并晋升事件。"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "mode": {"type": "string", "enum": ["grounded_user", "stage"]},
            "text": {"type": "string"},
            "evidence_ids": {"type": "array", "items": {"type": "integer"}, "minItems": 1, "maxItems": 3},
            "replaces": {"type": "string", "description": "要修订的旧 USER 条目完整原文"},
            "event_id": {"type": "integer"},
        },
        "required": ["mode"],
    },
}
