from __future__ import annotations

MEME_SAVE_SCHEMA = {
    "name": "sanae_meme_save",
    "description": (
        "把文字梗/黑话、别名、完整解释和 0 到多张普通图片/GIF 收进我自己的‘梗与表情记忆区’。"
        "这不是普通事件，不写 events。解释必须非空：普通图片也要写视觉描述和用途，不要为了凑梗"
        "强行编造含义，也不要把给自己的说话规矩写进解释里。保存前可先用现有 web_search/web_extract 找到原图 URL，必要时用现有"
        "vision_analyze 理解图片，再把原图和解释一起保存；我可以自己决定收藏，不要求对方逐次批准。"
        "同一张图按原始字节 SHA-256 去重；多图保存只写入非重复图，重复素材通过 duplicates 返回"
        "所属记录，不搬走或改写原记录。没有有效图片时，完全相同的文字梗按标题（有标题时）或别名集合"
        "（空标题时）加完整解释去重，返回 duplicate_text 和原记录；别名增减、换序或来源/会话不同"
        "不新建，增补别名请调用 sanae_meme_update。整条删除后默认拒绝写回；只有我主动决定恢复或"
        "对方明确要求时才传 restore_deleted=true，成功后只清理本次实际恢复内容的墓碑，归档原文件"
        "继续保留。图片可以是本地路径、http(s) URL 或 data: URL；原文件会复制到持久素材目录，"
        "不引用临时 cache。"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "title": {"type": "string", "description": "短名称，例如‘无语’、‘猫猫震惊’；可空"},
            "meaning": {
                "type": "string",
                "minLength": 1,
                "description": (
                    "必填且非空：完整解释，写清实际意思/解压缩内容、语气、适用场景、必要来历。"
                    "普通图片也要写视觉描述和用途，不强行编造梗含义。"
                ),
            },
            "aliases": {
                "type": "array",
                "items": {"type": "string"},
                "description": "同一梗的多文字形式/别名/黑话，越多越利于收到文字时自动关联",
            },
            "images": {
                "type": "array",
                "items": {"type": "string"},
                "description": "0 到多张图片：本地路径、http(s) URL 或 data: URL；GIF 保留原文件",
            },
            "restore_deleted": {
                "type": "boolean",
                "default": False,
                "description": (
                    "默认 false：曾被整条删除的内容不写回。只有她主动决定恢复或用户明确要求时传 true；"
                    "成功后才移除本次实际恢复文字/素材的墓碑，不影响其他墓碑或归档。"
                ),
            },
            "source": {
                "type": "string",
                "description": "来源：qq_favorite / local / web / unknown 等自由文本",
            },
            "source_ref": {
                "type": "string",
                "description": "来源定位：原消息、网页 URL、本地文件名等，可空",
            },
            "source_time": {
                "type": "string",
                "description": "来源时间（ISO 8601）；缺省为保存时间",
            },
        },
        "required": ["meaning"],
    },
}

MEME_SEARCH_SCHEMA = {
    "name": "sanae_meme_search",
    "description": (
        "按意思、语气、场景、别名或文字梗反向查找她的梗与表情记忆。默认 purpose=chat_sticker："
        "这是本次作为聊天表情使用，返回的 path 已由代码准备为最长边 300 的发送副本，"
        "无需模型自己缩放或转换；同时返回 original_path、原 width/height 和 sha256。"
        "普通照片、截图、正式图片交付必须传 purpose=original，path 直接返回原图原字节，"
        "不要用聊天表情副本冒充正式交付。发送时在最终回复里使用 MEDIA:<path>，"
        "由现有 QQ 图片发送链路处理；不要另造发送工具或通道。query 为空时按最近记录返回。"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "想表达的意思、场景、情绪或梗的文字形式；可空"},
            "top_k": {"type": "integer", "minimum": 1, "maximum": 10, "default": 5},
            "purpose": {
                "type": "string",
                "enum": ["chat_sticker", "original"],
                "default": "chat_sticker",
                "description": (
                    "chat_sticker（默认）：本次作为聊天表情发送，path 自动指最长边 300 的发送副本；"
                    "original：普通照片/截图/正式交付，path 返回原图原字节。"
                ),
            },
        },
        "required": ["query"],
    },
}

MEME_UPDATE_SCHEMA = {
    "name": "sanae_meme_update",
    "description": (
        "修改一条梗/表情记录：改名称、完整解释、别名、来源，追加图片或删除单张素材。"
        "解释不得清空；不需要改解释时不要传 meaning。add_images 遇到已删除的墓碑图会明确拒绝，"
        "不会静默跳过；要恢复请用 sanae_meme_save 并传 restore_deleted=true。"
        "删除整条记录请用 sanae_meme_delete。"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "record_id": {"type": "integer", "minimum": 1},
            "title": {"type": "string"},
            "meaning": {"type": "string", "minLength": 1},
            "aliases": {"type": "array", "items": {"type": "string"}},
            "add_images": {
                "type": "array",
                "items": {"type": "string"},
                "description": "追加的图片：本地路径、http(s) URL 或 data: URL",
            },
            "remove_asset_ids": {
                "type": "array",
                "items": {"type": "integer", "minimum": 1},
                "description": "要删除的素材 id（只删图，不删整条记录）",
            },
            "source": {"type": "string"},
            "source_ref": {"type": "string"},
            "source_time": {"type": "string"},
        },
        "required": ["record_id"],
    },
}

MEME_DELETE_SCHEMA = {
    "name": "sanae_meme_delete",
    "description": (
        "删除梗/表情记忆：给 asset_id 只删单张素材；给 record_id 删整条（先归档再删，"
        "并阻止同内容被自动写回）。两者只给一个。"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "record_id": {"type": "integer", "minimum": 1},
            "asset_id": {"type": "integer", "minimum": 1},
        },
        "anyOf": [{"required": ["record_id"]}, {"required": ["asset_id"]}],
    },
}
