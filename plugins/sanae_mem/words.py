from __future__ import annotations

import re

_DIMS = 512
# Chroma 集合的向量维度固定；更换维度时使用新集合名。
_COLLECTION = "sanae_events_r6"
_EMBED_MODEL_SUBDIR = "models/bge-small-zh-v1.5"
_EMBED_QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："
_EMBED_MAX_TOKENS = 512
_WRITE_IMPORTANCE_FLOOR = 0.45
_IMPORTANCE_CEILING = 0.95
_DUPLICATE_IMPORTANCE_INCREMENT = 0.2

_RECALL_LEXICAL_WEIGHT = 0.65
_RECALL_SEMANTIC_WEIGHT = 0.35
_RECALL_CHROMA_BONUS = 0.1
_CHROMA_BONUS_MIN_COSINE = 0.2
_RECALL_RECENCY_WEIGHT = 0.15
_RECALL_RECENCY_DECAY_DAYS = 30

_EXPLICIT_INJECT_MIN_RELEVANCE = 0.34
_EXPLICIT_INJECT_MIN_SEMANTIC = 0.60

_GAP_RECALL_WINDOW_DAYS = 14
_INJECTION_RECALL_WINDOW_DAYS = 30

_CONFIDENCE_DECAY_AFTER_DAYS = 30
_SCHEDULE_TTL_DAYS = 7
_DEFAULT_EVENT_TTL_DAYS = 14

_GAP_INJECT_MAX_CHARS = 100
_TRUST_WORDS = ("记住", "别忘了", "明天记得", "帮我", "我要", "想买", "打算", "考", "背单词")
_EMOTION_WORDS = ("烦", "累", "崩溃", "焦虑", "开心", "难过", "失眠")
_RELATION_WORDS = ("内心", "走进", "珍视", "庇护", "重要的人", "珍惜", "深处", "信任", "依靠")
_IMPORTANT_WORDS = ("面试", "考试", "大日子", "重要", "庇护", "约定")
_COMMAND_PREFIXES = ("查一下", "搜索", "运行")
_EXPLICIT_PAST_WORDS = ("上次", "之前", "那天", "那个", "还记得吗", "还记得", "我说过", "提过")
_TRIVIAL_MESSAGES = ("你好", "您好", "嗨", "哈喽", "在吗", "早", "早安", "晚安", "嗯", "嗯嗯", "哦", "好", "好的")
_BUSY_WORDS = ("忙", "工作", "开会", "加班", "赶工", "赶时间", "没空", "先处理", "晚点聊")
_TECH_WORDS = ("代码", "函数", "接口", "数据库", "服务器", "部署", "Python", "JavaScript", "API", "Git")
_SCHEDULE_WORDS = ("明天", "后天", "下周", "到时候", "日程", "预约", "面试", "考试")
_PROMOTION_WORDS = ("记住", "别忘", "约好了", "很重要", "我最")
_CONFLICT_WORDS = ("吵架", "争吵", "冲突", "闹矛盾", "分手", "绝交")
_IMPACT_WORDS = ("崩溃", "非常", "特别", "极其", "严重", "影响", "后果", "失去", "再也")
_DISTILL_TOPICS = {"关系", "身份", "偏好"}
_SYSTEM_META_WORDS = (
    "系统提示", "提示词", "工具调用", "数据库记录", "服务器状态", "服务器部署",
    "部署完成", "运维任务", "日志输出", "报错信息", "API调用", "token用量",
    "上下文窗口", "cron任务", "插件配置",
)
_TOPIC_WORDS = {
    "关系": _RELATION_WORDS,
    "身份": ("身份", "信仰者", "自我定位", "角色认同"),
    "偏好": ("喜欢", "不喜欢", "偏好", "习惯"),
    "购物": ("买", "购", "下单", "网购", "快递", "到货", "发货", "退货"),
    "学习": ("学习", "考试", "背单词", "作业", "复习", "课程"),
    "情绪": _EMOTION_WORDS,
    "健康": ("生病", "头疼", "胃疼", "失眠", "医院", "吃药", "发烧"),
    "生活": ("吃饭", "睡觉", "出门", "回家", "搬家", "旅行", "约会"),
}
_CN_RE = re.compile(r"[\u3400-\u9fff]")
_EN_RE = re.compile(r"[A-Za-z0-9_]+")
_META_TOKEN_RE = re.compile(
    r"\b(?:hermes|harness|codex|cron|plugin|token|provider|user\.md|memory\.md|memory|user)\b",
    re.IGNORECASE,
)
_META_EXCLUDE_WORDS = (
    "记忆系统", "事件流", "蒸馏", "蒸馏层",
    "插件", "情绪模块", "记忆温度", "增添情绪",
    "影响后续言行",
)

_NOISE_OBSERVE_RE = re.compile(
    r"(?:^(?:用户|对方|她|他)(?:提及|转述|表示|回应)|语气(?:中|里)?(?:带着|透着|显出))"
)
_DEV_ROUTING_NOTE_RE = re.compile(
    r"(?:(?:这条|本条|该条)属于|(?:待|等)\s*[A-Za-z0-9_.-]+\.(?:md|py|json|ya?ml)\b|落地后分流|分流(?:进|到))"
)
_TOOL_WHISPER_HEAD_RE = re.compile(
    r"^(?:先)?(?:获取|查询|读取|调取|取)(?:一下)?(?:当前|现在|系统)?(?:的)?(?:时间|日期|时间戳)(?=[，,。！!；;\s]|$)"
)
_ONEOFF_QUERY_MARKERS = (
    "的价格", "多少钱", "什么价格", "价格多少", "收费标准", "哪个好", "哪个划算", "值不值得", "性价比", "贵不贵",
)
_ONEOFF_QUERY_HEAD_RE = re.compile(
    r"^(?:帮我|请|麻烦|替我)?(?:对比|比较|比一比|查一下|查查|看看|看下|搜一下|算一下|问一下)"
)
_ENTRUST_WORDS = ("记住", "记得", "别忘了", "记一下", "记下", "要记", "记着")
_SELFFOOTNOTE_RECORD_RE = re.compile(r"(?:留档|记档|存档|留底|备案|记录在案)")
_SELFFOOTNOTE_MODAL_RE = re.compile(r"(?:必须|一定|切记|每次|一律|很重要|重要|无论如何)")
_SELFFOOTNOTE_DEICTIC_RE = re.compile(r"(?:^(?:毕竟|反正|总之)?\s*(?:这种|这类|这些|这点|这事|这件事)|(?:这种|这类|这些|这点|这事|这件事)[^，。]{0,8}(?:重要|必须|一定|留档|存档))")

_APPRAISE_SERIOUS_WORDS = {
    1: ("哈哈", "嘿嘿", "逗你", "开玩笑", "闹着玩", "调戏"),
    2: ("可能", "也许", "有点", "随便", "说说", "聊聊"),
    3: ("认真", "说真的", "郑重", "重要", "必须", "一定", "我需要", "拜托"),
}
_APPRAISE_SERIOUS_CONTEXT = ("因为", "所以", "但是", "希望", "需要", "重要", "难过", "焦虑")
_APPRAISE_INTENT_WORDS = {
    1: ("告诉你", "我发现", "原来", "今天", "最近"),
    2: ("请", "帮我", "能不能", "可以吗", "拜托", "记住", "别忘", "托付"),
    3: ("哈哈", "嘿嘿", "逗你", "开玩笑", "闹着玩", "调戏"),
    4: ("难过", "焦虑", "崩溃", "想哭", "安慰我", "陪陪我", "抱抱我"),
    5: ("不对", "不同意", "凭什么", "质疑", "反驳", "你错了", "骗我"),
    6: ("对不起", "抱歉", "我错了", "别生气", "原谅我", "没事的"),
}
_APPRAISE_REL_WORDS = {
    1: ("不信任", "失望", "别管我", "离我远点", "不需要你", "讨厌你", "绝交", "分手", "闭嘴", "滚"),
    2: ("没事", "还好", "一般", "随便聊", "换个话题"),
    3: ("信任", "托付", "喜欢你", "珍惜", "依靠", "想你", "抱抱", "谢谢你"),
}
_APPRAISE_SELF_WORDS = {
    1: ("我们", "咱们"),
    2: ("你的", "和你", "跟你", "对你"),
    3: ("早苗", "你自己", "你本人", "你是不是", "你会不会", "你怎么", "你为什么", "你错了", "喜欢你", "信任你", "讨厌你"),
}
_APPRAISE_CONFLICT_WORDS = {
    1: ("不太同意", "有点不对", "别这样", "先停一下"),
    2: ("不同意", "不对", "凭什么", "质疑", "反驳", "你错了", "骗我"),
    3: ("闭嘴", "滚", "废物", "恶心", "讨厌你", "背叛", "绝交", "分手"),
}
_APPRAISE_NEGATIVE_WORDS = ("烦", "累", "崩溃", "焦虑", "难过", "失望", "生气", "想哭", "讨厌", "痛苦", "绝望")
_APPRAISE_POSITIVE_WORDS = ("开心", "高兴", "喜欢", "信任", "珍惜", "安心", "幸福", "谢谢", "爱你")
_APPRAISE_STRONG_AFFECT_WORDS = ("崩溃", "绝望", "气死", "痛苦", "太开心", "爱死", "好幸福", "非常", "特别", "极其")


def _has_system_meta(text: str) -> bool:
    if _META_TOKEN_RE.search(text):
        return True
    return any(word in text for word in _META_EXCLUDE_WORDS) or any(
        word in text for word in _SYSTEM_META_WORDS
    )


def _is_oneoff_query(text: str) -> bool:
    if any(word in text for word in _ENTRUST_WORDS):
        return False
    if not any(marker in text for marker in _ONEOFF_QUERY_MARKERS):
        return False
    return bool(_ONEOFF_QUERY_HEAD_RE.search(text.strip()))


def _is_self_footnote(text: str, *, relax_deictic: bool) -> bool:
    if any(word in text for word in _ENTRUST_WORDS) or any(word in text for word in _EMOTION_WORDS):
        return False
    if not (_SELFFOOTNOTE_RECORD_RE.search(text) and _SELFFOOTNOTE_MODAL_RE.search(text)):
        return False
    return relax_deictic or bool(_SELFFOOTNOTE_DEICTIC_RE.search(text))


def _noise_shape(text: str, *, relax_deictic: bool = False) -> bool:
    if not text:
        return False
    stripped = text.strip()
    if _TOOL_WHISPER_HEAD_RE.search(stripped):
        return True
    if _is_oneoff_query(stripped):
        return True
    if _NOISE_OBSERVE_RE.search(stripped):
        return True
    if _DEV_ROUTING_NOTE_RE.search(stripped):
        return True
    return _is_self_footnote(stripped, relax_deictic=relax_deictic)


__all__ = [name for name in globals() if name.startswith('_') and not name.startswith('__')]
