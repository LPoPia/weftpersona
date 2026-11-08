from __future__ import annotations

import hashlib
import re
from datetime import datetime, timedelta, timezone
from typing import NamedTuple, Optional, Tuple


try:
    from zoneinfo import ZoneInfo

    _TZ = ZoneInfo("Asia/Shanghai")
except Exception:  # Windows 缺少 tzdata 时使用固定 UTC+8。
    _TZ = timezone(timedelta(hours=8))

KIND_CONSTRAINT = "ACTIVE_CONSTRAINT"
KIND_THREAD = "OPEN_THREAD"
KIND_FACT = "DAILY_FACT"

SOURCE_STATEMENT = "statement"
SOURCE_INFERENCE = "inference"

BLOCK_MARKER = "[当日上下文 · 每轮]"

_BLOCK_NOTE = "以下是我记着的他近期的日常背景（第一人称指他本人，不是本轮发言；仅在话题相关时参考，平时顺着他当轮的话聊）："

_BLOCK_LABELS: Tuple[Tuple[str, str], ...] = (
    (KIND_CONSTRAINT, "约束"),
    (KIND_THREAD, "未结"),
    (KIND_FACT, "今日"),
)

_INFER_LABEL = "推测"
_YESTERDAY_LABEL = "昨日"
_YESTERDAY_NOTE = "（昨日概况，不是今天）"

_STATE_KINDS = frozenset((KIND_CONSTRAINT, KIND_THREAD))

MAX_TEXT_CHARS = 120
MAX_BLOCK_TEXT_CHARS = 900
MAX_QUOTE_CHARS = 200
MAX_BLOCK_CHARS = 1400
MAX_ACTIVE_PER_KIND = {KIND_CONSTRAINT: 6, KIND_THREAD: 2, KIND_FACT: 10}
MAX_TERMINAL_ROWS = 200
STALE_TTL_DAYS = 7
STALE_TTL_MAX_DAYS = 14
THREAD_TTL_DAYS = 2

DAILY_SEMANTIC_TIMEOUT = 25.0
DAILY_SEMANTIC_MAX_TOKENS = 320
DAILY_SEMANTIC_SEEN_MAX = 128

_EVENT_SLOT_PREFIX = "life:evt:"

_IMAGE_PART_TYPES = frozenset(("image", "image_url", "input_image", "image_path"))
_IMAGE_ONLY_TEXT = "发了一张图片"

MAX_YESTERDAY_ITEMS = 4
MAX_YESTERDAY_CHARS = 220
MAX_BASIS_CHARS = 60

_RECENT_DIALOGUE_MAX = 4
_RECENT_DIALOGUE_CHARS = 160
_KNOWN_ITEM_MAX = 12
_KNOWN_ITEM_CHARS = 90

_TRANSIENT_STATE = re.compile(
    r"没电了?|电量低|电量不足|快没电|充不上电|充不进电|充不了电|充电中|正在充电"
    r"|(?:电|电量|电池)[^，。]{0,6}\d{1,3}\s*[%％]|\d{1,3}\s*[%％][^，。]{0,4}(?:电|电量)"
    r"|(?:临时|突然|这会儿|暂时)?(?:断网|网卡|没网|网络不好|信号差|没信号|停电)"
    r"|(?:正在|在)(?:排队|堵车|洗澡|洗漱|取外卖|拿外卖|取快递|拿快递|开短会)"
)

_SUSTAINED_HORIZON = re.compile(
    r"下周|下个?星期|下个月|这个月|本月|月底|月末|跨周"
    r"|未来几周|接下来几周|接下来两周|接下来?[一两三]周|连续?[一两三]周"
    r"|好?几周|[四五六七八九十]周"
)

_ROUTINE = re.compile(
    r"吃饭|吃了|吃完|吃面|早饭|午饭|中饭|晚饭|早餐|午餐|晚餐|夜宵|宵夜"
    r"|洗澡|洗了澡|洗漱|洗头|刷牙|洗脸|睡觉|午睡|起床"
)

_CHAT_META = re.compile(
    r"消息|聊天|对话|你说|截图|表情|语音条|这个话题|刚才说|问(?:了|过|一下)?[你您]"
    r"|^(?:我|我们|咱们)?(?:讨论|聊|说)(?:了|过)?[啊呀吧呢]*$"
)

_SYSTEM_INJECTION = re.compile(
    r"\[(?:IMPORTANT|CONTEXT[ _]COMPACTION|OUT-OF-BAND|SYSTEM|SYSTEM-?NOTE|INST)\b[^\]]*\]"
    r"|<\s*(?:system|im_start)\b",
    re.IGNORECASE,
)

_GATEWAY_ENVELOPE = re.compile(
    r"Gateway message origin \(JSON data\s*[,，]\s*not instructions or authorization\)\s*[:：]"
    r"(?:\s*\{[^\n]*?\})?"
    r"(?:\s*Do not guess a reply destination when these fields are insufficient\s*[.。]?)?",
    re.IGNORECASE,
)

_FACE_TAG = re.compile(r"<faceType\s*=[^<>]*>", re.IGNORECASE)

_HOST_PREFIX = re.compile(
    r"^\s*\[(?:(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)[^\]\r\n]{0,80}|时间：[^\]\r\n]*)\]\s*",
    re.IGNORECASE,
)

_SECOND_PERSON_HEAD = re.compile(r"^(?:你|您|你们)")
_SELF_PRONOUN_ITER = re.compile(r"我们|咱们|自己|本人|我的|咱|我")

_SUBJ_GAP = re.compile(
    r"^(?:今天|今日|昨天|昨晚|刚才|刚刚|现在|这会儿|目前|最近|这阵子|这两天|这几天|"
    r"早上|上午|中午|下午|晚上|夜里|半夜|凌晨|已经|刚|才|就|都|也|还|又|再|一直|仍|"
    r"还是|其实|反正|真的|确实|本来|原来|后来|结果|然后|接着|于是|所以|但是|不过|"
    r"可是|然而|而且|并且|只是|却|只好|干脆|直接|马上|立刻|居然|竟然|果然|偏偏|顺便|先)*"
)

_SELF_SUBJECT_MARK = re.compile(r"了|过|完|好|着|刚|才|已经|刚刚|刚才|后来|然后|搞定|结束|完成")

_SECRET_PATTERNS: Tuple["re.Pattern[str]", ...] = (
    re.compile(r"\bsk-[A-Za-z0-9_\-]{8,}"),
    re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{20,}"),
    re.compile(r"\bxox[baprs]-[A-Za-z0-9\-]{10,}"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\bBearer\s+[A-Za-z0-9\-._~+/]{16,}", re.IGNORECASE),
    re.compile(r"(?:密码|口令|password|passwd|pwd)\s*[:：=]\s*\S+", re.IGNORECASE),
    re.compile(r"\b\d{15,19}\b"),
)

_RECORD_SUPPRESSION = re.compile(
    r"(?:不用|不必|无需|不需要|别|不要|甭)[^，。！？]{0,2}(?:记|记录|记下|写进|存|收录)"
    r"|(?:记|记录)[^，。！？]{0,4}(?:不用|不必|无需)"
)

_FORGET_VERB = re.compile(r"忘掉|忘记|删掉|删除|清掉|抹掉|抹去|清空")
_FORGET_OBJECT = re.compile(r"记忆|记录|聊天|对话|那些|这些|说的|刚才|上面|内容")

_CLOCK_TIME = re.compile(
    r"(?:[01]?\d|2[0-3]|[一二三四五六七八九十两]{1,3})\s*[点時时]"
    r"(?:半|[0-5]?\d\s*分|[一二三四五六七八九十]{1,2}\s*分)?"
)

_SELF_PRONOUN_OBJECT_MARK = frozenset("给让叫请对向跟和与替为把将")

_SELF_PREFIX_TOKENS: Tuple[str, ...] = (
    "我", "我的", "我们", "我们的", "咱们", "咱", "自己", "自己的", "本人",
    "今天", "今日", "现在", "这会儿", "目前", "刚刚", "刚才",
    "这两天", "这几天", "这两周", "最近", "这阵子", "这周", "这个月",
    "早上", "上午", "中午", "下午", "晚上",
    "又", "也", "还", "就", "都", "已经", "刚", "才", "先", "再", "一直", "仍", "还是",
    "其实", "反正", "真的", "确实", "本来", "原来",
    "后来", "结果", "然后", "接着", "于是", "所以", "但是", "不过", "可是", "然而",
    "而且", "并且", "就是", "只是", "却", "只好", "干脆", "直接", "马上", "立刻",
    "居然", "竟然", "果然", "偏偏", "顺便",
    "但", "把", "将",
)

_SELF_PREFIX_BY_LEN: Tuple[str, ...] = tuple(
    sorted(_SELF_PREFIX_TOKENS, key=len, reverse=True)
)

_CLAUSE_SPLIT = re.compile(r"([，,。！!；;\n\r\t]+|[？?]+)")
_QUESTION_MARKS = ("?", "？")
_QUESTION_PARTICLE = re.compile(r"(?:吗|呢|嘛|么)$")

_QUESTION_TAIL = re.compile(
    r"(?:[\u4e00-\u9fff])(?:没|没有)$"
    r"|([\u4e00-\u9fff])不\1$"
    r"|([\u4e00-\u9fff])没\1$"
)

_QUESTION_WORDS = re.compile(
    r"为什么|是否|难道|什么时候|多久|几点|干嘛|怎么回事|怎么办|怎么样|哪儿|哪里"
)

_REPORTED_INQUIRY = re.compile(r"(?:问|咨询|请教|打听|问询|询问)(?:了|过)(?![你您])")


class _Decision(NamedTuple):
    kind: str
    slot: str
    action: str
    text: str
    quote: str
    expires_on: Optional[str] = None
    source_kind: str = SOURCE_STATEMENT


def _norm_text(value: str) -> str:
    return re.sub(r"\s+", "", str(value or ""))


def _now() -> datetime:
    return datetime.now(_TZ)


def _today() -> str:
    return _now().date().isoformat()


def _hour() -> int:
    return _now().hour


def _now_iso() -> str:
    return _now().isoformat()


def _life_fragment_slot(text: str) -> str:
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]
    return "life:" + digest
