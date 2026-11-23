from __future__ import annotations

import datetime
import hashlib
import importlib
import logging
import math
import os
import re
import struct
import threading
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

from .contracts import ViveMindError
from .words import (
    _CN_RE, _DIMS, _EN_RE,
    _EMBED_MAX_TOKENS, _EMBED_MODEL_SUBDIR, _EMBED_QUERY_PREFIX,
    _EMOTION_WORDS, _IMPORTANT_WORDS, _RELATION_WORDS, _TOPIC_WORDS, _TRUST_WORDS,
)

logger = logging.getLogger(__name__)


def _bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _load_plugin_config() -> dict:
    try:
        from .host_bridge import load_host_config

        return (load_host_config().get("plugins") or {}).get("sanae_mem") or {}
    except Exception:
        return {}


def _candidate_text(raw: str) -> str:
    return re.sub(r"^\[[^]]+\]\s*", "", raw).strip()


def _tokens(text: str) -> List[str]:
    chinese = "".join(_CN_RE.findall(text))
    parts = [chinese[i : i + 2] for i in range(max(0, len(chinese) - 1))]
    if len(chinese) == 1:
        parts.append(chinese)
    parts.extend(word.lower() for word in _EN_RE.findall(text))
    return parts


def event_keywords(text: str) -> str:
    values: List[str] = []
    for word in _TRUST_WORDS + _EMOTION_WORDS + _RELATION_WORDS + _IMPORTANT_WORDS + tuple(
        item for group in _TOPIC_WORDS.values() for item in group
    ):
        if word in text and word not in values:
            values.append(word)
    values.extend(token for token in _tokens(text) if token not in values)
    return " ".join(values[:16])

def _hash_embed(text: str, dims: int = _DIMS) -> List[float]:
    if dims < 8 or dims % 8:
        raise ValueError("dims must be a positive multiple of 8")
    vector = [0.0] * dims
    parts = _tokens(text)
    for token in parts:
        seed = hashlib.sha256(token.encode("utf-8")).digest()
        stream = bytearray()
        counter = 0
        while len(stream) < dims // 8:
            stream.extend(hashlib.sha256(seed + counter.to_bytes(2, "big")).digest())
            counter += 1
        for index in range(dims):
            vector[index] += 1.0 if stream[index // 8] & (1 << (index % 8)) else -1.0
    norm = math.sqrt(sum(value * value for value in vector))
    if norm:
        vector = [value / norm for value in vector]
    return vector


# BGE 的检索前缀只加在问句侧，文档侧不加。
EmbedDirLike = Optional[Union[str, os.PathLike[str]]]


@dataclass(frozen=True)
class _EmbedRuntime:
    dir: str
    session: Any
    tokenizer: Any
    numpy: Any
    inputs: Tuple[str, ...]


@dataclass(frozen=True)
class EmbedSlotSnapshot:
    dir: str
    session: Any = None
    failed: bool = False
    warned: bool = False


@dataclass
class _EmbedSlot:
    dir: str
    lock: threading.RLock = field(default_factory=threading.RLock)
    session: Any = None
    tokenizer: Any = None
    numpy: Any = None
    inputs: Tuple[str, ...] = ()
    failed: bool = False
    warned: bool = False

    def runtime(self) -> Optional[_EmbedRuntime]:
        if self.session is None:
            return None
        return _EmbedRuntime(
            dir=self.dir,
            session=self.session,
            tokenizer=self.tokenizer,
            numpy=self.numpy,
            inputs=self.inputs,
        )

    def snapshot(self) -> EmbedSlotSnapshot:
        return EmbedSlotSnapshot(
            dir=self.dir,
            session=self.session,
            failed=self.failed,
            warned=self.warned,
        )


_EMBED_LOCK = threading.RLock()
_EMBED_SLOTS: Dict[str, _EmbedSlot] = {}
_EMBED_ADDED_SITE_DIRS: set[str] = set()
_ACTIVE_EMBED_DIR: ContextVar[Optional[str]] = ContextVar("sanae_mem_active_embed_dir", default=None)
_DEFAULT_EMBED_DIR: str = ""


def _normalize_embed_dir(data_dir: EmbedDirLike) -> str:
    if data_dir is None:
        return ""
    raw = os.fspath(data_dir)
    if not raw:
        return ""
    return str(Path(raw).expanduser().resolve())


def _resolve_embed_dir(data_dir: EmbedDirLike = None) -> str:
    if data_dir is not None:
        return _normalize_embed_dir(data_dir)
    scoped = _ACTIVE_EMBED_DIR.get()
    if scoped is not None:
        return scoped
    with _EMBED_LOCK:
        return _DEFAULT_EMBED_DIR


def _get_or_create_embed_slot(norm_dir: str) -> _EmbedSlot:
    with _EMBED_LOCK:
        slot = _EMBED_SLOTS.get(norm_dir)
        if slot is None:
            slot = _EmbedSlot(dir=norm_dir)
            _EMBED_SLOTS[norm_dir] = slot
        return slot


def _configure_embedding(data_dir: EmbedDirLike) -> None:
    global _DEFAULT_EMBED_DIR
    norm_dir = _normalize_embed_dir(data_dir)
    _ACTIVE_EMBED_DIR.set(norm_dir)
    with _EMBED_LOCK:
        _DEFAULT_EMBED_DIR = norm_dir
        if norm_dir and norm_dir not in _EMBED_SLOTS:
            _EMBED_SLOTS[norm_dir] = _EmbedSlot(dir=norm_dir)


@contextmanager
def embedding_dir_scope(data_dir: EmbedDirLike) -> Iterator[str]:
    norm_dir = _normalize_embed_dir(data_dir)
    if norm_dir:
        _get_or_create_embed_slot(norm_dir)
    token = _ACTIVE_EMBED_DIR.set(norm_dir)
    try:
        yield norm_dir
    finally:
        _ACTIVE_EMBED_DIR.reset(token)


def get_embedding_state(data_dir: EmbedDirLike = None) -> EmbedSlotSnapshot:
    norm_dir = _resolve_embed_dir(data_dir)
    if not norm_dir:
        return EmbedSlotSnapshot(dir="")
    with _EMBED_LOCK:
        slot = _EMBED_SLOTS.get(norm_dir)
    if slot is None:
        return EmbedSlotSnapshot(dir=norm_dir)
    with slot.lock:
        return slot.snapshot()


def reset_embedding_state(data_dir: EmbedDirLike = None) -> None:
    global _DEFAULT_EMBED_DIR
    if data_dir is None:
        with _EMBED_LOCK:
            _EMBED_SLOTS.clear()
            _EMBED_ADDED_SITE_DIRS.clear()
            _DEFAULT_EMBED_DIR = ""
        _ACTIVE_EMBED_DIR.set(None)
        return
    norm_dir = _normalize_embed_dir(data_dir)
    if not norm_dir:
        _ACTIVE_EMBED_DIR.set(None)
        return
    with _EMBED_LOCK:
        _EMBED_SLOTS.pop(norm_dir, None)
        if _DEFAULT_EMBED_DIR == norm_dir:
            _DEFAULT_EMBED_DIR = ""
        prefix = norm_dir + os.sep
        _EMBED_ADDED_SITE_DIRS.difference_update(
            {path for path in _EMBED_ADDED_SITE_DIRS if path == norm_dir or path.startswith(prefix)}
        )
    if _ACTIVE_EMBED_DIR.get() == norm_dir:
        _ACTIVE_EMBED_DIR.set(None)


def _embed_site_packages(data_dir: Union[str, os.PathLike[str]]) -> None:
    import site as _site
    import sys as _sys

    norm_dir = _normalize_embed_dir(data_dir)
    if not norm_dir:
        return
    base = Path(norm_dir) / "venv"
    candidates = [
        base / sub / f"python{_sys.version_info.major}.{_sys.version_info.minor}" / "site-packages"
        for sub in ("lib", "lib64")
    ]
    with _EMBED_LOCK:
        added = False
        for candidate in candidates:
            candidate_str = str(candidate)
            if candidate_str in _EMBED_ADDED_SITE_DIRS:
                continue
            if candidate.is_dir():
                _site.addsitedir(candidate_str)
                _EMBED_ADDED_SITE_DIRS.add(candidate_str)
                added = True
        if added:
            importlib.invalidate_caches()


def _load_embedder(data_dir: EmbedDirLike = None) -> Optional[_EmbedRuntime]:
    norm_dir = _resolve_embed_dir(data_dir)
    if not norm_dir:
        return None
    slot = _get_or_create_embed_slot(norm_dir)
    with slot.lock:
        if slot.failed:
            return None
        existing = slot.runtime()
        if existing is not None:
            return existing
        try:
            model_dir = Path(slot.dir) / _EMBED_MODEL_SUBDIR
            model_path = model_dir / "model.onnx"
            tokenizer_path = model_dir / "tokenizer.json"
            if not (model_path.is_file() and tokenizer_path.is_file()):
                slot.failed = True
                if not slot.warned:
                    slot.warned = True
                    logger.warning(
                        "sanae_mem: 本地语义模型文件缺失，回退散列嵌入（dir=%s, model=%s）",
                        slot.dir,
                        _EMBED_MODEL_SUBDIR,
                    )
                return None
            _embed_site_packages(slot.dir)
            _np = importlib.import_module("numpy")
            _ort = importlib.import_module("onnxruntime")
            _Tokenizer = importlib.import_module("tokenizers").Tokenizer

            options = _ort.SessionOptions()
            options.intra_op_num_threads = 2
            options.graph_optimization_level = _ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            session = _ort.InferenceSession(
                str(model_path), sess_options=options, providers=["CPUExecutionProvider"]
            )
            tokenizer = _Tokenizer.from_file(str(tokenizer_path))
            tokenizer.enable_truncation(max_length=_EMBED_MAX_TOKENS)
            tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")
            slot.session = session
            slot.tokenizer = tokenizer
            slot.numpy = _np
            slot.inputs = tuple(item.name for item in session.get_inputs())
            logger.info("sanae_mem: 本地语义模型已加载（dir=%s, model=%s）", slot.dir, _EMBED_MODEL_SUBDIR)
            return slot.runtime()
        except Exception as exc:
            slot.failed = True
            if not slot.warned:
                slot.warned = True
                logger.warning(
                    "sanae_mem: 本地语义模型不可用，回退散列嵌入（dir=%s, err=%s）",
                    slot.dir,
                    type(exc).__name__,
                )
            return None


def _onnx_embed(texts: List[str], *, data_dir: EmbedDirLike = None) -> Optional[List[List[float]]]:
    runtime = _load_embedder(data_dir)
    if runtime is None:
        return None
    if not texts:
        return []
    try:
        numpy = runtime.numpy
        encoded = runtime.tokenizer.encode_batch(list(texts))
        ids = numpy.array([item.ids for item in encoded], dtype=numpy.int64)
        mask = numpy.array([item.attention_mask for item in encoded], dtype=numpy.int64)
        feed: Dict[str, Any] = {}
        if "input_ids" in runtime.inputs:
            feed["input_ids"] = ids
        if "attention_mask" in runtime.inputs:
            feed["attention_mask"] = mask
        if "token_type_ids" in runtime.inputs:
            feed["token_type_ids"] = numpy.zeros_like(ids)
        output = runtime.session.run(None, feed)
        hidden = numpy.asarray(output[0], dtype=numpy.float32)
        pooled = hidden[:, 0]  # bge 系：cls 池化
        norm = numpy.clip(numpy.linalg.norm(pooled, axis=1, keepdims=True), 1e-9, None)
        return (pooled / norm).astype(numpy.float32).tolist()
    except Exception as exc:
        logger.warning(
            "sanae_mem: 本地语义模型推理失败，本次回退散列嵌入（dir=%s, err=%s）",
            runtime.dir,
            type(exc).__name__,
        )
        return None


def _embedding_is_real(data_dir: EmbedDirLike = None) -> bool:
    return _load_embedder(data_dir) is not None


def _embed_text(
    text: str,
    *,
    query: bool = False,
    data_dir: EmbedDirLike = None,
) -> Tuple[List[float], bool]:
    payload = [(_EMBED_QUERY_PREFIX + text) if query else text]
    if data_dir is None:
        vectors = _onnx_embed(payload)
    else:
        vectors = _onnx_embed(payload, data_dir=data_dir)
    if vectors is not None:
        return vectors[0], True
    return _hash_embed(text, _DIMS), False


def _embed(text: str, *, data_dir: EmbedDirLike = None) -> List[float]:
    return _embed_text(text, data_dir=data_dir)[0]


def _embed_query(text: str, *, data_dir: EmbedDirLike = None) -> List[float]:
    return _embed_text(text, query=True, data_dir=data_dir)[0]


def _pack_vector(vector: List[float]) -> bytes:
    return struct.pack(f"<{len(vector)}f", *vector)


def _unpack_vector(blob: Optional[bytes]) -> List[float]:
    if not blob:
        return []
    return list(struct.unpack(f"<{len(blob) // 4}f", blob))


def _cosine(left: List[float], right: List[float]) -> float:
    if not left or len(left) != len(right):
        return 0.0
    return sum(a * b for a, b in zip(left, right))


def _fingerprint(text: str) -> str:
    normalized = re.sub(r"\W+", "", text, flags=re.UNICODE).lower()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _owner_key(platform: Any, **identity: Any) -> str:
    primary = str(identity.get("user_id") or "").strip()
    alternate = str(identity.get("user_id_alt") or "").strip()
    participant = alternate or primary
    if not participant:
        return ""
    label = f"{str(platform or '').strip()}:{participant}"
    # A caller that knows the platform scope (Slack workspace, …) gets that scope in the namespace:
    # the same participant ID in another scope is a different person, never the same owner.
    scope = str(identity.get("scope_id") or "").strip()
    if scope:
        label = f"{label}:{scope}"
    return hashlib.sha256(label.encode("utf-8")).hexdigest()[:16]


HOME_ENV = "HERMES_HOME"
HOME_OVERRIDE_ENV = "HERMES_HOME_OVERRIDE"

_HOME_DIAG_NAME = "home-resolution.log"

_HOME_DIAG_LOGGED = False


class HermesHomeConflict(ViveMindError, RuntimeError):
    """`HERMES_HOME` 与 `HERMES_HOME_OVERRIDE` 指向不同路径：不猜，直接停。"""


def _norm_home(value: str) -> str:
    return os.path.normcase(os.path.abspath(os.path.expanduser(str(value))))


def _env_home(name: str) -> str:
    return str(os.environ.get(name) or "").strip()


def _home_env_conflict():
    primary = _env_home(HOME_ENV)
    override = _env_home(HOME_OVERRIDE_ENV)
    if primary and override and _norm_home(primary) != _norm_home(override):
        raise HermesHomeConflict(
            "%s=%s 与 %s=%s 指向不同路径；根目录不唯一时拒绝启动，"
            "请让两者指向同一目录，或清掉其中之一。" % (HOME_ENV, primary, HOME_OVERRIDE_ENV, override)
        )
    return primary, override


def hermes_home(explicit: Any = None) -> str:
    """按固定优先级解析 Hermes 根目录，返回绝对路径串（唯一真源）。"""
    primary, override = _home_env_conflict()
    if explicit:
        return os.path.abspath(os.path.expanduser(str(explicit)))
    try:
        from . import host_bridge as _hb
        pinned = _hb._hermes_home_override.get()
        if pinned is not None:
            return os.path.abspath(os.path.expanduser(str(pinned)))
        if _hb._hc is not None:
            active = _hb._hc.get_hermes_home()
            if str(active) != str(_hb._hc.get_process_hermes_home()):
                return os.path.abspath(os.path.expanduser(str(active)))
    except Exception:
        pass
    return os.path.abspath(os.path.expanduser(primary or override or str(Path.home() / ".hermes")))


def provider_home(provider: Any) -> str:
    """取 provider 实例已解析的根（宿主显式路径的手里那份）。

    `initialize()` 把结果存进 `_hermes_home`；未初始化（直接构造的测试对象）
    才回落到同一解析器。混入类与钩子都走这里，保证与 provider 同源。
    """
    stored = getattr(provider, "_hermes_home", "")
    if stored:
        return str(stored)
    return hermes_home()


def record_home_resolution(explicit: Any = None, resolved: Any = None, *, log_dir: Any = None) -> str:
    """把"这次根目录从哪来、解析成什么"记一行（每进程一次，只有路径没有正文）。

    观察面不是必需的：留痕失败一律吞掉，绝不影响启动。
    """
    global _HOME_DIAG_LOGGED
    if _HOME_DIAG_LOGGED:
        return ""
    _HOME_DIAG_LOGGED = True
    try:
        home = str(resolved) if resolved else hermes_home(explicit)
        primary = _env_home(HOME_ENV)
        override = _env_home(HOME_OVERRIDE_ENV)
        source = ("explicit" if explicit else HOME_ENV if primary else
                  HOME_OVERRIDE_ENV if override else "home-default")
        line = "at=%s source=%s resolved=%s %s=%s %s=%s\n" % (
            datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
            source,
            home,
            HOME_ENV,
            primary or "-",
            HOME_OVERRIDE_ENV,
            override or "-",
        )
        target = Path(str(log_dir)) if log_dir else Path(home) / "sanae-mem"
        target.mkdir(parents=True, exist_ok=True)
        with (target / _HOME_DIAG_NAME).open("a", encoding="utf-8") as handle:
            handle.write(line)
        return line
    except Exception:
        return ""


def reset_home_diagnostics() -> None:
    global _HOME_DIAG_LOGGED
    _HOME_DIAG_LOGGED = False
