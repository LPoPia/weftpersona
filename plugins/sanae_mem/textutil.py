from __future__ import annotations

import datetime
import hashlib
import logging
import math
import os
import re
import struct
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from .words import (
        _CN_RE, _DIMS, _EN_RE,
        _EMBED_MAX_TOKENS, _EMBED_MODEL_SUBDIR, _EMBED_QUERY_PREFIX,
    )
except ImportError:
    from words import (
        _CN_RE, _DIMS, _EN_RE,
        _EMBED_MAX_TOKENS, _EMBED_MODEL_SUBDIR, _EMBED_QUERY_PREFIX,
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
        from hermes_cli.config import cfg_get, load_config_readonly

        return cfg_get(load_config_readonly(), "plugins", "sanae_mem", default={}) or {}
    except Exception:
        return {}


def _tokens(text: str) -> List[str]:
    chinese = "".join(_CN_RE.findall(text))
    parts = [chinese[i : i + 2] for i in range(max(0, len(chinese) - 1))]
    if len(chinese) == 1:
        parts.append(chinese)
    parts.extend(word.lower() for word in _EN_RE.findall(text))
    return parts


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
_EMBED_STATE: Dict[str, Any] = {"dir": "", "session": None, "tokenizer": None,
                                "numpy": None, "inputs": [], "failed": False, "warned": False}


def _configure_embedding(data_dir) -> None:
    _EMBED_STATE["dir"] = str(data_dir or "")


def _embed_site_packages(data_dir) -> None:
    import site as _site
    import sys as _sys

    base = Path(str(data_dir)) / "venv"
    for sub in ("lib", "lib64"):
        candidate = base / sub / f"python{_sys.version_info.major}.{_sys.version_info.minor}" / "site-packages"
        if candidate.is_dir():
            _site.addsitedir(str(candidate))


def _load_embedder():
    if _EMBED_STATE["failed"] or not _EMBED_STATE["dir"]:
        return None
    if _EMBED_STATE["session"] is not None:
        return _EMBED_STATE
    try:
        _embed_site_packages(_EMBED_STATE["dir"])
        import numpy as _np
        import onnxruntime as _ort
        from tokenizers import Tokenizer as _Tokenizer

        model_dir = Path(_EMBED_STATE["dir"]) / _EMBED_MODEL_SUBDIR
        model_path = model_dir / "model.onnx"
        tokenizer_path = model_dir / "tokenizer.json"
        if not (model_path.is_file() and tokenizer_path.is_file()):
            _EMBED_STATE["failed"] = True
            return None
        options = _ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.graph_optimization_level = _ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session = _ort.InferenceSession(
            str(model_path), sess_options=options, providers=["CPUExecutionProvider"]
        )
        tokenizer = _Tokenizer.from_file(str(tokenizer_path))
        tokenizer.enable_truncation(max_length=_EMBED_MAX_TOKENS)
        tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")
        _EMBED_STATE.update(
            session=session, tokenizer=tokenizer, numpy=_np,
            inputs=[item.name for item in session.get_inputs()],
        )
        logger.info("sanae_mem: 本地语义模型已加载（%s）", _EMBED_MODEL_SUBDIR)
        return _EMBED_STATE
    except Exception as exc:
        _EMBED_STATE["failed"] = True
        if not _EMBED_STATE["warned"]:
            _EMBED_STATE["warned"] = True
            logger.warning("sanae_mem: 本地语义模型不可用，回退散列嵌入（%s）", type(exc).__name__)
        return None


def _onnx_embed(texts: List[str]) -> Optional[List[List[float]]]:
    state = _load_embedder()
    if state is None:
        return None
    try:
        numpy = state["numpy"]
        encoded = state["tokenizer"].encode_batch(list(texts))
        ids = numpy.array([item.ids for item in encoded], dtype=numpy.int64)
        mask = numpy.array([item.attention_mask for item in encoded], dtype=numpy.int64)
        feed = {}
        if "input_ids" in state["inputs"]:
            feed["input_ids"] = ids
        if "attention_mask" in state["inputs"]:
            feed["attention_mask"] = mask
        if "token_type_ids" in state["inputs"]:
            feed["token_type_ids"] = numpy.zeros_like(ids)
        output = state["session"].run(None, feed)
        hidden = numpy.asarray(output[0], dtype=numpy.float32)
        pooled = hidden[:, 0]  # bge 系：cls 池化
        norm = numpy.clip(numpy.linalg.norm(pooled, axis=1, keepdims=True), 1e-9, None)
        return (pooled / norm).astype(numpy.float32).tolist()
    except Exception:
        _EMBED_STATE["failed"] = True
        return None


def _embedding_is_real() -> bool:
    _load_embedder()
    return _EMBED_STATE["session"] is not None


def _embed_text(text: str, *, query: bool = False):
    vectors = _onnx_embed([(_EMBED_QUERY_PREFIX + text) if query else text])
    if vectors is not None:
        return vectors[0], True
    return _hash_embed(text, _DIMS), False


def _embed(text: str) -> List[float]:
    return _embed_text(text)[0]


def _embed_query(text: str) -> List[float]:
    return _embed_text(text, query=True)[0]


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
    return hashlib.sha256(label.encode("utf-8")).hexdigest()[:16]


HOME_ENV = "HERMES_HOME"
HOME_OVERRIDE_ENV = "HERMES_HOME_OVERRIDE"

_HOME_DIAG_NAME = "home-resolution.log"

_HOME_DIAG_LOGGED = False


class HermesHomeConflict(RuntimeError):
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
    """测试用：清掉"本进程已留痕"标记。"""
    global _HOME_DIAG_LOGGED
    _HOME_DIAG_LOGGED = False


__all__ = [name for name in globals() if name.startswith('_') and not name.startswith('__')]
