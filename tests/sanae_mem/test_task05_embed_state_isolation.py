from __future__ import annotations

import asyncio
import importlib.util
import json
import logging
import math
import site
import sqlite3
import struct
import sys
import threading
import time
import types
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import pytest

_PLUGINS_ROOT = Path(__file__).resolve().parents[2] / "plugins"
if str(_PLUGINS_ROOT) not in sys.path:
    sys.path.insert(0, str(_PLUGINS_ROOT))

from sanae_mem import SanaeMemProvider
from sanae_mem.infra import host_bridge, textutil, words


class _FakeArray:
    """Minimal pure-Python fallback for numpy ndarray when numpy is absent."""

    def __init__(self, data: Any) -> None:
        self._data = data

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, tuple) and len(key) == 2 and key[0] == slice(None) and key[1] == 0:
            return _FakeArray([row[0] for row in self._data])
        return self._data[key]

    def __truediv__(self, other: "_FakeArray") -> "_FakeArray":
        rows: List[List[float]] = []
        for row, denom in zip(self._data, other._data):
            d = float(denom[0])
            rows.append([float(v) / d for v in row])
        return _FakeArray(rows)

    def astype(self, _dtype: Any) -> "_FakeArray":
        return self

    def tolist(self) -> Any:
        return self._data


class _FakeLinalg:
    @staticmethod
    def norm(arr: _FakeArray, axis: int = 1, keepdims: bool = True) -> _FakeArray:
        assert axis == 1 and keepdims
        norms = [[math.sqrt(sum(float(v) * float(v) for v in row))] for row in arr._data]
        return _FakeArray(norms)


class _FakeNumpyModule(types.ModuleType):
    int64: str = "int64"
    float32: str = "float32"
    linalg = _FakeLinalg()

    def __init__(self) -> None:
        super().__init__("numpy")

    @staticmethod
    def isscalar(obj: Any) -> bool:
        return isinstance(obj, (int, float, complex, bool, str, bytes))

    @staticmethod
    def array(data: Any, dtype: Any = None) -> _FakeArray:
        return _FakeArray(data)

    @staticmethod
    def asarray(data: Any, dtype: Any = None) -> _FakeArray:
        if isinstance(data, _FakeArray):
            return data
        return _FakeArray(data)

    @staticmethod
    def zeros_like(arr: _FakeArray) -> _FakeArray:
        return _FakeArray([[0 for _ in row] for row in arr._data])

    @staticmethod
    def clip(arr: _FakeArray, low: float, _high: Any) -> _FakeArray:
        return _FakeArray([[max(low, float(v)) for v in row] for row in arr._data])


class _FakeEncoding:
    def __init__(self, text: str) -> None:
        self.ids = [101, max(1, len(text)), 102]
        self.attention_mask = [1, 1, 1]


class _FakeInputMeta:
    def __init__(self, name: str) -> None:
        self.name = name


class _FakeTokenizer:
    def __init__(
        self,
        path: str,
        *,
        on_encode: Optional[Callable[[str, List[str]], None]] = None,
    ) -> None:
        self.path = path
        self._on_encode = on_encode
        self.max_length: Optional[int] = None
        self.pad_id: Optional[int] = None

    def enable_truncation(self, *, max_length: int) -> None:
        self.max_length = max_length

    def enable_padding(self, *, pad_id: int, pad_token: str) -> None:
        self.pad_id = pad_id

    def encode_batch(self, texts: List[str]) -> List[_FakeEncoding]:
        if self._on_encode is not None:
            self._on_encode(self.path, texts)
        return [_FakeEncoding(item) for item in texts]


class _FakeSession:
    def __init__(
        self,
        model_path: str,
        *,
        vector: Sequence[float],
        on_run: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    ) -> None:
        self.model_path = model_path
        self._vector = [float(v) for v in vector]
        self._on_run = on_run

    def get_inputs(self) -> List[_FakeInputMeta]:
        return [
            _FakeInputMeta("input_ids"),
            _FakeInputMeta("attention_mask"),
            _FakeInputMeta("token_type_ids"),
        ]

    def run(self, _outputs: Any, feed: Dict[str, Any]) -> List[Any]:
        if self._on_run is not None:
            self._on_run(self.model_path, feed)
        batch_size = len(feed["input_ids"])
        return [[[list(self._vector)] for _ in range(batch_size)]]


def _make_model_files(data_dir: Path) -> Path:
    model_dir = data_dir / words._EMBED_MODEL_SUBDIR
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "model.onnx").write_bytes(b"fake-onnx")
    (model_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
    return model_dir


def _install_fake_runtime(
    monkeypatch: pytest.MonkeyPatch,
    *,
    vectors_by_dir: Dict[str, Sequence[float]],
    load_counts: Optional[Dict[str, int]] = None,
    fail_load_dirs: Optional[set[str]] = None,
    load_delay_sec: float = 0.0,
    on_run: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    on_encode: Optional[Callable[[str, List[str]], None]] = None,
) -> None:
    counts_lock = threading.Lock()
    counts = load_counts if load_counts is not None else {}
    failing = fail_load_dirs or set()

    def _dir_for_model(model_path: str) -> str:
        resolved = Path(model_path).resolve()
        # <data_dir>/models/bge-small-zh-v1.5/model.onnx -> parents[2] is <data_dir>
        return str(resolved.parents[2])

    class _SessionOptions:
        intra_op_num_threads: int = 1
        graph_optimization_level: Any = None

    class _GraphOptimizationLevel:
        ORT_ENABLE_ALL = "ORT_ENABLE_ALL"

    def _create_session(
        model_path: str,
        *,
        sess_options: Any = None,
        providers: Optional[List[str]] = None,
    ) -> _FakeSession:
        assert sess_options is not None
        assert providers == ["CPUExecutionProvider"]
        data_key = _dir_for_model(model_path)
        with counts_lock:
            counts[data_key] = counts.get(data_key, 0) + 1
        if load_delay_sec > 0:
            time.sleep(load_delay_sec)
        if data_key in failing:
            raise RuntimeError(f"corrupt onnx model for {data_key}")
        vec = vectors_by_dir[data_key]
        return _FakeSession(model_path, vector=vec, on_run=on_run)

    ort_mod = types.ModuleType("onnxruntime")
    setattr(ort_mod, "SessionOptions", _SessionOptions)
    setattr(ort_mod, "GraphOptimizationLevel", _GraphOptimizationLevel)
    setattr(ort_mod, "InferenceSession", _create_session)

    class _TokenizerNamespace:
        @staticmethod
        def from_file(path: str) -> _FakeTokenizer:
            return _FakeTokenizer(path, on_encode=on_encode)

    tok_mod = types.ModuleType("tokenizers")
    setattr(tok_mod, "Tokenizer", _TokenizerNamespace)

    if importlib.util.find_spec("numpy") is None:
        monkeypatch.setitem(sys.modules, "numpy", _FakeNumpyModule())
    monkeypatch.setitem(sys.modules, "onnxruntime", ort_mod)
    monkeypatch.setitem(sys.modules, "tokenizers", tok_mod)


@pytest.fixture(autouse=True)
def _clean_embed_slots() -> Any:
    textutil.reset_embedding_state()
    try:
        yield
    finally:
        textutil.reset_embedding_state()


def _warning_records(caplog: pytest.LogCaptureFixture) -> List[logging.LogRecord]:
    return [
        rec
        for rec in caplog.records
        if rec.name == textutil.logger.name and rec.levelno == logging.WARNING
    ]


def test_missing_model_in_dir_a_does_not_block_dir_b_and_warns_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    dir_missing = tmp_path / "profile_missing"
    dir_missing.mkdir()
    dir_ready = tmp_path / "profile_ready"
    _make_model_files(dir_ready)

    ready_key = str(dir_ready.resolve())
    _install_fake_runtime(monkeypatch, vectors_by_dir={ready_key: [3.0, 4.0, 0.0, 0.0]})

    caplog.clear()
    with caplog.at_level(logging.WARNING, logger=textutil.logger.name):
        textutil._configure_embedding(dir_missing)
        vec1, is_real1 = textutil._embed_text("第一条文本")
        vec2, is_real2 = textutil._embed_text("第二条文本")

    assert is_real1 is False and is_real2 is False
    assert len(vec1) == words._DIMS and len(vec2) == words._DIMS
    missing_state = textutil.get_embedding_state(dir_missing)
    assert missing_state.failed is True
    assert missing_state.warned is True
    assert missing_state.session is None
    warnings = _warning_records(caplog)
    assert len(warnings) == 1
    assert str(dir_missing.resolve()) in warnings[0].getMessage()

    textutil._configure_embedding(dir_ready)
    real_vec, is_real3 = textutil._embed_text("切换到有模型的目录")
    assert is_real3 is True
    assert real_vec == pytest.approx([0.6, 0.8, 0.0, 0.0])

    ready_state = textutil.get_embedding_state(dir_ready)
    assert ready_state.failed is False
    assert ready_state.session is not None
    assert textutil._embedding_is_real(dir_ready) is True
    assert textutil._embedding_is_real(dir_missing) is False
    # 缺失目录的槽位依然独立保持 failed=True，不串用 dir_ready 的 session
    assert textutil.get_embedding_state(dir_missing).session is None
    assert textutil.get_embedding_state(dir_missing).failed is True


def test_two_valid_data_dirs_load_distinct_sessions_and_normalize_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dir_a = tmp_path / "profile_a"
    dir_b = tmp_path / "profile_b"
    _make_model_files(dir_a)
    _make_model_files(dir_b)

    key_a = str(dir_a.resolve())
    key_b = str(dir_b.resolve())
    load_counts: Dict[str, int] = {}
    _install_fake_runtime(
        monkeypatch,
        vectors_by_dir={
            key_a: [1.0, 0.0, 0.0, 0.0],
            key_b: [0.0, 1.0, 0.0, 0.0],
        },
        load_counts=load_counts,
    )

    textutil._configure_embedding(dir_a)
    vec_a, real_a = textutil._embed_text("来自目录A")
    assert real_a is True and vec_a == pytest.approx([1.0, 0.0, 0.0, 0.0])

    textutil._configure_embedding(dir_b)
    vec_b, real_b = textutil._embed_text("来自目录B")
    assert real_b is True and vec_b == pytest.approx([0.0, 1.0, 0.0, 0.0])

    # 使用未规范化路径回到目录 A，必须命中既有槽位而不是重复加载或串用 B
    unnormalized_a = dir_a / "." / ".." / dir_a.name
    textutil._configure_embedding(unnormalized_a)
    vec_a2, real_a2 = textutil._embed_text("再次命中目录A")
    assert real_a2 is True and vec_a2 == pytest.approx([1.0, 0.0, 0.0, 0.0])

    state_a = textutil.get_embedding_state(dir_a)
    state_b = textutil.get_embedding_state(dir_b)
    assert state_a.session is not None and state_b.session is not None
    assert state_a.session is not state_b.session
    assert load_counts == {key_a: 1, key_b: 1}


def test_load_failure_only_marks_failing_slot_and_warns_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    dir_corrupt = tmp_path / "corrupt"
    dir_healthy = tmp_path / "healthy"
    _make_model_files(dir_corrupt)
    _make_model_files(dir_healthy)

    corrupt_key = str(dir_corrupt.resolve())
    healthy_key = str(dir_healthy.resolve())
    load_counts: Dict[str, int] = {}
    _install_fake_runtime(
        monkeypatch,
        vectors_by_dir={healthy_key: [0.0, 0.0, 1.0, 0.0]},
        load_counts=load_counts,
        fail_load_dirs={corrupt_key},
    )

    caplog.clear()
    with caplog.at_level(logging.WARNING, logger=textutil.logger.name):
        _, ok1 = textutil._embed_text("坏模型第一次", data_dir=dir_corrupt)
        _, ok2 = textutil._embed_text("坏模型第二次", data_dir=dir_corrupt)

    assert ok1 is False and ok2 is False
    assert load_counts[corrupt_key] == 1
    warnings = _warning_records(caplog)
    assert len(warnings) == 1
    assert "RuntimeError" in warnings[0].getMessage()
    assert textutil.get_embedding_state(dir_corrupt).failed is True

    vec_ok, ok3 = textutil._embed_text("正常模型", data_dir=dir_healthy)
    assert ok3 is True and vec_ok == pytest.approx([0.0, 0.0, 1.0, 0.0])
    assert textutil.get_embedding_state(dir_healthy).failed is False


def test_missing_optional_dependency_marks_only_that_slot_and_warns_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    dir_no_dep = tmp_path / "no_dep"
    dir_with_dep = tmp_path / "with_dep"
    _make_model_files(dir_no_dep)
    _make_model_files(dir_with_dep)

    with_dep_key = str(dir_with_dep.resolve())
    _install_fake_runtime(monkeypatch, vectors_by_dir={with_dep_key: [0.0, 0.0, 0.0, 1.0]})

    saved_tok = sys.modules["tokenizers"]
    monkeypatch.setitem(sys.modules, "tokenizers", None)
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger=textutil.logger.name):
        _, ok1 = textutil._embed_text("缺少依赖1", data_dir=dir_no_dep)
        _, ok2 = textutil._embed_text("缺少依赖2", data_dir=dir_no_dep)

    assert ok1 is False and ok2 is False
    warnings = _warning_records(caplog)
    assert len(warnings) == 1
    assert "ModuleNotFoundError" in warnings[0].getMessage()
    assert textutil.get_embedding_state(dir_no_dep).failed is True

    # 另一目录具备依赖后可独立正常加载，不受 dir_no_dep 影响
    monkeypatch.setitem(sys.modules, "tokenizers", saved_tok)
    vec, ok3 = textutil._embed_text("具备依赖", data_dir=dir_with_dep)
    assert ok3 is True and vec == pytest.approx([0.0, 0.0, 0.0, 1.0])
    assert textutil.get_embedding_state(dir_with_dep).failed is False
    assert textutil.get_embedding_state(dir_no_dep).failed is True


def test_runtime_inference_error_warns_falls_back_once_and_recovers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    dir_ok = tmp_path / "recoverable"
    _make_model_files(dir_ok)
    dir_hash_ref = tmp_path / "hash_reference"
    dir_hash_ref.mkdir()
    ok_key = str(dir_ok.resolve())
    load_counts: Dict[str, int] = {}

    fail_run_once = {"armed": False}
    fail_encode_once = {"armed": False}

    def _on_run(_model_path: str, _feed: Dict[str, Any]) -> None:
        if fail_run_once["armed"]:
            fail_run_once["armed"] = False
            raise RuntimeError("transient ort execution error")

    def _on_encode(_tok_path: str, _texts: List[str]) -> None:
        if fail_encode_once["armed"]:
            fail_encode_once["armed"] = False
            raise ValueError("transient tokenizer failure")

    _install_fake_runtime(
        monkeypatch,
        vectors_by_dir={ok_key: [1.0, 0.0, 0.0, 0.0]},
        load_counts=load_counts,
        on_run=_on_run,
        on_encode=_on_encode,
    )

    textutil._configure_embedding(dir_ok)
    vec_warm, real_warm = textutil._embed_text("预热")
    assert real_warm is True and vec_warm == pytest.approx([1.0, 0.0, 0.0, 0.0])
    initial_session = textutil.get_embedding_state(dir_ok).session
    assert initial_session is not None

    # 1) session.run 单次异常：记 warning、本次回退散列、绝不永久置 failed=True
    fail_run_once["armed"] = True
    secret_text = "绝不可写入日志的敏感对话正文"
    expected_run_hash, expected_run_real = textutil._embed_text(secret_text, data_dir=dir_hash_ref)
    assert expected_run_real is False
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger=textutil.logger.name):
        vec_fallback, real_fallback = textutil._embed_text(secret_text)

    assert real_fallback is False
    assert vec_fallback == expected_run_hash
    run_warnings = _warning_records(caplog)
    assert len(run_warnings) == 1
    msg = run_warnings[0].getMessage()
    assert "RuntimeError" in msg
    assert secret_text not in msg
    state_after_err = textutil.get_embedding_state(dir_ok)
    assert state_after_err.failed is False
    assert state_after_err.session is initial_session

    # 下一次正常请求立即自动恢复 ONNX，且复用原 session（未重新加载）
    vec_recovered, real_recovered = textutil._embed_text("恢复后的正常查询")
    assert real_recovered is True and vec_recovered == pytest.approx([1.0, 0.0, 0.0, 0.0])
    assert load_counts[ok_key] == 1

    # 2) tokenizer.encode_batch 单次异常同样不得锁死槽位
    expected_tok_hash, expected_tok_real = textutil._embed_text("编码临时异常", data_dir=dir_hash_ref)
    assert expected_tok_real is False
    caplog.clear()
    fail_encode_once["armed"] = True
    with caplog.at_level(logging.WARNING, logger=textutil.logger.name):
        vec_tok_fb, real_tok_fb = textutil._embed_text("编码临时异常")
    assert real_tok_fb is False
    assert vec_tok_fb == expected_tok_hash
    tok_warnings = _warning_records(caplog)
    assert len(tok_warnings) == 1 and "ValueError" in tok_warnings[0].getMessage()
    assert textutil.get_embedding_state(dir_ok).failed is False

    vec_recovered2, real_recovered2 = textutil._embed_text("编码异常后立即恢复")
    assert real_recovered2 is True and vec_recovered2 == pytest.approx([1.0, 0.0, 0.0, 0.0])
    assert load_counts[ok_key] == 1


def test_multithreaded_concurrent_load_and_inference_isolated_and_race_free(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    dir_a = tmp_path / "mt_a"
    dir_b = tmp_path / "mt_b"
    dir_missing = tmp_path / "mt_missing"
    _make_model_files(dir_a)
    _make_model_files(dir_b)
    dir_missing.mkdir()

    key_a = str(dir_a.resolve())
    key_b = str(dir_b.resolve())
    load_counts: Dict[str, int] = {}
    _install_fake_runtime(
        monkeypatch,
        vectors_by_dir={
            key_a: [1.0, 0.0, 0.0, 0.0],
            key_b: [0.0, 1.0, 0.0, 0.0],
        },
        load_counts=load_counts,
        load_delay_sec=0.02,
    )

    assignments: Sequence[Tuple[Path, Optional[List[float]]]] = [
        *[(dir_a, [1.0, 0.0, 0.0, 0.0]) for _ in range(4)],
        *[(dir_b, [0.0, 1.0, 0.0, 0.0]) for _ in range(4)],
        *[(dir_missing, None) for _ in range(4)],
    ]
    n_threads = len(assignments)
    start_barrier = threading.Barrier(n_threads)
    configured_barrier = threading.Barrier(n_threads)
    errors: List[BaseException] = []
    errors_lock = threading.Lock()

    def _worker(idx: int, target_dir: Path, expected_vec: Optional[List[float]]) -> None:
        try:
            start_barrier.wait(timeout=5.0)
            textutil._configure_embedding(target_dir)
            configured_barrier.wait(timeout=5.0)
            for step in range(15):
                text = f"thread-{idx}-step-{step}"
                vec, is_real = textutil._embed_text(text)
                if expected_vec is None:
                    assert is_real is False
                    assert len(vec) == words._DIMS
                else:
                    assert is_real is True
                    assert vec == pytest.approx(expected_vec)
        except BaseException as exc:  # pragma: no cover - surfaced via assertion below
            with errors_lock:
                errors.append(exc)

    caplog.clear()
    with caplog.at_level(logging.WARNING, logger=textutil.logger.name):
        threads = [
            threading.Thread(target=_worker, args=(i, d, v))
            for i, (d, v) in enumerate(assignments)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=15.0)
            assert not t.is_alive()

    assert errors == [], f"并发线程出现错误: {errors!r}"
    assert load_counts == {key_a: 1, key_b: 1}
    assert textutil.get_embedding_state(dir_a).session is not textutil.get_embedding_state(dir_b).session
    assert textutil.get_embedding_state(dir_missing).failed is True
    missing_warnings = [
        rec for rec in _warning_records(caplog) if str(dir_missing.resolve()) in rec.getMessage()
    ]
    assert len(missing_warnings) == 1


def test_async_tasks_contextvar_isolation_on_same_thread(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dir_a = tmp_path / "async_a"
    dir_b = tmp_path / "async_b"
    _make_model_files(dir_a)
    _make_model_files(dir_b)

    key_a = str(dir_a.resolve())
    key_b = str(dir_b.resolve())
    _install_fake_runtime(
        monkeypatch,
        vectors_by_dir={
            key_a: [1.0, 0.0, 0.0, 0.0],
            key_b: [0.0, 1.0, 0.0, 0.0],
        },
    )

    async def _run_pair() -> Tuple[Tuple[List[float], bool], Tuple[List[float], bool]]:
        both_configured = asyncio.Event()
        configured = 0

        async def _task(target_dir: Path) -> Tuple[List[float], bool]:
            nonlocal configured
            textutil._configure_embedding(target_dir)
            configured += 1
            if configured == 2:
                both_configured.set()
            await both_configured.wait()
            await asyncio.sleep(0)
            return textutil._embed_text("协程隔离检查")

        res_a, res_b = await asyncio.gather(_task(dir_a), _task(dir_b))
        return res_a, res_b

    (vec_a, real_a), (vec_b, real_b) = asyncio.run(_run_pair())
    assert real_a is True and vec_a == pytest.approx([1.0, 0.0, 0.0, 0.0])
    assert real_b is True and vec_b == pytest.approx([0.0, 1.0, 0.0, 0.0])


def test_background_thread_falls_back_to_default_configured_dir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dir_main = tmp_path / "main_profile"
    _make_model_files(dir_main)
    main_key = str(dir_main.resolve())
    _install_fake_runtime(monkeypatch, vectors_by_dir={main_key: [0.6, 0.0, 0.8, 0.0]})

    textutil._configure_embedding(dir_main)
    result: Dict[str, Any] = {}

    def _bg() -> None:
        result["value"] = textutil._embed_text("后台线程写入")

    worker = threading.Thread(target=_bg)
    worker.start()
    worker.join(timeout=5.0)
    assert not worker.is_alive()
    vec, is_real = result["value"]
    assert is_real is True and vec == pytest.approx([0.6, 0.0, 0.8, 0.0])


def test_embed_site_packages_locked_and_deduplicated(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_dir = tmp_path / "with_venv"
    _make_model_files(data_dir)
    site_dir = (
        data_dir
        / "venv"
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
    )
    site_dir.mkdir(parents=True)

    data_dir_2 = tmp_path / "with_venv_2"
    _make_model_files(data_dir_2)
    site_dir_2 = (
        data_dir_2
        / "venv"
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
    )
    site_dir_2.mkdir(parents=True)

    key_1 = str(data_dir.resolve())
    key_2 = str(data_dir_2.resolve())
    _install_fake_runtime(
        monkeypatch,
        vectors_by_dir={
            key_1: [1.0, 0.0, 0.0, 0.0],
            key_2: [0.0, 1.0, 0.0, 0.0],
        },
        load_delay_sec=0.01,
    )

    calls: List[str] = []
    calls_lock = threading.Lock()

    def _spy_addsitedir(path: str, _known_paths: Any = None) -> None:
        time.sleep(0.01)
        with calls_lock:
            calls.append(path)

    monkeypatch.setattr(site, "addsitedir", _spy_addsitedir)

    barrier = threading.Barrier(8)
    errors: List[BaseException] = []
    errors_lock = threading.Lock()

    def _runner() -> None:
        try:
            barrier.wait(timeout=5.0)
            vec, is_real = textutil._embed_text("并发触发 venv site-packages 挂载", data_dir=data_dir)
            assert is_real is True and vec == pytest.approx([1.0, 0.0, 0.0, 0.0])
        except BaseException as exc:  # pragma: no cover - surfaced via assertion below
            with errors_lock:
                errors.append(exc)

    threads = [threading.Thread(target=_runner) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10.0)
        assert not t.is_alive()

    assert errors == []
    assert calls == [str(site_dir.resolve())]

    vec2, is_real2 = textutil._embed_text("第二个目录挂载独立 venv", data_dir=data_dir_2)
    assert is_real2 is True and vec2 == pytest.approx([0.0, 1.0, 0.0, 0.0])
    assert calls == [str(site_dir.resolve()), str(site_dir_2.resolve())]


def _read_event_vector(data_dir: Path, event_id: int) -> List[float]:
    with sqlite3.connect(data_dir / "sanae.db") as conn:
        row = conn.execute("SELECT vec FROM events WHERE id = ?", (event_id,)).fetchone()
    assert row is not None, f"event {event_id} missing from {data_dir / 'sanae.db'}"
    blob: bytes = row[0]
    assert blob and len(blob) % 4 == 0
    return list(struct.unpack(f"<{len(blob) // 4}f", blob))


def _provider_turn_scope(*, session_id: str, user_id: str, chat_id: str, message_id: str) -> Any:
    return host_bridge.turn_envelope_scope(
        platform="test",
        chat_type="dm",
        chat_id=chat_id,
        sender_id=user_id,
        session_id=session_id,
        session_key=f"test:{session_id}",
        message_id=message_id,
        is_messaging_surface=False,
    )


def test_provider_a_keeps_onnx_after_provider_b_without_model_initializes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    home_a = tmp_path / "home_a_with_model"
    data_a = home_a / "sanae-mem"
    _make_model_files(data_a)

    home_b = tmp_path / "home_b_missing_model"
    data_b = home_b / "sanae-mem"
    data_b.mkdir(parents=True)

    key_a = str(data_a.resolve())
    model_a = str((data_a / words._EMBED_MODEL_SUBDIR / "model.onnx").resolve())
    load_counts: Dict[str, int] = {}
    run_models: List[str] = []

    def _record_run(model_path: str, _feed: Dict[str, Any]) -> None:
        run_models.append(str(Path(model_path).resolve()))

    _install_fake_runtime(
        monkeypatch,
        vectors_by_dir={key_a: [1.0, 0.0, 0.0, 0.0]},
        load_counts=load_counts,
        on_run=_record_run,
    )
    for env_name in ("HERMES_HOME", "HERMES_HOME_OVERRIDE"):
        monkeypatch.setenv(env_name, str(home_a))

    provider_a = SanaeMemProvider(
        {"data_dir": str(data_a), "semantic_backend": "sqlite", "shadow": False, "read_gate": "explicit"}
    )
    provider_b = SanaeMemProvider(
        {"data_dir": str(data_b), "semantic_backend": "sqlite", "shadow": False, "read_gate": "explicit"}
    )
    try:
        # Step 1: Provider A 初始化并入库第一条事件
        with _provider_turn_scope(session_id="s-a", user_id="user-a", chat_id="chat-a", message_id="m-a1"):
            provider_a.initialize(
                "s-a", hermes_home=str(home_a), platform="test", agent_context="primary", user_id="user-a"
            )
            add_a1 = json.loads(
                provider_a.handle_tool_call(
                    "sanae_mem_add", {"content": "在旧书店买了一本植物图鉴", "importance": 0.85}
                )
            )
            assert add_a1["ok"] is True
            event_a1_id = int(add_a1["result"]["id"])

        assert _read_event_vector(data_a, event_a1_id) == pytest.approx([1.0, 0.0, 0.0, 0.0])
        assert run_models == [model_a]

        # Step 2: 同进程同线程初始化缺模型的 Provider B 并入库
        run_models.clear()
        with _provider_turn_scope(session_id="s-b", user_id="user-b", chat_id="chat-b", message_id="m-b1"):
            provider_b.initialize(
                "s-b", hermes_home=str(home_b), platform="test", agent_context="primary", user_id="user-b"
            )
            add_b1 = json.loads(
                provider_b.handle_tool_call(
                    "sanae_mem_add", {"content": "在便利店买了一瓶矿泉水", "importance": 0.85}
                )
            )
            assert add_b1["ok"] is True
            event_b1_id = int(add_b1["result"]["id"])

        vec_b1 = _read_event_vector(data_b, event_b1_id)
        assert len(vec_b1) == words._DIMS
        assert run_models == []
        assert textutil.get_embedding_state(data_b).failed is True

        # Step 3: 切回 Provider A 会话再次入库与检索（不重新 initialize）
        run_models.clear()
        with _provider_turn_scope(session_id="s-a", user_id="user-a", chat_id="chat-a", message_id="m-a2"):
            add_a2 = json.loads(
                provider_a.handle_tool_call(
                    "sanae_mem_add", {"content": "周末约好去海边看日出", "importance": 0.85}
                )
            )
            assert add_a2["ok"] is True
            event_a2_id = int(add_a2["result"]["id"])

            vec_a2 = _read_event_vector(data_a, event_a2_id)
            assert vec_a2 == pytest.approx([1.0, 0.0, 0.0, 0.0])
            assert run_models == [model_a]

            run_models.clear()
            search_res = json.loads(
                provider_a.handle_tool_call(
                    "sanae_mem_search", {"query": "完全没有字面重合的语义回忆", "top_k": 5}
                )
            )
            assert search_res["ok"] is True
            found_ids = {int(item["id"]) for item in search_res["result"]}
            assert {event_a1_id, event_a2_id} <= found_ids
            assert run_models == [model_a]

            run_models.clear()
            injected_block = provider_a.prefetch("还记得我之前提过的安排吗？", session_id="s-a")
            assert injected_block, "Provider A 仍走真 ONNX 嵌入，显式追问必须通过证据门注入而非散列 fail-closed"
            assert run_models == [model_a]

        assert load_counts == {key_a: 1}
        assert textutil.get_embedding_state(data_a).failed is False
        assert textutil.get_embedding_state(data_b).failed is True
    finally:
        provider_a.shutdown()
        provider_b.shutdown()


def test_provider_a_never_uses_provider_b_model_after_b_initializes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    home_a = tmp_path / "home_a_model_a"
    data_a = home_a / "sanae-mem"
    _make_model_files(data_a)

    home_b = tmp_path / "home_b_model_b"
    data_b = home_b / "sanae-mem"
    _make_model_files(data_b)

    key_a = str(data_a.resolve())
    key_b = str(data_b.resolve())
    model_a = str((data_a / words._EMBED_MODEL_SUBDIR / "model.onnx").resolve())
    model_b = str((data_b / words._EMBED_MODEL_SUBDIR / "model.onnx").resolve())
    load_counts: Dict[str, int] = {}
    run_models: List[str] = []

    def _record_run(model_path: str, _feed: Dict[str, Any]) -> None:
        run_models.append(str(Path(model_path).resolve()))

    _install_fake_runtime(
        monkeypatch,
        vectors_by_dir={
            key_a: [1.0, 0.0, 0.0, 0.0],
            key_b: [0.0, 1.0, 0.0, 0.0],
        },
        load_counts=load_counts,
        on_run=_record_run,
    )
    for env_name in ("HERMES_HOME", "HERMES_HOME_OVERRIDE"):
        monkeypatch.setenv(env_name, str(home_a))

    provider_a = SanaeMemProvider(
        {"data_dir": str(data_a), "semantic_backend": "sqlite", "shadow": False, "read_gate": "explicit"}
    )
    provider_b = SanaeMemProvider(
        {"data_dir": str(data_b), "semantic_backend": "sqlite", "shadow": False, "read_gate": "explicit"}
    )
    try:
        # Step 1: Provider A 初始化并入库第一条事件
        with _provider_turn_scope(session_id="s-a", user_id="user-a", chat_id="chat-a", message_id="m-a1"):
            provider_a.initialize(
                "s-a", hermes_home=str(home_a), platform="test", agent_context="primary", user_id="user-a"
            )
            add_a1 = json.loads(
                provider_a.handle_tool_call(
                    "sanae_mem_add", {"content": "在旧书店买了一本植物图鉴", "importance": 0.85}
                )
            )
            assert add_a1["ok"] is True
            event_a1_id = int(add_a1["result"]["id"])

        assert _read_event_vector(data_a, event_a1_id) == pytest.approx([1.0, 0.0, 0.0, 0.0])
        assert run_models == [model_a]

        # Step 2: 同进程同线程初始化拥有独立不同模型的 Provider B 并入库
        run_models.clear()
        with _provider_turn_scope(session_id="s-b", user_id="user-b", chat_id="chat-b", message_id="m-b1"):
            provider_b.initialize(
                "s-b", hermes_home=str(home_b), platform="test", agent_context="primary", user_id="user-b"
            )
            add_b1 = json.loads(
                provider_b.handle_tool_call(
                    "sanae_mem_add", {"content": "在公园拍了秋天的银杏叶", "importance": 0.85}
                )
            )
            assert add_b1["ok"] is True
            event_b1_id = int(add_b1["result"]["id"])

        assert _read_event_vector(data_b, event_b1_id) == pytest.approx([0.0, 1.0, 0.0, 0.0])
        assert run_models == [model_b]

        # Step 3: 切回 Provider A 会话再次入库与检索（不重新 initialize）
        run_models.clear()
        with _provider_turn_scope(session_id="s-a", user_id="user-a", chat_id="chat-a", message_id="m-a2"):
            add_a2 = json.loads(
                provider_a.handle_tool_call(
                    "sanae_mem_add", {"content": "周末约好去海边看日出", "importance": 0.85}
                )
            )
            assert add_a2["ok"] is True
            event_a2_id = int(add_a2["result"]["id"])

            vec_a2 = _read_event_vector(data_a, event_a2_id)
            assert vec_a2 == pytest.approx([1.0, 0.0, 0.0, 0.0])
            assert run_models == [model_a]

            run_models.clear()
            search_res = json.loads(
                provider_a.handle_tool_call(
                    "sanae_mem_search", {"query": "完全没有字面重合的语义回忆", "top_k": 5}
                )
            )
            assert search_res["ok"] is True
            found_ids = {int(item["id"]) for item in search_res["result"]}
            assert {event_a1_id, event_a2_id} <= found_ids
            assert run_models == [model_a]

            run_models.clear()
            injected_block = provider_a.prefetch("还记得我之前提过的安排吗？", session_id="s-a")
            assert injected_block, "Provider A 检索必须走模型 A 与库内向量匹配，绝不串用正交的模型 B"
            assert run_models == [model_a]

        assert load_counts == {key_a: 1, key_b: 1}
        assert textutil.get_embedding_state(data_a).session is not textutil.get_embedding_state(data_b).session
    finally:
        provider_a.shutdown()
        provider_b.shutdown()


def test_trailing_unicode_whitespace_data_dir_does_not_collide_with_stripped_sibling(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    home_plain = tmp_path / "home_plain"
    home_space = tmp_path / "home_space"
    home_plain.mkdir()
    home_space.mkdir()

    data_plain = tmp_path / "profile-data"
    data_space = tmp_path / "profile-data\u3000"
    _make_model_files(data_plain)
    _make_model_files(data_space)

    key_plain = str(data_plain.resolve())
    key_space = str(data_space.resolve())
    assert key_plain != key_space

    model_plain = str((data_plain / words._EMBED_MODEL_SUBDIR / "model.onnx").resolve())
    model_space = str((data_space / words._EMBED_MODEL_SUBDIR / "model.onnx").resolve())
    load_counts: Dict[str, int] = {}
    run_models: List[str] = []

    def _record_run(model_path: str, _feed: Dict[str, Any]) -> None:
        run_models.append(str(Path(model_path).resolve()))

    _install_fake_runtime(
        monkeypatch,
        vectors_by_dir={
            key_plain: [1.0, 0.0, 0.0, 0.0],
            key_space: [0.0, 1.0, 0.0, 0.0],
        },
        load_counts=load_counts,
        on_run=_record_run,
    )
    for env_name in ("HERMES_HOME", "HERMES_HOME_OVERRIDE"):
        monkeypatch.setenv(env_name, str(home_plain))

    provider_plain = SanaeMemProvider(
        {"data_dir": str(data_plain), "semantic_backend": "sqlite", "shadow": False, "read_gate": "explicit"}
    )
    provider_space = SanaeMemProvider(
        {"data_dir": str(data_space), "semantic_backend": "sqlite", "shadow": False, "read_gate": "explicit"}
    )
    try:
        with _provider_turn_scope(
            session_id="s-plain", user_id="user-plain", chat_id="chat-plain", message_id="m-p1"
        ):
            provider_plain.initialize(
                "s-plain",
                hermes_home=str(home_plain),
                platform="test",
                agent_context="primary",
                user_id="user-plain",
            )
            add_p1 = json.loads(
                provider_plain.handle_tool_call(
                    "sanae_mem_add", {"content": "在旧书店买了一本植物图鉴", "importance": 0.85}
                )
            )
            assert add_p1["ok"] is True
            event_p1_id = int(add_p1["result"]["id"])

        assert _read_event_vector(data_plain, event_p1_id) == pytest.approx([1.0, 0.0, 0.0, 0.0])
        assert run_models == [model_plain]

        run_models.clear()
        with _provider_turn_scope(
            session_id="s-space", user_id="user-space", chat_id="chat-space", message_id="m-s1"
        ):
            provider_space.initialize(
                "s-space",
                hermes_home=str(home_space),
                platform="test",
                agent_context="primary",
                user_id="user-space",
            )
            add_s1 = json.loads(
                provider_space.handle_tool_call(
                    "sanae_mem_add", {"content": "在公园拍了秋天的银杏叶", "importance": 0.85}
                )
            )
            assert add_s1["ok"] is True
            event_s1_id = int(add_s1["result"]["id"])

        assert _read_event_vector(data_space, event_s1_id) == pytest.approx([0.0, 1.0, 0.0, 0.0])
        assert run_models == [model_space]

        # 切回 profile-data 再次入库与检索，必须严格命中 profile-data 槽位
        run_models.clear()
        with _provider_turn_scope(
            session_id="s-plain", user_id="user-plain", chat_id="chat-plain", message_id="m-p2"
        ):
            add_p2 = json.loads(
                provider_plain.handle_tool_call(
                    "sanae_mem_add", {"content": "周末约好去海边看日出", "importance": 0.85}
                )
            )
            assert add_p2["ok"] is True
            event_p2_id = int(add_p2["result"]["id"])
            assert _read_event_vector(data_plain, event_p2_id) == pytest.approx([1.0, 0.0, 0.0, 0.0])
            assert run_models == [model_plain]

            run_models.clear()
            plain_search = json.loads(
                provider_plain.handle_tool_call(
                    "sanae_mem_search", {"query": "完全没有字面重合的语义回忆", "top_k": 5}
                )
            )
            assert plain_search["ok"] is True
            assert {event_p1_id, event_p2_id} <= {int(item["id"]) for item in plain_search["result"]}
            assert run_models == [model_plain]

            run_models.clear()
            assert provider_plain.prefetch("还记得我之前提过的安排吗？", session_id="s-plain")
            assert run_models == [model_plain]

        # 切回 profile-data\u3000 再次入库与检索，必须严格命中带尾随全角空白目录的独立槽位
        run_models.clear()
        with _provider_turn_scope(
            session_id="s-space", user_id="user-space", chat_id="chat-space", message_id="m-s2"
        ):
            add_s2 = json.loads(
                provider_space.handle_tool_call(
                    "sanae_mem_add", {"content": "晚上打算烤一盘曲奇饼干", "importance": 0.85}
                )
            )
            assert add_s2["ok"] is True
            event_s2_id = int(add_s2["result"]["id"])
            assert _read_event_vector(data_space, event_s2_id) == pytest.approx([0.0, 1.0, 0.0, 0.0])
            assert run_models == [model_space]

            run_models.clear()
            space_search = json.loads(
                provider_space.handle_tool_call(
                    "sanae_mem_search", {"query": "完全没有字面重合的语义回忆", "top_k": 5}
                )
            )
            assert space_search["ok"] is True
            assert {event_s1_id, event_s2_id} <= {int(item["id"]) for item in space_search["result"]}
            assert run_models == [model_space]

            run_models.clear()
            assert provider_space.prefetch("还记得我之前提过的安排吗？", session_id="s-space")
            assert run_models == [model_space]

        state_plain = textutil.get_embedding_state(data_plain)
        state_space = textutil.get_embedding_state(data_space)
        assert state_plain.dir == key_plain
        assert state_space.dir == key_space
        assert state_plain.session is not None and state_space.session is not None
        assert state_plain.session is not state_space.session
        assert load_counts == {key_plain: 1, key_space: 1}
    finally:
        provider_plain.shutdown()
        provider_space.shutdown()


def test_relative_unicode_whitespace_data_dir_resolves_and_shares_slot_with_absolute(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    home_ws = tmp_path / "home_rel_ws"
    home_reg = tmp_path / "home_regular"
    home_ws.mkdir()
    home_reg.mkdir()

    abs_ws_dir = tmp_path / "\u3000"
    reg_dir = tmp_path / "regular"
    _make_model_files(abs_ws_dir)
    _make_model_files(reg_dir)

    key_ws = str(abs_ws_dir.resolve())
    key_reg = str(reg_dir.resolve())
    assert key_ws != key_reg

    model_ws = str((abs_ws_dir / words._EMBED_MODEL_SUBDIR / "model.onnx").resolve())
    model_reg = str((reg_dir / words._EMBED_MODEL_SUBDIR / "model.onnx").resolve())
    load_counts: Dict[str, int] = {}
    run_models: List[str] = []

    def _record_run(model_path: str, _feed: Dict[str, Any]) -> None:
        run_models.append(str(Path(model_path).resolve()))

    _install_fake_runtime(
        monkeypatch,
        vectors_by_dir={
            key_ws: [0.0, 1.0, 0.0, 0.0],
            key_reg: [1.0, 0.0, 0.0, 0.0],
        },
        load_counts=load_counts,
        on_run=_record_run,
    )
    for env_name in ("HERMES_HOME", "HERMES_HOME_OVERRIDE"):
        monkeypatch.setenv(env_name, str(home_ws))
    monkeypatch.chdir(tmp_path)

    provider_ws = SanaeMemProvider(
        {"data_dir": "\u3000", "semantic_backend": "sqlite", "shadow": False, "read_gate": "explicit"}
    )
    provider_reg = SanaeMemProvider(
        {"data_dir": str(reg_dir), "semantic_backend": "sqlite", "shadow": False, "read_gate": "explicit"}
    )
    try:
        with _provider_turn_scope(
            session_id="s-ws", user_id="user-ws", chat_id="chat-ws", message_id="m-ws1"
        ):
            provider_ws.initialize(
                "s-ws",
                hermes_home=str(home_ws),
                platform="test",
                agent_context="primary",
                user_id="user-ws",
            )
            add_ws1 = json.loads(
                provider_ws.handle_tool_call(
                    "sanae_mem_add", {"content": "在旧书店买了一本植物图鉴", "importance": 0.85}
                )
            )
            assert add_ws1["ok"] is True
            event_ws1_id = int(add_ws1["result"]["id"])

        assert _read_event_vector(abs_ws_dir, event_ws1_id) == pytest.approx([0.0, 1.0, 0.0, 0.0])
        assert run_models == [model_ws]

        state_rel = textutil.get_embedding_state(Path("\u3000"))
        state_abs = textutil.get_embedding_state(abs_ws_dir)
        assert state_rel.dir == key_ws
        assert state_rel == state_abs
        assert state_rel.failed is False
        assert state_rel.session is not None

        # 用绝对路径调用必须命中同一已加载槽位（零重复加载）
        run_models.clear()
        vec_abs, real_abs = textutil._embed_text("绝对路径同槽校验", data_dir=abs_ws_dir)
        assert real_abs is True and vec_abs == pytest.approx([0.0, 1.0, 0.0, 0.0])
        assert run_models == [model_ws]
        assert load_counts == {key_ws: 1}

        # 同进程初始化常规目录 provider_reg，双方模型互不串扰
        run_models.clear()
        with _provider_turn_scope(
            session_id="s-reg", user_id="user-reg", chat_id="chat-reg", message_id="m-reg1"
        ):
            provider_reg.initialize(
                "s-reg",
                hermes_home=str(home_reg),
                platform="test",
                agent_context="primary",
                user_id="user-reg",
            )
            add_reg1 = json.loads(
                provider_reg.handle_tool_call(
                    "sanae_mem_add", {"content": "在公园拍了秋天的银杏叶", "importance": 0.85}
                )
            )
            assert add_reg1["ok"] is True
            event_reg1_id = int(add_reg1["result"]["id"])

        assert _read_event_vector(reg_dir, event_reg1_id) == pytest.approx([1.0, 0.0, 0.0, 0.0])
        assert run_models == [model_reg]

        # 切回相对空白目录 provider_ws 再次入库与检索，仍走独立模型不退化散列
        run_models.clear()
        with _provider_turn_scope(
            session_id="s-ws", user_id="user-ws", chat_id="chat-ws", message_id="m-ws2"
        ):
            add_ws2 = json.loads(
                provider_ws.handle_tool_call(
                    "sanae_mem_add", {"content": "周末约好去海边看日出", "importance": 0.85}
                )
            )
            assert add_ws2["ok"] is True
            event_ws2_id = int(add_ws2["result"]["id"])
            assert _read_event_vector(abs_ws_dir, event_ws2_id) == pytest.approx([0.0, 1.0, 0.0, 0.0])
            assert run_models == [model_ws]

            run_models.clear()
            search_ws = json.loads(
                provider_ws.handle_tool_call(
                    "sanae_mem_search", {"query": "完全没有字面重合的语义回忆", "top_k": 5}
                )
            )
            assert search_ws["ok"] is True
            assert {event_ws1_id, event_ws2_id} <= {int(item["id"]) for item in search_ws["result"]}
            assert run_models == [model_ws]

            run_models.clear()
            assert provider_ws.prefetch("还记得我之前提过的安排吗？", session_id="s-ws")
            assert run_models == [model_ws]

        assert load_counts == {key_ws: 1, key_reg: 1}
        assert textutil.get_embedding_state(Path("\u3000")).session is not textutil.get_embedding_state(reg_dir).session
    finally:
        provider_ws.shutdown()
        provider_reg.shutdown()


def test_two_providers_initialize_with_isolated_embedding_slots(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    home_missing = tmp_path / "home_missing"
    data_missing = home_missing / "sanae-mem"
    data_missing.mkdir(parents=True)

    home_ready = tmp_path / "home_ready"
    data_ready = home_ready / "sanae-mem"
    _make_model_files(data_ready)

    ready_key = str(data_ready.resolve())
    _install_fake_runtime(monkeypatch, vectors_by_dir={ready_key: [0.0, 0.6, 0.8, 0.0]})

    for env_name in ("HERMES_HOME", "HERMES_HOME_OVERRIDE"):
        monkeypatch.setenv(env_name, str(home_missing))

    provider_missing = SanaeMemProvider(
        {"data_dir": str(data_missing), "semantic_backend": "sqlite", "shadow": False}
    )
    provider_ready = SanaeMemProvider(
        {"data_dir": str(data_ready), "semantic_backend": "sqlite", "shadow": False}
    )
    try:
        with host_bridge.turn_envelope_scope(
            platform="test",
            chat_type="dm",
            chat_id="c1",
            sender_id="u1",
            session_id="s-missing",
            session_key="test:s-missing",
            message_id="m1",
            is_messaging_surface=False,
        ):
            provider_missing.initialize(
                "s-missing",
                hermes_home=str(home_missing),
                platform="test",
                agent_context="primary",
                user_id="u1",
            )
            vec_m, real_m = textutil._embed_text("缺失模型目录")
            assert real_m is False and len(vec_m) == words._DIMS
            assert textutil.get_embedding_state(data_missing).failed is True

        with host_bridge.turn_envelope_scope(
            platform="test",
            chat_type="dm",
            chat_id="c2",
            sender_id="u2",
            session_id="s-ready",
            session_key="test:s-ready",
            message_id="m2",
            is_messaging_surface=False,
        ):
            provider_ready.initialize(
                "s-ready",
                hermes_home=str(home_ready),
                platform="test",
                agent_context="primary",
                user_id="u2",
            )
            vec_r, real_r = textutil._embed_text("有模型目录")
            assert real_r is True and vec_r == pytest.approx([0.0, 0.6, 0.8, 0.0])
            assert textutil.get_embedding_state(data_ready).failed is False
            assert textutil.get_embedding_state(data_ready).session is not None
            assert textutil.get_embedding_state(data_missing).failed is True
            assert textutil.get_embedding_state(data_missing).session is None
    finally:
        provider_missing.shutdown()
        provider_ready.shutdown()
