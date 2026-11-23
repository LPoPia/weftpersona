"""Shared deterministic test harness for ViveMind integration and subprocess suites."""

from __future__ import annotations

from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
from typing import Any, Callable, Iterator, Mapping

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
PLUGIN_ROOT = REPO_ROOT / "plugins"
TESTS_ROOT = Path(__file__).resolve().parent

_HERMES_CANDIDATE_NAMES: tuple[str, ...] = (
    "swarm-private-chain-hermes",
    "host",
    "multiplatform-batch-hermes",
    "hermes-agent",
    "hermes-minimal-host",
    "hermes-minimal-host-20261112",
)


def _is_full_hermes_host(candidate: Path) -> bool:
    return (
        (candidate / "hermes_cli" / "plugins.py").is_file()
        and (candidate / "gateway" / "run.py").is_file()
    )


def _has_host_contracts(candidate: Path) -> bool:
    return (candidate / "hermes_host_contracts.py").is_file()


def resolve_hermes_host_root(*, require_full_host: bool = False) -> Path | None:
    """Return a Hermes host checkout directory, preferring full host checkouts over contracts-only roots."""
    contracts_fallback: Path | None = None

    for env_key in ("HERMES_TEST_ROOT", "VIVEMIND_HERMES_SOURCE"):
        env_root = os.environ.get(env_key)
        if not env_root:
            continue
        candidate = Path(env_root).expanduser().resolve()
        if _is_full_hermes_host(candidate):
            return candidate
        if _has_host_contracts(candidate) and contracts_fallback is None:
            contracts_fallback = candidate

    for mod_name in ("hermes_constants", "hermes_host_contracts"):
        spec = importlib.util.find_spec(mod_name)
        if spec is None or not spec.origin:
            continue
        candidate = Path(spec.origin).resolve().parent
        if _is_full_hermes_host(candidate):
            return candidate
        if _has_host_contracts(candidate) and contracts_fallback is None:
            contracts_fallback = candidate

    sibling_bases = [REPO_ROOT.parent]
    for ancestor in REPO_ROOT.parents[:4]:
        sibling_bases.append(ancestor / "源码")
    for base in sibling_bases:
        if not base.is_dir():
            continue
        for name in _HERMES_CANDIDATE_NAMES:
            cand = base / name
            if _is_full_hermes_host(cand):
                return cand

    tmp_fallback = Path(tempfile.gettempdir()) / "vivemind-hermes-6093627"
    if _is_full_hermes_host(tmp_fallback):
        return tmp_fallback
    if not require_full_host:
        if contracts_fallback is not None:
            return contracts_fallback
        if _has_host_contracts(tmp_fallback):
            return tmp_fallback
    return None


def resolve_vivemind_and_hermes_roots(*, skip_if_missing: bool = True) -> tuple[Path, Path]:
    """Locate the ViveMind checkout root and an adjacent/configured Hermes host checkout."""
    hermes_root = resolve_hermes_host_root(require_full_host=True)
    if hermes_root is None or not _is_full_hermes_host(hermes_root):
        if skip_if_missing:
            pytest.skip("hermes-agent checkout not present")
        raise FileNotFoundError(f"hermes-agent checkout not present (resolved={hermes_root})")
    return REPO_ROOT, hermes_root


def run_clean_subprocess(
    script: str,
    *,
    vivemind_root: Path,
    hermes_root: Path,
    home: Path,
    result_file: Path | None = None,
    timeout: int = 90,
) -> dict[str, Any]:
    """Execute *script* in an isolated Python subprocess and return its trailing JSON payload."""
    env = {
        **os.environ,
        "HERMES_HOME": str(home),
        "HERMES_TEST_ISOLATION": str(home),
        "PYTHONPATH": os.pathsep.join([
            str(vivemind_root),
            str(vivemind_root / "tests"),
            str(vivemind_root / "plugins"),
            str(hermes_root),
        ]),
        "PYTHONIOENCODING": "utf-8",
    }
    env.pop("HERMES_HOME_OVERRIDE", None)
    env.pop("HERMES_SAFE_MODE", None)
    proc = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(hermes_root),
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=timeout,
    )
    assert proc.returncode == 0, (
        f"Subprocess failed (rc={proc.returncode}):\n"
        f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    )
    if result_file is not None:
        assert result_file.is_file(), proc.stderr
        return json.loads(result_file.read_text(encoding="utf-8"))
    lines = [line for line in proc.stdout.splitlines() if line.strip()]
    return json.loads(lines[-1])


class LoopbackLlmServer:
    """Real loopback OpenAI-compatible HTTP server (outermost HTTP boundary only)."""

    def __init__(
        self,
        responder: Callable[[dict[str, Any], int], Any],
        *,
        before_response_body: Callable[[int], None] | None = None,
    ) -> None:
        self.responder = responder
        self.requests: list[dict[str, Any]] = []
        self.lock = threading.Lock()
        server_ref = self

        class _Handler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: object) -> None:  # noqa: A002
                return

            def do_GET(self) -> None:
                self.send_payload(200, {"data": [{"id": "vm-extract"}, {"id": "vm-flash"}]})

            def do_POST(self) -> None:
                length = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(length) if length > 0 else b"{}"
                payload = json.loads(raw.decode("utf-8"))
                with server_ref.lock:
                    server_ref.requests.append(payload)
                    call_index = len(server_ref.requests)
                if "messages" not in payload:
                    self.send_payload(200, {"name": "vm-extract"})
                    return
                outcome = server_ref.responder(payload, call_index)
                if (
                    isinstance(outcome, tuple)
                    and len(outcome) == 2
                    and isinstance(outcome[0], int)
                ):
                    status_code, result_obj = outcome
                else:
                    status_code, result_obj = 200, outcome
                if status_code != 200:
                    err_obj = (
                        result_obj
                        if isinstance(result_obj, dict) and "error" in result_obj
                        else {"error": {"message": "injected failure"}}
                    )
                    self.send_payload(status_code, err_obj)
                    return
                content = (
                    result_obj
                    if isinstance(result_obj, str)
                    else json.dumps(result_obj, ensure_ascii=False)
                )
                if payload.get("stream"):
                    chunk1 = json.dumps(
                        {
                            "id": f"fixture-completion-{call_index}",
                            "object": "chat.completion.chunk",
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {"role": "assistant", "content": content},
                                    "finish_reason": None,
                                }
                            ],
                        },
                        ensure_ascii=False,
                    )
                    chunk2 = json.dumps(
                        {
                            "id": f"fixture-completion-{call_index}",
                            "object": "chat.completion.chunk",
                            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                            "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
                        },
                        ensure_ascii=False,
                    )
                    sse = f"data: {chunk1}\n\ndata: {chunk2}\n\ndata: [DONE]\n\n".encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Content-Length", str(len(sse)))
                    self.end_headers()
                    if before_response_body is not None:
                        before_response_body(call_index)
                    try:
                        self.wfile.write(sse)
                    except OSError:
                        return
                    return
                self.send_payload(
                    200,
                    {
                        "id": f"fixture-completion-{call_index}",
                        "object": "chat.completion",
                        "model": payload.get("model", "vm-extract"),
                        "created": 1,
                        "choices": [
                            {
                                "index": 0,
                                "finish_reason": "stop",
                                "message": {"role": "assistant", "content": content},
                            }
                        ],
                        "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
                    },
                    call_index=call_index,
                )

            def send_payload(
                self, status_code: int, body_obj: dict[str, Any], *, call_index: int | None = None,
            ) -> None:
                body = json.dumps(body_obj, ensure_ascii=False).encode("utf-8")
                self.send_response(status_code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                if before_response_body is not None and call_index is not None:
                    before_response_body(call_index)
                try:
                    self.wfile.write(body)
                except OSError:
                    return

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        self.server.daemon_threads = True
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    @property
    def base_url(self) -> str:
        host, port = self.server.server_address[:2]
        return f"http://{host}:{port}/v1"

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5.0)

    def __enter__(self) -> LoopbackLlmServer:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


@contextmanager
def http_json_llm_server(
    responder: Callable[[dict[str, Any]], dict[str, Any] | tuple[int, dict[str, Any]]],
    *,
    chat_id: str = "chatcmpl-test",
) -> Iterator[str]:
    """Run an outermost OpenAI-compatible loopback HTTP server for real PluginLlm calls."""
    del chat_id
    with LoopbackLlmServer(lambda payload, _idx: responder(payload)) as server:
        yield server.base_url


def install_plugin_copy(
    home: Path,
    vivemind_root: Path = REPO_ROOT,
    *,
    plugin_name: str = "sanae_mem",
) -> Path:
    """Copy a plugin tree into *home*/plugins/*plugin_name* without test/cache artifacts."""
    plugins_dir = home / "plugins"
    plugins_dir.mkdir(parents=True, exist_ok=True)
    target = plugins_dir / plugin_name
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(
        vivemind_root / "plugins" / plugin_name,
        target,
        ignore=shutil.ignore_patterns("__pycache__", "tests", "*.pyc"),
    )
    return target


def write_home_config(home: Path, cfg: Mapping[str, Any]) -> Path:
    """Write *cfg* to *home*/config.yaml, creating *home* if needed."""
    home.mkdir(parents=True, exist_ok=True)
    config_path = home / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(dict(cfg), allow_unicode=True),
        encoding="utf-8",
    )
    return config_path
