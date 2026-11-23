"""Permanent AST architectural layering and bare-``_conn`` regression gates."""

from __future__ import annotations

import ast
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

PACKAGE_ROOT = Path(__file__).resolve().parents[2] / "plugins" / "sanae_mem"
PACKAGE_NAME = "sanae_mem"
MIXIN_LIMIT = 0
PROVIDER_MAX_LINES = 667
PROVIDER_MAX_METHODS = 80

LAYER_BY_SUBPACKAGE: dict[str, int] = {
    "infra": 0,
    "storage": 1,
    "governance": 2,
    "domains": 3,
    "cognitive": 3,
    "runtime": 4,
}
TOP_LEVEL_LAYER = 5

CONN_ALLOWLIST: frozenset[str] = frozenset({
    "runtime/provider.py",
    "storage/sqlite_facade.py",
})

BOUNDARY_MODULES: tuple[str, ...] = (
    "integrations.py",
    "runtime/private_delivery_settler.py",
    "runtime/group_ingest.py",
    "governance/source_intake.py",
)

_PROBE_CALL_NAMES: frozenset[str] = frozenset({"getattr", "hasattr", "setattr", "delattr"})

# Exact pre-existing public host/duck-type probes in boundary modules; any new or private probe fails.
_BOUNDARY_PUBLIC_PROBE_ALLOWLIST: dict[str, frozenset[str]] = {
    "integrations.py": frozenset({"plugin_id", "current_cron_execution", "has_plugin"}),
    "runtime/private_delivery_settler.py": frozenset(),
    "runtime/group_ingest.py": frozenset({"current_turn_envelope", "sender_name"}),
    "governance/source_intake.py": frozenset({
        "is_delegated_child",
        "restricted_audience",
        "is_test_isolated",
        "verified",
        "cancelled",
        "bound_owner",
        "to_dict",
        "source_plugin",
    }),
}


@dataclass(frozen=True)
class ImportEdge:
    src: str
    dst: str
    lineno: int


def _iter_source_files() -> list[Path]:
    files: list[Path] = []
    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        rel_parts = path.relative_to(PACKAGE_ROOT).parts
        if "tests" in rel_parts or "__pycache__" in rel_parts:
            continue
        files.append(path)
    return files


def _module_name_for_path(path: Path) -> str:
    rel = path.relative_to(PACKAGE_ROOT)
    if rel.name == "__init__.py":
        if len(rel.parts) == 1:
            return PACKAGE_NAME
        return f"{PACKAGE_NAME}.{'.'.join(rel.parts[:-1])}"
    return f"{PACKAGE_NAME}.{'.'.join(rel.with_suffix('').parts)}"


def _module_layer(module_name: str) -> int:
    parts = module_name.split(".")
    if len(parts) >= 2 and parts[1] in LAYER_BY_SUBPACKAGE:
        return LAYER_BY_SUBPACKAGE[parts[1]]
    return TOP_LEVEL_LAYER


def _is_type_checking_guard(test: ast.expr) -> bool:
    if isinstance(test, ast.Name):
        return test.id == "TYPE_CHECKING"
    if isinstance(test, ast.Attribute):
        return test.attr == "TYPE_CHECKING"
    return False


class _RuntimeImportVisitor(ast.NodeVisitor):
    def __init__(self, module_name: str, is_init: bool, known_modules: set[str]) -> None:
        self.module_name = module_name
        self.is_init = is_init
        self.known_modules = known_modules
        self.edges: list[ImportEdge] = []

    def visit_If(self, node: ast.If) -> None:
        if _is_type_checking_guard(node.test):
            for stmt in node.orelse:
                self.visit(stmt)
            return
        self.generic_visit(node)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            name = alias.name
            if name == PACKAGE_NAME or name.startswith(f"{PACKAGE_NAME}."):
                if name in self.known_modules:
                    self.edges.append(ImportEdge(self.module_name, name, node.lineno))

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.level > 0:
            parts = self.module_name.split(".")
            pkg_parts = parts if self.is_init else parts[:-1]
            up = node.level - 1
            if up > len(pkg_parts):
                return
            base_parts = pkg_parts[: len(pkg_parts) - up]
            if node.module:
                base_parts = [*base_parts, *node.module.split(".")]
            base = ".".join(base_parts)
        else:
            base = node.module or ""

        if base != PACKAGE_NAME and not base.startswith(f"{PACKAGE_NAME}."):
            return

        for alias in node.names:
            candidate = f"{base}.{alias.name}"
            if candidate in self.known_modules:
                self.edges.append(ImportEdge(self.module_name, candidate, node.lineno))
            elif base in self.known_modules:
                self.edges.append(ImportEdge(self.module_name, base, node.lineno))


def _collect_import_graph() -> tuple[set[str], list[ImportEdge]]:
    files = _iter_source_files()
    path_by_module = {_module_name_for_path(path): path for path in files}
    known_modules = set(path_by_module)
    edges: list[ImportEdge] = []
    for module_name, path in sorted(path_by_module.items()):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        visitor = _RuntimeImportVisitor(
            module_name=module_name,
            is_init=path.name == "__init__.py",
            known_modules=known_modules,
        )
        visitor.visit(tree)
        edges.extend(visitor.edges)
    return known_modules, edges


def _find_sccs(nodes: set[str], edges: list[ImportEdge]) -> list[list[str]]:
    adj: dict[str, set[str]] = defaultdict(set)
    for edge in edges:
        adj[edge.src].add(edge.dst)

    index = 0
    indices: dict[str, int] = {}
    lowlink: dict[str, int] = {}
    stack: list[str] = []
    on_stack: set[str] = set()
    cycles: list[list[str]] = []

    def strongconnect(node: str) -> None:
        nonlocal index
        indices[node] = index
        lowlink[node] = index
        index += 1
        stack.append(node)
        on_stack.add(node)

        for nxt in sorted(adj.get(node, ())):
            if nxt not in indices:
                strongconnect(nxt)
                lowlink[node] = min(lowlink[node], lowlink[nxt])
            elif nxt in on_stack:
                lowlink[node] = min(lowlink[node], indices[nxt])

        if lowlink[node] == indices[node]:
            component: list[str] = []
            while True:
                w = stack.pop()
                on_stack.remove(w)
                component.append(w)
                if w == node:
                    break
            if len(component) > 1 or (len(component) == 1 and component[0] in adj.get(component[0], ())):
                cycles.append(sorted(component))

    for node in sorted(nodes):
        if node not in indices:
            strongconnect(node)
    return cycles


def find_upward_dependencies() -> list[tuple[str, int, str, int, int]]:
    """Return all runtime imports where ``src_layer < dst_layer``."""
    _, edges = _collect_import_graph()
    violations: list[tuple[str, int, str, int, int]] = []
    for edge in edges:
        src_layer = _module_layer(edge.src)
        dst_layer = _module_layer(edge.dst)
        if src_layer < dst_layer:
            violations.append((edge.src, src_layer, edge.dst, dst_layer, edge.lineno))
    return sorted(violations)


def test_zero_runtime_circular_dependencies() -> None:
    nodes, edges = _collect_import_graph()
    cycles = _find_sccs(nodes, edges)
    assert cycles == [], f"Runtime circular dependencies detected: {cycles}"


def test_subpackage_init_files_have_no_runtime_imports() -> None:
    _, edges = _collect_import_graph()
    subpackage_inits = {f"{PACKAGE_NAME}.{name}" for name in LAYER_BY_SUBPACKAGE}
    init_edges = [edge for edge in edges if edge.src in subpackage_inits]
    assert init_edges == [], f"Subpackage __init__.py must stay empty: {init_edges}"

    for subpkg in sorted(LAYER_BY_SUBPACKAGE):
        init_path = PACKAGE_ROOT / subpkg / "__init__.py"
        assert init_path.is_file(), f"{init_path} must exist"
        assert init_path.stat().st_size == 0, f"{init_path} must be 0 bytes"


def test_zero_bare_conn_attribute_access() -> None:
    violations: list[tuple[str, int]] = []
    for path in _iter_source_files():
        rel_posix = path.relative_to(PACKAGE_ROOT).as_posix()
        if rel_posix in CONN_ALLOWLIST:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr == "_conn":
                violations.append((rel_posix, node.lineno))
    assert violations == [], f"Bare ._conn access forbidden outside allowlist: {violations}"


def test_zero_upward_dependencies() -> None:
    violations = find_upward_dependencies()
    assert violations == [], f"Upward (reverse) dependencies detected: {violations}"


def _package_mixin_definitions() -> list[tuple[str, str]]:
    definitions: list[tuple[str, str]] = []
    for path in _iter_source_files():
        module = _module_name_for_path(path)
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name.endswith("Mixin"):
                definitions.append((module, node.name))
    return sorted(definitions)


def test_backward_compatible_public_exports() -> None:
    import sys

    plugins_dir = str(PACKAGE_ROOT.parent)
    if plugins_dir not in sys.path:
        sys.path.insert(0, plugins_dir)

    from sanae_mem import _runtime_registry, plugin_entry, provider

    from agent.memory_provider import MemoryProvider

    assert provider.SanaeMemProvider is plugin_entry.SanaeMemProvider
    assert provider.SanaeMemProvider.__bases__ == (MemoryProvider,)
    for attr in (
        "owns_delivery",
        "register_active_provider",
        "register_delivery_owner",
        "register_explicit_source_sink",
        "register_source_intake_sink",
        "resolve_active_provider",
        "resolve_source_intake_sink",
        "unregister_explicit_source_sink",
    ):
        assert callable(getattr(_runtime_registry, attr))

    mro_mixins = sorted(
        (base.__module__, base.__name__)
        for base in provider.SanaeMemProvider.__mro__
        if (base.__module__ == PACKAGE_NAME or base.__module__.startswith(f"{PACKAGE_NAME}."))
        and base.__name__.endswith("Mixin")
    )
    defined_mixins = _package_mixin_definitions()
    assert len(mro_mixins) <= MIXIN_LIMIT, f"Provider MRO exceeds the Mixin limit: {mro_mixins}"
    assert len(defined_mixins) <= MIXIN_LIMIT, f"Package exceeds the Mixin limit: {defined_mixins}"
    assert mro_mixins == defined_mixins, "Provider MRO and package Mixin definitions differ"


def test_provider_size_and_method_count_stay_within_bounds() -> None:
    provider_path = PACKAGE_ROOT / "runtime" / "provider.py"
    source = provider_path.read_text(encoding="utf-8")
    line_count = len(source.splitlines())
    tree = ast.parse(source, filename=str(provider_path))
    provider_classes = [
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "SanaeMemProvider"
    ]
    assert len(provider_classes) == 1
    method_count = sum(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        for node in provider_classes[0].body
    )
    assert line_count <= PROVIDER_MAX_LINES, (
        f"runtime/provider.py grew to {line_count} lines (limit {PROVIDER_MAX_LINES})"
    )
    assert method_count <= PROVIDER_MAX_METHODS, (
        f"SanaeMemProvider grew to {method_count} methods (limit {PROVIDER_MAX_METHODS})"
    )


def _is_provider_or_worker_expr(expr: ast.expr) -> bool:
    if isinstance(expr, ast.Name):
        return expr.id in {"provider", "worker"}
    if isinstance(expr, ast.Attribute):
        return expr.attr in {"provider", "_provider", "worker", "_worker"}
    return False


def _probe_func_name(func: ast.expr) -> str | None:
    if isinstance(func, ast.Name) and func.id in _PROBE_CALL_NAMES:
        return func.id
    if isinstance(func, ast.Attribute) and func.attr in _PROBE_CALL_NAMES:
        return func.attr
    return None


_NESTED_SCOPE_NODES = (
    ast.FunctionDef,
    ast.AsyncFunctionDef,
    ast.ClassDef,
    ast.Lambda,
    ast.ListComp,
    ast.SetComp,
    ast.DictComp,
    ast.GeneratorExp,
)


def _walk_same_scope(fn_node: ast.AST) -> Iterator[ast.AST]:
    """Yield AST nodes inside *fn_node*'s own scope without entering nested scopes."""
    body = getattr(fn_node, "body", ())
    stack: list[ast.AST] = list(reversed(body))
    while stack:
        current = stack.pop()
        yield current
        if isinstance(current, _NESTED_SCOPE_NODES):
            continue
        for child in reversed(list(ast.iter_child_nodes(current))):
            stack.append(child)


def _is_immutable_for_session_client_handle(handle_fn: ast.FunctionDef) -> bool:
    """Return True iff `handle` binds `client` once to `for_session(...)` with zero shadowing or rebinding."""
    direct_bindings: list[tuple[ast.Name, ast.expr]] = []
    for node in _walk_same_scope(handle_fn):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "client":
                    direct_bindings.append((target, node.value))

    if len(direct_bindings) != 1:
        return False
    sole_target, rhs = direct_bindings[0]
    if not (
        isinstance(rhs, ast.Call)
        and isinstance(rhs.func, ast.Name)
        and rhs.func.id == "for_session"
    ):
        return False

    for node in ast.walk(handle_fn):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name == "client":
            return False
        if isinstance(node, ast.arg) and node.arg == "client":
            return False
        if isinstance(node, (ast.MatchAs, ast.MatchStar)) and node.name == "client":
            return False
        if isinstance(node, ast.MatchMapping) and node.rest == "client":
            return False
        if isinstance(node, ast.ExceptHandler) and node.name == "client":
            return False
        if isinstance(node, (ast.Global, ast.Nonlocal)) and "client" in node.names:
            return False
        if isinstance(node, ast.alias) and (node.asname or node.name.split(".", 1)[0]) == "client":
            return False
        if isinstance(node, ast.Name) and node.id == "client" and isinstance(node.ctx, (ast.Store, ast.Del)):
            if node is not sole_target:
                return False
    return True


def _allowed_bilibili_client_attr_nodes(tree: ast.Module) -> set[int]:
    """Return `id(ast.Attribute)` nodes for read-only `client._owner`/`_provider` in `register_bilibili_memory.handle`."""
    allowed_ids: set[int] = set()
    for top in tree.body:
        if not isinstance(top, ast.FunctionDef) or top.name != "register_bilibili_memory":
            continue
        for inner in top.body:
            if not isinstance(inner, ast.FunctionDef) or inner.name != "handle":
                continue
            if not _is_immutable_for_session_client_handle(inner):
                continue
            for node in _walk_same_scope(inner):
                if (
                    isinstance(node, ast.Attribute)
                    and isinstance(node.ctx, ast.Load)
                    and isinstance(node.value, ast.Name)
                    and node.value.id == "client"
                    and node.attr in {"_owner", "_provider"}
                ):
                    allowed_ids.add(id(node))
    return allowed_ids


def _allowed_source_intake_material_getattr_nodes(tree: ast.Module) -> set[int]:
    """Allow `getattr(material, field_name, None)` only inside `ingest_source_material`'s literal field tuple loop."""
    allowed_ids: set[int] = set()
    for top in tree.body:
        if not isinstance(top, ast.FunctionDef) or top.name != "ingest_source_material":
            continue
        for node in _walk_same_scope(top):
            if not isinstance(node, ast.DictComp):
                continue
            if len(node.generators) != 1:
                continue
            gen = node.generators[0]
            if not (
                isinstance(gen.target, ast.Name)
                and gen.target.id == "field_name"
                and isinstance(gen.iter, ast.Tuple)
                and gen.iter.elts
                and all(
                    isinstance(elt, ast.Constant)
                    and isinstance(elt.value, str)
                    and not elt.value.startswith("_")
                    for elt in gen.iter.elts
                )
            ):
                continue
            call = node.value
            if (
                isinstance(call, ast.Call)
                and isinstance(call.func, ast.Name)
                and call.func.id == "getattr"
                and len(call.args) >= 2
                and isinstance(call.args[0], ast.Name)
                and call.args[0].id == "material"
                and isinstance(call.args[1], ast.Name)
                and call.args[1].id == "field_name"
            ):
                allowed_ids.add(id(call))
    return allowed_ids


def find_boundary_private_access_violations(
    package_root: Path = PACKAGE_ROOT,
) -> list[tuple[str, int, str]]:
    """Return non-self private attribute accesses and disallowed getattr/hasattr probes in boundary modules."""
    violations: list[tuple[str, int, str]] = []
    for rel_posix in BOUNDARY_MODULES:
        path = package_root / rel_posix
        assert path.is_file(), f"Expected boundary module {path} to exist"
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        allowed_public_probes = _BOUNDARY_PUBLIC_PROBE_ALLOWLIST.get(rel_posix, frozenset())
        allowed_client_attr_ids = (
            _allowed_bilibili_client_attr_nodes(tree) if rel_posix == "integrations.py" else set()
        )
        allowed_material_calls = (
            _allowed_source_intake_material_getattr_nodes(tree)
            if rel_posix == "governance/source_intake.py"
            else set()
        )
        allowed_probe_callee_ids: set[int] = set()

        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute):
                if node.attr.startswith("_") and not node.attr.startswith("__"):
                    if isinstance(node.value, ast.Name) and node.value.id == "self":
                        continue
                    if id(node) in allowed_client_attr_ids:
                        continue
                    violations.append((rel_posix, node.lineno, f"private attr: {ast.unparse(node)}"))
            elif isinstance(node, ast.ImportFrom) and node.module == "builtins":
                for alias in node.names:
                    if alias.name in _PROBE_CALL_NAMES:
                        violations.append((rel_posix, node.lineno, f"probe import: {ast.unparse(node)}"))
            elif isinstance(node, ast.Call):
                func_name = _probe_func_name(node.func)
                if func_name is None:
                    continue
                if not isinstance(node.func, ast.Name):
                    violations.append((rel_posix, node.lineno, f"disallowed probe: {ast.unparse(node)}"))
                    continue
                if len(node.args) < 2:
                    violations.append((rel_posix, node.lineno, f"malformed probe: {ast.unparse(node)}"))
                    continue
                target_expr, attr_expr = node.args[0], node.args[1]
                if func_name in {"setattr", "delattr"} or _is_provider_or_worker_expr(target_expr):
                    violations.append((rel_posix, node.lineno, f"disallowed probe: {ast.unparse(node)}"))
                    continue
                if isinstance(attr_expr, ast.Constant) and isinstance(attr_expr.value, str):
                    attr_name = attr_expr.value
                    if attr_name.startswith("_") or attr_name not in allowed_public_probes:
                        violations.append((rel_posix, node.lineno, f"disallowed probe: {ast.unparse(node)}"))
                    else:
                        allowed_probe_callee_ids.add(id(node.func))
                    continue
                if id(node) in allowed_material_calls:
                    allowed_probe_callee_ids.add(id(node.func))
                    continue
                violations.append((rel_posix, node.lineno, f"dynamic probe: {ast.unparse(node)}"))

        # Catch non-call aliases/references such as `g: object = builtins.getattr` or `g = getattr`.
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id in _PROBE_CALL_NAMES and id(node) not in allowed_probe_callee_ids:
                violations.append((rel_posix, node.lineno, f"probe reference: {ast.unparse(node)}"))
            elif isinstance(node, ast.Attribute) and node.attr in _PROBE_CALL_NAMES:
                violations.append((rel_posix, node.lineno, f"probe reference: {ast.unparse(node)}"))
    return sorted(violations)


def test_boundary_modules_forbid_external_private_access_and_probes() -> None:
    violations = find_boundary_private_access_violations()
    assert violations == [], f"Boundary private access / probe violations detected: {violations}"


def test_boundary_ast_gate_rejects_builtins_probes_and_fake_client_rebind(tmp_path: Path) -> None:
    pkg_root = tmp_path / "pkg"
    for rel_posix in BOUNDARY_MODULES:
        target = pkg_root / rel_posix
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("", encoding="utf-8")

    (pkg_root / "integrations.py").write_text(
        "\n".join([
            "import builtins",
            "from builtins import getattr as _g",
            "def register_bilibili_memory(ctx):",
            "    def handle(args):",
            "        client = for_session('s')",
            "        client, tag = args, 1",
            "        def inner(client):",
            "            return client._provider",
            "        return (client._provider, inner(args), tag)",
            "    g_ann: object = builtins.getattr",
            "    h_ann: object = builtins.hasattr",
            "    x = builtins.getattr(ctx, '_secret', None)",
            "    y = builtins.hasattr(ctx, '_flag')",
            "    z = _g(ctx, '_secret2', None)",
            "    return (g_ann, h_ann, x, y, z, handle)",
        ]),
        encoding="utf-8",
    )
    (pkg_root / "runtime/private_delivery_settler.py").write_text(
        "import builtins\ndef g(p):\n    fn: object = builtins.setattr\n    return (p._conn, fn)\n",
        encoding="utf-8",
    )
    (pkg_root / "runtime/group_ingest.py").write_text(
        "import builtins\ndef h(x):\n    fn: object = builtins.delattr\n    return (getattr(x, 'new_public_probe', None), fn)\n",
        encoding="utf-8",
    )
    (pkg_root / "governance/source_intake.py").write_text(
        "\n".join([
            "def k(provider):",
            "    return getattr(provider, 'bound_owner', None)",
            "def rogue(material):",
            "    field_name = '_secret'",
            "    return getattr(material, field_name, None)",
        ]),
        encoding="utf-8",
    )

    violations = find_boundary_private_access_violations(package_root=pkg_root)
    summary = "\n".join(f"{rel}:{line}:{msg}" for rel, line, msg in violations)
    assert "integrations.py:2:probe import:" in summary
    assert "integrations.py:8:private attr: client._provider" in summary
    assert "integrations.py:9:private attr: client._provider" in summary
    assert "integrations.py:10:probe reference: builtins.getattr" in summary
    assert "integrations.py:11:probe reference: builtins.hasattr" in summary
    assert "integrations.py:12:disallowed probe: builtins.getattr" in summary
    assert "integrations.py:13:disallowed probe: builtins.hasattr" in summary
    assert "runtime/private_delivery_settler.py:3:probe reference: builtins.setattr" in summary
    assert "runtime/private_delivery_settler.py:4:private attr: p._conn" in summary
    assert "runtime/group_ingest.py:3:probe reference: builtins.delattr" in summary
    assert "runtime/group_ingest.py:4:disallowed probe: getattr" in summary
    assert "governance/source_intake.py:2:disallowed probe: getattr" in summary
    assert "governance/source_intake.py:5:dynamic probe: getattr(material, field_name, None)" in summary

    # Isolated checks inside an otherwise-valid `register_bilibili_memory.handle`:
    # attribute writes and nested-scope shadowing (lambda / comprehension / match) must all fail.
    for snippet in (
        "        client._provider = args\n        return client._provider",
        "        return (lambda client: client._provider)(args)",
        "        return [client._provider for client in args]",
        "        match args:\n            case {'client': client}:\n                return client._provider",
        "        def client():\n            pass\n        return client._provider",
        "        async def client():\n            pass\n        return client._provider",
        "        class client:\n            pass\n        return client._provider",
    ):
        probe_root = tmp_path / f"case_{abs(hash(snippet))}"
        for rel_posix in BOUNDARY_MODULES:
            target = probe_root / rel_posix
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("", encoding="utf-8")
        (probe_root / "integrations.py").write_text(
            "def register_bilibili_memory(ctx):\n"
            "    def handle(args):\n"
            "        client = for_session('s')\n"
            f"{snippet}\n",
            encoding="utf-8",
        )
        case_violations = find_boundary_private_access_violations(package_root=probe_root)
        assert any(
            rel == "integrations.py" and "private attr: client._provider" in msg
            for rel, _line, msg in case_violations
        ), (snippet, case_violations)
