from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
if os.environ.get("VIVEMIND_HERMES_SOURCE"):
    HERMES_ROOT = Path(os.environ["VIVEMIND_HERMES_SOURCE"]).expanduser().resolve()
else:
    _hermes_spec = importlib.util.find_spec("hermes_constants")
    if _hermes_spec is not None and _hermes_spec.origin:
        HERMES_ROOT = Path(_hermes_spec.origin).resolve().parent
    else:
        HERMES_ROOT = next(
            (path for path in (
                REPO_ROOT.parent / "hermes-minimal-host",
                REPO_ROOT.parent / "hermes-minimal-host-20261112",
            ) if path.is_dir()),
            Path(tempfile.gettempdir()) / "vivemind-hermes-6093627",
        )
if not HERMES_ROOT.is_dir():
    pytest.skip("pinned Hermes source is unavailable", allow_module_level=True)
sys.path.insert(0, str(HERMES_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "plugins"))

import hermes_host_contracts as hc
from sanae_mem.governance.cognitive_context import (
    DEFAULT_PUBLIC_ALLOWED_TOOLS,
    ViveMindCognitivePromptPolicy,
    build_public_social_turn_policy,
)
from sanae_mem.governance.memory_guard import ViveMindMemoryMutationGate, guard_builtin_memory_prompt


def _bundle():
    return {
        "identity_parts": ["SOUL"],
        "provider_memory_parts": ["BEHAVIOR"],
        "help_guidance": "TOOLS",
        "guidance_parts": [
            "# Finishing the job\nBuild working artifacts.",
            "You have persistent memory, carried across sessions.",
            "## Mid-turn user steering\n[OUT-OF-BAND USER MESSAGE]",
            "# Execution discipline\n<mandatory_tool_use>run date</mandatory_tool_use>",
        ],
        "skills_prompt": "SKILLS",
        "context_files_parts": ["ENGINEERING"],
        "builtin_memory_parts": ["MEMORY"],
        "user_profile_parts": ["USER"],
        "plugin_sections_after_memory": ["PLUGIN_CONTEXT"],
        "runtime_environment_block": "RUNTIME",
        "timestamp_line": "TIME",
    }


def _envelope(
    *,
    platform: str = "telegram",
    chat_type: str,
    sender_id: str,
    restricted_audience: str = "",
    is_delegated_child: bool = False,
):
    return hc.TurnEnvelope(
        platform=platform,
        chat_id="chat-1" if chat_type else "",
        chat_type=chat_type,
        sender_id=sender_id,
        session_id="session-1",
        restricted_audience=restricted_audience,
        is_delegated_child=is_delegated_child,
    )


def test_prompt_tiers_keep_behavior_on_every_surface_and_gate_profile_files():
    config = {"memory_identity": {"platform": "telegram", "user_id": "maintainer"}}
    policy = ViveMindCognitivePromptPolicy(lambda: config)
    agent = SimpleNamespace(platform="telegram")

    maintainer = policy.assemble_system_prompt_parts(
        agent, _bundle(), "PERSONA", _envelope(chat_type="dm", sender_id="maintainer")
    )
    assert maintainer is not None
    assert "TOOLS" not in maintainer["stable"]
    assert "Mid-turn user steering" not in maintainer["stable"]
    assert "Finishing the job" not in maintainer["stable"]
    assert "Execution discipline" not in maintainer["stable"]
    assert maintainer["stable"].index("You have persistent memory") < maintainer["stable"].index("SKILLS")
    assert "SOUL" not in maintainer["stable"] and "BEHAVIOR" not in maintainer["stable"]
    assert [maintainer["context"].index(marker) for marker in
            ("ENGINEERING", "MEMORY", "USER", "PLUGIN_CONTEXT", "RUNTIME", "TIME")] == sorted(
                maintainer["context"].index(marker) for marker in
                ("ENGINEERING", "MEMORY", "USER", "PLUGIN_CONTEXT", "RUNTIME", "TIME")
            )
    assert maintainer["volatile"] == "BEHAVIOR\n\nSOUL\n\nPERSONA"

    external = policy.assemble_system_prompt_parts(
        agent, _bundle(), "PERSONA", _envelope(chat_type="dm", sender_id="external")
    )
    assert external is not None
    assert "TOOLS" not in external["stable"]
    assert "Mid-turn user steering" not in external["stable"]
    assert "You have persistent memory" in external["stable"]
    assert "USER" in external["context"]
    assert "ENGINEERING" not in external["context"] and "MEMORY" not in external["context"]
    assert external["volatile"] == "BEHAVIOR\n\nSOUL\n\nPERSONA"

    delegated = policy.assemble_system_prompt_parts(
        agent,
        _bundle(),
        "PERSONA",
        _envelope(chat_type="dm", sender_id="maintainer", is_delegated_child=True),
    )
    assert delegated is not None
    assert delegated["stable"].index("TOOLS") < delegated["stable"].index("SKILLS")
    assert "Mid-turn user steering" in delegated["stable"]
    assert "Execution discipline" in delegated["stable"]

    cli_parts = policy.assemble_system_prompt_parts(
        SimpleNamespace(platform="cli"),
        _bundle(),
        "PERSONA",
        _envelope(platform="cli", chat_type="", sender_id=""),
    )
    assert cli_parts is not None
    assert cli_parts["stable"].index("TOOLS") < cli_parts["stable"].index("SKILLS")
    assert "Mid-turn user steering" in cli_parts["stable"]

    public = policy.assemble_system_prompt_parts(
        agent, _bundle(), "PERSONA", _envelope(chat_type="group", sender_id="external")
    )
    assert public is not None
    assert all(marker not in public["context"] for marker in ("ENGINEERING", "MEMORY", "USER"))
    assert public["volatile"] == "BEHAVIOR\n\nSOUL\n\nPERSONA"


def test_qq_main_dm_skills_index_filters_engineering_office_and_keeps_child_and_cli_full():
    config = {"memory_identity": {"platform": "qqbot", "user_id": "maintainer"}}
    policy = ViveMindCognitivePromptPolicy(lambda: config)
    skills_by_category = {
        "software-development": [
            ("systematic-debugging", "4-phase root cause debugging."),
            ("delegation-briefs", "给子 agent 派活时编写内联背景与验收标准的任务书模板。"),
        ],
        "autonomous-ai-agents": [
            ("hermes-agent", "Use, configure, theme, extend, and orchestrate Hermes Agent."),
        ],
        "apple": [("apple-notes", "Manage Apple Notes via memo CLI.")],
        "email": [("himalaya", "Himalaya CLI: IMAP/SMTP email.")],
        "note-taking": [("obsidian", "Read and edit Obsidian vault.")],
        "creative": [
            ("claude-design", "Design one-off HTML artifacts."),
            ("baoyu-infographic", "Infographics: 21 layouts x 21 styles (信息图, 可视化)."),
        ],
        "devops": [
            ("goat-usage-query", "查询 GOAT / Command Code 与 Kimi Coding 的额度用量。"),
            ("hermes-cron-ops", "使用 hermes cron 命令行增删改查定时任务与排查状态。"),
        ],
        "hermes-ops-docs": [
            ("hermes-ops-docs", "云服务器运维总手册、网关重启与定时任务排障参考。"),
        ],
        "media": [
            ("bilibili-video-analysis", "维护者发来 B 站视频链接时，抽帧看画面并结合弹幕分析内容。"),
            ("voice-cloning", "用 Qwen3-TTS 合成我的语音并发给维护者。"),
            ("gif-search", "Search/download GIFs from Tenor via curl + jq."),
        ],
        "productivity": [
            ("pdf", "PDF files: create, read, merge, fill, OCR, edit text."),
            ("xlsx", "Create, read, edit Excel .xlsx workbooks and CSVs."),
            ("remote-tech-support", "隔着聊天指导维护者排查家里路由器、电视盒子与网络故障。"),
            ("session-history-recall", "按维护者指定的时间段从 state.db 翻阅历史聊天记录以接续上下文。"),
        ],
        "research": [
            ("arxiv", "Search arXiv papers by keyword."),
            ("fandom-stats-lookup", "查询东方人气投票名次与 Pixiv 标签同人投稿数量统计。"),
        ],
        "social-media": [
            ("sanae-companion", "查询东方Project原作设定、同人曲目详情或向 QQ 发送文件与语音时查阅。"),
            ("bilibili-account-ops", "运营我的 B 站账号：刷首页视频、评论私信、发动态与助手分工。"),
            ("xurl", "X/Twitter via xurl CLI."),
        ],
        "web": [
            ("ai-search-combo", "选择 Exa 搜索与 DeepSeek 抓取双通道查资料。"),
            ("meme-lookup", "维护者问我梗/表情包含义时，双搜查证后分通义与语境两层回答。"),
            ("x-twitter-reading", "维护者发来 X/Twitter 链接时，用 fxtwitter 取图并用 vision 看图。"),
            ("blocked-page-recovery", "Use when a fetch fails: 403/429."),
        ],
    }
    unfiltered_index_lines = [
        "  creative [names only]: baoyu-infographic, claude-design",
        "  software-development:",
        "    - systematic-debugging: 4-phase root cause debugging.",
        "  devops:",
        "    - goat-usage-query: 查询 GOAT / Command Code 与 Kimi Coding 的额度用量。",
    ]
    raw_hidden_note = (
        "\n(Categories marked [names only] are outside the current coding context.)"
    )

    qq_dm_rendered = policy.format_skills_index_prompt(
        "DEFAULT",
        skills_by_category,
        None,
        _envelope(platform="qqbot", chat_type="dm", sender_id="maintainer"),
        index_lines=unfiltered_index_lines,
        hidden_note=raw_hidden_note,
    )
    assert qq_dm_rendered is not None
    for forbidden in (
        "systematic-debugging",
        "delegation-briefs",
        "hermes-agent",
        "apple-notes",
        "himalaya",
        "obsidian",
        "claude-design",
        "baoyu-infographic",
        "gif-search",
        "pdf:",
        "xlsx:",
        "arxiv",
        "xurl",
        "blocked-page-recovery",
        "[names only]",
        "software-development:",
    ):
        assert forbidden not in qq_dm_rendered
    for required in (
        "goat-usage-query",
        "hermes-cron-ops",
        "hermes-ops-docs",
        "bilibili-video-analysis",
        "voice-cloning",
        "remote-tech-support",
        "session-history-recall",
        "fandom-stats-lookup",
        "sanae-companion",
        "bilibili-account-ops",
        "ai-search-combo",
        "meme-lookup",
        "x-twitter-reading",
    ):
        assert required in qq_dm_rendered

    delegated_rendered = policy.format_skills_index_prompt(
        "DEFAULT",
        skills_by_category,
        None,
        _envelope(platform="qqbot", chat_type="dm", sender_id="maintainer", is_delegated_child=True),
        index_lines=unfiltered_index_lines,
        hidden_note=raw_hidden_note,
    )
    assert delegated_rendered is not None
    assert "systematic-debugging" in delegated_rendered
    assert "creative [names only]" in delegated_rendered
    assert "[names only]" in delegated_rendered

    cli_rendered = policy.format_skills_index_prompt(
        "DEFAULT",
        skills_by_category,
        None,
        _envelope(platform="cli", chat_type="", sender_id=""),
        index_lines=unfiltered_index_lines,
        hidden_note=raw_hidden_note,
    )
    assert cli_rendered is not None
    assert "systematic-debugging" in cli_rendered
    assert "creative [names only]" in cli_rendered


def test_qq_main_dm_is_skill_allowed_blocks_engineering_and_allows_companion():
    config = {"memory_identity": {"platform": "qqbot", "user_id": "maintainer"}}
    policy = ViveMindCognitivePromptPolicy(lambda: config)
    qq_dm = _envelope(platform="qqbot", chat_type="dm", sender_id="maintainer")
    delegated_qq_dm = _envelope(
        platform="qqbot", chat_type="dm", sender_id="maintainer", is_delegated_child=True,
    )
    cli_env = _envelope(platform="cli", chat_type="", sender_id="")

    blocked_cases = (
        ("systematic-debugging", "software-development", "4-phase root cause debugging."),
        ("codex", "software-development", "使用 Codex CLI 完成编程任务。"),
        ("docx", "productivity", "Create, read, edit Word .docx documents."),
        ("claude-design", "creative", "Design one-off HTML artifacts."),
        ("apple-notes", "apple", "Manage Apple Notes via memo CLI."),
    )
    allowed_cases = (
        ("sanae-companion", "social-media", "查询东方Project原作设定、同人曲目详情或向 QQ 发送文件与语音时查阅。"),
        ("meme-lookup", "web", "维护者问我梗/表情包含义时，双搜查证后分通义与语境两层回答。"),
        ("session-history-recall", "productivity", "按维护者指定的时间段从 state.db 翻阅历史聊天记录以接续上下文。"),
    )

    for name, cat, desc in blocked_cases:
        assert policy.is_skill_allowed(name, cat, desc, qq_dm) is False
        assert policy.is_skill_allowed(name, cat, desc, delegated_qq_dm) is True
        assert policy.is_skill_allowed(name, cat, desc, cli_env) is True

    for name, cat, desc in allowed_cases:
        assert policy.is_skill_allowed(name, cat, desc, qq_dm) is True
        assert policy.is_skill_allowed(name, cat, desc, delegated_qq_dm) is True
        assert policy.is_skill_allowed(name, cat, desc, cli_env) is True

    # Also verify envelope=None resolves current_turn_envelope() from host context.
    with hc.turn_envelope_scope(qq_dm):
        assert policy.is_skill_allowed("systematic-debugging", "software-development", "4-phase root cause debugging.") is False
        assert policy.is_skill_allowed("sanae-companion", "social-media", "查询东方Project原作设定。") is True


def test_external_surface_tool_allowlist_and_prompt_policy_preserve_host_prompt(monkeypatch):
    config = {"memory_identity": {"platform": "telegram", "user_id": "maintainer"}}
    prompt_policy = ViveMindCognitivePromptPolicy(lambda: config)
    external_policy = prompt_policy.resolve_turn_policy(
        _envelope(chat_type="dm", sender_id="external")
    )
    assert external_policy.override_system_prompt is False
    assert external_policy.allowed_tools == DEFAULT_PUBLIC_ALLOWED_TOOLS
    public_policy = prompt_policy.resolve_turn_policy(
        _envelope(chat_type="group", sender_id="external")
    )
    assert public_policy.override_system_prompt is False
    assert public_policy.allowed_tools == DEFAULT_PUBLIC_ALLOWED_TOOLS

    social = build_public_social_turn_policy(
        config=config,
        extra_allowed_tools=("safe_read", "terminal", "browser_navigate"),
        action_tool_name="browser_navigate",
    )
    assert social.override_system_prompt is False
    assert "safe_read" in social.allowed_tools
    assert not {"terminal", "browser_navigate"}.intersection(social.allowed_tools)

    agent = SimpleNamespace(platform="telegram")
    with hc.restricted_turn_scope(
        agent, [], social, session_id="session-1",
    ):
        assembled = prompt_policy.assemble_system_prompt_parts(
            agent,
            _bundle(),
            "PERSONA",
            _envelope(chat_type="group", sender_id="external", restricted_audience="public_social"),
        )
    assert assembled is not None
    assert "公开社交场合" in assembled["context"]
    assert assembled["volatile"] == "BEHAVIOR\n\nSOUL\n\nPERSONA"

    monkeypatch.setattr("sanae_mem.governance.memory_guard._has_enabled_builtin_notes", lambda: True)
    hc.register_cognitive_prompt_policy(prompt_policy)
    try:
        assert guard_builtin_memory_prompt(session_id="external-session") is None
    finally:
        hc.unregister_cognitive_prompt_policy(prompt_policy)


def test_memory_mutation_gate_blocks_user_profile_and_stages_only_protected_skills():
    gate = ViveMindMemoryMutationGate()
    blocked_user = gate.evaluate_memory_mutation(
        action="add", target="user", content="A profile fact",
    )
    assert blocked_user.blocked is True

    # Without an auxiliary LLM, non-static skill writes fail closed uniformly.
    unavailable = gate.evaluate_skill_mutation(
        action="create", name="calculator", args={"content": "Arithmetic steps."},
    )
    assert unavailable is not None
    if isinstance(unavailable, dict):
        assert unavailable["blocked"] is True and unavailable["allow"] is False
        assert "暂不可用" in str(unavailable.get("message") or "")
    else:
        assert unavailable.blocked is True and unavailable.allow is False
        assert "暂不可用" in str(unavailable.message or "")

    protected = gate.evaluate_skill_mutation(
        action="create",
        name="persona-rules",
        args={"content": "---\nname: persona-rules\ndescription: helper\n---\nBody."},
    )
    assert protected.blocked is True
    assert protected.require_approval is False
    assert protected.allow is False
    assert "暂不可用" in str(protected.message or "")

    description_protected = gate.evaluate_skill_mutation(
        action="update",
        name="conversation",
        args={"content": "---\nname: conversation\ndescription: 记忆卫生\n---\nBody."},
    )
    assert description_protected is not None
    if isinstance(description_protected, dict):
        assert description_protected["blocked"] is True
        assert description_protected.get("require_approval", False) is False
        assert description_protected["allow"] is False
        assert "暂不可用" in str(description_protected.get("message") or "")
    else:
        assert description_protected.blocked is True
        assert description_protected.require_approval is False
        assert description_protected.allow is False
        assert "暂不可用" in str(description_protected.message or "")


def test_load_host_config_ruamel_fallback_and_unbound_boot_probe(tmp_path, monkeypatch):
    from sanae_mem.infra import host_bridge

    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(
        "plugins:\n  sanae_mem:\n    memory_identity:\n      platform: qqbot\n      user_id: owner123\n",
        encoding="utf-8",
    )
    monkeypatch.setitem(sys.modules, "yaml", None)
    loaded = host_bridge.load_host_config(tmp_path)
    assert loaded["plugins"]["sanae_mem"]["memory_identity"] == {
        "platform": "qqbot",
        "user_id": "owner123",
    }

    json_cfg_dir = tmp_path / "json-only"
    json_cfg_dir.mkdir()
    (json_cfg_dir / "config.yaml").write_text(
        '{"plugins": {"sanae_mem": {"partner_name": "json-partner"}}}',
        encoding="utf-8",
    )
    monkeypatch.setitem(sys.modules, "ruamel", None)
    monkeypatch.setitem(sys.modules, "ruamel.yaml", None)
    assert host_bridge.load_host_config(json_cfg_dir)["plugins"]["sanae_mem"]["partner_name"] == "json-partner"
    assert host_bridge.load_host_config(tmp_path) == {}

    policy = ViveMindCognitivePromptPolicy(lambda: loaded["plugins"]["sanae_mem"])
    assert policy.resolve_turn_policy(hc.TurnEnvelope()) is None
    engaged_empty = policy.resolve_turn_policy(hc.TurnEnvelope(session_context_engaged=True))
    assert engaged_empty is not None
    assert engaged_empty.allowed_tools == DEFAULT_PUBLIC_ALLOWED_TOOLS
    maintainer_dm = hc.TurnEnvelope(
        platform="qqbot",
        chat_id="owner123",
        chat_type="dm",
        sender_id="owner123",
        session_id="sess-owner",
        session_context_engaged=True,
        is_messaging_surface=True,
    )
    assert policy.resolve_turn_policy(maintainer_dm) is None