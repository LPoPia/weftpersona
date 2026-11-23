from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


PLUGIN_ROOT = Path(__file__).resolve().parents[2] / "plugins"
sys.path.insert(0, str(PLUGIN_ROOT))

from sanae_mem.governance import source_intake


def _provider(agent_context="primary"):
    def _ctx():
        scope = source_intake.current_chat_scope()
        is_owner = scope.visibility == "private" and scope.owner == "owner-A"
        return source_intake.SourceIntakeContext(
            session_id="",
            agent_context=agent_context,
            shadow=False,
            daily_owner_key="owner-A",
            daily_owner_platform="telegram",
            daily_profile_key="",
            private_chat_owner="owner-A" if is_owner else "",
            is_configured_private_owner=is_owner,
        )

    return SimpleNamespace(
        sqlite=SimpleNamespace(is_open=True),
        source_intake_context=_ctx,
    )


def test_configured_owner_fallback_requires_verified_maintainer_private_route(monkeypatch):
    provider = _provider()
    envelope = SimpleNamespace(
        is_delegated_child=False,
        restricted_audience="",
        is_test_isolated=False,
    )
    monkeypatch.setattr(source_intake, "current_turn_envelope", lambda: envelope)
    monkeypatch.setattr(source_intake, "current_chat_scope", lambda: SimpleNamespace(
        visibility="private", platform="telegram", owner="owner-A",
    ))
    assert source_intake._resolve_intake_owner(
        provider, allow_configured_owner_fallback=True
    ) == "owner-A"

    monkeypatch.setattr(source_intake, "current_chat_scope", lambda: SimpleNamespace(
        visibility="private", platform="telegram", owner="external-owner",
    ))
    assert source_intake._resolve_intake_owner(
        provider, allow_configured_owner_fallback=True
    ) == ""


@pytest.mark.parametrize(
    "scope,envelope",
    [
        (SimpleNamespace(visibility="public", platform="telegram", owner=""), SimpleNamespace(
            is_delegated_child=False, restricted_audience="", is_test_isolated=False,
        )),
        (SimpleNamespace(visibility="unknown", platform="", owner=""), SimpleNamespace(
            is_delegated_child=False, restricted_audience="", is_test_isolated=False,
        )),
        (SimpleNamespace(visibility="private", platform="telegram", owner="owner-A"), SimpleNamespace(
            is_delegated_child=True, restricted_audience="", is_test_isolated=False,
        )),
        (SimpleNamespace(visibility="private", platform="telegram", owner="owner-A"), SimpleNamespace(
            is_delegated_child=False, restricted_audience="public_social", is_test_isolated=False,
        )),
    ],
)
def test_public_unknown_restricted_and_delegated_intake_never_falls_back(
    monkeypatch, scope, envelope,
):
    monkeypatch.setattr(source_intake, "current_chat_scope", lambda: scope)
    monkeypatch.setattr(source_intake, "current_turn_envelope", lambda: envelope)
    assert source_intake._resolve_intake_owner(
        _provider(), allow_configured_owner_fallback=True
    ) == ""


def test_cron_intake_requires_verified_binding_to_configured_owner(monkeypatch):
    provider = _provider("cron")
    envelope = SimpleNamespace(
        is_delegated_child=False,
        restricted_audience="",
        is_test_isolated=False,
    )
    cron = SimpleNamespace(
        verified=True,
        cancelled=False,
        bound_owner="owner-A",
    )
    monkeypatch.setattr(source_intake, "current_turn_envelope", lambda: envelope)
    monkeypatch.setattr(source_intake, "current_chat_scope", lambda: SimpleNamespace(
        visibility="unknown", platform="", owner="",
    ))
    monkeypatch.setattr(source_intake, "current_cron_binding", lambda: ("cron-session", "job", "execution"))
    monkeypatch.setattr(source_intake, "verified_cron_envelope", lambda: cron)
    assert source_intake._resolve_intake_owner(provider) == "owner-A"

    cron.bound_owner = "different-owner"
    assert source_intake._resolve_intake_owner(provider) == ""


def test_video_meme_evidence_requires_supported_occurrence_and_web_quote():
    material = source_intake.SourceMaterial(
        source_plugin="bilibili_public", platform="bilibili", material_kind="processed",
        privacy_scope="public", processing_type="video_meme_evidence",
        source_object_id="BV-synthetic", raw_locator="https://www.bilibili.com/video/BV-synthetic",
        title="synthetic phrase", content="comment says synthetic phrase here",
        extra_metadata={
            "meaning": "A synthetic meaning.", "occurrence_kind": "comment",
            "occurrence_quote": "comment says synthetic phrase here",
            "verification_url": "https://example.test/source",
            "verification_text": "The phrase synthetic phrase means: A synthetic meaning.",
            "verification_quote": "The phrase synthetic phrase means: A synthetic meaning.",
        },
    )
    assert source_intake._video_meme_evidence(material) is True

    bad_quote = source_intake.SourceMaterial.from_mapping({
        **material.__dict__,
        "extra_metadata": {**material.extra_metadata, "verification_quote": "invented quote"},
    })
    assert source_intake._video_meme_evidence(bad_quote) is False

    different_meaning = source_intake.SourceMaterial.from_mapping({
        **material.__dict__,
        "extra_metadata": {**material.extra_metadata, "meaning": "A different sense"},
    })
    assert source_intake._video_meme_evidence(different_meaning) is False
    paraphrased = source_intake.SourceMaterial.from_mapping({
        **material.__dict__, "processing_version": "2",
        "extra_metadata": {**material.extra_metadata, "meaning": "A plain-language restatement."},
    })
    assert source_intake._video_meme_evidence(paraphrased) is True
    assert source_intake._video_meme_evidence(source_intake.SourceMaterial.from_mapping({
        **paraphrased.__dict__,
        "extra_metadata": {**paraphrased.extra_metadata, "verification_quote": "synthetic phrase"},
    })) is False

    analysis_only = source_intake.SourceMaterial.from_mapping({
        **material.__dict__,
        "extra_metadata": {**material.extra_metadata, "occurrence_kind": "analysis"},
    })
    assert source_intake._video_meme_evidence(analysis_only) is False


def test_video_meme_evidence_rejects_untrusted_plugin_claim():
    material = source_intake.SourceMaterial(
        source_plugin="manual", platform="bilibili", material_kind="processed",
        processing_type="video_meme_evidence", content="synthetic", title="synthetic",
    )
    with pytest.raises(ValueError, match="untrusted_video_meme_route"):
        source_intake._video_meme_evidence(material)
