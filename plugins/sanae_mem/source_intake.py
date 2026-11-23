"""Public source material intake API."""

from .governance.source_intake import (
    VALID_MATERIAL_KINDS as VALID_MATERIAL_KINDS,
    VALID_PRIVACY_SCOPES as VALID_PRIVACY_SCOPES,
    SourceIntakeVerdict as SourceIntakeVerdict,
    SourceMaterial as SourceMaterial,
    build_material_source_context as build_material_source_context,
    classify_material_credibility as classify_material_credibility,
    ingest_source_material as ingest_source_material,
    register_source_intake_sink as register_source_intake_sink,
    submit_source_material as submit_source_material,
    unregister_source_intake_sink as unregister_source_intake_sink,
)
