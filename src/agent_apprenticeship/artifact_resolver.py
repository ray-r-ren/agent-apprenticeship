from __future__ import annotations

import re
from pathlib import Path


ARTIFACT_EXTENSIONS = (
    "csv",
    "tsv",
    "json",
    "jsonl",
    "xlsx",
    "xls",
    "md",
    "txt",
    "pdf",
    "docx",
    "pptx",
    "html",
    "xml",
    "yaml",
    "yml",
    "py",
    "js",
    "ts",
    "sql",
    "zip",
    "png",
    "jpg",
    "jpeg",
    "webp",
    "gif",
    "svg",
    "ipynb",
    "log",
)


def normalize_artifact_ref(ref: object) -> str:
    """Normalize display-style artifact evidence refs back to real paths."""

    value = "" if ref is None else str(ref)
    value = value.strip().strip('"').strip("'").replace("\\", "/").strip()
    while value.startswith("./"):
        value = value[2:]

    for prefix in (
        "artifact_content_previews_truncated:",
        "artifact_content_preview_truncated:",
        "artifact_content_previews:",
        "artifact_content_preview:",
        "artifact_previews_truncated:",
        "artifact_preview_truncated:",
        "artifact_previews:",
        "artifact_preview:",
    ):
        if value.startswith(prefix):
            value = value[len(prefix):].strip()

    match = re.search(
        r"((?:packages/[^/\s]+/)?attempts/[^/\s]+/artifacts/[^\s\)\],;]+|artifacts/[^\s\)\],;]+)",
        value,
    )
    if match:
        value = match.group(1).strip()

    for marker in (
        " (parse_error:",
        " (preview_error:",
        " (read_error:",
        " (open_error:",
        " (parse_status:",
        " (text_preview)",
        " (content_preview)",
        " (artifact_preview)",
        " (binary_preview)",
        " (sheet_preview)",
        " (truncated_preview)",
        " (preview_truncated)",
        " (truncated)",
        " (preview)",
    ):
        if marker in value:
            value = value.split(marker, 1)[0].strip()

    if "/artifacts/" in value or value.startswith("artifacts/"):
        extensions = "|".join(re.escape(ext) for ext in ARTIFACT_EXTENSIONS)
        ext_match = re.match(
            rf"^(.*?\.({extensions}))(?:[:#_\s\)].*)?$",
            value,
            flags=re.IGNORECASE,
        )
        if ext_match:
            value = ext_match.group(1).strip()

    return value


def is_artifact_evidence_ref(ref: object) -> bool:
    raw = str(ref).strip()
    normalized = normalize_artifact_ref(raw)
    metadata_prefixes = (
        "trace_summary_json:",
        "actual_outputs:",
        "actual_outputs.",
        "actual_outputs.json",
        "agent_trace.json",
        "rubric:",
        "rubric.",
        "score:",
        "score.",
        "metadata_json:",
        "metadata_json.",
    )
    if raw.startswith(metadata_prefixes) or normalized.startswith(metadata_prefixes):
        return False
    return (
        ("attempts/" in normalized and "/artifacts/" in normalized)
        or normalized.startswith("artifacts/")
        or normalized.startswith("input/")
        or normalized.startswith("task/")
        or ("packages/" in normalized and "/attempts/" in normalized and "/artifacts/" in normalized)
    )


def artifact_ref_resolves(ref: object, existing_refs: set[str]) -> bool:
    normalized = normalize_artifact_ref(ref).replace("\\", "/").lstrip("/")
    if not normalized:
        return False
    existing_normalized = {str(x).replace("\\", "/").lstrip("/") for x in existing_refs if x}
    if normalized in existing_normalized:
        return True
    return any(
        existing.endswith("/" + normalized)
        or normalized.endswith("/" + existing)
        for existing in existing_normalized
    )


def artifact_ref_candidates(ref: object) -> set[str]:
    normalized = normalize_artifact_ref(ref).replace("\\", "/").lstrip("/")
    if not normalized:
        return set()
    candidates = {normalized}
    if "/artifacts/" in normalized:
        candidates.add(normalized.split("/artifacts/", 1)[1])
    candidates.add(Path(normalized).name)
    return {c for c in candidates if c}
