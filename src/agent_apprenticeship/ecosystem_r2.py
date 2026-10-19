from __future__ import annotations

import json
import os
import re
import secrets
import shutil
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import (
    DEFAULT_R2_BUCKET,
    DEFAULT_R2_CONTRIBUTION_PREFIX,
    DEFAULT_R2_INDEX_PREFIX,
    DEFAULT_R2_SEED_PREFIX,
    Settings,
    get_settings,
)
from .env import redact_secrets
from .io import append_jsonl, read_workflow_json as read_json, read_workflow_jsonl as read_jsonl, write_json
from .public_sanitizer import sanitize_public_obj
from .version import get_package_version


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def safe_object_id(value: str) -> str:
    raw = str(value or "").strip()
    if "/" in raw or "\\" in raw or ".." in raw:
        raise ValueError("Unsafe ecosystem package id for object storage.")
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", raw).strip(".-")
    if not cleaned or cleaned in {".", ".."} or "/" in cleaned or "\\" in cleaned:
        raise ValueError("Unsafe ecosystem package id for object storage.")
    return cleaned[:160]


def identity_path(settings: Settings | None = None) -> Path:
    s = settings or get_settings()
    return s.app_home / "ecosystem_identity.json"


def shared_history_path(settings: Settings | None = None) -> Path:
    s = settings or get_settings()
    return s.app_home / "shared_contributions.jsonl"


def get_or_create_ecosystem_identity(settings: Settings | None = None) -> dict[str, Any]:
    s = settings or get_settings()
    path = identity_path(s)
    if path.exists():
        data = read_json(path)
        if isinstance(data, dict) and data.get("ecosystem_user_id"):
            return data
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "ecosystem_user_id": f"aa-user-{secrets.token_hex(12)}",
        "installation_id": f"aa-install-{secrets.token_hex(12)}",
        "created_at": utc_now(),
        "display_name": None,
        "metadata_json": {},
    }
    write_json(path, data)
    return data


def shared_history(settings: Settings | None = None) -> list[dict[str, Any]]:
    path = shared_history_path(settings)
    if not path.exists():
        return []
    return [row for row in read_jsonl(path) if isinstance(row, dict)]


def append_shared_history(record: dict[str, Any], settings: Settings | None = None) -> None:
    path = shared_history_path(settings)
    path.parent.mkdir(parents=True, exist_ok=True)
    append_jsonl(path, record)


def shared_count(settings: Settings | None = None) -> int:
    return len(shared_history(settings))


def r2_api_url(settings: Settings | None = None) -> str | None:
    s = settings or get_settings()
    value = (s.r2_api_url or "").strip().rstrip("/")
    return value or None


def r2_status(settings: Settings | None = None) -> dict[str, Any]:
    s = settings or get_settings()
    identity = get_or_create_ecosystem_identity(s)
    return {
        "storage_backend": s.ecosystem_storage_backend,
        "r2_api_url": r2_api_url(s),
        "r2_bucket": s.r2_bucket or DEFAULT_R2_BUCKET,
        "r2_index_prefix": s.r2_index_prefix or DEFAULT_R2_INDEX_PREFIX,
        "r2_contribution_prefix": s.r2_contribution_prefix or DEFAULT_R2_CONTRIBUTION_PREFIX,
        "r2_seed_prefix": s.r2_seed_prefix or DEFAULT_R2_SEED_PREFIX,
        "ecosystem_user_id": identity.get("ecosystem_user_id"),
        "shared_count": shared_count(s),
    }


def item_id_for_package(package_id: str) -> str:
    return f"aa-item-{safe_object_id(package_id)}"


def _bundle_manifest(bundle: Path) -> dict[str, Any]:
    manifest = bundle / "contribution_manifest.json"
    if not manifest.exists():
        raise FileNotFoundError(f"Experience Compilation manifest not found: {manifest}")
    return read_json(manifest)


def _first_text(*values: Any) -> str | None:
    for value in values:
        if isinstance(value, list):
            for item in value:
                if item not in (None, ""):
                    return str(item)
        elif value not in (None, ""):
            return str(value)
    return None


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if value == "":
        return []
    return [value]


def _first_list(*values: Any) -> list[str]:
    for value in values:
        items = [str(item) for item in _as_list(value) if item not in (None, "")]
        if items:
            return items
    return []


def _read_optional_json(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return read_json(path)
    except Exception:
        return None


def _economic_value_category(value: Any, rows_by_type: dict[str, Any], artifacts: int, traces: int) -> str:
    raw = str(value or "").strip().lower()
    if raw in {"low", "medium", "high", "unknown"}:
        return raw
    if raw and raw not in {"none", "null"}:
        if any(token in raw for token in ["high", "strategic", "critical", "enterprise"]):
            return "high"
        if any(token in raw for token in ["low", "small", "minor"]):
            return "low"
        return "medium"
    accepted = 0
    for count in (rows_by_type or {}).values():
        try:
            accepted += int(count or 0)
        except Exception:
            pass
    if accepted >= 12 or artifacts >= 4 or traces >= 3:
        return "high"
    if accepted >= 4 or artifacts >= 1 or traces >= 1:
        return "medium"
    return "unknown"


def ecosystem_item_from_bundle(
    bundle: Path,
    metadata: dict[str, Any],
    *,
    settings: Settings | None = None,
    identity: dict[str, Any] | None = None,
    source: str = "contribution",
) -> dict[str, Any]:
    s = settings or get_settings()
    identity = identity or get_or_create_ecosystem_identity(s)
    manifest = _bundle_manifest(bundle)
    package_id = safe_object_id(str(metadata.get("bundle_id") or manifest.get("bundle_id") or bundle.name))
    item_id = item_id_for_package(package_id)
    api_url = r2_api_url(s)
    root_prefix = (
        (os.environ.get("AA_R2_QA_PREFIX") or "qa").strip("/")
        if source == "qa"
        else (s.r2_contribution_prefix or DEFAULT_R2_CONTRIBUTION_PREFIX).strip("/")
    )
    prefix = f"{root_prefix}/{package_id}"
    download_url = f"{api_url}/v1/ecosystem/packages/{item_id}" if api_url else None
    compiler_dir = bundle / "experience_compiler"
    skill_title = manifest.get("title") or metadata.get("title") or package_id
    skill_summary = ""
    skill_path = compiler_dir / "skill_pack" / "skill.md"
    if skill_path.exists():
        for line in skill_path.read_text(errors="ignore").splitlines():
            clean = line.strip("# ").strip()
            if clean:
                skill_summary = clean[:240]
                break
    rows_by_type = manifest.get("experience_compiler_rows_by_type") or manifest.get("tdo_rows_by_type") or {}
    env_spec = _read_optional_json(compiler_dir / "environment" / "environment_spec.json") or {}
    skill_card = _read_optional_json(compiler_dir / "skill_pack" / "skill_card.json") or {}
    retrieval_card = _read_optional_json(compiler_dir / "skill_pack" / "retrieval_card.json") or {}
    loop_manifest = _read_optional_json(bundle / "loops" / "loop_manifest.json") or {}
    artifact_count = int(metadata.get("artifact_count") or manifest.get("artifact_count") or 0)
    trace_count = int(metadata.get("trace_count") or metadata.get("attempts") or manifest.get("trace_count") or manifest.get("attempts") or 0)
    domains = _first_list(metadata.get("domains"), manifest.get("domains"), [env_spec.get("environment_domain")])
    subdomains = _first_list(metadata.get("subdomains"), manifest.get("subdomains"))
    domain = _first_text(domains) or "unknown"
    subdomain = _first_text(subdomains) or "unknown"
    tags = sorted(set(_first_list(metadata.get("tags"), manifest.get("tags"), skill_card.get("tags"), retrieval_card.get("tags"))))
    surfaces = sorted(set(_first_list(env_spec.get("interaction_surfaces"), metadata.get("surfaces"), manifest.get("surfaces"))))
    tools = sorted(set(_first_list(env_spec.get("required_tools"), metadata.get("tools"), manifest.get("tools"))))
    expected_value = _economic_value_category(
        metadata.get("expected_economic_value") or manifest.get("expected_economic_value"),
        rows_by_type,
        artifact_count,
        trace_count,
    )
    economic_summary = (
        str(metadata.get("economic_value_summary") or manifest.get("economic_value_summary") or "").strip()
        or f"{expected_value.title()} reuse value based on trace evidence, artifact coverage, and generated training rows."
    )
    package_version = str(manifest.get("package_version") or manifest.get("agent_apprenticeship_version") or get_package_version())
    loop_count = int(loop_manifest.get("loop_iteration_count") or manifest.get("loop_iteration_count") or 0)
    has_skill = (compiler_dir / "skill_pack").exists()
    has_training = (compiler_dir / "training_time_package").exists()
    has_eval = (compiler_dir / "eval_package").exists() or (compiler_dir / "evals").exists()
    has_environment = (compiler_dir / "environment").exists()
    return sanitize_public_obj(
        {
            "id": item_id,
            "item_id": item_id,
            "kind": "agent_experience_package",
            "experience_compilation_id": package_id,
            "title": metadata.get("title") or manifest.get("title") or package_id,
            "task_title": manifest.get("title") or metadata.get("title") or package_id,
            "summary": metadata.get("summary") or metadata.get("title") or manifest.get("title") or "",
            "domain": domain,
            "domains": domains or [domain],
            "subdomain": subdomain,
            "subdomains": subdomains or ([subdomain] if subdomain != "unknown" else []),
            "expected_economic_value": expected_value,
            "economic_value_summary": economic_summary,
            "tags": tags,
            "surfaces": surfaces,
            "tools": tools,
            "created_at": metadata.get("created_at") or utc_now(),
            "updated_at": utc_now(),
            "source": source,
            "package_id": package_id,
            "package_version": package_version,
            "task_id": manifest.get("task_id"),
            "apprenticeship_mode": manifest.get("apprenticeship_mode") or metadata.get("apprenticeship_mode"),
            "mentor_mode": manifest.get("mentor_mode"),
            "apprentice_agent": metadata.get("apprentice_agent"),
            "mentor_model_provider": metadata.get("mentor_model_provider"),
            "task_status": metadata.get("task_status") or manifest.get("task_status"),
            "run_status": metadata.get("run_status") or manifest.get("run_status"),
            "attempts": metadata.get("attempts") or manifest.get("attempts") or 0,
            "traced_steps": metadata.get("traced_steps") or manifest.get("traced_steps") or 0,
            "tdo_status": manifest.get("tdo_status"),
            "trace_count": trace_count,
            "artifact_count": artifact_count,
            "loop_iteration_count": loop_count,
            "tdo_rows_by_type": rows_by_type,
            "training_row_counts": rows_by_type,
            "has_full_experience_compilation": True,
            "has_runtime_training": bool((compiler_dir / "skill_pack" / "skill.md").exists()),
            "has_skill_pack": has_skill,
            "has_training_time_package": has_training,
            "has_eval_package": has_eval,
            "has_environment_package": has_environment,
            "available_outputs": {
                "full_experience_compilation": True,
                "full_training_package": True,
                "runtime_training": bool((compiler_dir / "skill_pack" / "skill.md").exists()),
                "skill_pack": has_skill,
                "training_time_package": has_training,
                "eval_package": has_eval,
                "environment_package": has_environment,
            },
            "experience_compiler": {
                "enabled": bool(manifest.get("experience_compiler_enabled") or compiler_dir.exists()),
                "status": manifest.get("experience_compiler_status") or manifest.get("tdo_status"),
                "skill_title": skill_title,
                "skill_summary": skill_summary,
                "runtime_training_summary": skill_summary or f"Runtime training compiled from {skill_title}.",
                "training_data_counts": rows_by_type,
                "environment_domain": metadata.get("environment_domain") or manifest.get("environment_domain") or _first_text(metadata.get("domains"), manifest.get("domains")),
            },
            "contributor": {
                "ecosystem_user_id": identity.get("ecosystem_user_id"),
                "display_name": identity.get("display_name"),
            },
            "storage": {
                "backend": "cloudflare_r2",
                "bucket": s.r2_bucket or DEFAULT_R2_BUCKET,
                "path_prefix": prefix,
                "manifest_path": f"{prefix}/manifest.json",
                "package_path": f"{prefix}/contribution_package.zip",
                "download_url": download_url,
            },
        }
    )


def storage_ref_for_item(
    item: dict[str, Any],
    *,
    settings: Settings | None = None,
    dry_run: bool,
    uploaded_at: str | None = None,
    url: str | None = None,
) -> dict[str, Any]:
    s = settings or get_settings()
    storage = item.get("storage") or {}
    return sanitize_public_obj(
        {
            "storage_backend": "forsy-r2",
            "bucket_backend": "cloudflare_r2",
            "ingestion_api_url": r2_api_url(s),
            "bucket": s.r2_bucket or DEFAULT_R2_BUCKET,
            "path_prefix": storage.get("path_prefix"),
            "manifest_path": storage.get("manifest_path"),
            "package_path": storage.get("package_path"),
            "index_item_path": f"{(s.r2_index_prefix or DEFAULT_R2_INDEX_PREFIX).strip('/')}/items/{item.get('id')}.json",
            "item_id": item.get("id"),
            "package_id": item.get("package_id"),
            "ecosystem_user_id": (item.get("contributor") or {}).get("ecosystem_user_id"),
            "url": url or (r2_api_url(s) + f"/v1/ecosystem/items/{item.get('id')}" if r2_api_url(s) else None),
            "uploaded_at": uploaded_at,
            "dry_run": dry_run,
            "upload_mode": "dry_run" if dry_run else "r2_worker_ingestion",
        }
    )


def write_storage_ref(bundle: Path, submission_dir: Path, storage_ref: dict[str, Any]) -> Path:
    path = submission_dir / "storage_ref.json"
    write_json(path, storage_ref)
    write_json(bundle / "storage_ref.json", storage_ref)
    copied_bundle = submission_dir / "contribution_bundle"
    if copied_bundle.exists():
        write_json(copied_bundle / "storage_ref.json", storage_ref)
    return path


def _multipart_body(fields: dict[str, tuple[str, bytes, str] | str]) -> tuple[bytes, str]:
    boundary = f"----aa-r2-{secrets.token_hex(12)}"
    parts: list[bytes] = []
    for name, value in fields.items():
        parts.append(f"--{boundary}\r\n".encode())
        if isinstance(value, tuple):
            filename, content, content_type = value
            parts.append(
                f'Content-Disposition: form-data; name="{name}"; filename="{filename}"\r\n'
                f"Content-Type: {content_type}\r\n\r\n".encode()
            )
            parts.append(content)
            parts.append(b"\r\n")
        else:
            parts.append(f'Content-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode())
    parts.append(f"--{boundary}--\r\n".encode())
    return b"".join(parts), boundary


def upload_contribution_to_worker(
    *,
    api_url: str,
    package_zip: Path,
    ecosystem_item: dict[str, Any],
    manifest: dict[str, Any],
    storage_ref: dict[str, Any],
    qa: bool = False,
    timeout: int = 60,
) -> dict[str, Any]:
    api_url = api_url.rstrip("/")
    body, boundary = _multipart_body(
        {
            "package_zip": (package_zip.name, package_zip.read_bytes(), "application/zip"),
            "ecosystem_item_json": json.dumps(ecosystem_item, sort_keys=True),
            "manifest_json": json.dumps(manifest, sort_keys=True),
            "storage_ref_json": json.dumps(storage_ref, sort_keys=True),
            "qa": "true" if qa else "false",
        }
    )
    req = urllib.request.Request(
        f"{api_url}/v1/ecosystem/contributions",
        data=body,
        method="POST",
        headers={
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "User-Agent": "agent-apprenticeship-cli",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            payload = response.read().decode("utf-8")
            return json.loads(payload or "{}")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(redact_secrets(f"R2 ingestion failed: HTTP {exc.code} {detail}")) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(redact_secrets(f"R2 ingestion API unavailable: {exc}")) from exc


def _request_json(url: str, *, timeout: int = 30) -> dict[str, Any] | list[Any]:
    req = urllib.request.Request(url, headers={"User-Agent": "agent-apprenticeship-cli"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8") or "{}")


def _request_bytes(url: str, *, timeout: int = 60) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "agent-apprenticeship-cli"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return response.read()


def r2_index_cache_path(settings: Settings | None = None) -> Path:
    s = settings or get_settings()
    return s.app_home / "ecosystem" / "r2_index_cache.json"


def fetch_r2_index(settings: Settings | None = None, *, use_cache: bool = True) -> list[dict[str, Any]]:
    s = settings or get_settings()
    api_url = r2_api_url(s)
    if not api_url:
        if use_cache and r2_index_cache_path(s).exists():
            data = read_json(r2_index_cache_path(s))
            return [row for row in data.get("items", []) if isinstance(row, dict)]
        raise FileNotFoundError("R2 ingestion API is not configured. Run `apprentice ecosystem configure --r2-api-url <url>` or use `--storage github`.")
    data = _request_json(f"{api_url}/v1/ecosystem/index")
    rows = data.get("items") if isinstance(data, dict) else data
    rows = [row for row in rows or [] if isinstance(row, dict)]
    cache = r2_index_cache_path(s)
    cache.parent.mkdir(parents=True, exist_ok=True)
    write_json(cache, {"cached_at": utc_now(), "api_url": api_url, "items": rows})
    return rows


def fetch_r2_item(item_id: str, settings: Settings | None = None) -> dict[str, Any]:
    s = settings or get_settings()
    api_url = r2_api_url(s)
    if not api_url:
        for row in fetch_r2_index(s, use_cache=True):
            if row.get("id") == item_id or row.get("package_id") == item_id or row.get("bundle_id") == item_id:
                return row
        raise FileNotFoundError("R2 ingestion API is not configured and item is not in cache.")
    data = _request_json(f"{api_url}/v1/ecosystem/items/{urllib.parse.quote(item_id)}")
    return data if isinstance(data, dict) else {}


def download_r2_package(item_id: str, dest: Path, settings: Settings | None = None) -> Path:
    s = settings or get_settings()
    api_url = r2_api_url(s)
    if not api_url:
        raise FileNotFoundError("R2 ingestion API is not configured. Use `--storage github` for legacy fallback.")
    data = _request_bytes(f"{api_url}/v1/ecosystem/packages/{urllib.parse.quote(item_id)}")
    zip_path = dest.parent / f"{safe_object_id(item_id)}.zip"
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    zip_path.write_bytes(data)
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            target = dest / info.filename
            if not target.resolve().is_relative_to(dest.resolve()):
                raise RuntimeError("R2 package contains an unsafe path.")
        zf.extractall(dest)
    from .io import validate_trace_tree
    validate_trace_tree(dest)
    return dest


def append_successful_share(
    *,
    item: dict[str, Any],
    bundle: Path,
    storage_backend: str,
    url: str | None,
    sharing_mode: str,
    settings: Settings | None = None,
) -> dict[str, Any]:
    s = settings or get_settings()
    record = {
        "item_id": item.get("id"),
        "package_id": item.get("package_id"),
        "ecosystem_user_id": (item.get("contributor") or {}).get("ecosystem_user_id"),
        "uploaded_at": utc_now(),
        "storage_backend": storage_backend,
        "url": url,
        "local_bundle_path": str(bundle),
        "title": item.get("title"),
        "domain": item.get("domain"),
        "tdo_status": item.get("tdo_status"),
        "apprenticeship_mode": item.get("apprenticeship_mode"),
        "sharing_mode": sharing_mode,
    }
    append_shared_history(record, s)
    return record
