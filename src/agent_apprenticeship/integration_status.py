from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import get_settings, mentor_model_provider_readiness, apprentice_agent_readiness_status
from .io import append_jsonl, read_workflow_jsonl as read_jsonl
from .recipes import MODEL_PROVIDER_RECIPES, WORKER_AGENT_RECIPES


def smoke_cache_path(app_home: Path | None = None):
    return (app_home or get_settings().app_home) / "integration_smoke_results.jsonl"


def record_certification_result(
    *,
    provider_type: str,
    provider_id: str,
    result: str,
    certification_kind: str,
    agent_id: str | None = None,
    model_provider_id: str | None = None,
    error_type: str | None = None,
    error_summary: str | None = None,
    command_or_model: str | None = None,
    app_home: Path | None = None,
    metadata_json: dict[str, Any] | None = None,
) -> None:
    row = {
        "provider_type": provider_type,
        "provider_id": provider_id,
        "result": result,
        "certification_kind": certification_kind,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "agent_id": agent_id,
        "model_provider_id": model_provider_id,
        "error_type": error_type,
        "error_summary": error_summary,
        "command_or_model": command_or_model,
        "metadata_json": metadata_json or {},
    }
    try:
        append_jsonl(smoke_cache_path(app_home), row)
    except OSError:
        # Live smokes are often run from restricted CI/sandbox contexts. A
        # non-writable local app home should not hide the actual provider result.
        return


def record_live_smoke_result(
    *,
    provider_type: str,
    provider_id: str,
    result: str,
    error_type: str | None = None,
    error_summary: str | None = None,
    command_or_model: str | None = None,
) -> None:
    kind = {
        "apprentice_agent": "agent_live_smoke",
        "mentor_model_provider": "model_live_smoke",
    }.get(provider_type, "live_smoke")
    record_certification_result(
        provider_type=provider_type,
        provider_id=provider_id,
        result=result,
        certification_kind=kind,
        error_type=error_type,
        error_summary=error_summary,
        command_or_model=command_or_model,
    )


def latest_smoke_results(app_home: Path | None = None) -> dict[tuple[str, str], dict[str, Any]]:
    path = smoke_cache_path(app_home)
    if not path.exists():
        return {}
    latest: dict[tuple[str, str], dict[str, Any]] = {}
    for row in read_jsonl(path):
        key = (str(row.get("provider_type")), str(row.get("provider_id")))
        latest[key] = row
    return latest


def latest_certification_results(app_home: Path | None = None) -> list[dict[str, Any]]:
    path = smoke_cache_path(app_home)
    if not path.exists():
        return []
    return list(read_jsonl(path))


def _latest_for(rows: list[dict[str, Any]], **filters: str) -> dict[str, Any]:
    match = None
    for row in rows:
        if all(str(row.get(key)) == value for key, value in filters.items()):
            match = row
    return match or {}


def _latest_full_e2e_for(rows: list[dict[str, Any]], *, agent_id: str | None = None, provider_id: str | None = None) -> dict[str, Any]:
    match = None
    for row in rows:
        if row.get("certification_kind") != "full_e2e":
            continue
        if agent_id is not None and row.get("agent_id") != agent_id:
            continue
        if provider_id is not None and row.get("model_provider_id") != provider_id:
            continue
        match = row
    return match or {}


def _last_tested(*rows: dict[str, Any]) -> str | None:
    timestamps = [str(row.get("timestamp")) for row in rows if row.get("timestamp")]
    return max(timestamps) if timestamps else None


def integrations_report() -> dict[str, list[dict[str, Any]]]:
    latest = latest_smoke_results()
    rows = latest_certification_results()
    settings = get_settings()
    agents = []
    for agent_id, recipe in WORKER_AGENT_RECIPES.items():
        s = settings.model_copy(
            update={
                "worker_agent": agent_id,
                "worker_agent_command": settings.worker_agent_command if settings.worker_agent == agent_id else None,
                "custom_worker_command_template": settings.custom_worker_command_template if agent_id == "custom" else settings.custom_worker_command_template,
                "custom_worker_display_name": settings.custom_worker_display_name if agent_id == "custom" else settings.custom_worker_display_name,
            }
        )
        readiness = apprentice_agent_readiness_status(s)
        smoke = latest.get(("apprentice_agent", agent_id), {})
        if agent_id == "custom":
            fixture_smoke = _latest_for(rows, provider_type="apprentice_agent", provider_id="custom", certification_kind="agent_live_smoke_fixture")
            user_smoke = _latest_for(rows, provider_type="apprentice_agent", provider_id="custom", certification_kind="agent_live_smoke_user")
            display_smoke = fixture_smoke or smoke
        else:
            fixture_smoke = {}
            user_smoke = {}
            display_smoke = smoke
        full = _latest_full_e2e_for(rows, agent_id=agent_id)
        agents.append(
            {
                "id": agent_id,
                "display_name": recipe.display_name,
                "adapter_implemented": True,
                "command_expected": readiness.get("command") or recipe.command_name,
                "command_found": bool(readiness.get("command_found")),
                "fake_adapter_test_covered": True,
                "live_smoke_script_available": True,
                "latest_local_live_smoke_result": display_smoke.get("result") or "not_run",
                "latest_full_e2e_result": full.get("result") or "not_run",
                "last_tested_at": _last_tested(smoke, full, fixture_smoke, user_smoke),
                "latest_error_summary": display_smoke.get("error_summary"),
                "latest_full_e2e_error_summary": full.get("error_summary"),
                "custom_fixture_live_smoke_result": fixture_smoke.get("result") if agent_id == "custom" else None,
                "custom_user_live_smoke_result": user_smoke.get("result") if agent_id == "custom" else None,
            }
        )
    providers = []
    for provider_id, recipe in MODEL_PROVIDER_RECIPES.items():
        s = settings.model_copy(
            update={
                "model_provider": provider_id,
                "model_provider_api_key_env": settings.model_provider_api_key_env if settings.model_provider == provider_id else recipe.api_key_env_var,
                "model_provider_model": settings.model_provider_model if settings.model_provider == provider_id else recipe.default_model,
            }
        )
        readiness = mentor_model_provider_readiness(s)
        smoke = latest.get(("mentor_model_provider", provider_id), {})
        full = _latest_full_e2e_for(rows, provider_id=provider_id)
        full_result = full.get("result") or "not_run"
        if full_result == "passed" and str(smoke.get("result") or "").startswith("failed"):
            full_result = "not_certified_due_to_provider_failure"
        providers.append(
            {
                "id": provider_id,
                "display_name": recipe.display_name,
                "adapter_implemented": True,
                "key_env_var": readiness.get("api_key_env_var") or recipe.api_key_env_var,
                "key_visible": bool(readiness.get("api_key_visible")),
                "fake_provider_test_covered": True,
                "live_smoke_script_available": True,
                "latest_local_live_smoke_result": smoke.get("result") or "not_run",
                "latest_full_e2e_result": full_result,
                "last_tested_at": _last_tested(smoke, full),
                "latest_error_summary": smoke.get("error_summary"),
                "latest_full_e2e_error_summary": full.get("error_summary"),
            }
        )
    return {"apprentice_agents": agents, "mentor_model_providers": providers}
