from __future__ import annotations

import os
import shutil
import sys
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import get_settings, init_settings, update_settings
from .command_discovery import resolve_command, resolve_agent_command, gui_app_hint
from .env import redact_secrets
from .integration_status import record_certification_result
from .io import read_workflow_json as read_json
from .openai_structured import run_llm_smoke
from .public_run import run_prompt_task
from .recipes import MODEL_PROVIDER_RECIPES, WORKER_AGENT_RECIPES


RESULT_PASSED = "passed"
RESULT_FAILED = "failed"
RESULT_SKIPPED_MISSING_COMMAND = "skipped_missing_command"
RESULT_SKIPPED_MISSING_KEY = "skipped_missing_key"
RESULT_SKIPPED_NOT_CONFIGURED = "skipped_not_configured"
RESULT_FAILED_OUTPUT_CONTRACT = "failed_output_contract"
RESULT_FAILED_AUTH = "failed_auth"
RESULT_FAILED_QUOTA = "failed_quota"
RESULT_FAILED_TIMEOUT = "failed_timeout"
RESULT_FAILED_PROVIDER_ERROR = "failed_provider_error"
RESULT_FAILED_INSUFFICIENT_BALANCE = "failed_insufficient_balance"


def selected_agent_ids() -> list[str]:
    return list(WORKER_AGENT_RECIPES)


def selected_model_provider_ids() -> list[str]:
    return list(MODEL_PROVIDER_RECIPES)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def custom_fixture_command_template() -> str:
    fixture = _repo_root() / "scripts" / "fixtures" / "successful_custom_apprentice.py"
    return f"{sys.executable} {fixture} --workspace {{workspace}} --prompt-file {{prompt_file}}"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _smoke_timeout_seconds() -> int:
    try:
        return int(os.getenv("AA_TASK_TIMEOUT_SECONDS") or "240")
    except ValueError:
        return 240


def _google_key_visible() -> tuple[str, bool]:
    if os.getenv("GEMINI_API_KEY"):
        return "GEMINI_API_KEY", True
    if os.getenv("GOOGLE_API_KEY"):
        return "GOOGLE_API_KEY", True
    return "GEMINI_API_KEY", False


def model_key_status(provider_id: str) -> tuple[str, bool]:
    recipe = MODEL_PROVIDER_RECIPES[provider_id]
    if provider_id == "google":
        return _google_key_visible()
    return recipe.api_key_env_var, bool(os.getenv(recipe.api_key_env_var))


def _agent_command_for(agent_id: str, *, user_custom: bool = False) -> tuple[str | None, str | None, bool]:
    settings = get_settings()
    recipe = WORKER_AGENT_RECIPES[agent_id]
    if agent_id == "custom":
        if user_custom:
            template = settings.custom_worker_command_template
            if not template:
                return None, None, False
            command = template.split()[0]
            return (resolve_command(command) or command), template, bool(resolve_command(command))
        command = sys.executable
        return command, custom_fixture_command_template(), bool(resolve_command(command))
    configured = settings.worker_agent_command if settings.worker_agent == agent_id and settings.worker_agent_command else recipe.command_name
    _candidate, resolved = resolve_agent_command(agent_id, configured)
    return (resolved or configured), None, bool(resolved)


@contextmanager
def _isolated_app_home(prefix: str, cache_home: Path):
    previous_home = os.environ.get("AA_HOME")
    previous_disable = os.environ.get("AA_DISABLE_LOCAL_ENV")
    try:
        parent = cache_home / "certification_runs"
        parent.mkdir(parents=True, exist_ok=True)
        path = Path(tempfile.mkdtemp(prefix=f"{prefix}-", dir=parent))
    except OSError:
        path = Path(tempfile.mkdtemp(prefix=f"aa-{prefix}-"))
    os.environ["AA_HOME"] = str(path)
    # Certification runs should not accidentally read active app-home settings
    # unless we explicitly copy them below.
    os.environ.pop("AA_DISABLE_LOCAL_ENV", None)
    init_settings(path, overwrite=True)
    try:
        yield path
    finally:
        if previous_home is None:
            os.environ.pop("AA_HOME", None)
        else:
            os.environ["AA_HOME"] = previous_home
        if previous_disable is None:
            os.environ.pop("AA_DISABLE_LOCAL_ENV", None)
        else:
            os.environ["AA_DISABLE_LOCAL_ENV"] = previous_disable


def _base_row(kind: str, result: str, **extra: Any) -> dict[str, Any]:
    return {
        "certification_kind": kind,
        "result": result,
        "timestamp": _utc_now(),
        **extra,
    }


def _classify_error(text: str | None, *, output_contract_default: bool = False) -> tuple[str, str | None, str | None]:
    message = redact_secrets(text or "").strip()
    lower = message.lower()
    if (
        "auth" in lower
        or "not logged in" in lower
        or "login" in lower
        or "unauthorized" in lower
        or "api key" in lower
        or "setup required" in lower
        or "provider not configured" in lower
        or "model not configured" in lower
    ):
        return RESULT_FAILED_AUTH, "auth", message
    if "insufficient balance" in lower:
        return RESULT_FAILED_INSUFFICIENT_BALANCE, "provider_account_balance", message
    if "quota" in lower or "credit" in lower or "rate limit" in lower or "usage limit" in lower or "billing" in lower or "insufficient" in lower:
        return RESULT_FAILED_QUOTA, "quota", message
    if "timeout" in lower or "timed out" in lower:
        return RESULT_FAILED_TIMEOUT, "timeout", message
    if "output-contract" in lower or "did not produce required" in lower or "agent_trace.json" in lower:
        return RESULT_FAILED_OUTPUT_CONTRACT, "output_contract", message or "Apprentice Agent output-contract failure."
    if output_contract_default:
        return RESULT_FAILED_OUTPUT_CONTRACT, "output_contract", message or "Apprentice Agent output-contract failure."
    return RESULT_FAILED_PROVIDER_ERROR, "provider_error", message or "Provider failed."


def _result_from_run_status(run_root: Path) -> tuple[str, str | None, str | None]:
    status_path = run_root / "run_status.json"
    if not status_path.exists():
        return RESULT_FAILED_PROVIDER_ERROR, "missing_status", f"run_status.json was not created at {status_path}"
    status = read_json(status_path)
    if status.get("task_status") == "completed":
        return RESULT_PASSED, None, None
    reason = status.get("last_operational_error") or status.get("latest_message") or "Task did not complete."
    return _classify_error(str(reason), output_contract_default=True)


def _model_smoke_ok(counters: dict[str, Any]) -> bool:
    roles = ["intake", "rubric", "grader", "verifier", "evaluator"]
    return bool(counters.get("mentor_model_provider_available")) and all(
        bool(counters.get(f"{role}_live_call_ok")) and bool(counters.get(f"{role}_structured_output_validation_ok"))
        for role in roles
    ) and bool(counters.get("secret_scan_ok"))


def _model_error(counters: dict[str, Any]) -> tuple[str, str | None, str | None]:
    for key in sorted(counters):
        if key.endswith("_error_message") and counters.get(key):
            result, error_type, summary = _classify_error(str(counters[key]))
            if result == RESULT_FAILED_OUTPUT_CONTRACT:
                result = RESULT_FAILED_PROVIDER_ERROR
            return result, error_type, summary
    return RESULT_FAILED_PROVIDER_ERROR, "provider_error", "Mentor Model Provider live smoke failed."


def _record_agent_row(row: dict[str, Any], cache_home: Path) -> None:
    provider_id = row["agent_id"]
    record_certification_result(
        provider_type="apprentice_agent",
        provider_id=provider_id,
        result=row["result"],
        certification_kind=row["certification_kind"],
        agent_id=provider_id,
        error_type=row.get("error_type"),
        error_summary=row.get("error_summary"),
        command_or_model=row.get("command"),
        app_home=cache_home,
        metadata_json={k: v for k, v in row.items() if k not in {"result", "error_type", "error_summary"}},
    )


def _record_model_row(row: dict[str, Any], cache_home: Path) -> None:
    provider_id = row["provider_id"]
    record_certification_result(
        provider_type="mentor_model_provider",
        provider_id=provider_id,
        result=row["result"],
        certification_kind=row["certification_kind"],
        model_provider_id=provider_id,
        error_type=row.get("error_type"),
        error_summary=row.get("error_summary"),
        command_or_model=row.get("model"),
        app_home=cache_home,
        metadata_json={k: v for k, v in row.items() if k not in {"result", "error_type", "error_summary"}},
    )


def _record_full_row(row: dict[str, Any], cache_home: Path) -> None:
    record_certification_result(
        provider_type="full_e2e",
        provider_id=f"{row['agent_id']}+{row['provider_id']}",
        result=row["result"],
        certification_kind="full_e2e",
        agent_id=row["agent_id"],
        model_provider_id=row["provider_id"],
        error_type=row.get("error_type"),
        error_summary=row.get("error_summary"),
        command_or_model=f"{row.get('command') or row['agent_id']} + {row.get('model') or row['provider_id']}",
        app_home=cache_home,
        metadata_json={k: v for k, v in row.items() if k not in {"result", "error_type", "error_summary"}},
    )


def certify_agent(agent_id: str, *, strict: bool = False, user_custom: bool = False, cache_home: Path | None = None) -> dict[str, Any]:
    cache_home = cache_home or get_settings().app_home
    recipe = WORKER_AGENT_RECIPES[agent_id]
    mode = "user_configured" if agent_id == "custom" and user_custom else ("fixture" if agent_id == "custom" else "live")
    command, template, found = _agent_command_for(agent_id, user_custom=user_custom)
    kind = "agent_live_smoke_user" if mode == "user_configured" else ("agent_live_smoke_fixture" if mode == "fixture" else "agent_live_smoke")
    if command is None:
        row = _base_row(
            kind,
            RESULT_SKIPPED_NOT_CONFIGURED,
            agent_id=agent_id,
            display_name=recipe.display_name,
            mode=mode,
            command=None,
            command_found=False,
            error_type="not_configured",
            error_summary="User-configured Custom Apprentice Agent command template is not configured.",
        )
        _record_agent_row(row, cache_home)
        return row
    if not found:
        row = _base_row(
            kind,
            RESULT_SKIPPED_MISSING_COMMAND,
            agent_id=agent_id,
            display_name=recipe.display_name,
            mode=mode,
            command=command,
            command_found=False,
            error_type="missing_command",
            error_summary=(
                f"Apprentice Agent command not found: {command}"
                + (f". {gui_app_hint(agent_id)} Install or expose the headless CLI on PATH." if gui_app_hint(agent_id) else "")
            ),
        )
        _record_agent_row(row, cache_home)
        return row
    try:
        with _isolated_app_home(f"agent-{agent_id}", cache_home) as home:
            if agent_id == "custom":
                update_settings(
                    worker_agent="custom",
                    worker_agent_command=command,
                    custom_worker_display_name="Custom Fixture" if mode == "fixture" else "Custom",
                    custom_worker_command_template=template,
                    mentor_mode="expert_led",
                    max_improvement_loops=1,
                    task_timeout_seconds=_smoke_timeout_seconds(),
                )
            else:
                update_settings(
                    worker_agent=agent_id,
                    worker_agent_command=command,
                    worker_runner=("codex" if agent_id == "codex" else agent_id),
                    reviser_runner=("codex" if agent_id == "codex" else agent_id),
                    mentor_mode="expert_led",
                    max_improvement_loops=1,
                    task_timeout_seconds=_smoke_timeout_seconds(),
                )
            run_root, bundle = run_prompt_task(
                "Create a one-paragraph readiness note and save it under artifacts/readiness.md.",
                run_id=f"cert-agent-{agent_id}-{mode}",
                create_bundle=True,
            )
            result, error_type, error_summary = _result_from_run_status(run_root)
            row = _base_row(
                kind,
                result,
                agent_id=agent_id,
                display_name=recipe.display_name,
                mode=mode,
                command=command,
                command_found=True,
                run_path=str(run_root),
                bundle_path=str(bundle) if bundle else None,
                error_type=error_type,
                error_summary=error_summary,
            )
    except Exception as exc:
        result, error_type, error_summary = _classify_error(str(exc), output_contract_default=False)
        row = _base_row(
            kind,
            result if result != RESULT_FAILED_OUTPUT_CONTRACT else RESULT_FAILED_PROVIDER_ERROR,
            agent_id=agent_id,
            display_name=recipe.display_name,
            mode=mode,
            command=command,
            command_found=True,
            error_type=error_type,
            error_summary=error_summary,
        )
    _record_agent_row(row, cache_home)
    return row


def certify_model_provider(provider_id: str, *, strict: bool = False, cache_home: Path | None = None) -> dict[str, Any]:
    cache_home = cache_home or get_settings().app_home
    recipe = MODEL_PROVIDER_RECIPES[provider_id]
    key_env, visible = model_key_status(provider_id)
    if not visible:
        row = _base_row(
            "model_live_smoke",
            RESULT_SKIPPED_MISSING_KEY,
            provider_id=provider_id,
            display_name=recipe.display_name,
            model=recipe.default_model,
            api_key_env_var=key_env,
            api_key_visible=False,
            error_type="missing_key",
            error_summary=f"{key_env} is not visible.",
        )
        _record_model_row(row, cache_home)
        return row
    out_dir = cache_home / "certification_model_smokes" / provider_id
    try:
        counters = run_llm_smoke(out_dir, provider_id=provider_id)
        if _model_smoke_ok(counters):
            row = _base_row(
                "model_live_smoke",
                RESULT_PASSED,
                provider_id=provider_id,
                display_name=recipe.display_name,
                model=recipe.default_model,
                api_key_env_var=key_env,
                api_key_visible=True,
                output_dir=str(out_dir),
                counters=counters,
            )
        else:
            result, error_type, error_summary = _model_error(counters)
            row = _base_row(
                "model_live_smoke",
                result,
                provider_id=provider_id,
                display_name=recipe.display_name,
                model=recipe.default_model,
                api_key_env_var=key_env,
                api_key_visible=True,
                output_dir=str(out_dir),
                counters=counters,
                error_type=error_type,
                error_summary=error_summary,
            )
    except Exception as exc:
        result, error_type, error_summary = _classify_error(str(exc))
        row = _base_row(
            "model_live_smoke",
            result,
            provider_id=provider_id,
            display_name=recipe.display_name,
            model=recipe.default_model,
            api_key_env_var=key_env,
            api_key_visible=True,
            output_dir=str(out_dir),
            error_type=error_type,
            error_summary=error_summary,
        )
    _record_model_row(row, cache_home)
    return row


def bounded_full_e2e_pairs(
    *,
    all_combinations: bool = False,
    agent_ids: list[str] | None = None,
    provider_ids: list[str] | None = None,
) -> list[tuple[str, str]]:
    agents = agent_ids or selected_agent_ids()
    providers = provider_ids or selected_model_provider_ids()
    if all_combinations:
        return [(agent, provider) for agent in agents for provider in providers]
    pairs: list[tuple[str, str]] = [("codex", "openai"), ("openclaw", "openai")]
    pairs.extend(("custom", provider) for provider in providers)
    for agent in agents:
        if agent == "custom":
            continue
        command, _, found = _agent_command_for(agent)
        if found:
            pairs.append((agent, "openai"))
    deduped: list[tuple[str, str]] = []
    seen = set()
    for pair in pairs:
        if pair[0] not in agents or pair[1] not in providers:
            continue
        if pair not in seen:
            deduped.append(pair)
            seen.add(pair)
    return deduped


def certify_full_e2e_pair(agent_id: str, provider_id: str, *, cache_home: Path | None = None) -> dict[str, Any]:
    cache_home = cache_home or get_settings().app_home
    agent_recipe = WORKER_AGENT_RECIPES[agent_id]
    provider_recipe = MODEL_PROVIDER_RECIPES[provider_id]
    key_env, key_visible = model_key_status(provider_id)
    if not key_visible:
        row = _base_row(
            "full_e2e",
            RESULT_SKIPPED_MISSING_KEY,
            agent_id=agent_id,
            provider_id=provider_id,
            command=None,
            model=provider_recipe.default_model,
            api_key_env_var=key_env,
            error_type="missing_key",
            error_summary=f"{key_env} is not visible.",
        )
        _record_full_row(row, cache_home)
        return row
    command, template, found = _agent_command_for(agent_id)
    if not found:
        row = _base_row(
            "full_e2e",
            RESULT_SKIPPED_MISSING_COMMAND,
            agent_id=agent_id,
            provider_id=provider_id,
            command=command,
            model=provider_recipe.default_model,
            api_key_env_var=key_env,
            error_type="missing_command",
            error_summary=(
                f"Apprentice Agent command not found: {command}"
                + (f". {gui_app_hint(agent_id)} Install or expose the headless CLI on PATH." if gui_app_hint(agent_id) else "")
            ),
        )
        _record_full_row(row, cache_home)
        return row
    try:
        with _isolated_app_home(f"e2e-{agent_id}-{provider_id}", cache_home):
            task_timeout = max(_smoke_timeout_seconds(), 240) if agent_id == "openclaw" else _smoke_timeout_seconds()
            if agent_id == "custom":
                update_settings(
                    worker_agent="custom",
                    worker_agent_command=command,
                    custom_worker_display_name="Custom Fixture",
                    custom_worker_command_template=template,
                    mentor_mode="model_assisted",
                    model_provider=provider_id,
                    model_provider_api_key_env=key_env,
                    model_provider_model=provider_recipe.default_model,
                    max_improvement_loops=1,
                    task_timeout_seconds=task_timeout,
                )
            else:
                update_settings(
                    worker_agent=agent_id,
                    worker_agent_command=command,
                    worker_runner=("codex" if agent_id == "codex" else agent_id),
                    reviser_runner=("codex" if agent_id == "codex" else agent_id),
                    mentor_mode="model_assisted",
                    model_provider=provider_id,
                    model_provider_api_key_env=key_env,
                    model_provider_model=provider_recipe.default_model,
                    max_improvement_loops=1,
                    task_timeout_seconds=task_timeout,
                )
            run_root, bundle = run_prompt_task(
                "Create one sentence under artifacts/market_note.md.",
                run_id=f"cert-e2e-{agent_id}-{provider_id}",
                create_bundle=True,
            )
            result, error_type, error_summary = _result_from_run_status(run_root)
            row = _base_row(
                "full_e2e",
                result,
                agent_id=agent_id,
                provider_id=provider_id,
                display_name=f"{agent_recipe.display_name} + {provider_recipe.display_name}",
                command=command,
                model=provider_recipe.default_model,
                api_key_env_var=key_env,
                run_path=str(run_root),
                bundle_path=str(bundle) if bundle else None,
                error_type=error_type,
                error_summary=error_summary,
            )
    except Exception as exc:
        result, error_type, error_summary = _classify_error(str(exc))
        row = _base_row(
            "full_e2e",
            result,
            agent_id=agent_id,
            provider_id=provider_id,
            display_name=f"{agent_recipe.display_name} + {provider_recipe.display_name}",
            command=command,
            model=provider_recipe.default_model,
            api_key_env_var=key_env,
            error_type=error_type,
            error_summary=error_summary,
        )
    _record_full_row(row, cache_home)
    return row


def run_certification_matrix(
    *,
    include_agents: bool = False,
    include_models: bool = False,
    include_full_e2e: bool = False,
    all_combinations: bool = False,
    strict: bool = False,
    agent_ids: list[str] | None = None,
    provider_ids: list[str] | None = None,
    full_e2e_pairs: list[tuple[str, str]] | None = None,
) -> dict[str, Any]:
    if not (include_agents or include_models or include_full_e2e):
        include_agents = include_models = include_full_e2e = True
    cache_home = get_settings().app_home
    report: dict[str, Any] = {"agents": [], "models": [], "full_e2e": [], "summary": {}}
    if include_agents:
        for agent_id in agent_ids or selected_agent_ids():
            if agent_id == "custom":
                report["agents"].append(certify_agent(agent_id, strict=strict, user_custom=False, cache_home=cache_home))
                report["agents"].append(certify_agent(agent_id, strict=strict, user_custom=True, cache_home=cache_home))
            else:
                report["agents"].append(certify_agent(agent_id, strict=strict, cache_home=cache_home))
    if include_models:
        for provider_id in provider_ids or selected_model_provider_ids():
            report["models"].append(certify_model_provider(provider_id, strict=strict, cache_home=cache_home))
    if include_full_e2e:
        pairs = full_e2e_pairs or bounded_full_e2e_pairs(
            all_combinations=all_combinations,
            agent_ids=agent_ids,
            provider_ids=provider_ids,
        )
        for agent_id, provider_id in pairs:
            report["full_e2e"].append(certify_full_e2e_pair(agent_id, provider_id, cache_home=cache_home))
    all_rows = [*report["agents"], *report["models"], *report["full_e2e"]]
    report["summary"] = {
        "passed": sum(1 for row in all_rows if row["result"] == RESULT_PASSED),
        "failed": sum(1 for row in all_rows if str(row["result"]).startswith("failed")),
        "skipped": sum(1 for row in all_rows if str(row["result"]).startswith("skipped")),
        "strict": strict,
        "all_combinations": all_combinations,
    }
    return report


def certification_exit_code(report: dict[str, Any], *, strict: bool = False) -> int:
    rows = [*(report.get("agents") or []), *(report.get("models") or []), *(report.get("full_e2e") or [])]
    if any(str(row.get("result")).startswith("failed") for row in rows):
        return 1
    if strict and any(str(row.get("result")).startswith("skipped") for row in rows):
        return 1
    return 0
