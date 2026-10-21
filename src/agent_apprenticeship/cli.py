from __future__ import annotations
import base64
import hashlib
import os
import re
import select
import shlex
import sys
import subprocess
import shutil, json
import zipfile
import urllib.error
import urllib.parse
import urllib.request
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
import typer
from .schemas import RawTaskRecord, WorkflowTrace
from .io import read_workflow_json as read_json, read_workflow_jsonl as read_jsonl, write_json
from .loop import run_task as run_one
from .batch_runner import run_batch as run_batch_impl
from .bundle_exporter import create_contribution_bundle
from .release_exporter import create_release as create_release_impl
from .validation import validate_release as validate_release_impl, format_counters
from .config import (
    APPRENTICESHIP_MODES,
    DATA_SHARING_LEVELS,
    DEFAULT_ECOSYSTEM_STORAGE_BACKEND,
    DEFAULT_PUBLIC_ECOSYSTEM_REPO,
    DEFAULT_R2_BUCKET,
    DEFAULT_R2_CONTRIBUTION_PREFIX,
    DEFAULT_R2_INDEX_PREFIX,
    DEFAULT_R2_SEED_PREFIX,
    ECOSYSTEM_AUTO_SHARE_MODES,
    EVALUATION_MODES,
    MENTOR_MODES,
    SENSITIVE_INFO_MASKING_LEVELS,
    TRAINING_CONTRIBUTION_MODES,
    apprenticeship_mode_display,
    apprenticeship_mode_to_mentor_mode,
    ecosystem_auto_share_display,
    get_settings,
    init_settings,
    model_provider_ready,
    configured_model_provider_ready,
    apprentice_agent_readiness_status,
    mentor_model_provider_readiness,
    mentor_mode_display,
    mentor_mode_to_evaluation_mode,
    normalize_apprenticeship_mode,
    normalize_mentor_mode,
    normalize_ecosystem_auto_share,
    normalize_ecosystem_storage,
    normalize_sensitive_info_masking,
    normalize_training_contribution_mode,
    training_contribution_mode_display,
    debug_settings,
    public_settings,
    public_settings_text,
    update_settings,
)
from .ecosystem_r2 import (
    append_successful_share,
    download_r2_package,
    ecosystem_item_from_bundle,
    fetch_r2_index,
    fetch_r2_item,
    get_or_create_ecosystem_identity,
    r2_api_url,
    r2_status,
    shared_history,
    storage_ref_for_item,
    upload_contribution_to_worker,
    write_storage_ref,
)
from .openai_structured import run_llm_smoke, format_smoke_counters
from .env import contains_secret, redact_secrets
from .public_sanitizer import sanitize_public_obj, sanitize_public_text
from .command_discovery import AGENT_COMMAND_CANDIDATES, resolve_command, resolve_agent_command
from .integration_status import integrations_report
from .certification import certification_exit_code, run_certification_matrix
from .learning import (
    active_packs,
    compare_replay,
    compile_experience_pack,
    install_runtime_training_from_bundle,
    installed_skill_run_refs,
    list_installed_skills,
    list_packs,
    load_pack,
    pack_run_refs,
    replay_instruction_for_pack,
    remove_pack,
    resolve_learning_source,
    resolve_packs_for_run,
    search_learning_sources,
    select_runtime_training_for_run,
    set_installed_skill_enabled,
    slugify,
    uninstall_installed_skill,
    update_pack_status,
    write_before_after_result,
)
from .public_run import RunInterrupted, apprentice_agent_readiness, continue_session, finish_session, run_prompt_task, run_root_for
from .progress import append_progress_event, format_progress_event, format_run_status, read_run_status, watch_progress
from .recipes import MODEL_PROVIDER_RECIPES, WORKER_AGENT_RECIPES
app=typer.Typer(add_completion=False)
configure_app=typer.Typer(add_completion=False, help='Configure Apprentice Agents and Mentor Model Providers.')
ecosystem_app=typer.Typer(add_completion=False, help='Explore and share Agent Apprenticeship Experience Compilations.')
ecosystem_shared_app=typer.Typer(add_completion=False, help='Inspect local shared Experience Compilation history.')
bundle_app=typer.Typer(add_completion=False, help='Inspect local Experience Compilations.')
learn_app=typer.Typer(add_completion=False, help='Create and apply reversible Experience Packs from ecosystem experience.')
app.add_typer(configure_app, name='configure')
app.add_typer(ecosystem_app, name='ecosystem')
ecosystem_app.add_typer(ecosystem_shared_app, name='shared')
app.add_typer(bundle_app, name='bundle')
app.add_typer(learn_app, name='learn')

SLACK_LINK='https://join.slack.com/t/fsycommunity/shared_invite/zt-37417grrb-jFD6BQIYgC5wEMrW2bHssw'

FIRST_RUN_BANNER = """╭────────────────────────────────────────────╮
│                                            │
│          Agent Apprenticeship              │
│                                            │
╰────────────────────────────────────────────╯"""

MODEL_KEY_CANDIDATES = {
    "openai": ["OPENAI_API_KEY"],
    "anthropic": ["ANTHROPIC_API_KEY"],
    "google": ["GEMINI_API_KEY", "GOOGLE_API_KEY"],
    "openrouter": ["OPENROUTER_API_KEY"],
}

def _split_asset_prompt(text: str) -> list[Path]:
    return [Path(p.strip()) for p in text.split(',') if p.strip()]

def _print_bundle_ready(bundle: Path | None) -> None:
    if not bundle:
        return
    typer.echo(f'Experience Compilation path: {bundle}')
    typer.echo(f'Contribute to public ecosystem: apprentice ecosystem contribute {bundle}')
    typer.echo('Public ecosystem: Forsy ecosystem')
    typer.echo(f'View files: open {bundle}')


def _print_public_ecosystem_contribution_help(bundle: str | Path) -> None:
    typer.echo('Contribute to public ecosystem:')
    typer.echo(f'apprentice ecosystem contribute {bundle}')
    typer.echo('')
    typer.echo('Public ecosystem:')
    typer.echo('Forsy ecosystem')


def _print_private_internal_share_note(bundle: str | Path) -> None:
    typer.echo('Private Internal Only: no Public Ecosystem upload was made.')
    typer.echo('Optional manual share:')
    typer.echo(f'apprentice ecosystem contribute {bundle}')


def _print_first_run_banner() -> None:
    typer.echo(FIRST_RUN_BANNER)
    typer.echo("")

def _print_task_result(
    status: dict,
    *,
    followup: bool=False,
    record_only: bool=False,
    include_contribution_help: bool=True,
) -> None:
    task_status = status.get('task_status')
    bundle = status.get('contribution_bundle_path')
    artifacts = status.get('artifacts_path')
    error = status.get('last_operational_error') or status.get('latest_message')
    if followup:
        typer.echo('')
        if record_only:
            typer.echo('Follow-up recorded.')
            typer.echo('No Apprentice Agent loop was run. Use --run-loop to continue work.')
        else:
            typer.echo('Task updated.' if task_status != 'failed' else 'Task update failed.')
    elif task_status == 'completed':
        typer.echo('')
        typer.echo('Task completed.')
    elif task_status == 'failed':
        typer.echo('')
        typer.echo('Task failed.')
        if error:
            typer.echo(f'Reason: {error}')
    elif task_status == 'partial':
        typer.echo('')
        typer.echo('Task partially completed.')
        if error:
            typer.echo(f'Reason: {error}')
    if artifacts:
        label = 'Artifacts/partial files' if task_status == 'failed' else 'Artifacts'
        typer.echo(f'{label}:')
        typer.echo(str(artifacts))
    if bundle:
        typer.echo('Experience Compilation:')
        typer.echo(str(bundle))
        if include_contribution_help:
            typer.echo('')
            mode = get_settings().training_contribution_mode
            typer.echo(f'Agent Training Contribution Mode: {training_contribution_mode_display(mode)}')
            if mode == "private_internal_only":
                _print_private_internal_share_note(bundle)
            typer.echo('')
            typer.echo('View files:')
            typer.echo(f'open {bundle}')
    if not followup and status.get('run_id'):
        typer.echo('')
        typer.echo('Next:')
        typer.echo(f'- Add follow-up: apprentice continue {status["run_id"]}')
        typer.echo(f'- Finish session: apprentice finish {status["run_id"]}')


def _print_session_finished(status: dict, *, include_contribution_help: bool=True) -> None:
    typer.echo('Session finished.')
    typer.echo(f"Task Status: {status.get('task_status')}")
    if status.get('last_operational_error'):
        typer.echo(f"Reason: {status.get('last_operational_error')}")
    if status.get('artifacts_path'):
        typer.echo("Artifacts:")
        typer.echo(str(status.get('artifacts_path')))
    if status.get('contribution_bundle_path'):
        typer.echo("Experience Compilation:")
        typer.echo(str(status.get('contribution_bundle_path')))
        if include_contribution_help:
            typer.echo('')
            mode = get_settings().training_contribution_mode
            typer.echo(f'Agent Training Contribution Mode: {training_contribution_mode_display(mode)}')
            if mode == "private_internal_only":
                _print_private_internal_share_note(status.get('contribution_bundle_path'))
            typer.echo('')
            typer.echo('View files:')
            typer.echo(f"open {status.get('contribution_bundle_path')}")

def _yes_no(value: bool) -> str:
    return "yes" if value else "no"


def _status_label(status: str | None) -> str:
    if status == "ready":
        return "Ready"
    if status in {"not_ready", "missing_api_key", "missing_command"}:
        return "Not ready"
    if status == "untested":
        return "Untested"
    return (status or "not_ready").replace("_", " ").title()


def _status_line(label: str, status: str | None, reason: str | None = None) -> str:
    return f"{label}: {_status_label(status)}" + (f" - {reason}" if reason else "")


def _env_next_step(env_var: str | None) -> str:
    env = env_var or "YOUR_PROVIDER_API_KEY"
    return f"Next: export {env}=... or add {env}=... to ~/.agent-apprenticeship/.env.local"


def _stdin_allows_prompt() -> bool:
    if sys.stdin.isatty():
        return True
    try:
        ready, _, _ = select.select([sys.stdin], [], [], 0)
        if not ready:
            return False
    except Exception:
        pass
    try:
        pos = sys.stdin.tell()
        char = sys.stdin.read(1)
        sys.stdin.seek(pos)
        return bool(char)
    except Exception:
        return False


def _detect_apprentice_agents() -> list[dict]:
    detected = []
    for agent_id, commands in AGENT_COMMAND_CANDIDATES.items():
        recipe = WORKER_AGENT_RECIPES[agent_id]
        configured = next((command for command in commands if shutil.which(command)), None)
        resolved = shutil.which(configured) if configured else None
        if not resolved and os.getenv("AA_DISABLE_LOCAL_ENV") != "1":
            configured, resolved = resolve_agent_command(agent_id)
        if resolved:
            detected.append(
                {
                    "id": agent_id,
                    "display_name": recipe.display_name,
                    "command": configured,
                    "command_resolved": resolved,
                    "command_found": True,
                }
            )
    return detected


def _detect_model_provider_keys() -> list[dict]:
    # get_settings() loads .env.local from cwd and app home before we inspect env.
    get_settings()
    detected = []
    for provider_id, env_vars in MODEL_KEY_CANDIDATES.items():
        visible = next((name for name in env_vars if os.getenv(name)), None)
        if visible:
            recipe = MODEL_PROVIDER_RECIPES[provider_id]
            detected.append(
                {
                    "id": provider_id,
                    "display_name": recipe.display_name,
                    "api_key_env_var": visible,
                    "model": recipe.default_model,
                }
            )
    return detected


def _print_key_storage_guidance() -> None:
    typer.echo("Store Mentor Model Provider keys in ~/.agent-apprenticeship/.env.local, a gitignored repo .env.local, or exported shell env vars.")
    typer.echo("Example:")
    for name in ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY", "OPENROUTER_API_KEY"]:
        typer.echo(f'{name}="..."')
    typer.echo("Agent Apprenticeship records env var names only; it never stores or prints raw keys.")


def _choose_detected_index(count: int, default: int = 1) -> int | None:
    raw = typer.prompt("Choose", default=str(default)).strip().lower()
    if raw in {"y", "yes"}:
        return default if 1 <= default <= count else None
    if raw in {"", "skip", "none", "cancel"}:
        return None
    try:
        idx = int(raw)
    except ValueError:
        return None
    return idx if 1 <= idx <= count else None


def _configure_detected_apprentice_agent(row: dict) -> None:
    agent_id = str(row["id"])
    command = str(row["command"])
    runner = "codex" if agent_id == "codex" else agent_id
    update_settings(
        worker_agent=agent_id,
        worker_agent_command=command,
        worker_runner=runner,
        reviser_runner=runner,
        apprentice_agent_readiness_status="untested",
        apprentice_agent_readiness_reason="Command was auto-detected; run a live smoke before release certification.",
    )


def _configure_detected_model_provider(row: dict) -> None:
    provider_id = str(row["id"])
    recipe = MODEL_PROVIDER_RECIPES[provider_id]
    update_settings(
        model_provider=provider_id,
        model_provider_api_key_env=str(row["api_key_env_var"]),
        model_provider_model=str(row.get("model") or recipe.default_model),
        mentor_model_provider_readiness_status="untested",
        mentor_model_provider_readiness_reason="API key env var was auto-detected; run a live smoke before release certification.",
    )


def _first_run_setup(*, interactive: bool, defaults: bool=False) -> None:
    detected_agents = _detect_apprentice_agents()
    detected_providers = _detect_model_provider_keys()
    typer.echo("")
    if detected_agents:
        typer.echo("Detected Apprentice Agents:")
        for idx, row in enumerate(detected_agents, 1):
            typer.echo(f"{idx}. {row['display_name']} - command found ({row['command']})")
        custom_index = len(detected_agents) + 1
        typer.echo(f"{custom_index}. Custom - use a custom command template")
        if defaults:
            _configure_detected_apprentice_agent(detected_agents[0])
            typer.echo(f"Configured Apprentice Agent: {detected_agents[0]['display_name']}")
        elif interactive:
            choice = _choose_detected_index(custom_index, default=1)
            if choice and choice <= len(detected_agents):
                _configure_detected_apprentice_agent(detected_agents[choice - 1])
            elif choice == custom_index:
                typer.echo("Run: apprentice configure agent custom --command-template \"my-agent run --workspace {workspace} --prompt-file {prompt_file}\"")
        else:
            typer.echo("Run `apprentice configure agent <id>` or rerun `apprentice init --setup` to choose one.")
    else:
        typer.echo("No supported Apprentice Agent CLI was detected.")
        typer.echo("Install one of: Codex, Cursor, Claude Code, OpenClaw, OpenCode, Hermes Agent.")
        typer.echo("Or choose Custom to provide a command template.")
        typer.echo("Then rerun: apprentice init --setup")

    typer.echo("")
    if detected_providers:
        typer.echo("Detected Mentor Model Provider keys:")
        for idx, row in enumerate(detected_providers, 1):
            typer.echo(f"{idx}. {row['display_name']} - {row['api_key_env_var']} visible")
        if defaults:
            _configure_detected_model_provider(detected_providers[0])
            typer.echo(f"Configured Mentor Model Provider: {detected_providers[0]['display_name']}")
        elif interactive:
            choice = _choose_detected_index(len(detected_providers), default=1)
            if choice:
                _configure_detected_model_provider(detected_providers[choice - 1])
        else:
            typer.echo("Run `apprentice configure model <id>` or rerun `apprentice init --setup` to choose one.")
    else:
        typer.echo("No Mentor Model Provider API key was detected.")
        _print_key_storage_guidance()

    typer.echo("")
    default_mode = "autonomous" if detected_providers else "expert_led"
    if defaults:
        mode = default_mode
        update_settings(apprenticeship_mode=mode, training_contribution_mode="public_ecosystem")
        typer.echo(f"Configured Apprenticeship Mode: {apprenticeship_mode_display(mode)}")
        typer.echo("Configured Agent Training Contribution Mode: Public Ecosystem")
    elif interactive and sys.stdin.isatty():
        typer.echo("Choose Apprenticeship Mode:")
        typer.echo("1. Autonomous Apprenticeship")
        typer.echo("2. Expert-Led Apprenticeship")
        typer.echo("3. Organization Custom")
        mode_choice = typer.prompt("Default", default=get_settings().apprenticeship_mode or default_mode).strip().lower()
        mode = {"1": "autonomous", "2": "expert_led", "3": "org_custom"}.get(mode_choice, mode_choice)
        mode = normalize_apprenticeship_mode(mode)
        if mode == "org_custom":
            typer.echo("Organization Custom is available through private setup. Visit https://agentapprenticeship.org/custom")
        typer.echo("")
        typer.echo("Choose Agent Training Contribution Mode:")
        typer.echo("1. Public Ecosystem")
        typer.echo("2. Private Internal Only")
        contribution_choice = typer.prompt("Default", default=get_settings().training_contribution_mode).strip().lower()
        contribution = {"1": "public_ecosystem", "2": "private_internal_only"}.get(contribution_choice, contribution_choice)
        update_settings(apprenticeship_mode=mode, training_contribution_mode=contribution)
        typer.echo(f"Configured Apprenticeship Mode: {apprenticeship_mode_display(mode)}")
        typer.echo(f"Configured Agent Training Contribution Mode: {training_contribution_mode_display(contribution)}")
    elif interactive:
        update_settings(apprenticeship_mode=default_mode, training_contribution_mode="public_ecosystem")
        typer.echo(f"Configured Apprenticeship Mode: {apprenticeship_mode_display(default_mode)}")
        typer.echo("Configured Agent Training Contribution Mode: Public Ecosystem")


def _progress_callback(quiet: bool=False, verbose: bool=False, json_progress: bool=False):
    if quiet:
        return None
    def _emit(event: dict, status: dict) -> None:
        if json_progress:
            typer.echo(json.dumps(event, sort_keys=True))
            return
        event_type = event.get('event_type')
        if event_type == 'run_started':
            typer.echo('Agent Apprenticeship run started')
            typer.echo('')
            typer.echo(f'Run: {status.get("run_id")}')
            typer.echo(f'Apprentice Agent: {status.get("apprentice_agent") or status.get("worker_agent")}')
            typer.echo(f'Apprenticeship Mode: {apprenticeship_mode_display(status.get("apprenticeship_mode") or status.get("mentor_mode"))}')
            typer.echo(f'Maximum Improvement Loops: {status.get("maximum_improvement_loops")}')
            typer.echo(f'Task workspace: {status.get("task_workspace_path")}')
            typer.echo(f'Artifacts: {status.get("artifacts_path")}')
            typer.echo('')
            return
        if event_type in {'run_completed'}:
            _print_task_result(status)
            return
        if event_type in {'followup_started'}:
            typer.echo(format_progress_event(event))
            return
        if event_type in {'followup_completed'}:
            _print_task_result(status, followup=True, record_only=bool((event.get('metadata_json') or {}).get('record_only')))
            return
        if event_type == 'task_outputs_ready':
            typer.echo('')
            task_status = event.get('task_status') or status.get('task_status')
            typer.echo('Task completed.' if task_status == 'completed' else 'Task outputs are available.')
            if event.get('artifacts_path'):
                typer.echo(f"Artifacts are available at: {event.get('artifacts_path')}")
            return
        if event_type in {'tdo_started','tdo_round_generated','tdo_round_judged','tdo_round_improved','tdo_completed','tdo_failed'}:
            typer.echo(format_progress_event(event))
            return
        if event_type in {'apprentice_attempt_started','apprentice_attempt_completed','worker_attempt_started','worker_attempt_completed','revision_started','revision_completed','mentor_review_started','mentor_review_completed','task_workspace_prepared','contribution_bundle_completed','operational_error'} or verbose:
            typer.echo(format_progress_event(event))
    return _emit


def _interactive_checkpoint_requested(
    settings,
    *,
    expert_auto_approve: bool = False,
    hybrid_auto_approve: bool = False,
    expert_interactive: bool = False,
    hybrid_interactive: bool = False,
) -> bool:
    if settings.mentor_mode == "expert_led":
        if expert_auto_approve or os.getenv("AA_EXPERT_AUTO_APPROVE") == "1":
            return False
        return expert_interactive or os.getenv("AA_EXPERT_INTERACTIVE") == "1" or sys.stdin.isatty()
    if settings.mentor_mode == "hybrid":
        if hybrid_auto_approve or os.getenv("AA_HYBRID_AUTO_APPROVE") == "1":
            return False
        return hybrid_interactive or os.getenv("AA_HYBRID_INTERACTIVE") == "1" or sys.stdin.isatty()
    return False


@contextmanager
def _temporary_auto_approve_env(
    *,
    expert_auto_approve: bool = False,
    hybrid_auto_approve: bool = False,
    mentor_interactive_checkpoints: bool = False,
):
    old_expert = os.environ.get("AA_EXPERT_AUTO_APPROVE")
    old_hybrid = os.environ.get("AA_HYBRID_AUTO_APPROVE")
    old_interactive = os.environ.get("AA_MENTOR_INTERACTIVE_CHECKPOINTS")
    if expert_auto_approve:
        os.environ["AA_EXPERT_AUTO_APPROVE"] = "1"
    if hybrid_auto_approve:
        os.environ["AA_HYBRID_AUTO_APPROVE"] = "1"
    if mentor_interactive_checkpoints:
        os.environ["AA_MENTOR_INTERACTIVE_CHECKPOINTS"] = "1"
    try:
        yield
    finally:
        if old_expert is None:
            os.environ.pop("AA_EXPERT_AUTO_APPROVE", None)
        else:
            os.environ["AA_EXPERT_AUTO_APPROVE"] = old_expert
        if old_hybrid is None:
            os.environ.pop("AA_HYBRID_AUTO_APPROVE", None)
        else:
            os.environ["AA_HYBRID_AUTO_APPROVE"] = old_hybrid
        if old_interactive is None:
            os.environ.pop("AA_MENTOR_INTERACTIVE_CHECKPOINTS", None)
        else:
            os.environ["AA_MENTOR_INTERACTIVE_CHECKPOINTS"] = old_interactive

def _normalize_evaluation_mode(value: str | None) -> str | None:
    if value is None:
        return None
    normalized=value.strip().lower().replace('_','-')
    if normalized not in EVALUATION_MODES:
        raise typer.BadParameter(f'apprenticeship mode must be one of: {", ".join(APPRENTICESHIP_MODES)}')
    return normalized

def _normalize_mentor_mode(value: str | None) -> str | None:
    if value is None:
        return None
    try:
        return normalize_mentor_mode(value)
    except ValueError as exc:
        raise typer.BadParameter(f'apprenticeship mode must be one of: {", ".join(APPRENTICESHIP_MODES)}') from exc

def _normalize_data_sharing(value: str | None) -> str | None:
    if value is None:
        return None
    normalized=value.strip().lower().replace('_','-').replace(' ', '-')
    if normalized == 'full':
        normalized='full-context'
    if normalized not in DATA_SHARING_LEVELS:
        raise typer.BadParameter(
            f'sensitive info masking must be one of: {", ".join(SENSITIVE_INFO_MASKING_LEVELS)}'
        )
    return normalized

def _normalize_masking(value: str | None) -> str | None:
    if value is None:
        return None
    try:
        return normalize_sensitive_info_masking(value)
    except ValueError as exc:
        raise typer.BadParameter(
            f'sensitive info masking must be one of: {", ".join(SENSITIVE_INFO_MASKING_LEVELS)}'
        ) from exc


def _normalize_auto_share(value: str | None) -> str | None:
    if value is None:
        return None
    try:
        return normalize_ecosystem_auto_share(value)
    except ValueError as exc:
        raise typer.BadParameter(
            f'legacy sharing alias must be one of: {", ".join(ECOSYSTEM_AUTO_SHARE_MODES)}'
        ) from exc


def _normalize_apprenticeship_mode(value: str | None) -> str | None:
    if value is None:
        return None
    try:
        return normalize_apprenticeship_mode(value)
    except ValueError as exc:
        raise typer.BadParameter(
            f'apprenticeship mode must be one of: {", ".join(APPRENTICESHIP_MODES)}'
        ) from exc


def _normalize_training_contribution(value: str | None) -> str | None:
    if value is None:
        return None
    try:
        return normalize_training_contribution_mode(value)
    except ValueError as exc:
        raise typer.BadParameter(
            f'agent training contribution mode must be one of: {", ".join(TRAINING_CONTRIBUTION_MODES)}'
        ) from exc

def _settings_for_run(
    apprenticeship_mode: str | None=None,
    mentor_mode: str | None=None,
    sensitive_info_masking: str | None=None,
    max_loops: int | None=None,
    evaluation_mode: str | None=None,
    data_sharing_level: str | None=None,
):
    settings=get_settings()
    updates={}
    legacy_mentor_mode = _normalize_mentor_mode(mentor_mode) if mentor_mode is not None else None
    if apprenticeship_mode is not None:
        updates['apprenticeship_mode']=_normalize_apprenticeship_mode(apprenticeship_mode)
    if mentor_mode is not None:
        updates['apprenticeship_mode']=_normalize_apprenticeship_mode(mentor_mode)
    if evaluation_mode is not None:
        updates['evaluation_mode']=_normalize_evaluation_mode(evaluation_mode)
        if mentor_mode is None and apprenticeship_mode is None:
            updates['apprenticeship_mode']=_normalize_apprenticeship_mode(evaluation_mode)
    if sensitive_info_masking is not None:
        updates['sensitive_info_masking']=_normalize_masking(sensitive_info_masking)
    if data_sharing_level is not None:
        updates['data_sharing_level']=_normalize_data_sharing(data_sharing_level)
        if sensitive_info_masking is None:
            updates['sensitive_info_masking']=_normalize_masking(data_sharing_level)
    if max_loops is not None:
        if max_loops < 1:
            raise typer.BadParameter('maximum improvement loops must be at least 1')
        updates['max_improvement_loops']=max_loops
        updates['max_iterations']=max_loops
    if "apprenticeship_mode" in updates:
        updates["mentor_mode"] = apprenticeship_mode_to_mentor_mode(updates["apprenticeship_mode"])
        updates["evaluation_mode"] = mentor_mode_to_evaluation_mode(updates["mentor_mode"])
    if legacy_mentor_mode == "hybrid" and apprenticeship_mode is None:
        updates["mentor_mode"] = "hybrid"
        updates["evaluation_mode"] = "hybrid"
    return settings.model_copy(update=updates)

def _configure_model_impl(provider: str | None, model: str | None, api_key_env_var: str | None, test_connection: bool):
    if provider is None:
        detected = _detect_model_provider_keys()
        if detected:
            typer.echo('Detected Mentor Model Provider keys:')
            for idx, row in enumerate(detected, 1):
                typer.echo(f"{idx}. {row['display_name']} - {row['api_key_env_var']} visible")
            choice = typer.prompt('Choose Mentor Model Provider or press Enter to list all', default='', show_default=False).strip()
            if choice:
                try:
                    idx = int(choice)
                except ValueError:
                    provider = choice
                else:
                    if 1 <= idx <= len(detected):
                        provider = str(detected[idx - 1]["id"])
                        api_key_env_var = api_key_env_var or str(detected[idx - 1]["api_key_env_var"])
        if provider is not None:
            return _configure_model_impl(provider, model, api_key_env_var, test_connection)
        typer.echo('Mentor Model Provider options:')
        for key, recipe in MODEL_PROVIDER_RECIPES.items():
            typer.echo(f'- {key}: {recipe.display_name}')
        provider=typer.prompt('Mentor Model Provider')
    provider=provider.strip().lower()
    if provider not in MODEL_PROVIDER_RECIPES:
        raise typer.BadParameter(f'Mentor Model Provider must be one of: {", ".join(MODEL_PROVIDER_RECIPES)}')
    recipe=MODEL_PROVIDER_RECIPES[provider]
    env_var=api_key_env_var or recipe.api_key_env_var
    selected_model=model or recipe.default_model
    get_settings()
    key_visible=bool(os.getenv(env_var))
    if provider == "google" and not key_visible and os.getenv("GOOGLE_API_KEY"):
        key_visible=True
        env_var="GOOGLE_API_KEY"
    readiness_status="untested" if key_visible else "missing_api_key"
    readiness_reason=None if key_visible else f"{env_var} is not visible."
    settings=update_settings(
        model_provider=provider,
        model_provider_api_key_env=env_var,
        model_provider_model=selected_model,
        mentor_model_provider_readiness_status=readiness_status,
        mentor_model_provider_readiness_reason=readiness_reason,
    )
    if test_connection and key_visible:
        out_dir=settings.app_home / "model_provider_smoke" / provider
        counters=run_llm_smoke(out_dir, provider_id=provider)
        role_names=("intake","rubric","grader","verifier","evaluator")
        ready=bool(counters.get("secret_scan_ok")) and all(
            counters.get(f"{name}_live_call_ok") and counters.get(f"{name}_structured_output_validation_ok")
            for name in role_names
        )
        readiness_status="ready" if ready else "failed"
        readiness_reason=None if ready else "Live Mentor Model Provider check failed; inspect smoke artifacts under " + str(out_dir)
        settings=update_settings(
            mentor_model_provider_readiness_status=readiness_status,
            mentor_model_provider_readiness_reason=readiness_reason,
        )
    typer.echo('Mentor Model Provider configured')
    typer.echo(f'Provider: {recipe.display_name}')
    typer.echo(f'Model: {selected_model}')
    typer.echo(f'API key env var: {env_var}')
    typer.echo(f'API key visible: {_yes_no(key_visible)}')
    typer.echo(f'Live test run: {"yes" if test_connection else "no"}')
    typer.echo(_status_line('Mentor Model Provider Status', readiness_status, readiness_reason))
    if not key_visible:
        typer.echo(_env_next_step(env_var))
    else:
        typer.echo(f'Next: AA_RUN_LIVE_MODEL_PROVIDER_SMOKE={provider} bash scripts/live_model_provider_smoke.sh {provider}')
    if test_connection:
        if not key_visible:
            raise typer.Exit(1)
    return settings

def _ensure_evaluation_ready(settings, interactive: bool=True):
    if settings.mentor_mode == 'expert_led' or model_provider_ready(settings):
        return settings
    if not interactive:
        raise typer.BadParameter('Autonomous Apprenticeship requires a ready Mentor Model Provider; use configure model or --apprenticeship-mode expert-led')
    if not _stdin_allows_prompt():
        readiness = mentor_model_provider_readiness(settings)
        reason = readiness.get("reason") or "Mentor Model Provider is not ready."
        raise typer.BadParameter(
            f"Autonomous Apprenticeship requires a ready Mentor Model Provider. {reason} "
            "Run `apprentice doctor --live`, `apprentice configure model`, or use `--apprenticeship-mode expert-led`."
        )
    typer.echo('Autonomous Apprenticeship needs a ready Mentor Model Provider.')
    readiness=mentor_model_provider_readiness(settings)
    if readiness.get("provider"):
        typer.echo(_status_line('Mentor Model Provider Status', readiness.get("status"), readiness.get("reason")))
    typer.echo('')
    typer.echo('Choose:')
    typer.echo('1. Configure Mentor Model Provider')
    typer.echo('2. Run this session in expert-led mode')
    typer.echo('3. Cancel')
    typer.echo('')
    try:
        choice=typer.prompt('Default', default='expert-led').strip().lower()
    except typer.Abort:
        reason = readiness.get("reason") or "Mentor Model Provider is not ready."
        raise typer.BadParameter(
            f"Autonomous Apprenticeship requires a ready Mentor Model Provider. {reason} "
            "Run `apprentice doctor --live`, `apprentice configure model`, or use `--apprenticeship-mode expert-led`."
        ) from None
    if choice in {'1','setup','set up','configure','configure mentor model provider','set up model provider now'}:
        configured=_configure_model_impl(None, None, None, test_connection=False)
        return configured
    if choice in {'2','expert-led','expert','manual','run this session in expert-led mode'}:
        return settings.model_copy(update={'apprenticeship_mode':'expert_led', 'mentor_mode':'expert_led', 'evaluation_mode':'expert-led'})
    raise typer.Exit(1)


def _ensure_apprentice_agent_ready(settings, runner: str | None=None, interactive: bool=True):
    if runner == "deterministic":
        return
    status = apprentice_agent_readiness_status(settings)
    if status.get("status") == "ready":
        return
    reason = status.get("reason")
    if not interactive:
        raise typer.BadParameter(reason or 'Apprentice Agent is not ready.')
    if not _stdin_allows_prompt():
        if status.get("command_found"):
            typer.echo("Apprentice Agent readiness is untested; command was found, continuing without an interactive prompt.")
            return
        raise typer.BadParameter(
            f"Apprentice Agent is not ready. {reason or 'Command was not found.'} "
            "Run `apprentice configure agent` or install/authenticate a selected Apprentice Agent CLI."
        )
    if status.get("status") == "untested":
        typer.echo('Apprentice Agent Status: Untested')
        if reason:
            typer.echo(f'Reason: {reason}')
        typer.echo('Run a quick readiness check now?')
        typer.echo('')
        typer.echo('1. Run check')
        typer.echo('2. Run anyway')
        typer.echo('3. Configure Apprentice Agent')
        typer.echo('4. Cancel')
        typer.echo('')
        try:
            choice=typer.prompt('Default', default='run anyway').strip().lower()
        except typer.Abort:
            if status.get("command_found"):
                typer.echo("Apprentice Agent readiness is untested; command was found, continuing without an interactive prompt.")
                return
            raise typer.BadParameter(
                f"Apprentice Agent is not ready. {reason or 'Command was not found.'} "
                "Run `apprentice configure agent` or install/authenticate a selected Apprentice Agent CLI."
            ) from None
        if choice in {'1','run check','check'}:
            if status.get("command_found"):
                update_settings(apprentice_agent_readiness_status="ready", apprentice_agent_readiness_reason="Command availability check passed.")
                typer.echo('Apprentice Agent Status: Ready')
                return
            typer.echo(f'Reason: {reason}')
            raise typer.Exit(1)
        if choice in {'2','run anyway','continue','continue anyway'}:
            return
        if choice in {'3','configure','configure apprentice agent','configure agent'}:
            typer.echo('Run: agent-apprenticeship configure agent')
            raise typer.Exit(1)
        raise typer.Exit(1)
    typer.echo('Apprentice Agent is not ready.')
    typer.echo(f'Reason: {reason}')
    typer.echo('')
    typer.echo('Choose:')
    typer.echo('1. Configure Apprentice Agent')
    typer.echo('2. Continue anyway')
    typer.echo('3. Cancel')
    typer.echo('')
    try:
        choice=typer.prompt('Default', default='cancel').strip().lower()
    except typer.Abort:
        if status.get("command_found"):
            typer.echo("Apprentice Agent is not ready; command was found, continuing without an interactive prompt.")
            return
        raise typer.BadParameter(
            f"Apprentice Agent is not ready. {reason or 'Command was not found.'} "
            "Run `apprentice configure agent` or install/authenticate a selected Apprentice Agent CLI."
        ) from None
    if choice in {'1','configure','configure apprentice agent','configure agent'}:
        typer.echo('Run: agent-apprenticeship configure agent')
        raise typer.Exit(1)
    if choice in {'2','continue','continue anyway'}:
        return
    raise typer.Exit(1)

@app.command('init')
def init(
    overwrite: bool=typer.Option(False, help='Replace existing settings.json with defaults.'),
    setup: bool=typer.Option(False, '--setup', help='Detect installed Apprentice Agents and Mentor Model Provider keys.'),
    defaults: bool=typer.Option(False, '--defaults', '--non-interactive', help='Initialize with deterministic defaults and never prompt.'),
):
    _print_first_run_banner()
    path=init_settings(overwrite=overwrite)
    get_or_create_ecosystem_identity(get_settings())
    typer.echo(f'initialized Agent Apprenticeship settings at {path}')
    if defaults:
        _first_run_setup(interactive=False, defaults=True)
    elif setup or sys.stdin.isatty():
        _first_run_setup(interactive=bool(setup or sys.stdin.isatty()))

@app.command('settings')
def settings(
    as_json: bool=typer.Option(False, '--json', help='Print settings as JSON.'),
    debug: bool=typer.Option(False, '--debug', help='Include raw/internal compatibility settings.'),
):
    if debug:
        typer.echo(json.dumps(debug_settings(), indent=2, sort_keys=True))
    elif as_json:
        typer.echo(json.dumps(public_settings(), indent=2, sort_keys=True))
    else:
        typer.echo(_public_settings_text_with_ecosystem())


def _latest_run_path(settings) -> Path | None:
    runs = settings.app_home / "runs"
    if not runs.exists():
        return None
    candidates = [p for p in runs.iterdir() if p.is_dir()]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


@app.command('doctor')
def doctor(
    live: bool=typer.Option(False, '--live', help='Run live readiness checks when provider/tools are configured.'),
    integrations: bool=typer.Option(False, '--integrations', help='Print selected integration health matrix.'),
):
    if integrations:
        _print_integrations_report()
        return
    settings=get_settings()
    agent=apprentice_agent_readiness_status(settings)
    provider=mentor_model_provider_readiness(settings)
    typer.echo('Agent Apprenticeship Doctor')
    typer.echo('')
    typer.echo(f'App Home: {settings.app_home}')
    typer.echo(f'Apprentice Agent: {public_settings(settings)["apprentice_agent"]}')
    typer.echo(f'Apprentice Agent command: {agent.get("command") or "Not configured"}')
    typer.echo(f'Apprentice Agent command found: {_yes_no(bool(agent.get("command_found")))}')
    typer.echo(_status_line('Apprentice Agent readiness', agent.get('status'), agent.get('reason')))
    typer.echo(f'Apprenticeship Mode: {apprenticeship_mode_display(settings.apprenticeship_mode)}')
    typer.echo(f'Mentor Model Provider: {provider.get("provider_display")}')
    typer.echo(f'Mentor model: {provider.get("model") or "Not configured"}')
    typer.echo(f'API key env var: {provider.get("api_key_env_var") or "Not configured"}')
    typer.echo(f'API key visible: {_yes_no(bool(provider.get("api_key_visible")))}')
    typer.echo(_status_line('Mentor Model Provider readiness', provider.get('status'), provider.get('reason')))
    typer.echo(f'Maximum Improvement Loops: {settings.max_improvement_loops}')
    typer.echo(f'Sensitive Info Masking: {settings.sensitive_info_masking}')
    typer.echo(f'Public Ecosystem Repo: {settings.ecosystem_repo or DEFAULT_PUBLIC_ECOSYSTEM_REPO}')
    typer.echo(f'GitHub URL: {_ecosystem_repo_url(settings.ecosystem_repo)}')
    typer.echo(_ecosystem_auto_share_status_line(settings).replace("\n", "\n"))
    latest=_latest_run_path(settings)
    typer.echo(f'Latest run: {latest if latest else "None"}')
    if not live:
        return
    typer.echo('')
    typer.echo('Live Checks')
    if not provider.get('provider') or not provider.get('api_key_visible'):
        typer.echo('Mentor Model Provider live check: SKIPPED - provider/API key is not ready.')
    else:
        out_dir=settings.app_home / 'doctor_live_model' / str(provider.get('provider'))
        counters=run_llm_smoke(out_dir, provider_id=str(provider.get('provider')))
        roles=['intake','rubric','grader','verifier','evaluator']
        ok=bool(counters.get('mentor_model_provider_available')) and all(
            counters.get(f'{r}_live_call_ok') and counters.get(f'{r}_structured_output_validation_ok')
            for r in roles
        ) and bool(counters.get('secret_scan_ok'))
        typer.echo(f'Mentor Model Provider live check: {"PASSED" if ok else "FAILED"}')
        if ok:
            update_settings(
                mentor_model_provider_readiness_status="ready",
                mentor_model_provider_readiness_reason="doctor --live passed.",
            )
        if not ok:
            for key in sorted(k for k in counters if k.endswith('_error_message') or k.endswith('_error_type')):
                typer.echo(f'{key}: {counters[key]}')
    if agent.get('status') == 'missing_command':
        typer.echo('Apprentice Agent live check: SKIPPED - command not found.')
    elif agent.get('command_found'):
        typer.echo('Apprentice Agent live check: PASSED - command availability check passed.')
    else:
        typer.echo('Apprentice Agent live check: SKIPPED - Apprentice Agent is not configured.')


def _print_integrations_report() -> None:
    report = integrations_report()
    typer.echo("Agent Apprenticeship Integrations")
    typer.echo("")
    _print_integration_groups(report)
    typer.echo("")
    typer.echo("Apprentice Agents")
    for row in report["apprentice_agents"]:
        typer.echo(f"- {row['id']}: {row['display_name']}")
        typer.echo(f"  adapter implemented: {_yes_no(row['adapter_implemented'])}")
        typer.echo(f"  command expected: {row['command_expected']}")
        typer.echo(f"  command found: {_yes_no(row['command_found'])}")
        typer.echo(f"  fake adapter test covered: {_yes_no(row['fake_adapter_test_covered'])}")
        typer.echo(f"  live smoke script available: {_yes_no(row['live_smoke_script_available'])}")
        typer.echo(f"  latest local live smoke result: {row['latest_local_live_smoke_result']}")
        typer.echo(f"  latest full E2E result: {row['latest_full_e2e_result']}")
        if row.get("last_tested_at"):
            typer.echo(f"  last tested: {row['last_tested_at']}")
        if row["id"] == "custom":
            typer.echo(f"  custom fixture live smoke result: {row.get('custom_fixture_live_smoke_result') or 'not_run'}")
            typer.echo(f"  custom user-configured live smoke result: {row.get('custom_user_live_smoke_result') or 'not_run'}")
        if row.get("latest_error_summary"):
            typer.echo(f"  latest error summary: {_concise_integration_error(row['latest_error_summary'], provider_id=row['id'])}")
        if row.get("latest_full_e2e_error_summary"):
            typer.echo(f"  latest full E2E error summary: {_concise_integration_error(row['latest_full_e2e_error_summary'], provider_id=row['id'])}")
        next_action = _integration_next_action(row, provider_type="apprentice_agent")
        if next_action:
            typer.echo(f"  next action: {next_action}")
    typer.echo("")
    typer.echo("Mentor Model Providers")
    for row in report["mentor_model_providers"]:
        typer.echo(f"- {row['id']}: {row['display_name']}")
        typer.echo(f"  adapter implemented: {_yes_no(row['adapter_implemented'])}")
        typer.echo(f"  key env var: {row['key_env_var']}")
        typer.echo(f"  key visible: {_yes_no(row['key_visible'])}")
        typer.echo(f"  fake provider test covered: {_yes_no(row['fake_provider_test_covered'])}")
        typer.echo(f"  live smoke script available: {_yes_no(row['live_smoke_script_available'])}")
        typer.echo(f"  latest local live smoke result: {row['latest_local_live_smoke_result']}")
        typer.echo(f"  latest full E2E result: {row['latest_full_e2e_result']}")
        if row.get("last_tested_at"):
            typer.echo(f"  last tested: {row['last_tested_at']}")
        if row.get("latest_error_summary"):
            typer.echo(f"  latest error summary: {_concise_integration_error(row['latest_error_summary'], provider_id=row['id'])}")
        if row.get("latest_full_e2e_error_summary"):
            typer.echo(f"  latest full E2E error summary: {_concise_integration_error(row['latest_full_e2e_error_summary'], provider_id=row['id'])}")
        next_action = _integration_next_action(row, provider_type="mentor_model_provider")
        if next_action:
            typer.echo(f"  next action: {next_action}")


def _status_is_passed(value: str | None) -> bool:
    return value == "passed"


def _status_is_external_blocker(value: str | None) -> bool:
    return value in {
        "failed_auth",
        "failed_quota",
        "failed_insufficient_balance",
        "skipped_missing_key",
        "skipped_missing_command",
        "skipped_not_configured",
        "not_certified_due_to_provider_failure",
    }


def _status_is_framework_failure(value: str | None) -> bool:
    return value in {"failed", "failed_output_contract", "failed_timeout", "failed_provider_error"}


def _concise_integration_error(text: object, *, provider_id: str | None = None) -> str:
    raw = redact_secrets(str(text or "")).strip()
    low = raw.lower()
    if provider_id == "google" or "generativelanguage.googleapis.com" in low or "gemini" in low:
        if "quota" in low or "rate" in low or "resource_exhausted" in low:
            return "Google Gemini quota/rate limit reached."
        if "structured" in low or "json" in low:
            return "Google Gemini structured-output response could not be validated."
    if provider_id == "openrouter" or "openrouter" in low:
        if "requires more credits" in low or "can only afford" in low or "credit" in low or "402" in low:
            return "OpenRouter credits/request-size limit reached."
        if "parse" in low or "json" in low:
            return "OpenRouter provider response could not be parsed."
    if provider_id == "opencode" or "opencode setup required" in low:
        return "OpenCode needs provider/API-key setup."
    if provider_id == "cursor" or "cursor provider quota" in low:
        return "Cursor-side provider quota/credit limit reached."
    if "insufficient balance" in low:
        return "Provider account balance is insufficient."
    if "api key" in low or "auth" in low or "login" in low or "setup required" in low:
        return raw[:240]
    if "quota" in low or "credit" in low or "rate limit" in low:
        return "Provider quota/credit limit reached."
    return raw[:240]


def _integration_next_action(row: dict, *, provider_type: str) -> str | None:
    result_values = {
        str(row.get("latest_local_live_smoke_result") or ""),
        str(row.get("latest_full_e2e_result") or ""),
    }
    if not any(_status_is_external_blocker(value) for value in result_values):
        return None
    ident = str(row.get("id") or "")
    if provider_type == "mentor_model_provider":
        if ident == "google":
            return "Wait for Gemini quota reset or add billing/credits, then rerun `bash scripts/live_model_provider_smoke.sh google`."
        if ident == "openrouter":
            return "Add OpenRouter credits or choose a smaller model/request size, then rerun `bash scripts/live_model_provider_smoke.sh openrouter`."
        if str(row.get("latest_local_live_smoke_result")) == "skipped_missing_key":
            env_var = row.get("key_env_var") or "PROVIDER_API_KEY"
            return f"Add {env_var}=... to ~/.agent-apprenticeship/.env.local, then rerun the provider smoke."
    if ident == "opencode":
        return "Configure OpenCode's own provider/API key, then rerun `bash scripts/live_agent_provider_smoke.sh opencode`."
    if ident == "cursor":
        return "Resolve the Cursor-side quota/credit limit, then rerun `bash scripts/live_agent_provider_smoke.sh cursor`."
    if ident == "custom" and row.get("custom_user_live_smoke_result") == "skipped_not_configured":
        return "Configure a Custom command template if you want to certify your own custom agent command."
    if str(row.get("latest_local_live_smoke_result")).startswith("skipped_missing_command"):
        return "Install the CLI or expose it on PATH, then rerun the agent smoke."
    return None


def _join_or_none(items: list[str]) -> str:
    return ", ".join(items) if items else "none"


def _print_integration_groups(report: dict) -> None:
    agents = report["apprentice_agents"]
    providers = report["mentor_model_providers"]
    live_agents = [row["id"] for row in agents if _status_is_passed(row.get("latest_local_live_smoke_result"))]
    live_providers = [row["id"] for row in providers if _status_is_passed(row.get("latest_local_live_smoke_result"))]
    live_pairs = []
    blocked = []
    not_configured = []
    framework_failures = []
    for row in agents:
        full = row.get("latest_full_e2e_result")
        smoke = row.get("latest_local_live_smoke_result")
        if _status_is_passed(full):
            live_pairs.append(f"{row['id']}+configured-provider")
        if _status_is_external_blocker(smoke) or _status_is_external_blocker(full):
            target = not_configured if "skipped" in str(smoke) or "skipped" in str(full) else blocked
            reason = row.get("latest_error_summary") or row.get("latest_full_e2e_error_summary") or smoke or full
            target.append(f"{row['id']}: {_concise_integration_error(reason, provider_id=row['id'])}")
        if _status_is_framework_failure(smoke) or _status_is_framework_failure(full):
            reason = row.get("latest_error_summary") or row.get("latest_full_e2e_error_summary") or smoke or full
            framework_failures.append(f"{row['id']}: {_concise_integration_error(reason, provider_id=row['id'])}")
    for row in providers:
        full = row.get("latest_full_e2e_result")
        smoke = row.get("latest_local_live_smoke_result")
        if _status_is_passed(full):
            live_pairs.append(f"custom-or-agent+{row['id']}")
        if _status_is_external_blocker(smoke) or _status_is_external_blocker(full):
            target = not_configured if "skipped" in str(smoke) or "skipped" in str(full) else blocked
            reason = row.get("latest_error_summary") or row.get("latest_full_e2e_error_summary") or smoke or full
            target.append(f"{row['id']}: {_concise_integration_error(reason, provider_id=row['id'])}")
        if _status_is_framework_failure(smoke) or _status_is_framework_failure(full):
            reason = row.get("latest_error_summary") or row.get("latest_full_e2e_error_summary") or smoke or full
            framework_failures.append(f"{row['id']}: {_concise_integration_error(reason, provider_id=row['id'])}")
    typer.echo("Live Verified")
    typer.echo(f"- Apprentice Agents: {_join_or_none(live_agents)}")
    typer.echo(f"- Mentor Model Providers: {_join_or_none(live_providers)}")
    typer.echo(f"- Full E2E: {_join_or_none(live_pairs)}")
    typer.echo("")
    typer.echo("External Setup/Account Blocked")
    if blocked:
        for item in blocked:
            typer.echo(f"- {item}")
    else:
        typer.echo("- none")
    typer.echo("")
    typer.echo("Not Configured")
    if not_configured:
        for item in not_configured:
            typer.echo(f"- {item}")
    else:
        typer.echo("- none")
    typer.echo("")
    typer.echo("Framework/Contract Failures")
    if framework_failures:
        for item in framework_failures:
            typer.echo(f"- {item}")
    else:
        typer.echo("- none")


@app.command('integrations')
def integrations_command():
    _print_integrations_report()


def _print_certification_rows(title: str, rows: list[dict]) -> None:
    typer.echo(title)
    if not rows:
        typer.echo("- not run")
        return
    for row in rows:
        if "provider_id" in row and "agent_id" in row and row.get("certification_kind") == "full_e2e":
            label = f"{row['agent_id']} + {row['provider_id']}"
        else:
            label = row.get("agent_id") or row.get("provider_id") or "unknown"
            if row.get("mode"):
                label = f"{label} ({row['mode']})"
        provider_id = row.get("provider_id") or row.get("agent_id")
        reason = f" - {_concise_integration_error(row['error_summary'], provider_id=provider_id)}" if row.get("error_summary") else ""
        typer.echo(f"- {label}: {row['result']}{reason}")


def _print_certification_groups(report: dict) -> None:
    rows = [*(report.get("agents") or []), *(report.get("models") or []), *(report.get("full_e2e") or [])]
    live = []
    blocked = []
    not_configured = []
    framework_failures = []
    for row in rows:
        if "agent_id" in row and "provider_id" in row and row.get("certification_kind") == "full_e2e":
            label = f"{row['agent_id']}+{row['provider_id']}"
        else:
            label = row.get("agent_id") or row.get("provider_id") or "unknown"
            if row.get("mode"):
                label = f"{label} ({row['mode']})"
        result = str(row.get("result") or "")
        provider_id = row.get("provider_id") or row.get("agent_id")
        reason = _concise_integration_error(row.get("error_summary") or result, provider_id=provider_id)
        if result == "passed":
            live.append(label)
        elif result.startswith("skipped"):
            not_configured.append(f"{label}: {reason}")
        elif _status_is_external_blocker(result):
            blocked.append(f"{label}: {reason}")
        elif _status_is_framework_failure(result):
            framework_failures.append(f"{label}: {reason}")
    typer.echo("Certification Groups")
    typer.echo(f"- Live verified: {_join_or_none(live)}")
    typer.echo("- External setup/account blocked:")
    if blocked:
        for item in blocked:
            typer.echo(f"  - {item}")
    else:
        typer.echo("  - none")
    typer.echo("- Not configured:")
    if not_configured:
        for item in not_configured:
            typer.echo(f"  - {item}")
    else:
        typer.echo("  - none")
    typer.echo("- Framework/contract failures:")
    if framework_failures:
        for item in framework_failures:
            typer.echo(f"  - {item}")
    else:
        typer.echo("  - none")


@app.command('certify-integrations')
def certify_integrations(
    agents: bool=typer.Option(False, '--agents', help='Run live certification for selected Apprentice Agents.'),
    models: bool=typer.Option(False, '--models', help='Run live certification for selected Mentor Model Providers.'),
    full_e2e: bool=typer.Option(False, '--full-e2e', help='Run bounded full E2E integration certification.'),
    all_: bool=typer.Option(False, '--all', help='Run agents, models, and bounded full E2E certification.'),
    all_combinations: bool=typer.Option(False, '--all-combinations', help='Run every Apprentice Agent and Mentor Model Provider pair.'),
    agent: list[str] | None=typer.Option(None, '--agent', help='Limit certification to one Apprentice Agent id. Can be repeated.'),
    model_provider: list[str] | None=typer.Option(None, '--model-provider', help='Limit certification to one Mentor Model Provider id. Can be repeated.'),
    pair: list[str] | None=typer.Option(None, '--pair', help='Limit full E2E to agent+provider. Can be repeated.'),
    as_json: bool=typer.Option(False, '--json', help='Emit certification results as JSON.'),
    strict: bool=typer.Option(False, '--strict', help='Treat skipped integrations as failures.'),
):
    strict = strict or os.getenv("AA_CERTIFY_STRICT") == "1"
    include_agents = agents or all_
    include_models = models or all_
    include_full_e2e = full_e2e or all_
    if not (include_agents or include_models or include_full_e2e):
        include_agents = include_models = include_full_e2e = True
    agent_ids = agent or None
    provider_ids = model_provider or None
    if agent_ids:
        bad = [value for value in agent_ids if value not in WORKER_AGENT_RECIPES]
        if bad:
            raise typer.BadParameter(f'Apprentice Agent must be one of: {", ".join(WORKER_AGENT_RECIPES)}')
    if provider_ids:
        bad = [value for value in provider_ids if value not in MODEL_PROVIDER_RECIPES]
        if bad:
            raise typer.BadParameter(f'Mentor Model Provider must be one of: {", ".join(MODEL_PROVIDER_RECIPES)}')
    full_pairs = []
    for raw_pair in pair or []:
        if "+" not in raw_pair:
            raise typer.BadParameter('E2E pair must use agent+provider, for example codex+openai')
        agent_id, provider_id = [part.strip() for part in raw_pair.split("+", 1)]
        if agent_id not in WORKER_AGENT_RECIPES:
            raise typer.BadParameter(f'Apprentice Agent must be one of: {", ".join(WORKER_AGENT_RECIPES)}')
        if provider_id not in MODEL_PROVIDER_RECIPES:
            raise typer.BadParameter(f'Mentor Model Provider must be one of: {", ".join(MODEL_PROVIDER_RECIPES)}')
        full_pairs.append((agent_id, provider_id))
    report = run_certification_matrix(
        include_agents=include_agents,
        include_models=include_models,
        include_full_e2e=include_full_e2e,
        all_combinations=all_combinations,
        strict=strict,
        agent_ids=agent_ids,
        provider_ids=provider_ids,
        full_e2e_pairs=full_pairs or None,
    )
    if as_json:
        typer.echo(json.dumps(report, indent=2, sort_keys=True))
    else:
        typer.echo("Agent Apprenticeship Live Integration Certification")
        typer.echo("")
        _print_certification_groups(report)
        typer.echo("")
        _print_certification_rows("Apprentice Agents", report.get("agents") or [])
        typer.echo("")
        _print_certification_rows("Mentor Model Providers", report.get("models") or [])
        typer.echo("")
        _print_certification_rows("Full E2E", report.get("full_e2e") or [])
        typer.echo("")
        summary = report.get("summary") or {}
        typer.echo(
            f"Summary: passed={summary.get('passed', 0)} "
            f"failed={summary.get('failed', 0)} skipped={summary.get('skipped', 0)}"
        )
    code = certification_exit_code(report, strict=strict)
    if code:
        raise typer.Exit(code)

@configure_app.command('agent')
def configure_agent(
    agent: str | None=typer.Argument(None, help='Known Apprentice Agent id.'),
    command_path: str | None=typer.Option(None, '--command-path', help='Command path/name for the selected agent.'),
    model: str | None=typer.Option(None, '--model', help='Optional model name passed through to the agent recipe.'),
    display_name: str | None=typer.Option(None, '--display-name', help='Custom Apprentice Agent display name.'),
    command_template: str | None=typer.Option(None, '--command-template', help='Custom command template using {workspace} and {prompt_file}.'),
    can_write_files: bool=typer.Option(True, '--can-write-files/--read-only', help='Whether the custom Apprentice Agent can write files in the workspace.'),
    smoke_test: bool=typer.Option(False, '--smoke-test/--no-smoke-test', help='Check whether the configured command is available.'),
):
    if agent is None:
        typer.echo('Apprentice Agent options:')
        for key, recipe in WORKER_AGENT_RECIPES.items():
            typer.echo(f'- {key}: {recipe.display_name}')
        agent=typer.prompt('Apprentice Agent')
    agent=agent.strip().lower()
    if agent not in WORKER_AGENT_RECIPES:
        raise typer.BadParameter(f'Apprentice Agent must be one of: {", ".join(WORKER_AGENT_RECIPES)}')
    recipe=WORKER_AGENT_RECIPES[agent]
    command=command_path or recipe.command_name
    runner='codex' if agent == 'codex' else agent
    updates=dict(worker_agent=agent, worker_agent_command=command, worker_agent_model=model, worker_runner=runner, reviser_runner=runner)
    if agent == 'custom':
        if command_template is None:
            command_template=typer.prompt('Command template')
        missing=[token for token in ('{workspace}','{prompt_file}') if token not in command_template]
        if missing:
            raise typer.BadParameter('custom Apprentice Agent command template must include placeholders: '+', '.join(missing))
        updates.update(
            custom_worker_display_name=display_name or typer.prompt('Display name', default='Custom'),
            custom_worker_command_template=command_template,
            custom_worker_can_write_files=can_write_files,
            worker_agent_command=command_template.split()[0] if command_path is None else command_path,
        )
    check_command=command
    if agent == 'custom' and command_template:
        try:
            parts=shlex.split(command_template)
        except ValueError:
            parts=command_template.split()
        check_command=parts[0] if parts else command
    update_settings(**updates)
    typer.echo(f'Configured Apprentice Agent: {recipe.display_name if agent != "custom" else (display_name or "Custom")}')
    typer.echo(f'Command: {command}')
    resolved = resolve_command(check_command)
    typer.echo(f'Command found: {_yes_no(bool(resolved))}')
    if resolved and resolved != check_command:
        typer.echo(f'Resolved command: {resolved}')
    typer.echo('Readiness: not tested' if not smoke_test else 'Readiness check: command availability')
    typer.echo(f'Next: AA_RUN_LIVE_AGENT_PROVIDER_SMOKE={agent} bash scripts/live_agent_provider_smoke.sh {agent}')
    if smoke_test:
        ok=bool(resolve_command(check_command))
        update_settings(
            apprentice_agent_readiness_status=("ready" if ok else "missing_command"),
            apprentice_agent_readiness_reason=("Command availability check passed." if ok else f"Apprentice Agent command not found: {check_command}"),
        )
        typer.echo(f'apprentice_agent_command_available={str(ok).lower()}')
        typer.echo(_status_line('Apprentice Agent Status', 'ready' if ok else 'missing_command', None if ok else f"Apprentice Agent command not found: {check_command}"))
        if not ok:
            raise typer.Exit(1)
    else:
        update_settings(apprentice_agent_readiness_status=None, apprentice_agent_readiness_reason=None)

@configure_app.command('settings')
def configure_settings(
    apprenticeship_mode: str | None=typer.Option(None, '--apprenticeship-mode', help='autonomous, expert-led, or org-custom.'),
    mentor_mode: str | None=typer.Option(None, '--mentor-mode', help='Backward-compatible alias for --apprenticeship-mode.', hidden=True),
    sensitive_info_masking: str | None=typer.Option(None, '--sensitive-info-masking', help='standard or no-masking.'),
    max_loops: int | None=typer.Option(None, '--max-loops', help='Maximum improvement loops.'),
):
    updates={}
    if apprenticeship_mode is not None:
        updates['apprenticeship_mode']=_normalize_apprenticeship_mode(apprenticeship_mode)
    if mentor_mode is not None:
        updates['apprenticeship_mode']=_normalize_apprenticeship_mode(mentor_mode)
    if sensitive_info_masking is not None:
        updates['sensitive_info_masking']=_normalize_masking(sensitive_info_masking)
    if max_loops is not None:
        if max_loops < 1:
            raise typer.BadParameter('maximum improvement loops must be at least 1')
        updates['max_improvement_loops']=max_loops
    if not updates:
        typer.echo(public_settings_text())
        return
    update_settings(**updates)
    typer.echo(public_settings_text())


@configure_app.command('apprenticeship')
def configure_apprenticeship(
    mode: str | None=typer.Argument(None, help='autonomous, expert-led, or org-custom.'),
):
    if mode is None:
        typer.echo("Apprenticeship Mode options:")
        typer.echo("- autonomous: Autonomous Apprenticeship")
        typer.echo("- expert-led: Expert-Led Apprenticeship")
        typer.echo("- org-custom: Organization Custom")
        mode = typer.prompt("Apprenticeship Mode")
    normalized = _normalize_apprenticeship_mode(mode)
    updated = update_settings(apprenticeship_mode=normalized)
    typer.echo(f"Apprenticeship Mode: {apprenticeship_mode_display(updated.apprenticeship_mode)}")
    if updated.apprenticeship_mode == "org_custom":
        typer.echo("Organization Custom is available through private setup. Visit https://agentapprenticeship.org/custom")


@configure_app.command('contribution')
def configure_contribution(
    mode: str | None=typer.Argument(None, help='public-ecosystem or private-internal-only.'),
    sharing_behavior: str | None=typer.Option(None, "--sharing-behavior", help="Public Ecosystem behavior: manual or automatic."),
):
    if mode is None:
        typer.echo("Agent Training Contribution Mode options:")
        typer.echo("- public-ecosystem: share completed Experience Compilations to the Forsy ecosystem")
        typer.echo("- private-internal-only: keep completed packages local")
        mode = typer.prompt("Agent Training Contribution Mode")
    normalized = _normalize_training_contribution(mode)
    updates = {"training_contribution_mode": normalized}
    if sharing_behavior is not None:
        behavior = _normalize_auto_share(sharing_behavior)
        updates["ecosystem_auto_share"] = behavior
        if behavior != "disabled":
            updates["training_contribution_mode"] = "public_ecosystem"
    updated = update_settings(**updates)
    typer.echo(f"Agent Training Contribution Mode: {training_contribution_mode_display(updated.training_contribution_mode)}")
    if updated.training_contribution_mode == "public_ecosystem":
        typer.echo(f"Public Ecosystem sharing behavior: {ecosystem_auto_share_display(updated.ecosystem_auto_share)}")

@configure_app.command('model')
def configure_model(
    provider: str | None=typer.Argument(None, help='Known Mentor Model Provider id.'),
    model: str | None=typer.Option(None, '--model', help='Model name for evaluator/grader/verifier roles.'),
    api_key_env_var: str | None=typer.Option(None, '--api-key-env-var', help='Environment variable holding the API key.'),
    test_connection: bool=typer.Option(False, '--test-connection/--no-test-connection', help='Check whether the Mentor Model Provider API key env var is visible.'),
):
    _configure_model_impl(provider, model, api_key_env_var, test_connection)

@app.command('start')
def start(
    asset: list[Path]=typer.Option([], '--asset', '-a', help='Optional task asset file or directory.'),
    runner: str | None=typer.Option(None, '--runner', help='Advanced compatibility runner override.'),
    apprenticeship_mode: str | None=typer.Option(None, '--apprenticeship-mode', help='autonomous, expert-led, or org-custom.'),
    mentor_mode: str | None=typer.Option(None, '--mentor-mode', help='Backward-compatible alias for --apprenticeship-mode.', hidden=True),
    sensitive_info_masking: str | None=typer.Option(None, '--sensitive-info-masking', help='standard or no-masking.'),
    evaluation_mode: str | None=typer.Option(None, '--evaluation-mode', help='Backward-compatible alias for --apprenticeship-mode.', hidden=True),
    data_sharing_level: str | None=typer.Option(None, '--data-sharing-level', help='Backward-compatible alias for --sensitive-info-masking.', hidden=True),
    max_loops: int | None=typer.Option(None, '--max-loops', help='Maximum improvement loops for this run.'),
    quiet: bool=typer.Option(False, '--quiet', help='Print only the final task result.'),
    verbose: bool=typer.Option(False, '--verbose', help='Print additional progress details when available.'),
    json_progress: bool=typer.Option(False, '--json-progress', help='Emit progress events as JSONL.'),
    expert_auto_approve: bool=typer.Option(False, '--expert-auto-approve', help='Test/dev helper: auto-approve expert-led checkpoints.'),
    hybrid_auto_approve: bool=typer.Option(False, '--hybrid-auto-approve', help='Compatibility test/dev helper.', hidden=True),
    expert_interactive: bool=typer.Option(False, '--expert-interactive', help='Prompt for expert-led checkpoint inputs even in non-TTY test harnesses.'),
    hybrid_interactive: bool=typer.Option(False, '--hybrid-interactive', help='Compatibility test/dev helper.', hidden=True),
):
    current_settings = get_settings()
    current_agent = apprentice_agent_readiness_status(current_settings)
    if sys.stdin.isatty() and current_agent.get("status") in {"missing_command", "not_ready"}:
        _first_run_setup(interactive=True)
    instruction=typer.prompt('What should your agent work on?')
    extra=typer.prompt('Add assets? optional', default='', show_default=False)
    assets=list(asset)+_split_asset_prompt(extra)
    settings=_ensure_evaluation_ready(_settings_for_run(apprenticeship_mode, mentor_mode, sensitive_info_masking, max_loops, evaluation_mode, data_sharing_level), interactive=True)
    _ensure_apprentice_agent_ready(settings, runner, interactive=True)
    interactive_checkpoints = _interactive_checkpoint_requested(
        settings,
        expert_auto_approve=expert_auto_approve,
        hybrid_auto_approve=hybrid_auto_approve,
        expert_interactive=expert_interactive,
        hybrid_interactive=hybrid_interactive,
    )
    with _temporary_auto_approve_env(
        expert_auto_approve=expert_auto_approve,
        hybrid_auto_approve=hybrid_auto_approve,
        mentor_interactive_checkpoints=interactive_checkpoints,
    ):
        try:
            run_root,bundle=run_prompt_task(instruction, assets=assets, settings=settings, runner=runner, progress_callback=_progress_callback(quiet, verbose, json_progress))
        except RunInterrupted as exc:
            if not quiet and not json_progress:
                typer.echo("")
                typer.echo("Run interrupted.")
                typer.echo("Run directory:")
                typer.echo(str(exc.run_root))
            raise typer.Exit(130)
    status = read_run_status(run_root)
    if quiet:
        _print_task_result(status, include_contribution_help=False)
    elif not json_progress:
        _print_task_result(status)
    _handle_ecosystem_auto_share(status, quiet=quiet, json_progress=json_progress)

@app.command('run')
def run_prompt(
    instruction: str=typer.Argument(..., help='Task instruction for the Apprentice Agent.'),
    asset: list[Path]=typer.Option([], '--asset', '-a', help='Optional task asset file or directory.'),
    run_id: str | None=typer.Option(None, '--run-id', help='Optional run id/slug.'),
    runner: str | None=typer.Option(None, '--runner', help='Advanced compatibility runner override.'),
    apprenticeship_mode: str | None=typer.Option(None, '--apprenticeship-mode', help='autonomous, expert-led, or org-custom.'),
    mentor_mode: str | None=typer.Option(None, '--mentor-mode', help='Backward-compatible alias for --apprenticeship-mode.', hidden=True),
    sensitive_info_masking: str | None=typer.Option(None, '--sensitive-info-masking', help='standard or no-masking.'),
    evaluation_mode: str | None=typer.Option(None, '--evaluation-mode', help='Backward-compatible alias for --apprenticeship-mode.', hidden=True),
    data_sharing_level: str | None=typer.Option(None, '--data-sharing-level', help='Backward-compatible alias for --sensitive-info-masking.', hidden=True),
    max_loops: int | None=typer.Option(None, '--max-loops', help='Maximum improvement loops for this run.'),
    quiet: bool=typer.Option(False, '--quiet', help='Print only the final task result.'),
    verbose: bool=typer.Option(False, '--verbose', help='Print additional progress details when available.'),
    json_progress: bool=typer.Option(False, '--json-progress', help='Emit progress events as JSONL.'),
    experience_pack: list[str]=typer.Option([], '--experience-pack', help='Experience Pack id/path to apply to this run.'),
    no_experience_packs: bool=typer.Option(False, '--no-experience-packs', help='Do not apply Experience Packs to this run.'),
    use_active_experience_packs: bool=typer.Option(False, '--use-active-experience-packs', help='Apply active Experience Packs to this run.'),
    use_installed_skills: bool=typer.Option(False, '--use-installed-skills', help='Force relevant installed runtime training into this run.'),
    no_installed_skills: bool=typer.Option(False, '--no-installed-skills', help='Do not apply installed runtime training to this run.'),
    expert_auto_approve: bool=typer.Option(False, '--expert-auto-approve', help='Test/dev helper: auto-approve expert-led checkpoints.'),
    hybrid_auto_approve: bool=typer.Option(False, '--hybrid-auto-approve', help='Compatibility test/dev helper.', hidden=True),
    expert_interactive: bool=typer.Option(False, '--expert-interactive', help='Prompt for expert-led checkpoint inputs even in non-TTY test harnesses.'),
    hybrid_interactive: bool=typer.Option(False, '--hybrid-interactive', help='Compatibility test/dev helper.', hidden=True),
):
    settings=_ensure_evaluation_ready(_settings_for_run(apprenticeship_mode, mentor_mode, sensitive_info_masking, max_loops, evaluation_mode, data_sharing_level), interactive=True)
    _ensure_apprentice_agent_ready(settings, runner, interactive=True)
    for pack_id in experience_pack:
        try:
            _, pack = load_pack(pack_id, settings)
        except Exception:
            continue
        status = pack.get("status")
        if status in {"reverted", "removed"}:
            typer.echo(f"Experience Pack {pack.get('pack_id') or pack_id} is {status} and will not be applied.")
    selected_packs, pack_guidance = resolve_packs_for_run(
        experience_pack,
        use_active=use_active_experience_packs,
        no_packs=no_experience_packs,
        settings=settings,
    )
    installed_skills, runtime_guidance = ([], "")
    if not no_installed_skills:
        installed_skills, runtime_guidance = select_runtime_training_for_run(
            instruction,
            settings=settings,
            force=use_installed_skills,
        )
        if installed_skills and not quiet and not json_progress:
            typer.echo("Installed Runtime Training selected:")
            for skill in installed_skills:
                typer.echo(f"- {skill.get('installed_skill_id')}: {skill.get('title')}")
    effective_instruction = instruction.rstrip()
    if pack_guidance:
        effective_instruction += pack_guidance
    if runtime_guidance:
        effective_instruction += runtime_guidance
    if effective_instruction == instruction.rstrip():
        effective_instruction = instruction
    experience_refs = pack_run_refs(selected_packs)
    runtime_refs = installed_skill_run_refs(installed_skills)
    interactive_checkpoints = _interactive_checkpoint_requested(
        settings,
        expert_auto_approve=expert_auto_approve,
        hybrid_auto_approve=hybrid_auto_approve,
        expert_interactive=expert_interactive,
        hybrid_interactive=hybrid_interactive,
    )
    with _temporary_auto_approve_env(
        expert_auto_approve=expert_auto_approve,
        hybrid_auto_approve=hybrid_auto_approve,
        mentor_interactive_checkpoints=interactive_checkpoints,
    ):
        try:
            run_root,bundle=run_prompt_task(
                effective_instruction,
                assets=asset,
                run_id=run_id,
                settings=settings,
                runner=runner,
                progress_callback=_progress_callback(quiet, verbose, json_progress),
                experience_pack_refs=experience_refs,
                runtime_training_refs=runtime_refs,
            )
        except RunInterrupted as exc:
            if not quiet and not json_progress:
                typer.echo("")
                typer.echo("Run interrupted.")
                typer.echo("Run directory:")
                typer.echo(str(exc.run_root))
            raise typer.Exit(130)
    status = read_run_status(run_root)
    if quiet:
        _print_task_result(status, include_contribution_help=False)
    _handle_ecosystem_auto_share(status, quiet=quiet, json_progress=json_progress)

@app.command('continue')
def continue_run(
    run_id: str=typer.Argument(..., help='Run id or path to continue.'),
    followup_instruction: str | None=typer.Argument(None, help='Optional direct follow-up instruction.'),
    asset: list[Path]=typer.Option([], '--asset', '-a', help='Optional follow-up asset file or directory.'),
    run_loop: bool=typer.Option(False, '--run-loop', help='Run another improvement loop after recording the follow-up.'),
    record_only: bool=typer.Option(False, '--record-only', '--no-run-loop', help='Only record the follow-up and update the Experience Compilation.'),
    runner: str | None=typer.Option(None, '--runner', help='Advanced compatibility runner override.'),
    quiet: bool=typer.Option(False, '--quiet', help='Print only the final task result.'),
    verbose: bool=typer.Option(False, '--verbose', help='Print additional progress details when available.'),
    json_progress: bool=typer.Option(False, '--json-progress', help='Emit progress events as JSONL.'),
    expert_interactive: bool=typer.Option(False, '--expert-interactive', help='Prompt for expert-led checkpoint inputs even in non-TTY test harnesses.'),
    hybrid_interactive: bool=typer.Option(False, '--hybrid-interactive', help='Compatibility test/dev helper.', hidden=True),
    expert_auto_approve: bool=typer.Option(False, '--expert-auto-approve', help='Test/dev helper: auto-approve expert-led checkpoints.'),
    hybrid_auto_approve: bool=typer.Option(False, '--hybrid-auto-approve', help='Compatibility test/dev helper.', hidden=True),
):
    settings=get_settings()
    run_root=run_root_for(run_id, settings)
    if not quiet and not json_progress:
        typer.echo(f'Continuing run: {run_root.name}')
    if followup_instruction is None:
        followup_instruction=typer.prompt('Follow-up instruction')
        extra=typer.prompt('Add assets? optional', default='', show_default=False)
        asset=list(asset)+_split_asset_prompt(extra)
        if not run_loop and not record_only:
            run_loop=typer.confirm('Run another improvement loop?', default=False)
    elif not record_only and not run_loop:
        run_loop=True
    if record_only:
        run_loop=False
    if run_loop:
        _ensure_apprentice_agent_ready(settings, runner, interactive=True)
    if not quiet and not json_progress:
        typer.echo(f'Follow-up received: {followup_instruction[:80]}')
        typer.echo(f'Optional assets: {len(asset)}')
    interactive_checkpoints = _interactive_checkpoint_requested(
        settings,
        expert_auto_approve=expert_auto_approve,
        hybrid_auto_approve=hybrid_auto_approve,
        expert_interactive=expert_interactive,
        hybrid_interactive=hybrid_interactive,
    )
    with _temporary_auto_approve_env(
        expert_auto_approve=expert_auto_approve,
        hybrid_auto_approve=hybrid_auto_approve,
        mentor_interactive_checkpoints=interactive_checkpoints,
    ):
        try:
            run_root,bundle=continue_session(run_id, followup_instruction, assets=asset, run_loop=run_loop, settings=settings, runner=runner, progress_callback=_progress_callback(quiet, verbose, json_progress))
        except KeyboardInterrupt:
            previous_status = read_run_status(run_root)
            append_progress_event(
                run_root,
                "followup_interrupted",
                run_id=run_root.name,
                message="Follow-up interrupted by user.",
                phase="interrupted",
                run_status="partial",
                task_status="partial",
                contribution_bundle_path=previous_status.get("contribution_bundle_path"),
                operational_error="Follow-up interrupted by user.",
            )
            if not quiet and not json_progress:
                typer.echo("")
                typer.echo("Follow-up interrupted.")
                if previous_status.get("contribution_bundle_path"):
                    typer.echo("Previous Experience Compilation:")
                    typer.echo(str(previous_status.get("contribution_bundle_path")))
            raise typer.Exit(130)
    status = read_run_status(run_root)
    if quiet:
        _print_task_result(status, followup=True, record_only=not run_loop, include_contribution_help=False)
    _handle_ecosystem_auto_share(status, quiet=quiet, json_progress=json_progress)

@app.command('finish')
def finish(
    run_id: str=typer.Argument(..., help='Run id or path to finish.'),
    quiet: bool=typer.Option(False, '--quiet', help='Print only the final task result.'),
    json_progress: bool=typer.Option(False, '--json-progress', help='Emit progress events as JSONL.'),
    expert_interactive: bool=typer.Option(False, '--expert-interactive', help='Prompt for expert-led final checkpoint inputs even in non-TTY test harnesses.'),
    hybrid_interactive: bool=typer.Option(False, '--hybrid-interactive', help='Compatibility test/dev helper.', hidden=True),
    expert_auto_approve: bool=typer.Option(False, '--expert-auto-approve', help='Test/dev helper: auto-approve expert-led checkpoints.'),
    hybrid_auto_approve: bool=typer.Option(False, '--hybrid-auto-approve', help='Compatibility test/dev helper.', hidden=True),
):
    settings = get_settings()
    interactive_checkpoints = _interactive_checkpoint_requested(
        settings,
        expert_auto_approve=expert_auto_approve,
        hybrid_auto_approve=hybrid_auto_approve,
        expert_interactive=expert_interactive,
        hybrid_interactive=hybrid_interactive,
    )
    with _temporary_auto_approve_env(
        expert_auto_approve=expert_auto_approve,
        hybrid_auto_approve=hybrid_auto_approve,
        mentor_interactive_checkpoints=interactive_checkpoints,
    ):
        run_root,bundle=finish_session(run_id, settings=settings, progress_callback=None if json_progress else _progress_callback(quiet, False, False))
    status=read_run_status(run_root)
    if json_progress:
        for event in read_jsonl(run_root/'progress_events.jsonl'):
            typer.echo(json.dumps(event, sort_keys=True))
    elif quiet:
        _print_session_finished(status, include_contribution_help=False)
    else:
        _print_session_finished(status)
    _handle_ecosystem_auto_share(status, quiet=quiet, json_progress=json_progress)


@app.command('status')
def status(run_id: str=typer.Argument(..., help='Run id or path to inspect.')):
    run_root=run_root_for(run_id, get_settings())
    typer.echo(format_run_status(read_run_status(run_root)))


@app.command('watch', help='Watch live run progress, loop reviews, artifacts, and Experience Compiler status.')
def watch(
    run_id: str=typer.Argument(..., help='Run id or path to watch.'),
    interval: float=typer.Option(1.0, '--interval', help='Polling interval in seconds.'),
    timeout: float | None=typer.Option(None, '--timeout', help='Optional maximum watch time in seconds.'),
):
    run_root=run_root_for(run_id, get_settings())
    watch_progress(run_root, interval_seconds=interval, timeout_seconds=timeout, emit=typer.echo)


def _default_registry_path() -> Path:
    import os
    configured=os.getenv('AA_ECOSYSTEM_REGISTRY')
    if configured:
        return Path(configured).expanduser()
    return get_settings().app_home / 'ecosystem' / 'cache' / 'ecosystem_index.json'


def _configured_ecosystem_repo() -> str | None:
    return os.getenv("AA_ECOSYSTEM_REPO") or get_settings().ecosystem_repo or DEFAULT_PUBLIC_ECOSYSTEM_REPO


def _ecosystem_repo_url(repo: str | None = None) -> str:
    return f"https://github.com/{repo or _configured_ecosystem_repo() or DEFAULT_PUBLIC_ECOSYSTEM_REPO}"


def _configured_ecosystem_repo_path() -> Path | None:
    value = os.getenv("AA_ECOSYSTEM_REPO_PATH")
    if value:
        return Path(value).expanduser()
    return get_settings().ecosystem_repo_path


def _gh_path() -> str | None:
    return shutil.which("gh")


def _run_gh(args: list[str], *, timeout: int = 30) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["gh", *args], text=True, capture_output=True, timeout=timeout)


def _gh_authenticated() -> bool:
    if not _gh_path():
        return False
    try:
        return _run_gh(["auth", "status"], timeout=10).returncode == 0
    except Exception:
        return False


def _ecosystem_status() -> dict:
    repo = _configured_ecosystem_repo()
    repo_path = _configured_ecosystem_repo_path()
    gh_installed = bool(_gh_path())
    gh_authed = _gh_authenticated() if gh_installed else False
    settings = get_settings()
    r2 = r2_status(settings)
    return {
        "repo": repo,
        "repo_url": _ecosystem_repo_url(repo),
        "repo_path": str(repo_path) if repo_path else None,
        "repo_path_exists": bool(repo_path and repo_path.exists()),
        "gh_installed": gh_installed,
        "gh_authenticated": gh_authed,
        "automatic_contribution_ready": bool(repo and gh_installed and gh_authed),
        **r2,
    }


def _ecosystem_auto_share_readiness(repo: str | None = None) -> dict:
    settings = get_settings()
    if settings.ecosystem_storage_backend == "forsy-r2":
        api_url = r2_api_url(settings)
        return {
            "ready": bool(api_url),
            "reason": None if api_url else "R2 ingestion API is not configured",
            "next_action": "apprentice ecosystem configure --r2-api-url <url>" if not api_url else None,
            "repo": _configured_ecosystem_repo(),
            "repo_url": _ecosystem_repo_url(),
            "storage_backend": "forsy-r2",
        }
    configured_repo = repo or _configured_ecosystem_repo() or DEFAULT_PUBLIC_ECOSYSTEM_REPO
    gh_installed = bool(_gh_path())
    gh_authed = _gh_authenticated() if gh_installed else False
    if not configured_repo:
        return {
            "ready": False,
            "reason": "ecosystem repo is not configured",
            "next_action": "apprentice ecosystem configure --repo <owner>/<repo>",
            "repo": configured_repo,
            "repo_url": _ecosystem_repo_url(configured_repo),
            "gh_installed": gh_installed,
            "gh_authenticated": gh_authed,
        }
    if not gh_installed:
        return {
            "ready": False,
            "reason": "GitHub CLI is not installed",
            "next_action": "install GitHub CLI (`gh`)",
            "repo": configured_repo,
            "repo_url": _ecosystem_repo_url(configured_repo),
            "gh_installed": gh_installed,
            "gh_authenticated": gh_authed,
        }
    if not gh_authed:
        return {
            "ready": False,
            "reason": "GitHub CLI is not authenticated",
            "next_action": "gh auth login",
            "repo": configured_repo,
            "repo_url": _ecosystem_repo_url(configured_repo),
            "gh_installed": gh_installed,
            "gh_authenticated": gh_authed,
        }
    return {
        "ready": True,
        "reason": None,
        "next_action": None,
        "repo": configured_repo,
        "repo_url": _ecosystem_repo_url(configured_repo),
        "gh_installed": gh_installed,
        "gh_authenticated": gh_authed,
    }


def _ecosystem_auto_share_status_line(settings=None) -> str:
    s = settings or get_settings()
    readiness = _ecosystem_auto_share_readiness()
    label = training_contribution_mode_display(s.training_contribution_mode)
    if s.training_contribution_mode == "private_internal_only":
        return f"Agent Training Contribution Mode: {label}"
    behavior = ecosystem_auto_share_display(s.ecosystem_auto_share)
    behavior_label = "automatic" if behavior == "automatic" else "manual"
    if readiness["ready"]:
        return f"Agent Training Contribution Mode: {label}\nPublic Ecosystem sharing behavior: {behavior_label}\nPublic Ecosystem upload status: Ready"
    reason = readiness.get("reason") or "not ready"
    return f"Agent Training Contribution Mode: {label}\nPublic Ecosystem sharing behavior: {behavior_label}\nPublic Ecosystem upload status: Not ready - {reason}"


def _public_settings_text_with_ecosystem() -> str:
    settings = get_settings()
    lines = public_settings_text(settings).splitlines()
    status = _ecosystem_status()
    git_line = f"GitHub CLI: {'ready' if status['gh_installed'] and status['gh_authenticated'] else 'not ready'}"
    auto_status = _ecosystem_auto_share_status_line(settings).splitlines()
    out = []
    inserted = False
    for line in lines:
        out.append(line)
        if line.startswith("Ecosystem Repo:"):
            out.append(git_line)
            inserted = True
        if line.startswith("Public Ecosystem Repo:"):
            out.append(f"GitHub URL: {_ecosystem_repo_url(settings.ecosystem_repo)}")
            out.append(git_line)
            inserted = True
        if line.startswith("Agent Training Contribution Mode:"):
            out.pop()
            out.extend(auto_status)
    if not inserted:
        out.append(f"Public Ecosystem Repo: {settings.ecosystem_repo or DEFAULT_PUBLIC_ECOSYSTEM_REPO}")
        out.append(f"GitHub URL: {_ecosystem_repo_url(settings.ecosystem_repo)}")
        out.append(git_line)
        out.extend(auto_status)
    return "\n".join(out)


def _index_from_gh(repo: str) -> list[dict]:
    if not _gh_path():
        raise FileNotFoundError("GitHub CLI `gh` is not installed.")
    if not _gh_authenticated():
        raise FileNotFoundError("GitHub CLI `gh` is not authenticated.")
    last_error = ""
    for index_path in ["ecosystem_index.json", "bundles/index.json"]:
        cp = _run_gh(["api", f"repos/{repo}/contents/{index_path}"], timeout=30)
        if cp.returncode != 0:
            last_error = redact_secrets(cp.stderr or cp.stdout or "")
            continue
        payload = json.loads(cp.stdout or "{}")
        content = payload.get("content") or ""
        encoding = payload.get("encoding")
        if encoding == "base64":
            text = base64.b64decode(content).decode()
        else:
            text = content
        data = json.loads(text or "{}")
        rows = data.get("bundles") if isinstance(data, dict) else data
        return [row for row in rows or [] if isinstance(row, dict)]
    raise FileNotFoundError(
        f"Could not read public ecosystem index from {repo}. "
        f"Expected ecosystem_index.json or bundles/index.json. {last_error}".strip()
    )


def _read_public_url_text(url: str, *, timeout: int = 20) -> str:
    req = urllib.request.Request(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": "agent-apprenticeship-cli",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return response.read().decode("utf-8")


def _github_raw_url(repo: str, path: str, ref: str = "main") -> str:
    quoted = urllib.parse.quote(path.strip("/"), safe="/")
    return f"https://raw.githubusercontent.com/{repo}/{ref}/{quoted}"


def _github_contents_url(repo: str, path: str, ref: str = "main") -> str:
    quoted = urllib.parse.quote(path.strip("/"), safe="/")
    suffix = f"/{quoted}" if quoted else ""
    return f"https://api.github.com/repos/{repo}/contents{suffix}?ref={urllib.parse.quote(ref)}"


def _public_index_cache_path(repo: str) -> Path:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", repo)
    return get_settings().app_home / "ecosystem" / "public_index_cache" / f"{slug}.json"


def _load_public_index_cache(repo: str) -> list[dict] | None:
    path = _public_index_cache_path(repo)
    if not path.exists():
        return None
    try:
        data = read_json(path)
    except Exception:
        return None
    rows = data.get("rows") if isinstance(data, dict) else None
    if not isinstance(rows, list):
        return None
    return [row for row in rows if isinstance(row, dict)]


def _write_public_index_cache(repo: str, rows: list[dict]) -> None:
    path = _public_index_cache_path(repo)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(
        path,
        {
            "kind": "public_ecosystem_index_cache",
            "repo": repo,
            "repo_url": _ecosystem_repo_url(repo),
            "cached_at": datetime.now(timezone.utc).isoformat(),
            "rows": rows,
        },
    )


def _rows_from_index_text(text: str, *, suffix: str) -> list[dict]:
    if suffix == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    data = json.loads(text or "{}")
    if isinstance(data, list):
        return [row for row in data if isinstance(row, dict)]
    if isinstance(data, dict):
        rows = (
            data.get("bundles")
            or data.get("items")
            or data.get("entries")
            or data.get("contributions")
            or []
        )
        return [row for row in rows if isinstance(row, dict)]
    return []


def _index_from_public_repo_url(repo: str) -> list[dict]:
    rows: list[dict] = []
    errors: list[str] = []
    for path in [
        "seed_dataset/ecosystem_registry.jsonl",
        "seed_dataset/ecosystem_registry.json",
        "ecosystem/contributions/index.json",
        "ecosystem/contributions/index.jsonl",
    ]:
        try:
            text = _read_public_url_text(_github_raw_url(repo, path), timeout=20)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            errors.append(f"{path}: {exc}")
            continue
        for row in _rows_from_index_text(text, suffix=Path(path).suffix):
            out = dict(row)
            out.setdefault("public_repo_slug", repo)
            out.setdefault("public_repo_url", _ecosystem_repo_url(repo))
            if path.startswith("seed_dataset/"):
                out.setdefault("kind", "seed_task")
                out.setdefault("experience_source_type", "seed_task")
            else:
                out.setdefault("kind", "contribution_bundle")
                out.setdefault("experience_source_type", "contribution_bundle")
            rows.append(out)
    if not rows:
        detail = "; ".join(errors[:3])
        raise FileNotFoundError(
            f"Could not read public ecosystem index from https://github.com/{repo}. {detail}".strip()
        )
    deduped: list[dict] = []
    seen: set[str] = set()
    for row in rows:
        key = str(row.get("bundle_id") or row.get("seed_task_id") or row.get("task_id") or "")
        if key and key in seen:
            continue
        if key:
            seen.add(key)
        deduped.append(row)
    _write_public_index_cache(repo, deduped)
    return deduped


def _download_public_repo_path(repo: str, rel_path: str, dest: Path) -> None:
    data = json.loads(_read_public_url_text(_github_contents_url(repo, rel_path), timeout=20))
    if isinstance(data, list):
        dest.mkdir(parents=True, exist_ok=True)
        for item in data:
            item_type = item.get("type")
            item_path = item.get("path")
            item_name = item.get("name")
            if not item_path or not item_name:
                continue
            if item_type == "dir":
                _download_public_repo_path(repo, item_path, dest / item_name)
            elif item_type == "file":
                download_url = item.get("download_url") or _github_raw_url(repo, item_path)
                target = dest / item_name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(_read_public_url_text(download_url, timeout=20))
        return
    if isinstance(data, dict) and data.get("type") == "file":
        download_url = data.get("download_url") or _github_raw_url(repo, data.get("path") or rel_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(_read_public_url_text(download_url, timeout=20))
        return
    raise FileNotFoundError(f"Could not download public ecosystem path: {rel_path}")


def _load_json_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    if path.suffix == ".jsonl":
        return [row for row in read_jsonl(path) if isinstance(row, dict)]
    data = read_json(path)
    if isinstance(data, list):
        return [row for row in data if isinstance(row, dict)]
    if isinstance(data, dict):
        rows = (
            data.get("bundles")
            or data.get("items")
            or data.get("entries")
            or data.get("contributions")
            or []
        )
        return [row for row in rows if isinstance(row, dict)]
    return []


def _absolutize_public_repo_paths(row: dict, repo_path: Path) -> dict:
    out = dict(row)
    path_keys = {
        "task_path",
        "task_packet_path",
        "rubric_path",
        "worker_visible_rubric_path",
        "verifier_private_rubric_path",
        "attempts_path",
        "evaluation_path",
        "learning_signals_path",
        "local_bundle_path",
        "bundle_path",
        "bundle_path_or_url",
    }
    for key in path_keys:
        value = out.get(key)
        if value:
            candidate = Path(str(value)).expanduser()
            out[key] = str(candidate if candidate.is_absolute() else repo_path / candidate)
    if isinstance(out.get("trace_paths"), list):
        traces = []
        for value in out["trace_paths"]:
            candidate = Path(str(value)).expanduser()
            traces.append(str(candidate if candidate.is_absolute() else repo_path / candidate))
        out["trace_paths"] = traces
    return out


def _index_from_public_repo_path(repo_path: Path) -> list[dict]:
    repo_path = repo_path.expanduser()
    if not repo_path.exists():
        raise FileNotFoundError(f"Public ecosystem repo path not found: {repo_path}")
    rows: list[dict] = []
    seed_root = repo_path / "seed_dataset"
    if seed_root.exists():
        for path in [seed_root / "ecosystem_registry.jsonl", seed_root / "ecosystem_registry.json"]:
            rows.extend(_absolutize_public_repo_paths(row, repo_path) for row in _load_json_rows(path))
    contributions_root = repo_path / "ecosystem" / "contributions"
    if contributions_root.exists():
        for path in [contributions_root / "index.json", contributions_root / "index.jsonl"]:
            for row in _load_json_rows(path):
                bundle_id = row.get("bundle_id")
                out = dict(row)
                out.setdefault("kind", "contribution_bundle")
                out.setdefault("experience_source_type", "contribution_bundle")
                if bundle_id and not any(out.get(key) for key in ["local_bundle_path", "bundle_path", "bundle_path_or_url"]):
                    bundle_path = contributions_root / "bundles" / str(bundle_id)
                    if bundle_path.exists():
                        out["local_bundle_path"] = str(bundle_path)
                rows.append(_absolutize_public_repo_paths(out, repo_path))
    deduped: list[dict] = []
    seen: set[str] = set()
    for row in rows:
        key = str(row.get("bundle_id") or row.get("seed_task_id") or row.get("task_id") or "")
        if key and key in seen:
            continue
        if key:
            seen.add(key)
        deduped.append(row)
    return deduped


def _load_registry_file(path: Path) -> list[dict]:
    data = read_json(path)
    if isinstance(data, dict) and data.get("kind") == "public_ecosystem_index":
        repo_path = path.parent.parent if path.parent.name == "ecosystem" else path.parent
        return _index_from_public_repo_path(repo_path)
    if isinstance(data, dict):
        rows = (
            data.get('bundles')
            or data.get('items')
            or data.get('entries')
            or data.get('contributions')
            or []
        )
    else:
        rows = data
    return [row for row in rows if isinstance(row, dict)]


def _load_registry(registry: Path | None=None, storage: str | None = None) -> list[dict]:
    if registry is None:
        backend = normalize_ecosystem_storage(storage or get_settings().ecosystem_storage_backend)
        if backend == "forsy-r2":
            try:
                return fetch_r2_index(get_settings())
            except FileNotFoundError as exc:
                raise FileNotFoundError(f"{exc} Fallback: apprentice ecosystem list --storage github") from exc
            except Exception as exc:
                raise FileNotFoundError(f"R2 ecosystem index could not be read. {redact_secrets(str(exc))} Fallback: apprentice ecosystem list --storage github") from exc
        repo_path = _configured_ecosystem_repo_path()
        if repo_path:
            return _index_from_public_repo_path(repo_path)
        repo = _configured_ecosystem_repo()
        if repo:
            cached = _load_public_index_cache(repo)
            if cached:
                return cached
            try:
                return _index_from_public_repo_url(repo)
            except FileNotFoundError as exc:
                try:
                    return _index_from_gh(repo)
                except FileNotFoundError as gh_exc:
                    seed_rows = search_learning_sources(None)
                    if repo == DEFAULT_PUBLIC_ECOSYSTEM_REPO and seed_rows:
                        return seed_rows
                    raise FileNotFoundError(
                        f"Public ecosystem repo is configured as {repo}, but the index could not be read. {exc} {gh_exc}"
                    )
        path=_default_registry_path()
        if not path.exists():
            seed_rows = search_learning_sources(None)
            if seed_rows:
                return seed_rows
            raise FileNotFoundError(
                f"Public ecosystem repo is {DEFAULT_PUBLIC_ECOSYSTEM_REPO}, but no readable public index cache was found. "
                "Run `apprentice ecosystem configure --repo <owner>/<repo>` to override it, or pass "
                "`--registry <ecosystem_index.json>` for a local public-index file."
            )
    else:
        path=registry
    if not path.exists():
        raise FileNotFoundError(
            f'Public ecosystem index not found: {path}. '
            'Run `apprentice ecosystem guide` for setup steps.'
        )
    return _load_registry_file(path)


def _row_id(row: dict) -> str:
    return str(row.get("id") or row.get("bundle_id") or row.get("package_id") or row.get("seed_task_id") or row.get("task_id") or "")


def _row_values(row: dict, *keys: str) -> list[str]:
    values: list[str] = []
    for key in keys:
        value = row.get(key)
        if isinstance(value, (list, tuple, set)):
            values.extend(str(item) for item in value if item not in (None, ""))
        elif isinstance(value, dict):
            values.extend(str(item) for item in value.values() if item not in (None, ""))
        elif value not in (None, ""):
            values.append(str(value))
    return values


def _row_contains(row: dict, expected: str, *keys: str) -> bool:
    needle = str(expected or "").strip().lower()
    return bool(needle) and any(needle == value.lower() or needle in value.lower() for value in _row_values(row, *keys))


def _matches_registry_row(row: dict, query: str | None, filters: dict) -> bool:
    if filters.get("mine"):
        identity = get_or_create_ecosystem_identity(get_settings())
        contributor = row.get("contributor") or {}
        if str(contributor.get("ecosystem_user_id") or "") != str(identity.get("ecosystem_user_id")):
            return False
    if query:
        haystack=' '.join(_row_values(
            row,
            'id', 'item_id', 'bundle_id', 'package_id', 'experience_compilation_id',
            'title', 'task_title', 'summary', 'agent_apprentice_role',
            'expected_economic_value', 'economic_value_summary',
            'domain', 'domains', 'subdomain', 'subdomains', 'tags', 'surfaces', 'tools',
        ))
        if query.lower() not in haystack.lower():
            return False
    if filters.get("apprenticeship_mode"):
        expected = _normalize_apprenticeship_mode(str(filters["apprenticeship_mode"]))
        actual = row.get("apprenticeship_mode")
        if not actual and row.get("mentor_mode"):
            actual = normalize_apprenticeship_mode(str(row.get("mentor_mode")))
        if not actual:
            return False
        if normalize_apprenticeship_mode(str(actual or "")) != expected:
            return False
    for key in ['title','task_status','run_status']:
        value=filters.get(key)
        if value and str(row.get(key) or '').lower() != str(value).lower():
            return False
    if filters.get("mentor_mode"):
        expected = _normalize_apprenticeship_mode(str(filters["mentor_mode"]))
        actual = row.get("apprenticeship_mode") or row.get("mentor_mode")
        if not actual:
            return False
        if normalize_apprenticeship_mode(str(actual or "")) != expected:
            return False
    for key, fields in [('domain', ('domain', 'domains')), ('subdomain', ('subdomain', 'subdomains')), ('tag', ('tags',))]:
        value=filters.get(key)
        if value and not _row_contains(row, str(value), *fields):
            return False
    if filters.get('min_traced_steps') is not None and int(row.get('traced_steps') or 0) < filters['min_traced_steps']:
        return False
    if filters.get('max_traced_steps') is not None and int(row.get('traced_steps') or 0) > filters['max_traced_steps']:
        return False
    if filters.get('expected_economic_value') and str(filters['expected_economic_value']).lower() not in str(row.get('expected_economic_value') or '').lower():
        return False
    if filters.get('artifact_type'):
        types=[str(v).lower() for v in row.get('artifact_types') or row.get('artifacts') or []]
        if str(filters['artifact_type']).lower() not in types:
            return False
    if filters.get('apprentice_agent'):
        values=[
            row.get('apprentice_agent'),
            row.get('agent'),
            row.get('worker_agent'),
        ]
        if str(filters['apprentice_agent']).lower() not in [str(v).lower() for v in values if v]:
            return False
    if filters.get('mentor_model_provider'):
        values=[
            row.get('mentor_model_provider'),
            row.get('model_provider'),
            row.get('provider'),
        ]
        if str(filters['mentor_model_provider']).lower() not in [str(v).lower() for v in values if v]:
            return False
    return True


def _select_registry_row(rows: list[dict]) -> None:
    if not sys.stdin.isatty():
        typer.echo("Selection requires an interactive terminal. Use `--json` to select an item non-interactively.")
        return
    choice = typer.prompt("Choose Experience Compilation", default="1").strip()
    try:
        index = int(choice)
    except ValueError:
        typer.echo("Selection cancelled.")
        return
    if index < 1 or index > len(rows):
        typer.echo("Selection cancelled.")
        return
    row = rows[index - 1]
    item_id = _row_id(row)
    typer.echo("")
    typer.echo("Choose action:")
    typer.echo("1. Export Full Experience Compilation")
    typer.echo("2. Install Runtime Training")
    action = typer.prompt("Action", default="1").strip()
    if action == "1":
        output = typer.prompt("Output path", default=str(get_settings().app_home / "ecosystem" / "exports" / item_id)).strip()
        package_root = _package_for_ecosystem_item(item_id)
        exported = _copy_or_zip_package(package_root, Path(output), item_id=item_id, source="ecosystem_select")
        typer.echo(f"Export Full Experience Compilation complete: {exported}")
    elif action == "2":
        package_root = _package_for_ecosystem_item(item_id)
        manifest = install_runtime_training_from_bundle(item_id, package_root, settings=get_settings())
        typer.echo("Install Runtime Training complete.")
        typer.echo(f"installed_skill_id: {manifest.get('installed_skill_id')}")
    else:
        typer.echo("Selection cancelled.")


def _print_registry_rows(rows: list[dict], *, json_output: bool = False, select: bool = False) -> None:
    if json_output:
        typer.echo(json.dumps({"items": sanitize_public_obj(rows)}, indent=2, sort_keys=True))
        return
    if not rows:
        typer.echo('No Experience Compilations matched.')
        return
    for index, row in enumerate(rows, 1):
        typer.echo(_format_registry_row(row, index=index))
    if select:
        _select_registry_row(rows)


def _is_seed_registry_row(row: dict) -> bool:
    return bool(row.get("seed_task_id") or row.get("task_packet_path") or row.get("learning_signals_path"))


def _format_count(value) -> str:
    return str(int(value)) if isinstance(value, (int, float)) or str(value).isdigit() else "0"


def _format_seed_registry_row(row: dict, *, index: int | None = None) -> str:
    bundle_id = row.get("bundle_id") or row.get("seed_task_id") or row.get("task_id")
    status = row.get("task_status") or "ready_to_run"
    run_status = row.get("run_status") or "not_run"
    trace_count = row.get("trace_count")
    if trace_count is None:
        trace_count = len(row.get("trace_paths") or [])
    attempt_count = row.get("attempt_count") or row.get("attempts") or 0
    learning_count = row.get("training_signal_rows") or row.get("lessons_count") or 0
    learning_label = f"learning signals: {_format_count(learning_count)}" if learning_count else "learning signals: available"
    details = [
        f"{status} / {run_status}",
        f"traces: {_format_count(trace_count)}",
        f"attempts: {_format_count(attempt_count)}",
    ]
    if row.get("process_supervision_rows") is not None:
        details.append(f"process rows: {_format_count(row.get('process_supervision_rows'))}")
    if row.get("reward_modeling_rows") is not None:
        details.append(f"reward rows: {_format_count(row.get('reward_modeling_rows'))}")
    if row.get("revision_preference_pairs") is not None:
        details.append(f"preferences: {_format_count(row.get('revision_preference_pairs'))}")
    details.append(learning_label)
    prefix = f"{index}. " if index is not None else ""
    return f"{prefix}{bundle_id}: {row.get('title')} [{' · '.join(details)}]"


def _format_registry_row(row: dict, *, index: int | None = None) -> str:
    if _is_seed_registry_row(row):
        return _format_seed_registry_row(row, index=index)
    details = [
        str(row.get("task_status") or row.get("run_status") or "status_unknown"),
    ]
    mode = row.get("apprenticeship_mode") or row.get("mentor_mode")
    if mode:
        details.append(f"apprenticeship_mode={apprenticeship_mode_display(normalize_apprenticeship_mode(str(mode)))}")
    traces = row.get("traced_steps")
    if traces is not None:
        details.append(f"traces: {_format_count(traces)}")
    artifacts = row.get("artifact_count")
    if artifacts is not None:
        details.append(f"artifacts: {_format_count(artifacts)}")
    summary = str(row.get("summary") or "").strip()
    if len(summary) > 180:
        summary = summary[:177].rstrip() + "..."
    prefix = f"{index}. " if index is not None else ""
    return "\n".join([
        f"{prefix}{row.get('title') or _row_id(row)}",
        f"   item_id: {_row_id(row)}",
        f"   domain: {row.get('domain') or 'unknown'}",
        f"   subdomain: {row.get('subdomain') or 'unknown'}",
        f"   expected economic value: {row.get('expected_economic_value') or 'unknown'}",
        f"   summary: {summary or 'No summary provided.'}",
        f"   details: {' · '.join(details)}",
        "   actions: 1. Export Full Experience Compilation  2. Install Runtime Training",
    ])


def _safe_bundle_path(path: Path) -> Path:
    bundle = path.expanduser()
    manifest = bundle / "contribution_manifest.json"
    if not manifest.exists():
        raise FileNotFoundError(f"Experience Compilation manifest not found: {manifest}")
    from .io import validate_trace_tree
    validate_trace_tree(bundle)
    return bundle


def _artifact_index(bundle: Path) -> list[dict]:
    path = bundle / "outputs" / "artifacts_index.json"
    if not path.exists():
        return []
    data = read_json(path)
    return data if isinstance(data, list) else []


def _session_events(bundle: Path) -> list[dict]:
    return read_jsonl(bundle / "session_events.jsonl")


def _session_metadata(bundle: Path) -> dict:
    path = bundle / "session_metadata.json"
    return read_json(path) if path.exists() else {}


def _text_file_has_secret(path: Path) -> bool:
    if path.stat().st_size > 5_000_000:
        return False
    try:
        return contains_secret(path.read_text(errors="ignore"))
    except UnicodeDecodeError:
        return False


def _bundle_secret_hits(bundle: Path) -> list[str]:
    hits = []
    for path in bundle.rglob("*"):
        if path.is_file() and path.name not in {".env", ".env.local"} and _text_file_has_secret(path):
            hits.append(str(path.relative_to(bundle)))
    return hits


def _bundle_submission_metadata(bundle: Path, package_name: str | None = None, package_hash: str | None = None) -> dict:
    manifest = read_json(bundle / "contribution_manifest.json")
    session = _session_metadata(bundle)
    artifacts = _artifact_index(bundle)
    events = _session_events(bundle)
    followups = [event for event in events if event.get("event_type") == "user_followup"]
    created_at = datetime.now(timezone.utc).isoformat()
    metadata = {
        "bundle_id": manifest.get("bundle_id"),
        "title": manifest.get("title"),
        "task_status": manifest.get("task_status"),
        "run_status": manifest.get("run_status"),
        "apprentice_agent": session.get("apprentice_agent"),
        "apprenticeship_mode": manifest.get("apprenticeship_mode") or session.get("apprenticeship_mode"),
        "mentor_mode": manifest.get("mentor_mode"),
        "mentor_model_provider": session.get("model_provider"),
        "domains": manifest.get("domains") or [],
        "subdomains": manifest.get("subdomains") or [],
        "attempts": manifest.get("attempts"),
        "traced_steps": manifest.get("traced_steps"),
        "artifact_count": len(artifacts),
        "follow_up_count": len(followups),
        "process_supervision_rows": manifest.get("process_supervision_rows"),
        "reward_modeling_rows": manifest.get("reward_modeling_rows"),
        "revision_preference_pairs": manifest.get("revision_preference_pairs"),
        "expected_economic_value": manifest.get("expected_economic_value"),
        "expected_economic_value_for_agent_apprentice": manifest.get("expected_economic_value_for_agent_apprentice"),
        "created_at": created_at,
        "package_name": package_name,
        "package_hash": package_hash,
    }
    forbidden = {
        "reviewer",
        "review_decision",
        "accepted",
        "rejected",
        "quality_score",
        "featured",
        "curated",
        "source_url_or_ref",
        "source_kind",
        "source_url",
        "source_ref",
        "source_license",
        "source_release_schema_version",
        "trace_context",
    }
    return {key: value for key, value in metadata.items() if key not in forbidden}


def _submission_summary(metadata: dict, bundle: Path, package_zip: Path | None = None) -> str:
    lines = [
        "# Agent Apprenticeship Ecosystem Submission",
        "",
        "## ecosystem_submission.json summary",
        "",
        f"Bundle ID: {metadata.get('bundle_id')}",
        f"Title: {metadata.get('title')}",
        f"Task Status: {metadata.get('task_status')}",
        f"Run Status: {metadata.get('run_status')}",
        f"Apprentice Agent: {metadata.get('apprentice_agent') or 'unknown'}",
        f"Apprenticeship Mode: {apprenticeship_mode_display(metadata.get('apprenticeship_mode') or metadata.get('mentor_mode'))}",
        f"Mentor Model Provider: {metadata.get('mentor_model_provider') or 'none'}",
        f"Attempts: {metadata.get('attempts')}",
        f"Traced Steps: {metadata.get('traced_steps')}",
        f"Process Supervision Rows: {metadata.get('process_supervision_rows')}",
        f"Reward Modeling Rows: {metadata.get('reward_modeling_rows')}",
        f"Revision Preference Pairs: {metadata.get('revision_preference_pairs')}",
        f"Artifacts: {metadata.get('artifact_count')}",
        f"Follow-ups: {metadata.get('follow_up_count')}",
        f"Domains: {', '.join(metadata.get('domains') or []) or 'none'}",
        f"Subdomains: {', '.join(metadata.get('subdomains') or []) or 'none'}",
        f"Expected Economic Value: {metadata.get('expected_economic_value') or 'unknown'}",
        f"Expected Economic Value for Agent Apprentice: {metadata.get('expected_economic_value_for_agent_apprentice') or 'unknown'}",
        "",
        "## Bundle sharing",
        "",
        "This issue records the public ecosystem contribution metadata. Bundle files are packaged locally by Agent Apprenticeship v0.",
        f"Add the bundle under ecosystem/contributions/bundles/{metadata.get('bundle_id')}/.",
        "Update ecosystem/contributions/index.json or ecosystem/contributions/index.jsonl with the public metadata entry.",
        "Attach or share the generated submission package when using the GitHub issue path.",
    ]
    if package_zip:
        lines.append(f"Submission Package Name: {metadata.get('package_name') or package_zip.name}")
    if metadata.get("package_hash"):
        lines.append(f"Submission Package Hash: {metadata.get('package_hash')}")
    return "\n".join(lines) + "\n"


def _zip_dir(src: Path, dst: Path) -> str:
    if dst.exists():
        dst.unlink()
    with zipfile.ZipFile(dst, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(src.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(src.parent))
    digest = hashlib.sha256(dst.read_bytes()).hexdigest()
    return f"sha256:{digest}"


def _sanitize_submission_copy(root: Path) -> None:
    text_suffixes = {
        ".json",
        ".jsonl",
        ".md",
        ".txt",
        ".yaml",
        ".yml",
        ".csv",
        ".tsv",
        ".log",
    }
    for path in root.rglob("*"):
        if not path.is_file() or path.stat().st_size > 5_000_000:
            continue
        if path.suffix.lower() not in text_suffixes:
            continue
        raw = path.read_text(errors="ignore")
        if path.suffix.lower() == ".json":
            try:
                data = json.loads(raw or "{}")
            except Exception:
                path.write_text(sanitize_public_text(raw) or "")
            else:
                if isinstance(data, dict):
                    write_json(path, sanitize_public_obj(data))
                elif isinstance(data, list):
                    path.write_text(json.dumps(sanitize_public_obj({"items": data})["items"], indent=2, sort_keys=True) + "\n")
                else:
                    path.write_text(sanitize_public_text(raw) or "")
        elif path.suffix.lower() == ".jsonl":
            rows = []
            for line in raw.splitlines():
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                except Exception:
                    rows.append(json.dumps(sanitize_public_text(line) or "", sort_keys=True))
                else:
                    if isinstance(data, dict):
                        rows.append(json.dumps(sanitize_public_obj(data), sort_keys=True))
                    else:
                        rows.append(json.dumps(sanitize_public_obj({"item": data})["item"], sort_keys=True))
            path.write_text("\n".join(rows) + ("\n" if rows else ""))
        else:
            path.write_text(sanitize_public_text(raw) or "")


def _create_ecosystem_submission(bundle: Path, *, block_secret_like_values: bool = False) -> tuple[Path, Path, dict]:
    bundle = _safe_bundle_path(bundle)
    secret_hits = _bundle_secret_hits(bundle)
    if block_secret_like_values and secret_hits:
        raise RuntimeError(
            "Obvious secret-like values were found in the Experience Compilation. "
            "Remove or regenerate the affected files before public contribution: "
            + ", ".join(secret_hits[:10])
        )
    submission_dir = bundle.parent / "ecosystem_submission"
    if submission_dir.exists():
        shutil.rmtree(submission_dir)
    submission_dir.mkdir(parents=True)
    shutil.copytree(
        bundle,
        submission_dir / "contribution_bundle",
        ignore=shutil.ignore_patterns(".env", ".env.local", "__pycache__", "*.pyc"),
    )
    _sanitize_submission_copy(submission_dir / "contribution_bundle")
    package_zip = bundle.parent / f"{bundle.name}_ecosystem_submission.zip"
    metadata = _bundle_submission_metadata(bundle, package_name=package_zip.name)
    write_json(submission_dir / "ecosystem_submission.json", metadata)
    (submission_dir / "SUMMARY.md").write_text(_submission_summary(metadata, bundle))
    package_hash = _zip_dir(submission_dir, package_zip)
    metadata = _bundle_submission_metadata(bundle, package_name=package_zip.name, package_hash=package_hash)
    write_json(submission_dir / "ecosystem_submission.json", metadata)
    (submission_dir / "SUMMARY.md").write_text(_submission_summary(metadata, bundle, package_zip))
    return submission_dir, package_zip, metadata


def _ecosystem_contribution_record(
    *,
    metadata: dict,
    auto_share_mode: str,
    contribution_created: bool,
    contribution_url: str | None,
    contribution_method: str,
    skipped_reason: str | None,
) -> dict:
    return {
        "bundle_id": metadata.get("bundle_id"),
        "auto_share_mode": auto_share_mode,
        "training_contribution_mode": auto_share_mode,
        "contribution_created": contribution_created,
        "contribution_url": contribution_url,
        "contribution_method": contribution_method,
        "skipped_reason": skipped_reason,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def _write_ecosystem_contribution_record(bundle: Path, record: dict) -> None:
    bundle = Path(bundle)
    write_json(bundle / "ecosystem_contribution.json", record)
    if bundle.name == "contribution_bundle":
        write_json(bundle.parent / "ecosystem_contribution.json", record)


def _ecosystem_contribute_impl(
    bundle_path: Path,
    *,
    repo: str | None = None,
    dry_run: bool = False,
    auto_share_mode: str = "manual",
    storage: str | None = None,
    qa: bool = False,
) -> dict:
    bundle = _safe_bundle_path(bundle_path)
    qa = qa or str(os.environ.get("AA_R2_QA_UPLOAD") or "").strip().lower() in {"1", "true", "yes", "on"}
    submission_dir, package_zip, metadata = _create_ecosystem_submission(bundle)
    target_repo = repo or _configured_ecosystem_repo()
    settings = get_settings()
    storage_backend = normalize_ecosystem_storage(storage or settings.ecosystem_storage_backend or DEFAULT_ECOSYSTEM_STORAGE_BACKEND)
    issue_body = _gh_issue_body(metadata, submission_dir, package_zip)
    result = {
        "bundle": bundle,
        "submission_dir": submission_dir,
        "package_zip": package_zip,
        "metadata": metadata,
        "issue_body": issue_body,
        "target_repo": target_repo,
        "storage_backend": storage_backend,
        "storage_ref": None,
        "ecosystem_item": None,
        "upload_response": None,
        "contribution_created": False,
        "contribution_url": None,
        "contribution_method": "manual_package",
        "skipped_reason": None,
        "gh_error": None,
    }

    if storage_backend == "forsy-r2":
        identity = get_or_create_ecosystem_identity(settings)
        item = ecosystem_item_from_bundle(bundle, metadata, settings=settings, identity=identity, source="qa" if qa else "contribution")
        storage_ref = storage_ref_for_item(item, settings=settings, dry_run=dry_run)
        write_storage_ref(bundle, submission_dir, storage_ref)
        _zip_dir(submission_dir, package_zip)
        result["ecosystem_item"] = item
        result["storage_ref"] = storage_ref
        result["contribution_method"] = "r2_worker_ingestion" if not dry_run else "r2_worker_dry_run"
        if dry_run:
            result["skipped_reason"] = "dry run"
        else:
            api_url = r2_api_url(settings)
            if not api_url:
                result["skipped_reason"] = "R2 ingestion API is not configured"
                result["contribution_method"] = "skipped"
            else:
                response = upload_contribution_to_worker(
                    api_url=api_url,
                    package_zip=package_zip,
                    ecosystem_item=item,
                    manifest=read_json(bundle / "contribution_manifest.json"),
                    storage_ref=storage_ref,
                    qa=qa,
                )
                result["upload_response"] = response
                result["contribution_created"] = response.get("status") == "ok"
                result["contribution_url"] = (
                    (response.get("storage") or {}).get("url")
                    or storage_ref.get("url")
                    or (api_url.rstrip("/") + f"/v1/ecosystem/items/{item.get('id')}")
                )
                response_storage = response.get("storage") if isinstance(response.get("storage"), dict) else {}
                if response_storage:
                    item["storage"] = {**(item.get("storage") or {}), **response_storage}
                uploaded_ref = storage_ref_for_item(
                    item,
                    settings=settings,
                    dry_run=False,
                    uploaded_at=datetime.now(timezone.utc).isoformat(),
                    url=result["contribution_url"],
                )
                write_storage_ref(bundle, submission_dir, uploaded_ref)
                _zip_dir(submission_dir, package_zip)
                result["storage_ref"] = uploaded_ref
                append_successful_share(
                    item=item,
                    bundle=bundle,
                    storage_backend="forsy-r2",
                    url=result["contribution_url"],
                    sharing_mode=auto_share_mode,
                    settings=settings,
                )
        record = _ecosystem_contribution_record(
            metadata=metadata,
            auto_share_mode=auto_share_mode,
            contribution_created=bool(result["contribution_created"]),
            contribution_url=result.get("contribution_url"),
            contribution_method=str(result["contribution_method"]),
            skipped_reason=result.get("skipped_reason"),
        )
        result["record"] = record
        _write_ecosystem_contribution_record(bundle, record)
        return result

    if target_repo and _gh_path() and _gh_authenticated() and not dry_run:
        title = f"Experience Compilation: {metadata.get('bundle_id')}"
        cp = _run_gh(["issue", "create", "--repo", target_repo, "--title", title, "--body-file", str(issue_body)], timeout=60)
        if cp.returncode == 0:
            result["contribution_created"] = True
            result["contribution_url"] = (cp.stdout or "").strip()
            result["contribution_method"] = "github_issue"
        else:
            result["skipped_reason"] = "GitHub issue creation failed"
            result["gh_error"] = redact_secrets(cp.stderr or cp.stdout or "")
    elif target_repo and _gh_path() and _gh_authenticated() and dry_run:
        result["skipped_reason"] = "dry run"
    else:
        readiness = _ecosystem_auto_share_readiness(target_repo)
        result["skipped_reason"] = readiness.get("reason")
    record = _ecosystem_contribution_record(
        metadata=metadata,
        auto_share_mode=auto_share_mode,
        contribution_created=bool(result["contribution_created"]),
        contribution_url=result.get("contribution_url"),
        contribution_method=str(result["contribution_method"]),
        skipped_reason=result.get("skipped_reason"),
    )
    result["record"] = record
    _write_ecosystem_contribution_record(bundle, record)
    return result


def _bundle_from_status(status: dict) -> Path | None:
    raw = status.get("contribution_bundle_path")
    if not raw:
        return None
    bundle = Path(str(raw)).expanduser()
    return bundle if (bundle / "contribution_manifest.json").exists() else None


def _print_auto_share_summary(metadata: dict) -> None:
    typer.echo("Experience Compilation summary")
    typer.echo(f"bundle_id: {metadata.get('bundle_id')}")
    typer.echo(f"title: {metadata.get('title')}")
    typer.echo(f"task_status: {metadata.get('task_status')}")
    typer.echo(f"Apprentice Agent: {metadata.get('apprentice_agent') or 'unknown'}")
    typer.echo(f"Apprenticeship Mode: {apprenticeship_mode_display(metadata.get('apprenticeship_mode') or metadata.get('mentor_mode'))}")
    typer.echo(f"Mentor Model Provider: {metadata.get('mentor_model_provider') or 'none'}")
    typer.echo(f"artifact_count: {metadata.get('artifact_count')}")


def _record_auto_share_skip(bundle: Path, *, mode: str, reason: str) -> None:
    metadata = _bundle_submission_metadata(bundle)
    record = _ecosystem_contribution_record(
        metadata=metadata,
        auto_share_mode=mode,
        contribution_created=False,
        contribution_url=None,
        contribution_method="skipped",
        skipped_reason=reason,
    )
    _write_ecosystem_contribution_record(bundle, record)


def _handle_ecosystem_auto_share(status: dict, *, quiet: bool = False, json_progress: bool = False) -> None:
    settings = get_settings()
    mode = settings.training_contribution_mode
    if mode == "private_internal_only":
        return
    bundle = _bundle_from_status(status)
    if bundle is None:
        return
    emit = not quiet and not json_progress
    if mode == "public_ecosystem":
        behavior = normalize_ecosystem_auto_share(settings.ecosystem_auto_share)
        if behavior == "manual":
            if not emit or not sys.stdin.isatty():
                _record_auto_share_skip(bundle, mode=mode, reason="manual sharing behavior; upload not confirmed")
                if emit:
                    typer.echo("")
                    typer.echo("Public Ecosystem upload: manual; no upload was made.")
                    typer.echo(f"Experience Compilation: {bundle}")
                    typer.echo(f"Retry: apprentice ecosystem contribute {bundle}")
                return
            typer.echo("")
            typer.echo("Share this Experience Compilation to the Forsy ecosystem? [y/N]")
            choice = typer.prompt("Share", default="N", show_default=False).strip().lower()
            if choice not in {"y", "yes"}:
                _record_auto_share_skip(bundle, mode=mode, reason="user chose not to share")
                typer.echo("Kept local package only.")
                typer.echo(f"Retry: apprentice ecosystem contribute {bundle}")
                return
        try:
            result = _ecosystem_contribute_impl(bundle, auto_share_mode=mode)
        except Exception as exc:
            reason = redact_secrets(str(exc))
            _record_auto_share_skip(bundle, mode=mode, reason=reason)
            if emit:
                typer.echo("")
                typer.echo("Public Ecosystem upload skipped - package could not be prepared")
                typer.echo(f"Reason: {reason}")
                typer.echo(f"Experience Compilation: {bundle}")
                typer.echo(f"Retry: apprentice ecosystem contribute {bundle}")
            return
        if emit:
            typer.echo("")
            if result["contribution_created"]:
                typer.echo("Agent Training Contribution Mode: Public Ecosystem")
                typer.echo("Uploading Experience Compilation to Forsy ecosystem...")
                typer.echo(f"Uploaded: {result['contribution_url']}")
            else:
                reason = result.get("skipped_reason") or "public contribution was not created"
                typer.echo(f"Public Ecosystem upload skipped - {reason}")
                typer.echo(f"Experience Compilation: {bundle}")
                typer.echo(f"Retry: apprentice ecosystem contribute {bundle}")
        return


def _gh_issue_body(metadata: dict, submission_dir: Path, package_zip: Path) -> Path:
    body = submission_dir / "GITHUB_ISSUE_BODY.md"
    body.write_text(_submission_summary(metadata, submission_dir / "contribution_bundle", package_zip))
    return body


def _registry_entry_from_bundle(bundle: Path, metadata: dict | None = None) -> dict:
    metadata = metadata or _bundle_submission_metadata(bundle)
    entry = {
        **metadata,
        "bundle_path_or_url": metadata.get("bundle_path_or_url"),
        "local_bundle_path": str(bundle),
    }
    return {k: v for k, v in entry.items() if v is not None}


def _print_learning_sources(rows: list[dict]) -> None:
    if not rows:
        typer.echo("No ecosystem experience matched.")
        return
    for row in rows:
        bundle_id = row.get("bundle_id") or row.get("seed_task_id") or row.get("task_id")
        typer.echo(
            f"{bundle_id}: {row.get('title')} "
            f"[domain={','.join(map(str, row.get('domains') or [])) or 'unknown'}]"
        )


@learn_app.command("search")
def learn_search(query: str | None = typer.Argument(None, help="Search ecosystem experience for learning.")):
    """Search ecosystem entries that can become Experience Packs."""
    _print_learning_sources(search_learning_sources(query))


@learn_app.command("create")
def learn_create(
    sources: list[str] = typer.Argument(..., help="Ecosystem ids or local Experience Compilation paths."),
    title: str | None = typer.Option(None, "--title", help="Optional Experience Pack title."),
    replay: bool = typer.Option(False, "--replay", help="Run before/after replay after creating the pack."),
    runner: str | None = typer.Option(None, "--runner", help="Advanced compatibility runner override for replay."),
    apprenticeship_mode: str = typer.Option("expert-led", "--apprenticeship-mode", help="Apprenticeship Mode for replay."),
    mentor_mode: str | None = typer.Option(None, "--mentor-mode", help="Backward-compatible alias for --apprenticeship-mode.", hidden=True),
):
    settings = get_settings()
    resolved = [resolve_learning_source(source, settings) for source in sources]
    pack_path = compile_experience_pack(resolved, title=title, settings=settings)
    pack = read_json(pack_path / "experience_pack.json")
    typer.echo("Experience Pack created.")
    typer.echo(f"pack_id: {pack.get('pack_id')}")
    typer.echo(f"path: {pack_path}")
    typer.echo(f"title: {pack.get('title')}")
    typer.echo("Next:")
    typer.echo(f"- apprentice learn preview {pack.get('pack_id')}")
    typer.echo(f"- apprentice learn replay {pack.get('pack_id')}")
    if replay:
        _learn_replay_impl(str(pack.get("pack_id")), runner=runner, apprenticeship_mode=mentor_mode or apprenticeship_mode)


@learn_app.command("preview")
def learn_preview(pack_id: str = typer.Argument(..., help="Experience Pack id or path.")):
    path, pack = load_pack(pack_id)
    typer.echo(f"Experience Pack: {pack.get('title')}")
    typer.echo(f"pack_id: {pack.get('pack_id')}")
    typer.echo(f"status: {pack.get('status')}")
    typer.echo(f"path: {path}")
    typer.echo(f"sources: {', '.join(pack.get('source_ecosystem_ids') or [])}")
    typer.echo(f"domains: {', '.join(pack.get('domains') or []) or 'not specified'}")
    typer.echo("")
    typer.echo("Key lessons:")
    lessons = (pack.get("strategy_lessons") or [])[:5]
    if lessons:
        for lesson in lessons:
            typer.echo(f"- {lesson}")
    else:
        typer.echo("- No strategy lessons captured.")
    artifact_requirements = (pack.get("artifact_requirements") or [])[:5]
    if artifact_requirements:
        typer.echo("")
        typer.echo("Artifact requirements:")
        for requirement in artifact_requirements:
            typer.echo(f"- {requirement}")


def _learn_replay_impl(pack_id: str, *, runner: str | None, apprenticeship_mode: str) -> dict:
    settings = _settings_for_run(apprenticeship_mode=apprenticeship_mode, max_loops=1)
    _ensure_apprentice_agent_ready(settings, runner, interactive=False)
    pack_path, pack = load_pack(pack_id, settings)
    instruction = replay_instruction_for_pack(pack, settings)
    replay_slug = slugify(str(pack.get("pack_id") or "experience-pack"))[:48]
    before_id = f"learning-{replay_slug}-before"
    after_id = f"learning-{replay_slug}-after"
    typer.echo("Before/after replay started.")
    typer.echo(f"Experience Pack: {pack.get('title')}")
    with _temporary_auto_approve_env(expert_auto_approve=True, hybrid_auto_approve=True, mentor_interactive_checkpoints=False):
        before_root, _ = run_prompt_task(
            instruction,
            run_id=before_id,
            settings=settings,
            runner=runner,
            progress_callback=None,
        )
        selected, guidance = resolve_packs_for_run([str(pack.get("pack_id"))], settings=settings)
        after_instruction = instruction.rstrip() + guidance if guidance else instruction
        after_root, _ = run_prompt_task(
            after_instruction,
            run_id=after_id,
            settings=settings,
            runner=runner,
            progress_callback=None,
            experience_pack_refs=pack_run_refs(selected),
        )
    before_status = read_run_status(before_root)
    after_status = read_run_status(after_root)
    comparison = compare_replay(before_status, after_status)
    result = {
        **comparison,
        "pack_id": pack.get("pack_id"),
        "before_run_id": before_root.name,
        "after_run_id": after_root.name,
        "before_run_path": str(before_root),
        "after_run_path": str(after_root),
    }
    write_before_after_result(pack_path, result)
    typer.echo(f"before_status: {comparison.get('before_status')}")
    typer.echo(f"after_status: {comparison.get('after_status')}")
    typer.echo(f"before_after_result: {comparison.get('result')}")
    if comparison.get("result") == "improved":
        typer.echo(f"Suggested action: apprentice learn keep {pack.get('pack_id')}")
    elif comparison.get("result") == "regressed":
        typer.echo(f"Suggested action: apprentice learn revert {pack.get('pack_id')}")
    else:
        typer.echo("Suggested action: inspect the replay, then keep or revert learning.")
    return result


@learn_app.command("replay")
def learn_replay(
    pack_id: str = typer.Argument(..., help="Experience Pack id or path."),
    runner: str | None = typer.Option(None, "--runner", help="Advanced compatibility runner override."),
    apprenticeship_mode: str = typer.Option("expert-led", "--apprenticeship-mode", help="Apprenticeship Mode for replay."),
    mentor_mode: str | None = typer.Option(None, "--mentor-mode", help="Backward-compatible alias for --apprenticeship-mode.", hidden=True),
):
    _learn_replay_impl(pack_id, runner=runner, apprenticeship_mode=mentor_mode or apprenticeship_mode)


@learn_app.command("keep")
def learn_keep(pack_id: str = typer.Argument(..., help="Experience Pack id or path.")):
    path = update_pack_status(pack_id, "active")
    typer.echo("Keep learning: Experience Pack is active.")
    typer.echo(f"path: {path}")


@learn_app.command("revert")
def learn_revert(pack_id: str = typer.Argument(..., help="Experience Pack id or path.")):
    path = update_pack_status(pack_id, "reverted")
    typer.echo("Revert learning: Experience Pack is inactive.")
    typer.echo(f"path: {path}")


@learn_app.command("remove")
def learn_remove(pack_id: str = typer.Argument(..., help="Experience Pack id or path.")):
    path = remove_pack(pack_id)
    typer.echo("Experience Pack removed.")
    typer.echo(f"path: {path}")


@learn_app.command("list")
def learn_list():
    packs = list_packs()
    if not packs:
        typer.echo("No Experience Packs yet.")
        return
    for pack in packs:
        typer.echo(f"{pack.get('pack_id')}: {pack.get('title')} [status={pack.get('status')}]")


@learn_app.command("status")
def learn_status():
    packs = active_packs()
    if not packs:
        typer.echo("Experience Packs: Manual")
        typer.echo("Active Experience Packs: none")
        return
    typer.echo("Experience Packs: Manual")
    typer.echo("Active Experience Packs:")
    for pack in packs:
        typer.echo(f"- {pack.get('pack_id')}: {pack.get('title')}")


@learn_app.command("install")
def learn_install(
    item_id: str=typer.Argument(..., help="Ecosystem item id or local Experience Compilation path."),
    dry_run: bool=typer.Option(False, "--dry-run", help="Show what would be installed without writing files."),
    agent_target: str=typer.Option("current", "--agent-target", help="current, codex, claude, cursor, opencode, openclaw, hermes, or custom."),
    registry: Path | None=typer.Option(None, "--registry", help="Public ecosystem index JSON path for offline/test use."),
    storage: str | None=typer.Option(None, "--storage", help="Storage backend: forsy-r2 or github."),
):
    source_path = Path(item_id).expanduser()
    try:
        if source_path.exists():
            package = source_path
            source = str(source_path.name)
        else:
            package = _package_for_ecosystem_item(item_id, registry=registry, storage=storage)
            source = item_id
        result = install_runtime_training_from_bundle(source, package, settings=get_settings(), agent_target=agent_target, dry_run=dry_run)
    except Exception as exc:
        typer.echo(f"Could not install runtime training: {redact_secrets(str(exc))}")
        raise typer.Exit(1)
    if dry_run:
        typer.echo("Install Runtime Training dry run complete.")
        typer.echo(f"installed_skill_id: {result.get('installed_skill_id')}")
        typer.echo(f"target: {result.get('target')}")
        typer.echo("No files were installed.")
        return
    typer.echo("Install Runtime Training complete.")
    typer.echo(f"installed_skill_id: {result.get('installed_skill_id')}")
    typer.echo(f"enabled: {str(result.get('enabled')).lower()}")
    typer.echo("registry: local Agent Apprenticeship runtime")


@learn_app.command("installed")
def learn_installed(json_output: bool=typer.Option(False, "--json", help="Print installed runtime training as JSON.")):
    rows = list_installed_skills(get_settings())
    if json_output:
        typer.echo(json.dumps(rows, indent=2, sort_keys=True))
        return
    if not rows:
        typer.echo("Installed Runtime Training: none")
        return
    typer.echo("Installed Runtime Training")
    for row in rows:
        status = "enabled" if row.get("enabled", True) else "disabled"
        typer.echo(f"- {row.get('installed_skill_id')}: {row.get('title')} [{status}]")


@learn_app.command("enable")
def learn_enable(installed_skill_id: str=typer.Argument(..., help="Installed runtime training id.")):
    try:
        row = set_installed_skill_enabled(installed_skill_id, True, get_settings())
    except FileNotFoundError as exc:
        typer.echo(str(exc))
        raise typer.Exit(1)
    typer.echo("Runtime training enabled.")
    typer.echo(f"installed_skill_id: {row.get('installed_skill_id')}")


@learn_app.command("disable")
def learn_disable(installed_skill_id: str=typer.Argument(..., help="Installed runtime training id.")):
    try:
        row = set_installed_skill_enabled(installed_skill_id, False, get_settings())
    except FileNotFoundError as exc:
        typer.echo(str(exc))
        raise typer.Exit(1)
    typer.echo("Runtime training disabled.")
    typer.echo(f"installed_skill_id: {row.get('installed_skill_id')}")


@learn_app.command("uninstall")
def learn_uninstall(installed_skill_id: str=typer.Argument(..., help="Installed runtime training id.")):
    try:
        uninstall_installed_skill(installed_skill_id, get_settings())
    except FileNotFoundError as exc:
        typer.echo(str(exc))
        raise typer.Exit(1)
    typer.echo("Runtime training uninstalled.")
    typer.echo("registry: local Agent Apprenticeship runtime")


@ecosystem_app.command('guide')
def ecosystem_guide():
    typer.echo('Agent Apprenticeship Ecosystem')
    typer.echo('')
    typer.echo('The public ecosystem stores shared Experience Compilations in Forsy Cloudflare R2.')
    typer.echo('Run a task, then share the generated Experience Compilation:')
    typer.echo('  apprentice ecosystem contribute <bundle_path>')
    typer.echo('')
    typer.echo('Configure ecosystem storage:')
    typer.echo('  apprentice ecosystem configure --storage forsy-r2')
    typer.echo('  apprentice ecosystem configure --r2-api-url <worker-url>')
    typer.echo('')
    typer.echo('Agent Training Contribution Mode is Public Ecosystem by default.')
    typer.echo('  apprentice ecosystem configure --training-contribution public-ecosystem')
    typer.echo('  apprentice ecosystem configure --training-contribution private-internal-only')
    typer.echo('')
    typer.echo('Legacy GitHub issue sharing remains available only with:')
    typer.echo('  apprentice ecosystem contribute <bundle_path> --storage github')
    typer.echo(f'Community Slack: {SLACK_LINK}')


@ecosystem_app.command('configure')
def ecosystem_configure(
    repo: str | None=typer.Option(None, '--repo', help='Public ecosystem repo in owner/name form.'),
    repo_path: Path | None=typer.Option(None, '--repo-path', help='Public ecosystem repo path for a local checkout/export.'),
    apprenticeship_mode: str | None=typer.Option(None, '--apprenticeship-mode', help='autonomous, expert-led, or org-custom.'),
    training_contribution: str | None=typer.Option(None, '--training-contribution', help='public-ecosystem or private-internal-only.'),
    sharing_behavior: str | None=typer.Option(None, '--sharing-behavior', help='Public Ecosystem behavior: manual or automatic.'),
    auto_share: str | None=typer.Option(None, '--auto-share', help='Deprecated alias for --training-contribution.', hidden=True),
    storage: str | None=typer.Option(None, '--storage', help='Storage backend: forsy-r2 or github.'),
    r2_api_url: str | None=typer.Option(None, '--r2-api-url', help='Forsy R2 Worker ingestion API URL.'),
    r2_bucket: str | None=typer.Option(None, '--r2-bucket', help='R2 bucket name.'),
    r2_index_prefix: str | None=typer.Option(None, '--r2-index-prefix', help='R2 index prefix.'),
    r2_contribution_prefix: str | None=typer.Option(None, '--r2-contribution-prefix', help='R2 Experience Compilation storage prefix.'),
    r2_seed_prefix: str | None=typer.Option(None, '--r2-seed-prefix', help='R2 seed dataset prefix.'),
):
    settings = get_settings()
    if all(v is None for v in [repo, repo_path, apprenticeship_mode, training_contribution, sharing_behavior, auto_share, storage, r2_api_url, r2_bucket, r2_index_prefix, r2_contribution_prefix, r2_seed_prefix]):
        status = _ecosystem_status()
        typer.echo("Public Ecosystem Configuration")
        typer.echo(f"Storage backend: {settings.ecosystem_storage_backend}")
        typer.echo(f"R2 ingestion API: {settings.r2_api_url or 'not configured'}")
        typer.echo(f"R2 bucket: {settings.r2_bucket}")
        typer.echo(f"Current repo: {settings.ecosystem_repo or DEFAULT_PUBLIC_ECOSYSTEM_REPO}")
        typer.echo(f"GitHub URL: {_ecosystem_repo_url(settings.ecosystem_repo)}")
        typer.echo(f"Current public ecosystem repo path: {settings.ecosystem_repo_path or 'Not configured'}")
        typer.echo(f"GitHub CLI installed: {_yes_no(status['gh_installed'])}")
        typer.echo(f"GitHub CLI authenticated: {_yes_no(status['gh_authenticated'])}")
        typer.echo(f"Current Apprenticeship Mode: {apprenticeship_mode_display(settings.apprenticeship_mode)}")
        typer.echo(f"Current Agent Training Contribution Mode: {training_contribution_mode_display(settings.training_contribution_mode)}")
        entered_repo = typer.prompt("Public ecosystem repo, owner/name, optional", default=settings.ecosystem_repo or DEFAULT_PUBLIC_ECOSYSTEM_REPO, show_default=False).strip()
        if entered_repo:
            repo = entered_repo
        typer.echo("")
        typer.echo("Choose Agent Training Contribution Mode:")
        typer.echo("1. Public Ecosystem")
        typer.echo("2. Private Internal Only")
        choice = typer.prompt("Default", default=settings.training_contribution_mode).strip().lower()
        training_contribution = {"1": "public_ecosystem", "2": "private_internal_only"}.get(choice, choice)
    updates = {}
    candidate_repo = repo or settings.ecosystem_repo or DEFAULT_PUBLIC_ECOSYSTEM_REPO
    if repo is not None:
        if "/" not in repo:
            typer.echo("Repository must be in owner/name form.")
            raise typer.Exit(1)
        updates["ecosystem_repo"] = repo
    if repo_path is not None:
        expanded = repo_path.expanduser()
        if not expanded.exists():
            typer.echo(f"Public ecosystem repo path not found: {expanded}")
            raise typer.Exit(1)
        if not (expanded / "seed_dataset").exists() and not (expanded / "ecosystem").exists():
            typer.echo(f"Public ecosystem repo path does not look like an Agent Apprenticeship ecosystem repo: {expanded}")
            raise typer.Exit(1)
        updates["ecosystem_repo_path"] = expanded
    if apprenticeship_mode is not None:
        updates["apprenticeship_mode"] = _normalize_apprenticeship_mode(apprenticeship_mode)
    if training_contribution is not None:
        updates["training_contribution_mode"] = _normalize_training_contribution(training_contribution)
    if sharing_behavior is not None:
        updates["ecosystem_auto_share"] = _normalize_auto_share(sharing_behavior)
        if _normalize_auto_share(sharing_behavior) != "disabled":
            updates["training_contribution_mode"] = "public_ecosystem"
        else:
            updates["training_contribution_mode"] = "private_internal_only"
    if auto_share is not None:
        updates["ecosystem_auto_share"] = _normalize_auto_share(auto_share)
        updates["training_contribution_mode"] = "public_ecosystem" if _normalize_auto_share(auto_share) != "disabled" else "private_internal_only"
    if storage is not None:
        updates["ecosystem_storage_backend"] = normalize_ecosystem_storage(storage)
    if r2_api_url is not None:
        updates["r2_api_url"] = r2_api_url.rstrip("/")
    if r2_bucket is not None:
        updates["r2_bucket"] = r2_bucket
    if r2_index_prefix is not None:
        updates["r2_index_prefix"] = r2_index_prefix.strip("/")
    if r2_contribution_prefix is not None:
        updates["r2_contribution_prefix"] = r2_contribution_prefix.strip("/")
    if r2_seed_prefix is not None:
        updates["r2_seed_prefix"] = r2_seed_prefix.strip("/")
    if not updates:
        typer.echo("No ecosystem settings changed.")
        return
    updated = update_settings(**updates)
    if repo is not None:
        typer.echo(f"Public ecosystem repo configured: {repo}")
    if repo_path is not None:
        typer.echo(f"Public ecosystem repo path configured: {Path(repo_path).expanduser()}")
    if apprenticeship_mode is not None:
        typer.echo(f"Apprenticeship Mode: {apprenticeship_mode_display(updated.apprenticeship_mode)}")
    if training_contribution is not None or sharing_behavior is not None or auto_share is not None:
        typer.echo(f"Agent Training Contribution Mode: {training_contribution_mode_display(updated.training_contribution_mode)}")
        if updated.training_contribution_mode == "public_ecosystem":
            typer.echo(f"Public Ecosystem sharing behavior: {ecosystem_auto_share_display(updated.ecosystem_auto_share)}")
    if storage is not None:
        typer.echo(f"Storage backend: {updated.ecosystem_storage_backend}")
    if r2_api_url is not None:
        typer.echo(f"R2 ingestion API: {updated.r2_api_url or 'not configured'}")
    if any(v is not None for v in [r2_bucket, r2_index_prefix, r2_contribution_prefix, r2_seed_prefix]):
        typer.echo(f"R2 bucket: {updated.r2_bucket}")
        typer.echo(f"R2 index prefix: {updated.r2_index_prefix}")
        typer.echo(f"R2 Experience Compilation prefix: {updated.r2_contribution_prefix}")
        typer.echo(f"R2 seed prefix: {updated.r2_seed_prefix}")
    typer.echo("Next: apprentice ecosystem status")


@ecosystem_app.command('status')
def ecosystem_status_command():
    status = _ecosystem_status()
    settings = get_settings()
    typer.echo("Public Ecosystem Status")
    typer.echo(f"Storage backend: {status['storage_backend']}")
    typer.echo(f"R2 ingestion API: {status.get('r2_api_url') or 'not configured'}")
    typer.echo(f"R2 bucket: {status['r2_bucket']}")
    typer.echo(f"R2 index prefix: {status['r2_index_prefix']}")
    typer.echo(f"R2 Experience Compilation prefix: {status['r2_contribution_prefix']}")
    typer.echo(f"R2 seed prefix: {status['r2_seed_prefix']}")
    typer.echo(f"GitHub fallback repo: {status['repo'] or DEFAULT_PUBLIC_ECOSYSTEM_REPO}")
    typer.echo(f"GitHub URL: {status['repo_url']}")
    typer.echo(f"Public ecosystem repo path: {status.get('repo_path') or 'Not configured'}")
    if status.get("repo_path"):
        typer.echo(f"Public ecosystem repo path exists: {_yes_no(status.get('repo_path_exists'))}")
    typer.echo(f"GitHub CLI installed: {_yes_no(status['gh_installed'])}")
    typer.echo(f"GitHub CLI authenticated: {_yes_no(status['gh_authenticated'])}")
    typer.echo(f"Apprenticeship Mode: {apprenticeship_mode_display(settings.apprenticeship_mode)}")
    typer.echo(f"Agent Training Contribution Mode: {training_contribution_mode_display(settings.training_contribution_mode)}")
    typer.echo(f"Local ecosystem user: {status.get('ecosystem_user_id')}")
    typer.echo(f"Shared contributions: {status.get('shared_count', 0)}")
    if settings.training_contribution_mode == "public_ecosystem":
        readiness = _ecosystem_auto_share_readiness()
        typer.echo(f"Public Ecosystem upload status: {'Ready' if readiness['ready'] else 'Not ready - ' + str(readiness.get('reason') or 'not ready')}")


@ecosystem_app.command('identity')
def ecosystem_identity():
    settings = get_settings()
    identity = get_or_create_ecosystem_identity(settings)
    typer.echo("Local ecosystem identity")
    typer.echo(f"ecosystem_user_id: {identity.get('ecosystem_user_id')}")
    typer.echo(f"installation_id: {identity.get('installation_id')}")
    typer.echo(f"created_at: {identity.get('created_at')}")
    typer.echo(f"shared_contributions: {len(shared_history(settings))}")


def _print_shared_rows(rows: list[dict[str, Any]]) -> None:
    if not rows:
        typer.echo("No shared Experience Compilations recorded yet.")
        return
    for row in rows:
        typer.echo(f"{row.get('item_id')}: {row.get('title')} [{row.get('storage_backend')} · {row.get('uploaded_at')}]")


@ecosystem_shared_app.callback(invoke_without_command=True)
def ecosystem_shared(
    ctx: typer.Context,
    json_output: bool = typer.Option(False, "--json", help="Print local shared Experience Compilation history as JSON."),
):
    if ctx.invoked_subcommand is not None:
        return
    rows = shared_history(get_settings())
    if json_output:
        typer.echo(json.dumps(rows, indent=2, sort_keys=True))
        return
    _print_shared_rows(rows)


def _local_shared_record(item_id: str) -> dict[str, Any]:
    for row in shared_history(get_settings()):
        if item_id in {str(row.get("item_id") or ""), str(row.get("package_id") or "")}:
            return row
    raise FileNotFoundError(f"Shared contribution not found locally: {item_id}")


@ecosystem_shared_app.command("inspect")
def ecosystem_shared_inspect(item_id: str = typer.Argument(...)):
    try:
        row = _local_shared_record(item_id)
    except FileNotFoundError as exc:
        typer.echo(str(exc))
        raise typer.Exit(1)
    typer.echo(f"item_id: {row.get('item_id')}")
    typer.echo(f"package_id: {row.get('package_id')}")
    typer.echo(f"title: {row.get('title')}")
    typer.echo(f"storage_backend: {row.get('storage_backend')}")
    typer.echo(f"url: {row.get('url')}")
    typer.echo(f"local_bundle_path: {row.get('local_bundle_path')}")


@ecosystem_shared_app.command("pull")
def ecosystem_shared_pull(item_id: str = typer.Argument(...)):
    try:
        row = _local_shared_record(item_id)
    except FileNotFoundError as exc:
        typer.echo(str(exc))
        raise typer.Exit(1)
    local = Path(str(row.get("local_bundle_path") or "")).expanduser()
    dest = _prepare_ecosystem_pull_destination(str(row.get("item_id") or item_id))
    if local.exists():
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(local, dest)
        typer.echo(f"pulled_bundle={dest}")
        return
    try:
        download_r2_package(str(row.get("item_id") or item_id), dest, get_settings())
    except Exception as exc:
        typer.echo(f"Could not pull shared Experience Compilation: {redact_secrets(str(exc))}")
        raise typer.Exit(1)
    typer.echo(f"pulled_bundle={dest}")


@ecosystem_app.command('contribute')
def ecosystem_contribute(
    bundle_path: Path=typer.Argument(..., help='Experience Compilation folder.'),
    repo: str | None=typer.Option(None, '--repo', help='Public ecosystem repo in owner/name form.'),
    dry_run: bool=typer.Option(False, '--dry-run', help='Prepare upload package without uploading.'),
    storage: str | None=typer.Option(None, '--storage', help='Storage backend: forsy-r2 or github.'),
    qa: bool=typer.Option(False, '--qa', help='Use QA storage path when supported.'),
):
    try:
        result = _ecosystem_contribute_impl(bundle_path, repo=repo, dry_run=dry_run, auto_share_mode="manual", storage=storage, qa=qa)
    except Exception as exc:
        typer.echo(f"Could not prepare public ecosystem contribution: {redact_secrets(str(exc))}")
        raise typer.Exit(1)
    bundle = result["bundle"]
    submission_dir = result["submission_dir"]
    package_zip = result["package_zip"]
    metadata = result["metadata"]
    target_repo = result["target_repo"]
    issue_body = result["issue_body"]
    if result.get("storage_backend") == "forsy-r2":
        item = result.get("ecosystem_item") or {}
        storage_ref = result.get("storage_ref") or {}
        if dry_run:
            typer.echo("Forsy ecosystem upload dry run complete.")
            typer.echo("No files were uploaded.")
            typer.echo("")
            typer.echo("Storage backend:")
            typer.echo("Cloudflare R2")
            typer.echo("")
            typer.echo("Ingestion API:")
            typer.echo(str(storage_ref.get("ingestion_api_url") or "not configured"))
            typer.echo("")
            typer.echo("Local ecosystem user:")
            typer.echo(str(storage_ref.get("ecosystem_user_id")))
            typer.echo("")
            typer.echo("Prepared package:")
            typer.echo(str(package_zip))
            typer.echo("")
            typer.echo("Storage reference:")
            typer.echo("ecosystem_submission/storage_ref.json")
            return
        if result["contribution_created"]:
            typer.echo("Experience Compilation uploaded to the Forsy ecosystem.")
            typer.echo("")
            typer.echo("Item:")
            typer.echo(str(item.get("id")))
            typer.echo("")
            typer.echo("Local ecosystem user:")
            typer.echo(str(storage_ref.get("ecosystem_user_id")))
            typer.echo("")
            typer.echo("Storage:")
            typer.echo(f"Cloudflare R2 / {(item.get('storage') or {}).get('path_prefix')}")
            typer.echo("")
            typer.echo("URL:")
            typer.echo(str(result["contribution_url"]))
            return
        typer.echo("Forsy ecosystem upload package prepared.")
        typer.echo(f"Reason upload did not run: {result.get('skipped_reason') or 'not uploaded'}")
        typer.echo(f"Prepared package: {package_zip}")
        typer.echo("Storage reference: ecosystem_submission/storage_ref.json")
        typer.echo("Retry after configuring the Worker URL:")
        typer.echo("apprentice ecosystem configure --r2-api-url <url>")
        return

    typer.echo("Legacy GitHub ecosystem contribution prepared.")
    typer.echo("")
    typer.echo(f"Bundle ID: {metadata.get('bundle_id')}")
    typer.echo(f"Title: {metadata.get('title')}")
    typer.echo(f"Bundle folder: {bundle.name}")
    typer.echo(f"Submission package: {package_zip.name}")
    typer.echo("Submission metadata: ecosystem_submission/ecosystem_submission.json")
    if result["contribution_created"]:
        typer.echo(f"Public contribution URL: {result['contribution_url']}")
        typer.echo("No bundle files were uploaded automatically. Attach or submit the generated package according to the repo instructions.")
        return
    if target_repo and _gh_path() and _gh_authenticated() and not dry_run and result.get("gh_error"):
        typer.echo("GitHub issue creation failed; manual submission package is ready.")
        typer.echo(str(result.get("gh_error") or ""))
    elif target_repo and _gh_path() and _gh_authenticated() and dry_run:
        typer.echo("Dry run: no GitHub issue was created.")
        typer.echo(f"Would run: gh issue create --repo {target_repo} --title \"Experience Compilation: {metadata.get('bundle_id')}\" --body-file GITHUB_ISSUE_BODY.md")
    else:
        typer.echo("No GitHub issue was created.")
        typer.echo(f"Public ecosystem: {_ecosystem_repo_url(target_repo)}")
        if not _gh_path():
            typer.echo("GitHub CLI `gh` is not installed.")
        elif not _gh_authenticated():
            typer.echo("GitHub CLI `gh` is not authenticated.")
    typer.echo("")
    typer.echo("Manual public submission steps:")
    typer.echo(f"1. Open {_ecosystem_repo_url(target_repo)} in GitHub.")
    typer.echo("2. Create a new issue using: ecosystem_submission/GITHUB_ISSUE_BODY.md")
    typer.echo(f"3. Attach or share the submission package: {package_zip.name}")
    typer.echo(f"Community Slack: {SLACK_LINK}")


@ecosystem_app.command('list')
def ecosystem_list(
    registry: Path | None=typer.Option(None, '--registry', help='Public ecosystem index JSON path for offline/test use.'),
    storage: str | None=typer.Option(None, '--storage', help='Storage backend: forsy-r2 or github.'),
    mine: bool=typer.Option(False, '--mine', help='Show items shared by this local ecosystem user.'),
    domain: str | None=typer.Option(None, '--domain', help='Filter by domain.'),
    subdomain: str | None=typer.Option(None, '--subdomain', help='Filter by subdomain.'),
    tag: str | None=typer.Option(None, '--tag', help='Filter by tag.'),
    json_output: bool=typer.Option(False, '--json', help='Print matching Experience Compilations as JSON.'),
):
    try:
        rows=_load_registry(registry, storage=storage)
    except FileNotFoundError as exc:
        typer.echo(str(exc))
        raise typer.Exit(1)
    if mine:
        rows = [row for row in rows if _matches_registry_row(row, None, {"mine": True})]
    filters = {"domain": domain, "subdomain": subdomain, "tag": tag}
    rows = [row for row in rows if _matches_registry_row(row, None, filters)]
    _print_registry_rows(rows, json_output=json_output)


@ecosystem_app.command('search')
def ecosystem_search(
    query_arg: str | None=typer.Argument(None, help='Free-text query.'),
    query: str | None=typer.Option(None, '--query', help='Free-text query.'),
    registry: Path | None=typer.Option(None, '--registry', help='Public ecosystem index JSON path for offline/test use.'),
    storage: str | None=typer.Option(None, '--storage', help='Storage backend: forsy-r2 or github.'),
    mine: bool=typer.Option(False, '--mine', help='Search items shared by this local ecosystem user.'),
    title: str | None=typer.Option(None, '--title'),
    domain: str | None=typer.Option(None, '--domain'),
    subdomain: str | None=typer.Option(None, '--subdomain'),
    apprenticeship_mode: str | None=typer.Option(None, '--apprenticeship-mode', help='autonomous, expert-led, or org-custom.'),
    mentor_mode: str | None=typer.Option(None, '--mentor-mode', help='Backward-compatible alias for --apprenticeship-mode.', hidden=True),
    task_status: str | None=typer.Option(None, '--task-status'),
    run_status: str | None=typer.Option(None, '--run-status'),
    min_traced_steps: int | None=typer.Option(None, '--min-traced-steps'),
    max_traced_steps: int | None=typer.Option(None, '--max-traced-steps'),
    expected_economic_value: str | None=typer.Option(None, '--expected-economic-value'),
    tag: str | None=typer.Option(None, '--tag'),
    artifact_type: str | None=typer.Option(None, '--artifact-type'),
    apprentice_agent: str | None=typer.Option(None, '--apprentice-agent'),
    mentor_model_provider: str | None=typer.Option(None, '--mentor-model-provider'),
    json_output: bool=typer.Option(False, '--json', help='Print matching Experience Compilations as JSON.'),
    select: bool=typer.Option(False, '--select', help='Interactively choose a matching Experience Compilation and action.'),
):
    try:
        rows=_load_registry(registry, storage=storage)
    except FileNotFoundError as exc:
        typer.echo(str(exc))
        raise typer.Exit(1)
    effective_query = query or query_arg
    filters=locals()
    filters.pop('query_arg', None)
    filters.pop('query', None)
    filters.pop('registry', None)
    filters.pop('storage', None)
    filters.pop('effective_query', None)
    filters["mine"] = mine
    if mentor_mode and not apprenticeship_mode:
        filters["apprenticeship_mode"] = mentor_mode
    _print_registry_rows([row for row in rows if _matches_registry_row(row, effective_query, filters)], json_output=json_output, select=select)


def _find_registry_row(bundle_id: str, registry: Path | None=None, storage: str | None = None) -> dict:
    rows=_load_registry(registry, storage=storage)
    for row in rows:
        ids = {
            str(row.get("id") or ""),
            str(row.get("bundle_id") or ""),
            str(row.get("package_id") or ""),
            str(row.get("seed_task_id") or ""),
            str(row.get("task_id") or ""),
        }
        if bundle_id in ids:
            return row
    raise FileNotFoundError(f'Bundle id not found in registry: {bundle_id}')


def _prepare_ecosystem_pull_destination(bundle_id: str) -> Path:
    dest=get_settings().app_home / 'ecosystem' / 'bundles' / bundle_id
    try:
        if dest.exists():
            shutil.rmtree(dest)
        dest.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        typer.echo('Could not prepare local ecosystem pull destination.')
        typer.echo(f'Path: {dest}')
        typer.echo(f'Reason: {exc}')
        typer.echo('Next action: set AA_HOME to a writable directory or remove the existing pulled item and try again.')
        raise typer.Exit(1)
    return dest


def _locate_package_root(path: Path) -> Path:
    path = path.expanduser()
    if path.is_file():
        path = path.parent
    candidates = [path, path / "contribution_bundle", path / "package"]
    for candidate in candidates:
        if (candidate / "contribution_manifest.json").exists() or (candidate / "experience_compiler").exists():
            return candidate
    for manifest in path.rglob("contribution_manifest.json"):
        return manifest.parent
    return path


def _copy_or_zip_package(src: Path, output: Path, *, item_id: str, source: str) -> Path:
    src = _locate_package_root(src)
    from .agent_env_exporter import export_agent_env
    from .io import validate_trace_tree
    validate_trace_tree(src)
    output = output.expanduser()
    manifest = {
        "export_type": "full_experience_compilation",
        "item_id": item_id,
        "source": source,
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "package_root": "package",
        "includes": [
            "traces",
            "attempts",
            "artifacts",
            "outputs",
            "evaluation",
            "learning_signals",
            "experience_compiler",
            "skill_pack",
            "runtime_learning_package",
            "training_time_package",
            "environment",
            "manifests",
        ],
    }
    if output.suffix == ".zip":
        staging = output.parent / f".{output.stem}-export"
        if staging.exists():
            shutil.rmtree(staging)
        (staging / "package").mkdir(parents=True)
        shutil.copytree(src, staging / "package", dirs_exist_ok=True)
        export_agent_env(staging / "package")
        write_json(staging / "export_manifest.json", manifest)
        _zip_dir(staging, output)
        shutil.rmtree(staging)
        return output
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)
    shutil.copytree(src, output / "package", dirs_exist_ok=True)
    export_agent_env(output / "package")
    write_json(output / "export_manifest.json", manifest)
    return output


def _package_for_ecosystem_item(item_id: str, *, registry: Path | None = None, storage: str | None = None) -> Path:
    direct = Path(item_id).expanduser()
    if direct.exists():
        return _locate_package_root(direct)
    local = get_settings().app_home / "ecosystem" / "bundles" / item_id
    if local.exists():
        return _locate_package_root(local)
    backend = normalize_ecosystem_storage(storage or get_settings().ecosystem_storage_backend)
    if backend == "forsy-r2" and registry is None:
        dest = _prepare_ecosystem_pull_destination(item_id)
        download_r2_package(item_id, dest, get_settings())
        return _locate_package_root(dest)
    row = _find_registry_row(item_id, registry, storage=storage)
    src = row.get("local_bundle_path") or row.get("path") or row.get("bundle_path") or row.get("bundle_path_or_url")
    if src and Path(str(src)).expanduser().exists():
        return _locate_package_root(Path(str(src)).expanduser())
    raise FileNotFoundError(f"No full package is available for ecosystem item: {item_id}")


@ecosystem_app.command('pull')
def ecosystem_pull(
    bundle_id: str=typer.Argument(...),
    registry: Path | None=typer.Option(None, '--registry', help='Public ecosystem index JSON path for offline/test use.'),
    storage: str | None=typer.Option(None, '--storage', help='Storage backend: forsy-r2 or github.'),
    full: bool=typer.Option(False, '--full', help='Export the Full Experience Compilation after pulling.'),
    runtime: bool=typer.Option(False, '--runtime', help='Pull runtime training subset for install.'),
    install: bool=typer.Option(False, '--install', help='Install runtime training after pulling.'),
    mode: str | None=typer.Option(None, '--mode', help='full or runtime.'),
):
    try:
        row=_find_registry_row(bundle_id, registry, storage=storage)
    except FileNotFoundError as exc:
        typer.echo(str(exc))
        raise typer.Exit(1)
    backend = normalize_ecosystem_storage(storage or get_settings().ecosystem_storage_backend)
    if backend == "forsy-r2" and not registry:
        dest=_prepare_ecosystem_pull_destination(_row_id(row) or bundle_id)
        try:
            download_r2_package(str(row.get("id") or bundle_id), dest, get_settings())
        except Exception as exc:
            typer.echo(f"Could not pull ecosystem item from R2: {redact_secrets(str(exc))}")
            typer.echo("Fallback: apprentice ecosystem pull " + bundle_id + " --storage github")
            raise typer.Exit(1)
        from .io import validate_trace_tree
        validate_trace_tree(dest)
        typer.echo(f'pulled_bundle={dest}')
        if full or mode == "full":
            export_path = _copy_or_zip_package(dest, get_settings().app_home / "ecosystem" / "exports" / (_row_id(row) or bundle_id), item_id=_row_id(row) or bundle_id, source="r2_pull")
            typer.echo(f"full_experience_compilation={export_path}")
        if install or runtime or mode == "runtime":
            manifest = install_runtime_training_from_bundle(_row_id(row) or bundle_id, dest, settings=get_settings())
            typer.echo("Runtime training installed.")
            typer.echo(f"installed_skill_id: {manifest.get('installed_skill_id')}")
        return
    src=row.get('local_bundle_path') or row.get('path') or row.get('bundle_path') or row.get('bundle_path_or_url')
    if not src:
        task_path = row.get("task_path") or row.get("task_packet_path")
        if not task_path:
            typer.echo('This public index entry has no downloadable bundle path or URL.')
            raise typer.Exit(1)
        dest=_prepare_ecosystem_pull_destination(bundle_id)
        row = {**row, "experience_source_type": row.get("experience_source_type") or "seed_task", "pulled_item_type": "seed_task"}
        try:
            write_json(dest / "ecosystem_item.json", row)
            public_repo = row.get("public_repo_slug")

            def resolve_seed_path(raw: str | None) -> Path | None:
                if not raw:
                    return None
                path = Path(str(raw)).expanduser()
                return path if path.is_absolute() else Path.cwd() / path

            def copy_seed_path(raw: str | None, target_rel: str) -> None:
                if not raw:
                    return
                source_path = resolve_seed_path(raw)
                target = dest / target_rel
                target.parent.mkdir(parents=True, exist_ok=True)
                if source_path and source_path.exists() and source_path.is_dir():
                    shutil.copytree(source_path, target, dirs_exist_ok=True)
                elif source_path and source_path.exists():
                    shutil.copy2(source_path, target)
                elif public_repo:
                    _download_public_repo_path(str(public_repo), str(raw), target)

            copy_seed_path(row.get("task_path") or row.get("task_packet_path"), "task")
            copy_seed_path(row.get("rubric_path"), "rubric/rubric.json")
            copy_seed_path(row.get("worker_visible_rubric_path"), "rubric/worker_visible_rubric.md")
            copy_seed_path(row.get("verifier_private_rubric_path"), "rubric/verifier_private_rubric.json")
            copy_seed_path(row.get("attempts_path"), "attempts")
            copy_seed_path(row.get("evaluation_path"), "evaluation")
            copy_seed_path(row.get("learning_signals_path"), "learning_signals")
            for idx, trace_path in enumerate(row.get("trace_paths") or [], 1):
                trace_source = resolve_seed_path(str(trace_path))
                target_name = Path(trace_path).parent.name or f"trace_{idx}"
                if (trace_source and trace_source.exists()) or public_repo:
                    copy_seed_path(str(trace_path), f"traces/{target_name}/{Path(trace_path).name}")
        except OSError as exc:
            typer.echo('Could not pull ecosystem seed task.')
            typer.echo(f'Path: {dest}')
            typer.echo(f'Reason: {exc}')
            typer.echo('Next action: set AA_HOME to a writable directory and try again.')
            raise typer.Exit(1)
        from .io import validate_trace_tree
        validate_trace_tree(dest)
        typer.echo(f'pulled_bundle={dest}')
        return
    source=Path(src).expanduser()
    if not source.exists():
        typer.echo(f'Bundle path not found locally: {source}')
        typer.echo('If this is a public URL, download support depends on the ecosystem repo packaging convention for v0.')
        raise typer.Exit(1)
    dest=_prepare_ecosystem_pull_destination(bundle_id)
    try:
        shutil.rmtree(dest)
        shutil.copytree(source, dest)
    except OSError as exc:
        typer.echo('Could not pull ecosystem bundle.')
        typer.echo(f'Path: {dest}')
        typer.echo(f'Reason: {exc}')
        typer.echo('Next action: set AA_HOME to a writable directory and try again.')
        raise typer.Exit(1)
    from .io import validate_trace_tree
    validate_trace_tree(dest)
    typer.echo(f'pulled_bundle={dest}')
    if full or mode == "full":
        export_path = _copy_or_zip_package(dest, get_settings().app_home / "ecosystem" / "exports" / bundle_id, item_id=bundle_id, source="local_pull")
        typer.echo(f"full_experience_compilation={export_path}")
    if install or runtime or mode == "runtime":
        try:
            manifest = install_runtime_training_from_bundle(bundle_id, dest, settings=get_settings())
        except Exception as exc:
            typer.echo(f"Could not install runtime training: {redact_secrets(str(exc))}")
            raise typer.Exit(1)
        typer.echo("Runtime training installed.")
        typer.echo(f"installed_skill_id: {manifest.get('installed_skill_id')}")


@ecosystem_app.command('export')
def ecosystem_export(
    item_id: str=typer.Argument(..., help='Ecosystem item id.'),
    output: Path=typer.Option(..., '--output', '-o', help='Output folder or .zip path.'),
    full: bool=typer.Option(False, '--full', help='Export the Full Experience Compilation.'),
    mode: str | None=typer.Option(None, '--mode', help='Export mode: full.'),
    registry: Path | None=typer.Option(None, '--registry', help='Public ecosystem index JSON path for offline/test use.'),
    storage: str | None=typer.Option(None, '--storage', help='Storage backend: forsy-r2 or github.'),
):
    if not full and mode not in {None, "full"}:
        raise typer.BadParameter("Only Full Experience Compilation export is supported for this command.")
    try:
        package = _package_for_ecosystem_item(item_id, registry=registry, storage=storage)
        exported = _copy_or_zip_package(package, output, item_id=item_id, source="ecosystem_export")
    except Exception as exc:
        typer.echo(f"Could not export Full Experience Compilation: {redact_secrets(str(exc))}")
        raise typer.Exit(1)
    typer.echo("Export Full Experience Compilation complete.")
    typer.echo(f"item_id: {item_id}")
    typer.echo(f"output: {exported}")


def _summarize_bundle_manifest(manifest: dict) -> str:
    keys=[
        'bundle_id','title','attempts','traced_steps','process_supervision_rows',
        'reward_modeling_rows','revision_preference_pairs','domains','subdomains',
        'agent_apprentice_role','expected_economic_value',
        'expected_economic_value_for_agent_apprentice','task_status','run_status'
    ]
    lines = []
    for key in keys:
        if key not in manifest:
            continue
        lines.append(f"{key}: {manifest.get(key)}")
    mode = manifest.get("apprenticeship_mode") or manifest.get("mentor_mode")
    if mode:
        lines.append(f"apprenticeship_mode: {apprenticeship_mode_display(normalize_apprenticeship_mode(str(mode)))}")
    return "\n".join(lines)


def _summarize_seed_registry_row(row: dict) -> str:
    trace_count = row.get("trace_count")
    if trace_count is None:
        trace_count = len(row.get("trace_paths") or [])
    learning_count = row.get("training_signal_rows") or row.get("lessons_count") or 0
    lines = [
        "kind: seed_task",
        f"bundle_id: {row.get('bundle_id') or row.get('seed_task_id')}",
        f"title: {row.get('title')}",
        f"status: {row.get('task_status') or 'ready_to_run'} / {row.get('run_status') or 'not_run'}",
        f"traces: {_format_count(trace_count)}",
        f"attempts: {_format_count(row.get('attempt_count') or row.get('attempts') or 0)}",
        f"process_supervision_rows: {_format_count(row.get('process_supervision_rows') or 0)}",
        f"reward_modeling_rows: {_format_count(row.get('reward_modeling_rows') or 0)}",
        f"revision_preference_pairs: {_format_count(row.get('revision_preference_pairs') or 0)}",
        f"learning_signals: {_format_count(learning_count) if learning_count else 'available'}",
    ]
    if row.get("domains"):
        lines.append(f"domains: {row.get('domains')}")
    if row.get("subdomains"):
        lines.append(f"subdomains: {row.get('subdomains')}")
    lines.append("note: this is a seed task, not a completed Experience Compilation")
    return "\n".join(lines)


def _summarize_contribution_registry_row(row: dict) -> str:
    item_id = _row_id(row)
    compiler = row.get("experience_compiler") or {}
    available = row.get("available_outputs") or {}
    lines = [
        f"kind: {row.get('kind') or 'agent_experience_package'}",
        f"item_id: {item_id}",
        f"title: {row.get('title')}",
        f"task_status: {row.get('task_status') or 'unknown'}",
        f"run_status: {row.get('run_status') or 'unknown'}",
    ]
    for key in [
        "apprentice_agent",
        "mentor_model_provider",
        "attempts",
        "traced_steps",
        "artifact_count",
        "process_supervision_rows",
        "reward_modeling_rows",
        "revision_preference_pairs",
        "expected_economic_value",
        "expected_economic_value_for_agent_apprentice",
    ]:
        if row.get(key) is not None:
            lines.append(f"{key}: {row.get(key)}")
    mode = row.get("apprenticeship_mode") or row.get("mentor_mode")
    if mode:
        lines.append(f"apprenticeship_mode: {apprenticeship_mode_display(normalize_apprenticeship_mode(str(mode)))}")
    if row.get("domains"):
        lines.append(f"domains: {row.get('domains')}")
    if row.get("subdomains"):
        lines.append(f"subdomains: {row.get('subdomains')}")
    lines.extend(
        [
            f"has Full Experience Compilation: {'yes' if available.get('full_experience_compilation', available.get('full_training_package', row.get('kind') == 'agent_experience_package')) else 'no'}",
            f"has runtime training: {'yes' if available.get('runtime_training', row.get('kind') == 'agent_experience_package') else 'no'}",
            f"skill title: {compiler.get('skill_title') or row.get('title') or 'unknown'}",
            f"skill summary: {compiler.get('skill_summary') or compiler.get('runtime_training_summary') or 'available when Experience Compiler outputs are present'}",
            f"environment domain: {compiler.get('environment_domain') or row.get('environment_domain') or row.get('domain') or 'unknown'}",
            f"Experience Compiler status: {compiler.get('status') or row.get('experience_compiler_status') or row.get('tdo_status') or 'unknown'}",
        ]
    )
    installed = [
        skill for skill in list_installed_skills(get_settings())
        if item_id in {str(skill.get("source_item_id") or ""), str(skill.get("source_package_id") or "")}
    ]
    lines.append(f"local installed status: {'installed' if installed else 'not installed'}")
    lines.extend(
        [
            "",
            "Available actions:",
            "",
            "1. Export Full Experience Compilation",
            "   Use this when you want the complete traces, artifacts, training data, evaluation data, runtime learning, and generated data packages for research/training/evaluation pipelines.",
            "   Command:",
            f"   apprentice ecosystem export {item_id} --full --output <path>",
            "",
            "2. Install Runtime Training",
            "   Use this when you want your Apprentice Agent to use the compiled skill/runtime learning in future runs.",
            "   Command:",
            f"   apprentice learn install {item_id}",
        ]
    )
    return "\n".join(lines)


def _local_package_registry_row(path: Path) -> dict:
    package = _locate_package_root(path)
    manifest = {}
    for candidate in [
        package / "contribution_manifest.json",
        package / "task_manifest.json",
        package / "package_manifest.json",
        package / "manifest.json",
    ]:
        if candidate.exists():
            try:
                manifest = read_json(candidate)
                break
            except Exception:
                manifest = {}
    compiler = package / "experience_compiler"
    canonical = compiler / "skill_pack" / "canonical_skill.json"
    skill = read_json(canonical) if canonical.exists() else {}
    compiler_report = read_json(compiler / "compiler_report.json") if (compiler / "compiler_report.json").exists() else {}
    item_id = str(manifest.get("item_id") or manifest.get("package_id") or package.name)
    return {
        "id": item_id,
        "kind": "agent_experience_package",
        "title": skill.get("title") or manifest.get("title") or package.name,
        "summary": skill.get("summary") or compiler_report.get("description"),
        "domain": manifest.get("domain"),
        "subdomain": manifest.get("subdomain"),
        "package_id": manifest.get("package_id") or item_id,
        "task_id": manifest.get("task_id"),
        "local_bundle_path": str(package),
        "experience_compiler_status": compiler_report.get("compiler_status") or compiler_report.get("tdo_status") or ("completed" if compiler.exists() else "unknown"),
        "available_outputs": {
            "full_experience_compilation": True,
            "full_training_package": True,
            "runtime_training": bool((compiler / "skill_pack" / "skill.md").exists()),
            "skill_pack": (compiler / "skill_pack").exists(),
            "training_time_package": (compiler / "training_time_package").exists(),
            "eval_package": (compiler / "eval_package").exists(),
            "environment_package": (compiler / "environment").exists(),
        },
        "experience_compiler": {
            "enabled": compiler.exists(),
            "status": compiler_report.get("compiler_status") or compiler_report.get("tdo_status") or ("completed" if compiler.exists() else "unknown"),
            "skill_title": skill.get("title") or manifest.get("title") or package.name,
            "skill_summary": skill.get("summary") or compiler_report.get("description"),
            "runtime_training_summary": skill.get("summary") or "Runtime training available from local Experience Compiler outputs.",
            "training_data_counts": compiler_report.get("rows_accepted_by_type") or {},
            "environment_domain": manifest.get("environment_domain") or manifest.get("domain"),
        },
    }


@ecosystem_app.command('inspect')
def ecosystem_inspect(
    bundle_id: str=typer.Argument(...),
    registry: Path | None=typer.Option(None, '--registry', help='Public ecosystem index JSON path for offline/test use.'),
    storage: str | None=typer.Option(None, '--storage', help='Storage backend: forsy-r2 or github.'),
):
    local_arg = Path(bundle_id).expanduser()
    if local_arg.exists():
        typer.echo(_summarize_contribution_registry_row(_local_package_registry_row(local_arg)))
        return
    bundle_path=get_settings().app_home / 'ecosystem' / 'bundles' / bundle_id
    if bundle_path.exists() and (bundle_path/'contribution_manifest.json').exists():
        typer.echo(_summarize_bundle_manifest(read_json(bundle_path/'contribution_manifest.json')))
        return
    try:
        if normalize_ecosystem_storage(storage or get_settings().ecosystem_storage_backend) == "forsy-r2" and registry is None:
            row=fetch_r2_item(bundle_id, get_settings())
        else:
            row=_find_registry_row(bundle_id, registry, storage=storage)
    except FileNotFoundError as exc:
        typer.echo(str(exc))
        raise typer.Exit(1)
    if _is_seed_registry_row(row):
        typer.echo(_summarize_seed_registry_row(row))
        return
    typer.echo(_summarize_contribution_registry_row(row))


@bundle_app.command('inspect')
def bundle_inspect(path: Path=typer.Argument(..., help='Experience Compilation folder.')):
    from .loop_review import summarize_loop_reviews

    manifest=path/'contribution_manifest.json'
    if not manifest.exists():
        typer.echo(f'Experience Compilation manifest not found: {manifest}')
        raise typer.Exit(1)
    manifest_data=read_json(manifest)
    artifacts=_artifact_index(path)
    followups=[event for event in _session_events(path) if event.get("event_type") == "user_followup"]
    checkpoints=list((path / "mentor_checkpoints").glob("*.json")) if (path / "mentor_checkpoints").exists() else []
    typer.echo(_summarize_bundle_manifest(manifest_data))
    session=_session_metadata(path)
    if session.get("apprentice_agent"):
        typer.echo(f"apprentice_agent: {session.get('apprentice_agent')}")
    if session.get("model_provider"):
        typer.echo(f"mentor_model_provider: {session.get('model_provider')}")
    typer.echo(f"artifact_count: {len(artifacts)}")
    typer.echo(f"follow_up_count: {len(followups)}")
    if checkpoints:
        typer.echo(f"mentor_checkpoint_count: {len(checkpoints)}")
    loop_summary = summarize_loop_reviews(path)
    if loop_summary.get("has_loop_review_packets"):
        typer.echo(f"loop_review_packets: {loop_summary.get('loop_iteration_count')}")
        typer.echo(f"loop_final_verdict: {loop_summary.get('final_loop_verdict') or 'unknown'}")
        reviewers = ", ".join(loop_summary.get("reviewer_types") or []) or "unknown"
        typer.echo(f"loop_reviewer_types: {reviewers}")
        refs = loop_summary.get("loop_review_refs") or []
        if refs:
            typer.echo(f"latest_review_packet: {refs[-1]}")
    top=[row.get("package_relative_path") or row.get("artifact_ref") for row in artifacts[:5]]
    if top:
        typer.echo("top_artifacts:")
        for ref in top:
            typer.echo(f"- {ref}")
    typer.echo("")
    typer.echo(f"Next: apprentice ecosystem contribute {path}")


@bundle_app.command('check')
def bundle_check(path: Path=typer.Argument(..., help='Experience Compilation folder.')):
    from .artifact_resolver import artifact_ref_candidates, artifact_ref_resolves, normalize_artifact_ref

    issues: list[str] = []
    bundle = Path(path).expanduser()
    from .io import validate_trace_tree
    try:
        validate_trace_tree(bundle)
    except ValueError as exc:
        issues.append(str(exc))
    manifest_candidates = [
        bundle / "contribution_manifest.json",
        bundle / "task_manifest.json",
        bundle / "package_manifest.json",
        bundle / "manifest.json",
        bundle / "experience_compiler" / "compiler_manifest.json",
    ]
    manifest_path = next((candidate for candidate in manifest_candidates if candidate.exists()), manifest_candidates[0])
    if not bundle.exists():
        issues.append(f"Bundle path does not exist: {bundle}")
    elif not manifest_path.exists():
        candidate_names = ", ".join(str(candidate.relative_to(bundle)) for candidate in manifest_candidates)
        issues.append(f"Experience Compilation manifest not found. Checked: {candidate_names}")
    manifest_data = {}
    if manifest_path.exists():
        try:
            manifest_data = read_json(manifest_path)
        except Exception as exc:
            issues.append(f"{manifest_path.relative_to(bundle)} could not be parsed: {exc}")
    artifact_index_path = bundle / "outputs" / "artifacts_index.json"
    artifacts: list[dict] = []
    if artifact_index_path.exists():
        try:
            raw_artifacts = read_json(artifact_index_path)
            if isinstance(raw_artifacts, list):
                artifacts = raw_artifacts
            else:
                issues.append("outputs/artifacts_index.json must contain a JSON list.")
        except Exception as exc:
            issues.append(f"outputs/artifacts_index.json could not be parsed: {exc}")
    for row in artifacts:
        ref = row.get("package_relative_path") or row.get("artifact_ref")
        if ref:
            normalized = normalize_artifact_ref(ref)
            existing_refs = {
                str(p.relative_to(bundle)).replace("\\", "/")
                for p in bundle.rglob("*")
                if p.is_file()
            }
            candidates = artifact_ref_candidates(normalized) | {normalized}
            if not any((bundle / candidate).exists() for candidate in candidates) and not artifact_ref_resolves(normalized, existing_refs):
                issues.append(f"Referenced artifact is missing: {ref}")
    for optional in ["session_events.jsonl", "progress_events.jsonl"]:
        file_path = bundle / optional
        if file_path.exists():
            try:
                read_jsonl(file_path)
            except Exception as exc:
                issues.append(f"{optional} could not be parsed: {exc}")
    for ref in manifest_data.get("experience_compiler_output_refs") or []:
        normalized = normalize_artifact_ref(ref)
        if not (bundle / normalized).exists():
            issues.append(f"Referenced Experience Compiler output is missing: {ref}")
    for ref in manifest_data.get("tdo_output_refs") or []:
        normalized = normalize_artifact_ref(ref)
        if not (bundle / normalized).exists():
            issues.append(f"Referenced legacy learning output is missing: {ref}")
    for ref in manifest_data.get("loop_review_refs") or []:
        normalized = normalize_artifact_ref(ref)
        if not (bundle / normalized).exists():
            issues.append(f"Referenced loop review packet is missing: {ref}")
    tdo_dir = bundle / "tdo"
    if tdo_dir.exists():
        for file_path in tdo_dir.rglob("*.json"):
            try:
                read_json(file_path)
            except Exception as exc:
                issues.append(f"{file_path.relative_to(bundle)} could not be parsed: {exc}")
        for file_path in tdo_dir.rglob("*.jsonl"):
            try:
                read_jsonl(file_path)
            except Exception as exc:
                issues.append(f"{file_path.relative_to(bundle)} could not be parsed: {exc}")
    compiler_dir = bundle / "experience_compiler"
    if compiler_dir.exists():
        for file_path in compiler_dir.rglob("*.json"):
            try:
                read_json(file_path)
            except Exception as exc:
                issues.append(f"{file_path.relative_to(bundle)} could not be parsed: {exc}")
        source_map_path = compiler_dir / "source_evidence_map.json"
        if source_map_path.exists():
            try:
                source_map = read_json(source_map_path)
                evidence_ids = {
                    str(item.get("evidence_id"))
                    for item in source_map.get("evidence_catalog") or []
                    if isinstance(item, dict) and item.get("evidence_id")
                }
                for link in source_map.get("row_evidence_links") or []:
                    if not isinstance(link, dict):
                        continue
                    link_ids = link.get("evidence_ref_ids") or link.get("source_evidence_ids") or []
                    for evidence_id in link_ids:
                        if str(evidence_id) not in evidence_ids:
                            issues.append(f"source_evidence_map row {link.get('row_id')} references unknown evidence id: {evidence_id}")
                orphan_count = (
                    (source_map.get("coverage_statistics") or {}).get("orphan_evidence_id_count")
                    if isinstance(source_map.get("coverage_statistics"), dict)
                    else None
                )
                if orphan_count is None:
                    orphan_count = (source_map.get("coverage") or {}).get("orphan_evidence_id_count")
                if orphan_count not in (None, 0):
                    issues.append(f"source_evidence_map has orphan evidence ids: {orphan_count}")
            except Exception as exc:
                issues.append(f"{source_map_path.relative_to(bundle)} could not be validated: {exc}")
        seen_training_ids: set[str] = set()
        for file_path in compiler_dir.rglob("*.jsonl"):
            try:
                rows = read_jsonl(file_path)
            except Exception as exc:
                issues.append(f"{file_path.relative_to(bundle)} could not be parsed: {exc}")
                continue
            if "training_time_package" in file_path.parts or "eval_package" in file_path.parts or "loop_learning" in file_path.parts:
                for row in rows:
                    if not isinstance(row, dict):
                        continue
                    row_id = row.get("row_id") or row.get("pair_id")
                    if row_id:
                        key = f"{file_path.relative_to(compiler_dir)}:{row_id}"
                        if key in seen_training_ids:
                            issues.append(f"Duplicate generated row id: {row_id}")
                        seen_training_ids.add(key)
                    if row.get("data_type") and not (
                        row.get("source_evidence_ref_ids")
                        or row.get("source_evidence_ids")
                        or row.get("evidence_refs")
                    ):
                        issues.append(f"{file_path.relative_to(bundle)} row lacks evidence refs: {row_id or row.get('data_type')}")
    loops_dir = bundle / "loops"
    if loops_dir.exists():
        for file_path in loops_dir.rglob("*.json"):
            try:
                read_json(file_path)
            except Exception as exc:
                issues.append(f"{file_path.relative_to(bundle)} could not be parsed: {exc}")
        for file_path in loops_dir.rglob("*.jsonl"):
            try:
                read_jsonl(file_path)
            except Exception as exc:
                issues.append(f"{file_path.relative_to(bundle)} could not be parsed: {exc}")
    surface_warnings: list[str] = []
    for trace_path in [bundle / "traces" / "agent_traces.jsonl", bundle / "traces" / "raw_agent_traces.jsonl"]:
        if not trace_path.exists():
            continue
        try:
            for trace in read_jsonl(trace_path):
                for step in trace.get("steps") or []:
                    missing = step.get("missing_surface_fields") or []
                    status = step.get("surface_capture_status")
                    surface = step.get("interaction_surface")
                    if surface and status in {"partial", "unavailable"} and missing:
                        surface_warnings.append(
                            f"{trace_path.relative_to(bundle)} step {step.get('step')}: {surface} capture {status}; missing {', '.join(str(x) for x in missing[:6])}"
                        )
        except Exception:
            pass
    secret_hits = _bundle_secret_hits(bundle) if bundle.exists() else []
    if secret_hits:
        issues.append("Obvious secret-like values were found: " + ", ".join(secret_hits[:10]))
    if issues:
        typer.echo("Experience Compilation check: needs attention")
        typer.echo(f"Experience Compilation: {bundle}")
        for issue in issues:
            typer.echo(f"Reason: {redact_secrets(issue)}")
        typer.echo("Next action: fix the packaging issue, then rerun `apprentice bundle check`.")
        raise typer.Exit(1)
    status = manifest_data.get("task_status") or manifest_data.get("run_status") or "unknown"
    typer.echo("Experience Compilation check: passed")
    typer.echo(f"Experience Compilation: {bundle}")
    typer.echo(f"Task status: {status}")
    for warning in surface_warnings[:10]:
        typer.echo(f"Surface completeness warning: {redact_secrets(warning)}")
    typer.echo("This Experience Compilation is ready for local export, Runtime Training install, or explicit sharing.")


@bundle_app.command('contribute')
def bundle_contribute(path: Path=typer.Argument(..., help='Experience Compilation folder.')):
    manifest=path/'contribution_manifest.json'
    if not manifest.exists():
        typer.echo(f'Experience Compilation manifest not found: {manifest}')
        raise typer.Exit(1)
    try:
        submission_dir, package_zip, metadata = _create_ecosystem_submission(path)
    except Exception as exc:
        typer.echo(f"Could not prepare public ecosystem contribution: {redact_secrets(str(exc))}")
        raise typer.Exit(1)
    typer.echo('Experience Compilation ready.')
    typer.echo('')
    typer.echo(f"Bundle ID: {metadata.get('bundle_id')}")
    typer.echo(f"Title: {metadata.get('title')}")
    typer.echo('Experience Compilation:')
    typer.echo(str(path))
    typer.echo('')
    typer.echo('No upload was performed by this command.')
    typer.echo(f"Submission package: {package_zip.name}")
    typer.echo("Submission metadata: ecosystem_submission/ecosystem_submission.json")
    typer.echo('')
    typer.echo('Contribute to public ecosystem:')
    typer.echo(f'apprentice ecosystem contribute {path}')
    typer.echo('')
    typer.echo('Public ecosystem:')
    typer.echo(_ecosystem_repo_url())
    typer.echo('')
    typer.echo('View files:')
    typer.echo(f'open {path}')

@app.command('init-env', hidden=True)
def init_env():
    dst=Path('.env.local')
    if not dst.exists(): shutil.copyfile('.env.example', dst); typer.echo('created .env.local from .env.example')
    else: typer.echo('.env.local already exists')

@app.command('run-task', hidden=True)
def run_task(input: Path=typer.Option(Path('data/seed_tasks/hard_finance_reconciliation.jsonl')), output_root: Path=Path('outputs'), runner: str='deterministic'):
    raw=RawTaskRecord.model_validate(read_jsonl(input)[0])
    run_root=output_root/'runs'/'single'
    pkg=run_one(raw, run_root, runner=runner); typer.echo(str(pkg))

@app.command('run-batch', hidden=True)
def run_batch(input: Path=typer.Option(...), limit: int|None=None, resume: bool=False, max_parallel: int=1, retry_limit: int=0, task_timeout_seconds: int=900, runner: str='deterministic', release_id: str|None=None, output_root: Path=Path('outputs'), max_iterations: int|None=None):
    typer.echo(str(run_batch_impl(input, output_root, limit, resume, max_parallel, retry_limit, task_timeout_seconds, runner, release_id, max_iterations=max_iterations)))

@app.command('run-many', hidden=True)
def run_many(input: Path=typer.Option(...), limit: int|None=None, resume: bool=False, max_parallel: int=1, retry_limit: int=0, task_timeout_seconds: int=900, runner: str='deterministic', release_id: str|None=None, output_root: Path=Path('outputs'), max_iterations: int|None=None):
    typer.echo(str(run_batch_impl(input, output_root, limit, resume, max_parallel, retry_limit, task_timeout_seconds, runner, release_id, max_iterations=max_iterations)))

@app.command('create-bundle', hidden=True)
def create_bundle(
    run_root: Path=typer.Option(...),
    bundle_root: Path|None=typer.Option(None),
    include_debug: bool=typer.Option(False, '--include-debug', help='Include debug validation reports in the bundle.'),
    release_style: bool=typer.Option(False, '--release-style', help='Create the full internal release-style export.'),
):
    typer.echo(str(create_contribution_bundle(run_root, bundle_root, include_debug=include_debug, release_style=release_style)))

@app.command('create-release', hidden=True)
def create_release(run_root: Path=typer.Option(Path('outputs/runs/single')), release_root: Path=typer.Option(Path('outputs/releases/manual'))):
    typer.echo(str(create_release_impl(run_root, release_root)))

@app.command('validate-release', hidden=True)
def validate_release(release_root: Path=typer.Option(...)):
    c=validate_release_impl(release_root); typer.echo(format_counters(c)); raise typer.Exit(0 if c['release_valid'] else 1)

@app.command('validate-public-release', hidden=True)
def validate_public_release(release_root: Path=typer.Option(...)):
    public=release_root/'public' if (release_root/'public').exists() else release_root
    c=validate_release_impl(release_root if (release_root/'public').exists() else release_root.parent) if public.name == 'public' else validate_release_impl(release_root)
    typer.echo(format_counters({k:v for k,v in c.items() if k.startswith('public_') or k in ['secret_scan_ok']}))
    raise typer.Exit(0 if c.get('public_release_valid') else 1)


@app.command('repair-roles', hidden=True)
def repair_roles(run_root: Path=typer.Option(...), task_id: str=typer.Option(...), roles: str=typer.Option('evaluator_agent,grader_agent,verifier_agent'), attempts: str=typer.Option('baseline,revised'), rebuild_release: Path|None=typer.Option(None)):
    """Re-run model evaluation roles from existing task package artifacts/traces."""
    from .schemas import RubricSpec, ActualOutputs, WorkflowTrace, GraderResult, VerifierResult
    from .grader import grade_attempt, apply_score_reliability
    from .verifier import verify_attempt
    from .evaluator import evaluate_attempt
    from .io import read_workflow_json as read_json, write_json
    from datetime import datetime, timezone
    import shutil
    pkg=run_root/'packages'/task_id
    rubric=RubricSpec.model_validate(read_json(pkg/'rubric/rubric.json'))
    selected={x.strip() for x in roles.split(',') if x.strip()}
    selected_attempts=[x.strip() for x in attempts.split(',') if x.strip()]
    role_root=run_root/'roles'/task_id
    archive_root=role_root/'repair_archive'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    for attempt in selected_attempts:
        actual_path=pkg/'attempts'/attempt/'actual_outputs.json'
        trace_path=pkg/'attempts'/attempt/'agent_trace.json'
        if not actual_path.exists() or not trace_path.exists():
            typer.echo(f'skipping {attempt}: missing existing attempt outputs')
            continue
        actual=ActualOutputs.model_validate(read_json(actual_path))
        trace=WorkflowTrace.model_validate(read_json(trace_path))
        grade_path=pkg/'grading'/f'{attempt}_grader_result.json'
        ver_path=pkg/'grading'/f'{attempt}_verifier_result.json'
        grader=GraderResult.model_validate(read_json(grade_path)) if grade_path.exists() else None
        verifier=VerifierResult.model_validate(read_json(ver_path)) if ver_path.exists() else None
        if 'grader_agent' in selected or grader is None:
            old=role_root/'grader_agent'/attempt
            if old.exists():
                archive_root.mkdir(parents=True, exist_ok=True); shutil.copytree(old, archive_root/f'grader_agent_{attempt}', dirs_exist_ok=True)
            grader=grade_attempt(rubric, actual, attempt, trace, role_root, pkg)
        if 'verifier_agent' in selected or verifier is None:
            old=role_root/'verifier_agent'/attempt
            if old.exists():
                archive_root.mkdir(parents=True, exist_ok=True); shutil.copytree(old, archive_root/f'verifier_agent_{attempt}', dirs_exist_ok=True)
            verifier=verify_attempt(grader, actual, trace, role_root, pkg)
        grader=apply_score_reliability(grader, verifier)
        grader.metadata_json['repair_run']=True; verifier.metadata_json['repair_run']=True
        write_json(grade_path, grader); write_json(ver_path, verifier)
        if 'evaluator_agent' in selected:
            old=role_root/'evaluator_agent'/attempt
            if old.exists():
                archive_root.mkdir(parents=True, exist_ok=True); shutil.copytree(old, archive_root/f'evaluator_agent_{attempt}', dirs_exist_ok=True)
            target=f'{task_id}_revised' if attempt == 'baseline' else f'{task_id}_{attempt}_followup'
            fb,rp=evaluate_attempt(grader, verifier, actual, trace, target, role_root, pkg)
            fb.metadata_json['repair_run']=True; rp.metadata_json['repair_run']=True
            if attempt == 'baseline':
                write_json(pkg/'feedback/baseline_evaluator_feedback.json', fb); write_json(pkg/'feedback/revision_plan.json', rp)
            else:
                write_json(pkg/'feedback'/f'{attempt}_evaluator_feedback.json', fb)
        typer.echo(f'repaired roles for task_id={task_id} attempt={attempt}')
    if rebuild_release is not None:
        from .release_exporter import create_release
        create_release(run_root, rebuild_release)
        typer.echo(f'rebuilt release={rebuild_release}')


@app.command('summarize-releases', hidden=True)
def summarize_releases(release_root: Path=typer.Option(Path('outputs/releases')), pattern: str=typer.Option('tasks-*')):
    """Summarize release readiness across a release directory."""
    from .validation import validate_release
    totals={'total_releases':0,'green_scale_ready_count':0,'release_valid_false_count':0,'fallback_only_count':0,'model_role_incomplete_count':0,'model_score_count_lt_2_count':0,'failed_verification_count':0}
    needing=[]; warnings=[]; clean=[]
    for rel in sorted(release_root.glob(pattern)):
        if not rel.is_dir():
            continue
        totals['total_releases'] += 1
        c=validate_release(rel)
        task_ids=[]
        try:
            task_ids=[str(r.get('task_id') or r.get('raw_task_id')) for r in read_jsonl(rel/'tasks.jsonl')]
        except Exception:
            task_ids=[rel.name]
        label=','.join(task_ids) or rel.name
        if c.get('scale_ready'):
            totals['green_scale_ready_count'] += 1
            if c.get('scale_warnings'):
                warnings.append(label)
            else:
                clean.append(label)
        else:
            needing.append(label)
        if not c.get('release_valid'): totals['release_valid_false_count'] += 1
        if c.get('fallback_only_task_count'): totals['fallback_only_count'] += 1
        if not c.get('model_role_completeness_ok'): totals['model_role_incomplete_count'] += 1
        if int(c.get('model_score_count') or 0) < 2: totals['model_score_count_lt_2_count'] += 1
        if c.get('verifier_failed_count'): totals['failed_verification_count'] += 1
    for k,v in totals.items():
        typer.echo(f'{k}={v}')
    typer.echo('task_ids_needing_rerun=' + ','.join(needing))
    typer.echo('task_ids_publishable_with_warnings=' + ','.join(warnings))
    typer.echo('task_ids_clean_publishable=' + ','.join(clean))

@app.command('inspect-trace')
def inspect_trace(trace_path: Path):
    from .atif import load_trace
    t=WorkflowTrace.model_validate(load_trace(trace_path)); typer.echo(f'trace_id={t.trace_id}\ntask_id={t.task_id}\nsteps={len(t.steps)}')

@app.command('codex-smoke', hidden=True)
def codex_smoke():
    if not shutil.which('codex'):
        typer.echo('codex_available=false'); raise typer.Exit(1)
    typer.echo('codex_available=true')

@app.command('llm-smoke', hidden=True)
def llm_smoke(
    output_dir: Path=Path('outputs/llm_smoke'),
    provider: str | None=typer.Option(None, '--provider', help='Mentor Model Provider id to test. Defaults to configured provider.'),
):
    if provider is not None and provider not in MODEL_PROVIDER_RECIPES:
        raise typer.BadParameter(f'Mentor Model Provider must be one of: {", ".join(MODEL_PROVIDER_RECIPES)}')
    counters=run_llm_smoke(output_dir, provider_id=provider)
    typer.echo(format_smoke_counters(counters))
    roles=['intake','rubric','grader','verifier','evaluator']
    ok=counters.get('mentor_model_provider_available') and all(counters.get(f'{r}_live_call_ok') and counters.get(f'{r}_structured_output_validation_ok') for r in roles) and counters.get('secret_scan_ok')
    raise typer.Exit(0 if ok else 1)

def main(): app()
if __name__=='__main__': main()
