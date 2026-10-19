from __future__ import annotations

import re
import shlex
import shutil
import os
from datetime import datetime, timezone
from pathlib import Path

from .bundle_exporter import create_contribution_bundle
from .config import Settings, apprentice_agent_display_name, apprentice_agent_readiness_status, get_settings, settings_override
from .env import redact_secrets
from .io import read_workflow_json as read_json, write_json
from .loop import run_task
from .loop_review import record_followup_review_update, record_human_loop_feedback
from .mentor_checkpoints import write_mentor_checkpoints
from .progress import ProgressCallback, append_progress_event, update_run_status
from .recipes import WORKER_AGENT_RECIPES
from .schemas import RawTaskRecord
from .session_events import append_session_event, backfill_session_event_task_ids, next_followup_index
from .tdo import run_tdo_for_run, utc_now


def _write_failed_tdo_reports(run_root: Path, settings: Settings, error: str) -> None:
    packages = sorted((run_root / "packages").glob("*")) if (run_root / "packages").exists() else []
    for pkg in packages:
        tdo_dir = pkg / "tdo"
        tdo_dir.mkdir(parents=True, exist_ok=True)
        safe_pkg = re.sub(r"[^A-Za-z0-9_]+", "_", pkg.name).strip("_") or "package"
        report = {
            "tdo_id": f"tdo_{safe_pkg}_failed",
            "source_package_id": pkg.name,
            "source_task_id": pkg.name,
            "created_at": utc_now(),
            "generator_mode": settings.worker_agent,
            "tdo_judge_source": "apprentice_self_judge",
            "mentor_audited": False,
            "rounds_run": 0,
            "stop_reason": "tdo_runtime_error",
            "tdo_status": "failed",
            "rows_generated_by_type": {},
            "rows_accepted_by_type": {},
            "rows_rejected_by_type": {},
            "schema_validation_passed": False,
            "sanitization_passed": False,
            "weak_strong_proxy_available": False,
            "grpo_suitability_distribution": {},
            "omitted_outputs": ["experience_compiler_outputs"],
            "generation_notes": [error],
            "recommended_uses": [],
            "metadata_json": {"failure_recorded_by": "runtime_guardrail"},
        }
        write_json(tdo_dir / "tdo_report.json", report)
        write_json(tdo_dir / "tdo_manifest.json", {
            "schema_version": "aa-experience-compiler-v0.2",
            "tdo_id": report["tdo_id"],
            "tdo_status": "failed",
            "source_package_id": pkg.name,
            "source_task_id": pkg.name,
            "tdo_output_refs": ["tdo/tdo_report.json", "tdo/tdo_manifest.json"],
            "tdo_judge_source": "apprentice_self_judge",
            "mentor_audited": False,
            "rounds_run": 0,
            "rows_by_type": {},
            "created_at": report["created_at"],
        })
        compiler_dir = pkg / "experience_compiler"
        compiler_dir.mkdir(parents=True, exist_ok=True)
        write_json(compiler_dir / "compiler_report.json", {
            "compiler_status": "failed",
            "source_package_id": pkg.name,
            "error": error,
            "created_at": report["created_at"],
        })
        write_json(compiler_dir / "compiler_manifest.json", {
            "schema_version": "aa-experience-compiler-v0.2",
            "compiler_status": "failed",
            "source_package_id": pkg.name,
            "compiler_output_refs": ["experience_compiler/compiler_report.json", "experience_compiler/compiler_manifest.json"],
            "created_at": report["created_at"],
        })


def _safe_run_tdo_for_run(
    run_root: Path,
    settings: Settings,
    *,
    progress_callback: ProgressCallback | None = None,
    followup_index: int | None = None,
    record_only: bool = False,
) -> dict:
    try:
        return run_tdo_for_run(
            run_root,
            settings,
            progress_callback=progress_callback,
            followup_index=followup_index,
            record_only=record_only,
        )
    except Exception as exc:
        safe = redact_secrets(str(exc))[:500]
        _write_failed_tdo_reports(run_root, settings, safe)
        write_json(run_root / "tdo_status.json", {"tdo_status": "failed", "error": safe, "updated_at": utc_now()})
        append_progress_event(
            run_root,
            "tdo_failed",
            run_id=run_root.name,
            message=f"Experience Compiler failed: {safe}",
            phase="experience_compiler_failed",
            metadata_json={"followup_index": followup_index} if followup_index else None,
            callback=progress_callback,
        )
        return {"tdo_status": "failed", "error": safe}


class RunInterrupted(Exception):
    def __init__(self, run_root: Path, message: str = "Run interrupted by user."):
        super().__init__(message)
        self.run_root = run_root
        self.message = message


def slugify(text: str, fallback: str = "task") -> str:
    value = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return (value[:64].strip("-") or fallback)


def make_run_id(instruction: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d-%H%M%S")
    return f"{stamp}-{slugify(instruction)}"


def run_root_for(run_id: str, settings: Settings | None = None) -> Path:
    settings = settings or get_settings()
    candidate = Path(run_id).expanduser()
    if candidate.exists() or candidate.is_absolute() or "/" in run_id:
        return candidate
    return settings.app_home / "runs" / run_id


def _unique_target(dest_dir: Path, name: str) -> Path:
    target = dest_dir / name
    if not target.exists():
        return target
    stem = target.stem
    suffix = target.suffix
    for i in range(2, 1000):
        candidate = dest_dir / f"{stem}-{i}{suffix}"
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Could not create unique asset target for {name}")


def copy_assets(assets: list[Path] | None, dest_dir: Path, rel_prefix: str) -> list[str]:
    copied: list[str] = []
    if not assets:
        return copied
    dest_dir.mkdir(parents=True, exist_ok=True)
    for raw_path in assets:
        src = Path(raw_path).expanduser()
        if not src.exists():
            raise FileNotFoundError(f"Asset does not exist: {src}")
        target = _unique_target(dest_dir, src.name)
        if src.is_dir():
            shutil.copytree(src, target)
        else:
            shutil.copy2(src, target)
        copied.append(f"{rel_prefix.rstrip('/')}/{target.name}")
    return copied


def _asset_abs_refs(run_root: Path, rel_refs: list[str]) -> list[str]:
    refs = []
    for rel in rel_refs:
        p = run_root / rel
        refs.append(str(p))
    return refs


def prompt_to_raw_task(run_id: str, instruction: str, asset_refs: list[str] | None = None) -> RawTaskRecord:
    title = instruction.strip().splitlines()[0][:90] or "Agent Apprenticeship task"
    return RawTaskRecord(
        raw_task_id=f"raw_{slugify(run_id)}",
        source_kind="user_prompt",
        raw_title=title,
        raw_description=instruction,
        raw_payload={
            "expected_deliverable": "Completed task deliverables under artifacts/.",
            "input_requirements": [Path(ref).name for ref in asset_refs or []],
            "output_requirements": ["final deliverable"],
            "privacy_classification": "unknown",
            "task_entrypoint": "direct_prompt",
        },
        input_artifact_refs=asset_refs or [],
        task_id=f"task_{slugify(run_id)}",
        normalized_title=title,
        normalized_instruction=instruction,
        expected_deliverable="Completed task deliverables under artifacts/.",
        metadata_json={"created_by": "agent-apprenticeship-run"},
    )


def task_id_for_run_id(run_id: str) -> str:
    return f"task_{slugify(run_id)}"


def _worker_display(settings: Settings) -> str:
    return settings.custom_worker_display_name or settings.worker_agent.replace("-", " ").title()


def apprentice_agent_display(settings: Settings) -> str:
    return apprentice_agent_display_name(settings)


def _configured_apprentice_command(settings: Settings, override: str | None = None) -> str | None:
    if override == "deterministic":
        return None
    if settings.worker_agent == "custom":
        template = settings.custom_worker_command_template or ""
        try:
            return shlex.split(template)[0] if template else settings.worker_agent_command
        except ValueError:
            return template.split()[0] if template.split() else settings.worker_agent_command
    recipe = WORKER_AGENT_RECIPES.get(settings.worker_agent)
    return settings.worker_agent_command or (recipe.command_name if recipe else settings.worker_agent)


def apprentice_agent_readiness(settings: Settings, override: str | None = None) -> tuple[bool, str | None]:
    if override == "deterministic":
        return True, None
    status = apprentice_agent_readiness_status(settings)
    if status["status"] == "ready":
        return True, None
    return False, status.get("reason") or str(status["status"])


def _human_checkpoint_mode(settings: Settings) -> bool:
    return settings.mentor_mode in {"expert_led", "hybrid"}


def _checkpoint_auto_approve(settings: Settings) -> bool:
    return (
        (settings.mentor_mode == "expert_led" and os.getenv("AA_EXPERT_AUTO_APPROVE") == "1")
        or (settings.mentor_mode == "hybrid" and os.getenv("AA_HYBRID_AUTO_APPROVE") == "1")
    )


def _checkpoint_review_labels(settings: Settings, revision_requested: bool | None = None) -> tuple[str, str]:
    if settings.mentor_mode == "hybrid":
        pending = "Expert-led model audit draft complete - human confirmation pending"
        if revision_requested is None:
            return pending, "Expert-led human confirmation complete"
        return pending, "Expert-led human confirmation complete - revision requested" if revision_requested else "Expert-led human confirmation complete - finish selected"
    pending = "Expert audit pending"
    if revision_requested is None:
        return pending, "Expert audit complete"
    return pending, "Expert audit complete - revision requested" if revision_requested else "Expert audit complete - finish selected"


def _pre_attempt_checkpoint_callback(
    run_root: Path,
    settings: Settings,
    *,
    progress_callback: ProgressCallback | None,
    followup_index: int | None = None,
):
    def _callback(_pkg: Path) -> None:
        append_progress_event(
            run_root,
            "apprentice_task_spec_ready",
            run_id=run_root.name,
            message=("Follow-up %s Apprentice Agent updated task-intake/rubric draft" % followup_index if followup_index else "Apprentice Agent drafted task intake and rubric"),
            current_loop=1,
            maximum_improvement_loops=settings.max_improvement_loops,
            phase="task_intake_rubric",
            metadata_json={"followup_index": followup_index, "draft_author": "apprentice_agent"} if followup_index else {"draft_author": "apprentice_agent"},
            callback=progress_callback,
        )
        if _human_checkpoint_mode(settings):
            write_mentor_checkpoints(
                run_root,
                settings,
                auto_approve=_checkpoint_auto_approve(settings),
                stages=("task_intake", "rubric"),
                preserve_interactive=followup_index is None,
            )
        append_progress_event(
            run_root,
            "apprentice_attempt_started",
            run_id=run_root.name,
            message=(f"Follow-up {followup_index} Apprentice attempt started" if followup_index else "Apprentice attempt started"),
            current_loop=1,
            maximum_improvement_loops=settings.max_improvement_loops,
            phase="apprentice_attempt",
            metadata_json={"followup_index": followup_index} if followup_index else None,
            callback=progress_callback,
        )

    return _callback


def _append_mentor_preparation_started(
    run_root: Path,
    settings: Settings,
    *,
    progress_callback: ProgressCallback | None,
    followup_index: int | None = None,
) -> None:
    append_progress_event(
        run_root,
        "mentor_preparation_started",
        run_id=run_root.name,
        message=(f"Follow-up {followup_index} Mentor preparation started" if followup_index else "Mentor preparation started"),
        current_loop=1,
        maximum_improvement_loops=settings.max_improvement_loops,
        phase="mentor_preparation",
        metadata_json={"followup_index": followup_index} if followup_index else None,
        callback=progress_callback,
    )


def _revision_decision_callback(
    run_root: Path,
    settings: Settings,
    *,
    progress_callback: ProgressCallback | None,
    followup_index: int | None = None,
):
    if not _human_checkpoint_mode(settings):
        return None, {}
    state: dict[str, bool] = {}

    def _callback(pkg: Path) -> bool:
        status, _reason = _session_status_for_package(pkg)
        traced_steps, artifact_count, artifacts_path, operational_error = _package_progress_summary(run_root, pkg)
        if not state.get("apprentice_completed_emitted"):
            append_progress_event(
                run_root,
                "apprentice_attempt_completed",
                run_id=run_root.name,
                message=("Apprentice attempt failed - operational error" if operational_error else ("Follow-up %s Apprentice attempt complete" % followup_index if followup_index else "Apprentice attempt complete")),
                current_loop=1,
                maximum_improvement_loops=settings.max_improvement_loops,
                phase="apprentice_attempt_complete",
                traced_steps=traced_steps,
                artifact_count=artifact_count,
                artifacts_path=artifacts_path,
                operational_error=operational_error,
                metadata_json={"followup_index": followup_index} if followup_index else None,
                callback=progress_callback,
            )
            state["apprentice_completed_emitted"] = True
        if operational_error:
            return False
        pending, _ = _checkpoint_review_labels(settings)
        review_packet = pkg / "loops" / "iterations" / "001" / "review_packet.md"
        append_progress_event(
            run_root,
            "mentor_review_started",
            run_id=run_root.name,
            message=f"{pending} - review packet ready",
            current_loop=1,
            maximum_improvement_loops=settings.max_improvement_loops,
            phase="expert_review" if settings.mentor_mode == "expert_led" else "hybrid_human_approval",
            metadata_json={
                **({"followup_index": followup_index} if followup_index else {}),
                "review_packet_ref": str(review_packet.relative_to(pkg)) if review_packet.exists() else None,
                "review_packet_path": str(review_packet) if review_packet.exists() else None,
            },
            callback=progress_callback,
        )
        update_run_status(
            run_root,
            run_status=status,
            task_status=status,
            current_phase="expert_review_pending" if settings.mentor_mode == "expert_led" else "hybrid_human_approval_pending",
            latest_message=pending,
            traced_steps=traced_steps,
            artifact_count=artifact_count,
            artifacts_path=str(artifacts_path),
        )
        write_mentor_checkpoints(
            run_root,
            settings,
            auto_approve=_checkpoint_auto_approve(settings),
            stages=("evaluation", "revision"),
            preserve_interactive=followup_index is None,
        )
        revision_path = run_root / "mentor_checkpoints" / "revision_checkpoint.json"
        revision = read_json(revision_path) if revision_path.exists() else {}
        evaluation_path = run_root / "mentor_checkpoints" / "evaluation_checkpoint.json"
        evaluation = read_json(evaluation_path) if evaluation_path.exists() else {}
        revision_requested = bool(revision.get("revision_should_run")) and settings.max_improvement_loops > 1
        record_human_loop_feedback(
            pkg,
            iteration=1,
            checkpoint={**evaluation, **revision, "evidence_refs": ["loops/iterations/001/review_packet.json", "loops/iterations/001/review_packet.md"]},
            revision_requested=revision_requested,
        )
        _, complete = _checkpoint_review_labels(settings, revision_requested)
        append_progress_event(
            run_root,
            "mentor_review_completed",
            run_id=run_root.name,
            message=complete,
            current_loop=1,
            maximum_improvement_loops=settings.max_improvement_loops,
            phase="expert_review_complete" if settings.mentor_mode == "expert_led" else "hybrid_human_approval_complete",
            task_status=status,
            traced_steps=traced_steps,
            artifact_count=artifact_count,
            artifacts_path=artifacts_path,
            metadata_json={"followup_index": followup_index} if followup_index else None,
            callback=progress_callback,
        )
        if revision_requested:
            append_progress_event(
                run_root,
                "revision_started",
                run_id=run_root.name,
                message=("Follow-up %s revision attempt started" % followup_index if followup_index else "Revision attempt started"),
                current_loop=2,
                maximum_improvement_loops=settings.max_improvement_loops,
                phase="revision_attempt",
                metadata_json={"followup_index": followup_index} if followup_index else None,
                callback=progress_callback,
            )
            state["revision_started_emitted"] = True
        return revision_requested

    return _callback, state


def runner_for_settings(settings: Settings, override: str | None = None) -> str:
    if override:
        return override
    if settings.worker_agent == "custom":
        return "custom"
    if settings.worker_runner == "deterministic":
        return "deterministic"
    if settings.worker_agent == "codex":
        return "codex"
    if settings.worker_agent in WORKER_AGENT_RECIPES:
        return settings.worker_agent
    raise RuntimeError(f"Unsupported Apprentice Agent: {settings.worker_agent}")


def _loop_settings_for_run(settings: Settings) -> Settings:
    if settings.mentor_mode != "expert_led":
        return settings
    return settings.model_copy(
        update={
            "rubric_mode": "deterministic",
            "llm_task_intake_enabled": False,
            "llm_rubric_generation_enabled": False,
            "llm_evaluator_enabled": False,
            "llm_grader_enabled": False,
            "llm_verifier_enabled": False,
        }
    )


def _settings_for_session(settings: Settings, session: dict) -> Settings:
    updates = {}
    if session.get("mentor_mode"):
        updates["mentor_mode"] = session["mentor_mode"]
    if session.get("sensitive_info_masking"):
        updates["sensitive_info_masking"] = session["sensitive_info_masking"]
    if session.get("model_provider"):
        updates["model_provider"] = session["model_provider"]
    if session.get("max_improvement_loops"):
        loops = int(session["max_improvement_loops"])
        updates["max_improvement_loops"] = loops
        updates["max_iterations"] = loops
    return settings.model_copy(update=updates) if updates else settings


def _session_status_for_package(pkg: Path) -> tuple[str, str | None]:
    actual_paths = sorted((pkg / "attempts").glob("*/actual_outputs.json"))
    if not actual_paths:
        return "partial", "No attempt outputs were found."
    statuses = []
    operational_errors = []
    for path in actual_paths:
        try:
            data = read_json(path)
        except Exception:
            return "partial", f"Could not read {path.relative_to(pkg)}."
        metadata = data.get("metadata_json") or {}
        op_error = metadata.get("apprentice_agent_operational_error") or metadata.get("worker_agent_operational_error")
        if op_error:
            operational_errors.append(str(op_error))
        statuses.append(str(data.get("status") or "failed"))
    if operational_errors:
        return "failed", operational_errors[0]
    if all(status == "success" for status in statuses):
        return "completed", None
    if all(status in {"failed", "timeout", "error"} for status in statuses):
        return "failed", "All recorded attempts failed."
    return "partial", "One or more attempts were partial or failed."


def _sync_run_artifacts(run_root: Path, pkg: Path) -> Path:
    public_artifacts = run_root / "artifacts"
    public_artifacts.mkdir(parents=True, exist_ok=True)
    for existing in public_artifacts.iterdir():
        if existing.is_dir():
            shutil.rmtree(existing)
        else:
            existing.unlink()
    manifest = read_json(pkg / "package_manifest.json") if (pkg / "package_manifest.json").exists() else {}
    selected = str(manifest.get("selected_attempt_id") or "")
    selected_kind = "revised" if selected.endswith("_revised") else "baseline"
    candidates = [pkg / "attempts" / selected_kind / "artifacts"]
    candidates.extend(sorted(pkg.glob("attempts/*/artifacts"), reverse=True))
    source = next((path for path in candidates if path.exists() and any(path.rglob("*"))), None)
    if not source:
        return public_artifacts
    for item in source.rglob("*"):
        if not item.is_file():
            continue
        rel = item.relative_to(source)
        target = public_artifacts / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(item, target)
    return public_artifacts


def _package_progress_summary(run_root: Path, pkg: Path) -> tuple[int, int, Path, str | None]:
    traced_steps = 0
    for trace_path in pkg.glob("attempts/*/agent_trace.json"):
        try:
            traced_steps += len(read_json(trace_path).get("steps") or [])
        except Exception:
            pass
    artifact_count = 0
    for art_dir in pkg.glob("attempts/*/artifacts"):
        if art_dir.exists():
            files = [p for p in art_dir.rglob("*") if p.is_file()]
            artifact_count += len(files)
    artifacts_path = _sync_run_artifacts(run_root, pkg)
    operational_error = None
    for actual_path in sorted((pkg / "attempts").glob("*/actual_outputs.json")):
        try:
            actual = read_json(actual_path)
        except Exception:
            continue
        metadata = actual.get("metadata_json") or {}
        operational_error = metadata.get("apprentice_agent_operational_error") or metadata.get("worker_agent_operational_error")
        if operational_error:
            break
        if actual.get("status") in {"failed", "timeout", "error"}:
            operational_error = actual.get("error_message") or actual.get("error_type")
            break
    return traced_steps, artifact_count, artifacts_path, operational_error


def _attempt_complete_callback(
    run_root: Path,
    settings: Settings,
    *,
    progress_callback: ProgressCallback | None,
    followup_index: int | None = None,
    state: dict[str, bool] | None = None,
):
    def _callback(pkg: Path, attempt_kind: str) -> None:
        if attempt_kind != "baseline":
            return
        traced_steps, artifact_count, artifacts_path, operational_error = _package_progress_summary(run_root, pkg)
        append_progress_event(
            run_root,
            "apprentice_attempt_completed",
            run_id=run_root.name,
            message=("Apprentice attempt failed - operational error" if operational_error else (f"Follow-up {followup_index} Apprentice attempt complete" if followup_index else "Apprentice attempt complete")),
            current_loop=1,
            maximum_improvement_loops=settings.max_improvement_loops,
            phase="apprentice_attempt_complete",
            traced_steps=traced_steps,
            artifact_count=artifact_count,
            artifacts_path=artifacts_path,
            operational_error=operational_error,
            metadata_json={"followup_index": followup_index} if followup_index else None,
            callback=progress_callback,
        )
        if state is not None:
            state["apprentice_completed_emitted"] = True
        if not operational_error and not _human_checkpoint_mode(settings):
            append_progress_event(
                run_root,
                "mentor_review_started",
                run_id=run_root.name,
                message=(f"Follow-up {followup_index} Mentor audit started" if followup_index else "Mentor audit started"),
                current_loop=1,
                maximum_improvement_loops=settings.max_improvement_loops,
                phase="mentor_review",
                traced_steps=traced_steps,
                artifact_count=artifact_count,
                artifacts_path=artifacts_path,
                metadata_json={"followup_index": followup_index} if followup_index else None,
                callback=progress_callback,
            )
            if state is not None:
                state["mentor_review_started_emitted"] = True

    return _callback


def run_prompt_task(
    instruction: str,
    assets: list[Path] | None = None,
    run_id: str | None = None,
    settings: Settings | None = None,
    runner: str | None = None,
    create_bundle: bool = True,
    progress_callback: ProgressCallback | None = None,
    experience_pack_refs: list[dict] | None = None,
    runtime_training_refs: list[dict] | None = None,
) -> tuple[Path, Path | None]:
    settings = settings or get_settings()
    run_id = run_id or make_run_id(instruction)
    run_root = run_root_for(run_id, settings)
    run_root.mkdir(parents=True, exist_ok=True)
    public_artifacts = run_root / "artifacts"
    public_artifacts.mkdir(parents=True, exist_ok=True)
    title = instruction.strip().splitlines()[0][:90] or "Agent Apprenticeship task"
    update_run_status(
        run_root,
        run_id=run_id,
        run_status="running",
        task_status="running",
        current_phase="starting",
        current_loop=1,
        maximum_improvement_loops=settings.max_improvement_loops,
        latest_message="Agent Apprenticeship run started.",
        apprentice_agent=apprentice_agent_display(settings),
        apprenticeship_mode=settings.apprenticeship_mode,
        mentor_mode=settings.mentor_mode,
        task_title=title,
        task_workspace_path=str(run_root),
        artifacts_path=str(public_artifacts),
        **_experience_session_fields(experience_pack_refs),
        **_runtime_training_session_fields(runtime_training_refs),
    )
    append_progress_event(
        run_root,
        "run_started",
        run_id=run_id,
        message="Agent Apprenticeship run started",
        current_loop=1,
        maximum_improvement_loops=settings.max_improvement_loops,
        phase="starting",
        run_status="running",
        task_status="running",
        metadata_json={
            "apprenticeship_mode": settings.apprenticeship_mode,
            "mentor_mode": settings.mentor_mode,
        },
        callback=progress_callback,
    )
    (run_root / "task").mkdir(exist_ok=True)
    append_progress_event(
        run_root,
        "task_workspace_prepared",
        run_id=run_id,
        message="Preparing task workspace",
        current_loop=1,
        maximum_improvement_loops=settings.max_improvement_loops,
        phase="preparing_workspace",
        callback=progress_callback,
    )
    (run_root / "task" / "task_instruction.md").write_text(instruction.rstrip() + "\n")
    task_id = task_id_for_run_id(run_id)

    try:
        asset_refs = copy_assets(
            assets,
            run_root / "task" / "task_instruction_assets",
            "task/task_instruction_assets",
        )
    except Exception as exc:
        append_progress_event(
            run_root,
            "operational_error",
            run_id=run_id,
            message="File-copy failure while preparing task assets",
            current_loop=1,
            maximum_improvement_loops=settings.max_improvement_loops,
            phase="asset_copy_failed",
            run_status="failed",
            task_status="failed",
            operational_error=str(exc),
            callback=progress_callback,
        )
        raise
    append_session_event(
        run_root,
        event_type="task_instruction",
        run_id=run_id,
        task_id=task_id,
        session_id=run_id,
        instruction=instruction,
    )
    if asset_refs:
        append_session_event(
            run_root,
            event_type="task_assets_added",
            run_id=run_id,
            task_id=task_id,
            session_id=run_id,
            assets=asset_refs,
        )

    raw = prompt_to_raw_task(run_id, instruction, _asset_abs_refs(run_root, asset_refs))
    write_json(
        run_root / "session.json",
        {
            "run_id": run_id,
            "session_id": run_id,
            "run_status": "started",
            "task_id": raw.task_id,
            "task_instruction": instruction,
            "task_assets": asset_refs,
            "mentor_mode": settings.mentor_mode,
            "apprenticeship_mode": settings.apprenticeship_mode,
            "sensitive_info_masking": settings.sensitive_info_masking,
            "max_improvement_loops": settings.max_improvement_loops,
            "apprentice_agent": apprentice_agent_display(settings),
            "model_provider": settings.model_provider,
            **_experience_session_fields(experience_pack_refs),
            **_runtime_training_session_fields(runtime_training_refs),
        },
    )
    _append_mentor_preparation_started(run_root, settings, progress_callback=progress_callback)
    revision_decider, checkpoint_state = _revision_decision_callback(
        run_root,
        settings,
        progress_callback=progress_callback,
    )
    try:
        with settings_override(_loop_settings_for_run(settings)):
            pkg = run_task(
                raw,
                run_root,
                runner=runner_for_settings(settings, runner),
                max_iterations=settings.max_improvement_loops,
                pre_attempt_callback=_pre_attempt_checkpoint_callback(
                    run_root,
                    settings,
                    progress_callback=progress_callback,
                ),
                attempt_complete_callback=_attempt_complete_callback(
                    run_root,
                    settings,
                    progress_callback=progress_callback,
                    state=checkpoint_state,
                ),
                revision_decision_callback=revision_decider,
            )
    except KeyboardInterrupt as exc:
        append_progress_event(
            run_root,
            "run_interrupted",
            run_id=run_id,
            message="Run interrupted by user.",
            current_loop=1,
            maximum_improvement_loops=settings.max_improvement_loops,
            phase="interrupted",
            run_status="partial",
            task_status="partial",
            operational_error="Run interrupted by user.",
            callback=progress_callback,
        )
        session = read_json(run_root / "session.json") if (run_root / "session.json").exists() else {}
        session.update({"run_status": "partial", "task_status": "partial", "status_reason": "Run interrupted by user."})
        write_json(run_root / "session.json", {k: v for k, v in session.items() if v is not None})
        raise RunInterrupted(run_root) from exc
    manifest = read_json(pkg / "package_manifest.json") if (pkg / "package_manifest.json").exists() else {}
    run_status, partial_reason = _session_status_for_package(pkg)
    actual_iterations = int(manifest.get("actual_iterations") or 1)
    traced_steps, artifact_count, artifacts_path, operational_error = _package_progress_summary(run_root, pkg)
    if operational_error or not checkpoint_state.get("apprentice_completed_emitted"):
        append_progress_event(
            run_root,
            "apprentice_attempt_completed",
            run_id=run_id,
            message=("Apprentice attempt failed - operational error" if operational_error else "Apprentice attempt complete"),
            current_loop=1,
            maximum_improvement_loops=settings.max_improvement_loops,
            phase="apprentice_attempt_complete",
            traced_steps=traced_steps,
            artifact_count=artifact_count,
            artifacts_path=artifacts_path,
            operational_error=operational_error,
            callback=progress_callback,
        )
    if operational_error:
        append_progress_event(
            run_root,
            "operational_error",
            run_id=run_id,
            message="Apprentice Agent operational error",
            current_loop=1,
            maximum_improvement_loops=settings.max_improvement_loops,
            phase="operational_error",
            run_status=run_status,
            task_status=run_status,
            operational_error=operational_error,
            artifacts_path=artifacts_path,
            callback=progress_callback,
        )
    should_run_mentor_review = not (operational_error and run_status == "failed")
    if should_run_mentor_review and actual_iterations > 1:
        if not checkpoint_state.get("revision_started_emitted"):
            append_progress_event(
                run_root,
                "revision_started",
                run_id=run_id,
                message="Revision attempt started",
                current_loop=2,
                maximum_improvement_loops=settings.max_improvement_loops,
                phase="revision_attempt",
                callback=progress_callback,
            )
        append_progress_event(
            run_root,
            "revision_completed",
            run_id=run_id,
            message="Revision attempt complete",
            current_loop=2,
            maximum_improvement_loops=settings.max_improvement_loops,
            phase="revision_attempt_complete",
            traced_steps=traced_steps,
            artifact_count=artifact_count,
            artifacts_path=artifacts_path,
            callback=progress_callback,
        )
    if should_run_mentor_review and not _human_checkpoint_mode(settings):
        if not checkpoint_state.get("mentor_review_started_emitted"):
            append_progress_event(
                run_root,
                "mentor_review_started",
                run_id=run_id,
                message="Mentor audit started",
                current_loop=actual_iterations,
                maximum_improvement_loops=settings.max_improvement_loops,
                phase="mentor_review",
                callback=progress_callback,
            )
        append_progress_event(
            run_root,
            "mentor_review_completed",
            run_id=run_id,
            message=f"Mentor audit complete - task {run_status}",
            current_loop=actual_iterations,
            maximum_improvement_loops=settings.max_improvement_loops,
            phase="mentor_review_complete",
            task_status=run_status,
            traced_steps=traced_steps,
            artifact_count=artifact_count,
            artifacts_path=artifacts_path,
            callback=progress_callback,
        )
    if should_run_mentor_review and _human_checkpoint_mode(settings):
        write_mentor_checkpoints(
            run_root,
            settings,
            auto_approve=_checkpoint_auto_approve(settings),
            stages=("final_approval",),
        )
        _record_final_approval_loop_feedback(run_root, settings)
    append_session_event(
        run_root,
        event_type="agent_attempt",
        run_id=run_id,
        task_id=pkg.name,
        session_id=run_id,
        attempt_id=manifest.get("selected_attempt_id"),
        metadata_json={"package_path": str(pkg.relative_to(run_root))},
    )
    backfill_session_event_task_ids(run_root, pkg.name)
    session_data = {
        "run_id": run_id,
        "session_id": run_id,
        "run_status": run_status,
        "task_status": run_status,
        "task_id": pkg.name,
        "task_instruction": instruction,
        "task_assets": asset_refs,
        "latest_package": str(pkg.relative_to(run_root)),
        "latest_attempt_id": manifest.get("selected_attempt_id"),
        "mentor_mode": settings.mentor_mode,
        "apprenticeship_mode": settings.apprenticeship_mode,
        "sensitive_info_masking": settings.sensitive_info_masking,
        "max_improvement_loops": settings.max_improvement_loops,
        "apprentice_agent": apprentice_agent_display(settings),
        "model_provider": settings.model_provider,
        **_experience_session_fields(experience_pack_refs),
        **_runtime_training_session_fields(runtime_training_refs),
    }
    if partial_reason:
        session_data["status_reason"] = partial_reason
    write_json(
        run_root / "session.json",
        session_data,
    )
    bundle = None
    if create_bundle:
        append_progress_event(
            run_root,
            "task_outputs_ready",
            run_id=run_id,
            message=f"Task {run_status}. Artifacts are available at: {artifacts_path}",
            current_loop=actual_iterations,
            maximum_improvement_loops=settings.max_improvement_loops,
            phase="task_outputs_ready",
            traced_steps=traced_steps,
            artifact_count=artifact_count,
            artifacts_path=artifacts_path,
            callback=progress_callback,
        )
        _safe_run_tdo_for_run(run_root, settings, progress_callback=progress_callback)
        if _human_checkpoint_mode(settings) and not should_run_mentor_review:
            write_mentor_checkpoints(
                run_root,
                settings,
                auto_approve=_checkpoint_auto_approve(settings),
                stages=("final_approval",),
            )
            _record_final_approval_loop_feedback(run_root, settings)
        append_progress_event(
            run_root,
            "contribution_bundle_started",
            run_id=run_id,
            message="Experience Compilation packaging started",
            current_loop=actual_iterations,
            maximum_improvement_loops=settings.max_improvement_loops,
            phase="contribution_bundle",
            callback=progress_callback,
        )
        bundle = create_contribution_bundle(run_root, settings=settings)
        _update_bundle_experience_metadata(bundle, experience_pack_refs, runtime_training_refs)
        append_progress_event(
            run_root,
            "contribution_bundle_completed",
            run_id=run_id,
            message="Experience Compilation ready",
            current_loop=actual_iterations,
            maximum_improvement_loops=settings.max_improvement_loops,
            phase="contribution_bundle_complete",
            contribution_bundle_path=bundle,
            callback=progress_callback,
        )
    append_progress_event(
        run_root,
        "run_completed",
        run_id=run_id,
        message=f"Task {run_status}.",
        current_loop=actual_iterations,
        maximum_improvement_loops=settings.max_improvement_loops,
        phase="completed" if run_status == "completed" else run_status,
        run_status=run_status,
        task_status=run_status,
        traced_steps=traced_steps,
        artifact_count=artifact_count,
        artifacts_path=artifacts_path,
        contribution_bundle_path=bundle,
        operational_error=operational_error,
        callback=progress_callback,
    )
    update_run_status(run_root, **_runtime_training_session_fields(runtime_training_refs))
    return run_root, bundle


def _experience_session_fields(experience_pack_refs: list[dict] | None) -> dict:
    refs = experience_pack_refs or []
    if not refs:
        return {}
    return {
        "experience_pack_ids": [ref.get("pack_id") for ref in refs if ref.get("pack_id")],
        "experience_pack_titles": [ref.get("title") for ref in refs if ref.get("title")],
        "experience_pack_sources": [
            source
            for ref in refs
            for source in (ref.get("source_refs") or [])
        ],
    }


def _runtime_training_session_fields(runtime_training_refs: list[dict] | None) -> dict:
    refs = runtime_training_refs or []
    if not refs:
        return {
            "runtime_training_used": False,
            "installed_skill_refs": [],
            "installed_skill_ids": [],
        }
    injected_sections: list[str] = []
    confidence_values: list[str] = []
    selection_reasons: list[str] = []
    total_chars = 0
    content_hashes: list[str] = []
    for ref in refs:
        for section in ref.get("runtime_training_injected_sections") or []:
            if section not in injected_sections:
                injected_sections.append(section)
        source_gate = ref.get("source_quality_gate") if isinstance(ref.get("source_quality_gate"), dict) else {}
        confidence = ref.get("runtime_training_confidence") or ref.get("transfer_confidence") or source_gate.get("transfer_confidence")
        if confidence:
            confidence_values.append(str(confidence))
        reason = ref.get("runtime_training_selection_reason") or ref.get("selection_reason")
        if reason:
            selection_reasons.append(str(reason))
        total_chars += int(ref.get("runtime_training_context_char_count") or 0)
        if ref.get("runtime_training_content_hash"):
            content_hashes.append(str(ref.get("runtime_training_content_hash")))
    return {
        "runtime_training_used": True,
        "installed_skill_refs": refs,
        "installed_skill_ids": [ref.get("installed_skill_id") for ref in refs if ref.get("installed_skill_id")],
        "runtime_training_selection_reason": "; ".join(selection_reasons) or "enabled installed runtime training matched this task or was explicitly requested",
        "runtime_training_confidence": "low" if "low" in confidence_values else "high" if confidence_values else "unknown",
        "runtime_training_injected_sections": injected_sections,
        "runtime_training_injected_char_count": total_chars,
        "runtime_training_content_hashes": content_hashes,
        "output_contract_used": "output_contract" in injected_sections,
        "artifact_contract_used": "artifact_contract" in injected_sections,
        "prefinal_verification_status": "not_run",
        "repair_loop_count": max((int(ref.get("repair_loop_count") or 0) for ref in refs), default=0),
        "contract_failures_json": [failure for ref in refs for failure in (ref.get("contract_failures_json") or [])],
    }


def _update_json_experience_metadata(path: Path, experience_pack_refs: list[dict] | None, runtime_training_refs: list[dict] | None = None) -> None:
    if not (experience_pack_refs or runtime_training_refs) or not path.exists():
        return
    data = read_json(path)
    if not isinstance(data, dict):
        return
    if experience_pack_refs:
        data.update(_experience_session_fields(experience_pack_refs))
        data["experience_learning_status"] = "experience_pack_applied"
    if runtime_training_refs:
        data.update(_runtime_training_session_fields(runtime_training_refs))
    write_json(path, data)


def _update_trace_experience_metadata(path: Path, experience_pack_refs: list[dict] | None, runtime_training_refs: list[dict] | None = None) -> None:
    if not (experience_pack_refs or runtime_training_refs) or not path.exists():
        return
    data = read_json(path)
    if not isinstance(data, dict):
        return
    metadata = data.setdefault("metadata_json", {})
    if not isinstance(metadata, dict):
        metadata = {}
        data["metadata_json"] = metadata
    if experience_pack_refs:
        metadata.update(_experience_session_fields(experience_pack_refs))
        metadata["experience_learning_status"] = "experience_pack_applied"
    if runtime_training_refs:
        metadata.update(_runtime_training_session_fields(runtime_training_refs))
    write_json(path, data)


def _update_bundle_experience_metadata(bundle: Path | None, experience_pack_refs: list[dict] | None, runtime_training_refs: list[dict] | None = None) -> None:
    if not bundle or not (experience_pack_refs or runtime_training_refs):
        return
    for rel in ("contribution_manifest.json", "session_metadata.json"):
        _update_json_experience_metadata(bundle / rel, experience_pack_refs, runtime_training_refs)
    for trace_path in bundle.glob("attempts/*/agent_trace.json"):
        _update_trace_experience_metadata(trace_path, experience_pack_refs, runtime_training_refs)


def _latest_attempt_id(run_root: Path) -> str | None:
    packages = sorted((run_root / "packages").glob("*")) if (run_root / "packages").exists() else []
    for pkg in reversed(packages):
        manifest = pkg / "package_manifest.json"
        if manifest.exists():
            data = read_json(manifest)
            if data.get("selected_attempt_id"):
                return str(data["selected_attempt_id"])
    return None


def _latest_package(run_root: Path) -> Path | None:
    packages = sorted((run_root / "packages").glob("*")) if (run_root / "packages").exists() else []
    return packages[-1] if packages else None


def _record_final_approval_loop_feedback(run_root: Path, settings: Settings) -> None:
    if not _human_checkpoint_mode(settings):
        return
    pkg = _latest_package(run_root)
    if not pkg:
        return
    final_path = run_root / "mentor_checkpoints" / "final_approval_checkpoint.json"
    if not final_path.exists():
        return
    try:
        final = read_json(final_path)
    except Exception:
        return
    manifest_path = pkg / "loops" / "loop_manifest.json"
    if not manifest_path.exists():
        return
    manifest = read_json(manifest_path)
    iterations = [item for item in manifest.get("iterations") or [] if isinstance(item, dict)]
    if not iterations:
        return
    iteration = max(int(item.get("loop_iteration") or 0) for item in iterations)
    if iteration <= 0:
        return
    review_refs = [
        f"loops/iterations/{iteration:03d}/review_packet.json",
        f"loops/iterations/{iteration:03d}/review_packet.md",
    ]
    final_with_refs = {**final, "evidence_refs": review_refs}
    approved = bool(final.get("approved_for_local_bundle", True))
    record_human_loop_feedback(pkg, iteration=iteration, checkpoint=final_with_refs, revision_requested=not approved)


def _session_task_id(run_root: Path) -> str | None:
    session_path = run_root / "session.json"
    if session_path.exists():
        try:
            data = read_json(session_path)
            if data.get("task_id"):
                return str(data["task_id"])
        except Exception:
            pass
    packages = sorted((run_root / "packages").glob("*")) if (run_root / "packages").exists() else []
    return packages[-1].name if packages else None


def continue_session(
    run_id: str,
    followup_instruction: str,
    assets: list[Path] | None = None,
    run_loop: bool = False,
    settings: Settings | None = None,
    runner: str | None = None,
    progress_callback: ProgressCallback | None = None,
) -> tuple[Path, Path | None]:
    settings = settings or get_settings()
    run_root = run_root_for(run_id, settings)
    if not run_root.exists():
        raise FileNotFoundError(f"Run not found: {run_id}")
    session = read_json(run_root / "session.json") if (run_root / "session.json").exists() else {}
    settings = _settings_for_session(settings, session)
    followup_index = next_followup_index(run_root)
    task_id = _session_task_id(run_root)
    append_progress_event(
        run_root,
        "followup_started",
        run_id=run_root.name,
        message=f"Follow-up {followup_index} received",
        current_loop=1,
        maximum_improvement_loops=settings.max_improvement_loops,
        phase="followup",
        run_status="running",
        task_status="running",
        metadata_json={"followup_index": followup_index},
        callback=progress_callback,
    )
    try:
        asset_refs = copy_assets(
            assets,
            run_root / "task" / "task_instruction_assets" / f"followup_{followup_index}",
            f"task/task_instruction_assets/followup_{followup_index}",
        )
    except Exception as exc:
        append_progress_event(
            run_root,
            "operational_error",
            run_id=run_root.name,
            message="File-copy failure while preparing follow-up assets",
            current_loop=1,
            maximum_improvement_loops=settings.max_improvement_loops,
            phase="asset_copy_failed",
            run_status="failed",
            task_status="failed",
            operational_error=str(exc),
            metadata_json={"followup_index": followup_index},
            callback=progress_callback,
        )
        raise
    append_session_event(
        run_root,
        event_type="user_followup",
        run_id=run_root.name,
        task_id=task_id,
        session_id=run_root.name,
        feedback_source="user",
        feedback_type="followup_instruction",
        applies_to_attempt=_latest_attempt_id(run_root),
        followup_index=followup_index,
        followup_instruction=followup_instruction,
        followup_assets=asset_refs,
    )
    if asset_refs:
        append_session_event(
            run_root,
            event_type="followup_assets_added",
            run_id=run_root.name,
            task_id=task_id,
            session_id=run_root.name,
            feedback_source="user",
            feedback_type="followup_instruction",
            applies_to_attempt=_latest_attempt_id(run_root),
            followup_index=followup_index,
            followup_assets=asset_refs,
        )
    bundle: Path | None = None
    if run_loop:
        original = session.get("task_instruction") or ""
        combined = (
            "Continue the same Agent Apprenticeship session.\n\n"
            f"Original task instruction:\n{original}\n\n"
            f"Follow-up instruction {followup_index}:\n{followup_instruction}\n"
        )
        raw = prompt_to_raw_task(
            f"{run_root.name}-followup-{followup_index}",
            combined,
            _asset_abs_refs(run_root, asset_refs),
        )
        _append_mentor_preparation_started(
            run_root,
            settings,
            progress_callback=progress_callback,
            followup_index=followup_index,
        )
        revision_decider, checkpoint_state = _revision_decision_callback(
            run_root,
            settings,
            progress_callback=progress_callback,
            followup_index=followup_index,
        )
        with settings_override(_loop_settings_for_run(settings)):
            pkg = run_task(
                raw,
                run_root,
                runner=runner_for_settings(settings, runner),
                max_iterations=settings.max_improvement_loops,
                pre_attempt_callback=_pre_attempt_checkpoint_callback(
                    run_root,
                    settings,
                    progress_callback=progress_callback,
                    followup_index=followup_index,
                ),
                attempt_complete_callback=_attempt_complete_callback(
                    run_root,
                    settings,
                    progress_callback=progress_callback,
                    followup_index=followup_index,
                    state=checkpoint_state,
                ),
                revision_decision_callback=revision_decider,
            )
        manifest = read_json(pkg / "package_manifest.json") if (pkg / "package_manifest.json").exists() else {}
        status, _reason = _session_status_for_package(pkg)
        actual_iterations = int(manifest.get("actual_iterations") or 1)
        traced_steps, artifact_count, artifacts_path, operational_error = _package_progress_summary(run_root, pkg)
        if operational_error or not checkpoint_state.get("apprentice_completed_emitted"):
            append_progress_event(
                run_root,
                "apprentice_attempt_completed",
                run_id=run_root.name,
                message=(f"Follow-up {followup_index} Apprentice attempt failed - operational error" if operational_error else f"Follow-up {followup_index} Apprentice attempt complete"),
                current_loop=1,
                maximum_improvement_loops=settings.max_improvement_loops,
                phase="apprentice_attempt_complete",
                traced_steps=traced_steps,
                artifact_count=artifact_count,
                artifacts_path=artifacts_path,
                operational_error=operational_error,
                metadata_json={"followup_index": followup_index},
                callback=progress_callback,
            )
        if operational_error:
            append_progress_event(
                run_root,
                "operational_error",
                run_id=run_root.name,
                message=f"Follow-up {followup_index} Apprentice Agent operational error",
                current_loop=1,
                maximum_improvement_loops=settings.max_improvement_loops,
                phase="operational_error",
                run_status=status,
                task_status=status,
                operational_error=operational_error,
                artifacts_path=artifacts_path,
                metadata_json={"followup_index": followup_index},
                callback=progress_callback,
            )
        should_run_mentor_review = not (operational_error and status == "failed")
        if should_run_mentor_review and actual_iterations > 1:
            if not checkpoint_state.get("revision_started_emitted"):
                append_progress_event(
                    run_root,
                    "revision_started",
                    run_id=run_root.name,
                    message=f"Follow-up {followup_index} revision attempt started",
                    current_loop=2,
                    maximum_improvement_loops=settings.max_improvement_loops,
                    phase="revision_attempt",
                    metadata_json={"followup_index": followup_index},
                    callback=progress_callback,
                )
            append_progress_event(
                run_root,
                "revision_completed",
                run_id=run_root.name,
                message=f"Follow-up {followup_index} revision attempt complete",
                current_loop=2,
                maximum_improvement_loops=settings.max_improvement_loops,
                phase="revision_attempt_complete",
                traced_steps=traced_steps,
                artifact_count=artifact_count,
                artifacts_path=artifacts_path,
                metadata_json={"followup_index": followup_index},
                callback=progress_callback,
            )
        if should_run_mentor_review and not _human_checkpoint_mode(settings):
            append_progress_event(
                run_root,
                "mentor_review_completed",
                run_id=run_root.name,
                message=f"Follow-up {followup_index} mentor audit complete",
                current_loop=actual_iterations,
                maximum_improvement_loops=settings.max_improvement_loops,
                phase="mentor_review_complete",
                task_status=status,
                traced_steps=traced_steps,
                artifact_count=artifact_count,
                artifacts_path=artifacts_path,
                metadata_json={"followup_index": followup_index},
                callback=progress_callback,
            )
        if should_run_mentor_review and _human_checkpoint_mode(settings):
            write_mentor_checkpoints(
                run_root,
                settings,
                auto_approve=_checkpoint_auto_approve(settings),
                stages=("final_approval",),
            )
            _record_final_approval_loop_feedback(run_root, settings)
        append_session_event(
            run_root,
            event_type="agent_attempt",
            run_id=run_root.name,
            task_id=task_id or pkg.name,
            session_id=run_root.name,
            attempt_id=manifest.get("selected_attempt_id"),
            metadata_json={"package_path": str(pkg.relative_to(run_root)), "followup_index": followup_index},
        )
        backfill_session_event_task_ids(run_root, task_id or pkg.name)
        final_status = status
        final_artifacts_path = artifacts_path
        final_traced_steps = traced_steps
        final_artifact_count = artifact_count
        final_operational_error = operational_error
        session = read_json(run_root / "session.json") if (run_root / "session.json").exists() else {}
        session.update(
            {
                "run_status": status,
                "task_status": status,
                "task_id": task_id or pkg.name,
                "latest_package": str(pkg.relative_to(run_root)),
                "latest_attempt_id": manifest.get("selected_attempt_id"),
                "status_reason": _reason if status != "completed" and _reason else None,
            }
        )
        session = {k: v for k, v in session.items() if v is not None}
        write_json(run_root / "session.json", session)
    else:
        backfill_session_event_task_ids(run_root, task_id)
        session = read_json(run_root / "session.json") if (run_root / "session.json").exists() else {}
        final_status = session.get("task_status") or session.get("run_status") or "completed"
        final_artifacts_path = None
        status_file = run_root / "run_status.json"
        if status_file.exists():
            status_data = read_json(status_file)
            final_artifacts_path = status_data.get("artifacts_path")
        final_traced_steps = None
        final_artifact_count = None
        final_operational_error = None
        record_only_message = "Follow-up recorded. No Apprentice Agent loop was run. Use --run-loop to continue work."
        record_followup_review_update(
            run_root,
            followup_index=followup_index,
            record_only=True,
            note=record_only_message,
        )
    if run_loop:
        record_followup_review_update(
            run_root,
            followup_index=followup_index,
            record_only=False,
            note=f"Follow-up {followup_index} ran a new Apprentice Agent loop and produced a new review packet.",
        )
    _safe_run_tdo_for_run(
        run_root,
        settings,
        progress_callback=progress_callback,
        followup_index=followup_index,
        record_only=not run_loop,
    )
    if run_loop and _human_checkpoint_mode(settings):
        write_mentor_checkpoints(
            run_root,
            settings,
            auto_approve=_checkpoint_auto_approve(settings),
            stages=("final_approval",),
            preserve_interactive=False,
        )
        _record_final_approval_loop_feedback(run_root, settings)
    append_progress_event(
        run_root,
        "contribution_bundle_started",
        run_id=run_root.name,
        message="Experience Compilation update started",
        current_loop=1,
        maximum_improvement_loops=settings.max_improvement_loops,
        phase="contribution_bundle",
        metadata_json={"followup_index": followup_index},
        callback=progress_callback,
    )
    bundle = create_contribution_bundle(run_root, settings=settings)
    append_progress_event(
        run_root,
        "contribution_bundle_completed",
        run_id=run_root.name,
        message="Experience Compilation updated",
        current_loop=1,
        maximum_improvement_loops=settings.max_improvement_loops,
        phase="contribution_bundle_complete",
        contribution_bundle_path=bundle,
        metadata_json={"followup_index": followup_index},
        callback=progress_callback,
    )
    append_progress_event(
        run_root,
        "followup_completed",
        run_id=run_root.name,
        message=(record_only_message if not run_loop else f"Follow-up {followup_index} complete"),
        current_loop=1,
        maximum_improvement_loops=settings.max_improvement_loops,
        phase="followup_complete",
        run_status=final_status,
        task_status=final_status,
        traced_steps=final_traced_steps,
        artifact_count=final_artifact_count,
        artifacts_path=final_artifacts_path,
        contribution_bundle_path=bundle,
        operational_error=final_operational_error,
        metadata_json={"followup_index": followup_index, "record_only": not run_loop},
        callback=progress_callback,
    )
    return run_root, bundle


def finish_session(run_id: str, settings: Settings | None = None, progress_callback: ProgressCallback | None = None) -> tuple[Path, Path]:
    settings = settings or get_settings()
    run_root = run_root_for(run_id, settings)
    if not run_root.exists():
        raise FileNotFoundError(f"Run not found: {run_id}")
    task_id = _session_task_id(run_root)
    append_session_event(
        run_root,
        event_type="session_finished",
        run_id=run_root.name,
        task_id=task_id,
        session_id=run_root.name,
        applies_to_attempt=_latest_attempt_id(run_root),
    )
    if (run_root / "session.json").exists():
        data = read_json(run_root / "session.json")
    else:
        data = {"run_id": run_root.name, "session_id": run_root.name}
    status_data = read_json(run_root / "run_status.json") if (run_root / "run_status.json").exists() else {}
    existing_task_status = data.get("task_status") or status_data.get("task_status") or "partial"
    existing_run_status = data.get("run_status") or status_data.get("run_status") or existing_task_status
    data["run_status"] = existing_run_status
    data["task_status"] = existing_task_status
    data["session_status"] = "finished"
    if task_id:
        data["task_id"] = task_id
    write_json(run_root / "session.json", data)
    backfill_session_event_task_ids(run_root, task_id)
    write_mentor_checkpoints(
        run_root,
        settings,
        auto_approve=_checkpoint_auto_approve(settings),
        stages=("final_approval",),
    )
    _record_final_approval_loop_feedback(run_root, settings)
    _safe_run_tdo_for_run(run_root, settings, progress_callback=progress_callback)
    bundle = create_contribution_bundle(run_root, settings=settings)
    append_progress_event(
        run_root,
        "run_completed",
        run_id=run_root.name,
        message="Session finished.",
        phase="session_finished",
        run_status=existing_run_status,
        task_status=existing_task_status,
        artifacts_path=status_data.get("artifacts_path") or str(run_root / "artifacts"),
        contribution_bundle_path=bundle,
    )
    return run_root, bundle
