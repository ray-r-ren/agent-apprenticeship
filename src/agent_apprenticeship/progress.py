from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Literal

from .io import append_jsonl, read_workflow_json as read_json, read_workflow_jsonl as read_jsonl, write_json

RUN_TERMINAL_STATUSES = {"completed", "partial", "failed"}

RunStatus = Literal["running", "completed", "partial", "failed"]
TaskStatus = Literal["running", "completed", "partial", "failed"]
ProgressCallback = Callable[[dict[str, Any], dict[str, Any]], None]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def run_status_path(run_root: Path) -> Path:
    return run_root / "run_status.json"


def progress_events_path(run_root: Path) -> Path:
    return run_root / "progress_events.jsonl"


def read_run_status(run_root: Path) -> dict[str, Any]:
    path = run_status_path(run_root)
    return read_json(path) if path.exists() else {}


def read_progress_events(run_root: Path) -> list[dict[str, Any]]:
    path = progress_events_path(run_root)
    return read_jsonl(path) if path.exists() else []


def _event_index(run_root: Path) -> int:
    return len(read_progress_events(run_root)) + 1


def _base_status(run_id: str, run_root: Path, **kwargs: Any) -> dict[str, Any]:
    now = utc_now()
    return {
        "run_id": run_id,
        "run_status": "running",
        "task_status": "running",
        "current_phase": kwargs.get("current_phase") or "starting",
        "current_loop": kwargs.get("current_loop") or 1,
        "maximum_improvement_loops": kwargs.get("maximum_improvement_loops"),
        "latest_message": kwargs.get("latest_message") or "Run starting.",
        "apprentice_agent": kwargs.get("apprentice_agent") or kwargs.get("worker_agent"),
        "apprenticeship_mode": kwargs.get("apprenticeship_mode"),
        "mentor_mode": kwargs.get("mentor_mode"),
        "task_title": kwargs.get("task_title"),
        "task_workspace_path": str(kwargs.get("task_workspace_path") or run_root),
        "artifacts_path": str(kwargs.get("artifacts_path") or run_root / "artifacts"),
        "contribution_bundle_path": kwargs.get("contribution_bundle_path"),
        "traced_steps": kwargs.get("traced_steps"),
        "artifact_count": kwargs.get("artifact_count"),
        "started_at": kwargs.get("started_at") or now,
        "updated_at": now,
        "last_operational_error": kwargs.get("last_operational_error"),
    }


def update_run_status(run_root: Path, **updates: Any) -> dict[str, Any]:
    run_root.mkdir(parents=True, exist_ok=True)
    current = read_run_status(run_root)
    updates = dict(updates)
    clear_operational_error = bool(updates.pop("clear_operational_error", False))
    if "worker_agent" in updates and "apprentice_agent" not in updates:
        updates["apprentice_agent"] = updates["worker_agent"]
    updates.pop("worker_agent", None)
    run_id = str(updates.get("run_id") or current.get("run_id") or run_root.name)
    if current:
        data = dict(current)
        data.update({k: v for k, v in updates.items() if v is not None})
        if clear_operational_error:
            data.pop("last_operational_error", None)
        if data.get("apprentice_agent"):
            data.pop("worker_agent", None)
        data.setdefault("run_id", run_id)
        data.setdefault("started_at", current.get("started_at") or utc_now())
        data["updated_at"] = utc_now()
    else:
        base_updates = {k: v for k, v in updates.items() if k != "run_id"}
        data = _base_status(run_id, run_root, **base_updates)
    write_json(run_status_path(run_root), data)
    return data


def append_progress_event(
    run_root: Path,
    event_type: str,
    *,
    message: str,
    run_id: str | None = None,
    current_loop: int | None = None,
    maximum_improvement_loops: int | None = None,
    phase: str | None = None,
    run_status: RunStatus | None = None,
    task_status: TaskStatus | None = None,
    traced_steps: int | None = None,
    artifact_count: int | None = None,
    artifacts_path: str | Path | None = None,
    contribution_bundle_path: str | Path | None = None,
    operational_error: str | None = None,
    metadata_json: dict[str, Any] | None = None,
    callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    run_root.mkdir(parents=True, exist_ok=True)
    event = {
        "event_type": event_type,
        "event_index": _event_index(run_root),
        "created_at": utc_now(),
        "run_id": run_id or run_root.name,
        "current_loop": current_loop,
        "maximum_improvement_loops": maximum_improvement_loops,
        "phase": phase,
        "message": message,
        "run_status": run_status,
        "task_status": task_status,
        "traced_steps": traced_steps,
        "artifact_count": artifact_count,
        "artifacts_path": str(artifacts_path) if artifacts_path else None,
        "contribution_bundle_path": str(contribution_bundle_path) if contribution_bundle_path else None,
        "operational_error": operational_error,
        "metadata_json": metadata_json or {},
    }
    event = {k: v for k, v in event.items() if v is not None}
    append_jsonl(progress_events_path(run_root), event)
    status_updates: dict[str, Any] = {
        "run_id": event["run_id"],
        "current_phase": phase or event_type,
        "latest_message": message,
    }
    for key, value in {
        "current_loop": current_loop,
        "maximum_improvement_loops": maximum_improvement_loops,
        "run_status": run_status,
        "task_status": task_status,
        "traced_steps": traced_steps,
        "artifact_count": artifact_count,
        "artifacts_path": str(artifacts_path) if artifacts_path else None,
        "contribution_bundle_path": str(contribution_bundle_path) if contribution_bundle_path else None,
        "last_operational_error": operational_error,
    }.items():
        if value is not None:
            status_updates[key] = value
    if operational_error is None and (run_status == "completed" or task_status == "completed"):
        status_updates["clear_operational_error"] = True
    status = update_run_status(run_root, **status_updates)
    if callback:
        callback(event, status)
    return event


def format_progress_event(event: dict[str, Any]) -> str:
    loop = event.get("current_loop")
    max_loop = event.get("maximum_improvement_loops")
    metadata = event.get("metadata_json") or {}
    followup_index = metadata.get("followup_index")
    followup_prefix = f"[Follow-up {followup_index}]" if followup_index else ""
    loop_prefix = f"[{loop}/{max_loop}] " if loop and max_loop else ""
    prefix = f"{followup_prefix}{loop_prefix}"
    message = event.get("message") or event.get("event_type") or "Progress update"
    details: list[str] = []
    if event.get("traced_steps") is not None:
        details.append(f"{event['traced_steps']} traced steps")
    if event.get("artifact_count") is not None:
        details.append(f"{event['artifact_count']} artifacts")
    if event.get("operational_error"):
        details.append(f"error: {event['operational_error']}")
    return prefix + message + (f" - {', '.join(details)}" if details else "")


def format_run_status(status: dict[str, Any]) -> str:
    if not status:
        return "No run_status.json found."
    lines = [
        f"Run: {status.get('run_id')}",
        f"Apprentice Agent: {status.get('apprentice_agent') or status.get('worker_agent')}",
        f"Run Status: {status.get('run_status')}",
        f"Task Status: {status.get('task_status')}",
        f"Current Phase: {status.get('current_phase')}",
        f"Loop: {status.get('current_loop')}/{status.get('maximum_improvement_loops')}",
        f"Latest: {status.get('latest_message')}",
        f"Task Workspace: {status.get('task_workspace_path')}",
        f"Artifacts: {status.get('artifacts_path')}",
    ]
    if status.get("contribution_bundle_path"):
        lines.append(f"Experience Compilation: {status.get('contribution_bundle_path')}")
    if status.get("traced_steps") is not None:
        lines.append(f"Traced Steps: {status.get('traced_steps')}")
    if status.get("artifact_count") is not None:
        lines.append(f"Artifacts Indexed: {status.get('artifact_count')}")
    if status.get("last_operational_error"):
        lines.append(f"Operational Error: {status.get('last_operational_error')}")
    return "\n".join(lines)


def watch_progress(
    run_root: Path,
    *,
    interval_seconds: float = 1.0,
    timeout_seconds: float | None = None,
    emit: Callable[[str], None] = print,
) -> None:
    seen = 0
    started = time.monotonic()
    while True:
        events = read_progress_events(run_root)
        for event in events[seen:]:
            emit(format_progress_event(event))
        seen = len(events)
        status = read_run_status(run_root)
        if status.get("run_status") in RUN_TERMINAL_STATUSES:
            return
        if timeout_seconds is not None and time.monotonic() - started >= timeout_seconds:
            return
        time.sleep(interval_seconds)
