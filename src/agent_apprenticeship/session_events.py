from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from .io import append_jsonl, read_workflow_jsonl as read_jsonl


SessionEventType = Literal[
    "task_instruction",
    "task_assets_added",
    "agent_attempt",
    "evaluation",
    "revision",
    "user_followup",
    "followup_assets_added",
    "session_finished",
]


class SessionEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event_type: SessionEventType
    event_index: int
    created_at: str
    run_id: str
    task_id: str | None = None
    session_id: str | None = None
    attempt_id: str | None = None
    feedback_source: str | None = None
    feedback_type: str | None = None
    applies_to_attempt: str | None = None
    followup_index: int | None = None
    instruction: str | None = None
    followup_instruction: str | None = None
    assets: list[str] = Field(default_factory=list)
    followup_assets: list[str] = Field(default_factory=list)
    metadata_json: dict[str, Any] = Field(default_factory=dict)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def events_path(run_root: Path) -> Path:
    return run_root / "session_events.jsonl"


def read_session_events(run_root: Path) -> list[dict[str, Any]]:
    return read_jsonl(events_path(run_root))


def next_event_index(run_root: Path) -> int:
    return len(read_session_events(run_root)) + 1


def append_session_event(run_root: Path, **kwargs: Any) -> SessionEvent:
    run_root.mkdir(parents=True, exist_ok=True)
    event = SessionEvent(
        event_index=next_event_index(run_root),
        created_at=utc_now(),
        **kwargs,
    )
    append_jsonl(events_path(run_root), event)
    return event


def backfill_session_event_task_ids(run_root: Path, task_id: str | None) -> None:
    if not task_id:
        return
    path = events_path(run_root)
    rows = read_jsonl(path)
    if not rows:
        return
    changed = False
    for row in rows:
        if isinstance(row, dict) and not row.get("task_id"):
            row["task_id"] = task_id
            changed = True
    if not changed:
        return
    path.write_text("")
    for row in rows:
        append_jsonl(path, row)


def next_followup_index(run_root: Path) -> int:
    events = read_session_events(run_root)
    values = [
        int(e.get("followup_index") or 0)
        for e in events
        if e.get("event_type") == "user_followup"
    ]
    return (max(values) if values else 0) + 1
