from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .config import Settings
from .io import read_workflow_json as read_json, write_json


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _latest_package(run_root: Path) -> Path | None:
    packages = run_root / "packages"
    if not packages.exists():
        return None
    candidates = [p for p in packages.iterdir() if p.is_dir()]
    return max(candidates, key=lambda p: p.stat().st_mtime) if candidates else None


def _attempt_summary(pkg: Path | None) -> dict[str, Any]:
    if not pkg:
        return {"attempts": 0, "traced_steps": 0, "artifacts": []}
    attempts = sorted((pkg / "attempts").glob("*"))
    traced_steps = 0
    artifacts: list[str] = []
    for attempt in attempts:
        trace = attempt / "agent_trace.json"
        if trace.exists():
            try:
                traced_steps += len(read_json(trace).get("steps") or [])
            except Exception:
                pass
        art_dir = attempt / "artifacts"
        if art_dir.exists():
            for path in art_dir.rglob("*"):
                if path.is_file():
                    artifacts.append(str(path.relative_to(pkg)))
    return {"attempts": len(attempts), "traced_steps": traced_steps, "artifacts": artifacts[:50]}


def _mark_source_artifact_audited(pkg: Path | None, *, source: str, mode: str, input_source: str, stages: set[str]) -> None:
    if not pkg:
        return
    updates = {
        "mentor_audited": True,
        "mentor_audit_source": source,
        "audit_status": f"{mode}_audit_complete",
        "checkpoint_input_source": input_source,
    }
    targets: list[Path] = []
    if "task_intake" in stages:
        targets.append(pkg / "task" / "task_intake_spec.json")
    if "rubric" in stages:
        targets.append(pkg / "rubric" / "rubric.json")
    for path in targets:
        if not path.exists():
            continue
        try:
            data = read_json(path)
            md = dict(data.get("metadata_json") or {})
            md.update(updates)
            data["metadata_json"] = md
            write_json(path, data)
        except Exception:
            continue


def _truthy_env(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _load_checkpoint_inputs() -> dict[str, Any]:
    raw = os.getenv("AA_MENTOR_CHECKPOINT_INPUTS_JSON")
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _prompt(label: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    try:
        value = input(f"{label}{suffix}: ").strip()
    except EOFError:
        return default
    return value if value else default


def _split_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def _parse_score(value: str, default: float) -> float:
    try:
        score = float(value)
    except ValueError:
        return default
    if score > 1:
        score = score / 100
    return max(0.0, min(1.0, score))


APPROVE_WORDS = {"accept", "approve", "approved", "confirm", "confirmed", "yes", "y", "ok", "okay"}


def _is_approve(value: str) -> bool:
    return value.strip().lower() in APPROVE_WORDS


def _is_edit(value: str) -> bool:
    return value.strip().lower().startswith("edit")


def _stage_set(stages: Iterable[str] | None) -> set[str]:
    return set(stages or {"task_intake", "rubric", "evaluation", "revision", "final_approval"})


def _collect_interactive_inputs(
    mode: str,
    session: dict[str, Any],
    status: dict[str, Any],
    summary: dict[str, Any],
    stages: set[str],
) -> dict[str, Any]:
    label = "Expert-led"
    default_passed = "pass" if status.get("task_status") == "completed" else "fail"
    default_score = "1.0" if default_passed == "pass" else "0.0"
    collected: dict[str, Any] = {}
    if "task_intake" in stages:
        print(f"{label} checkpoint: task intake")
        intake_decision = _prompt("Confirm or edit task intake", "confirm")
        if _is_edit(intake_decision):
            edited_title = _prompt("Edited title, optional", "")
            intake_notes = _prompt("Task intake notes, optional", "")
            decision = "edited"
        else:
            edited_title = ""
            intake_notes = "Confirmed as-is." if _is_approve(intake_decision) else f"Confirmed as-is. Input: {intake_decision}"
            decision = "confirmed"
        collected["task_intake"] = {
            "decision": decision,
            "edited_title": edited_title or None,
            "notes": intake_notes or "Confirmed as-is.",
        }

    if "rubric" in stages:
        print(f"{label} checkpoint: rubric")
        rubric_decision = _prompt("Confirm or edit rubric", "confirm")
        if _is_edit(rubric_decision):
            additional_criterion = _prompt("Add rubric criterion, optional", "")
            rubric_notes = _prompt("Rubric notes, optional", "")
            rubric_choice = "edited"
        else:
            additional_criterion = ""
            rubric_notes = "Rubric confirmed." if _is_approve(rubric_decision) else f"Rubric confirmed. Input: {rubric_decision}"
            rubric_choice = "confirmed"
        collected["rubric"] = {
            "decision": rubric_choice,
            "additional_criteria": [additional_criterion] if additional_criterion else [],
            "notes": rubric_notes or "Rubric confirmed.",
        }

    if "evaluation" in stages:
        print(f"{label} checkpoint: evaluation/verifier")
        passed_raw = _prompt("Pass or fail", default_passed).strip().lower()
        passed = passed_raw.startswith("pass") or passed_raw in {"yes", "y", "true", "approve", "approved", "ok"}
        score = _parse_score(_prompt("Score 0-1 or 0-100", default_score), 1.0 if default_passed == "pass" else 0.0)
        feedback = _prompt("Expert feedback", status.get("latest_message") or "Reviewed artifacts and trace summary.")
        failed_criteria = [] if passed else _split_csv(_prompt("Failed criteria, comma-separated optional", ""))
        collected["evaluation"] = {
            "passed": passed,
            "score": score,
            "failed_criteria": failed_criteria,
            "feedback": feedback,
            "attempt_summary": summary,
        }

    if "revision" in stages:
        print(f"{label} checkpoint: revision")
        revision_decision = _prompt("Revise or finish", "finish").strip().lower()
        revision_notes = _prompt("Revision notes, optional", "")
        collected["revision"] = {
            "revision_should_run": revision_decision.startswith("revise"),
            "decision": "revise" if revision_decision.startswith("revise") else "finish",
            "notes": revision_notes or ("Revision requested by expert." if revision_decision.startswith("revise") else "Expert chose to finish."),
        }

    if "final_approval" in stages:
        print(f"{label} checkpoint: final bundle approval")
        final_decision = _prompt("Confirm final Experience Compilation or hold", "confirm").strip().lower()
        final_notes = _prompt("Final approval notes, optional", "")
        collected["final_approval"] = {
            "approved_for_local_bundle": _is_approve(final_decision) or final_decision.startswith("approve"),
            "decision": "confirmed" if (_is_approve(final_decision) or final_decision.startswith("approve") or final_decision.startswith("confirm")) else "held",
            "notes": final_notes or "Final Experience Compilation checkpoint completed.",
        }

    return collected


def _checkpoint_inputs(
    mode: str,
    session: dict[str, Any],
    status: dict[str, Any],
    summary: dict[str, Any],
    stages: set[str],
) -> dict[str, Any]:
    provided = _load_checkpoint_inputs()
    if provided:
        return provided
    if _truthy_env("AA_MENTOR_INTERACTIVE_CHECKPOINTS"):
        return _collect_interactive_inputs(mode, session, status, summary, stages)
    return {}


def _section(inputs: dict[str, Any], name: str) -> dict[str, Any]:
    value = inputs.get(name)
    return value if isinstance(value, dict) else {}


def _write_checkpoint_history(path: Path, payload: dict[str, Any]) -> None:
    if not path.name.endswith("_checkpoint.json"):
        return
    history_dir = path.parent / "history"
    history_dir.mkdir(parents=True, exist_ok=True)
    created = str(payload.get("created_at") or _now())
    safe_created = "".join(ch if ch.isalnum() else "-" for ch in created).strip("-")
    stem = path.stem.removesuffix("_checkpoint")
    candidate = history_dir / f"{safe_created}_{stem}.json"
    index = 2
    while candidate.exists():
        candidate = history_dir / f"{safe_created}_{stem}_{index}.json"
        index += 1
    write_json(candidate, payload)


def _write_checkpoint(path: Path, payload: dict[str, Any], *, preserve_interactive: bool = True) -> None:
    _write_checkpoint_history(path, payload)
    if preserve_interactive and path.exists() and payload.get("checkpoint_input_source") != "interactive":
        try:
            existing = read_json(path)
        except Exception:
            existing = {}
        if existing.get("checkpoint_input_source") == "interactive":
            return
    write_json(path, payload)


def write_mentor_checkpoints(
    run_root: Path,
    settings: Settings,
    *,
    auto_approve: bool = False,
    human_approved: bool = True,
    stages: Iterable[str] | None = None,
    preserve_interactive: bool = True,
) -> Path | None:
    mode = settings.mentor_mode
    if mode not in {"expert_led", "hybrid"}:
        return None
    enabled_stages = _stage_set(stages)
    pkg = _latest_package(run_root)
    session = read_json(run_root / "session.json") if (run_root / "session.json").exists() else {}
    status = read_json(run_root / "run_status.json") if (run_root / "run_status.json").exists() else {}
    root = run_root / "mentor_checkpoints"
    root.mkdir(parents=True, exist_ok=True)
    source = "human_expert" if mode == "expert_led" else "hybrid_human_approval"
    model_draft_source = "mentor_model_provider" if mode == "hybrid" else None
    common = {
        "mentor_mode": mode,
        "source": source,
        "auto_approved": bool(auto_approve),
        "human_approved": bool(human_approved),
        "model_draft_source": model_draft_source,
        "run_id": session.get("run_id") or status.get("run_id"),
        "task_id": session.get("task_id"),
    }
    summary = _attempt_summary(pkg)
    inputs = _checkpoint_inputs(mode, session, status, summary, enabled_stages)
    intake_input = _section(inputs, "task_intake")
    rubric_input = _section(inputs, "rubric")
    evaluation_input = _section(inputs, "evaluation")
    revision_input = _section(inputs, "revision")
    final_input = _section(inputs, "final_approval")
    input_source = "interactive" if inputs else ("auto_approve" if auto_approve else "recorded")
    if "task_intake" in enabled_stages:
        _write_checkpoint(
            root / "task_intake_checkpoint.json",
            {
                **common,
                "created_at": _now(),
                "checkpoint_type": "task_intake",
                "title": intake_input.get("edited_title") or status.get("task_title"),
                "instruction": session.get("task_instruction"),
                "decision": intake_input.get("decision") or "confirmed",
                "notes": intake_input.get("notes") or ("Confirmed as-is." if auto_approve else "Human expert task-intake audit recorded."),
                "checkpoint_input_source": input_source,
            },
            preserve_interactive=preserve_interactive,
        )
    if "rubric" in enabled_stages:
        _write_checkpoint(
            root / "rubric_checkpoint.json",
            {
                **common,
                "created_at": _now(),
                "checkpoint_type": "rubric",
                "rubric_refs": [
                    "rubric/worker_visible_rubric.md",
                    "rubric/rubric.json",
                ],
                "decision": rubric_input.get("decision") or "confirmed",
                "additional_criteria": rubric_input.get("additional_criteria") or [],
                "notes": rubric_input.get("notes") or ("Apprentice-generated rubric confirmed." if auto_approve else "Human expert rubric audit recorded."),
                "checkpoint_input_source": input_source,
            },
            preserve_interactive=preserve_interactive,
        )
    default_passed = status.get("task_status") == "completed"
    if "evaluation" in enabled_stages:
        passed = evaluation_input.get("passed", default_passed)
        _write_checkpoint(
            root / "evaluation_checkpoint.json",
            {
                **common,
                "created_at": _now(),
                "checkpoint_type": "evaluation_verifier",
                "task_status": status.get("task_status"),
                "run_status": status.get("run_status"),
                "score": evaluation_input.get("score", 1.0 if passed else 0.0),
                "passed": passed,
                "failed_criteria": evaluation_input.get("failed_criteria") or ([] if passed else ["task_status_not_completed"]),
                "feedback": evaluation_input.get("feedback") or status.get("last_operational_error") or status.get("latest_message") or "Human checkpoint recorded.",
                "attempt_summary": evaluation_input.get("attempt_summary") or summary,
                "checkpoint_input_source": input_source,
            },
            preserve_interactive=preserve_interactive,
        )
    if "revision" in enabled_stages:
        _write_checkpoint(
            root / "revision_checkpoint.json",
            {
                **common,
                "created_at": _now(),
                "checkpoint_type": "revision_decision",
                "revision_should_run": bool(revision_input.get("revision_should_run", False)),
                "decision": revision_input.get("decision") or "finish",
                "notes": revision_input.get("notes") or "No additional human-requested revision recorded by this checkpoint.",
                "checkpoint_input_source": input_source,
            },
            preserve_interactive=preserve_interactive,
        )
    if "final_approval" in enabled_stages:
        _write_checkpoint(
            root / "final_approval_checkpoint.json",
            {
                **common,
                "created_at": _now(),
                "checkpoint_type": "final_approval",
                "approved_for_local_bundle": final_input.get("approved_for_local_bundle", True),
                "contribution_bundle_path": status.get("contribution_bundle_path"),
                "decision": final_input.get("decision") or "confirmed",
                "notes": final_input.get("notes") or "Final Experience Compilation checkpoint recorded.",
                "checkpoint_input_source": input_source,
            },
            preserve_interactive=preserve_interactive,
        )
    (root / "README.md").write_text(
        "# Mentor Checkpoints\n\n"
        "These files record Expert-Led or Organization Custom audit checkpoints for the apprenticeship session. "
        "They are local audit artifacts and do not contain raw provider secrets.\n"
    )
    _mark_source_artifact_audited(pkg, source=source, mode=mode, input_source=input_source, stages=enabled_stages)
    return root
