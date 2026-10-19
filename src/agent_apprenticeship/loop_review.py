from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .io import read_workflow_json as read_json, write_json
from .public_sanitizer import sanitize_public_obj, sanitize_public_text


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _as_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    return {}


def _read(path: Path) -> dict[str, Any]:
    try:
        obj = read_json(path)
    except Exception:
        return {}
    return obj if isinstance(obj, dict) else {}


def _rel(pkg: Path, path: Path) -> str:
    try:
        return path.relative_to(pkg).as_posix()
    except ValueError:
        return path.as_posix()


def _existing_ref(pkg: Path, ref: str | None) -> str | None:
    if not ref:
        return None
    text = str(ref).replace("\\", "/").strip()
    if not text or text.startswith("/") or ".." in Path(text).parts:
        return None
    return text


def _list_refs(pkg: Path, values: Any) -> list[str]:
    raw = values if isinstance(values, list) else [values]
    refs: list[str] = []
    for value in raw:
        ref = _existing_ref(pkg, str(value) if value is not None else None)
        if ref and ref not in refs:
            refs.append(ref)
    return refs


def _attempt_kind_from_actual(actual: dict[str, Any]) -> str:
    return str(actual.get("attempt_kind") or "baseline")


def _loop_id(pkg: Path) -> str:
    return f"loop_{pkg.name}"


def _iteration_dir(pkg: Path, iteration: int) -> Path:
    return pkg / "loops" / "iterations" / f"{iteration:03d}"


def _trace_refs(pkg: Path, attempt_kind: str) -> dict[str, str | None]:
    base = pkg / "attempts" / attempt_kind
    refs = {
        "canonical": f"attempts/{attempt_kind}/agent_trace.json" if (base / "agent_trace.json").exists() else None,
        "raw": f"attempts/{attempt_kind}/agent_trace.raw.json" if (base / "agent_trace.raw.json").exists() else None,
        "normalized": f"attempts/{attempt_kind}/agent_trace.normalized.json" if (base / "agent_trace.normalized.json").exists() else None,
        "normalization_report": f"attempts/{attempt_kind}/trace_normalization_report.json" if (base / "trace_normalization_report.json").exists() else None,
    }
    return refs


def _key_trace_steps(trace: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for step in trace.get("steps") or []:
        if not isinstance(step, dict):
            continue
        if step.get("action") == "user_message":
            continue
        out.append(
            sanitize_public_obj(
                {
                    "step": step.get("step"),
                    "actor": step.get("actor"),
                    "actor_kind": step.get("actor_kind"),
                    "operation": step.get("operation"),
                    "interaction_surface": step.get("interaction_surface"),
                    "environment_domain": step.get("environment_domain"),
                    "input_source": step.get("input_source"),
                    "output": sanitize_public_text(str(step.get("output") or ""))[:500],
                    "state_change": sanitize_public_text(str(step.get("state_change") or ""))[:500],
                    "step_outcome": step.get("step_outcome"),
                    "success": step.get("success"),
                    "artifact_refs": step.get("artifact_refs") or [],
                    "missing_surface_fields": step.get("missing_surface_fields") or [],
                    "trace_quality_warnings": step.get("trace_quality_warnings") or [],
                }
            )
        )
        if len(out) >= 12:
            break
    return out


def _surface_summary(trace: dict[str, Any]) -> dict[str, Any]:
    surfaces: Counter[str] = Counter()
    domains: Counter[str] = Counter()
    warnings: list[dict[str, Any]] = []
    for step in trace.get("steps") or []:
        if not isinstance(step, dict):
            continue
        surface = step.get("interaction_surface") or "unknown"
        domain = step.get("environment_domain") or "unknown"
        surfaces[str(surface)] += 1
        domains[str(domain)] += 1
        missing = step.get("missing_surface_fields") or []
        if missing:
            warnings.append(
                {
                    "step": step.get("step"),
                    "interaction_surface": surface,
                    "surface_capture_status": step.get("surface_capture_status"),
                    "missing_surface_fields": missing[:12],
                }
            )
    return {
        "interaction_surfaces": dict(surfaces),
        "environment_domains": dict(domains),
        "surface_capture_warnings": warnings[:20],
    }


def _artifact_index(pkg: Path, actual: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    index = pkg / "artifacts_index.json"
    if index.exists():
        raw = read_json(index)
        if isinstance(raw, list):
            for row in raw[:100]:
                if isinstance(row, dict):
                    rows.append(sanitize_public_obj(row))
    if rows:
        return rows
    refs = []
    for key in ["primary_output_ref", "deliverable_refs", "artifact_refs", "files_created", "input_artifact_refs"]:
        refs.extend(_list_refs(pkg, actual.get(key)))
    for ref in dict.fromkeys(refs):
        rows.append({"artifact_ref": ref, "exists": (pkg / ref).exists()})
    return rows


def _rubric_status(pkg: Path) -> dict[str, Any]:
    rubric = _read(pkg / "rubric" / "rubric.json")
    quality = _read(pkg / "rubric" / "rubric_quality_report.json")
    return sanitize_public_obj(
        {
            "rubric_refs": [ref for ref in ["rubric/rubric.json", "rubric/worker_visible_rubric.md"] if (pkg / ref).exists()],
            "rubric_item_count": len(rubric.get("rubric_items") or []),
            "mentor_audited": bool((rubric.get("metadata_json") or {}).get("mentor_audited")),
            "audit_status": (rubric.get("metadata_json") or {}).get("audit_status"),
            "quality_report": quality,
        }
    )


def _status_from_grader_verifier(grader: dict[str, Any], verifier: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    failed = list(grader.get("failed_criteria") or [])
    verifier_failed = list(verifier.get("failed_checks") or verifier.get("failed_criteria") or [])
    known_failures = [str(v) for v in failed + verifier_failed if v]
    grade_score = grader.get("final_score", grader.get("score"))
    rubric_status = {
        "grader_result_ref": None,
        "score": grade_score,
        "passed": grader.get("passed"),
        "failed_criteria": failed,
        "score_reliability": grader.get("score_reliability"),
    }
    verifier_status = {
        "verifier_result_ref": None,
        "passed": verifier.get("passed"),
        "verdict": verifier.get("verdict"),
        "failed_checks": verifier_failed,
        "evidence_refs": verifier.get("evidence_refs") or [],
        "notes": verifier.get("notes") or verifier.get("verifier_notes"),
    }
    return rubric_status, verifier_status, known_failures


def _evaluator_status(feedback: dict[str, Any]) -> dict[str, Any]:
    return sanitize_public_obj(
        {
            "feedback_ref": "feedback/baseline_evaluator_feedback.json" if feedback else None,
            "feedback_type": feedback.get("feedback_type"),
            "revision_priority": feedback.get("revision_priority"),
            "failed_rubric_items": feedback.get("failed_rubric_items") or feedback.get("failed_or_weak_rubric_items") or [],
            "summary": feedback.get("feedback_summary"),
            "actionable_feedback": feedback.get("actionable_feedback") or [],
            "provider": feedback.get("provider"),
            "model": feedback.get("model"),
        }
    )


def _output_refs(pkg: Path, actual: dict[str, Any], attempt_kind: str) -> list[str]:
    refs = [f"attempts/{attempt_kind}/actual_outputs.json"]
    for key in ["primary_output_ref", "deliverable_refs", "artifact_refs", "files_created", "stdout_ref", "stderr_ref", "raw_log_refs"]:
        refs.extend(_list_refs(pkg, actual.get(key)))
    return list(dict.fromkeys(refs))


def _missing_outputs(actual: dict[str, Any]) -> list[str]:
    missing: list[str] = []
    for key in ["primary_output_ref", "deliverable_refs", "artifact_refs", "files_created"]:
        value = actual.get(key)
        if value in (None, "", []):
            missing.append(key)
    if actual.get("status") not in {"success", "completed"}:
        missing.append("successful_status")
    return missing


def _diff_from_previous(pkg: Path, iteration: int, actual: dict[str, Any]) -> dict[str, Any]:
    if iteration <= 1:
        return {"available": False, "reason": "First loop iteration has no previous output snapshot."}
    previous_path = _iteration_dir(pkg, iteration - 1) / "output_snapshot.json"
    previous = _read(previous_path)
    previous_refs = set(_output_refs(pkg, previous, _attempt_kind_from_actual(previous)))
    current_refs = set(_output_refs(pkg, actual, _attempt_kind_from_actual(actual)))
    return sanitize_public_obj(
        {
            "available": True,
            "previous_iteration_ref": f"loops/iterations/{iteration - 1:03d}/output_snapshot.json",
            "added_refs": sorted(current_refs - previous_refs),
            "removed_refs": sorted(previous_refs - current_refs),
            "changed_summary": {
                "previous_output_summary": previous.get("output_summary"),
                "current_output_summary": actual.get("output_summary"),
            },
        }
    )


def reviewer_feedback_from_evaluator(
    feedback: Any,
    grader: Any,
    verifier: Any,
    *,
    reviewer_type: str = "mentor_model_provider",
    review_source: str = "review_packet",
    fallback_reviewer_type: str | None = None,
) -> dict[str, Any]:
    fb = _as_dict(feedback)
    gr = _as_dict(grader)
    ve = _as_dict(verifier)
    score = gr.get("final_score", gr.get("score"))
    try:
        numeric_score = float(score)
    except (TypeError, ValueError):
        numeric_score = 0.0
    failed = list(fb.get("failed_rubric_items") or fb.get("failed_or_weak_rubric_items") or gr.get("failed_criteria") or [])
    blocking = [str(item) for item in failed if item]
    verdict = "pass"
    if blocking or ve.get("passed") is False or numeric_score < 0.75:
        verdict = "revise"
    if fb.get("metadata_json", {}).get("mentor_evaluator_status") == "unavailable":
        reviewer_type = fallback_reviewer_type or "mentor_model_unavailable"
        verdict = "needs_expert" if blocking else "stop_with_recorded_outcome"
    return sanitize_public_obj(
        {
            "reviewer_type": reviewer_type,
            "review_source": review_source,
            "verdict": verdict,
            "score": numeric_score,
            "blocking_issues": blocking,
            "non_blocking_issues": [str(x) for x in fb.get("artifact_specific_comments") or []],
            "evidence_refs": list(dict.fromkeys((fb.get("evidence_refs") or []) + (fb.get("artifact_refs") or []) + (ve.get("evidence_refs") or []))),
            "feedback": fb.get("feedback_summary") or "No reviewer feedback summary was available.",
            "revision_instructions": fb.get("actionable_feedback") or [],
            "stop_condition": "pass" if verdict == "pass" else "revise_or_request_expert_feedback",
            "training_value_notes": "Loop reviewer feedback can supervise critique, revision, verifier, and reward-model rows.",
            "created_at": _now(),
        }
    )


def human_feedback_from_checkpoint(checkpoint: dict[str, Any], *, loop_id: str, iteration: int) -> dict[str, Any]:
    input_source = checkpoint.get("checkpoint_input_source")
    if input_source == "auto_approve":
        reviewer_type = "simulated_for_test"
        source = "auto_expert_checkpoint"
    elif checkpoint.get("source") == "hybrid_human_approval":
        reviewer_type = "org_custom_reviewer"
        source = "hybrid_human_approval"
    else:
        reviewer_type = "human_expert"
        source = "interactive" if input_source == "interactive" else "recorded"
    return sanitize_public_obj(
        {
            "feedback_id": f"feedback_{loop_id}_{iteration:03d}",
            "loop_id": loop_id,
            "loop_iteration": iteration,
            "reviewer_type": reviewer_type,
            "expert_feedback_source": source,
            "feedback_text": checkpoint.get("feedback") or checkpoint.get("notes") or "Expert checkpoint recorded.",
            "requested_changes": [checkpoint.get("notes")] if checkpoint.get("revision_should_run") else [],
            "accepted_outputs": [] if checkpoint.get("revision_should_run") else ["current_attempt_outputs"],
            "rejected_outputs": ["current_attempt_outputs"] if checkpoint.get("revision_should_run") else [],
            "evidence_refs": checkpoint.get("evidence_refs") or [],
            "created_at": checkpoint.get("created_at") or _now(),
        }
    )


def _default_loop_decision(
    *,
    loop_id: str,
    iteration: int,
    reviewer_feedback: dict[str, Any] | None,
    stop_reason: str | None,
    continue_loop: bool | None,
) -> dict[str, Any]:
    verdict = (reviewer_feedback or {}).get("verdict")
    if continue_loop is None:
        continue_loop = verdict == "revise"
    return sanitize_public_obj(
        {
            "loop_id": loop_id,
            "loop_iteration": iteration,
            "decision": "continue_loop" if continue_loop else "stop",
            "verdict": verdict or ("revise" if continue_loop else "pass"),
            "stop_condition": stop_reason or ("revision_requested" if continue_loop else "review_complete"),
            "created_at": _now(),
        }
    )


def _normalized_revision_plan(
    raw_plan: Any,
    *,
    loop_id: str,
    iteration: int,
    reviewer_feedback: dict[str, Any] | None,
    loop_decision: dict[str, Any],
) -> dict[str, Any]:
    plan = _as_dict(raw_plan)
    decision = str(loop_decision.get("decision") or "")
    should_continue = decision == "continue_loop"
    feedback_ref = f"loops/iterations/{iteration:03d}/reviewer_feedback.json"
    issues = (
        plan.get("issues_to_fix")
        or plan.get("revision_items")
        or (reviewer_feedback or {}).get("blocking_issues")
        or (reviewer_feedback or {}).get("revision_instructions")
        or []
    )
    verification_steps = (
        plan.get("verification_steps")
        or plan.get("checks_to_rerun")
        or ["Reinspect actual_outputs.json, artifacts, trace refs, rubric status, and verifier status."]
    )
    deferred_feedback_reason = plan.get("feedback_not_applied_reason")
    if issues and not should_continue and not deferred_feedback_reason:
        deferred_feedback_reason = (
            loop_decision.get("stop_condition")
            or "Loop stopped because the reviewer verdict did not require another revision iteration."
        )
    defaults = {
        "loop_id": loop_id,
        "from_iteration": iteration,
        "to_iteration": iteration + 1 if should_continue else None,
        "feedback_refs": [feedback_ref] if reviewer_feedback else [],
        "issues_to_fix": issues,
        "artifacts_to_update": plan.get("artifacts_to_update") or [],
        "outputs_to_update": plan.get("outputs_to_update") or [],
        "verification_steps": verification_steps,
        "stop_conditions": plan.get("stop_conditions") or [loop_decision.get("stop_condition") or "review_complete"],
        "revision_required": should_continue,
        "feedback_applied": plan.get("feedback_applied") if plan.get("feedback_applied") is not None else (None if should_continue or not issues else False),
        "feedback_not_applied_reason": deferred_feedback_reason,
        "created_at": plan.get("created_at") or _now(),
    }
    merged = {**defaults, **{k: v for k, v in plan.items() if v not in (None, "", [])}}
    return sanitize_public_obj(merged)


def write_loop_review_iteration(
    package_root: Path,
    *,
    iteration: int,
    mode: str | None,
    actual_outputs: Any,
    trace: Any,
    grader_result: Any | None = None,
    verifier_result: Any | None = None,
    evaluator_feedback: Any | None = None,
    revision_plan: Any | None = None,
    reviewer_feedback: dict[str, Any] | None = None,
    loop_decision: dict[str, Any] | None = None,
    stop_reason: str | None = None,
    continue_loop: bool | None = None,
) -> Path:
    package_root = Path(package_root)
    actual = _as_dict(actual_outputs)
    trace_data = _as_dict(trace)
    grader = _as_dict(grader_result)
    verifier = _as_dict(verifier_result)
    feedback = _as_dict(evaluator_feedback)
    attempt_kind = _attempt_kind_from_actual(actual)
    loop_id = _loop_id(package_root)
    out_dir = _iteration_dir(package_root, iteration)
    out_dir.mkdir(parents=True, exist_ok=True)

    rubric_result_status, verifier_status, known_failures = _status_from_grader_verifier(grader, verifier)
    if attempt_kind:
        rubric_result_status["grader_result_ref"] = f"grading/{attempt_kind}_grader_result.json" if (package_root / "grading" / f"{attempt_kind}_grader_result.json").exists() else None
        verifier_status["verifier_result_ref"] = f"grading/{attempt_kind}_verifier_result.json" if (package_root / "grading" / f"{attempt_kind}_verifier_result.json").exists() else None
    output_refs = _output_refs(package_root, actual, attempt_kind)
    diff = _diff_from_previous(package_root, iteration, actual)
    if reviewer_feedback is None and feedback:
        reviewer_feedback = reviewer_feedback_from_evaluator(feedback, grader, verifier)
    if loop_decision is None:
        loop_decision = _default_loop_decision(
            loop_id=loop_id,
            iteration=iteration,
            reviewer_feedback=reviewer_feedback,
            stop_reason=stop_reason,
            continue_loop=continue_loop,
        )
    plan = _normalized_revision_plan(
        revision_plan,
        loop_id=loop_id,
        iteration=iteration,
        reviewer_feedback=reviewer_feedback,
        loop_decision=loop_decision,
    )

    packet = sanitize_public_obj(
        {
            "loop_id": loop_id,
            "loop_iteration": iteration,
            "mode": mode,
            "task_id": package_root.name,
            "package_id": package_root.name,
            "run_id": package_root.parent.parent.name if package_root.parent.name == "packages" else None,
            "current_goal": (_read(package_root / "task" / "task_intake_spec.json").get("normalized_instruction") or _read(package_root / "task" / "raw_task_record.json").get("raw_description")),
            "current_attempt_summary": actual.get("output_summary"),
            "current_output_refs": output_refs,
            "actual_outputs_ref": f"attempts/{attempt_kind}/actual_outputs.json",
            "artifact_refs": list(dict.fromkeys((actual.get("artifact_refs") or []) + (actual.get("deliverable_refs") or []) + (actual.get("files_created") or []))),
            "artifact_previews": _artifact_index(package_root, actual)[:25],
            "trace_refs": _trace_refs(package_root, attempt_kind),
            "key_trace_steps": _key_trace_steps(trace_data),
            "rubric_refs": ["rubric/rubric.json", "rubric/worker_visible_rubric.md"],
            "rubric_status": {**_rubric_status(package_root), "attempt_result": rubric_result_status},
            "verifier_status": verifier_status,
            "evaluator_status": _evaluator_status(feedback),
            "known_failures": known_failures,
            "missing_outputs": _missing_outputs(actual),
            "open_questions": [] if not known_failures else ["Can the listed failures be repaired in the next iteration?"],
            "surface_summary": _surface_summary(trace_data),
            "environment_summary": _surface_summary(trace_data).get("environment_domains"),
            "previous_iteration_refs": [f"loops/iterations/{iteration - 1:03d}/review_packet.json"] if iteration > 1 else [],
            "diff_from_previous_iteration": diff,
            "reviewer_instructions": [
                "Inspect actual_outputs.json, artifacts, trace evidence, rubric status, verifier status, and this packet before judging.",
                "Do not rely on Apprentice Agent self-evaluation as outcome truth.",
                "Choose accept, revise, ask for expert feedback, stop with a recorded outcome, or continue loop based on grounded evidence.",
            ],
            "recommended_review_focus": known_failures or ["Confirm deliverables, evidence refs, and verifier/rubric alignment."],
            "stop_continue_options": ["accept", "revise", "ask_for_expert_feedback", "stop_with_recorded_outcome", "continue_loop"],
            "created_at": _now(),
        }
    )

    write_json(out_dir / "output_snapshot.json", actual)
    write_json(out_dir / "artifact_index.json", _artifact_index(package_root, actual))
    write_json(out_dir / "trace_refs.json", _trace_refs(package_root, attempt_kind))
    write_json(out_dir / "rubric_status.json", packet["rubric_status"])
    write_json(out_dir / "verifier_status.json", verifier_status)
    write_json(out_dir / "evaluator_feedback.json", feedback)
    write_json(out_dir / "review_packet.json", packet)
    write_json(out_dir / "diff_report.json", diff)
    write_json(out_dir / "reviewer_feedback.json", reviewer_feedback or {})
    write_json(out_dir / "revision_plan.json", plan)
    write_json(out_dir / "loop_decision.json", loop_decision)
    _write_review_packet_md(out_dir / "review_packet.md", packet)
    _write_loop_manifest(package_root)
    return out_dir / "review_packet.json"


def _write_review_packet_md(path: Path, packet: dict[str, Any]) -> None:
    lines = [
        f"# Loop Iteration {packet.get('loop_iteration')} Review Packet",
        "",
        f"Task: {packet.get('task_id')}",
        f"Mode: {packet.get('mode') or 'unknown'}",
        "",
        "## Task Goal",
        str(packet.get("current_goal") or "No goal recorded."),
        "",
        "## What Changed This Loop",
        str((packet.get("diff_from_previous_iteration") or {}).get("changed_summary") or "Initial attempt."),
        "",
        "## Outputs Produced",
    ]
    for ref in packet.get("current_output_refs") or []:
        lines.append(f"- {ref}")
    lines.extend(["", "## Evidence To Inspect"])
    for ref in [packet.get("actual_outputs_ref"), *((packet.get("trace_refs") or {}).values())]:
        if ref:
            lines.append(f"- {ref}")
    lines.extend(["", "## Rubric And Verifier Status"])
    lines.append(f"- Rubric score/status: {(packet.get('rubric_status') or {}).get('attempt_result')}")
    lines.append(f"- Verifier status: {packet.get('verifier_status')}")
    lines.extend(["", "## Problems Or Uncertainty"])
    failures = packet.get("known_failures") or packet.get("missing_outputs") or []
    if failures:
        lines.extend(f"- {item}" for item in failures)
    else:
        lines.append("- No blocking issue was recorded in this packet.")
    lines.extend(["", "## Suggested Next Decision"])
    lines.append("- accept")
    lines.append("- revise")
    lines.append("- ask for expert feedback")
    lines.append("- stop with recorded outcome")
    lines.append("- continue loop")
    path.write_text("\n".join(lines).rstrip() + "\n")


def _write_loop_manifest(pkg: Path) -> None:
    iterations: list[dict[str, Any]] = []
    reviewer_types: list[str] = []
    final_verdict = None
    for packet_path in sorted((pkg / "loops" / "iterations").glob("*/review_packet.json")):
        iteration_dir = packet_path.parent
        packet = _read(packet_path)
        feedback = _read(iteration_dir / "reviewer_feedback.json")
        decision = _read(iteration_dir / "loop_decision.json")
        reviewer_type = feedback.get("reviewer_type")
        if reviewer_type and reviewer_type not in reviewer_types:
            reviewer_types.append(str(reviewer_type))
        final_verdict = decision.get("verdict") or feedback.get("verdict") or final_verdict
        iterations.append(
            {
                "loop_iteration": packet.get("loop_iteration"),
                "review_packet_ref": _rel(pkg, packet_path),
                "review_packet_md_ref": _rel(pkg, iteration_dir / "review_packet.md"),
                "reviewer_feedback_ref": _rel(pkg, iteration_dir / "reviewer_feedback.json"),
                "revision_plan_ref": _rel(pkg, iteration_dir / "revision_plan.json"),
                "loop_decision_ref": _rel(pkg, iteration_dir / "loop_decision.json"),
                "diff_report_ref": _rel(pkg, iteration_dir / "diff_report.json"),
                "reviewer_type": reviewer_type,
                "verdict": decision.get("verdict") or feedback.get("verdict"),
            }
        )
    manifest = sanitize_public_obj(
        {
            "loop_id": _loop_id(pkg),
            "package_id": pkg.name,
            "loop_iteration_count": len(iterations),
            "has_loop_review_packets": bool(iterations),
            "reviewer_types": reviewer_types,
            "final_loop_verdict": final_verdict,
            "iterations": iterations,
            "created_at": _now(),
            "updated_at": _now(),
        }
    )
    (pkg / "loops").mkdir(parents=True, exist_ok=True)
    write_json(pkg / "loops" / "loop_manifest.json", manifest)


def record_human_loop_feedback(package_root: Path, *, iteration: int, checkpoint: dict[str, Any], revision_requested: bool) -> None:
    loop_id = _loop_id(package_root)
    feedback = human_feedback_from_checkpoint(checkpoint, loop_id=loop_id, iteration=iteration)
    decision = _default_loop_decision(
        loop_id=loop_id,
        iteration=iteration,
        reviewer_feedback={"verdict": "revise" if revision_requested else "pass"},
        stop_reason="expert_requested_revision" if revision_requested else "expert_finished_loop",
        continue_loop=revision_requested,
    )
    out_dir = _iteration_dir(package_root, iteration)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "reviewer_feedback.json", feedback)
    write_json(out_dir / "loop_decision.json", decision)
    existing_plan = _read(out_dir / "revision_plan.json")
    requested_changes = feedback.get("requested_changes") or []
    existing_plan.update(
        {
            "loop_id": loop_id,
            "from_iteration": iteration,
            "to_iteration": iteration + 1 if revision_requested else None,
            "feedback_refs": [f"loops/iterations/{iteration:03d}/reviewer_feedback.json"],
            "issues_to_fix": requested_changes,
            "revision_required": revision_requested,
            "feedback_applied": (True if not revision_requested and not requested_changes else existing_plan.get("feedback_applied")),
            "feedback_not_applied_reason": (None if not revision_requested and not requested_changes else existing_plan.get("feedback_not_applied_reason")),
            "created_at": existing_plan.get("created_at") or _now(),
        }
    )
    write_json(out_dir / "revision_plan.json", sanitize_public_obj(existing_plan))
    _write_loop_manifest(package_root)


def record_followup_review_update(run_root: Path, *, followup_index: int, record_only: bool, note: str) -> Path:
    root = run_root / "loops" / "followups" / f"{followup_index:03d}"
    root.mkdir(parents=True, exist_ok=True)
    packages = sorted((run_root / "packages").glob("*")) if (run_root / "packages").exists() else []
    latest = packages[-1] if packages else None
    refs: list[str] = []
    if latest and (latest / "loops" / "loop_manifest.json").exists():
        refs.append(f"packages/{latest.name}/loops/loop_manifest.json")
    payload = sanitize_public_obj(
        {
            "followup_index": followup_index,
            "record_only": record_only,
            "note": note,
            "previous_loop_refs": refs,
            "new_execution_data_recorded": not record_only,
            "reason_no_new_loop_execution": "Record-only follow-up captured user feedback without running the Apprentice Agent." if record_only else None,
            "created_at": _now(),
        }
    )
    write_json(root / "followup_review_update.json", payload)
    (root / "followup_review_update.md").write_text(
        "# Follow-up Review Update\n\n"
        f"Follow-up: {followup_index}\n\n"
        f"Record-only: {record_only}\n\n"
        f"{payload.get('reason_no_new_loop_execution') or note}\n"
    )
    return root / "followup_review_update.json"


def summarize_loop_reviews(root: Path) -> dict[str, Any]:
    root = Path(root)
    manifests = list(root.glob("loops/loop_manifest.json"))
    manifests.extend(root.glob("packages/*/loops/loop_manifest.json"))
    if not manifests:
        return {
            "has_loop_review_packets": False,
            "loop_iteration_count": 0,
            "reviewer_types": [],
            "final_loop_verdict": None,
            "loop_review_refs": [],
        }
    total = 0
    reviewers: list[str] = []
    refs: list[str] = []
    final = None
    package_manifest_count = len(list(root.glob("packages/*/loops/loop_manifest.json")))
    for path in sorted(manifests):
        data = _read(path)
        total += int(data.get("loop_iteration_count") or 0)
        for reviewer in data.get("reviewer_types") or []:
            if reviewer and reviewer not in reviewers:
                reviewers.append(str(reviewer))
        final = data.get("final_loop_verdict") or final
        for iteration in data.get("iterations") or []:
            ref = iteration.get("review_packet_ref")
            if ref:
                if path.parent.parent.parent.name == "packages":
                    if package_manifest_count > 1:
                        inner = str(ref).removeprefix("loops/")
                        refs.append(f"loops/{path.parent.parent.name}/{inner}")
                    else:
                        refs.append(str(ref))
                else:
                    refs.append(str(ref))
    return {
        "has_loop_review_packets": total > 0,
        "loop_iteration_count": total,
        "reviewer_types": reviewers,
        "final_loop_verdict": final,
        "loop_review_refs": refs[:50],
    }
