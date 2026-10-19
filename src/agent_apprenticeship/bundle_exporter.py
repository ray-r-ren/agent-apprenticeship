from __future__ import annotations
from .io import validate_trace_tree

import shutil
import tempfile
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import Settings, apprenticeship_mode_display, get_settings, normalize_apprenticeship_mode, normalize_mentor_mode
from .io import append_jsonl, read_workflow_json as read_json, read_workflow_jsonl as read_jsonl, write_json
from .public_sanitizer import sanitize_public_obj, sanitize_public_text
from .release_exporter import create_release
from .tdo import tdo_manifest_fields_from_report
from .loop_review import summarize_loop_reviews
from .version import get_package_version


def _count_jsonl(path: Path) -> int:
    return len(read_jsonl(path))


def _trace_step_count(path: Path) -> int:
    return sum(len(row.get("steps") or []) for row in read_jsonl(path))


def _is_nonempty_jsonl(path: Path) -> bool:
    return _count_jsonl(path) > 0


def _copy_jsonl_if_nonempty(src: Path, dst: Path) -> bool:
    if not src.exists() or not _is_nonempty_jsonl(src):
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text("")
    for row in read_jsonl(src):
        append_jsonl(dst, sanitize_public_obj(row) if isinstance(row, dict) else row)
    return True


def _sanitize_bundle_json(data: Any) -> Any:
    if isinstance(data, dict):
        return sanitize_public_obj(data)
    if isinstance(data, list):
        return [sanitize_public_obj(row) if isinstance(row, dict) else row for row in data]
    return data


def _copy_json_if_exists(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    write_json(dst, _sanitize_bundle_json(read_json(src)))
    return True


def _copy_public_file(src: Path, dst: Path) -> bool:
    if not src.exists() or not src.is_file():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.suffix == ".json":
        try:
            write_json(dst, _sanitize_bundle_json(read_json(src)))
        except Exception:
            dst.write_text(sanitize_public_text(src.read_text(errors="replace")) or "")
    elif src.suffix == ".jsonl":
        try:
            dst.write_text("")
            for row in read_jsonl(src):
                append_jsonl(dst, sanitize_public_obj(row) if isinstance(row, dict) else row)
        except Exception:
            dst.write_text(sanitize_public_text(src.read_text(errors="replace")) or "")
    else:
        try:
            dst.write_text(sanitize_public_text(src.read_text(errors="replace")) or "")
        except UnicodeDecodeError:
            shutil.copy2(src, dst)
    return True


def _run_status(release_root: Path) -> str:
    rows = read_jsonl(release_root / "packages_index.jsonl")
    statuses = [row.get("task_status") for row in rows if row.get("task_status")]
    if statuses and all(status == "completed" for status in statuses):
        return "completed"
    if statuses and all(status == "failed" for status in statuses):
        return "failed"
    return "partial" if statuses else "failed"


def contribution_counts(release_root: Path) -> dict[str, int]:
    trace_count = _count_jsonl(release_root / "agent_traces.jsonl")
    actual_outputs_count = _count_jsonl(release_root / "actual_outputs.jsonl")
    return {
        "attempts": actual_outputs_count,
        "actual_outputs_count": actual_outputs_count,
        "trace_count": trace_count,
        "traced_steps": _trace_step_count(release_root / "agent_traces.jsonl"),
        "process_supervision_rows": _count_jsonl(release_root / "process_supervision.jsonl"),
        "reward_modeling_rows": _count_jsonl(release_root / "reward_modeling.jsonl"),
        "revision_preference_pairs": _count_jsonl(release_root / "revision_preference_pairs.jsonl"),
    }


def _write_card(bundle_root: Path) -> None:
    (bundle_root / "contribution_card.md").write_text(
        "# Agent Apprenticeship Agent Experience Package\n\n"
        "This local bundle packages an apprenticeship session for public ecosystem contribution. "
        "It includes session events, task instructions, traces, actual outputs, evaluation results, "
        "and learning-data views when available.\n\n"
        "No automatic upload is performed. Inspect this folder locally before sharing.\n\n"
        "To contribute, submit the bundle to the Agent Apprenticeship ecosystem repo, "
        "or get help in Slack: https://join.slack.com/t/fsycommunity/shared_invite/zt-37417grrb-jFD6BQIYgC5wEMrW2bHssw\n"
    )


def _session_mentor_mode(session: dict[str, Any], settings: Settings) -> str:
    return normalize_mentor_mode(session.get("mentor_mode") or session.get("evaluation_mode") or settings.mentor_mode)


def _session_apprenticeship_mode(session: dict[str, Any], settings: Settings) -> str:
    return normalize_apprenticeship_mode(
        session.get("apprenticeship_mode")
        or session.get("mentor_mode")
        or session.get("evaluation_mode")
        or settings.apprenticeship_mode
    )


def _clean_session_metadata(session: dict[str, Any], settings: Settings) -> dict[str, Any]:
    allowed = {
        "run_id",
        "session_id",
        "run_status",
        "task_status",
        "task_id",
        "task_instruction",
        "task_assets",
        "latest_package",
        "latest_attempt_id",
        "max_improvement_loops",
        "apprentice_agent",
        "model_provider",
        "status_reason",
        "session_status",
    }
    clean = {k: v for k, v in session.items() if k in allowed and v is not None}
    clean["apprenticeship_mode"] = _session_apprenticeship_mode(session, settings)
    clean["apprenticeship_mode_display"] = apprenticeship_mode_display(clean["apprenticeship_mode"])
    clean["mentor_mode"] = _session_mentor_mode(session, settings)
    return clean


def _copy_session_files(run_root: Path, bundle_root: Path, settings: Settings) -> None:
    if (run_root / "session_events.jsonl").exists():
        shutil.copy2(run_root / "session_events.jsonl", bundle_root / "session_events.jsonl")
    if (run_root / "session.json").exists():
        write_json(bundle_root / "session_metadata.json", _clean_session_metadata(read_json(run_root / "session.json"), settings))
    if (run_root / "task").exists():
        shutil.copytree(run_root / "task", bundle_root / "task", dirs_exist_ok=True)
    if (run_root / "mentor_checkpoints").exists():
        shutil.copytree(run_root / "mentor_checkpoints", bundle_root / "mentor_checkpoints", dirs_exist_ok=True)
    if (run_root / "loops").exists():
        shutil.copytree(run_root / "loops", bundle_root / "loops", dirs_exist_ok=True)


def _copy_clean_bundle_files(release_root: Path, bundle_root: Path, include_debug: bool) -> None:
    for dirname in ["task", "traces", "outputs", "evaluation", "learning_data"]:
        (bundle_root / dirname).mkdir(parents=True, exist_ok=True)

    _copy_jsonl_if_nonempty(release_root / "agent_traces.jsonl", bundle_root / "traces" / "agent_traces.jsonl")
    _copy_jsonl_if_nonempty(release_root / "raw_agent_traces.jsonl", bundle_root / "traces" / "raw_agent_traces.jsonl")

    _copy_jsonl_if_nonempty(release_root / "actual_outputs.jsonl", bundle_root / "outputs" / "actual_outputs.jsonl")
    _copy_json_if_exists(release_root / "artifacts_index.json", bundle_root / "outputs" / "artifacts_index.json")
    _copy_indexed_package_files(release_root, bundle_root)
    _copy_tdo_outputs(release_root, bundle_root)
    _copy_experience_compiler_outputs(release_root, bundle_root)
    _copy_loop_review_outputs(release_root, bundle_root)

    for name in [
        "grader_results.jsonl",
        "verifier_results.jsonl",
        "evaluator_feedback.jsonl",
        "revision_plans.jsonl",
        "hillclimb_results.jsonl",
    ]:
        _copy_jsonl_if_nonempty(release_root / name, bundle_root / "evaluation" / name)

    for name in [
        "process_supervision.jsonl",
        "reward_modeling.jsonl",
        "revision_preference_pairs.jsonl",
        "training_signals.jsonl",
    ]:
        _copy_jsonl_if_nonempty(release_root / name, bundle_root / "learning_data" / name)

    if include_debug:
        copied = False
        for name in [
            "trace_normalization_reports.jsonl",
            "actual_outputs_normalization_reports.jsonl",
            "role_results_index.jsonl",
        ]:
            copied = _copy_jsonl_if_nonempty(release_root / name, bundle_root / "debug" / name) or copied
        copied = _copy_json_if_exists(release_root / "quality_report.json", bundle_root / "debug" / "quality_report.json") or copied
        if not copied and (bundle_root / "debug").exists():
            shutil.rmtree(bundle_root / "debug")


def _copy_indexed_package_files(release_root: Path, bundle_root: Path) -> None:
    index_path = release_root / "artifacts_index.json"
    if not index_path.exists():
        return
    raw_rows = read_json(index_path)
    if not isinstance(raw_rows, list):
        return
    task_ids = sorted({str(row.get("task_id")) for row in raw_rows if isinstance(row, dict) and row.get("task_id")})
    multiple_tasks = len(task_ids) > 1
    rewritten: list[dict[str, Any]] = []
    for row in raw_rows:
        if not isinstance(row, dict):
            continue
        rel = str(row.get("package_relative_path") or "")
        task_id = str(row.get("task_id") or "")
        if not rel or not task_id or is_unsafe_bundle_ref(rel):
            rewritten.append(sanitize_public_obj(row))
            continue
        src = release_root / "packages" / task_id / rel
        bundle_rel = f"packages/{task_id}/{rel}" if multiple_tasks else rel
        clean_row = sanitize_public_obj(dict(row))
        clean_row["package_relative_path"] = bundle_rel
        if src.exists() and _copy_public_file(src, bundle_root / bundle_rel):
            clean_row.pop("artifact_missing", None)
        else:
            clean_row["artifact_missing"] = True
        rewritten.append(clean_row)
    write_json(bundle_root / "outputs" / "artifacts_index.json", rewritten)


def _copy_tdo_outputs(release_root: Path, bundle_root: Path) -> None:
    package_tdo_dirs = sorted(path for path in (release_root / "packages").glob("*/tdo") if path.is_dir())
    if not package_tdo_dirs:
        return
    if len(package_tdo_dirs) == 1:
        shutil.copytree(package_tdo_dirs[0], bundle_root / "tdo", dirs_exist_ok=True)
        return
    for src in package_tdo_dirs:
        target = bundle_root / "tdo" / src.parent.name
        shutil.copytree(src, target, dirs_exist_ok=True)


def _copy_experience_compiler_outputs(release_root: Path, bundle_root: Path) -> None:
    package_dirs = sorted(path for path in (release_root / "packages").glob("*/experience_compiler") if path.is_dir())
    if not package_dirs:
        return
    if len(package_dirs) == 1:
        shutil.copytree(package_dirs[0], bundle_root / "experience_compiler", dirs_exist_ok=True)
        return
    for src in package_dirs:
        target = bundle_root / "experience_compiler" / src.parent.name
        shutil.copytree(src, target, dirs_exist_ok=True)


def _copy_loop_review_outputs(release_root: Path, bundle_root: Path) -> None:
    package_dirs = sorted(path for path in (release_root / "packages").glob("*/loops") if path.is_dir())
    if not package_dirs:
        return
    if len(package_dirs) == 1:
        shutil.copytree(package_dirs[0], bundle_root / "loops", dirs_exist_ok=True)
        return
    for src in package_dirs:
        target = bundle_root / "loops" / src.parent.name
        shutil.copytree(src, target, dirs_exist_ok=True)


def is_unsafe_bundle_ref(ref: str) -> bool:
    path = Path(ref)
    return path.is_absolute() or ".." in path.parts


def _first_jsonl_row(path: Path) -> dict[str, Any]:
    rows = read_jsonl(path) if path.exists() else []
    for row in rows:
        if isinstance(row, dict):
            return row
    return {}


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:80].strip("-") or "bundle"


def _list_value(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value if v]
    return [str(value)]


def _bundle_metadata_shape(bundle_root: Path, release_root: Path, settings: Settings) -> dict[str, Any]:
    counts = contribution_counts(release_root)
    session = read_json(bundle_root / "session_metadata.json") if (bundle_root / "session_metadata.json").exists() else {}
    task = _first_jsonl_row(release_root / "tasks.jsonl")
    run_status = session.get("run_status") or _run_status(release_root)
    task_status = session.get("task_status") or run_status
    instruction = str(session.get("task_instruction") or "")
    title = (
        task.get("normalized_title")
        or task.get("raw_title")
        or (instruction.strip().splitlines()[0][:90] if instruction.strip() else None)
        or "Agent Apprenticeship session"
    )
    run_id = str(session.get("run_id") or bundle_root.parent.name or _slug(title))
    role = task.get("agent_apprentice_role") or task.get("apprenticeship_role")
    metadata = {
        "bundle_id": f"aa-bundle-{_slug(run_id)}",
        "experience_compilation_id": f"aa-compilation-{_slug(run_id)}",
        "compilation_id": f"aa-compilation-{_slug(run_id)}",
        "schema_version": "aa-experience-compilation-v0.2",
        "package_version": get_package_version(),
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "title": title,
        "task_title": title,
        **counts,
        "domains": _list_value(task.get("domain")),
        "subdomains": _list_value(task.get("subdomain")),
        "domain": (_list_value(task.get("domain")) or ["general"])[0],
        "subdomain": (_list_value(task.get("subdomain")) or ["general"])[0],
        "workflow_family": task.get("task_family") or task.get("workflow_family") or task.get("workflow_type") or "general_workflow",
        "tags": sorted(set([str(task.get("domain") or "general"), str(task.get("subdomain") or "general"), str(task.get("workflow_type") or "workflow")]))[:12],
        "surfaces": [],
        "tools": [],
        "agent_apprentice_role": role,
        "expected_economic_value": task.get("expected_economic_value"),
        "expected_economic_value_for_agent_apprentice": task.get("expected_economic_value_for_agent_apprentice"),
        "apprenticeship_mode": _session_apprenticeship_mode(session, settings),
        "apprenticeship_mode_display": apprenticeship_mode_display(_session_apprenticeship_mode(session, settings)),
        "mentor_mode": _session_mentor_mode(session, settings),
        "public_private_mode": getattr(settings, "contribution_mode", None),
        "sharing_upload_mode": getattr(settings, "contribution_mode", None),
        "private_internal_no_upload_confirmed": getattr(settings, "contribution_mode", None) == "private_internal",
        "task_status": task_status,
        "run_status": run_status,
    }
    return metadata


def _write_manifest(bundle_root: Path, release_root: Path, settings: Settings, include_internal_schema: bool = False) -> None:
    manifest = _bundle_metadata_shape(bundle_root, release_root, settings)
    release_manifest: dict[str, Any] = {}
    if (release_root / "dataset_manifest.json").exists():
        release_manifest = read_json(release_root / "dataset_manifest.json")
    session = read_json(bundle_root / "session_metadata.json") if (bundle_root / "session_metadata.json").exists() else {}
    if session.get("status_reason") and manifest.get("run_status") != "completed":
        manifest["status_reason"] = session["status_reason"]
    if include_internal_schema:
        manifest["source_release_schema_version"] = release_manifest.get("schema_version")
    manifest.update(_tdo_manifest_shape(release_root))
    manifest.update(_loop_manifest_shape(release_root))
    write_json(bundle_root / "contribution_manifest.json", manifest)


def _loop_manifest_shape(release_root: Path) -> dict[str, Any]:
    summary = summarize_loop_reviews(release_root)
    return {
        "has_loop_review_packets": summary.get("has_loop_review_packets"),
        "loop_iteration_count": summary.get("loop_iteration_count"),
        "loop_reviewer_types": summary.get("reviewer_types") or [],
        "final_loop_verdict": summary.get("final_loop_verdict"),
        "loop_review_refs": summary.get("loop_review_refs") or [],
    }


def _tdo_manifest_shape(release_root: Path) -> dict[str, Any]:
    reports = []
    for path in sorted((release_root / "packages").glob("*/tdo/tdo_report.json")):
        try:
            reports.append(read_json(path))
        except Exception:
            continue
    if not reports:
        return {
            "tdo_enabled": False,
            "tdo_status": "not_run",
            "tdo_output_refs": [],
            "experience_compiler_enabled": False,
            "experience_compiler_status": "not_run",
            "experience_compiler_output_refs": [],
        }
    latest = reports[-1]
    output_refs: list[str] = []
    package_tdo_dirs = sorted(path for path in (release_root / "packages").glob("*/tdo") if path.is_dir())
    if len(package_tdo_dirs) == 1:
        for child in sorted(package_tdo_dirs[0].iterdir()):
            if child.is_file():
                output_refs.append(f"tdo/{child.name}")
    else:
        for src in package_tdo_dirs:
            for child in sorted(src.iterdir()):
                if child.is_file():
                    output_refs.append(f"tdo/{src.parent.name}/{child.name}")
    fields = tdo_manifest_fields_from_report(latest)
    fields["tdo_output_refs"] = output_refs
    fields.update(_experience_compiler_manifest_shape(release_root, latest))
    if len(reports) > 1:
        fields["tdo_package_count"] = len(reports)
    return fields


def _experience_compiler_manifest_shape(release_root: Path, latest_report: dict[str, Any]) -> dict[str, Any]:
    compiler_dirs = sorted(path for path in (release_root / "packages").glob("*/experience_compiler") if path.is_dir())
    refs: list[str] = []
    compiler_manifest: dict[str, Any] = {}
    source_evidence_coverage: dict[str, Any] = {}
    quality_report: dict[str, Any] = {}
    if len(compiler_dirs) == 1:
        compiler_manifest = read_json(compiler_dirs[0] / "compiler_manifest.json") if (compiler_dirs[0] / "compiler_manifest.json").exists() else {}
        source_map = read_json(compiler_dirs[0] / "source_evidence_map.json") if (compiler_dirs[0] / "source_evidence_map.json").exists() else {}
        source_evidence_coverage = source_map.get("coverage_statistics") or {}
        quality_report = read_json(compiler_dirs[0] / "quality_report.json") if (compiler_dirs[0] / "quality_report.json").exists() else {}
        for child in sorted(compiler_dirs[0].rglob("*")):
            if child.is_file():
                refs.append(f"experience_compiler/{child.relative_to(compiler_dirs[0]).as_posix()}")
    else:
        for src in compiler_dirs:
            for child in sorted(src.rglob("*")):
                if child.is_file():
                    refs.append(f"experience_compiler/{src.parent.name}/{child.relative_to(src).as_posix()}")
    return {
        "experience_compiler_enabled": bool(compiler_dirs),
        "experience_compiler_status": latest_report.get("tdo_status"),
        "experience_compiler_id": str(latest_report.get("tdo_id") or "").replace("tdo_", "compiler_", 1),
        "experience_compiler_output_refs": refs,
        "experience_compiler_rows_by_type": latest_report.get("rows_accepted_by_type") or {},
        "training_row_counts": latest_report.get("rows_accepted_by_type") or {},
        "training_row_counts_by_file": compiler_manifest.get("training_row_counts_by_file") or quality_report.get("training_row_counts_by_file") or {},
        "source_evidence_map_count": source_evidence_coverage.get("evidence_ref_count"),
        "source_evidence_row_link_count": source_evidence_coverage.get("row_link_count"),
        "experience_compiler_rounds_run": latest_report.get("rounds_run"),
        "experience_compiler_average_quality_score": latest_report.get("average_quality_score"),
        "experience_compiler_average_reuse_value_score": latest_report.get("average_reuse_value_score"),
        "experience_compiler_judge_source": latest_report.get("tdo_judge_source"),
        "compiler_version": compiler_manifest.get("compiler_version"),
        "training_row_schema_version": compiler_manifest.get("training_row_schema_version"),
        "source_evidence_map_schema_version": compiler_manifest.get("source_evidence_map_schema_version"),
    }


def create_contribution_bundle(
    run_root: Path,
    bundle_root: Path | None = None,
    settings: Settings | None = None,
    include_debug: bool = False,
    release_style: bool = False,
) -> Path:
    settings = settings or get_settings()
    bundle_root = bundle_root or (run_root / "contribution_bundle")
    if bundle_root.exists():
        shutil.rmtree(bundle_root)
    bundle_root.mkdir(parents=True, exist_ok=True)

    validate_trace_tree(run_root)
    if release_style:
        create_release(run_root, bundle_root)
        _copy_session_files(run_root, bundle_root, settings)
        _write_manifest(bundle_root, bundle_root, settings, include_internal_schema=True)
        _write_card(bundle_root)
        validate_trace_tree(bundle_root)
        return bundle_root

    with tempfile.TemporaryDirectory(prefix="aa-release-for-bundle-") as tmp:
        release_root = Path(tmp) / "release"
        create_release(run_root, release_root)
        _copy_session_files(run_root, bundle_root, settings)
        _copy_clean_bundle_files(release_root, bundle_root, include_debug=include_debug)
        _write_manifest(bundle_root, release_root, settings, include_internal_schema=include_debug)
        _write_card(bundle_root)

    validate_trace_tree(bundle_root)
    return bundle_root
