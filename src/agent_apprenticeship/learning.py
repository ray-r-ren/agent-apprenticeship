from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import Settings, get_settings
from .env import contains_secret
from .io import append_jsonl, read_workflow_json as read_json, read_workflow_jsonl as read_jsonl, write_json
from .version import get_package_version


PACK_STATUSES = {"draft", "active", "reverted", "removed"}
RESULT_LABELS = {"improved", "no_observed_change", "regressed", "inconclusive"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def slugify(value: str, fallback: str = "experience") -> str:
    chars = []
    for char in value.lower():
        if char.isalnum():
            chars.append(char)
        elif chars and chars[-1] != "-":
            chars.append("-")
    slug = "".join(chars).strip("-")
    return (slug or fallback)[:72]


def repo_root() -> Path:
    configured = os.getenv("AA_ECOSYSTEM_REPO_PATH")
    if configured:
        path = Path(configured).expanduser()
        if (path / "seed_dataset").exists():
            return path
    try:
        settings = get_settings()
        if settings.ecosystem_repo_path and (settings.ecosystem_repo_path / "seed_dataset").exists():
            return settings.ecosystem_repo_path
    except Exception:
        pass
    candidates = [Path.cwd(), Path(__file__).resolve().parents[2]]
    for candidate in candidates:
        if (candidate / "seed_dataset").exists():
            return candidate
    return Path.cwd()


def seed_dataset_root() -> Path:
    return repo_root() / "seed_dataset"


def seed_registry_path() -> Path:
    root = seed_dataset_root()
    if (root / "ecosystem_registry.jsonl").exists():
        return root / "ecosystem_registry.jsonl"
    return root / "ecosystem_registry.json"


def experience_root(settings: Settings | None = None) -> Path:
    s = settings or get_settings()
    root = s.app_home / "experience_packs"
    root.mkdir(parents=True, exist_ok=True)
    return root


def installed_skills_root(settings: Settings | None = None) -> Path:
    s = settings or get_settings()
    root = s.app_home / "installed_skills"
    root.mkdir(parents=True, exist_ok=True)
    return root


def installed_skills_index_path(settings: Settings | None = None) -> Path:
    s = settings or get_settings()
    return s.app_home / "installed_skills_index.jsonl"


def _load_json_or_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    if path.suffix == ".jsonl":
        return [row for row in read_jsonl(path) if isinstance(row, dict)]
    data = read_json(path)
    if isinstance(data, list):
        return [row for row in data if isinstance(row, dict)]
    if isinstance(data, dict):
        entries = data.get("entries") or data.get("bundles") or data.get("items") or data.get("contributions")
        if isinstance(entries, list):
            return [row for row in entries if isinstance(row, dict)]
    return []


def load_seed_registry() -> list[dict[str, Any]]:
    return _load_json_or_jsonl(seed_registry_path())


def load_contribution_registry() -> list[dict[str, Any]]:
    root = repo_root()
    rows: list[dict[str, Any]] = []
    for path in [
        root / "ecosystem" / "contributions" / "index.json",
        root / "ecosystem" / "contributions" / "index.jsonl",
    ]:
        rows.extend(_load_json_or_jsonl(path))
    normalized: list[dict[str, Any]] = []
    for row in rows:
        bundle_id = row.get("bundle_id")
        if not bundle_id:
            continue
        out = dict(row)
        out.setdefault("source_kind", "contribution_bundle")
        out.setdefault("experience_source_type", "contribution_bundle")
        for key in ["local_bundle_path", "bundle_path", "bundle_path_or_url"]:
            value = out.get(key)
            if value:
                candidate = Path(str(value)).expanduser()
                out[key] = str(candidate if candidate.is_absolute() else root / candidate)
        if not any(out.get(key) for key in ["local_bundle_path", "bundle_path", "bundle_path_or_url"]):
            bundle_path = root / "ecosystem" / "contributions" / "bundles" / str(bundle_id)
            if bundle_path.exists():
                out["local_bundle_path"] = str(bundle_path)
        normalized.append(out)
    return normalized


def search_learning_sources(query: str | None = None) -> list[dict[str, Any]]:
    query_text = (query or "").strip().lower()
    rows = load_seed_registry() + load_contribution_registry()
    if not query_text:
        return rows
    terms = [term for term in query_text.split() if term]
    matches: list[dict[str, Any]] = []
    for row in rows:
        haystack = " ".join(
            [
                str(row.get("bundle_id") or ""),
                str(row.get("seed_task_id") or ""),
                str(row.get("task_id") or ""),
                str(row.get("title") or ""),
                " ".join(map(str, row.get("domains") or [])),
                " ".join(map(str, row.get("subdomains") or [])),
                str(row.get("agent_apprentice_role") or ""),
            ]
        ).lower()
        if all(term in haystack for term in terms):
            matches.append(row)
    return matches


def _resolve_relative(path_value: str | None) -> Path | None:
    if not path_value:
        return None
    path = Path(path_value)
    if path.is_absolute():
        return path
    return repo_root() / path


def _source_id(row: dict[str, Any]) -> str:
    return str(row.get("bundle_id") or row.get("seed_task_id") or row.get("task_id") or "source")


def _bundle_source(path: Path) -> dict[str, Any]:
    from .io import validate_trace_tree
    validate_trace_tree(path)
    manifest_path = path / "contribution_manifest.json"
    manifest = read_json(manifest_path) if manifest_path.exists() else {}
    bundle_id = str(manifest.get("bundle_id") or path.name)
    artifacts_index = path / "outputs" / "artifacts_index.json"
    artifact_count = 0
    if artifacts_index.exists():
        try:
            data = read_json(artifacts_index)
            if isinstance(data, list):
                artifact_count = len(data)
        except Exception:
            artifact_count = 0
    return {
        "source_kind": "contribution_bundle",
        "bundle_id": bundle_id,
        "title": manifest.get("title") or bundle_id,
        "domains": manifest.get("domains") or [],
        "subdomains": manifest.get("subdomains") or [],
        "agent_apprentice_role": manifest.get("agent_apprentice_role"),
        "task_status": manifest.get("task_status"),
        "run_status": manifest.get("run_status"),
        "attempt_count": manifest.get("attempts") or 0,
        "trace_count": manifest.get("traced_steps") or 0,
        "artifact_count": artifact_count,
        "bundle_path": str(path),
        "manifest_path": str(manifest_path),
        "artifact_index_path": str(artifacts_index) if artifacts_index.exists() else None,
    }


def resolve_learning_source(source: str, settings: Settings | None = None) -> dict[str, Any]:
    text = source.strip()
    path = Path(text).expanduser()
    if path.exists():
        if path.is_file():
            path = path.parent
        return _bundle_source(path)
    rows = load_seed_registry()
    rows.extend(load_contribution_registry())
    for row in rows:
        keys = {
            str(row.get("bundle_id") or ""),
            str(row.get("seed_task_id") or ""),
            str(row.get("task_id") or ""),
        }
        if text in keys:
            resolved = dict(row)
            resolved["source_kind"] = resolved.get("source_kind") or (
                "seed_task" if resolved.get("seed_task_id") or resolved.get("task_packet_path") else "contribution_bundle"
            )
            if resolved["source_kind"] == "contribution_bundle":
                bundle_path = (
                    resolved.get("local_bundle_path")
                    or resolved.get("bundle_path")
                    or resolved.get("bundle_path_or_url")
                )
                if bundle_path and Path(str(bundle_path)).expanduser().exists():
                    return _bundle_source(Path(str(bundle_path)).expanduser())
            return resolved
    s = settings or get_settings()
    pulled = s.app_home / "ecosystem" / "bundles" / text
    if pulled.exists():
        return _bundle_source(pulled)
    raise FileNotFoundError(f"Ecosystem experience source not found: {source}")


def _safe_read(path: Path | None, limit: int = 12000) -> str:
    if not path or not path.exists() or not path.is_file():
        return ""
    text = path.read_text(errors="replace")
    return text[:limit]


def _read_task_packet(source: dict[str, Any]) -> dict[str, Any]:
    path = _resolve_relative(source.get("task_packet_path"))
    if path and path.exists():
        try:
            data = read_json(path)
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}
    return {}


def _read_learning_lessons(source: dict[str, Any]) -> list[dict[str, Any]]:
    lessons_root = _resolve_relative(source.get("learning_signals_path"))
    if not lessons_root or not lessons_root.exists():
        return []
    lessons: list[dict[str, Any]] = []
    for path in sorted(lessons_root.glob("*.jsonl")):
        lessons.extend([row for row in read_jsonl(path) if isinstance(row, dict)])
    return lessons


def _read_rubric_reminders(source: dict[str, Any]) -> list[str]:
    path = _resolve_relative(source.get("worker_visible_rubric_path"))
    text = _safe_read(path, limit=6000)
    reminders: list[str] = []
    for line in text.splitlines():
        line = line.strip("-*# ").strip()
        if line and len(line) <= 220:
            reminders.append(line)
        if len(reminders) >= 8:
            break
    return reminders


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        clean = str(value).strip()
        if not clean or clean in seen:
            continue
        seen.add(clean)
        out.append(clean)
    return out


def _extend_from_lessons(lessons: list[dict[str, Any]], key: str) -> list[str]:
    values: list[str] = []
    for lesson in lessons:
        item = lesson.get(key)
        if isinstance(item, list):
            values.extend([str(v) for v in item])
        elif item:
            values.append(str(item))
    return values


def _trace_refs(source: dict[str, Any]) -> list[dict[str, Any]]:
    refs = []
    for path_value in source.get("trace_paths") or []:
        path = _resolve_relative(str(path_value))
        refs.append({"path": str(path_value), "exists": bool(path and path.exists())})
    return refs


def _source_ref(source: dict[str, Any]) -> dict[str, Any]:
    return {
        "source_id": _source_id(source),
        "experience_source_type": source.get("source_kind") or "seed_task",
        "title": source.get("title"),
        "domains": source.get("domains") or [],
        "subdomains": source.get("subdomains") or [],
        "task_packet_path": source.get("task_packet_path"),
        "worker_visible_rubric_path": source.get("worker_visible_rubric_path"),
        "learning_signals_path": source.get("learning_signals_path"),
        "trace_refs": _trace_refs(source),
        "bundle_path": source.get("bundle_path"),
    }


def compile_experience_pack(
    sources: list[dict[str, Any]],
    *,
    title: str | None = None,
    settings: Settings | None = None,
) -> Path:
    if not sources:
        raise ValueError("At least one ecosystem experience source is required.")
    s = settings or get_settings()
    source_ids = [_source_id(source) for source in sources]
    first_title = str(sources[0].get("title") or source_ids[0])
    pack_title = title or (first_title if len(sources) == 1 else f"{first_title} + {len(sources) - 1} more")
    digest = hashlib.sha256("|".join(source_ids + [pack_title]).encode()).hexdigest()[:10]
    pack_id = f"aa-exp-{slugify(pack_title)}-{digest}"
    pack_path = experience_root(s) / pack_id
    suffix = 2
    while pack_path.exists():
        pack_path = experience_root(s) / f"{pack_id}-{suffix}"
        suffix += 1
    pack_id = pack_path.name
    domains: list[str] = []
    subdomains: list[str] = []
    task_patterns: list[str] = []
    artifact_requirements: list[str] = []
    rubric_reminders: list[str] = []
    common_failure_modes: list[str] = []
    strategy_lessons: list[str] = []
    revision_lessons: list[str] = []
    verifier_lessons: list[str] = []
    good_bad_signals: list[str] = []
    original_rubric_refs: list[str] = []
    original_learning_signal_refs: list[str] = []
    source_refs: list[dict[str, Any]] = []
    for source in sources:
        domains.extend(map(str, source.get("domains") or []))
        subdomains.extend(map(str, source.get("subdomains") or []))
        packet = _read_task_packet(source)
        if packet.get("task_family"):
            task_patterns.append(str(packet["task_family"]))
        if packet.get("expected_deliverable"):
            artifact_requirements.append(str(packet["expected_deliverable"]))
        artifact_requirements.extend(map(str, packet.get("deliverables") or []))
        lessons = _read_learning_lessons(source)
        artifact_requirements.extend(_extend_from_lessons(lessons, "artifact_requirements"))
        common_failure_modes.extend(_extend_from_lessons(lessons, "common_failure_modes"))
        strategy_lessons.extend(_extend_from_lessons(lessons, "strategy_lessons"))
        revision_lessons.extend(_extend_from_lessons(lessons, "revision_lessons"))
        verifier_lessons.extend(_extend_from_lessons(lessons, "verifier_feedback_summary"))
        for lesson in lessons:
            if lesson.get("lesson_summary"):
                good_bad_signals.append(str(lesson["lesson_summary"]))
            if lesson.get("rubric_reminders"):
                item = lesson["rubric_reminders"]
                if isinstance(item, list):
                    rubric_reminders.extend(map(str, item))
        rubric_reminders.extend(_read_rubric_reminders(source))
        if source.get("worker_visible_rubric_path"):
            original_rubric_refs.append(str(source["worker_visible_rubric_path"]))
        if source.get("learning_signals_path"):
            original_learning_signal_refs.append(str(source["learning_signals_path"]))
        source_refs.append(_source_ref(source))
    domains = _dedupe(domains)
    subdomains = _dedupe(subdomains)
    task_patterns = _dedupe(task_patterns)
    artifact_requirements = _dedupe(artifact_requirements)[:12]
    rubric_reminders = _dedupe(rubric_reminders)[:12]
    common_failure_modes = _dedupe(common_failure_modes)[:12]
    strategy_lessons = _dedupe(strategy_lessons)[:12]
    revision_lessons = _dedupe(revision_lessons)[:12]
    verifier_lessons = _dedupe(verifier_lessons)[:12]
    good_bad_signals = _dedupe(good_bad_signals)[:12]
    pack = {
        "pack_id": pack_id,
        "title": pack_title,
        "source_ecosystem_ids": source_ids,
        "experience_source_types": _dedupe([str(source.get("source_kind") or "seed_task") for source in sources]),
        "domains": domains,
        "subdomains": subdomains,
        "task_patterns": task_patterns,
        "artifact_requirements": artifact_requirements,
        "rubric_reminders": rubric_reminders,
        "common_failure_modes": common_failure_modes,
        "strategy_lessons": strategy_lessons,
        "revision_lessons": revision_lessons,
        "verifier_evaluator_lessons": verifier_lessons,
        "examples_of_good_bad_attempt_signals": good_bad_signals,
        "original_trace_refs": [ref for source in sources for ref in _trace_refs(source)],
        "original_rubric_refs": original_rubric_refs,
        "original_learning_signal_refs": original_learning_signal_refs,
        "created_at": utc_now(),
        "status": "draft",
    }
    skill = render_skill_markdown(pack)
    summary = render_pack_summary(pack)
    for name, text in {"skill.md": skill, "SUMMARY.md": summary}.items():
        if contains_secret(text):
            raise ValueError(f"Experience Pack blocked because {name} appears to contain a secret.")
    pack_path.mkdir(parents=True)
    write_json(pack_path / "experience_pack.json", pack)
    write_json(pack_path / "source_refs.json", {"sources": source_refs})
    (pack_path / "skill.md").write_text(skill)
    (pack_path / "SUMMARY.md").write_text(summary)
    return pack_path


def render_skill_markdown(pack: dict[str, Any]) -> str:
    def bullets(key: str, fallback: str = "None captured.") -> str:
        values = pack.get(key) or []
        if not values:
            return f"- {fallback}"
        return "\n".join(f"- {value}" for value in values[:8])

    return (
        f"# Experience Pack: {pack.get('title')}\n\n"
        "Use these concise lessons when working on similar tasks. Do not treat this as hidden answer key content.\n\n"
        "## Key Lessons\n"
        f"{bullets('strategy_lessons')}\n\n"
        "## Artifact Requirements\n"
        f"{bullets('artifact_requirements')}\n\n"
        "## Rubric Reminders\n"
        f"{bullets('rubric_reminders')}\n\n"
        "## Common Failure Modes To Avoid\n"
        f"{bullets('common_failure_modes')}\n\n"
        "## Revision Lessons\n"
        f"{bullets('revision_lessons')}\n\n"
        "## Source Refs\n"
        + "\n".join(f"- {source_id}" for source_id in pack.get("source_ecosystem_ids", [])[:12])
        + "\n"
    )


def render_pack_summary(pack: dict[str, Any]) -> str:
    return (
        f"# {pack.get('title')}\n\n"
        f"Pack ID: {pack.get('pack_id')}\n\n"
        f"Status: {pack.get('status')}\n\n"
        f"Sources: {', '.join(pack.get('source_ecosystem_ids') or [])}\n\n"
        f"Domains: {', '.join(pack.get('domains') or []) or 'not specified'}\n\n"
        "This Experience Pack contains compact Apprentice Agent guidance plus source references for inspection.\n"
    )


def load_pack(pack_id_or_path: str, settings: Settings | None = None) -> tuple[Path, dict[str, Any]]:
    s = settings or get_settings()
    candidate = Path(pack_id_or_path).expanduser()
    if not candidate.exists():
        candidate = experience_root(s) / pack_id_or_path
    if not candidate.exists() and (experience_root(s) / "_removed" / pack_id_or_path).exists():
        candidate = experience_root(s) / "_removed" / pack_id_or_path
    data_path = candidate / "experience_pack.json"
    if not data_path.exists():
        raise FileNotFoundError(f"Experience Pack not found: {pack_id_or_path}")
    data = read_json(data_path)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid Experience Pack: {pack_id_or_path}")
    return candidate, data


def list_packs(settings: Settings | None = None, include_removed: bool = False) -> list[dict[str, Any]]:
    root = experience_root(settings)
    paths = [path for path in root.iterdir() if path.is_dir() and path.name != "_removed"]
    if include_removed and (root / "_removed").exists():
        paths.extend([path for path in (root / "_removed").iterdir() if path.is_dir()])
    packs = []
    for path in sorted(paths):
        data_path = path / "experience_pack.json"
        if data_path.exists():
            try:
                data = read_json(data_path)
                if isinstance(data, dict):
                    data = dict(data)
                    data["path"] = str(path)
                    packs.append(data)
            except Exception:
                continue
    return packs


def update_pack_status(pack_id: str, status: str, settings: Settings | None = None) -> Path:
    if status not in PACK_STATUSES:
        raise ValueError(f"Unsupported Experience Pack status: {status}")
    path, data = load_pack(pack_id, settings)
    data["status"] = status
    data["updated_at"] = utc_now()
    write_json(path / "experience_pack.json", data)
    (path / "SUMMARY.md").write_text(render_pack_summary(data))
    return path


def remove_pack(pack_id: str, settings: Settings | None = None) -> Path:
    path, data = load_pack(pack_id, settings)
    data["status"] = "removed"
    data["updated_at"] = utc_now()
    write_json(path / "experience_pack.json", data)
    removed_root = experience_root(settings) / "_removed"
    removed_root.mkdir(exist_ok=True)
    target = removed_root / path.name
    if target.exists():
        shutil.rmtree(target)
    shutil.move(str(path), str(target))
    return target


def active_packs(settings: Settings | None = None) -> list[dict[str, Any]]:
    return [pack for pack in list_packs(settings) if pack.get("status") == "active"]


def pack_prompt_section(pack: dict[str, Any]) -> str:
    skill = render_skill_markdown(pack)
    lines = skill.splitlines()
    trimmed = "\n".join(lines[:80])
    return (
        "\n\n---\n"
        f"Experience Pack: {pack.get('title')}\n"
        "Apply these lessons when useful. Keep the task's original requirements authoritative.\n\n"
        f"{trimmed}\n"
        "---\n"
    )


def resolve_packs_for_run(
    pack_ids: list[str],
    *,
    use_active: bool = False,
    no_packs: bool = False,
    settings: Settings | None = None,
) -> tuple[list[dict[str, Any]], str]:
    if no_packs:
        return [], ""
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    s = settings or get_settings()
    for pack_id in pack_ids:
        path, pack = load_pack(pack_id, s)
        status = pack.get("status")
        if status in {"reverted", "removed"}:
            continue
        pack = dict(pack)
        pack["path"] = str(path)
        if pack.get("pack_id") not in seen:
            seen.add(str(pack.get("pack_id")))
            selected.append(pack)
    if use_active:
        for pack in active_packs(s):
            if pack.get("pack_id") not in seen:
                seen.add(str(pack.get("pack_id")))
                selected.append(pack)
    guidance = "".join(pack_prompt_section(pack) for pack in selected)
    return selected, guidance


def pack_run_refs(packs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    refs = []
    for pack in packs:
        refs.append(
            {
                "pack_id": pack.get("pack_id"),
                "title": pack.get("title"),
                "source_refs": pack.get("source_ecosystem_ids") or [],
                "status": pack.get("status"),
            }
        )
    return refs


def replay_instruction_for_pack(pack: dict[str, Any], settings: Settings | None = None) -> str:
    source_ids = pack.get("source_ecosystem_ids") or []
    if source_ids:
        try:
            source = resolve_learning_source(str(source_ids[0]), settings)
            packet = _read_task_packet(source)
            instruction = packet.get("instruction") or packet.get("task_instruction") or packet.get("normalized_instruction")
            if instruction:
                deliverables = packet.get("deliverables") or []
                extra = f"\n\nRequired deliverables: {', '.join(map(str, deliverables))}" if deliverables else ""
                return str(instruction) + extra
        except Exception:
            pass
    artifacts = ", ".join(pack.get("artifact_requirements") or []) or "a concise artifact summary"
    return f"Complete a small task inspired by {pack.get('title')}. Produce {artifacts}."


def compare_replay(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    before_status = before.get("task_status") or before.get("run_status")
    after_status = after.get("task_status") or after.get("run_status")
    before_artifacts = int(before.get("artifact_count") or 0)
    after_artifacts = int(after.get("artifact_count") or 0)
    label = "inconclusive"
    if before_status != "completed" and after_status == "completed":
        label = "improved"
    elif before_status == "completed" and after_status != "completed":
        label = "regressed"
    elif before_status == after_status == "completed":
        if after_artifacts > before_artifacts:
            label = "improved"
        elif after_artifacts < before_artifacts:
            label = "regressed"
        else:
            label = "no_observed_change"
    elif before_status == after_status:
        label = "no_observed_change"
    return {
        "result": label,
        "before_status": before_status,
        "after_status": after_status,
        "before_artifact_count": before_artifacts,
        "after_artifact_count": after_artifacts,
        "score_available": False,
        "score_note": "No comparable evaluator score was available; comparison used status and artifact completeness.",
    }


def _find_bundle_root(path: Path) -> Path:
    path = path.expanduser()
    if path.is_file():
        path = path.parent
    candidates = [
        path,
        path / "contribution_bundle",
        path / "package",
    ]
    for candidate in candidates:
        if (candidate / "contribution_manifest.json").exists() or (candidate / "experience_compiler").exists():
            return candidate
    for candidate in path.rglob("contribution_manifest.json"):
        return candidate.parent
    raise FileNotFoundError(f"Experience Compilation not found: {path}")


def _safe_copy_file(src: Path, dst: Path, files: list[str], root: Path) -> None:
    if not src.exists() or not src.is_file():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    files.append(dst.relative_to(root).as_posix())


def _safe_copytree(src: Path, dst: Path, files: list[str], root: Path) -> None:
    if not src.exists() or not src.is_dir():
        return
    shutil.copytree(src, dst, dirs_exist_ok=True)
    for file_path in sorted(dst.rglob("*")):
        if file_path.is_file():
            files.append(file_path.relative_to(root).as_posix())


def _read_manifest_field(bundle: Path, key: str, default: Any = None) -> Any:
    for rel in ["contribution_manifest.json", "experience_compiler/compiler_manifest.json"]:
        path = bundle / rel
        if path.exists():
            try:
                data = read_json(path)
            except Exception:
                continue
            if isinstance(data, dict) and data.get(key) is not None:
                return data.get(key)
    return default


def install_runtime_training_from_bundle(
    source_item_id: str,
    bundle_path: Path,
    *,
    settings: Settings | None = None,
    agent_target: str = "current",
    dry_run: bool = False,
) -> dict[str, Any]:
    """Install runtime-useful Experience Compiler outputs into the local registry."""
    s = settings or get_settings()
    bundle = _find_bundle_root(bundle_path)
    from .io import validate_trace_tree
    validate_trace_tree(bundle)
    compiler = bundle / "experience_compiler"
    skill_dir = compiler / "skill_pack"
    runtime_dir = compiler / "runtime_learning_package"
    skill_md = skill_dir / "skill.md"
    if not skill_md.exists():
        raise FileNotFoundError(f"Runtime training skill.md not found in {bundle}")
    quality = _runtime_training_quality(bundle)
    package_id = str(_read_manifest_field(bundle, "bundle_id") or _read_manifest_field(bundle, "source_package_id") or bundle.name)
    title = str(_read_manifest_field(bundle, "title") or _read_manifest_field(bundle, "source_task_id") or source_item_id)
    digest = hashlib.sha256(f"{source_item_id}|{package_id}|{skill_md.read_text(errors='replace')[:5000]}".encode()).hexdigest()[:10]
    installed_skill_id = f"aa-skill-{slugify(title, 'runtime-training')}-{digest}"
    target = installed_skills_root(s) / installed_skill_id
    files: list[str] = []
    if dry_run:
        return {
            "installed_skill_id": installed_skill_id,
            "source_item_id": source_item_id,
            "source_package_id": package_id,
            "dry_run": True,
            "would_install": [
                "skill.md",
                "canonical_skill.json",
                "retrieval_card.json",
                "runtime_learning_package/",
                "install_manifest.json",
            ],
            "target": str(target),
        }
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)
    _safe_copy_file(skill_md, target / "skill.md", files, target)
    _safe_copytree(runtime_dir, target / "runtime_learning_package", files, target)
    for src_name, dst_name in [
        ("canonical_skill.json", "canonical_skill.json"),
        ("retrieval_card.json", "retrieval_card.json"),
        ("skill_card.json", "skill_card.json"),
        ("verifier_spec.json", "verifier_spec.json"),
        ("verifier_checklist.json", "verifier_checklist.json"),
        ("tool_recipe.json", "tool_recipe.json"),
    ]:
        _safe_copy_file(skill_dir / src_name, target / dst_name, files, target)
    if not (target / "canonical_skill.json").exists():
        write_json(
            target / "canonical_skill.json",
            {
                "skill_id": installed_skill_id,
                "title": title,
                "source_item_id": source_item_id,
                "source_package_id": package_id,
                "skill_ref": "skill.md",
            },
        )
        files.append("canonical_skill.json")
    if not (target / "retrieval_card.json").exists():
        write_json(
            target / "retrieval_card.json",
            {
                "skill_id": installed_skill_id,
                "title": title,
                "domains": _read_manifest_field(bundle, "domains", []) or [],
                "source_item_id": source_item_id,
                "summary": f"Runtime training compiled from {title}.",
            },
        )
        files.append("retrieval_card.json")
    manifest = {
        "installed_skill_id": installed_skill_id,
        "source_item_id": source_item_id,
        "source_package_id": package_id,
        "source_url": _read_manifest_field(bundle, "source_url"),
        "package_version": get_package_version(),
        "agent_target": agent_target,
        "install_mode": "agent_apprenticeship_runtime",
        "installed_at": utc_now(),
        "enabled": True,
        "title": title,
        "files_installed": sorted(set(files + ["install_manifest.json"])),
        "backup_refs": [],
        "reversible": True,
        "generation_notes": ["native_agent_files_unchanged"],
        "transfer_confidence": quality["transfer_confidence"],
        "runtime_training_confidence": quality["transfer_confidence"],
        "low_confidence_reasons": [quality["low_confidence_reason"]] if quality.get("low_confidence_reason") else [],
        "source_quality_gate": quality,
    }
    write_json(target / "install_manifest.json", manifest)
    append_jsonl(installed_skills_index_path(s), manifest)
    return manifest


def _runtime_training_quality(bundle: Path) -> dict[str, Any]:
    runtime = bundle / "experience_compiler" / "runtime_learning_package"
    compiler = bundle / "experience_compiler"
    has_actual = any((bundle / "attempts").glob("*/actual_outputs.json")) if (bundle / "attempts").exists() else False
    has_artifacts = any((bundle / "attempts").glob("*/artifacts/*")) if (bundle / "attempts").exists() else False
    required = {
        "transfer_runbook": runtime / "transfer_runbook.md",
        "minimal_prompt_context": runtime / "minimal_prompt_context.md",
        "output_contract": runtime / "output_contract.json",
        "artifact_contract": runtime / "artifact_contract.json",
        "prefinal_verification_checklist": runtime / "prefinal_verification_checklist.json",
    }
    missing = [name for name, path in required.items() if not path.exists()]
    evidence_map = compiler / "source_evidence_map.json"
    source_score = _runtime_training_source_score(bundle)
    checks = {
        "actual_outputs_exist": has_actual,
        "artifacts_exist": has_artifacts,
        "source_evidence_map_exists": evidence_map.exists(),
        "runtime_package_non_generic": bool((runtime / "transfer_runbook.md").exists() and len((runtime / "transfer_runbook.md").read_text(errors="replace")) > 500),
        "required_runtime_contract_files_present": not missing,
        "missing_runtime_contract_files": missing,
        "source_score": source_score,
        "source_score_passes_transfer_gate": source_score is None or source_score >= 0.7,
    }
    confidence = "high" if all(v for k, v in checks.items() if k != "missing_runtime_contract_files") else "low"
    checks["transfer_confidence"] = confidence
    reasons = list(missing)
    if source_score is not None and source_score < 0.7:
        reasons.append(f"source_score_below_transfer_gate:{source_score:.3f}")
    checks["low_confidence_reason"] = "; ".join(reasons) if reasons else None
    return checks


def _runtime_training_source_score(bundle: Path) -> float | None:
    for rel in [
        "grading/baseline_grader_result.json",
        "evaluation/grader_result.json",
        "evaluation/grader_results.json",
    ]:
        path = bundle / rel
        if not path.exists():
            continue
        try:
            payload = read_json(path)
        except Exception:
            continue
        if isinstance(payload, dict):
            value = payload.get("score")
            try:
                return float(value)
            except Exception:
                pass
    jsonl = bundle / "evaluation/grader_results.jsonl"
    if jsonl.exists():
        for row in read_jsonl(jsonl):
            if isinstance(row, dict) and row.get("score") is not None:
                try:
                    return float(row.get("score"))
                except Exception:
                    continue
    return None


def list_installed_skills(settings: Settings | None = None, include_disabled: bool = True) -> list[dict[str, Any]]:
    root = installed_skills_root(settings)
    rows: list[dict[str, Any]] = []
    for path in sorted(root.iterdir()) if root.exists() else []:
        manifest = path / "install_manifest.json"
        if not manifest.exists():
            continue
        try:
            data = read_json(manifest)
        except Exception:
            continue
        if not include_disabled and not data.get("enabled", True):
            continue
        data = dict(data)
        data["path"] = str(path)
        rows.append(data)
    return rows


def _installed_skill_path(installed_skill_id: str, settings: Settings | None = None) -> Path:
    path = installed_skills_root(settings) / installed_skill_id
    if not (path / "install_manifest.json").exists():
        raise FileNotFoundError(f"Installed runtime training not found: {installed_skill_id}")
    return path


def set_installed_skill_enabled(installed_skill_id: str, enabled: bool, settings: Settings | None = None) -> dict[str, Any]:
    path = _installed_skill_path(installed_skill_id, settings)
    manifest = read_json(path / "install_manifest.json")
    manifest["enabled"] = bool(enabled)
    manifest["updated_at"] = utc_now()
    write_json(path / "install_manifest.json", manifest)
    append_jsonl(installed_skills_index_path(settings), manifest)
    return manifest


def uninstall_installed_skill(installed_skill_id: str, settings: Settings | None = None) -> Path:
    root = installed_skills_root(settings)
    path = _installed_skill_path(installed_skill_id, settings)
    manifest = read_json(path / "install_manifest.json")
    manifest["enabled"] = False
    manifest["uninstalled_at"] = utc_now()
    removed = root / "_removed" / installed_skill_id
    removed.parent.mkdir(parents=True, exist_ok=True)
    if removed.exists():
        shutil.rmtree(removed)
    write_json(path / "install_manifest.json", manifest)
    shutil.move(str(path), str(removed))
    append_jsonl(installed_skills_index_path(settings), manifest)
    return removed


def _skill_keywords(skill: dict[str, Any]) -> set[str]:
    text = " ".join(
        str(skill.get(key) or "")
        for key in ["installed_skill_id", "title", "source_item_id", "source_package_id"]
    )
    path = Path(str(skill.get("path") or ""))
    for rel in ["retrieval_card.json", "canonical_skill.json", "skill.md"]:
        file_path = path / rel
        if file_path.exists():
            text += " " + file_path.read_text(errors="ignore")[:3000]
    return {token.lower() for token in re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}", text)}


def select_runtime_training_for_run(
    instruction: str,
    *,
    settings: Settings | None = None,
    force: bool = False,
    limit: int = 2,
) -> tuple[list[dict[str, Any]], str]:
    enabled = list_installed_skills(settings, include_disabled=False)
    if not enabled:
        return [], ""
    query_terms = {token.lower() for token in re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}", instruction)}
    scored: list[tuple[int, dict[str, Any], str]] = []
    for skill in enabled:
        overlap = len(query_terms & _skill_keywords(skill))
        if force or overlap:
            confidence = skill.get("transfer_confidence") or (skill.get("source_quality_gate") or {}).get("transfer_confidence") or "unknown"
            source_gate = skill.get("source_quality_gate") if isinstance(skill.get("source_quality_gate"), dict) else {}
            source_score = source_gate.get("source_score")
            quality_penalty = 0
            reason_bits = ["matched runtime training against the task request"]
            if confidence == "low":
                quality_penalty -= 50
                reason_bits.append("low-confidence source; use cautiously")
            if isinstance(source_score, (int, float)) and float(source_score) < 0.7:
                quality_penalty -= 50
                reason_bits.append(f"source score {float(source_score):.2f} below transfer gate")
            if force:
                reason_bits.append("explicit --use-installed-skills requested")
            scored.append((overlap + quality_penalty, skill, "; ".join(reason_bits)))
    scored.sort(key=lambda item: item[0], reverse=True)
    selected = [dict(skill, runtime_training_selection_reason=reason) for _, skill, reason in scored[:limit]]
    if not selected:
        return [], ""
    sections = ["\n\n---\nInstalled Runtime Training\nUse these enabled skills only when they are relevant. The task instruction remains authoritative.\n"]
    for idx, skill in enumerate(selected):
        path = Path(str(skill.get("path")))
        runtime_text, runtime_meta = _runtime_prompt_context(path)
        skill = dict(skill)
        skill.update(runtime_meta)
        if runtime_text:
            skill_text = runtime_text
        else:
            skill_text = (path / "skill.md").read_text(errors="replace") if (path / "skill.md").exists() else ""
        sections.append(
            f"\n## {skill.get('title') or skill.get('installed_skill_id')}\n"
            f"installed_skill_id: {skill.get('installed_skill_id')}\n"
            f"selection_reason: {skill.get('runtime_training_selection_reason') or 'matched runtime training against the task request'}.\n\n"
            f"{skill_text[:5000]}\n"
        )
        selected[idx] = skill
    sections.append("---\n")
    return selected, "\n".join(sections)


def installed_skill_run_refs(skills: list[dict[str, Any]]) -> list[dict[str, Any]]:
    refs: list[dict[str, Any]] = []
    for skill in skills:
        refs.append(
            {
                "installed_skill_id": skill.get("installed_skill_id"),
                "title": skill.get("title"),
                "source_item_id": skill.get("source_item_id"),
                "source_package_id": skill.get("source_package_id"),
                "runtime_training_injected_sections": skill.get("runtime_training_injected_sections") or [],
                "runtime_training_context_char_count": skill.get("runtime_training_context_char_count") or 0,
                "runtime_training_content_hash": skill.get("runtime_training_content_hash"),
                "runtime_training_selection_reason": skill.get("runtime_training_selection_reason"),
                "transfer_confidence": skill.get("transfer_confidence"),
                "runtime_training_confidence": skill.get("runtime_training_confidence") or skill.get("transfer_confidence"),
                "low_confidence_reasons": skill.get("low_confidence_reasons") or [],
                "source_quality_gate": skill.get("source_quality_gate"),
                "selected_transfer_runbook_path": skill.get("selected_transfer_runbook_path"),
                "selected_transfer_checklist_path": skill.get("selected_transfer_checklist_path"),
                "injected_minimal_prompt_context": bool(skill.get("injected_minimal_prompt_context")),
                "injected_output_contract": bool(skill.get("injected_output_contract")),
                "injected_artifact_contract": bool(skill.get("injected_artifact_contract")),
                "injected_prefinal_checklist": bool(skill.get("injected_prefinal_checklist")),
                "prefinal_verification_passed": skill.get("prefinal_verification_passed"),
                "repair_loop_count": skill.get("repair_loop_count", 0),
                "contract_failures_json": skill.get("contract_failures_json") or [],
                "runtime_training_injection_mode": skill.get("runtime_training_injection_mode"),
            }
        )
    return refs


def _jsonl_preview(path: Path, limit: int = 8) -> str:
    if not path.exists() or not path.is_file():
        return ""
    lines: list[str] = []
    for line in path.read_text(errors="replace").splitlines():
        line = line.strip()
        if line:
            lines.append(line)
        if len(lines) >= limit:
            break
    return "\n".join(lines)


def _runtime_prompt_context(skill_root: Path) -> tuple[str, dict[str, Any]]:
    """Return compact runtime-training context that is actually injected."""
    runtime = skill_root / "runtime_learning_package"
    injection_mode = (os.getenv("AA_RUNTIME_TRAINING_INJECTION_MODE") or "contract").strip().lower()
    if injection_mode == "full":
        ordered_files = [
            ("minimal_prompt_context.md", "minimal_prompt_context"),
            ("transfer_runbook.md", "transfer_runbook"),
            ("transfer_checklist.json", "transfer_checklist"),
            ("tool_execution_recipe.json", "tool_execution_recipe"),
            ("output_verification_steps.jsonl", "output_verification_steps"),
            ("common_errors_to_avoid.jsonl", "common_errors_to_avoid"),
            ("runtime_instructions.md", "runtime_instructions"),
        ]
    else:
        ordered_files = [
            ("minimal_prompt_context.md", "minimal_prompt_context"),
            ("output_contract.json", "output_contract"),
            ("artifact_contract.json", "artifact_contract"),
            ("prefinal_verification_checklist.json", "prefinal_verification_checklist"),
        ]
    parts: list[str] = []
    injected_sections: list[str] = []
    selected_runbook = None
    selected_checklist = None
    for rel, section in ordered_files:
        path = runtime / rel
        if not path.exists() or not path.is_file():
            continue
        if rel.endswith(".jsonl"):
            text = _jsonl_preview(path)
        elif rel.endswith(".json"):
            try:
                payload = read_json(path)
                text = json.dumps(payload, indent=2, sort_keys=True)
            except Exception:
                text = path.read_text(errors="replace")
        else:
            text = path.read_text(errors="replace")
        text = text.strip()
        if not text:
            continue
        if rel == "transfer_runbook.md":
            selected_runbook = f"runtime_learning_package/{rel}"
        if rel == "transfer_checklist.json":
            selected_checklist = f"runtime_learning_package/{rel}"
        injected_sections.append(section)
        section_limit = 2200 if injection_mode == "full" else 950
        parts.append(f"### {section.replace('_', ' ').title()}\n{text[:section_limit]}")
    if selected_runbook is None and (runtime / "transfer_runbook.md").exists():
        selected_runbook = "runtime_learning_package/transfer_runbook.md"
    if selected_checklist is None:
        if (runtime / "prefinal_verification_checklist.json").exists():
            selected_checklist = "runtime_learning_package/prefinal_verification_checklist.json"
        elif (runtime / "transfer_checklist.json").exists():
            selected_checklist = "runtime_learning_package/transfer_checklist.json"
    combined = "\n\n".join(parts)
    digest = hashlib.sha256(combined.encode()).hexdigest()[:16] if combined else None
    return combined, {
        "runtime_training_injected_sections": injected_sections,
        "runtime_training_context_char_count": len(combined),
        "runtime_training_content_hash": digest,
        "selected_transfer_runbook_path": selected_runbook,
        "selected_transfer_checklist_path": selected_checklist,
        "injected_minimal_prompt_context": "minimal_prompt_context" in injected_sections,
        "injected_output_contract": "output_contract" in injected_sections,
        "injected_artifact_contract": "artifact_contract" in injected_sections,
        "injected_prefinal_checklist": "prefinal_verification_checklist" in injected_sections,
        "prefinal_verification_passed": None,
        "repair_loop_count": 0,
        "contract_failures_json": [],
        "runtime_training_injection_mode": injection_mode,
    }


def write_before_after_result(pack_path: Path, result: dict[str, Any]) -> Path:
    result = dict(result)
    result["created_at"] = utc_now()
    write_json(pack_path / "before_after_result.json", result)
    return pack_path / "before_after_result.json"
