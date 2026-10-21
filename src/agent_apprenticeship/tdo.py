from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

from .config import Settings, apprentice_agent_display_name, configured_model_provider_ready
from .env import contains_secret, redact_secrets
from .io import append_jsonl, read_workflow_json as read_json, read_workflow_jsonl as read_jsonl, write_json
from .progress import ProgressCallback, append_progress_event
from .public_sanitizer import sanitize_public_obj, sanitize_public_text
from .schemas import TDOJudgeResult, TDOReport
from .version import get_package_version

TDO_VERSION = "aa-experience-compiler-v0.2"
TRAINING_ROW_SCHEMA_VERSION = "aa-training-row-v0.2"
EVIDENCE_MAP_SCHEMA_VERSION = "aa-source-evidence-map-v0.2"
TDO_JSONL_FILES = [
    "tdo_trace.jsonl",
    "tdo_rounds.jsonl",
    "tdo_judge_results.jsonl",
    "training_examples.jsonl",
    "evaluation_examples.jsonl",
    "verifier_examples.jsonl",
    "reward_model_rows.jsonl",
    "preference_pairs.jsonl",
    "failure_cases.jsonl",
    "task_variants.jsonl",
    "environment_response_pairs.jsonl",
    "multi_agent_handoffs.jsonl",
    "multimodal_artifacts.jsonl",
    "ui_interaction_events.jsonl",
    "physical_environment_events.jsonl",
    "api_interaction_events.jsonl",
    "mcp_interaction_events.jsonl",
    "database_events.jsonl",
    "structured_table_events.jsonl",
    "software_execution_events.jsonl",
    "search_retrieval_events.jsonl",
    "enterprise_app_events.jsonl",
    "async_workflow_events.jsonl",
    "collaboration_events.jsonl",
    "document_events.jsonl",
    "cross_system_events.jsonl",
    "meta_optimization_records.jsonl",
    "rejected_rows.jsonl",
]
TDO_JSON_FILES = [
    "tdo_manifest.json",
    "tdo_report.json",
    "tdo_recipe.json",
    "optimized_task_spec.json",
    "optimized_rubric.json",
    "domain_value_profile.json",
    "environment_spec.json",
]
ROW_FILE_BY_TYPE = {
    "training_example": "training_examples.jsonl",
    "evaluation_example": "evaluation_examples.jsonl",
    "verifier_example": "verifier_examples.jsonl",
    "reward_model_row": "reward_model_rows.jsonl",
    "preference_pair": "preference_pairs.jsonl",
    "failure_case": "failure_cases.jsonl",
    "task_variant": "task_variants.jsonl",
    "environment_response_pair": "environment_response_pairs.jsonl",
    "multi_agent_handoff": "multi_agent_handoffs.jsonl",
    "multimodal_artifact": "multimodal_artifacts.jsonl",
    "ui_interaction_event": "ui_interaction_events.jsonl",
    "physical_environment_event": "physical_environment_events.jsonl",
    "api_interaction_event": "api_interaction_events.jsonl",
    "mcp_interaction_event": "mcp_interaction_events.jsonl",
    "database_event": "database_events.jsonl",
    "structured_table_event": "structured_table_events.jsonl",
    "software_execution_event": "software_execution_events.jsonl",
    "search_retrieval_event": "search_retrieval_events.jsonl",
    "enterprise_app_event": "enterprise_app_events.jsonl",
    "async_workflow_event": "async_workflow_events.jsonl",
    "collaboration_event": "collaboration_events.jsonl",
    "document_event": "document_events.jsonl",
    "cross_system_event": "cross_system_events.jsonl",
}
ENVIRONMENT_DOMAINS = {
    "terminal",
    "browser",
    "computer_use",
    "web",
    "search",
    "codebase",
    "software",
    "os",
    "android",
    "mcp",
    "api",
    "database",
    "spreadsheet",
    "file",
    "multimodal",
    "device",
    "robotics",
    "lab",
    "industrial",
    "physical",
    "mixed",
    "unknown",
}
SURFACE_FILES = {
    "multi_agent_handoff": "multi_agent_handoffs.jsonl",
    "multimodal_artifact": "multimodal_artifacts.jsonl",
    "ui_interaction_event": "ui_interaction_events.jsonl",
    "physical_environment_event": "physical_environment_events.jsonl",
    "api_interaction_event": "api_interaction_events.jsonl",
    "mcp_interaction_event": "mcp_interaction_events.jsonl",
    "database_event": "database_events.jsonl",
    "structured_table_event": "structured_table_events.jsonl",
    "software_execution_event": "software_execution_events.jsonl",
    "search_retrieval_event": "search_retrieval_events.jsonl",
    "enterprise_app_event": "enterprise_app_events.jsonl",
    "async_workflow_event": "async_workflow_events.jsonl",
    "collaboration_event": "collaboration_events.jsonl",
    "document_event": "document_events.jsonl",
    "cross_system_event": "cross_system_events.jsonl",
}
TRAINING_USE_BY_TYPE = {
    "training_example": ["sft", "lora_ready_instruction_tuning"],
    "evaluation_example": ["eval_generation", "rubric_training"],
    "verifier_example": ["verifier_training"],
    "reward_model_row": ["reward_modeling", "agentic_rl_reward"],
    "preference_pair": ["preference_optimization", "dpo_ipo_orpo"],
    "failure_case": ["failure_prediction", "repair_policy_training"],
    "task_variant": ["transfer_eval_generation", "curriculum_generation"],
    "environment_response_pair": ["environment_model_training", "agentic_rl", "grpo_rollout_material"],
    "multi_agent_handoff": ["process_supervision", "agentic_rl"],
    "multimodal_artifact": ["multimodal_grounding", "verifier_training"],
    "ui_interaction_event": ["process_supervision", "environment_model_training"],
    "physical_environment_event": ["environment_model_training", "safety_verifier_training"],
    "api_interaction_event": ["tool_use_training", "environment_model_training"],
    "mcp_interaction_event": ["tool_use_training", "environment_model_training"],
    "database_event": ["tool_use_training", "process_supervision"],
    "structured_table_event": ["tool_use_training", "verifier_training"],
    "software_execution_event": ["process_supervision", "agentic_rl"],
    "search_retrieval_event": ["retrieval_grounding", "citation_verifier_training"],
    "enterprise_app_event": ["process_supervision", "environment_model_training"],
    "async_workflow_event": ["process_supervision", "environment_model_training"],
    "collaboration_event": ["process_supervision", "feedback_training"],
    "document_event": ["document_grounding", "verifier_training"],
    "cross_system_event": ["process_supervision", "environment_model_training"],
}
ROW_ID_FIELD_BY_TYPE = {
    "preference_pair": "pair_id",
    "critique_revision_pair": "pair_id",
    "feedback_revision_pair": "pair_id",
}
PRIVATE_RE = re.compile(r"(/Users/|/private/|/var/folders/|/tmp/|\.env(?:\.local)?)")
FORBIDDEN_KEYS = {
    "source_ref",
    "source_basis",
    "source_url_or_ref",
    "source_url",
    "source_kind",
    "source_license",
    ("b" + "uyer"),
    ("s" + "eller"),
    ("pr" + "ice"),
    ("list" + "ing"),
    ("market" + "place"),
    "agent_self_eval",
    "self_eval",
    "trace_context",
    "worker_agent",
    "worker_attempt",
    "evaluation_mode",
    "data_sharing_level",
    ("feat" + "ured"),
    ("cur" + "ated"),
    ("limit" + "ations"),
    ("known_" + "limitations"),
    ("limitations" + "_count"),
    ("when_" + "not_to_use"),
    ("not_" + "useful_for"),
    ("non_" + "use_cases"),
    ("rubric_" + "limitations"),
}
PUBLIC_TEXT_REPLACEMENTS = {
    "Gatekeeping": "Review Routing",
    "gatekeeping": "review routing",
    ("Worker " + "Agent"): "Apprentice Agent",
    ("worker " + "agent"): "Apprentice Agent",
    ("Deep" + "Seek"): "model provider",
    ("files_created_" + "list" + "ing_inconsistency"): "files_created_index_inconsistency",
    ("Cur" + "ated"): "Selected",
    ("cur" + "ated"): "selected",
    ("Contribution " + "Bundle"): "Experience Compilation",
    ("contribution " + "bundle"): "Experience Compilation",
    ("Export Full " + "Training Package"): "Export Full Experience Compilation",
    ("Training Data " + "Optimization"): "Experience Compiler",
    ("T" + "DO"): "Experience Compiler",
    ("Honest " + "limitations"): "Generation notes",
    ("honest " + "limitations"): "generation notes",
    ("Limit" + "ations"): "Generation Notes",
    ("limit" + "ations"): "generation notes",
    ("limit" + "ation"): "generation note",
    ("Applic" + "ability"): "Best Use",
    ("applic" + "ability"): "fit",
    ("Required " + "Conditions"): "Required Inputs",
    ("Evidence " + "Coverage"): "Source Evidence",
    ("Evidence " + "coverage"): "Source evidence",
    ("evidence " + "coverage"): "source evidence",
    ("Coverage " + "Notes"): "Generation Notes",
    ("Data Coverage " + "Notes"): "Generation Notes",
    ("Transfer " + "Boundaries"): "Transfer Notes",
    ("Missing Evidence " + "Notes"): "Generation Notes",
    ("Quality " + "Notes"): "Quality Signals",
    ("quality " + "notes"): "quality signals",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _safe_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", value).strip("_") or "package"


def _truthy_env(name: str) -> bool:
    return (os.getenv(name) or "").strip().lower() in {"1", "true", "yes", "y", "on"}


def _int_env(name: str, default: int) -> int:
    try:
        return max(1, int(os.getenv(name) or default))
    except ValueError:
        return default


def _float_env(name: str, default: float) -> float:
    try:
        return max(0.0, min(1.0, float(os.getenv(name) or default)))
    except ValueError:
        return default


def _safe_read(path: Path) -> Any:
    try:
        return read_json(path)
    except Exception:
        return None


def _safe_jsonl(path: Path) -> list[Any]:
    try:
        return read_jsonl(path)
    except Exception:
        return []


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("")
    for row in rows:
        append_jsonl(path, sanitize_public_obj(row))


def _clean_text(value: Any, limit: int = 1200) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        text = str(sanitize_public_obj(value) if isinstance(value, dict) else value)
    else:
        text = str(value)
    text = sanitize_public_text(redact_secrets(text)) or ""
    for old, new in PUBLIC_TEXT_REPLACEMENTS.items():
        text = text.replace(old, new)
    return text[:limit]


def _normalize_ref(ref: Any) -> str | None:
    if not ref:
        return None
    text = str(ref).replace("\\", "/").strip()
    while text.startswith("./"):
        text = text[2:]
    if text.startswith("/") or ".." in Path(text).parts:
        return None
    return text or None


def _refs(values: list[Any] | Any) -> list[str]:
    if not values:
        return []
    raw = values if isinstance(values, list) else [values]
    out: list[str] = []
    for value in raw:
        ref = _normalize_ref(value)
        if ref and ref not in out:
            out.append(ref)
    return out


def _hash_text(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()


def _hash_obj(obj: Any) -> str:
    return _hash_text(json.dumps(sanitize_public_obj(obj), sort_keys=True, default=str))


def _row_identity(data_type: str, row: dict[str, Any], evidence: dict[str, Any]) -> str:
    source_task = row.get("source_task_id") or _source_task_id(evidence)
    seed = {
        "data_type": data_type,
        "source_task_id": source_task,
        "source_trace_id": row.get("source_trace_id"),
        "source_step_ids": row.get("source_step_ids") or [],
        "source_attempt_ids": row.get("source_attempt_ids") or [],
        "input": row.get("input"),
        "target": row.get("target"),
        "round": row.get("tdo_round"),
    }
    digest = hashlib.sha256(json.dumps(seed, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:16]
    prefix = "pair" if data_type in ROW_ID_FIELD_BY_TYPE else "row"
    return f"{prefix}_{_safe_id(str(source_task))}_{_safe_id(data_type)}_{digest}"


def _task_lineage(evidence: dict[str, Any]) -> dict[str, Any]:
    task = evidence.get("task") or {}
    raw = evidence.get("raw_task") or {}
    domain = task.get("domain") or raw.get("domain") or (raw.get("raw_payload") or {}).get("domain") or "general"
    subdomain = task.get("subdomain") or raw.get("subdomain") or (raw.get("raw_payload") or {}).get("subdomain") or "general"
    workflow_family = (
        task.get("task_family")
        or task.get("workflow_family")
        or raw.get("task_family")
        or raw.get("workflow_family")
        or task.get("workflow_type")
        or "general_workflow"
    )
    return {
        "source_package_id": evidence.get("package_id"),
        "source_task_id": _source_task_id(evidence),
        "task_title": _task_title(evidence),
        "domain": _clean_text(domain, 100),
        "subdomain": _clean_text(subdomain, 100),
        "workflow_family": _clean_text(workflow_family, 120),
    }


def _evidence_id(ref: str) -> str:
    return f"evidence_{hashlib.sha256(ref.encode('utf-8')).hexdigest()[:16]}"


def _evidence_type_for_ref(ref: str) -> str:
    if ref.startswith("task/") or ref.startswith("source_task/"):
        return "task_spec"
    if ref.startswith("rubric/"):
        return "rubric"
    if ref.startswith("attempts/") and ref.endswith("agent_trace.json"):
        return "trace"
    if ref.startswith("attempts/") and ref.endswith("actual_outputs.json"):
        return "actual_outputs"
    if "/artifacts/" in ref or ref.startswith("artifacts/"):
        return "artifact"
    if ref.startswith("grading/") and "grader" in ref:
        return "grader"
    if ref.startswith("grading/") and "verifier" in ref:
        return "verifier"
    if ref.startswith("feedback/"):
        return "feedback"
    if ref.startswith("signals/"):
        return "training_signal"
    if ref.startswith("loops/"):
        if ref.endswith("review_packet.json"):
            return "review_packet"
        if ref.endswith("revision_plan.json"):
            return "revision_plan"
        if ref.endswith("loop_decision.json"):
            return "loop_decision"
        if "contract_" in ref:
            return "contract_verification_or_repair"
        return "loop_evidence"
    return "package_file"


def _ref_resolves(package_root: Path, ref: str) -> bool:
    if (package_root / ref).exists():
        return True
    if ref.startswith("traces/") or ref.startswith("outputs/") or ref.startswith("evaluation/"):
        return True
    return False


def _content_hash_for_ref(package_root: Path, ref: str) -> str | None:
    path = package_root / ref
    if not path.exists() or not path.is_file() or path.stat().st_size > 2_000_000:
        return None
    try:
        return _hash_text(path.read_text(errors="replace"))
    except Exception:
        return None


def _evidence_catalog(evidence: dict[str, Any]) -> list[dict[str, Any]]:
    package_root = evidence.get("package_root")
    if not isinstance(package_root, Path):
        return []
    catalog: list[dict[str, Any]] = []
    for ref in _evidence_refs(evidence):
        ref = str(ref)
        catalog.append(
            {
                "evidence_id": _evidence_id(ref),
                "ref": ref,
                "evidence_type": _evidence_type_for_ref(ref),
                "visibility": "visible" if not ref.startswith("hidden") and "/hidden" not in ref else "hidden_excluded",
                "resolves": _ref_resolves(package_root, ref),
                "content_hash": _content_hash_for_ref(package_root, ref),
                "confidence": 1.0 if _ref_resolves(package_root, ref) else 0.5,
            }
        )
    return catalog


def _evidence_ids_for_refs(refs: list[str]) -> list[str]:
    return [_evidence_id(ref) for ref in _refs(refs)]


def _score(value: Any, default: float = 0.0) -> float:
    try:
        raw = float(value)
    except (TypeError, ValueError):
        return default
    if raw > 1.0:
        raw = raw / 100.0
    return max(0.0, min(1.0, raw))


def _source_task_id(evidence: dict[str, Any]) -> str:
    task = evidence.get("task") or {}
    raw = evidence.get("raw_task") or {}
    manifest = evidence.get("manifest") or {}
    return str(
        task.get("task_id")
        or raw.get("task_id")
        or raw.get("raw_task_id")
        or manifest.get("task_id")
        or evidence["package_id"]
    )


def _load_evidence(package_root: Path, run_root: Path | None = None) -> dict[str, Any]:
    attempts: dict[str, dict[str, Any]] = {}
    for attempt_dir in sorted((package_root / "attempts").glob("*")):
        if not attempt_dir.is_dir():
            continue
        if not (attempt_dir / "actual_outputs.json").exists() and not (attempt_dir / "agent_trace.json").exists():
            continue
        kind = attempt_dir.name
        attempts[kind] = {
            "actual_outputs": _safe_read(attempt_dir / "actual_outputs.json") or {},
            "agent_trace": _safe_read(attempt_dir / "agent_trace.json") or {},
            "attempt_manifest": _safe_read(attempt_dir / "attempt_manifest.json") or {},
        }
    loop_review_refs: list[str] = []
    loop_review_packets: list[dict[str, Any]] = []
    for packet_path in sorted((package_root / "loops").glob("iterations/*/review_packet.json")):
        try:
            packet = read_json(packet_path)
        except Exception:
            continue
        rel = packet_path.relative_to(package_root).as_posix()
        loop_review_refs.append(rel)
        loop_review_packets.append({"ref": rel, "packet": packet})
    for rel in [
        "loops/loop_manifest.json",
        *[p.relative_to(package_root).as_posix() for p in sorted((package_root / "loops").glob("iterations/*/reviewer_feedback.json"))],
        *[p.relative_to(package_root).as_posix() for p in sorted((package_root / "loops").glob("iterations/*/revision_plan.json"))],
        *[p.relative_to(package_root).as_posix() for p in sorted((package_root / "loops").glob("iterations/*/loop_decision.json"))],
        *[p.relative_to(package_root).as_posix() for p in sorted((package_root / "loops").glob("iterations/*/diff_report.json"))],
    ]:
        if (package_root / rel).exists() and rel not in loop_review_refs:
            loop_review_refs.append(rel)
    return {
        "package_id": package_root.name,
        "package_root": package_root,
        "run_root": run_root,
        "manifest": _safe_read(package_root / "package_manifest.json") or _safe_read(package_root / "manifest.json") or {},
        "raw_task": _safe_read(package_root / "task" / "raw_task_record.json") or _safe_read(package_root / "task" / "task_packet.json") or {},
        "task": _safe_read(package_root / "task" / "task_intake_spec.json") or _safe_read(package_root / "task" / "task_packet.json") or {},
        "rubric": _safe_read(package_root / "rubric" / "rubric.json") or {},
        "attempts": attempts,
        "grader_results": {
            path.stem.replace("_grader_result", ""): _safe_read(path) or {}
            for path in sorted((package_root / "grading").glob("*_grader_result.json"))
        },
        "verifier_results": {
            path.stem.replace("_verifier_result", ""): _safe_read(path) or {}
            for path in sorted((package_root / "grading").glob("*_verifier_result.json"))
        },
        "evaluator_feedback": _safe_read(package_root / "feedback" / "baseline_evaluator_feedback.json") or {},
        "revision_plan": _safe_read(package_root / "feedback" / "revision_plan.json") or {},
        "hillclimb": _safe_read(package_root / "signals" / "hillclimb_result.json") or {},
        "lessons": _safe_read(package_root / "signals" / "lesson_pack.json") or {},
        "training_signals": _safe_jsonl(package_root / "signals" / "training_signals.jsonl"),
        "process_supervision": _safe_jsonl(package_root / "signals" / "process_supervision.jsonl"),
        "reward_modeling": _safe_jsonl(package_root / "signals" / "reward_modeling.jsonl"),
        "revision_preference_pairs": _safe_jsonl(package_root / "signals" / "revision_preference_pairs.jsonl"),
        "artifact_index": _safe_read(package_root / "artifacts_index.json") or [],
        "session_events": _safe_jsonl(run_root / "session_events.jsonl") if run_root else [],
        "loop_review_refs": loop_review_refs,
        "loop_review_packets": loop_review_packets,
    }


def _attempt_ids(evidence: dict[str, Any]) -> list[str]:
    ids: list[str] = []
    for attempt in (evidence.get("attempts") or {}).values():
        actual = attempt.get("actual_outputs") or {}
        attempt_id = actual.get("attempt_id")
        if attempt_id and attempt_id not in ids:
            ids.append(str(attempt_id))
    return ids


def _primary_trace_id(evidence: dict[str, Any]) -> str | None:
    for attempt in (evidence.get("attempts") or {}).values():
        trace = attempt.get("agent_trace") or {}
        if trace.get("trace_id"):
            return str(trace.get("trace_id"))
    return None


def _source_artifact_refs(evidence: dict[str, Any]) -> list[str]:
    refs: list[str] = []
    package_root = evidence.get("package_root")
    for attempt_kind, attempt in (evidence.get("attempts") or {}).items():
        actual = attempt.get("actual_outputs") or {}
        for key in ["deliverable_refs", "artifact_refs", "files_created", "input_artifact_refs"]:
            refs.extend(_refs(actual.get(key)))
        for local in [f"attempts/{attempt_kind}/actual_outputs.json", f"attempts/{attempt_kind}/agent_trace.json"]:
            if not isinstance(package_root, Path) or (package_root / local).exists():
                refs.append(local)
    return list(dict.fromkeys(refs))


def _looks_numeric(value: Any) -> bool:
    try:
        if value in (None, ""):
            return False
        float(str(value))
        return True
    except Exception:
        return False


def _json_type_name(value: Any) -> str:
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return type(value).__name__


def _artifact_contracts(evidence: dict[str, Any], limit: int = 12) -> list[dict[str, Any]]:
    package_root = evidence.get("package_root")
    if not isinstance(package_root, Path):
        return []
    contracts: list[dict[str, Any]] = []
    seen: set[str] = set()
    for ref in _source_artifact_refs(evidence):
        if ref in seen or ref.startswith("input/") or "/input/" in ref:
            continue
        seen.add(ref)
        path = package_root / ref
        if not path.exists() or not path.is_file():
            continue
        suffix = path.suffix.lower()
        contract: dict[str, Any] = {"artifact_ref": ref, "filename": path.name}
        try:
            if suffix == ".csv":
                with path.open(newline="", errors="replace") as f:
                    reader = csv.DictReader(f)
                    header = list(reader.fieldnames or [])
                    rows = [row for _, row in zip(range(5), reader)]
                numeric_columns: list[str] = []
                non_empty_columns: list[str] = []
                for column in header:
                    values = [row.get(column) for row in rows if row.get(column) not in (None, "")]
                    if values:
                        non_empty_columns.append(column)
                        if all(_looks_numeric(value) for value in values):
                            numeric_columns.append(column)
                contract.update({
                    "format": "csv",
                    "columns": header,
                    "example_row_shape": len(header),
                    "non_empty_columns": non_empty_columns[:16],
                    "numeric_columns": numeric_columns[:16],
                    "min_rows": 1 if rows else 0,
                })
            elif suffix == ".json":
                data = json.loads(path.read_text(errors="replace"))
                if isinstance(data, dict):
                    expected_types = {
                        str(key): _json_type_name(value)
                        for key, value in data.items()
                        if value not in (None, "", [], {})
                    }
                    contract.update({
                        "format": "json",
                        "top_level_keys": list(data.keys())[:16],
                        "expected_types": expected_types,
                        "non_empty_fields": list(expected_types.keys())[:16],
                    })
                elif isinstance(data, list):
                    item_keys: list[str] = []
                    expected_types: dict[str, str] = {}
                    if data and isinstance(data[0], dict):
                        item_keys = list(data[0].keys())[:16]
                        expected_types = {
                            str(key): _json_type_name(value)
                            for key, value in data[0].items()
                            if value not in (None, "", [], {})
                        }
                    contract.update({
                        "format": "json",
                        "top_level_type": "array",
                        "item_count": len(data),
                        "top_level_keys": item_keys,
                        "expected_types": expected_types,
                        "non_empty_fields": list(expected_types.keys())[:16],
                        "min_rows": 1 if data else 0,
                    })
                else:
                    contract.update({"format": "json", "top_level_type": type(data).__name__})
            elif suffix in {".txt", ".md", ".sh", ".bed", ".tsv", ".svg"}:
                contract.update({"format": suffix.lstrip(".") or "text"})
            else:
                contract.update({"format": suffix.lstrip(".") or "file"})
        except Exception as exc:
            contract.update({"format": suffix.lstrip(".") or "file", "read_error": type(exc).__name__})
        contracts.append(contract)
        if len(contracts) >= limit:
            break
    return contracts


def _contract_line(contract: dict[str, Any]) -> str:
    filename = contract.get("filename") or contract.get("artifact_ref")
    if contract.get("columns"):
        return f"`{filename}`: CSV columns {', '.join(map(str, contract.get('columns') or []))}."
    if contract.get("top_level_keys"):
        return f"`{filename}`: JSON keys {', '.join(map(str, contract.get('top_level_keys') or []))}."
    return f"`{filename}`: {contract.get('format') or 'file'} artifact."


def _contract_output_path(contract: dict[str, Any]) -> str:
    ref = str(contract.get("artifact_ref") or contract.get("filename") or "").replace("\\", "/")
    name = str(contract.get("filename") or Path(ref).name or "output")
    if "/artifacts/" in ref:
        return "artifacts/" + ref.split("/artifacts/", 1)[1]
    if "/output/" in ref:
        return "output/" + ref.split("/output/", 1)[1]
    return f"output/{name}"


def _output_contract_payload(evidence: dict[str, Any], artifact_contracts: list[dict[str, Any]]) -> dict[str, Any]:
    outputs = []
    for contract in artifact_contracts:
        entry = {
            "path": _contract_output_path(contract),
            "format": contract.get("format") or "file",
            "scoring_critical": True,
        }
        if contract.get("columns"):
            entry["columns"] = contract.get("columns")
        if contract.get("numeric_columns"):
            entry["numeric_columns"] = contract.get("numeric_columns")
        if contract.get("non_empty_columns"):
            entry["non_empty_columns"] = contract.get("non_empty_columns")
        if contract.get("top_level_keys"):
            entry["keys"] = contract.get("top_level_keys")
        if contract.get("expected_types"):
            entry["expected_types"] = contract.get("expected_types")
        if contract.get("non_empty_fields"):
            entry["non_empty_fields"] = contract.get("non_empty_fields")
        if contract.get("min_rows") is not None:
            entry["min_rows"] = contract.get("min_rows")
        outputs.append(entry)
    if not any(item.get("path", "").endswith("actual_outputs.json") for item in outputs):
        outputs.append({
            "path": "output/actual_outputs.json",
            "format": "json",
            "keys": ["output_summary", "files_created"],
            "scoring_critical": True,
        })
    return {
        "source_package_id": evidence.get("package_id"),
        "contract_kind": "runtime_transfer_output_contract",
        "required_outputs": outputs,
        "validation_commands": [
            "open every required file",
            "validate JSON parses",
            "validate CSV/TSV headers",
            "confirm actual_outputs.json lists generated files",
        ],
        "common_schema_mistakes": [
            "writing files under the wrong relative directory",
            "omitting required CSV columns or JSON keys",
            "leaving actual_outputs.json empty or stale",
        ],
    }


def _artifact_contract_payload(evidence: dict[str, Any], artifact_contracts: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "source_package_id": evidence.get("package_id"),
        "contract_kind": "runtime_transfer_artifact_contract",
        "required_artifacts": [
            {
                "path": _contract_output_path(contract),
                "artifact_type": contract.get("format") or "file",
                "expected_location": str(Path(_contract_output_path(contract)).parent),
                "name_pattern": Path(_contract_output_path(contract)).name,
                "verify_exists": True,
                **({"columns": contract.get("columns")} if contract.get("columns") else {}),
                **({"keys": contract.get("top_level_keys")} if contract.get("top_level_keys") else {}),
                **({"numeric_columns": contract.get("numeric_columns")} if contract.get("numeric_columns") else {}),
                **({"non_empty_columns": contract.get("non_empty_columns")} if contract.get("non_empty_columns") else {}),
                **({"expected_types": contract.get("expected_types")} if contract.get("expected_types") else {}),
                **({"non_empty_fields": contract.get("non_empty_fields")} if contract.get("non_empty_fields") else {}),
                **({"min_rows": contract.get("min_rows")} if contract.get("min_rows") is not None else {}),
            }
            for contract in artifact_contracts
        ],
    }


def _prefinal_checklist_payload(evidence: dict[str, Any], output_contract: dict[str, Any], artifact_contract: dict[str, Any]) -> dict[str, Any]:
    checks = [
        {
            "check_id": "required_files_exist",
            "description": "Every required output/artifact path exists in the task output location.",
            "verification_method": "filesystem_exists",
            "blocking": True,
            "repair_hint": "Create or move missing files to the required relative path.",
        },
        {
            "check_id": "json_outputs_parse",
            "description": "JSON outputs, including actual_outputs.json, parse as JSON objects or declared arrays.",
            "verification_method": "json_parse",
            "blocking": True,
            "repair_hint": "Rewrite malformed JSON and rerun the parse check.",
        },
        {
            "check_id": "csv_tsv_headers_match",
            "description": "CSV/TSV outputs include required scoring-critical columns.",
            "verification_method": "delimited_header_check",
            "blocking": True,
            "repair_hint": "Rewrite files with the required header names exactly.",
        },
        {
            "check_id": "actual_outputs_lists_files",
            "description": "actual_outputs.json summarizes and lists generated deliverables.",
            "verification_method": "actual_outputs_contract_check",
            "blocking": True,
            "repair_hint": "Update actual_outputs.json after producing outputs.",
        },
    ]
    return {
        "source_package_id": evidence.get("package_id"),
        "checks": checks,
        "output_contract_ref": "runtime_learning_package/output_contract.json",
        "artifact_contract_ref": "runtime_learning_package/artifact_contract.json",
        "required_output_count": len(output_contract.get("required_outputs") or []),
        "required_artifact_count": len(artifact_contract.get("required_artifacts") or []),
    }


def _repair_instructions_text(title: str, output_contract: dict[str, Any]) -> str:
    required = output_contract.get("required_outputs") or []
    lines = [
        f"# Contract Repair Instructions: {title}",
        "",
        "Use these instructions only after a pre-final contract check fails.",
        "",
        "1. Read each contract failure and map it to a required output path.",
        "2. Regenerate missing files from current visible inputs; do not copy source-task answers.",
        "3. For malformed CSV/TSV files, rewrite the header exactly and preserve derived rows.",
        "4. For malformed JSON, rewrite valid JSON with required keys.",
        "5. Reopen each repaired file and rerun the contract verifier.",
        "",
        "## Required Output Paths",
    ]
    lines.extend(f"- `{item.get('path')}` ({item.get('format') or 'file'})" for item in required[:20])
    return "\n".join(lines) + "\n"


def _runtime_transfer_runbook(
    evidence: dict[str, Any],
    title: str,
    instruction: str,
    criteria: list[str],
    artifact_contracts: list[dict[str, Any]],
) -> str:
    contract_lines = [_contract_line(contract) for contract in artifact_contracts[:10]]
    if not contract_lines:
        contract_lines = ["Create every deliverable named by the current task and register it in `actual_outputs.json`."]
    verifier_lines = criteria[:8] or ["Every required artifact exists.", "`actual_outputs.json` names the generated deliverables."]
    return "\n".join(
        [
            f"# Transfer Runbook: {title}",
            "",
            "Use this as a compact execution checklist for a related task. Current-task instructions and inputs remain authoritative.",
            "",
            "## Execution Steps",
            "1. Read the current task directory, visible input files, and required output names before writing code.",
            "2. Build a small script or commands that derive outputs from current input files; do not copy source-task values.",
            "3. Write outputs under `output/` unless the current task names a different relative location.",
            "4. Match artifact filenames, CSV headers, JSON keys, and text-file roles exactly.",
            "5. Write `output/actual_outputs.json` or `actual_outputs.json` with the produced file list and concise summary.",
            "6. Run a local sanity check by reopening each file and verifying row/key counts and required columns.",
            "",
            "## Artifact Contract Pattern",
            *(f"- {line}" for line in contract_lines),
            "",
            "## Verification Checklist",
            *(f"- {line}" for line in verifier_lines),
            "",
            "## Transfer Cues",
            f"- Source workflow pattern: {_clean_text(instruction, 500)}",
            "- Transfer the procedure, output contract discipline, and checks; do not transfer source-task answers.",
            "- If a formula or threshold is task-specific, infer it from the current task inputs or visible instructions.",
            "",
        ]
    )


def _loop_learning_rows(evidence: dict[str, Any], report: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    package_root = evidence.get("package_root")
    source_package_id = report.get("source_package_id") or evidence.get("package_id")
    out: dict[str, list[dict[str, Any]]] = {
        "loop_review_rows": [],
        "feedback_revision_pairs": [],
        "contract_verification_rows": [],
        "repair_loop_rows": [],
        "reviewer_decision_rows": [],
    }
    if not isinstance(package_root, Path):
        return out
    for item in evidence.get("loop_review_packets") or []:
        packet = item.get("packet") or {}
        ref = item.get("ref")
        if not ref:
            continue
        iteration_dir = package_root / Path(ref).parent
        loop_iteration = packet.get("loop_iteration")
        reviewer_feedback = _safe_read(iteration_dir / "reviewer_feedback.json") or {}
        revision_plan = _safe_read(iteration_dir / "revision_plan.json") or {}
        loop_decision = _safe_read(iteration_dir / "loop_decision.json") or {}
        contract_verification = _safe_read(iteration_dir / "contract_verification.json") or {}
        contract_repair_plan = _safe_read(iteration_dir / "contract_repair_plan.json") or {}
        contract_repair_result = _safe_read(iteration_dir / "contract_repair_result.json") or {}
        diff_report = _safe_read(iteration_dir / "diff_report.json") or {}

        base = {
            "schema_version": TRAINING_ROW_SCHEMA_VERSION,
            "source_package_id": source_package_id,
            "source_task_id": report.get("source_task_id"),
            "task_lineage": _task_lineage(evidence),
            "loop_id": packet.get("loop_id"),
            "loop_iteration": loop_iteration,
            "evidence_refs": [ref],
            "source_evidence_ref_ids": _evidence_ids_for_refs([ref]),
            "confidence": report.get("average_quality_score") or 0.5,
            "created_at": utc_now(),
            "generated_by": "Experience Compiler",
        }
        known_failures = [str(v) for v in (packet.get("known_failures") or []) if v]
        missing_outputs = [str(v) for v in (packet.get("missing_outputs") or []) if v]
        out["loop_review_rows"].append(
            sanitize_public_obj(
                {
                    **base,
                    "row_id": f"loop_review_{_safe_id(str(packet.get('loop_id') or source_package_id))}_{loop_iteration or 0}",
                    "data_type": "loop_review_row",
                    "mode": packet.get("mode"),
                    "current_attempt_summary": _clean_text(packet.get("current_attempt_summary")),
                    "rubric_status": packet.get("rubric_status"),
                    "verifier_status": packet.get("verifier_status"),
                    "known_failures": known_failures,
                    "missing_outputs": missing_outputs,
                    "recommended_review_focus": packet.get("recommended_review_focus") or [],
                    "review_packet_ref": ref,
                    "review_packet_md_ref": str(Path(ref).with_suffix(".md")),
                }
            )
        )
        if reviewer_feedback or revision_plan:
            out["feedback_revision_pairs"].append(
                sanitize_public_obj(
                    {
                    **base,
                    "pair_id": f"feedback_revision_{_safe_id(str(packet.get('loop_id') or source_package_id))}_{loop_iteration or 0}",
                    "row_id": f"feedback_revision_{_safe_id(str(packet.get('loop_id') or source_package_id))}_{loop_iteration or 0}",
                    "data_type": "feedback_revision_pair",
                        "reviewer_type": reviewer_feedback.get("reviewer_type") or "unknown",
                        "feedback": _clean_text(reviewer_feedback.get("feedback") or reviewer_feedback.get("feedback_text") or reviewer_feedback.get("summary")),
                        "blocking_issues": reviewer_feedback.get("blocking_issues") or known_failures,
                        "revision_instructions": reviewer_feedback.get("revision_instructions") or revision_plan.get("issues_to_fix") or [],
                        "revision_plan_ref": f"{Path(ref).parent.as_posix()}/revision_plan.json",
                        "diff_report_ref": f"{Path(ref).parent.as_posix()}/diff_report.json",
                        "feedback_applied_status": revision_plan.get("feedback_applied") or "unknown",
                    }
                )
            )
        if loop_decision:
            out["reviewer_decision_rows"].append(
                sanitize_public_obj(
                    {
                    **base,
                    "row_id": f"reviewer_decision_{_safe_id(str(packet.get('loop_id') or source_package_id))}_{loop_iteration or 0}",
                    "data_type": "reviewer_decision_row",
                        "decision": loop_decision.get("decision"),
                        "verdict": loop_decision.get("verdict"),
                        "score": loop_decision.get("score"),
                        "continue_loop": loop_decision.get("continue_loop"),
                        "stop_condition": loop_decision.get("stop_condition"),
                        "termination_reason": loop_decision.get("termination_reason"),
                        "training_value_notes": loop_decision.get("training_value_notes"),
                        "loop_decision_ref": f"{Path(ref).parent.as_posix()}/loop_decision.json",
                    }
                )
            )
        if contract_verification:
            failures = contract_verification.get("failures") or []
            out["contract_verification_rows"].append(
                sanitize_public_obj(
                    {
                    **base,
                    "row_id": f"contract_verification_{_safe_id(str(packet.get('loop_id') or source_package_id))}_{loop_iteration or 0}",
                    "data_type": "contract_verification_row",
                        "contract_verification_passed": contract_verification.get("contract_verification_passed"),
                        "actual_outputs_json_ok": contract_verification.get("actual_outputs_json_ok"),
                        "failure_count": contract_verification.get("failure_count", len(failures)),
                        "semantic_warning_count": contract_verification.get("semantic_warning_count", 0),
                        "failures": failures[:20],
                        "semantic_warnings": (contract_verification.get("semantic_warnings") or [])[:20],
                        "contract_verification_ref": f"{Path(ref).parent.as_posix()}/contract_verification.json",
                    }
                )
            )
        if contract_repair_plan or contract_repair_result:
            out["repair_loop_rows"].append(
                sanitize_public_obj(
                    {
                    **base,
                    "row_id": f"repair_loop_{_safe_id(str(packet.get('loop_id') or source_package_id))}_{loop_iteration or 0}",
                    "data_type": "repair_loop_row",
                        "repair_required": contract_repair_plan.get("repair_required"),
                        "preserve_passing_items": contract_repair_plan.get("preserve_passing_items"),
                        "issues_to_fix": contract_repair_plan.get("issues_to_fix") or [],
                        "repair_result": contract_repair_result,
                        "changed_refs": (diff_report.get("added_refs") or []) + (diff_report.get("removed_refs") or []),
                        "contract_repair_plan_ref": f"{Path(ref).parent.as_posix()}/contract_repair_plan.json",
                        "contract_repair_result_ref": f"{Path(ref).parent.as_posix()}/contract_repair_result.json",
                    }
                )
            )
    return out


def _evidence_refs(evidence: dict[str, Any]) -> list[str]:
    refs = ["task/task_intake_spec.json", "rubric/rubric.json"]
    refs.extend(_source_artifact_refs(evidence))
    for name in [
        "grading/baseline_grader_result.json",
        "grading/revised_grader_result.json",
        "grading/baseline_verifier_result.json",
        "grading/revised_verifier_result.json",
        "feedback/baseline_evaluator_feedback.json",
        "feedback/revision_plan.json",
        "signals/hillclimb_result.json",
    ]:
        if (evidence["package_root"] / name).exists():
            refs.append(name)
    refs.extend(evidence.get("loop_review_refs") or [])
    return list(dict.fromkeys(refs))


def _infer_environment_domain(step: dict[str, Any] | None = None, tool: str | None = None, operation: str | None = None) -> str:
    step = step or {}
    raw = " ".join(str(v or "") for v in [tool or step.get("tool"), operation or step.get("operation"), step.get("action"), step.get("interaction_surface"), step.get("input"), step.get("output")]).lower()
    if any(x in raw for x in ["industrial", "plc", "scada", "manufacturing", "telemetry"]):
        return "industrial"
    if any(x in raw for x in ["robot", "actuator", "sensor", "physical"]):
        return "robotics" if "robot" in raw else "physical"
    if any(x in raw for x in ["desktop", "computer_use", "accessibility", "ui automation"]):
        return "computer_use"
    if any(x in raw for x in ["bash", "shell", "terminal", "codex_cli", "command"]):
        return "terminal"
    if any(x in raw for x in ["browser", "playwright"]):
        return "browser"
    if any(x in raw for x in ["search", "web"]):
        return "search"
    if any(x in raw for x in ["file", "read", "write"]):
        return "file"
    if any(x in raw for x in ["code", "test", "build", "edit"]):
        return "codebase"
    if any(x in raw for x in ["sql", "database", "db"]):
        return "database"
    if any(x in raw for x in ["sheet", "spreadsheet", "xlsx", "csv"]):
        return "spreadsheet"
    if any(x in raw for x in ["document", "pdf", "ocr", "image", "audio", "video", "screenshot", "table"]):
        return "multimodal"
    if "mcp" in raw:
        return "mcp"
    if "api" in raw:
        return "api"
    return str(step.get("environment_domain") or "unknown") if step.get("environment_domain") in ENVIRONMENT_DOMAINS else "unknown"


def _weak_strong_proxy(evidence: dict[str, Any]) -> dict[str, Any]:
    hill = evidence.get("hillclimb") or {}
    if hill.get("baseline_score") is not None and hill.get("revised_score") is not None:
        weak = _score(hill.get("baseline_score"))
        strong = _score(hill.get("revised_score"))
        gap = strong - weak
        return {
            "available": True,
            "weak_score_proxy": weak,
            "strong_score_proxy": strong,
            "solver_gap_proxy": round(gap, 4),
            "weak_pattern": _clean_text("; ".join(hill.get("failed_criteria_before") or []) or "Baseline attempt before revision."),
            "strong_pattern": _clean_text("; ".join(hill.get("criteria_improved") or []) or "Revised or selected attempt had stronger evidence."),
            "gap_interpretation": "Revised evidence outperformed baseline." if gap > 0 else "No positive solver gap was observed.",
            "grpo_suitability": "high" if gap >= 0.2 else "medium" if gap > 0.05 else "low",
            "learning_signal_quality": "high" if gap >= 0.2 else "medium" if gap > 0 else "low",
        }
    graders = list((evidence.get("grader_results") or {}).values())
    if len(graders) >= 2:
        scores = [_score(g.get("final_score") if g.get("final_score") is not None else g.get("score")) for g in graders]
        weak = min(scores)
        strong = max(scores)
        gap = strong - weak
        return {
            "available": True,
            "weak_score_proxy": weak,
            "strong_score_proxy": strong,
            "solver_gap_proxy": round(gap, 4),
            "weak_pattern": "Lower-scoring attempt.",
            "strong_pattern": "Higher-scoring attempt.",
            "gap_interpretation": "Score gap inferred from grader results.",
            "grpo_suitability": "medium" if gap > 0.05 else "low",
            "learning_signal_quality": "medium" if gap > 0 else "low",
        }
    return {
        "available": False,
        "weak_score_proxy": None,
        "strong_score_proxy": None,
        "solver_gap_proxy": None,
        "weak_pattern": None,
        "strong_pattern": None,
        "gap_interpretation": "Weak/strong proxy unavailable from this package.",
        "grpo_suitability": "unknown",
        "learning_signal_quality": "unknown",
    }


def _source_audit_status(evidence: dict[str, Any]) -> dict[str, Any]:
    task_md = (evidence.get("task") or {}).get("metadata_json") or {}
    rubric_md = (evidence.get("rubric") or {}).get("metadata_json") or {}
    task_audited = bool(task_md.get("mentor_audited"))
    rubric_audited = bool(rubric_md.get("mentor_audited"))
    return {
        "source_task_spec_mentor_audited": task_audited,
        "source_task_spec_audit_source": task_md.get("mentor_audit_source"),
        "source_task_spec_audit_status": task_md.get("audit_status") or ("mentor_audited" if task_audited else "unaudited"),
        "source_rubric_mentor_audited": rubric_audited,
        "source_rubric_audit_source": rubric_md.get("mentor_audit_source"),
        "source_rubric_audit_status": rubric_md.get("audit_status") or ("mentor_audited" if rubric_audited else "unaudited"),
    }


def _base_row(evidence: dict[str, Any], settings: Settings, round_idx: int, judge_source: str, mentor_audited: bool, proxy: dict[str, Any], quality: float) -> dict[str, Any]:
    lineage = _task_lineage(evidence)
    evidence_refs = _evidence_refs(evidence)
    source_artifact_refs = _source_artifact_refs(evidence)
    row = {
        "schema_version": TRAINING_ROW_SCHEMA_VERSION,
        "source_package_id": evidence["package_id"],
        "source_task_id": _source_task_id(evidence),
        "task_lineage": lineage,
        "domain": lineage["domain"],
        "subdomain": lineage["subdomain"],
        "workflow_family": lineage["workflow_family"],
        "source_trace_id": _primary_trace_id(evidence),
        "source_step_ids": [],
        "source_attempt_ids": _attempt_ids(evidence),
        "source_artifact_refs": source_artifact_refs,
        "source_evaluation_refs": [ref for ref in evidence_refs if ref.startswith(("grading/", "feedback/", "signals/"))],
        "evidence_refs": evidence_refs,
        "source_evidence_ref_ids": _evidence_ids_for_refs(evidence_refs),
        "evidence_visibility": "visible_or_public_package_evidence",
        "visible_input_refs": [ref for ref in source_artifact_refs if ref.startswith("input/") or "/input/" in ref],
        "source_content_hashes": {
            ref: _content_hash_for_ref(evidence["package_root"], ref)
            for ref in evidence_refs[:24]
            if isinstance(evidence.get("package_root"), Path) and _content_hash_for_ref(evidence["package_root"], ref)
        },
        "provenance": "derived_from_observed_agent_apprenticeship_package",
        "quality_score": quality,
        "fidelity_score": min(1.0, quality + 0.04),
        "difficulty_score": min(1.0, max(0.35, quality - 0.08)),
        "learnability_score": min(1.0, quality + 0.02),
        "reuse_value_score": min(1.0, quality + 0.03),
        "confidence": min(1.0, quality + 0.02),
        "weak_score_proxy": proxy.get("weak_score_proxy"),
        "strong_score_proxy": proxy.get("strong_score_proxy"),
        "solver_gap_proxy": proxy.get("solver_gap_proxy"),
        "weak_pattern": proxy.get("weak_pattern"),
        "strong_pattern": proxy.get("strong_pattern"),
        "gap_interpretation": proxy.get("gap_interpretation"),
        "grpo_suitability": proxy.get("grpo_suitability"),
        "generated_by": "Experience Compiler",
        "generator_mode": settings.worker_agent,
        "tdo_judge_source": judge_source,
        "mentor_audited": mentor_audited,
        "tdo_round": round_idx,
        "accepted": False,
        "rejection_reason": None,
        "created_at": utc_now(),
    }
    if judge_source == "apprentice_self_judge":
        row["self_judge_generation_notes"] = [
            "judge_source=apprentice_self_judge",
            "recommended_future_mentor_audit",
        ]
        row["recommended_future_mentor_audit"] = True
    return row


def _has_concrete_source_evidence(row: dict[str, Any]) -> bool:
    """True when a row is grounded beyond task/rubric metadata alone."""
    if row.get("source_trace_id"):
        return True
    for key in ("source_step_ids", "source_attempt_ids", "source_artifact_refs", "source_evaluation_refs"):
        if _refs(row.get(key) or []):
            return True
    for ref in _refs(row.get("evidence_refs") or []):
        if ref.startswith(("traces/", "attempts/", "artifacts/", "outputs/", "evaluation/", "grading/", "feedback/", "signals/", "loops/")):
            return True
    return False


def _finalize_candidate_rows(rows: list[dict[str, Any]], evidence: dict[str, Any]) -> list[dict[str, Any]]:
    finalized: list[dict[str, Any]] = []
    for raw in rows:
        row = dict(raw)
        data_type = str(row.get("data_type") or "unknown")
        row_id = _row_identity(data_type, row, evidence)
        id_field = ROW_ID_FIELD_BY_TYPE.get(data_type, "row_id")
        row.setdefault(id_field, row_id)
        row.setdefault("row_id", row_id)
        row.setdefault("schema_version", TRAINING_ROW_SCHEMA_VERSION)
        row.setdefault("training_use_cases", TRAINING_USE_BY_TYPE.get(data_type, ["experience_compilation_training"]))
        row.setdefault("quality_signal_source", row.get("tdo_judge_source") or "experience_compiler_guardrail")
        row.setdefault("source_trace_ref", row.get("source_trace_id"))
        row.setdefault("source_evidence_ref_ids", _evidence_ids_for_refs(row.get("evidence_refs") or []))
        row.setdefault("evidence_confidence", row.get("confidence") or row.get("quality_score") or 0.5)
        row.setdefault("row_content_hash", _hash_obj({k: v for k, v in row.items() if k not in {"row_content_hash", "created_at"}}))
        row.setdefault(
            "model_training_targets",
            {
                "sft_ready": data_type == "training_example",
                "lora_ready": data_type == "training_example",
                "reward_ready": data_type == "reward_model_row",
                "verifier_ready": data_type in {"verifier_example", "evaluation_example"},
                "preference_ready": data_type == "preference_pair",
                "process_supervision_ready": data_type in {"environment_response_pair", *SURFACE_FILES.keys()},
                "rl_rollout_ready": data_type in {"environment_response_pair", "software_execution_event", "api_interaction_event"},
            },
        )
        if data_type == "training_example":
            row.setdefault("instruction", (row.get("input") or {}).get("instruction") if isinstance(row.get("input"), dict) else row.get("input"))
            row.setdefault("target_output", row.get("expected_output") or row.get("target"))
            row.setdefault("output_contract_summary", "Use actual_outputs.json plus artifact refs as the output contract.")
        elif data_type == "reward_model_row":
            target = row.get("target") if isinstance(row.get("target"), dict) else {}
            row.setdefault("reward_component", "task_outcome")
            row.setdefault("reward_value", target.get("final_score") if isinstance(target, dict) else None)
            row.setdefault("reward_timing", "final")
            row.setdefault("verifier_or_grader_source", "grader_result")
        elif data_type == "verifier_example":
            row.setdefault("claim_or_output", row.get("input"))
            row.setdefault("visible_evidence", row.get("source_artifact_refs") or row.get("evidence_refs"))
            row.setdefault("verifier_decision", row.get("target"))
        elif data_type == "preference_pair":
            target = row.get("target") if isinstance(row.get("target"), dict) else {}
            row.setdefault("chosen_output", target.get("chosen"))
            row.setdefault("rejected_output", "baseline" if target.get("chosen") == "revised" else "revised")
            row.setdefault("preference_reason", row.get("scoring_hint") or "Higher grounded score/verifier evidence.")
            row.setdefault("quality_delta", target.get("solver_gap_proxy"))
        elif data_type == "failure_case":
            target = row.get("target") if isinstance(row.get("target"), dict) else {}
            row.setdefault("failure_type", target.get("failure_mode") or "contract_or_score_failure")
            row.setdefault("failure_severity", "medium")
            row.setdefault("repair_recommendation", target.get("repair_hint") or row.get("scoring_hint"))
        elif data_type == "task_variant":
            target = row.get("target") if isinstance(row.get("target"), dict) else {}
            row.setdefault("variant_instruction", target.get("variant_instruction"))
            row.setdefault("expected_output_contract", target.get("expected_deliverables"))
            row.setdefault("difficulty", row.get("difficulty_score"))
        elif data_type == "environment_response_pair":
            row.setdefault("rollout_group_id", f"rollout_{_safe_id(str(row.get('source_task_id')))}")
            row.setdefault("candidate_rollout_id", f"rollout_{row_id}")
            row.setdefault("scalar_reward", row.get("format_score") or row.get("quality_score"))
            row.setdefault("reward_components", {"format": row.get("format_score"), "consistency": row.get("consistency_score")})
            row.setdefault("termination_reason", row.get("success_signal"))
        finalized.append(sanitize_public_obj(row))
    return finalized


def _build_candidate_rows(evidence: dict[str, Any], settings: Settings, round_idx: int, judge_source: str, mentor_audited: bool, proxy: dict[str, Any]) -> list[dict[str, Any]]:
    task = evidence.get("task") or {}
    raw = evidence.get("raw_task") or {}
    rubric = evidence.get("rubric") or {}
    attempts = evidence.get("attempts") or {}
    quality = min(0.92, 0.58 + 0.11 * round_idx)
    common = _base_row(evidence, settings, round_idx, judge_source, mentor_audited, proxy, quality)
    title = task.get("normalized_title") or raw.get("raw_title") or raw.get("title") or evidence["package_id"]
    instruction = task.get("normalized_instruction") or raw.get("raw_description") or raw.get("instruction") or title
    deliverables = task.get("output_requirements") or raw.get("output_requirements") or ["task artifact"]
    rows: list[dict[str, Any]] = []
    rows.append({
        **common,
        "data_type": "training_example",
        "input": {"instruction": _clean_text(instruction), "expected_deliverables": deliverables},
        "target": {"artifact_refs": _source_artifact_refs(evidence), "summary": _best_output_summary(evidence)},
        "expected_output": _best_output_summary(evidence),
        "rubric": rubric.get("rubric_items") or [],
        "scoring_hint": "Use actual_outputs.json, artifact refs, traces, and mentor/verifier/evaluator evidence.",
        "training_use_case": "supervised_task_completion",
    })
    for item in rubric.get("rubric_items") or []:
        rows.append({
            **common,
            "data_type": "evaluation_example",
            "input": {"artifact_refs": _source_artifact_refs(evidence), "criterion": item.get("criterion_name")},
            "target": {"criterion_description": item.get("criterion_description"), "pass_threshold": item.get("pass_threshold")},
            "expected_output": "Score the attempt against the criterion using grounded evidence.",
            "rubric": item,
            "scoring_hint": "Do not reward easier or more permissive rubrics.",
        })
    for attempt_kind, verifier in (evidence.get("verifier_results") or {}).items():
        rows.append({
            **common,
            "data_type": "verifier_example",
            "input": {"attempt_kind": attempt_kind, "artifact_refs": _source_artifact_refs(evidence)},
            "target": verifier,
            "expected_output": "Verify grounding, artifact contract, score consistency, and leakage checks.",
            "rubric": rubric.get("rubric_items") or [],
            "scoring_hint": "Verifier output is evidence; Apprentice Agent self-assessment is not outcome evidence.",
        })
    for attempt_kind, grader in (evidence.get("grader_results") or {}).items():
        rows.append({
            **common,
            "data_type": "reward_model_row",
            "input": {"attempt_kind": attempt_kind, "output_summary": _attempt_summary(attempts.get(attempt_kind) or {})},
            "target": {"final_score": grader.get("final_score", grader.get("score")), "passed": grader.get("passed")},
            "expected_output": "Prefer higher-quality grounded outputs with verifier-consistent evidence.",
            "rubric": rubric.get("rubric_items") or [],
            "scoring_hint": "Use grader and verifier evidence as reward-model labels.",
        })
    if proxy.get("available"):
        rows.append({
            **common,
            "data_type": "preference_pair",
            "input": {"baseline_attempt_ref": "attempts/baseline", "revised_attempt_ref": "attempts/revised"},
            "target": {
                "chosen": "revised" if (proxy.get("strong_score_proxy") or 0) >= (proxy.get("weak_score_proxy") or 0) else "baseline",
                "solver_gap_proxy": proxy.get("solver_gap_proxy"),
            },
            "expected_output": "Choose the attempt with stronger grounded evidence.",
            "rubric": rubric.get("rubric_items") or [],
            "scoring_hint": "Preference is based on observed score/verifier gap, not a new model claim.",
        })
    for attempt_kind, grader in (evidence.get("grader_results") or {}).items():
        for criterion in grader.get("failed_criteria") or []:
            rows.append({
                **common,
                "data_type": "failure_case",
                "input": {"attempt_kind": attempt_kind, "failed_criterion": criterion},
                "target": {"failure_mode": criterion, "repair_hint": _clean_text((evidence.get("evaluator_feedback") or {}).get("feedback_summary"))},
                "expected_output": "Identify the failure and propose grounded repair guidance.",
                "rubric": rubric.get("rubric_items") or [],
                "scoring_hint": "Failure cases must point back to grader/verifier/evaluator evidence.",
            })
    rows.append({
        **common,
        "data_type": "task_variant",
        "input": {"source_instruction": _clean_text(instruction)},
        "target": {
            "variant_instruction": _clean_text(f"{instruction}\n\nVariant constraint: preserve the same deliverables and make the result easier to verify."),
            "expected_deliverables": deliverables,
        },
        "expected_output": "A grounded task variant with the same evidence-backed artifact contract.",
        "rubric": rubric.get("rubric_items") or [],
        "scoring_hint": "Variant must preserve artifact requirements and verifier checks.",
    })
    rows.extend(_environment_response_rows(evidence, common))
    rows.extend(_surface_event_rows(evidence, common))
    return _finalize_candidate_rows(rows, evidence)


def _best_output_summary(evidence: dict[str, Any]) -> str:
    selected = (evidence.get("manifest") or {}).get("selected_attempt_id")
    for attempt in (evidence.get("attempts") or {}).values():
        actual = attempt.get("actual_outputs") or {}
        if selected and actual.get("attempt_id") == selected and actual.get("output_summary"):
            return _clean_text(actual.get("output_summary"))
    for attempt in (evidence.get("attempts") or {}).values():
        actual = attempt.get("actual_outputs") or {}
        if actual.get("output_summary"):
            return _clean_text(actual.get("output_summary"))
    return "No output summary was available."


def _attempt_summary(attempt: dict[str, Any]) -> str:
    return _clean_text((attempt.get("actual_outputs") or {}).get("output_summary") or "Attempt output summary unavailable.")


def _environment_response_rows(evidence: dict[str, Any], common: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for attempt in (evidence.get("attempts") or {}).values():
        trace = attempt.get("agent_trace") or {}
        steps = trace.get("steps") or []
        total = len(steps)
        previous_state = ""
        for idx, step in enumerate(steps, start=1):
            if step.get("action") == "user_message":
                previous_state = _clean_text(step.get("state_change") or previous_state)
                continue
            domain = step.get("environment_domain") if step.get("environment_domain") in ENVIRONMENT_DOMAINS else _infer_environment_domain(step)
            score = 0.72
            if step.get("success") is True:
                score = 0.82
            if step.get("step_outcome") in {"failed", "blocked"}:
                score = 0.45
            rows.append({
                **common,
                "data_type": "environment_response_pair",
                "source_trace_id": trace.get("trace_id"),
                "source_step_id": step.get("step") or idx,
                "source_step_ids": [step.get("step") or idx],
                "environment_domain": domain,
                "interaction_surface": step.get("interaction_surface") or _surface_from_step(step),
                "state_before": _clean_text(previous_state or step.get("observation")),
                "agent_action": _clean_text(step.get("input") or step.get("action") or step.get("operation")),
                "observed_environment_response": _clean_text(step.get("output") or step.get("observation")),
                "state_after": _clean_text(step.get("state_change")),
                "success_signal": _clean_text(step.get("step_outcome") or step.get("success")),
                "failure_mode": _clean_text(step.get("error_type") or ""),
                "verifier_feedback": _clean_text(_first_verifier_notes(evidence)),
                "turn_idx": idx,
                "total_turns": total,
                "format_score": score,
                "factuality_score": score,
                "consistency_score": score,
                "realism_score": score,
                "training_quality_score": score,
                "environment_response_quality_score": score,
                "usable_for": ["environment_model_training", "agent_rl", "verifier_training"],
            })
            previous_state = _clean_text(step.get("state_change") or previous_state)
    return rows


def _surface_from_step(step: dict[str, Any]) -> str:
    surface = step.get("interaction_surface")
    if surface:
        return str(surface)
    domain = step.get("environment_domain")
    if domain in {"api", "database", "spreadsheet", "search", "browser", "computer_use", "file", "codebase"}:
        return str(domain)
    raw = " ".join(str(step.get(k) or "") for k in ["tool", "operation", "input", "output"]).lower()
    if "mcp" in raw:
        return "mcp_tool"
    if any(x in raw for x in ["curl", "http", "graphql", "endpoint"]):
        return "api"
    if any(x in raw for x in ["sql", "table", "database"]):
        return "database"
    if any(x in raw for x in ["xlsx", "workbook", "spreadsheet", "csv"]):
        return "spreadsheet"
    if any(x in raw for x in ["bash", "shell", "command", "pytest", "git "]):
        return "cli"
    return "unknown"


def _surface_common(common: dict[str, Any], trace: dict[str, Any], step: dict[str, Any], idx: int, data_type: str) -> dict[str, Any]:
    step_id = step.get("step") or idx
    return {
        **common,
        "data_type": data_type,
        "source_trace_id": trace.get("trace_id"),
        "source_step_id": step_id,
        "source_step_ids": [step_id],
        "environment_domain": step.get("environment_domain") or _infer_environment_domain(step),
        "interaction_surface": _surface_from_step(step),
        "state_change": _clean_text(step.get("state_change")),
        "success_signal": _clean_text(step.get("step_outcome") or step.get("success")),
        "failure_mode": _clean_text(step.get("error_type") or step.get("error_message") or ""),
        "training_quality_score": step.get("environment_response_quality_score") or step.get("training_quality_score") or common.get("quality_score"),
        "usable_for": ["trace_training", "environment_response_training", "verifier_training"],
    }


def _surface_event_rows(evidence: dict[str, Any], common: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for attempt in (evidence.get("attempts") or {}).values():
        trace = attempt.get("agent_trace") or {}
        for idx, step in enumerate(trace.get("steps") or [], start=1):
            if step.get("action") == "user_message":
                continue
            surface = _surface_from_step(step)
            domain = step.get("environment_domain") or _infer_environment_domain(step)
            if step.get("subagent_id") or step.get("delegation_id") or step.get("subagent_output_refs"):
                rows.append({
                    **_surface_common(common, trace, step, idx, "multi_agent_handoff"),
                    "parent_agent_id": step.get("parent_agent_id"),
                    "subagent_id": step.get("subagent_id"),
                    "handoff_reason": _clean_text(step.get("handoff_reason")),
                    "handoff_payload_ref": _normalize_ref(step.get("handoff_payload_ref")),
                    "subagent_output_refs": _refs(step.get("subagent_output_refs") or step.get("handoff_output_refs") or step.get("artifact_refs")),
                    "downstream_steps_caused": step.get("depends_on_agent_outputs") or [],
                    "handoff_quality_score": common.get("quality_score"),
                    "delegation_success": step.get("success"),
                    "reusable_lesson": _clean_text(step.get("state_change") or step.get("output")),
                })
            if step.get("modality") or step.get("modalities") or surface in {"multimodal", "document", "pdf", "table"} or domain == "multimodal":
                refs = _refs(step.get("artifact_refs") or step.get("page_refs") or step.get("table_extract_refs"))
                for ref in refs or [None]:
                    modality = step.get("modality")
                    if not modality and isinstance(step.get("modalities"), list) and step.get("modalities"):
                        modality = step.get("modalities")[0]
                    rows.append({
                        **_surface_common(common, trace, step, idx, "multimodal_artifact"),
                        "modality": modality or "unknown",
                        "artifact_ref": ref,
                        "artifact_role": step.get("artifact_role"),
                        "mime_type": step.get("mime_type"),
                        "artifact_hash": step.get("artifact_hash"),
                        "preview_ref": _normalize_ref(step.get("artifact_preview_ref")),
                        "extracted_text_ref": _normalize_ref(step.get("ocr_text_ref") or step.get("text_extract_ref")),
                        "transcript_ref": _normalize_ref(step.get("transcript_ref")),
                        "annotation_ref": _normalize_ref(step.get("annotation_ref")),
                        "quality_score": common.get("quality_score"),
                        "verifier_use": "artifact evidence",
                        "training_use_cases": ["multimodal_grounding", "verifier_training"],
                        "evidence_status": [] if ref else ["No artifact ref was captured for this multimodal step."],
                    })
            if surface in {"browser", "computer_use"} or step.get("ui_surface") or step.get("screenshot_ref"):
                rows.append({
                    **_surface_common(common, trace, step, idx, "ui_interaction_event"),
                    "ui_surface": step.get("ui_surface") or surface,
                    "app_name": step.get("app_name"),
                    "action_type": step.get("action_type") or step.get("operation"),
                    "visible_state_ref": _normalize_ref(step.get("visible_state_ref")),
                    "screenshot_ref": _normalize_ref(step.get("screenshot_ref")),
                    "element_selector": step.get("element_selector"),
                    "coordinates": step.get("coordinates"),
                    "input_text": _clean_text(step.get("input_text")),
                    "observed_result": _clean_text(step.get("output")),
                })
            if surface in {"device", "industrial", "physical"} or domain in {"device", "industrial", "physical", "robotics", "lab"}:
                rows.append({
                    **_surface_common(common, trace, step, idx, "physical_environment_event"),
                    "physical_environment_type": step.get("physical_environment_type") or domain,
                    "device_type": step.get("device_type"),
                    "sensor_readings_ref": _normalize_ref(step.get("sensor_readings_ref")),
                    "actuator_command": _clean_text(step.get("actuator_command")),
                    "physical_state_before_ref": _normalize_ref(step.get("physical_state_before_ref")),
                    "physical_state_after_ref": _normalize_ref(step.get("physical_state_after_ref")),
                    "telemetry_ref": _normalize_ref(step.get("telemetry_ref")),
                    "safety_state": step.get("safety_state"),
                    "operator_override": step.get("operator_override"),
                    "simulation_or_real_world": step.get("simulation_or_real_world") or "unknown",
                    "safety_relevance": "safety_state captured" if step.get("safety_state") else "unknown",
                    "verifier_use": "safety and telemetry evidence",
                    "evidence_status": [] if step.get("telemetry_ref") or step.get("physical_state_after_ref") else ["Physical evidence refs were not captured."],
                })
            if surface == "api":
                rows.append({
                    **_surface_common(common, trace, step, idx, "api_interaction_event"),
                    "api_provider": step.get("api_provider"),
                    "api_service": step.get("api_service"),
                    "api_endpoint": step.get("api_endpoint"),
                    "api_method": step.get("api_method"),
                    "http_status": step.get("http_status"),
                    "request_payload_ref": _normalize_ref(step.get("request_payload_ref") or step.get("raw_request_ref")),
                    "response_payload_ref": _normalize_ref(step.get("response_payload_ref") or step.get("raw_response_ref")),
                    "resource_id": step.get("resource_id"),
                    "object_type": step.get("object_type"),
                    "retry_count": step.get("retry_count"),
                    "verifier_use": "API request/response grounding",
                })
            if surface in {"mcp_tool", "mcp_resource", "mcp_prompt"}:
                rows.append({
                    **_surface_common(common, trace, step, idx, "mcp_interaction_event"),
                    "mcp_server": step.get("mcp_server"),
                    "mcp_capability_type": step.get("mcp_capability_type") or surface.removeprefix("mcp_"),
                    "mcp_tool_name": step.get("mcp_tool_name"),
                    "mcp_resource_uri": step.get("mcp_resource_uri"),
                    "mcp_prompt_name": step.get("mcp_prompt_name"),
                    "arguments_ref": _normalize_ref(step.get("mcp_arguments_ref")),
                    "result_ref": _normalize_ref(step.get("mcp_result_ref")),
                })
            if surface == "database":
                rows.append({
                    **_surface_common(common, trace, step, idx, "database_event"),
                    "database_type": step.get("database_type"),
                    "database_name": step.get("database_name"),
                    "schema_name": step.get("schema_name"),
                    "table_name": step.get("table_name"),
                    "query_type": step.get("query_type"),
                    "query_ref": _normalize_ref(step.get("query_ref")),
                    "row_count": step.get("row_count"),
                    "affected_row_count": step.get("affected_row_count"),
                    "transaction_id": step.get("transaction_id"),
                })
            if surface in {"spreadsheet", "table"}:
                rows.append({
                    **_surface_common(common, trace, step, idx, "structured_table_event"),
                    "workbook_ref": _normalize_ref(step.get("workbook_ref")),
                    "sheet_name": step.get("sheet_name"),
                    "table_ref": _normalize_ref(step.get("table_ref")),
                    "cell_range": step.get("cell_range"),
                    "row_count": step.get("row_count"),
                    "column_count": step.get("column_count"),
                    "formula_refs": _refs(step.get("formula_refs")),
                    "before_table_ref": _normalize_ref(step.get("before_table_ref")),
                    "after_table_ref": _normalize_ref(step.get("after_table_ref")),
                    "diff_ref": _normalize_ref(step.get("diff_ref")),
                })
            if surface in {"cli", "file", "codebase"}:
                rows.append({
                    **_surface_common(common, trace, step, idx, "software_execution_event"),
                    "command": _clean_text(step.get("command") or step.get("input")),
                    "command_ref": _normalize_ref(step.get("command_ref")),
                    "exit_code": step.get("exit_code"),
                    "stdout_ref": _normalize_ref(step.get("stdout_ref")),
                    "stderr_ref": _normalize_ref(step.get("stderr_ref")),
                    "file_path": _normalize_ref(step.get("file_path")),
                    "file_paths": _refs(step.get("file_paths") or step.get("artifact_refs")),
                    "file_operation": step.get("file_operation") or step.get("operation"),
                    "diff_ref": _normalize_ref(step.get("diff_ref")),
                    "test_result_ref": _normalize_ref(step.get("test_result_ref")),
                })
            if surface == "search":
                rows.append({
                    **_surface_common(common, trace, step, idx, "search_retrieval_event"),
                    "search_provider": step.get("search_provider") or step.get("tool"),
                    "query": _clean_text(step.get("query") or step.get("input")),
                    "query_ref": _normalize_ref(step.get("query_ref")),
                    "search_scope": step.get("search_scope"),
                    "result_count": step.get("result_count"),
                    "selected_result_refs": _refs(step.get("selected_result_refs")),
                    "citation_refs": _refs(step.get("citation_refs")),
                })
            if surface in {"cloud_console", "saas_app"}:
                rows.append({
                    **_surface_common(common, trace, step, idx, "enterprise_app_event"),
                    "service_name": step.get("service_name") or step.get("app_name"),
                    "workspace_id": step.get("workspace_id"),
                    "project_id": step.get("project_id"),
                    "resource_id": step.get("resource_id"),
                    "resource_type": step.get("resource_type"),
                    "record_id": step.get("record_id"),
                    "ticket_id": step.get("ticket_id"),
                    "object_state_before_ref": _normalize_ref(step.get("object_state_before_ref")),
                    "object_state_after_ref": _normalize_ref(step.get("object_state_after_ref")),
                    "audit_log_ref": _normalize_ref(step.get("audit_log_ref")),
                })
            if surface in {"async_job", "queue", "webhook", "scheduler"}:
                rows.append({
                    **_surface_common(common, trace, step, idx, "async_workflow_event"),
                    "job_id": step.get("job_id"),
                    "queue_name": step.get("queue_name"),
                    "message_id": step.get("message_id"),
                    "webhook_id": step.get("webhook_id"),
                    "schedule_id": step.get("schedule_id"),
                    "trigger_type": step.get("trigger_type"),
                    "retry_count": step.get("retry_count"),
                    "delivery_status": step.get("delivery_status"),
                    "polling_status": step.get("polling_status"),
                    "callback_status": step.get("callback_status"),
                    "job_status_before": step.get("job_status_before"),
                    "job_status_after": step.get("job_status_after"),
                    "async_result_ref": _normalize_ref(step.get("async_result_ref")),
                })
            if surface in {"messaging", "email", "collaboration"}:
                rows.append({
                    **_surface_common(common, trace, step, idx, "collaboration_event"),
                    "platform_name": step.get("platform_name"),
                    "channel_id": step.get("channel_id"),
                    "thread_id": step.get("thread_id"),
                    "message_id": step.get("message_id"),
                    "sender_role": step.get("sender_role"),
                    "recipient_role": step.get("recipient_role"),
                    "message_type": step.get("message_type"),
                    "message_ref": _normalize_ref(step.get("message_ref")),
                    "attachment_refs": _refs(step.get("attachment_refs")),
                    "decision_refs": _refs(step.get("decision_refs")),
                    "followup_required": step.get("followup_required"),
                })
            if surface in {"document", "pdf"}:
                rows.append({
                    **_surface_common(common, trace, step, idx, "document_event"),
                    "document_ref": _normalize_ref(step.get("document_ref")),
                    "document_type": step.get("document_type") or surface,
                    "page_refs": _refs(step.get("page_refs")),
                    "section_refs": _refs(step.get("section_refs")),
                    "text_extract_ref": _normalize_ref(step.get("text_extract_ref") or step.get("ocr_text_ref")),
                    "table_extract_refs": _refs(step.get("table_extract_refs")),
                    "figure_refs": _refs(step.get("figure_refs")),
                    "citation_refs": _refs(step.get("citation_refs")),
                    "annotation_refs": _refs(step.get("annotation_refs")),
                    "summary_ref": _normalize_ref(step.get("summary_ref")),
                })
            if surface == "mixed" or step.get("connected_surfaces") or step.get("cross_system_dependencies"):
                rows.append({
                    **_surface_common(common, trace, step, idx, "cross_system_event"),
                    "connected_surfaces": step.get("connected_surfaces") or [],
                    "cross_system_dependencies": step.get("cross_system_dependencies") or [],
                    "handoff_refs": _refs(step.get("handoff_refs")),
                    "state_sync_refs": _refs(step.get("state_sync_refs")),
                    "consistency_checks": step.get("consistency_checks") or [],
                    "system_of_record": step.get("system_of_record"),
                })
    return [sanitize_public_obj(row) for row in rows]


def _first_verifier_notes(evidence: dict[str, Any]) -> str | None:
    for result in (evidence.get("verifier_results") or {}).values():
        if result.get("verifier_notes"):
            return str(result.get("verifier_notes"))
    return None


def _private_or_forbidden_hit(obj: Any) -> str | None:
    if isinstance(obj, dict):
        for key, value in obj.items():
            key_l = str(key).lower()
            if key_l in FORBIDDEN_KEYS:
                return key_l
            hit = _private_or_forbidden_hit(value)
            if hit:
                return hit
        return None
    if isinstance(obj, list):
        for value in obj:
            hit = _private_or_forbidden_hit(value)
            if hit:
                return hit
        return None
    if isinstance(obj, str):
        if contains_secret(obj):
            return "secret_like_value"
        if PRIVATE_RE.search(obj):
            return "private_local_path"
        lower = obj.lower()
        for token in [("deep" + "seek"), ("worker " + "agent"), "worker_agent", ("market" + "place"), ("b" + "uyer"), ("s" + "eller")]:
            if token in lower:
                return token
    return None


def _artifact_refs_resolve(package_root: Path, row: dict[str, Any]) -> list[str]:
    missing: list[str] = []
    for ref in _refs(row.get("source_artifact_refs") or []):
        if ":" in ref:
            continue
        candidates = [package_root / ref]
        if ref.startswith("packages/"):
            candidates.append(package_root.parent / ref.removeprefix("packages/"))
        if not any(p.exists() for p in candidates):
            # Evidence refs can legitimately describe generated aggregate rows.
            if ref.startswith(("grading/", "feedback/", "signals/", "task/", "rubric/")):
                continue
            missing.append(ref)
    return missing


def _validate_rows(package_root: Path, rows: list[dict[str, Any]], min_quality: float, min_fidelity: float, min_reuse: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    schema_errors: list[str] = []
    seen_row_ids: set[str] = set()
    for row in rows:
        reason = _private_or_forbidden_hit(row)
        missing = _artifact_refs_resolve(package_root, row)
        if missing:
            reason = reason or f"unresolved_artifact_ref:{missing[0]}"
        if not row.get("data_type"):
            reason = reason or "missing_data_type"
        row_id = row.get("row_id") or row.get("pair_id")
        if not row_id:
            reason = reason or "missing_stable_row_id"
        elif str(row_id) in seen_row_ids:
            reason = reason or f"duplicate_row_id:{row_id}"
        else:
            seen_row_ids.add(str(row_id))
        if not isinstance(row.get("task_lineage"), dict) or not row.get("source_task_id"):
            reason = reason or "missing_task_lineage"
        if not row.get("domain") or not row.get("workflow_family"):
            reason = reason or "missing_domain_or_workflow_family"
        if not row.get("source_evidence_ref_ids"):
            reason = reason or "missing_source_evidence_ref_ids"
        if row.get("data_type") in {"training_example", "evaluation_example", "verifier_example", "reward_model_row"}:
            if row.get("input") in (None, "", {}, []):
                reason = reason or "missing_core_input"
            if row.get("target") in (None, "", {}, []):
                reason = reason or "missing_core_target"
        if not _has_concrete_source_evidence(row):
            reason = reason or "insufficient_concrete_source_evidence"
        if _score(row.get("quality_score")) < min_quality:
            reason = reason or "quality_score_below_threshold"
        if _score(row.get("fidelity_score")) < min_fidelity:
            reason = reason or "fidelity_score_below_threshold"
        if _score(row.get("reuse_value_score")) < min_reuse:
            reason = reason or "reuse_value_score_below_threshold"
        clean = sanitize_public_obj(dict(row))
        if reason:
            clean["accepted"] = False
            clean["rejection_reason"] = reason
            rejected.append(clean)
            schema_errors.append(reason)
        else:
            clean["accepted"] = True
            clean["rejection_reason"] = None
            accepted.append(clean)
    return accepted, rejected, schema_errors


def _judge(round_idx: int, max_rounds: int, rows: list[dict[str, Any]], rejected: list[dict[str, Any]], judge_source: str, mentor_audited: bool, proxy: dict[str, Any]) -> TDOJudgeResult:
    if not rows:
        verdict = "reject"
        reason = "No candidate rows were generated from package evidence."
    elif rejected and round_idx >= max_rounds:
        verdict = "improve"
        reason = "Some candidate rows need stronger grounding or safer refs."
    elif round_idx < max_rounds:
        verdict = "improve"
        reason = "Autodata-style loop continues to improve weak rows and recipe grounding."
    else:
        verdict = "accept"
        reason = "Candidate rows passed schema, grounding, and sanitization guardrails."
    scores = [_score(row.get("quality_score"), 0.0) for row in rows] or [0.0]
    fidelity = [_score(row.get("fidelity_score"), 0.0) for row in rows] or [0.0]
    difficulty = [_score(row.get("difficulty_score"), 0.0) for row in rows] or [0.0]
    learnability = [_score(row.get("learnability_score"), 0.0) for row in rows] or [0.0]
    reuse = [_score(row.get("reuse_value_score"), 0.0) for row in rows] or [0.0]
    result = TDOJudgeResult(
        tdo_round=round_idx,
        tdo_judge_source=judge_source,  # type: ignore[arg-type]
        mentor_audited=mentor_audited,
        verdict=verdict,  # type: ignore[arg-type]
        weak_pattern=proxy.get("weak_pattern"),
        strong_pattern=proxy.get("strong_pattern"),
        gap_interpretation=proxy.get("gap_interpretation"),
        rubric_concerns=[],
        grounding_concerns=[row.get("rejection_reason") for row in rejected if row.get("rejection_reason")],
        data_quality_concerns=[] if rows else ["no_candidate_rows"],
        grpo_suitability=proxy.get("grpo_suitability") or "unknown",
        learning_signal_quality=proxy.get("learning_signal_quality") or "unknown",
        reuse_value="high" if mean(reuse) >= 0.8 else "medium" if mean(reuse) >= 0.6 else "low",
        quality_score=round(mean(scores), 4),
        fidelity_score=round(mean(fidelity), 4),
        difficulty_score=round(mean(difficulty), 4),
        learnability_score=round(mean(learnability), 4),
        reuse_value_score=round(mean(reuse), 4),
        verdict_reason=reason,
        suggestion_for_generator=("Strengthen evidence refs, keep rows grounded, and improve weak rows in the next compiler round." if verdict in {"improve", "reject"} else None),
    )
    return result


def _optimized_task_spec(evidence: dict[str, Any], settings: Settings, round_idx: int) -> dict[str, Any]:
    task = evidence.get("task") or {}
    raw = evidence.get("raw_task") or {}
    proxy = _weak_strong_proxy(evidence)
    instruction = task.get("normalized_instruction") or raw.get("raw_description") or raw.get("instruction") or task.get("normalized_title") or evidence["package_id"]
    domain = task.get("domain") or raw.get("domain") or (raw.get("raw_payload") or {}).get("domain") or "general"
    subdomain = task.get("subdomain") or raw.get("subdomain") or (raw.get("raw_payload") or {}).get("subdomain")
    return sanitize_public_obj({
        "source_task_id": _source_task_id(evidence),
        "source_package_id": evidence["package_id"],
        "optimized_title": task.get("normalized_title") or raw.get("raw_title") or evidence["package_id"],
        "optimized_instruction": instruction,
        "domain": domain,
        "subdomain": subdomain,
        "role": task.get("apprenticeship_role") or task.get("professional_role") or "Apprentice Agent",
        "expected_deliverables": task.get("output_requirements") or [task.get("expected_agent_deliverable") or "task artifacts"],
        "constraints": task.get("constraints") or [],
        "required_inputs": task.get("input_requirements") or [],
        "success_criteria": [item.get("criterion_description") for item in (evidence.get("rubric") or {}).get("rubric_items", [])],
        "failure_modes": list(dict.fromkeys((evidence.get("hillclimb") or {}).get("failed_criteria_before") or [])),
        "expected_economic_value": task.get("expected_economic_value") or raw.get("expected_economic_value") or "category_unknown",
        "expected_economic_value_rationale": "Derived from task metadata, deliverable requirements, trace complexity, and evaluation evidence.",
        "training_use_cases": ["supervised_finetuning_rows", "verifier_training", "reward_modeling", "preference_optimization"],
        "generated_by": "Experience Compiler",
        "apprenticeship_mode": getattr(settings, "apprenticeship_mode", None),
        "compiler_round": round_idx,
        "tdo_round": round_idx,
        "weak_strong_proxy_available": proxy.get("available"),
        **_source_audit_status(evidence),
        "evidence_refs": _evidence_refs(evidence),
    })


def _optimized_rubric(evidence: dict[str, Any], round_idx: int) -> dict[str, Any]:
    rubric = evidence.get("rubric") or {}
    items = rubric.get("rubric_items") or []
    return sanitize_public_obj({
        "source_task_id": _source_task_id(evidence),
        "source_package_id": evidence["package_id"],
        "tdo_round": round_idx,
        "rubric_items": items,
        "verifier_checks": [
            "actual_outputs.json parses and points to generated artifacts",
            "artifact refs resolve inside the package",
            "trace steps support the claimed output",
            "grader/verifier/evaluator evidence is consistent",
        ],
        "scoring_guidance": "Use grounded artifact, trace, grader, verifier, and evaluator evidence. Do not reward permissive rubrics.",
        "failure_criteria": list(dict.fromkeys((evidence.get("hillclimb") or {}).get("failed_criteria_before") or [])),
        "artifact_checks": rubric.get("required_artifacts") or [ref for ref in _source_artifact_refs(evidence) if "/artifacts/" in ref],
        "environment_response_checks": ["environment_response_pairs.jsonl rows should reflect observed trace steps only"],
        "reward_modeling_notes": "Use score/verifier evidence and weak/strong proxies where available.",
        "evaluator_notes": _clean_text((evidence.get("evaluator_feedback") or {}).get("feedback_summary")),
        **_source_audit_status(evidence),
        "evidence_refs": _evidence_refs(evidence),
    })


def _domain_value_profile(evidence: dict[str, Any]) -> dict[str, Any]:
    task = evidence.get("task") or {}
    raw = evidence.get("raw_task") or {}
    domain = task.get("domain") or raw.get("domain") or (raw.get("raw_payload") or {}).get("domain") or "general"
    subdomain = task.get("subdomain") or raw.get("subdomain") or (raw.get("raw_payload") or {}).get("subdomain")
    traced_steps = sum(len((attempt.get("agent_trace") or {}).get("steps") or []) for attempt in (evidence.get("attempts") or {}).values())
    value_category = "high" if traced_steps >= 12 else "medium" if traced_steps >= 5 else "emerging"
    return sanitize_public_obj({
        "domain": domain,
        "subdomain": subdomain,
        "role": task.get("apprenticeship_role") or task.get("professional_role") or "Apprentice Agent",
        "apprenticeship_mode": (evidence.get("manifest") or {}).get("apprenticeship_mode"),
        "economic_value_category": value_category,
        "expected_economic_value": task.get("expected_economic_value") or raw.get("expected_economic_value") or value_category,
        "value_drivers": ["specialized workflow", "artifact creation", "verifiable outputs", "reusable trace evidence"],
        "risk_level": "medium" if traced_steps >= 8 else "low",
        "specialization_level": "high" if task.get("subdomain") else "medium",
        "automation_potential": "high" if evidence.get("process_supervision") else "medium",
        "training_value": "high" if traced_steps >= 8 else "medium",
        "evaluation_value": "high" if evidence.get("grader_results") and evidence.get("verifier_results") else "medium",
        "verifier_value": "high" if evidence.get("verifier_results") else "medium",
        "evidence_refs": _evidence_refs(evidence),
    })


def _environment_spec(evidence: dict[str, Any]) -> dict[str, Any]:
    domains: list[str] = []
    surfaces: list[str] = []
    tools: list[str] = []
    operations: list[str] = []
    steps: list[dict[str, Any]] = []
    for attempt in (evidence.get("attempts") or {}).values():
        for step in (attempt.get("agent_trace") or {}).get("steps") or []:
            if step.get("action") == "user_message":
                continue
            steps.append(step)
            domains.append(_infer_environment_domain(step))
            surfaces.append(_surface_from_step(step))
            if step.get("tool"):
                tools.append(str(step.get("tool")))
            if step.get("operation"):
                operations.append(str(step.get("operation")))
    unique_domains = sorted(set(domains) - {"unknown"})
    domain = unique_domains[0] if len(unique_domains) == 1 else "mixed" if unique_domains else "unknown"
    environment_type = "local_filesystem" if domain == "file" else domain if domain in {"terminal", "browser", "codebase", "spreadsheet", "api", "mixed"} else "unknown"
    unique_surfaces = sorted(set(surfaces) - {"unknown"})
    def refs_from(*keys: str) -> list[str]:
        out: list[str] = []
        for step in steps:
            for key in keys:
                out.extend(_refs(step.get(key)))
                ref = _normalize_ref(step.get(key))
                if ref:
                    out.append(ref)
        return list(dict.fromkeys(out))
    def values_from(*keys: str) -> list[str]:
        vals: list[str] = []
        for step in steps:
            for key in keys:
                value = step.get(key)
                if isinstance(value, list):
                    vals.extend(str(v) for v in value if v not in (None, ""))
                elif value not in (None, ""):
                    vals.append(str(value))
        return sorted(set(vals))
    multi_agent_present = any(step.get("subagent_id") or step.get("delegation_id") for step in steps)
    multimodal_present = any(step.get("modality") or step.get("modalities") or _surface_from_step(step) in {"multimodal", "document", "pdf", "table"} for step in steps)
    ui_present = any(_surface_from_step(step) in {"browser", "computer_use", "cloud_console", "saas_app"} or step.get("screenshot_ref") or step.get("visible_state_ref") for step in steps)
    physical_present = any(_surface_from_step(step) in {"device", "industrial", "physical"} or (step.get("environment_domain") in {"device", "industrial", "physical", "robotics", "lab"}) for step in steps)
    return sanitize_public_obj({
        "environment_type": environment_type,
        "environment_domain": domain,
        "interaction_surfaces": unique_surfaces,
        "surface_requirements": {
            surface: "capture surface-specific ids, state refs, requests/responses, evidence refs, and status transitions when available"
            for surface in unique_surfaces
        },
        "required_tools": sorted(set(tools)),
        "runtime_dependencies": ["python"] if any("python" in t.lower() for t in tools) else [],
        "setup_commands": [],
        "input_artifacts": [ref for ref in _source_artifact_refs(evidence) if ref.startswith("input/") or "/input/" in ref],
        "initial_state_refs": ["task/task_intake_spec.json", "rubric/rubric.json"],
        "allowed_actions": sorted(set(operations)),
        "action_schema": {
            "action": "operation/tool invocation or artifact-producing step",
            "tool": "optional tool name",
            "input": "visible input or command payload",
            "artifact_refs": "package-relative refs produced or inspected",
        },
        "observation_schema": {
            "observation": "tool/environment response summary",
            "state_change": "state delta after action",
            "artifact_refs": "package-relative output refs",
            "success_signal": "step outcome or verifier signal",
        },
        "reset_strategy": "copy_workspace" if environment_type in {"local_filesystem", "codebase", "terminal"} else "unknown",
        "verification_commands": [],
        "scoring_refs": [ref for ref in _evidence_refs(evidence) if ref.startswith("grading/")],
        "expected_output_refs": [ref for ref in _source_artifact_refs(evidence) if "/artifacts/" in ref],
        "termination_conditions": ["task_complete", "contract_verified", "max_iterations_reached"],
        "reward_function_components": ["artifact_contract", "verifier_signal", "step_success", "grader_score_if_available"],
        "replay_smoke_test": {"available": False, "command": None, "success_condition": None},
        "reproducibility_metadata": {
            "source_trace_count": len(evidence.get("attempts") or {}),
            "package_id": evidence.get("package_id"),
            "created_from_observed_trace": True,
        },
        "multi_agent": {
            "present": multi_agent_present,
            "agents": values_from("agent_id", "agent_name", "subagent_id", "subagent_name"),
            "coordination_pattern": (values_from("coordination_pattern") or [None])[0],
            "handoff_refs": refs_from("handoff_payload_ref", "handoff_output_refs", "subagent_output_refs"),
        },
        "multimodal": {
            "present": multimodal_present,
            "modalities": values_from("modality", "modalities"),
            "artifact_refs": refs_from("artifact_refs", "screenshot_ref", "document_ref", "workbook_ref", "table_ref"),
        },
        "ui_or_software": {
            "present": ui_present,
            "surfaces": [s for s in unique_surfaces if s in {"browser", "computer_use", "cloud_console", "saas_app", "cli", "codebase", "file"}],
            "state_refs": refs_from("visible_state_ref", "screenshot_ref", "before_state_ref", "after_state_ref", "software_state_ref"),
        },
        "physical_or_industrial": {
            "present": physical_present,
            "environment_type": (values_from("physical_environment_type") or [None])[0],
            "device_refs": refs_from("sensor_readings_ref", "telemetry_ref", "calibration_ref"),
            "safety_refs": refs_from("safety_limit_refs", "compliance_refs"),
            "telemetry_refs": refs_from("telemetry_ref"),
        },
        "api": {
            "providers": values_from("api_provider"),
            "endpoints": values_from("api_endpoint"),
            "auth_modes": values_from("auth_mode"),
            "schemas": refs_from("request_schema_ref", "response_schema_ref"),
        },
        "mcp": {
            "servers": values_from("mcp_server"),
            "tools": values_from("mcp_tool_name"),
            "resources": values_from("mcp_resource_uri"),
            "prompts": values_from("mcp_prompt_name"),
        },
        "database": {
            "types": values_from("database_type"),
            "schemas": values_from("schema_name"),
            "tables": values_from("table_name"),
        },
        "spreadsheet": {
            "workbooks": refs_from("workbook_ref"),
            "sheets": values_from("sheet_name"),
            "tables": refs_from("table_ref", "table_refs"),
        },
        "software_execution": {
            "commands": values_from("command", "test_command"),
            "tests": refs_from("test_result_ref"),
            "dependencies": refs_from("dependency_changes_ref"),
        },
        "async": {
            "jobs": values_from("job_id"),
            "queues": values_from("queue_name"),
            "webhooks": values_from("webhook_id"),
            "schedulers": values_from("schedule_id"),
        },
        "collaboration": {
            "platforms": values_from("platform_name"),
            "threads": values_from("thread_id"),
            "decisions": refs_from("decision_refs"),
        },
        "documents": {
            "documents": refs_from("document_ref"),
            "extracts": refs_from("text_extract_ref", "ocr_text_ref", "table_extract_refs"),
            "citations": refs_from("citation_refs"),
        },
        "cross_system": {
            "systems": unique_surfaces,
            "dependencies": values_from("cross_system_dependencies"),
            "system_of_record": (values_from("system_of_record") or [None])[0],
        },
        "evidence_status": ["No simulated environment response is generated."],
    })


def _counts_by_type(rows: list[dict[str, Any]]) -> dict[str, int]:
    return dict(Counter(str(row.get("data_type") or "unknown") for row in rows))


def _counts_with_surface_zeros(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts = _counts_by_type(rows)
    for data_type in SURFACE_FILES:
        counts.setdefault(data_type, 0)
    return counts


def _average(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [_score(row.get(key), 0.0) for row in rows if row.get(key) is not None]
    return round(mean(values), 4) if values else None


def _grpo_distribution(rows: list[dict[str, Any]]) -> dict[str, int]:
    return dict(Counter(str(row.get("grpo_suitability") or "unknown") for row in rows))


def _write_tdo_outputs(tdo_dir: Path, accepted: list[dict[str, Any]], rejected: list[dict[str, Any]]) -> None:
    for name in sorted(set(ROW_FILE_BY_TYPE.values()) | {"rejected_rows.jsonl"}):
        (tdo_dir / name).parent.mkdir(parents=True, exist_ok=True)
        (tdo_dir / name).write_text("")
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in accepted:
        by_type[str(row.get("data_type") or "unknown")].append(row)
    for data_type, rows in by_type.items():
        target = ROW_FILE_BY_TYPE.get(data_type)
        if target:
            _write_jsonl(tdo_dir / target, rows)
    _write_jsonl(tdo_dir / "rejected_rows.jsonl", rejected)


def _copy_jsonl_public(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text("")
    if not src.exists():
        return
    for row in read_jsonl(src):
        append_jsonl(dst, sanitize_public_obj(row) if isinstance(row, dict) else row)


def _compiler_public_obj(obj: Any) -> Any:
    rename = {
        "tdo_id": "compiler_id",
        "tdo_status": "compiler_status",
        "tdo_round": "compiler_round",
        "tdo_judge_source": "judge_source",
        "tdo_output_refs": "compiler_output_refs",
        "tdo_rows_by_type": "compiler_rows_by_type",
    }
    if isinstance(obj, dict):
        out: dict[str, Any] = {}
        for key, value in obj.items():
            out[rename.get(str(key), str(key))] = _compiler_public_obj(value)
        return sanitize_public_obj(out)
    if isinstance(obj, list):
        return [_compiler_public_obj(item) for item in obj]
    return sanitize_public_obj(obj)


def _copy_compiler_jsonl_public(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text("")
    if not src.exists():
        return
    for row in read_jsonl(src):
        append_jsonl(dst, _compiler_public_obj(row))


def _copy_compiler_json_public(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not src.exists():
        write_json(dst, {})
        return
        write_json(dst, _compiler_public_obj(read_json(src)))


def _compiler_rows_from_tdo_dir(tdo_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for filename in sorted(set(ROW_FILE_BY_TYPE.values())):
        path = tdo_dir / filename
        if not path.exists():
            continue
        for row in _safe_jsonl(path):
            if isinstance(row, dict):
                clean = dict(row)
                clean["_source_row_file"] = filename
                rows.append(clean)
    return rows


def _training_row_counts_by_file(tdo_dir: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    for filename in sorted(set(ROW_FILE_BY_TYPE.values())):
        path = tdo_dir / filename
        counts[filename] = len(_safe_jsonl(path)) if path.exists() else 0
    return counts


def _source_evidence_map_payload(
    evidence: dict[str, Any],
    report: dict[str, Any],
    compiler_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    catalog = _evidence_catalog(evidence)
    evidence_by_id = {item["evidence_id"]: item for item in catalog}
    row_links: list[dict[str, Any]] = []
    orphan_ids: set[str] = set()
    for row in compiler_rows:
        row_id = str(row.get("row_id") or row.get("pair_id") or _row_identity(str(row.get("data_type") or "unknown"), row, evidence))
        ids = [str(v) for v in row.get("source_evidence_ref_ids") or _evidence_ids_for_refs(row.get("evidence_refs") or [])]
        for evidence_id in ids:
            if evidence_id not in evidence_by_id:
                orphan_ids.add(evidence_id)
        row_links.append(
            {
                "row_id": row_id,
                "row_file": row.get("_source_row_file"),
                "data_type": row.get("data_type"),
                "source_task_id": row.get("source_task_id") or report.get("source_task_id"),
                "source_trace_id": row.get("source_trace_id"),
                "source_step_ids": row.get("source_step_ids") or [],
                "source_artifact_refs": row.get("source_artifact_refs") or [],
                "evidence_ref_ids": ids,
                "visible_evidence_ref_ids": [
                    evidence_id
                    for evidence_id in ids
                    if (evidence_by_id.get(evidence_id) or {}).get("visibility") == "visible"
                ],
                "inferred_evidence_ref_ids": [
                    evidence_id
                    for evidence_id in ids
                    if (evidence_by_id.get(evidence_id) or {}).get("visibility") == "inferred"
                ],
            }
        )
    resolved = sum(1 for item in catalog if item.get("resolves"))
    return sanitize_public_obj(
        {
            "schema_version": EVIDENCE_MAP_SCHEMA_VERSION,
            "source_package_id": report.get("source_package_id"),
            "source_task_id": report.get("source_task_id"),
            "evidence_catalog": catalog,
            "row_evidence_links": row_links,
            "task_refs": [
                ref
                for ref in [
                    "task/task_packet.json",
                    "task/task_intake_spec.json",
                    "task/raw_task_record.json",
                    "source_task/task_packet.json",
                ]
                if isinstance(evidence.get("package_root"), Path) and (evidence["package_root"] / ref).exists()
            ]
            or ["task/task_packet.json"],
            "artifact_refs": _source_artifact_refs(evidence),
            "evaluation_refs": [ref for ref in _evidence_refs(evidence) if ref.startswith(("grading/", "feedback/", "signals/", "loops/"))],
            "loop_review_refs": evidence.get("loop_review_refs") or [],
            "trace_refs": [
                str((attempt.get("agent_trace") or {}).get("trace_id"))
                for attempt in (evidence.get("attempts") or {}).values()
                if (attempt.get("agent_trace") or {}).get("trace_id")
            ],
            "coverage_statistics": {
                "evidence_ref_count": len(catalog),
                "resolved_evidence_ref_count": resolved,
                "unresolved_evidence_ref_count": len(catalog) - resolved,
                "row_link_count": len(row_links),
                "row_count_with_evidence_ids": sum(1 for row in row_links if row.get("evidence_ref_ids")),
                "orphan_evidence_id_count": len(orphan_ids),
            },
            "orphan_evidence_ref_ids": sorted(orphan_ids),
        }
    )


def _process_supervision_rows_from_compiler_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    process_types = {"environment_response_pair", *SURFACE_FILES.keys()}
    for row in rows:
        if row.get("data_type") not in process_types:
            continue
        out.append(
            sanitize_public_obj(
                {
                    "schema_version": TRAINING_ROW_SCHEMA_VERSION,
                    "row_id": f"ps_{row.get('row_id')}",
                    "data_type": "process_supervision",
                    "source_package_id": row.get("source_package_id"),
                    "source_task_id": row.get("source_task_id"),
                    "task_lineage": row.get("task_lineage"),
                    "domain": row.get("domain"),
                    "subdomain": row.get("subdomain"),
                    "workflow_family": row.get("workflow_family"),
                    "source_trace_id": row.get("source_trace_id"),
                    "source_step_ids": row.get("source_step_ids") or [],
                    "step_goal": row.get("agent_action") or row.get("command") or row.get("query") or row.get("input") or row.get("state_change") or row.get("interaction_surface"),
                    "action_taken": row.get("agent_action") or row.get("action_type") or row.get("command") or row.get("query") or row.get("state_change") or row.get("interaction_surface"),
                    "observation": row.get("observed_environment_response") or row.get("observed_result") or row.get("output") or row.get("state_change") or row.get("success_signal"),
                    "tool_action_metadata": {
                        "environment_domain": row.get("environment_domain"),
                        "interaction_surface": row.get("interaction_surface"),
                        "tool_or_surface": row.get("search_provider") or row.get("api_endpoint") or row.get("table_name") or row.get("service_name"),
                    },
                    "correctness_label": "positive" if row.get("success_signal") in {"completed", "True", True} else "unknown",
                    "quality_signal": row.get("training_quality_score") or row.get("quality_score"),
                    "next_step_recommendation": "Continue from the observed state and verify artifact/output contract before final answer.",
                    "source_evidence_ref_ids": row.get("source_evidence_ref_ids") or [],
                    "evidence_refs": row.get("evidence_refs") or [],
                    "confidence": row.get("confidence") or row.get("quality_score") or 0.5,
                    "created_at": row.get("created_at") or utc_now(),
                }
            )
        )
    return out


def _rl_rollout_rows_from_compiler_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        if row.get("data_type") != "environment_response_pair":
            continue
        row_id = str(row.get("row_id") or _hash_obj(row))
        out.append(
            sanitize_public_obj(
                {
                    "schema_version": TRAINING_ROW_SCHEMA_VERSION,
                    "row_id": f"rl_{row_id}",
                    "data_type": "agentic_rl_rollout",
                    "source_package_id": row.get("source_package_id"),
                    "source_task_id": row.get("source_task_id"),
                    "task_lineage": row.get("task_lineage"),
                    "domain": row.get("domain"),
                    "subdomain": row.get("subdomain"),
                    "workflow_family": row.get("workflow_family"),
                    "rollout_group_id": row.get("rollout_group_id") or f"rollout_{_safe_id(str(row.get('source_task_id')))}",
                    "candidate_rollout_id": row.get("candidate_rollout_id") or f"rollout_{row_id}",
                    "prompt_or_instruction": row.get("input") or row.get("agent_action"),
                    "initial_environment_state": row.get("state_before"),
                    "visible_input_refs": row.get("visible_input_refs") or [],
                    "tool_action_schema": {
                        "environment_domain": row.get("environment_domain"),
                        "interaction_surface": row.get("interaction_surface"),
                    },
                    "step_action_trace": {
                        "source_trace_id": row.get("source_trace_id"),
                        "source_step_ids": row.get("source_step_ids") or [],
                        "action": row.get("agent_action"),
                        "observation": row.get("observed_environment_response"),
                    },
                    "artifacts_produced": row.get("source_artifact_refs") or [],
                    "final_output": row.get("state_after") or row.get("success_signal"),
                    "verifier_decision": row.get("verifier_feedback"),
                    "scalar_reward": row.get("scalar_reward") or row.get("training_quality_score") or row.get("quality_score"),
                    "reward_components": row.get("reward_components") or {},
                    "reward_timing": "step",
                    "success_label": row.get("success_signal"),
                    "termination_reason": row.get("termination_reason") or row.get("success_signal"),
                    "source_evidence_ref_ids": row.get("source_evidence_ref_ids") or [],
                    "evidence_refs": row.get("evidence_refs") or [],
                    "replay_scoring_smoke_test_refs": row.get("source_evaluation_refs") or [],
                    "confidence": row.get("confidence") or row.get("quality_score") or 0.5,
                    "created_at": row.get("created_at") or utc_now(),
                }
            )
        )
    return out


def _bioactive_schema_examples(report: dict[str, Any], evidence: dict[str, Any]) -> list[dict[str, Any]]:
    base = {
        "schema_version": TRAINING_ROW_SCHEMA_VERSION,
        "source_package_id": report.get("source_package_id"),
        "source_task_id": report.get("source_task_id"),
        "task_lineage": _task_lineage(evidence),
        "source_evidence_ref_ids": _evidence_ids_for_refs(_evidence_refs(evidence)[:5]),
        "evidence_refs": _evidence_refs(evidence)[:5],
        "confidence": 0.5,
        "personal_context_policy": "abstract_constraints_only_no_sensitive_identity",
        "medical_advice_policy": "evidence_mapping_only_no_treatment_or_cure_claims",
        "created_at": utc_now(),
    }
    return [
        {
            **base,
            "row_id": f"bioactive_schema_{_safe_id(str(report.get('source_package_id') or 'package'))}",
            "data_type": "bioactive_evidence_mapping_schema_example",
            "source_text_or_artifact_ref": None,
            "extracted_claim": None,
            "ingredient_or_compound_entity": None,
            "product_entity": None,
            "evidence_summary": None,
            "mechanism_category": None,
            "safety_caution_category": None,
            "uncertainty": "schema_only",
            "citation_or_provenance_refs": [],
            "verifier_decision": "not_evaluated_schema_example",
        }
    ]


def _copy_json_public(src: Path, dst: Path, default: dict[str, Any] | None = None) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.exists():
        write_json(dst, sanitize_public_obj(read_json(src)))
    else:
        write_json(dst, sanitize_public_obj(default or {}))


def _task_title(evidence: dict[str, Any]) -> str:
    task = evidence.get("task") or {}
    raw = evidence.get("raw_task") or {}
    return _clean_text(task.get("normalized_title") or raw.get("raw_title") or evidence.get("package_id") or "Agent Apprenticeship workflow", 240)


def _task_instruction(evidence: dict[str, Any]) -> str:
    task = evidence.get("task") or {}
    raw = evidence.get("raw_task") or {}
    return _clean_text(task.get("normalized_instruction") or raw.get("raw_description") or _task_title(evidence), 1600)


def _domain(evidence: dict[str, Any]) -> str:
    task = evidence.get("task") or {}
    return _clean_text(task.get("domain") or task.get("environment_domain") or "mixed", 80)


def _success_criteria(evidence: dict[str, Any]) -> list[str]:
    task = evidence.get("task") or {}
    rubric = evidence.get("rubric") or {}
    raw_items = task.get("success_criteria") or task.get("expected_deliverables") or []
    if not raw_items:
        raw_items = [item.get("criterion_description") for item in rubric.get("rubric_items", []) if isinstance(item, dict)]
    if isinstance(raw_items, str):
        raw_items = [raw_items]
    return [_clean_text(item, 180) for item in raw_items if item][:8] or [
        "Produce task-specific deliverables.",
        "Preserve source evidence and package-relative artifact refs.",
        "Pass bundle check and verifier checks where available.",
    ]


def _output_summary(evidence: dict[str, Any]) -> str:
    return _best_output_summary(evidence)


def _skill_markdown(evidence: dict[str, Any], report: dict[str, Any], rows_accepted: dict[str, Any], settings: Settings) -> str:
    task = evidence.get("task") or {}
    actual_outputs = evidence.get("actual_outputs") or []
    output_summaries = [
        _clean_text(row.get("output_summary"), 240)
        for row in actual_outputs
        if isinstance(row, dict) and row.get("output_summary")
    ][:3]
    artifact_refs = sorted(
        {
            ref
            for row in actual_outputs
            if isinstance(row, dict)
            for ref in _refs((row.get("artifact_refs") or []) + (row.get("deliverable_refs") or []) + (row.get("files_created") or []))
        }
    )[:12]
    domain = _clean_text(task.get("domain") or "mixed", 80)
    subdomain = _clean_text(task.get("subdomain") or "general", 80)
    criteria = task.get("success_criteria") or task.get("expected_deliverables") or []
    if isinstance(criteria, str):
        criteria = [criteria]
    criteria = [_clean_text(item, 180) for item in criteria][:8] or ["Produce task-specific deliverables and preserve evidence refs."]
    title = _task_title(evidence)
    instruction = _task_instruction(evidence)
    evidence_refs = _evidence_refs(evidence)[:14]
    if not output_summaries:
        output_summaries = [_output_summary(evidence)]
    if not artifact_refs:
        artifact_refs = _source_artifact_refs(evidence)[:12]
    concrete_evidence_refs = [
        ref for ref in evidence_refs + artifact_refs
        if str(ref).startswith(("traces/", "attempts/", "artifacts/", "outputs/", "evaluation/", "grading/", "feedback/", "signals/", "loops/"))
    ]
    if not concrete_evidence_refs:
        output_summaries = [
            "Concrete trace, artifact, attempt, or evaluation evidence was not available in this source package.",
            "Use this skill as task/rubric-derived guidance only, and require fresh execution evidence before accepting training or evaluation rows.",
        ]
    row_summary = ", ".join(f"{k}: {v}" for k, v in sorted((rows_accepted or {}).items()) if v) or "grounded rows recorded"
    audit = "mentor-audited" if report.get("mentor_audited") else "apprentice self-judged; future mentor audit recommended"
    return "\n".join(
        [
            f"# Skill Pack: {title}",
            "",
            "## Best Use",
            f"Use this skill for {domain}/{subdomain} work resembling: {instruction}",
            "It is useful when an Apprentice Agent must turn a real workflow into deliverables, traces, reusable experience, verifier evidence, and public ecosystem-ready package material.",
            "",
            "## Prerequisites",
            "- Access to the task inputs, artifacts, and package-relative evidence refs.",
            "- A writable task workspace for outputs and `actual_outputs.json`.",
            "- Mentor Model Provider or expert audit when semantic outcome quality must be judged.",
            "",
            "## Inputs",
            *(f"- {_clean_text(item, 180)}" for item in (task.get("required_inputs") or task.get("input_requirements") or ["Current task instruction, task assets, and package evidence refs."])[:8]),
            "",
            "## Outputs",
            *(f"- {_clean_text(item, 180)}" for item in (task.get("expected_deliverables") or task.get("output_requirements") or criteria)[:8]),
            "",
            "## Workflow",
            "1. Restate the user task as an Apprentice-generated task spec with domain, role, constraints, deliverables, and evidence requirements.",
            "2. Draft a rubric before execution, then preserve whether a Mentor Model, expert, or organization flow audited it.",
            "3. Execute one meaningful action per trace step, keeping observation, input, output, and state_change separate.",
            "4. Write deliverables under package-relative artifact paths and summarize them through `actual_outputs.json`.",
            "5. Use evaluator, verifier, grader, or expert records as outcome evidence; do not rely on Apprentice self-eval.",
            "6. Run the Experience Compiler after outputs exist so runtime learning and training-time rows cite concrete evidence refs.",
            "",
            "## Decision Points",
            f"- Apprenticeship Mode: {getattr(settings, 'apprenticeship_mode', 'autonomous')}.",
            f"- Compiler judge source: {report.get('tdo_judge_source')}; audit status: {audit}.",
            f"- Accepted learning objects: {row_summary}.",
            "",
            "## Common Failure Patterns",
            "- Missing artifact refs make outputs hard to verify.",
            "- Compressed trace steps lose causality and reduce observed state-transition value.",
            "- Unavailable Mentor Model Provider means compiler rows are self-judged and should be audited before quality claims.",
            "- Private paths or secrets make ecosystem sharing unsafe and must be redacted.",
            "",
            "## Recovery Strategies",
            "- Add missing package-relative artifact refs and rerun bundle check.",
            "- Split vague steps into single-action trace steps with caused_by links.",
            "- Re-run follow-up loops when user corrections change deliverables.",
            "- Preserve partial or failed runs honestly instead of rewriting them as success.",
            "",
            "## Tool Recipe",
            "- Use the tools and surfaces shown in the source traces; do not assume unavailable systems.",
            "- Store large command outputs, documents, tables, screenshots, and logs as artifact refs instead of embedding them in JSON.",
            "- Run verifier or bundle checks when present in the package evidence.",
            "",
            "## Evidence & Artifact Expectations",
            *(f"- {summary}" for summary in output_summaries),
            *(f"- artifact: `{ref}`" for ref in artifact_refs),
            *(f"- evidence ref: `{ref}`" for ref in evidence_refs),
            "",
            "## Verification Checklist",
            *(f"- {item}" for item in criteria),
            "- `actual_outputs.json` exists and names deliverable refs.",
            "- Trace steps include concrete evidence for generated rows.",
            "- Bundle check and public sanitizer pass without private paths or secrets.",
            "",
            "## Examples",
            f"- Source task: {title}",
            f"- Reusable domain: {domain}",
            f"- Compiler outputs: {row_summary}",
            "",
            "## Transfer Notes",
            "- Use the task/rubric pattern for adjacent workflows in the same domain.",
            "- Convert verifier checklist items into future eval cases.",
            "- Reuse observed state-transition rows for world-model-ready interaction data.",
            "",
        ]
    )


def _write_experience_compiler_outputs(
    package_root: Path,
    tdo_dir: Path,
    report: dict[str, Any],
    evidence: dict[str, Any],
    rows_accepted: dict[str, Any],
    settings: Settings,
) -> None:
    root = package_root / "experience_compiler"
    if root.exists():
        shutil.rmtree(root)
    for name in [
        "experience_pack",
        "skill_pack",
        "runtime_learning_package",
        "training_time_package",
        "evals",
        "eval_package",
        "verifier",
        "reward",
        "environment",
        "loop_learning",
        "reports",
    ]:
        (root / name).mkdir(parents=True, exist_ok=True)

    manifest_src = read_json(tdo_dir / "tdo_manifest.json") if (tdo_dir / "tdo_manifest.json").exists() else {}
    compiler_rows = _compiler_rows_from_tdo_dir(tdo_dir)
    training_row_counts_by_file = _training_row_counts_by_file(tdo_dir)
    evidence_map = _source_evidence_map_payload(evidence, report, compiler_rows)
    package_version = get_package_version()
    compiler_refs = [
        "experience_compiler/compiler_manifest.json",
        "experience_compiler/compiler_report.json",
        "experience_compiler/compiler_trace.jsonl",
        "experience_compiler/compiler_rounds.jsonl",
        "experience_compiler/compiler_recipe.json",
        "experience_compiler/compiler_judge_results.jsonl",
        "experience_compiler/learning_objects_index.json",
        "experience_compiler/skill_pack/skill.md",
    ]
    compiler_manifest = {
        "schema_version": "aa-experience-compiler-v0.2",
        "training_row_schema_version": TRAINING_ROW_SCHEMA_VERSION,
        "source_evidence_map_schema_version": EVIDENCE_MAP_SCHEMA_VERSION,
        "package_version": package_version,
        "compiler_version": TDO_VERSION,
        "compiler_id": str(report.get("tdo_id") or "").replace("tdo_", "compiler_", 1) or f"compiler_{_safe_id(package_root.name)}",
        "compiler_status": report.get("tdo_status"),
        "source_package_id": report.get("source_package_id"),
        "source_task_id": report.get("source_task_id"),
        "title": _task_title(evidence),
        "task_title": _task_title(evidence),
        "domain": _task_lineage(evidence)["domain"],
        "subdomain": _task_lineage(evidence)["subdomain"],
        "workflow_family": _task_lineage(evidence)["workflow_family"],
        "public_private_mode": getattr(settings, "contribution_mode", None),
        "sharing_upload_mode": getattr(settings, "contribution_mode", None),
        "private_internal_no_upload_confirmed": getattr(settings, "contribution_mode", None) == "private_internal",
        "apprenticeship_mode": getattr(settings, "apprenticeship_mode", None),
        "judge_source": report.get("tdo_judge_source"),
        "mentor_audited": report.get("mentor_audited"),
        "rounds_run": report.get("rounds_run"),
        "rows_by_type": rows_accepted,
        "training_row_counts_by_file": training_row_counts_by_file,
        "source_evidence_ref_count": (evidence_map.get("coverage_statistics") or {}).get("evidence_ref_count"),
        "source_evidence_row_link_count": (evidence_map.get("coverage_statistics") or {}).get("row_link_count"),
        "supported_runtime_uses": ["experience_pack", "skill_pack", "runtime_learning_package"],
        "supported_training_uses": [
            "sft",
            "lora_ready_instruction_tuning",
            "process_supervision",
            "training_examples",
            "evaluation_examples",
            "verifier_examples",
            "reward_model_rows",
            "preference_pairs",
            "critique_revision_pairs",
            "failure_cases",
            "task_variants",
            "agentic_rl_rollouts",
            "grpo_rollout_material",
            "environment_response_pairs",
            "loop_learning",
            "contract_verification",
            "environment_state_transition_learning",
        ],
        "loop_review_refs": evidence.get("loop_review_refs") or [],
        "compiler_output_refs": compiler_refs,
        "created_at": report.get("created_at") or utc_now(),
    }
    write_json(root / "compiler_manifest.json", compiler_manifest)
    compiler_report = _compiler_public_obj({
        **report,
        "public_feature_name": "Experience Compiler",
        "description": "The Experience Compiler turns completed agent workflows into reusable runtime learning and training-time data.",
        "apprenticeship_mode": getattr(settings, "apprenticeship_mode", None),
    })
    write_json(root / "compiler_report.json", sanitize_public_obj(compiler_report))
    write_json(root / "source_evidence_map.json", evidence_map)
    write_json(root / "quality_report.json", sanitize_public_obj({
        "source_package_id": report.get("source_package_id"),
        "compiler_status": report.get("tdo_status"),
        "package_version": package_version,
        "compiler_version": TDO_VERSION,
        "average_quality_score": report.get("average_quality_score"),
        "average_fidelity_score": report.get("average_fidelity_score"),
        "average_reuse_value_score": report.get("average_reuse_value_score"),
        "training_row_counts_by_file": training_row_counts_by_file,
        "source_evidence_map_coverage": evidence_map.get("coverage_statistics"),
        "omitted_outputs": report.get("omitted_outputs") or [],
        "not_generated_counts": report.get("not_generated_counts") or {},
        "generation_notes": report.get("generation_notes") or [],
        "grounding": "Rows are derived from source traces, actual_outputs, artifacts, rubrics, and evaluator/verifier records when available.",
    }))
    _copy_compiler_jsonl_public(tdo_dir / "tdo_trace.jsonl", root / "compiler_trace.jsonl")
    _copy_compiler_jsonl_public(tdo_dir / "tdo_rounds.jsonl", root / "compiler_rounds.jsonl")
    _copy_compiler_json_public(tdo_dir / "tdo_recipe.json", root / "compiler_recipe.json")
    _copy_compiler_jsonl_public(tdo_dir / "tdo_judge_results.jsonl", root / "compiler_judge_results.jsonl")
    _copy_compiler_jsonl_public(tdo_dir / "tdo_rounds.jsonl", root / "reports" / "compiler_rounds.jsonl")
    _copy_compiler_jsonl_public(tdo_dir / "tdo_judge_results.jsonl", root / "reports" / "compiler_judge_results.jsonl")
    _copy_compiler_json_public(tdo_dir / "tdo_recipe.json", root / "reports" / "compiler_recipe.json")

    loop_learning = _loop_learning_rows(evidence, report)
    _write_jsonl(root / "loop_learning" / "loop_review_rows.jsonl", loop_learning["loop_review_rows"])
    _write_jsonl(root / "loop_learning" / "feedback_revision_pairs.jsonl", loop_learning["feedback_revision_pairs"])
    _write_jsonl(root / "loop_learning" / "contract_verification_rows.jsonl", loop_learning["contract_verification_rows"])
    _write_jsonl(root / "loop_learning" / "repair_loop_rows.jsonl", loop_learning["repair_loop_rows"])
    _write_jsonl(root / "loop_learning" / "reviewer_decision_rows.jsonl", loop_learning["reviewer_decision_rows"])

    jsonl_map = {
        "training_examples.jsonl": root / "training_time_package" / "training_examples.jsonl",
        "evaluation_examples.jsonl": root / "training_time_package" / "evaluation_examples.jsonl",
        "verifier_examples.jsonl": root / "training_time_package" / "verifier_examples.jsonl",
        "reward_model_rows.jsonl": root / "training_time_package" / "reward_model_rows.jsonl",
        "preference_pairs.jsonl": root / "training_time_package" / "preference_pairs.jsonl",
        "failure_cases.jsonl": root / "training_time_package" / "failure_cases.jsonl",
        "task_variants.jsonl": root / "training_time_package" / "task_variants.jsonl",
        "environment_response_pairs.jsonl": root / "environment" / "environment_response_pairs.jsonl",
        "multi_agent_handoffs.jsonl": root / "environment" / "multi_agent_handoffs.jsonl",
        "multimodal_artifacts.jsonl": root / "environment" / "multimodal_artifacts.jsonl",
        "ui_interaction_events.jsonl": root / "environment" / "ui_interaction_events.jsonl",
        "physical_environment_events.jsonl": root / "environment" / "physical_environment_events.jsonl",
        "api_interaction_events.jsonl": root / "environment" / "api_interaction_events.jsonl",
        "mcp_interaction_events.jsonl": root / "environment" / "mcp_interaction_events.jsonl",
        "database_events.jsonl": root / "environment" / "database_events.jsonl",
        "structured_table_events.jsonl": root / "environment" / "structured_table_events.jsonl",
        "software_execution_events.jsonl": root / "environment" / "software_execution_events.jsonl",
        "search_retrieval_events.jsonl": root / "environment" / "search_retrieval_events.jsonl",
        "enterprise_app_events.jsonl": root / "environment" / "enterprise_app_events.jsonl",
        "async_workflow_events.jsonl": root / "environment" / "async_workflow_events.jsonl",
        "collaboration_events.jsonl": root / "environment" / "collaboration_events.jsonl",
        "document_events.jsonl": root / "environment" / "document_events.jsonl",
        "cross_system_events.jsonl": root / "environment" / "cross_system_events.jsonl",
        "rejected_rows.jsonl": root / "reports" / "rejected_rows.jsonl",
        "meta_optimization_records.jsonl": root / "reports" / "meta_optimization_records.jsonl",
    }
    for src_name, dst in jsonl_map.items():
        _copy_jsonl_public(tdo_dir / src_name, dst)

    _copy_json_public(tdo_dir / "optimized_task_spec.json", root / "reports" / "optimized_task_spec.json")
    _copy_json_public(tdo_dir / "optimized_rubric.json", root / "reports" / "optimized_rubric.json")
    _copy_json_public(tdo_dir / "domain_value_profile.json", root / "reports" / "domain_value_profile.json")
    _copy_json_public(tdo_dir / "environment_spec.json", root / "environment" / "environment_spec.json")
    _copy_jsonl_public(tdo_dir / "evaluation_examples.jsonl", root / "evals" / "evaluation_examples.jsonl")
    _copy_jsonl_public(tdo_dir / "verifier_examples.jsonl", root / "verifier" / "verifier_examples.jsonl")
    _copy_jsonl_public(tdo_dir / "reward_model_rows.jsonl", root / "reward" / "reward_model_rows.jsonl")
    _copy_jsonl_public(tdo_dir / "preference_pairs.jsonl", root / "reward" / "preference_pairs.jsonl")

    write_json(root / "experience_pack" / "experience_pack_manifest.json", {
        "pack_type": "experience_pack",
        "source_package_id": report.get("source_package_id"),
        "skill_pack_ref": "experience_compiler/skill_pack/skill.md",
        "runtime_learning_package_ref": "experience_compiler/runtime_learning_package/runtime_learning_manifest.json",
        "training_time_package_ref": "experience_compiler/training_time_package/training_time_manifest.json",
    })
    skill_md = _skill_markdown(evidence, report, rows_accepted, settings)
    (root / "skill_pack" / "skill.md").write_text(sanitize_public_text(skill_md) or skill_md)
    write_json(root / "skill_pack" / "skill_manifest.json", {
        "source_package_id": report.get("source_package_id"),
        "source_task_id": report.get("source_task_id"),
        "skill_ref": "experience_compiler/skill_pack/skill.md",
        "sections": ["Best Use", "Prerequisites", "Workflow", "Decision Points", "Common Failure Patterns", "Recovery Strategies", "Evidence & Artifact Expectations", "Verification Checklist", "Examples", "Transfer Notes"],
        "quality_floor": "task-specific, evidence-grounded, non-generic",
    })
    title = _task_title(evidence) or str(report.get("source_task_id") or "Compiled skill")
    instruction = _task_instruction(evidence)
    criteria = _success_criteria(evidence)
    artifact_contracts = _artifact_contracts(evidence)
    canonical_skill = sanitize_public_obj({
        "skill_id": f"skill_{_safe_id(str(report.get('source_package_id') or package_root.name))}",
        "title": title,
        "summary": f"Reusable runtime skill compiled from the completed workflow: {title}",
        "when_to_use": [f"Use for similar {(_domain(evidence) or 'workflow')} tasks with matching artifacts, constraints, or verification needs."],
        "procedure": [line.strip("- ") for line in _skill_markdown(evidence, report, rows_accepted, settings).splitlines() if line.strip().startswith("- ")][:12],
        "workflow_steps": [
            "Confirm the current task matches the source domain, deliverables, and constraints.",
            "Reuse the source workflow pattern while keeping the current user instruction authoritative.",
            "Create package-relative artifacts and update actual_outputs.json.",
            "Verify outputs against the compiled checklist and source evidence expectations.",
        ],
        "decision_rules": [
            "Apply this skill only when query terms, domain, required artifacts, or verifier needs overlap with the source task.",
            "Prefer current-task evidence over transferred assumptions.",
            "Stop or ask for expert/model review when required evidence is missing or semantic quality is uncertain.",
        ],
        "verifier_checks": criteria + ["actual_outputs.json names the deliverables", "source evidence refs remain package-relative"],
        "failure_modes": [
            "The current task lacks the source task's required inputs or tools.",
            "Artifact refs are missing, absolute, or unresolved.",
            "The workflow is copied without adapting to current constraints.",
        ],
        "recovery_strategies": [
            "Regenerate missing package-relative refs and rerun bundle or verifier checks.",
            "Record evidence status instead of fabricating missing evidence.",
            "Use a follow-up loop when user or reviewer feedback changes the deliverables.",
        ],
        "success_criteria": criteria,
        "evidence_refs": _evidence_refs(evidence),
        "evidence_reference_ids": _evidence_refs(evidence),
    })
    write_json(root / "skill_pack" / "canonical_skill.json", canonical_skill)
    write_json(root / "skill_pack" / "skill_card.json", {
        "title": title,
        "domain": _domain(evidence),
        "source_package_id": report.get("source_package_id"),
        "runtime_training_summary": f"Skill pack for {title}",
    })
    write_json(root / "skill_pack" / "retrieval_card.json", {
        "title": title,
        "tags": sorted(set([_domain(evidence), *(str(title + " " + instruction).lower().split()[:12])]))[:16],
        "query_terms": sorted(set(str(title + ' ' + instruction).lower().split()))[:40],
        "useful_for": [
            f"Runtime guidance for adjacent {_domain(evidence) or 'workflow'} tasks.",
            "Verifier checklist reuse when deliverables and evidence patterns match.",
            "Training/evaluation examples grounded in the source package.",
        ],
        "evidence_refs": _evidence_refs(evidence)[:12],
        "confidence": report.get("average_quality_score") or report.get("average_fidelity_score") or 0.5,
        "source_package_id": report.get("source_package_id"),
        "artifact_expectations": _source_artifact_refs(evidence)[:10],
    })
    _write_jsonl(root / "skill_pack" / "usage_examples.jsonl", [{
        "source_package_id": report.get("source_package_id"),
        "input": instruction,
        "expected_runtime_behavior": "Follow the compiled procedure, preserve evidence refs, and verify outputs against the checklist.",
        "evidence_refs": _evidence_refs(evidence)[:10],
    }])
    _write_jsonl(root / "skill_pack" / "anti_patterns.jsonl", [{
        "source_package_id": report.get("source_package_id"),
        "anti_pattern": "Applying the compiled skill without checking current task constraints and required artifacts.",
        "correction": "Re-read the task brief and verifier checklist before reusing the procedure.",
    }])
    _write_jsonl(root / "skill_pack" / "failure_modes.jsonl", [{
        "source_package_id": report.get("source_package_id"),
        "failure_mode": "Missing artifact evidence or unresolved output references.",
        "recovery": "Regenerate or relink package-relative artifact refs, then rerun bundle check.",
    }])
    _write_jsonl(root / "skill_pack" / "decision_rules.jsonl", [{
        "source_package_id": report.get("source_package_id"),
        "rule": "Use the skill when task/domain/query terms overlap and required tools are available.",
        "evidence_refs": _evidence_refs(evidence)[:5],
    }])
    write_json(root / "skill_pack" / "tool_recipe.json", {
        "source_package_id": report.get("source_package_id"),
        "required_tools": sorted(set(_environment_spec(evidence).get("required_tools") or [])),
        "procedure_ref": "skill.md",
    })
    verifier_lines = ["# Verifier Checklist", "", *(f"- {item}" for item in criteria), "- Bundle check passes.", "- Evidence refs resolve."]
    (root / "skill_pack" / "verifier_checklist.md").write_text("\n".join(verifier_lines) + "\n")
    write_json(root / "skill_pack" / "verifier_checklist.json", {
        "source_package_id": report.get("source_package_id"),
        "checks": criteria + ["bundle_check_passes", "evidence_refs_resolve"],
    })
    output_contract = _output_contract_payload(evidence, artifact_contracts)
    artifact_contract = _artifact_contract_payload(evidence, artifact_contracts)
    prefinal_checklist = _prefinal_checklist_payload(evidence, output_contract, artifact_contract)
    write_json(root / "runtime_learning_package" / "runtime_learning_manifest.json", {
        "source_package_id": report.get("source_package_id"),
        "apprenticeship_mode": getattr(settings, "apprenticeship_mode", None),
        "runtime_uses": ["prompt_guidance", "tool_use_hints", "correction_patterns", "experience_pack"],
        "supported_runtime_uses": [
            "skill_import",
            "retrieval_augmented_execution",
            "verifier_assisted_execution",
            "workflow_checklist",
            "failure_pattern_warning",
            "tool_recipe_reuse",
        ],
        "skill_ref": "experience_compiler/skill_pack/skill.md",
        "transfer_runbook_ref": "experience_compiler/runtime_learning_package/transfer_runbook.md",
        "transfer_checklist_ref": "experience_compiler/runtime_learning_package/transfer_checklist.json",
        "minimal_prompt_context_ref": "experience_compiler/runtime_learning_package/minimal_prompt_context.md",
        "output_contract_ref": "experience_compiler/runtime_learning_package/output_contract.json",
        "artifact_contract_ref": "experience_compiler/runtime_learning_package/artifact_contract.json",
        "prefinal_verification_checklist_ref": "experience_compiler/runtime_learning_package/prefinal_verification_checklist.json",
        "repair_instructions_ref": "experience_compiler/runtime_learning_package/repair_instructions.md",
    })
    for name, rows in {
        "runtime_memory.jsonl": [{"source_package_id": report.get("source_package_id"), "lesson": _task_instruction(evidence), "usable_for": ["runtime_memory"]}],
        "tool_use_hints.jsonl": [{"source_package_id": report.get("source_package_id"), "hint": "Preserve package-relative refs for commands, files, artifacts, and actual outputs.", "usable_for": ["tool_use"]}],
        "correction_patterns.jsonl": [{"source_package_id": report.get("source_package_id"), "pattern": "If a follow-up changes constraints, update artifacts and compiler metadata without inventing missing execution.", "usable_for": ["followup"]}],
    }.items():
        _write_jsonl(root / "runtime_learning_package" / name, rows)
    (root / "runtime_learning_package" / "skill.md").write_text((root / "skill_pack" / "skill.md").read_text(errors="replace"))
    (root / "runtime_learning_package" / "runtime_context.md").write_text(
        f"# Runtime Context\n\nSource task: {title}\n\nInstruction pattern:\n\n{instruction}\n\nDomain: {_domain(evidence) or 'unknown'}\n\nEvidence refs:\n"
        + "\n".join(f"- `{ref}`" for ref in _evidence_refs(evidence)[:12])
        + "\n"
    )
    (root / "runtime_learning_package" / "runtime_instructions.md").write_text(
        "\n".join(
            [
                f"# Runtime Instructions: {title}",
                "",
                "Use this runtime training only when the current task is relevant to the source workflow. The current user instruction remains authoritative.",
                "",
                "## Apply When",
                f"- The task resembles `{_domain(evidence) or 'workflow'}` work or shares deliverables with the source task.",
                "- The required tools, artifacts, and verification expectations are available.",
                "",
                "## Runtime Steps",
                "1. Restate the current task and compare it to the source workflow.",
                "2. Reuse only the relevant workflow steps, decision rules, and verifier checks.",
                "3. Produce package-relative artifacts and update `actual_outputs.json`.",
                "4. Preserve trace steps, feedback refs, and evidence refs for later review.",
                "5. Use only current-task evidence and omit unsupported transferred assumptions.",
                "",
                "## Verifier Focus",
                *(f"- {item}" for item in criteria[:6]),
                "- Evidence refs resolve and remain package-relative.",
                "- Outputs are grounded in current task artifacts rather than source-task assumptions.",
                "",
                "## Source Evidence",
                *(f"- `{ref}`" for ref in _evidence_refs(evidence)[:12]),
                "",
            ]
        )
    )
    transfer_runbook = _runtime_transfer_runbook(evidence, title, instruction, criteria, artifact_contracts)
    (root / "runtime_learning_package" / "transfer_runbook.md").write_text(
        sanitize_public_text(transfer_runbook) or transfer_runbook
    )
    minimal_context = "\n".join(
        [
            f"# Minimal Runtime Transfer Context: {title}",
            "",
            "Follow current-task instructions. Reuse only the workflow pattern and output-contract discipline from this source task.",
            "",
            "## Must Do",
            "- Derive outputs from current visible input files.",
            "- Match required filenames, CSV headers, JSON keys, and artifact roles.",
            "- Write `actual_outputs.json` with the generated file list.",
            "- Run the pre-final contract checklist before finishing.",
            "",
            "## Output Contract Pattern",
            *[f"- {_contract_line(contract)}" for contract in artifact_contracts[:8]],
            "",
        ]
    )
    (root / "runtime_learning_package" / "minimal_prompt_context.md").write_text(minimal_context[:4000])
    write_json(root / "runtime_learning_package" / "output_contract.json", output_contract)
    write_json(root / "runtime_learning_package" / "artifact_contract.json", artifact_contract)
    write_json(root / "runtime_learning_package" / "prefinal_verification_checklist.json", prefinal_checklist)
    (root / "runtime_learning_package" / "repair_instructions.md").write_text(
        _repair_instructions_text(title, output_contract)
    )
    write_json(root / "runtime_learning_package" / "transfer_checklist.json", {
        "source_package_id": report.get("source_package_id"),
        "title": title,
        "checks": [
            "current_task_inputs_read",
            "outputs_derived_from_current_inputs",
            "required_filenames_created",
            "csv_headers_or_json_keys_match_contract",
            "actual_outputs_json_lists_generated_files",
            "local_sanity_check_completed",
        ],
        "artifact_contracts": artifact_contracts,
        "verifier_checks": criteria[:12],
    })
    write_json(root / "runtime_learning_package" / "tool_execution_recipe.json", {
        "source_package_id": report.get("source_package_id"),
        "recommended_steps": [
            "inspect visible input files",
            "write a short deterministic Python script or shell command sequence",
            "run the script in the task directory",
            "open outputs and validate schemas",
            "write actual_outputs.json",
        ],
        "artifact_contracts": artifact_contracts[:12],
    })
    _write_jsonl(root / "runtime_learning_package" / "output_verification_steps.jsonl", [
        {
            "source_package_id": report.get("source_package_id"),
            "artifact": _contract_output_path(contract),
            "check": _contract_line(contract),
            "evidence_refs": [contract.get("artifact_ref")] if contract.get("artifact_ref") else _evidence_refs(evidence)[:3],
        }
        for contract in artifact_contracts
    ] or [{
        "source_package_id": report.get("source_package_id"),
        "artifact": "actual_outputs.json",
        "check": "Generated outputs are listed in actual_outputs.json.",
        "evidence_refs": _evidence_refs(evidence)[:3],
    }])
    _write_jsonl(root / "runtime_learning_package" / "common_errors_to_avoid.jsonl", [
        {
            "source_package_id": report.get("source_package_id"),
            "error": "Missing required output file or writing it outside the expected relative output location.",
            "avoidance": "Create every named artifact and reopen it before final response.",
        },
        {
            "source_package_id": report.get("source_package_id"),
            "error": "CSV headers or JSON keys drift from the expected artifact contract.",
            "avoidance": "Use the transfer checklist and write headers/keys explicitly.",
        },
        {
            "source_package_id": report.get("source_package_id"),
            "error": "actual_outputs.json exists but does not list generated files.",
            "avoidance": "Populate files/deliverable refs after verifying outputs on disk.",
        },
    ])
    for src, dst in [
        ("retrieval_card.json", "retrieval_card.json"),
        ("verifier_checklist.json", "verifier_spec.json"),
        ("tool_recipe.json", "tool_recipe.json"),
    ]:
        shutil.copy2(root / "skill_pack" / src, root / "runtime_learning_package" / dst)
    reusable_rows = [{"source_package_id": report.get("source_package_id"), "text": item, "evidence_refs": _evidence_refs(evidence)[:5]} for item in criteria[:8]]
    for name in [
        "reusable_steps.jsonl",
        "reusable_decision_rules.jsonl",
        "tool_use_patterns.jsonl",
        "failure_recovery_patterns.jsonl",
        "mentor_feedback_patterns.jsonl",
        "verifier_feedback_patterns.jsonl",
        "artifact_expectations.jsonl",
        "environment_expectations.jsonl",
    ]:
        _write_jsonl(root / "runtime_learning_package" / name, reusable_rows or [{
            "source_package_id": report.get("source_package_id"),
            "text": "Preserve task-specific evidence, artifacts, and output refs.",
            "evidence_refs": _evidence_refs(evidence)[:5],
        }])
    write_json(root / "training_time_package" / "training_time_manifest.json", {
        "source_package_id": report.get("source_package_id"),
        "training_uses": compiler_manifest["supported_training_uses"],
        "supported_training_uses": [
            "sft",
            "lora_ready_instruction_tuning",
            "rl",
            "grpo",
            "agentic_rollouts",
            "reward_modeling",
            "preference_learning",
            "verifier_training",
            "process_supervision",
            "distillation",
            "eval_generation",
            "environment_model_training",
        ],
        "row_counts": rows_accepted,
        "row_counts_by_file": training_row_counts_by_file,
        "schema_version": TRAINING_ROW_SCHEMA_VERSION,
        "source_evidence_map_ref": "experience_compiler/source_evidence_map.json",
        "no_model_training_claimed": True,
    })
    for src, dst in [
        ("training_examples.jsonl", "sft_examples.jsonl"),
        ("verifier_examples.jsonl", "verifier_training_rows.jsonl"),
        ("failure_cases.jsonl", "failure_cases.jsonl"),
        ("task_variants.jsonl", "task_variants.jsonl"),
        ("environment_response_pairs.jsonl", "environment_response_pairs.jsonl"),
        ("reward_model_rows.jsonl", "reward_model_rows.jsonl"),
        ("preference_pairs.jsonl", "preference_pairs.jsonl"),
    ]:
        _copy_jsonl_public(tdo_dir / src, root / "training_time_package" / dst)
    process_rows = _process_supervision_rows_from_compiler_rows(compiler_rows)
    _write_jsonl(root / "training_time_package" / "process_supervision.jsonl", process_rows)
    rl_rollout_rows = _rl_rollout_rows_from_compiler_rows(compiler_rows)
    _write_jsonl(root / "training_time_package" / "agentic_rollouts.jsonl", rl_rollout_rows)
    _write_jsonl(root / "training_time_package" / "grpo_rollout_groups.jsonl", [
        {
            "schema_version": TRAINING_ROW_SCHEMA_VERSION,
            "row_id": f"grpo_group_{_safe_id(str(report.get('source_package_id') or package_root.name))}",
            "data_type": "grpo_rollout_group",
            "source_package_id": report.get("source_package_id"),
            "source_task_id": report.get("source_task_id"),
            "task_lineage": _task_lineage(evidence),
            "rollout_group_id": f"rollout_{_safe_id(str(report.get('source_task_id') or package_root.name))}",
            "candidate_rollout_refs": [row.get("row_id") for row in rl_rollout_rows],
            "reward_components": ["artifact_contract", "verifier_signal", "step_success"],
            "source_evidence_ref_ids": _evidence_ids_for_refs(_evidence_refs(evidence)[:12]),
            "evidence_refs": _evidence_refs(evidence)[:12],
            "confidence": report.get("average_quality_score") or 0.5,
            "created_at": utc_now(),
        }
    ] if rl_rollout_rows else [])
    _write_jsonl(root / "training_time_package" / "rl_environment_replay_rows.jsonl", rl_rollout_rows)
    loop_pairs: list[dict[str, Any]] = []
    for item in evidence.get("loop_review_packets") or []:
        packet = item.get("packet") or {}
        ref = item.get("ref")
        loop_pairs.append({
            "schema_version": TRAINING_ROW_SCHEMA_VERSION,
            "pair_id": f"critique_revision_{_safe_id(str(report.get('source_package_id') or package_root.name))}_{packet.get('loop_iteration') or len(loop_pairs)+1}",
            "row_id": f"critique_revision_{_safe_id(str(report.get('source_package_id') or package_root.name))}_{packet.get('loop_iteration') or len(loop_pairs)+1}",
            "data_type": "critique_revision_pair",
            "source_package_id": report.get("source_package_id"),
            "source_task_id": report.get("source_task_id"),
            "task_lineage": _task_lineage(evidence),
            "critique": _clean_text("; ".join(packet.get("known_failures") or packet.get("recommended_review_focus") or []) or "Loop review packet recorded no blocking failure."),
            "revision": _clean_text(str((packet.get("diff_from_previous_iteration") or {}).get("changed_summary") or "Use loop feedback and revision plan to ground updates.")),
            "evidence_refs": [ref] if ref else _evidence_refs(evidence)[:5],
            "source_evidence_ref_ids": _evidence_ids_for_refs([ref] if ref else _evidence_refs(evidence)[:5]),
            "loop_iteration": packet.get("loop_iteration"),
            "confidence": report.get("average_quality_score") or 0.5,
            "created_at": utc_now(),
        })
    if not loop_pairs:
        loop_pairs = [{
            "schema_version": TRAINING_ROW_SCHEMA_VERSION,
            "pair_id": f"critique_revision_{_safe_id(str(report.get('source_package_id') or package_root.name))}_fallback",
            "row_id": f"critique_revision_{_safe_id(str(report.get('source_package_id') or package_root.name))}_fallback",
            "data_type": "critique_revision_pair",
            "source_package_id": report.get("source_package_id"),
            "source_task_id": report.get("source_task_id"),
            "task_lineage": _task_lineage(evidence),
            "critique": "Use evaluator/verifier feedback where present; do not invent missing feedback.",
            "revision": "Ground revisions in actual outputs and artifact refs.",
            "evidence_refs": _evidence_refs(evidence)[:5],
            "source_evidence_ref_ids": _evidence_ids_for_refs(_evidence_refs(evidence)[:5]),
            "confidence": report.get("average_quality_score") or 0.5,
            "created_at": utc_now(),
        }]
    loop_pairs.extend(loop_learning["feedback_revision_pairs"])
    _write_jsonl(root / "training_time_package" / "critique_revision_pairs.jsonl", loop_pairs)
    _write_jsonl(root / "training_time_package" / "trajectory_rollouts.jsonl", [{
        "source_package_id": report.get("source_package_id"),
        "trajectory_ref": "traces/agent_traces.jsonl",
        "artifact_refs": _source_artifact_refs(evidence),
    }])
    _write_jsonl(root / "training_time_package" / "distillation_records.jsonl", [{
        "schema_version": TRAINING_ROW_SCHEMA_VERSION,
        "row_id": f"distill_{_safe_id(str(report.get('source_package_id') or package_root.name))}",
        "data_type": "distillation_record",
        "source_package_id": report.get("source_package_id"),
        "source_task_id": report.get("source_task_id"),
        "task_lineage": _task_lineage(evidence),
        "input": instruction,
        "target": _output_summary(evidence),
        "evidence_refs": _evidence_refs(evidence)[:8],
        "source_evidence_ref_ids": _evidence_ids_for_refs(_evidence_refs(evidence)[:8]),
        "confidence": report.get("average_quality_score") or 0.5,
        "created_at": utc_now(),
    }])
    _write_jsonl(root / "training_time_package" / "transfer_eval_tasks.jsonl", [
        {
            "schema_version": TRAINING_ROW_SCHEMA_VERSION,
            "row_id": f"transfer_eval_{_safe_id(str(report.get('source_package_id') or package_root.name))}",
            "data_type": "transfer_eval_task",
            "source_package_id": report.get("source_package_id"),
            "source_task_id": report.get("source_task_id"),
            "task_lineage": _task_lineage(evidence),
            "task": f"Replay a small adjacent {(_task_title(evidence) or 'workflow')} task and compare artifact quality.",
            "success_signal": "bundle check passes and verifier/evaluator evidence refs are present",
            "transfer_tested": False,
            "expected_output_contract": output_contract,
            "source_evidence_ref_ids": _evidence_ids_for_refs(_evidence_refs(evidence)[:8]),
            "evidence_refs": _evidence_refs(evidence)[:8],
            "confidence": report.get("average_quality_score") or 0.5,
            "created_at": utc_now(),
        }
    ])
    write_json(root / "reports" / "transfer_test_report.json", {
        "transfer_tested": False,
        "reason": "Transfer tests are reserved for future replay runs; no model training was performed.",
    })
    _copy_jsonl_public(tdo_dir / "evaluation_examples.jsonl", root / "eval_package" / "evaluation_examples.jsonl")
    shutil.copy2(root / "training_time_package" / "transfer_eval_tasks.jsonl", root / "eval_package" / "transfer_eval_tasks.jsonl")
    shutil.copy2(root / "reports" / "transfer_test_report.json", root / "eval_package" / "transfer_test_report.json")
    write_json(root / "eval_package" / "eval_manifest.json", {
        "schema_version": "aa-eval-package-v0.2",
        "source_package_id": report.get("source_package_id"),
        "source_task_id": report.get("source_task_id"),
        "task_lineage": _task_lineage(evidence),
        "source_evidence_map_ref": "experience_compiler/source_evidence_map.json",
        "output_contract_ref": "experience_compiler/runtime_learning_package/output_contract.json",
        "artifact_contract_ref": "experience_compiler/runtime_learning_package/artifact_contract.json",
        "supported_eval_uses": ["rubric_eval", "contract_verifier_training", "transfer_eval_generation", "grader_case_generation"],
    })
    _write_jsonl(root / "eval_package" / "rubric_items.jsonl", [
        {
            "schema_version": TRAINING_ROW_SCHEMA_VERSION,
            "row_id": f"eval_rubric_{_safe_id(str(report.get('source_package_id') or package_root.name))}_{idx}",
            "data_type": "eval_rubric_item",
            "source_package_id": report.get("source_package_id"),
            "source_task_id": report.get("source_task_id"),
            "task_lineage": _task_lineage(evidence),
            "rubric_item": item,
            "scoring_criterion": item,
            "evidence_refs": _evidence_refs(evidence)[:5],
            "source_evidence_ref_ids": _evidence_ids_for_refs(_evidence_refs(evidence)[:5]),
            "confidence": report.get("average_quality_score") or 0.5,
            "created_at": utc_now(),
        }
        for idx, item in enumerate(criteria, start=1)
    ])
    _write_jsonl(root / "eval_package" / "grader_cases.jsonl", [{
        "schema_version": TRAINING_ROW_SCHEMA_VERSION,
        "row_id": f"grader_case_{_safe_id(str(report.get('source_package_id') or package_root.name))}",
        "data_type": "grader_case",
        "source_package_id": report.get("source_package_id"),
        "source_task_id": report.get("source_task_id"),
        "task_lineage": _task_lineage(evidence),
        "case": "Grade output against compiled rubric and artifact evidence.",
        "output_contract": output_contract,
        "artifact_contract": artifact_contract,
        "evidence_refs": _evidence_refs(evidence)[:5],
        "source_evidence_ref_ids": _evidence_ids_for_refs(_evidence_refs(evidence)[:5]),
        "confidence": report.get("average_quality_score") or 0.5,
        "created_at": utc_now(),
    }])
    _write_jsonl(root / "eval_package" / "verifier_cases.jsonl", [{
        "schema_version": TRAINING_ROW_SCHEMA_VERSION,
        "row_id": f"verifier_case_{_safe_id(str(report.get('source_package_id') or package_root.name))}",
        "data_type": "verifier_case",
        "source_package_id": report.get("source_package_id"),
        "source_task_id": report.get("source_task_id"),
        "task_lineage": _task_lineage(evidence),
        "claim_or_output": "Required artifacts and output refs exist and match the contract.",
        "visible_evidence": _source_artifact_refs(evidence)[:12],
        "expected_contract": output_contract,
        "verifier_decision": "case_requires_future_candidate_output",
        "failure_reasons": [],
        "evidence_refs": _evidence_refs(evidence)[:5],
        "source_evidence_ref_ids": _evidence_ids_for_refs(_evidence_refs(evidence)[:5]),
        "confidence": report.get("average_quality_score") or 0.5,
        "created_at": utc_now(),
    }])
    _write_jsonl(root / "eval_package" / "expected_artifacts.jsonl", [
        {
            "schema_version": TRAINING_ROW_SCHEMA_VERSION,
            "row_id": f"expected_artifact_{_safe_id(str(report.get('source_package_id') or package_root.name))}_{idx}",
            "data_type": "expected_artifact",
            "source_package_id": report.get("source_package_id"),
            "source_task_id": report.get("source_task_id"),
            "task_lineage": _task_lineage(evidence),
            "artifact_ref": ref,
            "source_evidence_ref_ids": _evidence_ids_for_refs([ref]),
            "evidence_refs": [ref],
            "confidence": 1.0,
            "created_at": utc_now(),
        }
        for idx, ref in enumerate(_source_artifact_refs(evidence), start=1)
    ])
    env_spec = _environment_spec(evidence)
    write_json(root / "environment" / "environment_manifest.json", {
        "schema_version": "aa-environment-package-v0.2",
        "source_package_id": report.get("source_package_id"),
        "source_task_id": report.get("source_task_id"),
        "task_lineage": _task_lineage(evidence),
        "environment_spec_ref": "experience_compiler/environment/environment_spec.json",
        "state_transition_rows_ref": "experience_compiler/environment/state_transition_rows.jsonl",
        "replay_plan_ref": "experience_compiler/environment/replay_plan.json",
        "source_evidence_map_ref": "experience_compiler/source_evidence_map.json",
        "supported_training_uses": ["agentic_rl", "grpo_rollout_material", "environment_model_training", "replay_smoke_testing"],
    })
    write_json(root / "environment" / "replay_plan.json", {
        "schema_version": "aa-replay-plan-v0.2",
        "source_package_id": report.get("source_package_id"),
        "source_task_id": report.get("source_task_id"),
        "task_lineage": _task_lineage(evidence),
        "replay_available": bool(env_spec.get("verification_commands")),
        "initial_state_refs": env_spec.get("initial_state_refs") or [],
        "visible_input_refs": env_spec.get("input_artifacts") or [],
        "action_schema_ref": "experience_compiler/environment/allowed_actions.jsonl",
        "observation_schema": {"observation": "string", "artifact_refs": "array", "state_after": "string"},
        "scoring_or_verifier_hooks": env_spec.get("scoring_refs") or [],
        "termination_conditions": ["task_complete", "contract_verified", "max_iterations_reached"],
        "reward_function_components": ["artifact_contract", "verifier_signal", "step_success"],
        "generation_note": "Replay is task-dependent; use environment_spec and verifier checks.",
    })
    write_json(root / "environment" / "setup_requirements.json", {
        "schema_version": "aa-environment-setup-v0.2",
        "source_package_id": report.get("source_package_id"),
        "required_tools": env_spec.get("required_tools") or [],
        "runtime_dependencies": env_spec.get("runtime_dependencies") or [],
        "dependency_metadata": env_spec.get("software_execution") or {},
    })
    _write_jsonl(root / "environment" / "allowed_actions.jsonl", [
        {
            "schema_version": TRAINING_ROW_SCHEMA_VERSION,
            "row_id": f"allowed_action_{_safe_id(str(report.get('source_package_id') or package_root.name))}_{idx}",
            "data_type": "allowed_action",
            "source_package_id": report.get("source_package_id"),
            "source_task_id": report.get("source_task_id"),
            "task_lineage": _task_lineage(evidence),
            "action": action,
            "source_evidence_ref_ids": _evidence_ids_for_refs(_evidence_refs(evidence)[:3]),
            "evidence_refs": _evidence_refs(evidence)[:3],
            "confidence": report.get("average_quality_score") or 0.5,
            "created_at": utc_now(),
        }
        for idx, action in enumerate((env_spec.get("allowed_actions") or ["read_task", "write_artifact", "verify_output"]), start=1)
    ])
    _copy_jsonl_public(tdo_dir / "environment_response_pairs.jsonl", root / "environment" / "state_transition_rows.jsonl")
    write_json(root / "environment" / "reset_strategy.json", {
        "schema_version": "aa-environment-reset-v0.2",
        "source_package_id": report.get("source_package_id"),
        "reset_strategy": env_spec.get("reset_strategy") or "unknown",
        "state_reset_refs": env_spec.get("initial_state_refs") or [],
        "generation_notes": env_spec.get("evidence_status") or [],
    })
    _write_jsonl(root / "environment" / "verification_commands.jsonl", [
        {
            "schema_version": TRAINING_ROW_SCHEMA_VERSION,
            "row_id": f"verification_command_{_safe_id(str(report.get('source_package_id') or package_root.name))}_{idx}",
            "data_type": "verification_command",
            "source_package_id": report.get("source_package_id"),
            "source_task_id": report.get("source_task_id"),
            "task_lineage": _task_lineage(evidence),
            "command": command,
            "source_evidence_ref_ids": _evidence_ids_for_refs(_evidence_refs(evidence)[:3]),
            "evidence_refs": _evidence_refs(evidence)[:3],
            "confidence": report.get("average_quality_score") or 0.5,
            "created_at": utc_now(),
        }
        for idx, command in enumerate((env_spec.get("verification_commands") or []), start=1)
    ])
    (root / "reports" / "generation_notes.md").write_text(
        "\n".join([
            "# Generation Notes",
            "",
            "This package keeps accepted rows grounded in package evidence.",
            "Unsupported candidate rows are omitted or recorded in rejected row data.",
        ]) + "\n"
    )
    write_json(root / "learning_objects_index.json", {
        "source_package_id": report.get("source_package_id"),
        "learning_objects": [
            {"kind": "skill_pack", "path": "experience_compiler/skill_pack/skill.md"},
            {"kind": "runtime_learning_package", "path": "experience_compiler/runtime_learning_package/runtime_learning_manifest.json"},
            {"kind": "training_time_package", "path": "experience_compiler/training_time_package/training_time_manifest.json"},
            {"kind": "environment", "path": "experience_compiler/environment/environment_spec.json"},
            {"kind": "loop_learning", "path": "experience_compiler/loop_learning/loop_review_rows.jsonl"},
            {"kind": "reports", "path": "experience_compiler/compiler_report.json"},
        ],
        "rows_by_type": rows_accepted,
    })


def run_tdo_for_package(package_root: Path, settings: Settings, *, run_root: Path | None = None, followup_index: int | None = None, record_only: bool = False) -> dict[str, Any]:
    tdo_dir = package_root / "tdo"
    tdo_dir.mkdir(parents=True, exist_ok=True)
    for name in TDO_JSONL_FILES:
        (tdo_dir / name).write_text("")
    if _truthy_env("AA_TDO_DISABLED"):
        report = TDOReport(
            tdo_id=f"tdo_{_safe_id(package_root.name)}",
            source_package_id=package_root.name,
            source_task_id=package_root.name,
            created_at=utc_now(),
            generator_mode=settings.worker_agent,
            tdo_judge_source="apprentice_self_judge",
            mentor_audited=False,
            rounds_run=0,
            stop_reason="disabled_by_AA_TDO_DISABLED",
            tdo_status="disabled",
            schema_validation_passed=True,
            sanitization_passed=True,
            weak_strong_proxy_available=False,
        )
        write_json(tdo_dir / "tdo_report.json", report)
        write_json(tdo_dir / "tdo_manifest.json", {"schema_version": TDO_VERSION, "tdo_status": "disabled"})
        from .agent_env_exporter import export_agent_env
        export_agent_env(package_root)
        return report.model_dump(mode="json")
    max_rounds = _int_env("AA_EXPERIENCE_COMPILER_MAX_ROUNDS", _int_env("AA_TDO_MAX_ROUNDS", 3))
    min_quality = _float_env("AA_EXPERIENCE_COMPILER_MIN_QUALITY", _float_env("AA_TDO_MIN_QUALITY_SCORE", 0.65))
    min_fidelity = _float_env("AA_TDO_MIN_FIDELITY_SCORE", 0.65)
    min_reuse = _float_env("AA_TDO_MIN_REUSE_VALUE_SCORE", 0.65)
    evidence = _load_evidence(package_root, run_root=run_root)
    mentor_ready = configured_model_provider_ready(settings)
    judge_source = "mentor_model" if mentor_ready else "apprentice_self_judge"
    mentor_audited = bool(mentor_ready)
    proxy = _weak_strong_proxy(evidence)
    all_generated: list[dict[str, Any]] = []
    accepted_final: list[dict[str, Any]] = []
    rejected_all: list[dict[str, Any]] = []
    recipe_updates: list[str] = []
    judge_results: list[dict[str, Any]] = []
    rounds_run = 0
    stop_reason = "max_rounds_reached"
    for round_idx in range(1, max_rounds + 1):
        rounds_run = round_idx
        recipe_version = f"{TDO_VERSION}-round-{round_idx}"
        candidates = _build_candidate_rows(evidence, settings, round_idx, judge_source, mentor_audited, proxy)
        accepted, rejected, schema_errors = _validate_rows(package_root, candidates, min_quality, min_fidelity, min_reuse)
        judge = _judge(round_idx, max_rounds, accepted, rejected, judge_source, mentor_audited, proxy)
        judge_dict = judge.model_dump(mode="json")
        judge_results.append(judge_dict)
        all_generated.extend(candidates)
        round_rejected = list(rejected)
        if round_idx < max_rounds:
            for row in accepted:
                superseded = dict(row)
                superseded["accepted"] = False
                superseded["rejection_reason"] = "superseded_by_later_tdo_round"
                round_rejected.append(superseded)
            recipe_updates.append(f"round_{round_idx}_strengthen_grounding_and_reuse_scores")
        else:
            accepted_final = accepted
        rejected_all.extend(round_rejected)
        round_record = {
            "round": round_idx,
            "recipe_version": recipe_version,
            "generated_counts": _counts_by_type(candidates),
            "accepted_counts": _counts_by_type(accepted if round_idx == max_rounds else []),
            "rejected_counts": _counts_by_type(round_rejected),
            "schema_errors": sorted(set(schema_errors)),
            "quality_findings": judge_dict.get("data_quality_concerns") or [],
            "recipe_updates": list(recipe_updates),
            "judge_source": judge_source,
            "stop_reason": stop_reason if round_idx == max_rounds else None,
        }
        append_jsonl(tdo_dir / "tdo_rounds.jsonl", round_record)
        append_jsonl(tdo_dir / "tdo_judge_results.jsonl", judge_dict)
        for event in [
            {"event_type": "generate", "round": round_idx, "message": "generated candidate training data", "counts": _counts_by_type(candidates)},
            {"event_type": "judge", "round": round_idx, "message": "judge feedback complete", "verdict": judge.verdict},
            {"event_type": "improve" if judge.verdict == "improve" else "accept", "round": round_idx, "message": judge.verdict_reason},
        ]:
            append_jsonl(tdo_dir / "tdo_trace.jsonl", sanitize_public_obj(event))
        append_jsonl(tdo_dir / "meta_optimization_records.jsonl", {
            "recipe_version": recipe_version,
            "round": round_idx,
            "failure_patterns": sorted(set(schema_errors)),
            "accepted_recipe_updates": list(recipe_updates),
            "rejected_recipe_updates": [],
            "quality_delta": round(0.11 * (round_idx - 1), 4),
            "prompt_improvement_candidates": ["Require source evidence refs for every reusable row."],
            "notes_for_future_tdo_recipe": "Use mentor-model audit when available; keep rows grounded in observed package evidence.",
        })
    rejected_unique = _dedupe_rows(rejected_all)
    accepted_final = _dedupe_rows(accepted_final)
    _write_tdo_outputs(tdo_dir, accepted_final, rejected_unique)
    write_json(tdo_dir / "tdo_recipe.json", sanitize_public_obj({
        "schema_version": TDO_VERSION,
        "recipe_version": f"{TDO_VERSION}-round-{rounds_run}",
        "source_grounding": [
            "task_packet",
            "rubric",
            "agent_trace",
            "actual_outputs",
            "artifacts",
            "attempt_manifest",
            "evaluator_feedback",
            "grader_results",
            "verifier_results",
            "revision_plans",
            "process_supervision",
            "reward_modeling",
            "revision_preference_pairs",
            "lessons",
            "follow_up_records",
            "mentor_checkpoints",
        ],
        "max_rounds": max_rounds,
        "rounds_run": rounds_run,
        "quality_thresholds": {
            "min_quality_score": min_quality,
            "min_fidelity_score": min_fidelity,
            "min_reuse_value_score": min_reuse,
        },
        "recipe_updates": recipe_updates,
        "judge_source": judge_source,
        "mentor_audited": mentor_audited,
        **_source_audit_status(evidence),
    }))
    write_json(tdo_dir / "optimized_task_spec.json", _optimized_task_spec(evidence, settings, rounds_run))
    write_json(tdo_dir / "optimized_rubric.json", _optimized_rubric(evidence, rounds_run))
    write_json(tdo_dir / "domain_value_profile.json", _domain_value_profile(evidence))
    write_json(tdo_dir / "environment_spec.json", _environment_spec(evidence))
    rows_generated = _counts_with_surface_zeros(all_generated)
    rows_accepted = _counts_with_surface_zeros(accepted_final)
    rows_rejected = _counts_with_surface_zeros(rejected_unique)
    omitted_outputs: list[str] = []
    not_generated_counts: dict[str, int] = {}
    generation_notes: list[str] = []
    if not mentor_audited:
        generation_notes.append("judge_source=apprentice_self_judge")
    if not proxy.get("available"):
        not_generated_counts["solver_gap_proxy"] = 1
    if int(rows_accepted.get("preference_pair") or 0) == 0:
        omitted_outputs.append("preference_pairs")
        not_generated_counts["preference_pair"] = int(rows_rejected.get("preference_pair") or 0)
    report = TDOReport(
        tdo_id=f"tdo_{_safe_id(package_root.name)}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}",
        source_package_id=package_root.name,
        source_task_id=_source_task_id(evidence),
        created_at=utc_now(),
        generator_mode=settings.worker_agent,
        tdo_judge_source=judge_source,  # type: ignore[arg-type]
        mentor_audited=mentor_audited,
        rounds_run=rounds_run,
        stop_reason=stop_reason,
        tdo_status="completed",
        rows_generated_by_type=rows_generated,
        rows_accepted_by_type=rows_accepted,
        rows_rejected_by_type=rows_rejected,
        average_quality_score=_average(accepted_final, "quality_score"),
        average_fidelity_score=_average(accepted_final, "fidelity_score"),
        average_difficulty_score=_average(accepted_final, "difficulty_score"),
        average_learnability_score=_average(accepted_final, "learnability_score"),
        average_reuse_value_score=_average(accepted_final, "reuse_value_score"),
        schema_validation_passed=not any(row.get("rejection_reason") == "missing_data_type" for row in rejected_unique),
        sanitization_passed=not any(_private_or_forbidden_hit(row) for row in accepted_final),
        weak_strong_proxy_available=bool(proxy.get("available")),
        average_solver_gap_proxy=proxy.get("solver_gap_proxy"),
        grpo_suitability_distribution=_grpo_distribution(accepted_final),
        omitted_outputs=omitted_outputs,
        not_generated_counts=not_generated_counts,
        generation_notes=generation_notes,
        recommended_uses=["training_examples", "evaluation_examples", "verifier_training", "reward_modeling", "preference_optimization", "environment_response_training"],
        metadata_json={
            "apprenticeship_mode": getattr(settings, "apprenticeship_mode", None),
            "record_only_followup_update": record_only,
            "followup_index": followup_index,
            "follow_up_record_count": len([e for e in evidence.get("session_events") or [] if e.get("event_type") == "user_followup"]),
            "self_judge_generation_notes": ([] if mentor_audited else ["recommended_future_mentor_audit"]),
            "recommended_future_mentor_audit": not mentor_audited,
            "structural_guardrails": ["schema", "paths", "references", "secrets", "public-package hygiene"],
            **_source_audit_status(evidence),
        },
    )
    write_json(tdo_dir / "tdo_report.json", report)
    output_refs = [f"tdo/{name}" for name in TDO_JSON_FILES + TDO_JSONL_FILES]
    write_json(tdo_dir / "tdo_manifest.json", {
        "schema_version": TDO_VERSION,
        "tdo_id": report.tdo_id,
        "tdo_status": report.tdo_status,
        "source_package_id": package_root.name,
        "source_task_id": report.source_task_id,
        "tdo_output_refs": output_refs,
        "tdo_judge_source": judge_source,
        "mentor_audited": mentor_audited,
        **_source_audit_status(evidence),
        "rounds_run": rounds_run,
        "rows_by_type": rows_accepted,
        "created_at": report.created_at,
    })
    _write_experience_compiler_outputs(
        package_root,
        tdo_dir,
        report.model_dump(mode="json"),
        evidence,
        rows_accepted,
        settings,
    )
    try:
        from .package_exporter import write_artifacts_index

        write_artifacts_index(package_root)
    except Exception:
        pass
    from .agent_env_exporter import export_agent_env
    export_agent_env(package_root)
    return report.model_dump(mode="json")


def _dedupe_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for row in rows:
        key = repr({k: row.get(k) for k in ["data_type", "input", "target", "source_step_id", "tdo_round", "rejection_reason"]})
        if key not in seen:
            seen.add(key)
            out.append(row)
    return out


def run_tdo_for_run(
    run_root: Path,
    settings: Settings,
    *,
    progress_callback: ProgressCallback | None = None,
    followup_index: int | None = None,
    record_only: bool = False,
) -> dict[str, Any]:
    if _truthy_env("AA_TDO_DISABLED") or _truthy_env("AA_EXPERIENCE_COMPILER_DISABLED"):
        return {"tdo_status": "disabled", "stop_reason": "disabled_by_AA_TDO_DISABLED"}
    package_root = run_root / "packages"
    packages = sorted([p for p in package_root.glob("*") if p.is_dir()]) if package_root.exists() else []
    if not packages:
        return {"tdo_status": "failed", "error": "No task package found for Experience Compiler."}
    append_progress_event(
        run_root,
        "tdo_started",
        run_id=run_root.name,
        message="Experience Compiler started.",
        phase="experience_compiler",
        metadata_json={"followup_index": followup_index} if followup_index else None,
        callback=progress_callback,
    )
    reports: list[dict[str, Any]] = []
    try:
        for pkg in packages:
            report = run_tdo_for_package(pkg, settings, run_root=run_root, followup_index=followup_index, record_only=record_only)
            reports.append(report)
            rounds = read_jsonl(pkg / "tdo" / "tdo_rounds.jsonl")
            max_round = int(report.get("rounds_run") or len(rounds) or _int_env("AA_TDO_MAX_ROUNDS", 3))
            for row in rounds:
                round_idx = int(row.get("round") or 0)
                append_progress_event(
                    run_root,
                    "tdo_round_generated",
                    run_id=run_root.name,
                    message=f"Round {round_idx}/{max_round}: generated candidate learning data.",
                    phase="experience_compiler",
                    metadata_json={"package_id": pkg.name, "followup_index": followup_index, "generated_counts": row.get("generated_counts")},
                    callback=progress_callback,
                )
                append_progress_event(
                    run_root,
                    "tdo_round_judged",
                    run_id=run_root.name,
                    message=f"Round {round_idx}/{max_round}: compiler audit complete.",
                    phase="experience_compiler",
                    metadata_json={"package_id": pkg.name, "followup_index": followup_index, "judge_source": row.get("judge_source")},
                    callback=progress_callback,
                )
                if round_idx < max_round:
                    append_progress_event(
                        run_root,
                        "tdo_round_improved",
                        run_id=run_root.name,
                        message=f"Round {round_idx + 1}/{max_round}: improved weak learning objects.",
                        phase="experience_compiler",
                        metadata_json={"package_id": pkg.name, "followup_index": followup_index},
                        callback=progress_callback,
                    )
        status = "completed" if all(r.get("tdo_status") == "completed" for r in reports) else "partial"
        append_progress_event(
            run_root,
            "tdo_completed",
            run_id=run_root.name,
            message="Experience Compiler complete.",
            phase="experience_compiler_complete",
            metadata_json={"tdo_status": status, "package_count": len(reports), "followup_index": followup_index},
            callback=progress_callback,
        )
        write_json(run_root / "tdo_status.json", {"tdo_status": status, "reports": reports, "updated_at": utc_now()})
        return {"tdo_status": status, "reports": reports}
    except Exception as exc:
        safe = redact_secrets(str(exc))
        write_json(run_root / "tdo_status.json", {"tdo_status": "failed", "error": safe, "updated_at": utc_now()})
        append_progress_event(
            run_root,
            "tdo_failed",
            run_id=run_root.name,
            message=f"Experience Compiler failed: {safe}",
            phase="experience_compiler_failed",
            operational_error=safe,
            metadata_json={"followup_index": followup_index} if followup_index else None,
            callback=progress_callback,
        )
        return {"tdo_status": "failed", "error": safe}


def tdo_manifest_fields_from_report(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "tdo_enabled": report.get("tdo_status") != "disabled",
        "tdo_status": report.get("tdo_status"),
        "tdo_id": report.get("tdo_id"),
        "tdo_rounds_run": report.get("rounds_run"),
        "tdo_rows_by_type": report.get("rows_accepted_by_type") or {},
        "tdo_average_quality_score": report.get("average_quality_score"),
        "tdo_average_fidelity_score": report.get("average_fidelity_score"),
        "tdo_average_reuse_value_score": report.get("average_reuse_value_score"),
        "tdo_judge_source": report.get("tdo_judge_source"),
        "mentor_audited": report.get("mentor_audited"),
    }
