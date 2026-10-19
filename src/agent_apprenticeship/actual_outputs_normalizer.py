from __future__ import annotations
import json
from pathlib import Path
from typing import Any
from pydantic import BaseModel, Field
from .schemas import ActualOutputs
from .io import write_json
from .openai_structured import extract_json_object

CANONICAL_ACTUAL_FIELDS = set(ActualOutputs.model_fields.keys())

class ActualOutputsNormalizationContext(BaseModel):
    task_id: str
    attempt_id: str
    attempt_kind: str
    package_root: Path
    required_artifacts: list[str] = Field(default_factory=list)

class ActualOutputsNormalizationReport(BaseModel):
    task_id: str
    attempt_id: str
    attempt_kind: str
    raw_outputs_ref: str | None = None
    invalid_outputs_ref: str | None = None
    normalized_outputs_ref: str | None = None
    canonical_outputs_ref: str | None = None
    actual_outputs_schema_valid: bool = False
    actual_outputs_normalized: bool = False
    actual_outputs_fallback: bool = False
    actual_outputs_raw_count: int = 0
    actual_outputs_normalized_count: int = 0
    actual_outputs_schema_valid_count: int = 0
    actual_outputs_fallback_count: int = 0
    actual_outputs_inferred_artifact_count: int = 0
    actual_outputs_discarded_field_count: int = 0
    validation_errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    metadata_json: dict[str, Any] = Field(default_factory=dict)

class ActualOutputsNormalizationResult(BaseModel):
    actual_outputs: dict[str, Any] | None = None
    report: ActualOutputsNormalizationReport
    fallback_required: bool = False
    parse_error: str | None = None


def _as_list(v: Any) -> list[Any]:
    if v is None:
        return []
    if isinstance(v, list):
        return v
    return [v]


def _compact(v: Any, max_len: int=2000) -> str:
    if v is None:
        return ''
    if isinstance(v, str):
        return v[:max_len]
    try:
        return json.dumps(v, sort_keys=True)[:max_len]
    except Exception:
        return str(v)[:max_len]


def _validation_errors(exc: Exception) -> list[str]:
    if hasattr(exc, 'errors'):
        try:
            return [_compact(e, 2000) for e in exc.errors()]
        except Exception:
            pass
    return [str(exc)]


def _norm_ref(ref: str, attempt_kind: str) -> str:
    ref=str(ref).replace("\\", "/").strip().lstrip("/")
    while ref.startswith("./"):
        ref = ref[2:]
    if not ref:
        return ref
    if ref in {"actual_outputs.json", "agent_trace.json", "stdout.txt", "stderr.txt", "final_message.txt"}:
        return f"attempts/{attempt_kind}/{ref}"
    if ref.startswith("input/") or ref.startswith("task/"):
        return ref
    if ref.startswith(f'attempts/{attempt_kind}/'):
        return ref
    if ref.startswith('artifacts/'):
        return f'attempts/{attempt_kind}/{ref}'
    if '/' not in ref:
        return f'attempts/{attempt_kind}/artifacts/{ref}'
    return ref


def _is_input_ref(ref: str) -> bool:
    return ref.startswith("input/") or ref.startswith("task/") or "/input/" in ref or "/task/" in ref


def _is_attempt_metadata_ref(ref: str, attempt_kind: str) -> bool:
    return ref in {
        f"attempts/{attempt_kind}/actual_outputs.json",
        f"attempts/{attempt_kind}/agent_trace.json",
        f"attempts/{attempt_kind}/stdout.txt",
        f"attempts/{attempt_kind}/stderr.txt",
        f"attempts/{attempt_kind}/final_message.txt",
    }


def _generated_ref_list(value: Any, attempt_kind: str) -> list[str]:
    refs=[]
    for item in _as_list(value):
        if not isinstance(item, str):
            continue
        normalized=_norm_ref(item, attempt_kind)
        if _is_input_ref(normalized) or _is_attempt_metadata_ref(normalized, attempt_kind):
            continue
        if normalized and normalized not in refs:
            refs.append(normalized)
    return refs


def _input_ref_list(value: Any, attempt_kind: str) -> list[str]:
    refs=[]
    for item in _as_list(value):
        if not isinstance(item, str):
            continue
        normalized=_norm_ref(item, attempt_kind)
        if normalized and _is_input_ref(normalized) and normalized not in refs:
            refs.append(normalized)
    return refs


def _artifact_paths_from_raw(raw: dict[str, Any]) -> dict[str, Any]:
    found={}
    for k,v in raw.items():
        key = k[2:] if isinstance(k, str) and k.startswith("./") else k
        if isinstance(key, str) and (key.startswith('artifacts/') or '/artifacts/' in key):
            found[key]=v
    return found


def _scan_existing_artifacts(ctx: ActualOutputsNormalizationContext) -> list[str]:
    artifact_dir=ctx.package_root/'attempts'/ctx.attempt_kind/'artifacts'
    refs=[]
    if artifact_dir.exists():
        required_names={Path(x).name for x in ctx.required_artifacts if x}
        files=[p for p in artifact_dir.iterdir() if p.is_file()]
        if required_names:
            files=[p for p in files if p.name in required_names] or files
        refs=[f'attempts/{ctx.attempt_kind}/artifacts/{p.name}' for p in files]
    return refs


def _extract_refs(raw: dict[str, Any], ctx: ActualOutputsNormalizationContext) -> tuple[list[str], list[str], dict[str, Any], list[str]]:
    warnings=[]
    refs=[]
    input_refs=[]
    original_path_fields=_artifact_paths_from_raw(raw)
    for k in original_path_fields:
        normalized = _norm_ref(k, ctx.attempt_kind)
        (input_refs if _is_input_ref(normalized) else refs).append(normalized)
    for key in ['deliverable_refs','artifact_refs','files_created','outputs','artifacts','output_files']:
        val=raw.get(key)
        if isinstance(val, dict):
            for k in val:
                if isinstance(k, str) and (k.startswith('artifacts/') or '/artifacts/' in k or '.' in Path(k).name):
                    normalized = _norm_ref(k, ctx.attempt_kind)
                    if _is_input_ref(normalized):
                        input_refs.append(normalized)
                    elif not _is_attempt_metadata_ref(normalized, ctx.attempt_kind):
                        refs.append(normalized)
        else:
            for item in _as_list(val):
                if isinstance(item, str):
                    normalized = _norm_ref(item, ctx.attempt_kind)
                    if _is_input_ref(normalized):
                        input_refs.append(normalized)
                    elif not _is_attempt_metadata_ref(normalized, ctx.attempt_kind):
                        refs.append(normalized)
    for item in _as_list(raw.get("input_artifact_refs")):
        if isinstance(item, str):
            input_refs.append(_norm_ref(item, ctx.attempt_kind))
    existing=_scan_existing_artifacts(ctx)
    refs.extend(existing)
    dedup=[]
    for r in refs:
        if r and r not in dedup:
            dedup.append(r)
    input_dedup=[]
    for r in input_refs:
        if r and r not in input_dedup:
            input_dedup.append(r)
    return dedup, input_dedup, original_path_fields, warnings


def normalize_actual_outputs(raw: dict[str, Any] | None, ctx: ActualOutputsNormalizationContext) -> ActualOutputsNormalizationResult:
    raw = dict(raw or {})
    report=ActualOutputsNormalizationReport(task_id=ctx.task_id, attempt_id=ctx.attempt_id, attempt_kind=ctx.attempt_kind, actual_outputs_raw_count=1)
    refs, input_refs, original_path_fields, warnings = _extract_refs(raw, ctx)
    original_fields={k:v for k,v in raw.items() if k not in CANONICAL_ACTUAL_FIELDS}
    raw_status=str(raw.get('status') or '').lower()
    failure_status=raw_status in {'failed','error','timeout'}
    status = raw.get('status') if raw.get('status') in {'success','partial','failed','timeout','error'} else None
    if status is None:
        status='success' if refs and not failure_status else ('failed' if failure_status else ('partial' if refs else 'failed'))
    if refs and not failure_status and status == 'failed':
        status='success'
    summary=raw.get('output_summary') or raw.get('summary') or raw.get('final_summary') or ('Normalized actual outputs from artifact files.' if refs else 'No canonical actual outputs were produced.')
    md=dict(raw.get('metadata_json') or {}) if isinstance(raw.get('metadata_json'), dict) else {}
    if original_fields:
        md['original_fields']=original_fields
    if original_path_fields:
        md['original_artifact_path_fields']=original_path_fields
    md['raw_actual_outputs']=raw
    md['actual_outputs_discarded_field_count']=0
    md['actual_outputs_normalized']=True
    md['expected_deliverable_items']=[Path(str(x)).name for x in ctx.required_artifacts if x]
    md['produced_deliverable_items']=[Path(str(x)).name for x in refs]
    actual={
        'task_id': str(raw.get('task_id') or ctx.task_id),
        'attempt_id': str(raw.get('attempt_id') or ctx.attempt_id),
        'attempt_kind': str(raw.get('attempt_kind') or ctx.attempt_kind),
        'status': status,
        'output_summary': str(summary),
        'primary_output_ref': _generated_ref_list(raw.get('primary_output_ref'), ctx.attempt_kind)[0] if _generated_ref_list(raw.get('primary_output_ref'), ctx.attempt_kind) else (refs[0] if refs else None),
        'input_artifact_refs': _input_ref_list(raw.get('input_artifact_refs'), ctx.attempt_kind) or input_refs,
        'deliverable_refs': _generated_ref_list(raw.get('deliverable_refs'), ctx.attempt_kind) or refs,
        'final_message_ref': raw.get('final_message_ref') or f'attempts/{ctx.attempt_kind}/final_message.txt',
        'artifact_refs': _generated_ref_list(raw.get('artifact_refs'), ctx.attempt_kind) or refs,
        'files_created': _generated_ref_list(raw.get('files_created'), ctx.attempt_kind) or refs,
        'files_modified': [_norm_ref(str(x), ctx.attempt_kind) for x in (raw.get('files_modified') if isinstance(raw.get('files_modified'), list) else [])],
        'files_deleted': [_norm_ref(str(x), ctx.attempt_kind) for x in (raw.get('files_deleted') if isinstance(raw.get('files_deleted'), list) else [])],
        'stdout_ref': raw.get('stdout_ref') or f'attempts/{ctx.attempt_kind}/stdout.txt',
        'stderr_ref': raw.get('stderr_ref') or f'attempts/{ctx.attempt_kind}/stderr.txt',
        'raw_log_refs': [str(x) for x in (raw.get('raw_log_refs') if isinstance(raw.get('raw_log_refs'), list) else [])] or [f'attempts/{ctx.attempt_kind}/stdout.txt', f'attempts/{ctx.attempt_kind}/stderr.txt', f'attempts/{ctx.attempt_kind}/final_message.txt'],
        'error_type': raw.get('error_type') if status in {'failed','timeout','error'} else None,
        'error_message': raw.get('error_message') if status in {'failed','timeout','error'} else None,
        'metadata_json': md,
    }
    try:
        obj=ActualOutputs.model_validate(actual)
        normalized=obj.model_dump(mode='json')
        report.actual_outputs_schema_valid=True
        report.actual_outputs_schema_valid_count=1
    except Exception as exc:
        normalized=actual
        report.validation_errors.extend(_validation_errors(exc))
    report.actual_outputs_normalized=True
    report.actual_outputs_normalized_count=1
    report.actual_outputs_inferred_artifact_count=len(refs)
    report.actual_outputs_discarded_field_count=0
    report.warnings.extend(warnings)
    return ActualOutputsNormalizationResult(actual_outputs=normalized, report=report, fallback_required=not report.actual_outputs_schema_valid)


def repair_actual_outputs_file(path: Path, ctx: ActualOutputsNormalizationContext) -> ActualOutputsNormalizationResult:
    report=ActualOutputsNormalizationReport(task_id=ctx.task_id, attempt_id=ctx.attempt_id, attempt_kind=ctx.attempt_kind, raw_outputs_ref='actual_outputs.raw.json')
    if not path.exists():
        existing=_scan_existing_artifacts(ctx)
        if existing:
            result=normalize_actual_outputs({}, ctx)
            result.report.raw_outputs_ref=None
            return result
        report.actual_outputs_fallback=True
        report.actual_outputs_fallback_count=1
        report.validation_errors.append('actual_outputs.json missing and no artifact evidence was available')
        return ActualOutputsNormalizationResult(actual_outputs=None, report=report, fallback_required=True, parse_error='missing actual_outputs.json')
    text=path.read_text()
    raw_path=path.with_name('actual_outputs.raw.json')
    raw_path.write_text(text)
    json_meta: dict[str, Any] = {}
    parse_warnings: list[str] = []
    try:
        raw=json.loads(text)
        if not isinstance(raw, dict):
            raise ValueError('actual_outputs JSON was not an object')
    except Exception as first_exc:
        try:
            raw, json_meta = extract_json_object(text, return_metadata=True)
            if not isinstance(raw, dict):
                raise ValueError('actual_outputs JSON was not an object')
            parse_warnings.append(f"actual_outputs parsed from embedded JSON after raw parse failed: {first_exc}")
        except Exception as exc:
            report.actual_outputs_fallback=True
            report.actual_outputs_fallback_count=1
            report.validation_errors.append(str(first_exc))
            if str(exc) != str(first_exc):
                report.validation_errors.append(str(exc))
            return ActualOutputsNormalizationResult(actual_outputs=None, report=report, fallback_required=True, parse_error=str(first_exc))
    try:
        ActualOutputs.model_validate(raw)
        result=normalize_actual_outputs(raw, ctx)
        result.report.raw_outputs_ref='actual_outputs.raw.json'
        result.report.canonical_outputs_ref='actual_outputs.json'
        result.report.warnings.extend(parse_warnings)
        result.report.metadata_json.update(json_meta)
        return result
    except Exception as exc:
        path.with_name('actual_outputs.invalid.json').write_text(text)
        result=normalize_actual_outputs(raw, ctx)
        result.report.raw_outputs_ref='actual_outputs.raw.json'
        result.report.invalid_outputs_ref='actual_outputs.invalid.json'
        result.report.normalized_outputs_ref='actual_outputs.normalized.json'
        result.report.canonical_outputs_ref='actual_outputs.json'
        result.report.validation_errors.extend(_validation_errors(exc))
        result.report.warnings.extend(parse_warnings)
        result.report.metadata_json.update(json_meta)
        return result


def write_actual_outputs_normalization(attempt_dir: Path, result: ActualOutputsNormalizationResult) -> None:
    if result.actual_outputs is not None:
        write_json(attempt_dir/'actual_outputs.normalized.json', result.actual_outputs)
        write_json(attempt_dir/'actual_outputs.json', result.actual_outputs)
        result.report.normalized_outputs_ref=result.report.normalized_outputs_ref or 'actual_outputs.normalized.json'
        result.report.canonical_outputs_ref=result.report.canonical_outputs_ref or 'actual_outputs.json'
    write_json(attempt_dir/'actual_outputs_normalization_report.json', result.report)
