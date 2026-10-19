"""Validate inbound ATIF; never guess or repair an incompatible trace schema."""
from __future__ import annotations
from pathlib import Path
from typing import Any
from pydantic import BaseModel, Field
from .schemas import ActualOutputs, WorkflowTrace
from .atif import load_trace, to_workflow, from_workflow, validate_trace

class TraceNormalizationContext(BaseModel):
    task_id: str
    attempt_id: str
    attempt_kind: str
    task: str
    actual_outputs: ActualOutputs | None = None
    trace_id_prefix: str = 'trace'

class TraceNormalizationReport(BaseModel):
    task_id: str
    attempt_id: str
    attempt_kind: str
    raw_trace_ref: str | None = None
    normalized_trace_ref: str | None = None
    canonical_trace_ref: str | None = None
    trace_schema_valid: bool = False
    trace_normalized: bool = False
    trace_lossless: bool = True
    fallback_trace: bool = False
    raw_step_count: int = 0
    normalized_step_count: int = 0
    discarded_step_count: int = 0
    raw_trace_parse_error: bool = False
    trace_normalization_error: bool = False
    trace_normalization_partial: bool = False
    validation_errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    field_mappings: dict[str,str] = Field(default_factory=dict)
    input_dict_to_string_count: int = 0
    output_dict_to_string_count: int = 0
    causal_type_repair_count: int = 0
    operation_repair_count: int = 0
    action_repair_count: int = 0
    metadata_json: dict[str, Any] = Field(default_factory=dict)

class NormalizationResult(BaseModel):
    normalized_trace: dict[str, Any] | None = None
    report: TraceNormalizationReport
    fallback_required: bool = False
    parse_error: str | None = None


def infer_attempt_status(trace: dict[str, Any], report: dict[str, Any] | None = None) -> str:
    existing=trace.get('attempt_status')
    if existing in {'completed', 'failed', 'blocked', 'fallback', 'partial'}:
        return existing
    termination=trace.get('termination_reason')
    md=trace.get('metadata_json') if isinstance(trace.get('metadata_json'), dict) else {}
    report=report or {}
    if termination == 'task_complete':
        return 'completed'
    if termination == 'agent_blocked':
        return 'blocked'
    actual=trace.get('actual_outputs') if isinstance(trace.get('actual_outputs'), dict) else {}
    if termination in {'timeout','error_unrecoverable','provider_usage_limit','verifier_failed'} or actual.get('status') in {'failed','timeout','error'}:
        return 'failed'
    if report.get('fallback_trace') or md.get('fallback_trace') or md.get('fallback_trace_created') or md.get('fallback_reason') or report.get('fallback_reason') or (isinstance(report.get('metadata_json'), dict) and report['metadata_json'].get('fallback_reason')):
        return 'fallback'
    steps=trace.get('steps') or []
    if any(s.get('step_outcome') == 'blocked' for s in steps if isinstance(s, dict)):
        return 'blocked'
    if any(s.get('step_outcome') == 'failed' or s.get('success') is False for s in steps if isinstance(s, dict)):
        return 'failed' if not any(s.get('success') is True for s in steps if isinstance(s, dict)) else 'partial'
    if steps:
        return 'partial'
    return 'failed'

def normalize_trace_for_export(trace: dict, report: dict | None = None) -> dict:
    projection = to_workflow(trace) if "agent" in trace else dict(trace)
    projection["attempt_status"] = infer_attempt_status(projection, report)
    return from_workflow(projection)


def repair_agent_trace_file(raw_trace_path: Path, context: TraceNormalizationContext) -> NormalizationResult:
    report = TraceNormalizationReport(task_id=context.task_id, attempt_id=context.attempt_id,
        attempt_kind=context.attempt_kind, raw_trace_ref=raw_trace_path.name)
    try:
        atif = load_trace(raw_trace_path)
        trace = to_workflow(atif)
        trace.update(task_id=context.task_id, attempt_id=context.attempt_id,
                     attempt_kind=context.attempt_kind, task=context.task)
        if context.actual_outputs is not None:
            trace['actual_outputs'] = context.actual_outputs.model_dump(mode='json')
        WorkflowTrace.model_validate(trace)
        normalized = from_workflow(trace)
    except (ValueError, TypeError, OSError) as error:
        report.trace_normalization_error = True
        report.trace_lossless = False
        report.validation_errors.append(str(error))
        return NormalizationResult(report=report, fallback_required=True, parse_error=str(error))
    report.raw_step_count = report.normalized_step_count = len(atif['steps'])
    report.trace_schema_valid = report.trace_normalized = True
    report.normalized_trace_ref = 'agent_trace.normalized.json'
    report.canonical_trace_ref = 'agent_trace.json'
    return NormalizationResult(normalized_trace=normalized, report=report)
