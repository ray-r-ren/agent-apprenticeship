from __future__ import annotations
import json
from pathlib import Path
from pydantic import BaseModel
from .schemas import RawTaskRecord, TaskIntakeSpec, TaskIntakeQualityReport
from .config import apprentice_agent_display_name, get_settings
from .io import read_workflow_json as read_json, write_json
from .openai_structured import extract_json_object, get_model_provider_status, run_structured_role
from .public_sanitizer import sanitize_public_obj, sha256_text

class LLMTaskIntakeOutput(BaseModel):
    task_intake_spec: TaskIntakeSpec
    task_intake_quality_report: TaskIntakeQualityReport

SOURCE_FIELD_KEYS={'source_url_or_ref','source_kind','source_url','source_ref','source_basis','source_license'}

def _mentor_provider_can_attempt() -> bool:
    return bool(get_model_provider_status().get('provider_available'))

def _mentor_provider_id() -> str:
    settings=get_settings()
    return settings.model_provider or 'openai'

def _drop_source_fields(obj):
    if isinstance(obj, list):
        return [_drop_source_fields(v) for v in obj]
    if isinstance(obj, dict):
        return {k:_drop_source_fields(v) for k,v in obj.items() if k not in SOURCE_FIELD_KEYS and v is not None}
    return obj

def _task_record_for_intake(raw: RawTaskRecord) -> dict:
    data=raw.model_dump(mode='json')
    return sanitize_public_obj(_drop_source_fields(data))

def _sanitize_intake_output_obj(obj):
    return sanitize_public_obj(_drop_source_fields(obj))

def _sanitize_intake_role_artifacts(role_dir: Path) -> None:
    for name in ("parsed_output.json", "raw_parsed_output.json"):
        path = role_dir / name
        if path.exists():
            try:
                write_json(path, _sanitize_intake_output_obj(read_json(path)))
            except Exception:
                pass
    for name in ("raw_output.txt", "raw_output.retry.txt"):
        path = role_dir / name
        if not path.exists():
            continue
        try:
            parsed = extract_json_object(path.read_text(errors="ignore"))
            path.write_text(json.dumps(_sanitize_intake_output_obj(parsed), indent=2, sort_keys=True) + "\n")
        except Exception:
            text = path.read_text(errors="ignore")
            for key in SOURCE_FIELD_KEYS:
                text = text.replace(key, "reference_id")
            path.write_text(text)

def _sanitize_spec_and_quality(
    spec: TaskIntakeSpec,
    quality: TaskIntakeQualityReport | None = None,
) -> tuple[TaskIntakeSpec, TaskIntakeQualityReport | None]:
    spec_updates = {
        "metadata_json": _sanitize_intake_output_obj(spec.metadata_json or {}),
        "expected_pay": None,
        "expected_apprentice_pay": None,
    }
    sanitized_quality = None
    if quality is not None:
        sanitized_quality = quality.model_copy(
            update={"metadata_json": _sanitize_intake_output_obj(quality.metadata_json or {})}
        )
    return spec.model_copy(update=spec_updates), sanitized_quality

def direct_task_sheet_metadata(raw: RawTaskRecord) -> dict[str, object]:
    payload=raw.raw_payload or {}
    expected_economic_value = raw.expected_economic_value or payload.get('expected_economic_value') or raw.expected_pay or payload.get('expected_pay')
    expected_economic_value_for_agent_apprentice = (
        raw.expected_economic_value_for_agent_apprentice
        or payload.get('expected_economic_value_for_agent_apprentice')
        or raw.expected_apprentice_pay
        or payload.get('expected_apprentice_pay')
    )
    return {
        'domain': raw.normalized_domain or payload.get('normalized_domain') or payload.get('domain'),
        'subdomain': raw.normalized_subdomain or payload.get('normalized_subdomain') or payload.get('subdomain'),
        'apprenticeship_role': raw.apprenticeship_role or payload.get('apprenticeship_role'),
        'expected_economic_value': expected_economic_value,
        'expected_economic_value_for_agent_apprentice': expected_economic_value_for_agent_apprentice,
        'expected_pay': expected_economic_value,
        'expected_apprentice_pay': expected_economic_value_for_agent_apprentice,
        'task_family': raw.task_family or payload.get('task_family'),
        'difficulty_tier': raw.difficulty_tier or payload.get('difficulty_tier'),
        'needs_expert_review': raw.needs_expert_review if raw.needs_expert_review is not None else payload.get('needs_expert_review'),
    }

def apply_direct_task_sheet_metadata(spec: TaskIntakeSpec, raw: RawTaskRecord) -> TaskIntakeSpec:
    direct={k:v for k,v in direct_task_sheet_metadata(raw).items() if v is not None}
    updates={}
    for src,dst in [('domain','domain'),('subdomain','subdomain'),('apprenticeship_role','apprenticeship_role'),('expected_economic_value','expected_economic_value'),('expected_economic_value_for_agent_apprentice','expected_economic_value_for_agent_apprentice'),('expected_pay','expected_pay'),('expected_apprentice_pay','expected_apprentice_pay'),('task_family','task_family'),('difficulty_tier','difficulty_tier'),('needs_expert_review','needs_expert_review')]:
        if src in direct:
            updates[dst]=direct[src]
    md=dict(spec.metadata_json or {})
    md['direct_task_sheet_fields']={k:v for k,v in direct.items() if k not in {'expected_pay','expected_apprentice_pay'}}
    md['direct_task_sheet_fields_preserved']=bool(direct)
    return spec.model_copy(update={**updates, 'metadata_json': md})

def _apprentice_draft_metadata(raw: RawTaskRecord, expected: str) -> dict[str, object]:
    settings = get_settings()
    payload = raw.raw_payload or {}
    constraints = payload.get('constraints') or []
    output_requirements = payload.get('output_requirements') or []
    environment_domain = payload.get('environment_domain') or payload.get('environment') or payload.get('domain') or 'unknown'
    environment_type = payload.get('environment_type') or environment_domain or 'unknown'
    role = payload.get('role') or payload.get('professional_role') or payload.get('apprenticeship_role') or raw.apprenticeship_role
    return {
        'intake_source': 'apprentice_agent_draft',
        'draft_author': 'apprentice_agent',
        'apprentice_agent_id': settings.worker_agent,
        'apprentice_agent_name': apprentice_agent_display_name(settings),
        'mentor_audited': False,
        'mentor_audit_source': None,
        'audit_status': 'unaudited',
        'unaudited_reason': 'No Mentor Model or human audit was completed for this task-intake draft.',
        'role': role or 'Apprentice Agent',
        'expected_deliverables': [expected],
        'success_criteria': output_requirements or [f'Produce {expected}.'],
        'failure_modes': payload.get('failure_modes') or ['Missing required deliverable', 'Output does not satisfy the task instruction'],
        'expected_economic_value_rationale': payload.get('expected_economic_value_rationale') or 'Estimated from task domain, deliverable scope, and specialization signals.',
        'expected_economic_value_for_apprentice_agent_rationale': payload.get('expected_economic_value_for_agent_apprentice_rationale') or payload.get('expected_economic_value_for_apprentice_agent_rationale') or 'Estimated from the agent-executable portion of the workflow.',
        'environment_domain': environment_domain,
        'environment_type': environment_type,
        'training_use_cases': payload.get('training_use_cases') or ['task_following', 'artifact_generation', 'workflow_trace_learning'],
        'evaluation_use_cases': payload.get('evaluation_use_cases') or ['rubric_based_evaluation', 'artifact_contract_checking'],
        'verifier_use_cases': payload.get('verifier_use_cases') or ['artifact_presence_verification', 'evidence_grounding_verification'],
        'artifact_requirements': output_requirements or [expected],
        'trace_requirements': payload.get('trace_requirements') or ['record one meaningful action per step', 'reference generated artifacts', 'separate observation, input, output, and state_change'],
        'constraints': constraints,
    }

def deterministic_intake(raw: RawTaskRecord) -> tuple[TaskIntakeSpec, TaskIntakeQualityReport]:
    tid = raw.raw_task_id.replace('raw_','task_')
    direct=direct_task_sheet_metadata(raw)
    expected=raw.raw_payload.get('expected_deliverable') or raw.expected_deliverable or raw.raw_payload.get('expected_agent_deliverable','Completed deliverables and audit notes')
    spec=TaskIntakeSpec(task_id=tid, normalized_title=raw.raw_title, normalized_instruction=raw.raw_description, domain=direct.get('domain') or raw.raw_payload.get('domain','general'), subdomain=direct.get('subdomain') or raw.raw_payload.get('subdomain'), professional_role=raw.raw_payload.get('professional_role'), apprenticeship_role=direct.get('apprenticeship_role'), task_family=direct.get('task_family'), expected_economic_value=direct.get('expected_economic_value'), expected_economic_value_for_agent_apprentice=direct.get('expected_economic_value_for_agent_apprentice'), expected_pay=direct.get('expected_pay'), expected_apprentice_pay=direct.get('expected_apprentice_pay'), workflow_type=raw.raw_payload.get('workflow_type','analysis'), skill_targets=raw.raw_payload.get('skill_targets',['analysis','artifact_generation']), difficulty_tier=direct.get('difficulty_tier') or raw.raw_payload.get('difficulty_tier','medium'), expected_human_deliverable=raw.raw_payload.get('expected_human_deliverable', expected), expected_agent_deliverable=expected, input_requirements=raw.raw_payload.get('input_requirements',[]), output_requirements=raw.raw_payload.get('output_requirements',[]), required_context=raw.raw_payload.get('required_context',[]), assumptions=[], constraints=raw.raw_payload.get('constraints',[]), allowed_tools=['python','file_read','file_write','bash'], disallowed_tools=['browser'], privacy_classification=raw.raw_payload.get('privacy_classification','unknown'), license=None, allowed_use='local apprenticeship data generation', rubricability_score=0.8, verifiability_score=0.8, artifactability_score=0.9, needs_expert_review=bool(direct.get('needs_expert_review')) if direct.get('needs_expert_review') is not None else False, metadata_json=_apprentice_draft_metadata(raw, expected))
    spec=apply_direct_task_sheet_metadata(spec, raw)
    settings=get_settings()
    if settings.rubric_mode in {'hybrid','llm_default'} and _mentor_provider_can_attempt() and settings.llm_task_intake_enabled:
        spec.metadata_json.update({'llm_task_intake_enabled': True, 'mentor_audit_status': 'not_completed', 'provider': _mentor_provider_id()})
    q=TaskIntakeQualityReport(task_id=tid, instruction_clarity_score=0.8, input_completeness_score=0.8, output_contract_score=0.9, rubricability_score=spec.rubricability_score, verifiability_score=spec.verifiability_score, artifactability_score=spec.artifactability_score, privacy_risk_score=0.1, license_risk_score=0.1, ambiguity_score=0.2, overall_intake_quality_score=0.82, quality_flags=['mentor_audit_pending'], blockers=[], recommended_fix=None, metadata_json={'draft_author':'apprentice_agent','mentor_audited':False})
    return spec,q

def _intake_prompt(raw: RawTaskRecord) -> str:
    return """Return only valid JSON. Do not include markdown. Do not add extra top-level fields; place extras under metadata_json.extra_model_fields.
Required skeleton: {"task_intake_spec":{"task_id":"...","normalized_title":"...","normalized_instruction":"...","domain":"general","workflow_type":"artifact_generation","difficulty_tier":"medium","expected_human_deliverable":"...","expected_agent_deliverable":"...","input_requirements":[],"output_requirements":[],"required_context":[],"assumptions":[],"constraints":[],"allowed_tools":[],"disallowed_tools":[],"privacy_classification":"unknown","rubricability_score":0.7,"verifiability_score":0.7,"artifactability_score":0.7,"needs_expert_review":false,"metadata_json":{}},"task_intake_quality_report":{"task_id":"...","instruction_clarity_score":0.7,"input_completeness_score":0.7,"output_contract_score":0.7,"rubricability_score":0.7,"verifiability_score":0.7,"artifactability_score":0.7,"privacy_risk_score":0.2,"license_risk_score":0.2,"ambiguity_score":0.3,"overall_intake_quality_score":0.7,"quality_flags":[],"blockers":[],"metadata_json":{}}}.
Audit and improve an Apprentice Agent-generated task intake draft for reusable agent work experience. Keep the Apprentice Agent as the draft author; your role is Mentor audit/improvement. Populate title, domain, subdomain, agent apprenticeship role, expected_economic_value, expected_economic_value_for_apprentice_agent, expected_economic_value rationale, environment_domain, environment_type, training_use_cases, evaluation_use_cases, verifier_use_cases, artifact_requirements, required_artifacts, trace_requirements, success_criteria, and failure_modes when inferable from the task. Put fields that are not first-class schema fields under metadata_json. Do not include commerce metadata, source URL fields, or agent self-evaluation claims.
Raw task JSON:
""" + json.dumps(_task_record_for_intake(raw), sort_keys=True)

def task_intake(raw: RawTaskRecord, role_root: Path | None=None) -> tuple[TaskIntakeSpec, TaskIntakeQualityReport]:
    settings=get_settings()
    role_root=role_root or Path('outputs/roles')
    if settings.llm_task_intake_enabled and settings.rubric_mode in {'hybrid','llm_default','llm_required'} and _mentor_provider_can_attempt():
        prompt=_intake_prompt(raw)
        try:
            provider=_mentor_provider_id()
            model_override=settings.llm_task_intake_model if provider == 'openai' else None
            rr=run_structured_role('intake_agent', prompt, LLMTaskIntakeOutput, role_root/'intake_agent', allow_fallback=settings.allow_deterministic_eval_fallback, model_override=model_override, normalizer_context={'task_id': raw.raw_task_id.replace('raw_','task_'), 'task_title': raw.raw_title, 'task_instruction': raw.raw_description, 'model': model_override or settings.model_provider_model, 'provider':provider})
            if rr.live_call_ok and rr.structured_output_validation_ok:
                _sanitize_intake_role_artifacts(role_root/'intake_agent')
                parsed=read_json(role_root/'intake_agent/parsed_output.json')
                spec=TaskIntakeSpec.model_validate(parsed['task_intake_spec'])
                spec=apply_direct_task_sheet_metadata(spec, raw)
                q=TaskIntakeQualityReport.model_validate(parsed['task_intake_quality_report'])
                draft_md = _apprentice_draft_metadata(raw, spec.expected_agent_deliverable)
                spec.metadata_json.update({**draft_md,'intake_source':'apprentice_agent_draft','draft_author':'apprentice_agent','mentor_audited':True,'mentor_audit_source':'mentor_model','audit_status':'mentor_audited','provider':rr.provider,'model':rr.model,'llm_prompt_ref_internal':str(role_root/'intake_agent/prompt.md'),'llm_response_ref_internal':str(role_root/'intake_agent/raw_output.txt'),'prompt_template_id':'task_intake_agent_v0','prompt_template_version':'0.1','prompt_hash':sha256_text(prompt),'public_response_summary':'Mentor Model audited and improved an Apprentice-generated task intake draft.'})
                q.metadata_json.update({'draft_author':'apprentice_agent','mentor_audited':True,'mentor_audit_source':'mentor_model','role_result_ref_internal':str(role_root/'intake_agent/role_result.json')})
                spec, q = _sanitize_spec_and_quality(spec, q)
                return spec,q
            if settings.rubric_mode == 'llm_required' or settings.llm_fail_closed:
                raise RuntimeError(rr.error_message or 'Model task intake failed')
        except Exception:
            if settings.rubric_mode == 'llm_required' or settings.llm_fail_closed:
                raise
    spec,q=deterministic_intake(raw)
    if settings.llm_task_intake_enabled and _mentor_provider_can_attempt():
        spec.metadata_json.update({'mentor_audit_status':'not_completed','provider':_mentor_provider_id()})
    spec, q = _sanitize_spec_and_quality(spec, q)
    return spec,q
