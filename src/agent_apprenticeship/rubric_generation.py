from __future__ import annotations
import json
from pathlib import Path
from pydantic import BaseModel
from .schemas import TaskIntakeSpec, RubricItem, RubricSpec, RubricQualityReport
from .config import apprentice_agent_display_name, get_settings
from .io import read_workflow_json as read_json
from .openai_structured import get_model_provider_status, run_structured_role
from .public_sanitizer import sanitize_public_obj, sha256_text

class LLMRubricOutput(BaseModel):
    rubric_spec: RubricSpec
    rubric_quality_report: RubricQualityReport

def _mentor_provider_can_attempt() -> bool:
    return bool(get_model_provider_status().get('provider_available'))

def _mentor_provider_id() -> str:
    settings=get_settings()
    return settings.model_provider or 'openai'

def deterministic_rubric(spec: TaskIntakeSpec) -> tuple[RubricSpec, RubricQualityReport]:
    reqs = spec.output_requirements or ['final deliverable']
    weight = 1.0/len(reqs)
    settings=get_settings()
    apprentice_name = apprentice_agent_display_name(settings)
    items=[RubricItem(rubric_item_id=f'ri_{i+1}', criterion_name=r, criterion_description=f'Output satisfies {r}', weight=weight, score_min=0, score_max=1, pass_threshold=0.7, observable_evidence=[r], required_artifacts=[r], scoring_method='hybrid', worker_visible=True, verifier_only=False, hidden_reference_required=False, failure_modes=['missing','incorrect'], partial_credit_rules=['partial if substantially present'], edge_cases=[], anti_cheat_notes=[], metadata_json={'draft_author':'apprentice_agent','trace_quality_checks':['artifact_refs point to produced files','actual_outputs.json summarizes deliverables'],'environment_response_checks':['trace includes observed environment responses for meaningful tool/file actions']}) for i,r in enumerate(reqs)]
    rub=RubricSpec(rubric_id=f'rubric_{spec.task_id}', task_id=spec.task_id, task_family_id=None, rubric_version='v0.1', rubric_items=items, total_weight=1.0, pass_threshold=0.7, worker_visible_rubric_ref='rubric/worker_visible_rubric.md', verifier_private_rubric_ref='rubric/verifier_private_rubric.json', hidden_reference_policy='No hidden references for this Apprentice-generated draft.', scoring_aggregation='weighted_sum', required_artifacts=reqs, disqualifying_errors=['secret leak'], partial_credit_allowed=True, grader_kind='hybrid', rubric_generation_source='agent_assisted', rubric_generation_agent_provider=None, rubric_generation_agent_model=None, rubric_generation_confidence=0.55, metadata_json={'rubric_source':'apprentice_agent_draft','draft_author':'apprentice_agent','apprentice_agent_id':settings.worker_agent,'apprentice_agent_name':apprentice_name,'mentor_audited':False,'mentor_audit_source':None,'audit_status':'unaudited','rubric_limitations':['Mentor audit has not completed for this rubric draft.'],'rubric_confidence':'medium','verifier_checks':reqs,'artifact_checks':reqs,'trace_quality_checks':['one meaningful action per trace step','artifact_refs and deliverable_refs are package-relative','actual_outputs.json is present'],'environment_response_checks':['environment_response_pairs can be extracted from observed trace output/state_change'],'reward_modeling_notes':['Use mentor/verifier/grader evidence when available; structural checks alone are not semantic quality labels.'],'process_supervision_notes':['Use trace step outcomes and evaluator feedback as labels when available.'],'revision_preference_notes':['Prefer revised attempts only when mentor/verifier/grader evidence supports improvement.'],'economic_value_checks':['Use expected economic value fields from task intake as context, not as outcome proof.']})
    qr=RubricQualityReport(rubric_id=rub.rubric_id, task_id=spec.task_id, criteria_count=len(items), total_weight=1.0, weights_sum_valid=True, has_observable_evidence=True, has_required_artifacts=True, has_partial_credit_rules=True, has_disqualifying_errors=True, has_hidden_reference_policy=True, has_worker_visible_view=True, has_verifier_private_view=True, ambiguous_criteria_count=0, unverifiable_criteria_count=0, rubric_quality_score=0.8, quality_flags=['mentor_audit_pending'], blockers=[], metadata_json={'draft_author':'apprentice_agent','mentor_audited':False})
    if settings.rubric_mode in {'hybrid','llm_default'} and _mentor_provider_can_attempt() and settings.llm_rubric_generation_enabled:
        provider=_mentor_provider_id()
        rub.grader_kind='hybrid'; rub.rubric_generation_agent_provider=provider; rub.rubric_generation_agent_model=settings.model_provider_model or settings.openai_model; rub.metadata_json.update({'llm_rubric_generation_enabled': True, 'mentor_audit_status': 'not_completed', 'provider': provider})
        qr.quality_flags.append('mentor_audit_not_completed')
    return rub, qr

def deterministic_rubric_quality_check(rubric: RubricSpec) -> RubricQualityReport:
    vague=sum(1 for i in rubric.rubric_items if 'good quality' in (i.criterion_description or '').lower() and not i.observable_evidence)
    return RubricQualityReport(rubric_id=rubric.rubric_id, task_id=rubric.task_id, criteria_count=len(rubric.rubric_items), total_weight=rubric.total_weight, weights_sum_valid=abs(sum(i.weight for i in rubric.rubric_items)-1.0)<1e-6, has_observable_evidence=all(bool(i.observable_evidence) for i in rubric.rubric_items), has_required_artifacts=all(bool(i.required_artifacts) for i in rubric.rubric_items), has_partial_credit_rules=all(bool(i.partial_credit_rules) for i in rubric.rubric_items), has_disqualifying_errors=bool(rubric.disqualifying_errors), has_hidden_reference_policy=bool(rubric.hidden_reference_policy), has_worker_visible_view=any(i.worker_visible for i in rubric.rubric_items), has_verifier_private_view=bool(rubric.verifier_private_rubric_ref), ambiguous_criteria_count=vague, unverifiable_criteria_count=sum(1 for i in rubric.rubric_items if not i.observable_evidence), rubric_quality_score=0.0 if vague else 0.85, quality_flags=['vague_rubric_item'] if vague else [], blockers=['vague criteria without evidence'] if vague else [], metadata_json={})

def _rubric_prompt(spec: TaskIntakeSpec) -> str:
    public_spec=sanitize_public_obj(spec.model_dump(mode='json'))
    return """Return only valid JSON. Do not include markdown. Do not add extra top-level fields; place extras under metadata_json.extra_model_fields.
Required skeleton: {"rubric_spec":{"rubric_id":"rubric_<task_id>","task_id":"...","rubric_version":"0.1","rubric_items":[{"rubric_item_id":"ri_1","criterion_name":"...","criterion_description":"...","weight":1.0,"score_min":0,"score_max":1,"pass_threshold":0.7,"observable_evidence":[],"required_artifacts":[],"scoring_method":"llm_rubric_judge","worker_visible":true,"verifier_only":false,"hidden_reference_required":false,"failure_modes":[],"partial_credit_rules":[],"edge_cases":[],"anti_cheat_notes":[],"metadata_json":{}}],"total_weight":1.0,"pass_threshold":0.7,"worker_visible_rubric_ref":"rubric/worker_visible_rubric.md","verifier_private_rubric_ref":"rubric/verifier_private_rubric.json","hidden_reference_policy":"no_hidden_reference_available","scoring_aggregation":"weighted_sum","required_artifacts":[],"disqualifying_errors":[],"partial_credit_allowed":true,"grader_kind":"llm_rubric_judge","rubric_generation_source":"task_specific_agent_draft","metadata_json":{}},"rubric_quality_report":{"rubric_id":"...","task_id":"...","criteria_count":1,"total_weight":1.0,"weights_sum_valid":true,"has_observable_evidence":true,"has_required_artifacts":true,"has_partial_credit_rules":true,"has_disqualifying_errors":false,"has_hidden_reference_policy":true,"has_worker_visible_view":true,"has_verifier_private_view":true,"ambiguous_criteria_count":0,"unverifiable_criteria_count":0,"rubric_quality_score":0.75,"quality_flags":[],"blockers":[],"metadata_json":{}}}.
Audit and improve an Apprentice Agent-generated task rubric draft. Keep the Apprentice Agent as the draft author; your role is Mentor audit/improvement. Create a task-specific rubric with weighted rubric_items that sum to 1.0. Include explicit required_artifacts, observable_evidence, success criteria, failure modes, trace_quality_checks, environment_response_checks, reward_modeling_notes, process_supervision_notes, revision_preference_notes, economic_value_checks, and grader/verifier/evaluator guidance in metadata. Do not rely on Apprentice Agent self-evaluation as outcome evidence. Do not leak hidden/reference answers to Apprentice-visible fields. Do not include commerce metadata or off-brand selection language.
TaskIntakeSpec JSON:
""" + json.dumps(public_spec, sort_keys=True)

def generate_rubric(spec: TaskIntakeSpec, role_root: Path | None=None) -> tuple[RubricSpec, RubricQualityReport]:
    settings=get_settings(); role_root=role_root or Path('outputs/roles')
    if settings.llm_rubric_generation_enabled and settings.rubric_mode in {'hybrid','llm_default','llm_required'} and _mentor_provider_can_attempt():
        prompt=_rubric_prompt(spec)
        try:
            provider=_mentor_provider_id()
            model_override=settings.llm_rubric_model if provider == 'openai' else None
            rr=run_structured_role('rubric_agent', prompt, LLMRubricOutput, role_root/'rubric_agent', allow_fallback=settings.allow_deterministic_eval_fallback, model_override=model_override, normalizer_context={'task_id': spec.task_id, 'task_title': spec.normalized_title, 'task_instruction': spec.normalized_instruction, 'model': model_override or settings.model_provider_model, 'provider':provider})
            if rr.live_call_ok and rr.structured_output_validation_ok:
                parsed=read_json(role_root/'rubric_agent/parsed_output.json')
                rub=RubricSpec.model_validate(parsed['rubric_spec'])
                qr=deterministic_rubric_quality_check(rub)
                rub.grader_kind='hybrid'; rub.rubric_generation_source='agent_assisted'; rub.rubric_generation_agent_provider=rr.provider; rub.rubric_generation_agent_model=rr.model; rub.rubric_generation_confidence=rub.rubric_generation_confidence or 0.75
                rub.metadata_json.update({'rubric_source':'apprentice_agent_draft','draft_author':'apprentice_agent','mentor_audited':True,'mentor_audit_source':'mentor_model','audit_status':'mentor_audited','provider':rr.provider,'model':rr.model,'llm_prompt_ref_internal':str(role_root/'rubric_agent/prompt.md'),'llm_response_ref_internal':str(role_root/'rubric_agent/raw_output.txt'),'prompt_template_id':'rubric_generation_agent_v0','prompt_template_version':'0.1','prompt_hash':sha256_text(prompt),'public_response_summary':'Mentor Model audited and improved an Apprentice-generated rubric draft.'})
                qr.metadata_json.update({'draft_author':'apprentice_agent','mentor_audited':True,'mentor_audit_source':'mentor_model','role_result_ref_internal':str(role_root/'rubric_agent/role_result.json')})
                return rub, qr
            if settings.rubric_mode == 'llm_required' or settings.llm_fail_closed:
                raise RuntimeError(rr.error_message or 'Model rubric generation failed')
        except Exception:
            if settings.rubric_mode == 'llm_required' or settings.llm_fail_closed:
                raise
    return deterministic_rubric(spec)

def worker_visible_markdown(rubric: RubricSpec) -> str:
    lines=[f'# Worker-visible rubric for {rubric.task_id}', '']
    for item in rubric.rubric_items:
        if item.worker_visible and not item.verifier_only:
            lines.append(f'- **{item.criterion_name}** ({item.weight}): {item.criterion_description}')
    return '\n'.join(lines)+'\n'
