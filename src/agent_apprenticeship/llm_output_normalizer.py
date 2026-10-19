from __future__ import annotations
from dataclasses import dataclass
from typing import Any
from .schemas import GraderResult, VerifierResult, RubricItemScore, TaskIntakeSpec, TaskIntakeQualityReport, RubricSpec, RubricQualityReport, RubricItem, EvaluatorFeedback, RevisionPlan

GRADER_ALLOWED=set(GraderResult.__annotations__.keys())
VERIFIER_ALLOWED=set(VerifierResult.__annotations__.keys())
ITEM_ALLOWED=set(RubricItemScore.__annotations__.keys())
INTAKE_ALLOWED=set(TaskIntakeSpec.__annotations__.keys())
QUALITY_ALLOWED=set(TaskIntakeQualityReport.__annotations__.keys())
RUBRIC_ALLOWED=set(RubricSpec.__annotations__.keys())
RUBRIC_ITEM_ALLOWED=set(RubricItem.__annotations__.keys())
RUBRIC_QUALITY_ALLOWED=set(RubricQualityReport.__annotations__.keys())
EVALUATOR_ALLOWED=set(EvaluatorFeedback.__annotations__.keys())
REVISION_ALLOWED=set(RevisionPlan.__annotations__.keys())

@dataclass
class RoleContext:
    task_id: str = 'smoke_task'
    attempt_id: str = 'smoke_attempt'
    attempt_kind: str = 'baseline'
    rubric_id: str = 'smoke_rubric'
    grader_result_id: str | None = None
    artifact_contract_score: float | None = None
    evidence_refs: list[str] | None = None
    artifact_content_refs: list[str] | None = None
    artifact_content_previews: list[dict[str, Any]] | None = None
    artifact_content_hashes: dict[str, str] | None = None
    artifact_content_preview_truncated: bool | None = None
    model_grading_basis: str | None = None
    model: str | None = None
    provider: str = 'openai'
    task_title: str | None = None
    task_instruction: str | None = None
    target_attempt_id: str | None = None
    verifier_result_id: str | None = None
    review_packet_ref: str | None = None


def _as_list(v: Any) -> list[Any]:
    if v is None: return []
    if isinstance(v, list): return v
    return [v]

def _float(v: Any, default: float=0.0) -> float:
    try:
        if v is None: return default
        return float(v)
    except Exception:
        return default

def _bool(v: Any, default: bool=False) -> bool:
    if isinstance(v, bool): return v
    if isinstance(v, str):
        if v.lower() in {'true','yes','passed','pass','ok'}: return True
        if v.lower() in {'false','no','failed','fail'}: return False
    return default

def _metadata(raw: dict[str, Any], allowed: set[str]) -> dict[str, Any]:
    md=dict(raw.get('metadata_json') or {}) if isinstance(raw.get('metadata_json'), dict) else {}
    extra={k:v for k,v in raw.items() if k not in allowed}
    if extra:
        md['extra_model_fields']=extra
    md['raw_model_output']=raw
    md['raw_llm_output']=raw
    return md

def _normalize_item(raw: Any, idx: int, fallback_refs: list[str]) -> dict[str, Any]:
    if not isinstance(raw, dict): raw={'notes': str(raw)}
    md=_metadata(raw, ITEM_ALLOWED)
    score=_float(raw.get('score', raw.get('semantic_correctness_score', raw.get('final_score'))), 0.0)
    max_score=_float(raw.get('max_score'), 1.0) or 1.0
    refs=[str(x) for x in _as_list(raw.get('evidence_refs'))] or fallback_refs
    passed=_bool(raw.get('passed'), score >= 0.7*max_score)
    return {
        'rubric_item_id': str(raw.get('rubric_item_id') or raw.get('id') or f'ri_{idx}'),
        'criterion_name': str(raw.get('criterion_name') or raw.get('name') or raw.get('criterion') or f'criterion_{idx}'),
        'score': score,
        'max_score': max_score,
        'passed': passed,
        'evidence_refs': refs,
        'failure_mode': raw.get('failure_mode'),
        'notes': raw.get('notes') or raw.get('reasoning_summary'),
        'confidence': _float(raw.get('confidence'), 0.7),
        'artifact_presence_ok': raw.get('artifact_presence_ok') if isinstance(raw.get('artifact_presence_ok'), bool) else None,
        'semantic_correctness_score': raw.get('semantic_correctness_score') if raw.get('semantic_correctness_score') is not None else score,
        'reasoning_summary': raw.get('reasoning_summary') or raw.get('reasoning') or raw.get('notes'),
        'improvement_suggestion': raw.get('improvement_suggestion') or raw.get('suggestion'),
    }



def _first_dict(raw: dict[str, Any], *keys: str) -> dict[str, Any]:
    for key in keys:
        val=raw.get(key)
        if isinstance(val, dict):
            return val
    return raw

def _summary(v: Any, default: str) -> str:
    if v is None:
        return default
    if isinstance(v, str):
        return v
    if isinstance(v, list):
        return ', '.join(str(x) for x in v) or default
    if isinstance(v, dict):
        return ', '.join(f'{k}: {v[k]}' for k in list(v)[:5]) or default
    return str(v)

def _score(v: Any, default: float=0.8) -> float:
    x=_float(v, default)
    if x < 0: return 0.0
    if x > 1: return 1.0
    return x

def normalize_task_intake_result(raw: dict[str, Any], context: RoleContext | None=None) -> dict[str, Any]:
    context=context or RoleContext()
    raw=dict(raw or {})
    spec_raw=_first_dict(raw, 'task_intake_spec', 'intake_spec', 'task')
    quality_raw=raw.get('task_intake_quality_report') if isinstance(raw.get('task_intake_quality_report'), dict) else raw.get('quality_report') if isinstance(raw.get('quality_report'), dict) else {}
    task_id=str(spec_raw.get('task_id') or raw.get('task_id') or context.task_id)
    title=spec_raw.get('normalized_title') or spec_raw.get('task_title') or spec_raw.get('title') or raw.get('task_title') or context.task_title or task_id
    instruction=spec_raw.get('normalized_instruction') or spec_raw.get('instruction') or spec_raw.get('task_instruction') or spec_raw.get('description') or raw.get('description') or context.task_instruction or title
    expected_outputs=spec_raw.get('expected_outputs') or spec_raw.get('output_requirements') or spec_raw.get('required_artifacts') or []
    required_artifacts=spec_raw.get('required_artifacts') or spec_raw.get('artifacts') or expected_outputs or []
    metadata=_metadata(spec_raw, INTAKE_ALLOWED)
    metadata.update({
        'task_family_guess': spec_raw.get('task_family_guess') or spec_raw.get('task_family_id') or spec_raw.get('domain') or 'unknown',
        'task_type': spec_raw.get('task_type') or spec_raw.get('workflow_type') or 'artifact_generation',
        'required_artifacts': _as_list(required_artifacts),
        'hidden_reference_policy': spec_raw.get('hidden_reference_policy') or 'no_hidden_reference_available',
        'risk_notes': _as_list(spec_raw.get('risk_notes') or spec_raw.get('risk_safety_notes')),
        'subjectivity_level': spec_raw.get('subjectivity_level') or 'medium',
        'evaluation_difficulty': spec_raw.get('evaluation_difficulty') or 'medium',
        'suggested_evaluator_type': spec_raw.get('suggested_evaluator_type') or 'model_judged',
    })
    spec={
        'task_id': task_id,
        'normalized_title': str(title),
        'normalized_instruction': str(instruction),
        'domain': str(spec_raw.get('domain') or raw.get('domain') or 'general'),
        'subdomain': spec_raw.get('subdomain'),
        'professional_role': spec_raw.get('professional_role'),
        'workflow_type': str(spec_raw.get('workflow_type') or spec_raw.get('task_type') or 'artifact_generation'),
        'skill_targets': [str(x) for x in _as_list(spec_raw.get('skill_targets'))],
        'difficulty_tier': spec_raw.get('difficulty_tier') if spec_raw.get('difficulty_tier') in {'easy','medium','hard','expert'} else 'medium',
        'expected_human_deliverable': _summary(spec_raw.get('expected_human_deliverable') or expected_outputs, 'Review the generated artifacts and final answer.'),
        'expected_agent_deliverable': _summary(spec_raw.get('expected_agent_deliverable') or expected_outputs, 'Produce required task artifacts and actual_outputs.json.'),
        'input_requirements': [str(x) for x in _as_list(spec_raw.get('input_requirements') or spec_raw.get('required_inputs'))],
        'output_requirements': [str(x) for x in _as_list(spec_raw.get('output_requirements') or expected_outputs or required_artifacts)],
        'required_context': [str(x) for x in _as_list(spec_raw.get('required_context'))],
        'assumptions': [str(x) for x in _as_list(spec_raw.get('assumptions'))],
        'constraints': [str(x) for x in _as_list(spec_raw.get('constraints'))],
        'allowed_tools': [str(x) for x in _as_list(spec_raw.get('allowed_tools'))] or ['python','file_read','file_write','bash'],
        'disallowed_tools': [str(x) for x in _as_list(spec_raw.get('disallowed_tools'))],
        'privacy_classification': spec_raw.get('privacy_classification') if spec_raw.get('privacy_classification') in {'public','synthetic','sensitive_possible','contains_pii','unknown'} else 'unknown',
        'license': spec_raw.get('license'),
        'allowed_use': spec_raw.get('allowed_use') or 'local research dataset generation',
        'rubricability_score': _score(spec_raw.get('rubricability_score') or spec_raw.get('rubricability'), 0.7),
        'verifiability_score': _score(spec_raw.get('verifiability_score') or spec_raw.get('verifiability'), 0.7),
        'artifactability_score': _score(spec_raw.get('artifactability_score') or spec_raw.get('artifactability'), 0.7),
        'needs_expert_review': _bool(spec_raw.get('needs_expert_review'), False),
        'metadata_json': metadata,
    }
    qmd=_metadata(quality_raw, QUALITY_ALLOWED) if isinstance(quality_raw, dict) else {}
    quality={
        'task_id': task_id,
        'instruction_clarity_score': _score(quality_raw.get('instruction_clarity_score') if isinstance(quality_raw, dict) else None, 0.7),
        'input_completeness_score': _score(quality_raw.get('input_completeness_score') if isinstance(quality_raw, dict) else None, 0.7),
        'output_contract_score': _score(quality_raw.get('output_contract_score') if isinstance(quality_raw, dict) else None, 0.7),
        'rubricability_score': spec['rubricability_score'],
        'verifiability_score': spec['verifiability_score'],
        'artifactability_score': spec['artifactability_score'],
        'privacy_risk_score': _score(quality_raw.get('privacy_risk_score') if isinstance(quality_raw, dict) else None, 0.2),
        'license_risk_score': _score(quality_raw.get('license_risk_score') if isinstance(quality_raw, dict) else None, 0.2),
        'ambiguity_score': _score(quality_raw.get('ambiguity_score') if isinstance(quality_raw, dict) else None, 0.3),
        'overall_intake_quality_score': _score(quality_raw.get('overall_intake_quality_score') if isinstance(quality_raw, dict) else None, 0.7),
        'quality_flags': [str(x) for x in _as_list(quality_raw.get('quality_flags') if isinstance(quality_raw, dict) else None)],
        'blockers': [str(x) for x in _as_list(quality_raw.get('blockers') if isinstance(quality_raw, dict) else None)],
        'recommended_fix': quality_raw.get('recommended_fix') if isinstance(quality_raw, dict) else None,
        'metadata_json': qmd,
    }
    return {'task_intake_spec': spec, 'task_intake_quality_report': quality}

def _rubric_item(raw_item: Any, idx: int, weight: float, required_artifacts: list[str]) -> dict[str, Any]:
    if not isinstance(raw_item, dict): raw_item={'criterion_name': str(raw_item)}
    md=_metadata(raw_item, RUBRIC_ITEM_ALLOWED)
    evidence=raw_item.get('observable_evidence') or raw_item.get('evidence') or raw_item.get('evidence_requirements') or raw_item.get('success_criteria') or required_artifacts or ['final answer and artifacts']
    artifacts=raw_item.get('required_artifacts') or raw_item.get('artifacts') or required_artifacts or ['final_answer']
    scoring_method=raw_item.get('scoring_method')
    if not isinstance(scoring_method, str) or scoring_method not in {'llm_rubric_judge','deterministic','schema_match','regex','unit_test','hybrid','human_future','structural_guardrail'}:
        scoring_method='llm_rubric_judge'
    return {
        'rubric_item_id': str(raw_item.get('rubric_item_id') or raw_item.get('id') or f'ri_{idx}'),
        'criterion_name': str(raw_item.get('criterion_name') or raw_item.get('name') or raw_item.get('criterion') or f'criterion_{idx}'),
        'criterion_description': str(raw_item.get('criterion_description') or raw_item.get('description') or raw_item.get('scoring_guidance') or 'Evaluate observable task success evidence.'),
        'weight': weight,
        'score_min': _float(raw_item.get('score_min'), 0.0),
        'score_max': _float(raw_item.get('score_max') or raw_item.get('max_score'), 1.0) or 1.0,
        'pass_threshold': _float(raw_item.get('pass_threshold'), 0.7),
        'observable_evidence': [str(x) for x in _as_list(evidence)],
        'required_artifacts': [str(x) for x in _as_list(artifacts)],
        'scoring_method': scoring_method,
        'worker_visible': _bool(raw_item.get('worker_visible'), True),
        'verifier_only': _bool(raw_item.get('verifier_only'), False),
        'hidden_reference_required': _bool(raw_item.get('hidden_reference_required'), False),
        'failure_modes': [str(x) for x in _as_list(raw_item.get('failure_modes'))] or ['missing evidence','incorrect output'],
        'partial_credit_rules': [str(x) for x in _as_list(raw_item.get('partial_credit_rules'))] or ['Award partial credit for partially correct, evidence-backed artifacts.'],
        'edge_cases': [str(x) for x in _as_list(raw_item.get('edge_cases'))],
        'anti_cheat_notes': [str(x) for x in _as_list(raw_item.get('anti_cheat_notes'))],
        'metadata_json': md,
    }

def normalize_rubric_result(raw: dict[str, Any], context: RoleContext | None=None) -> dict[str, Any]:
    context=context or RoleContext()
    raw=dict(raw or {})
    rub_raw=_first_dict(raw, 'rubric_spec', 'rubric')
    quality_raw=raw.get('rubric_quality_report') if isinstance(raw.get('rubric_quality_report'), dict) else raw.get('quality_report') if isinstance(raw.get('quality_report'), dict) else {}
    task_id=str(rub_raw.get('task_id') or raw.get('task_id') or context.task_id)
    required=[str(x) for x in _as_list(rub_raw.get('required_artifacts') or raw.get('required_artifacts') or rub_raw.get('expected_outputs'))]
    raw_items=rub_raw.get('rubric_items') or rub_raw.get('criteria') or rub_raw.get('items') or rub_raw.get('checks') or []
    if not raw_items:
        base=required or [context.task_title or 'final deliverable']
        raw_items=[{'name': f'{name} quality', 'description': f'{name} is present, correct, and supported by evidence.', 'required_artifacts':[name], 'evidence':[name]} for name in base]
    raw_weights=[_float(i.get('weight'), 0.0) if isinstance(i, dict) else 0.0 for i in raw_items]
    total=sum(raw_weights)
    weight_normalized=False
    if total <= 0:
        weights=[1.0/len(raw_items)]*len(raw_items)
        weight_normalized=True
    else:
        weights=[w/total for w in raw_weights]
        weight_normalized=abs(total-1.0)>1e-6
    items=[_rubric_item(item, i+1, weights[i], required) for i,item in enumerate(raw_items)]
    all_required=[]
    for item in items: all_required.extend(item['required_artifacts'])
    required=list(dict.fromkeys(required or all_required))
    md=_metadata(rub_raw, RUBRIC_ALLOWED)
    md.update({'grader_guidance': rub_raw.get('grader_guidance') or 'Grade against observable artifacts, trace evidence, and task requirements.', 'verifier_guidance': rub_raw.get('verifier_guidance') or 'Verify evidence grounding, score consistency, and leakage safety.', 'evaluator_guidance': rub_raw.get('evaluator_guidance') or 'Provide actionable feedback for revision.', 'limitations': [str(x) for x in _as_list(rub_raw.get('limitations'))]})
    if weight_normalized: md['weight_normalization_applied']=True
    rubric={
        'rubric_id': str(rub_raw.get('rubric_id') or f'rubric_{task_id}'),
        'task_id': task_id,
        'task_family_id': rub_raw.get('task_family_id'),
        'rubric_version': str(rub_raw.get('rubric_version') or '0.1'),
        'rubric_items': items,
        'total_weight': 1.0,
        'pass_threshold': _float(rub_raw.get('pass_threshold'), 0.7),
        'worker_visible_rubric_ref': rub_raw.get('worker_visible_rubric_ref') or 'rubric/worker_visible_rubric.md',
        'verifier_private_rubric_ref': rub_raw.get('verifier_private_rubric_ref') or 'rubric/verifier_private_rubric.json',
        'hidden_reference_policy': rub_raw.get('hidden_reference_policy') or 'no_hidden_reference_available',
        'scoring_aggregation': rub_raw.get('scoring_aggregation') if rub_raw.get('scoring_aggregation') in {'weighted_sum','sum','all_required','custom'} else 'weighted_sum',
        'required_artifacts': required,
        'disqualifying_errors': [str(x) for x in _as_list(rub_raw.get('disqualifying_errors'))],
        'partial_credit_allowed': _bool(rub_raw.get('partial_credit_allowed'), True),
        'grader_kind': rub_raw.get('grader_kind') if rub_raw.get('grader_kind') in {'llm_rubric_judge','deterministic','hybrid','human_future','structural_guardrail','unavailable'} else 'llm_rubric_judge',
        'rubric_generation_source': rub_raw.get('rubric_generation_source') if rub_raw.get('rubric_generation_source') in {'agent_assisted','family_template','task_specific_agent_draft','expert_override','deterministic_seed'} else 'task_specific_agent_draft',
        'rubric_generation_agent_provider': rub_raw.get('rubric_generation_agent_provider') or context.provider,
        'rubric_generation_agent_model': rub_raw.get('rubric_generation_agent_model') or context.model,
        'rubric_generation_confidence': _float(rub_raw.get('rubric_generation_confidence'), 0.7),
        'metadata_json': md,
    }
    quality={
        'rubric_id': rubric['rubric_id'], 'task_id': task_id, 'criteria_count': len(items), 'total_weight': 1.0,
        'weights_sum_valid': True, 'has_observable_evidence': all(bool(i['observable_evidence']) for i in items), 'has_required_artifacts': all(bool(i['required_artifacts']) for i in items),
        'has_partial_credit_rules': any(bool(i['partial_credit_rules']) for i in items), 'has_disqualifying_errors': bool(rubric['disqualifying_errors']), 'has_hidden_reference_policy': bool(rubric['hidden_reference_policy']),
        'has_worker_visible_view': True, 'has_verifier_private_view': True, 'ambiguous_criteria_count': int(quality_raw.get('ambiguous_criteria_count') or 0) if isinstance(quality_raw, dict) else 0,
        'unverifiable_criteria_count': int(quality_raw.get('unverifiable_criteria_count') or 0) if isinstance(quality_raw, dict) else 0, 'rubric_quality_score': _score(quality_raw.get('rubric_quality_score') if isinstance(quality_raw, dict) else None, 0.75),
        'quality_flags': [str(x) for x in _as_list(quality_raw.get('quality_flags') if isinstance(quality_raw, dict) else None)], 'blockers': [str(x) for x in _as_list(quality_raw.get('blockers') if isinstance(quality_raw, dict) else None)], 'metadata_json': _metadata(quality_raw if isinstance(quality_raw, dict) else {}, RUBRIC_QUALITY_ALLOWED),
    }
    return {'rubric_spec': rubric, 'rubric_quality_report': quality}

def normalize_evaluator_result(raw: dict[str, Any], context: RoleContext | None=None) -> dict[str, Any]:
    context=context or RoleContext()
    raw=dict(raw or {})
    fb_raw=_first_dict(raw, 'evaluator_feedback', 'feedback')
    rp_raw=raw.get('revision_plan') if isinstance(raw.get('revision_plan'), dict) else {}
    task_id=str(fb_raw.get('task_id') or raw.get('task_id') or context.task_id)
    attempt_id=str(fb_raw.get('attempt_id') or raw.get('attempt_id') or context.attempt_id)
    summary=str(fb_raw.get('feedback_summary') or fb_raw.get('summary') or fb_raw.get('feedback') or 'LLM evaluator feedback generated.')
    actionable=fb_raw.get('actionable_feedback') or fb_raw.get('suggestions') or fb_raw.get('recommendations') or []
    actionable=[str(x) for x in _as_list(actionable)]
    weak=[str(x) for x in _as_list(fb_raw.get('failed_or_weak_rubric_items') or fb_raw.get('failed_rubric_items'))]
    md=_metadata(fb_raw, EVALUATOR_ALLOWED)
    md.update({'attempt_kind': fb_raw.get('attempt_kind') or context.attempt_kind, 'grader_result_id': context.grader_result_id, 'verifier_result_id': context.verifier_result_id, 'review_source': 'review_packet' if context.review_packet_ref else md.get('review_source'), 'review_packet_ref': context.review_packet_ref})
    feedback={
        'feedback_id': str(fb_raw.get('feedback_id') or f'feedback_{task_id}_{context.attempt_kind}'),
        'task_id': task_id,
        'attempt_id': attempt_id,
        'target_actor': fb_raw.get('target_actor') if fb_raw.get('target_actor') in {'worker','reviser','apprentice'} else 'apprentice',
        'feedback_type': fb_raw.get('feedback_type') if fb_raw.get('feedback_type') in {'criteria_failure','artifact_missing','format_error','logic_error','tool_error','quality_gap','strategy_gap','safety_or_privacy','other'} else 'other',
        'failed_rubric_items': weak,
        'evidence_refs': [str(x) for x in _as_list(fb_raw.get('evidence_refs'))],
        'artifact_refs': [str(x) for x in _as_list(fb_raw.get('artifact_refs'))],
        'feedback_summary': summary,
        'actionable_feedback': actionable,
        'suggested_revision': str(fb_raw.get('suggested_revision') or (actionable[0] if actionable else summary)),
        'revision_priority': fb_raw.get('revision_priority') if fb_raw.get('revision_priority') in {'low','medium','high'} else 'medium',
        'confidence': _float(fb_raw.get('confidence'), 0.7),
        'hidden_reference_used': _bool(fb_raw.get('hidden_reference_used'), False),
        'hidden_reference_leaked': _bool(fb_raw.get('hidden_reference_leaked'), False),
        'failed_or_weak_rubric_items': weak,
        'artifact_specific_comments': [str(x) for x in _as_list(fb_raw.get('artifact_specific_comments'))],
        'trace_specific_comments': [str(x) for x in _as_list(fb_raw.get('trace_specific_comments'))],
        'revision_plan': fb_raw.get('revision_plan') if isinstance(fb_raw.get('revision_plan'), str) else summary,
        'model': fb_raw.get('model') or context.model,
        'provider': fb_raw.get('provider') or context.provider,
        'metadata_json': md,
    }
    rp_md=_metadata(rp_raw, REVISION_ALLOWED) if isinstance(rp_raw, dict) else {}
    revision={
        'revision_plan_id': str(rp_raw.get('revision_plan_id') or f'revision_plan_{task_id}_{context.attempt_kind}'),
        'task_id': task_id,
        'source_attempt_id': str(rp_raw.get('source_attempt_id') or attempt_id),
        'target_attempt_id': str(rp_raw.get('target_attempt_id') or context.target_attempt_id or f'{task_id}_revised'),
        'revision_kind': rp_raw.get('revision_kind') if rp_raw.get('revision_kind') in {'local_fix','strategy_shift','tool_change','decomposition_change','artifact_rebuild','format_repair','other'} else 'local_fix',
        'revision_reason': str(rp_raw.get('revision_reason') or summary),
        'failed_rubric_items': [str(x) for x in _as_list(rp_raw.get('failed_rubric_items'))] or weak,
        'planned_changes': [str(x) for x in _as_list(rp_raw.get('planned_changes') or rp_raw.get('instructions'))] or actionable,
        'expected_score_improvement': rp_raw.get('expected_score_improvement'),
        'risk_of_regression': rp_raw.get('risk_of_regression') if rp_raw.get('risk_of_regression') in {'low','medium','high'} else 'medium',
        'uses_evaluator_feedback': _bool(rp_raw.get('uses_evaluator_feedback'), True),
        'metadata_json': rp_md,
    }
    revision['metadata_json'].update({'source_attempt_kind': context.attempt_kind, 'revision_goal': summary, 'priority_items': weak, 'expected_improvements': [], 'review_source': 'review_packet' if context.review_packet_ref else revision['metadata_json'].get('review_source'), 'review_packet_ref': context.review_packet_ref})
    return {'evaluator_feedback': feedback, 'revision_plan': revision}

def normalize_grader_result(raw: dict[str, Any], context: RoleContext | None=None) -> dict[str, Any]:
    context=context or RoleContext()
    raw=dict(raw or {})
    md=_metadata(raw, GRADER_ALLOWED)
    refs=[str(x) for x in _as_list(raw.get('evidence_refs'))] or list(context.evidence_refs or [])
    artifact_contract_score = context.artifact_contract_score if context.artifact_contract_score is not None else raw.get('artifact_contract_score')
    semantic=raw.get('semantic_score', raw.get('final_score', raw.get('score')))
    semantic_score=_float(semantic, 0.0)
    final_score=_float(raw.get('final_score'), semantic_score)
    limitations=[str(x) for x in _as_list(raw.get('limitations'))]
    if artifact_contract_score == 0.0 and not (raw.get('evidence_refs') or context.evidence_refs):
        semantic_score=0.0
        final_score=0.0
        limitations.append('Semantic grading impossible because artifact content was unavailable; score reflects missing outputs/log evidence.')
    md['structural_guardrail']={'artifact_contract_score': artifact_contract_score}
    basis=context.model_grading_basis or ('artifact_content' if artifact_contract_score else ('logs_only' if refs else 'missing_outputs'))
    md['evidence_materialization_status']='materialized' if artifact_contract_score else 'missing_or_unverified'
    md['referenced_but_missing_artifacts']=[str(x) for x in _as_list(raw.get('referenced_but_missing_artifacts'))]
    md['artifact_content_available']=basis in {'artifact_content','artifact_preview'}
    md['semantic_grading_basis']=basis
    md['model_grading_basis']=basis
    md['artifact_content_refs']=context.artifact_content_refs or []
    md['artifact_content_previews']=context.artifact_content_previews or []
    md['artifact_content_hashes']=context.artifact_content_hashes or {}
    md['artifact_content_preview_truncated']=bool(context.artifact_content_preview_truncated)
    items=[_normalize_item(x, i+1, refs) for i,x in enumerate(_as_list(raw.get('rubric_item_scores')))]
    passed=_bool(raw.get('passed'), final_score >= 0.7)
    failed=[str(x) for x in _as_list(raw.get('failed_criteria'))]
    passed_criteria=[str(x) for x in _as_list(raw.get('passed_criteria'))]
    if not failed and items:
        failed=[i['rubric_item_id'] for i in items if not i['passed']]
    if not passed_criteria and items:
        passed_criteria=[i['rubric_item_id'] for i in items if i['passed']]
    return {
        'grader_result_id': str(raw.get('grader_result_id') or f'grader_{context.task_id}_{context.attempt_kind}'),
        'task_id': str(raw.get('task_id') or context.task_id),
        'attempt_id': str(raw.get('attempt_id') or context.attempt_id),
        'attempt_kind': str(raw.get('attempt_kind') or context.attempt_kind),
        'rubric_id': str(raw.get('rubric_id') or context.rubric_id),
        'grader_kind': 'model',
        'score_source': 'model_judged',
        'score': final_score,
        'max_score': _float(raw.get('max_score'), 1.0) or 1.0,
        'passed': passed,
        'rubric_item_scores': items,
        'failed_criteria': failed,
        'passed_criteria': passed_criteria,
        'evidence_refs': refs,
        'confidence': _float(raw.get('confidence'), 0.7),
        'reasoning_summary': raw.get('reasoning_summary') or raw.get('summary') or raw.get('notes'),
        'limitations': limitations,
        'hidden_reference_used': _bool(raw.get('hidden_reference_used'), False),
        'hidden_reference_leaked': _bool(raw.get('hidden_reference_leaked'), False),
        'artifact_contract_score': artifact_contract_score,
        'semantic_score': semantic_score,
        'model_score': semantic_score,
        'legacy_semantic_score': semantic_score,
        'legacy_score_source': 'llm_semantic',
        'final_score': final_score,
        'model': raw.get('model') or context.model,
        'provider': raw.get('provider') or context.provider,
        'deterministic_precheck_ref': None,
        'llm_prompt_ref_internal': raw.get('llm_prompt_ref_internal'),
        'llm_response_ref_internal': raw.get('llm_response_ref_internal'),
        'public_prompt_hash': raw.get('public_prompt_hash'),
        'public_response_summary': raw.get('public_response_summary') or raw.get('reasoning_summary') or 'Model grader result normalized.',
        'metadata_json': md,
    }

def _first_present(raw: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in raw and raw.get(key) is not None:
            return raw.get(key)
    return None

def normalize_verifier_result(raw: dict[str, Any], context: RoleContext | None=None) -> dict[str, Any]:
    context=context or RoleContext()
    raw=dict(raw or {})
    md=_metadata(raw, VERIFIER_ALLOWED)
    unsupported=[str(x) for x in _as_list(raw.get('unsupported_claims'))]
    issues=[str(x) for x in _as_list(raw.get('issues'))]
    if raw.get('unsupported_claims_found') is True and not unsupported:
        unsupported.append('unsupported claims found')
    if unsupported:
        issues.extend([f'unsupported_claim: {x}' for x in unsupported])
    passed_raw=_first_present(raw, 'passed', 'grade_accepted', 'accepted')
    if passed_raw is False and not issues:
        issues.append('verifier marked attempt as not passed')
    artifact_ok=_bool(_first_present(raw, 'artifact_contract_ok'), True)
    grounding=_bool(_first_present(raw, 'semantic_evidence_grounding_ok', 'evidence_grounding_ok', 'grounded', 'evidence_grounded'), True)
    consistency=_bool(_first_present(raw, 'score_consistency_ok', 'score_consistent'), True)
    leaked=_bool(_first_present(raw, 'hidden_reference_leaked', 'hidden_reference_leak'), False)
    if not grounding and not any('ground' in i.lower() for i in issues):
        issues.append('semantic evidence was not grounded')
    if not consistency and not any('score' in i.lower() for i in issues):
        issues.append('score was not internally consistent')
    if leaked and not any('hidden_reference' in i.lower() or 'leak' in i.lower() for i in issues):
        issues.append('hidden reference leakage detected')
    ok=artifact_ok and grounding and consistency and not leaked and passed_raw is not False and not unsupported
    if raw.get('verification_status') in {'verified','partially_verified','failed','not_run'}:
        status=raw.get('verification_status')
    elif passed_raw is False or leaked or unsupported:
        status='failed'
    else:
        status='verified' if ok else 'partially_verified'
    return {
        'verifier_result_id': str(raw.get('verifier_result_id') or f'verifier_{context.task_id}_{context.attempt_kind}'),
        'task_id': str(raw.get('task_id') or context.task_id),
        'attempt_id': str(raw.get('attempt_id') or context.attempt_id),
        'attempt_kind': str(raw.get('attempt_kind') or context.attempt_kind),
        'grader_result_id': raw.get('grader_result_id') or context.grader_result_id,
        'verification_status': status,
        'artifact_contract_ok': artifact_ok,
        'evidence_grounding_ok': grounding,
        'score_consistency_ok': consistency,
        'hidden_reference_leaked': leaked,
        'issues': list(dict.fromkeys(issues)),
        'confidence': _float(raw.get('confidence'), 0.7),
        'verifier_notes': raw.get('verifier_notes') or raw.get('notes') or raw.get('summary'),
        'semantic_evidence_grounding_ok': grounding,
        'unsupported_claims': unsupported,
        'leakage_check_ok': raw.get('leakage_check_ok') if isinstance(raw.get('leakage_check_ok'), bool) else not leaked,
        'model': raw.get('model') or context.model,
        'provider': raw.get('provider') or context.provider,
        'metadata_json': md,
    }

def normalize_role_output(role: str, raw: dict[str, Any], context: RoleContext | dict[str, Any] | None=None) -> dict[str, Any]:
    if isinstance(context, dict): context=RoleContext(**context)
    if role == 'intake_agent':
        return normalize_task_intake_result(raw, context)
    if role == 'rubric_agent':
        return normalize_rubric_result(raw, context)
    if role == 'evaluator_agent':
        return normalize_evaluator_result(raw, context)
    if role == 'grader_agent':
        return normalize_grader_result(raw, context)
    if role == 'verifier_agent':
        return normalize_verifier_result(raw, context)
    return raw
