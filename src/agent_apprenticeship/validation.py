from __future__ import annotations
import json
from pathlib import Path
from .schemas import WorkflowTrace
from .io import read_workflow_jsonl as read_jsonl
from .env import contains_secret
from .public_sanitizer import has_prompt_leak
from .artifact_resolver import artifact_ref_resolves, is_artifact_evidence_ref, normalize_artifact_ref
REQUIRED=['dataset_manifest.json','dataset_card.md','quality_report.json','full_task_records.jsonl','tasks.jsonl','task_intake_specs.jsonl','rubrics.jsonl','rubric_items.jsonl','raw_agent_traces.jsonl','agent_traces.jsonl','trace_normalization_reports.jsonl','actual_outputs_normalization_reports.jsonl','actual_outputs.jsonl','grader_results.jsonl','verifier_results.jsonl','hillclimb_results.jsonl','training_signals.jsonl','process_supervision.jsonl','reward_modeling.jsonl','revision_preference_pairs.jsonl','role_results_index.jsonl','artifacts_index.json','packages_index.jsonl','forsy_like_collections.jsonl']

_normalize_artifact_evidence_ref = normalize_artifact_ref
_is_artifact_evidence_ref = is_artifact_evidence_ref
_artifact_ref_resolves = artifact_ref_resolves

def scan_tree_for_secrets(root: Path) -> bool:
    for p in root.rglob('*'):
        if p.is_file() and p.stat().st_size < 5_000_000:
            if contains_secret(p.read_text(errors='ignore')): return False
    return True


def validate_release(root: Path) -> dict[str, object]:
    counters={'release_valid': True,'public_release_valid': True, 'task_count':0,'trace_count':0,'trace_valid_count':0,'trace_invalid_count':0,'trace_missing_count':0,'incomplete_package_count':0,'raw_trace_count':0,'raw_trace_step_count':0,'normalized_trace_count':0,'normalized_trace_step_count':0,'fallback_trace_count':0,'fallback_trace_step_count':0,'discarded_step_count':0,'raw_trace_parse_error_count':0,'trace_normalization_error_count':0,'trace_normalization_partial_count':0,'trace_lossless_count':0,'trace_lossless_failure_count':0,'trace_step_count_total':0,'trace_steps_with_tool_count':0,'trace_steps_with_input_count':0,'trace_steps_with_output_count':0,'trace_steps_with_state_change_count':0,'trace_steps_with_reasoning_count':0,'artifact_count':0,'artifact_missing_count':0,'grader_result_count':0,'verifier_result_count':0,'hillclimb_result_count':0,'process_supervision_count':0,'reward_modeling_count':0,'revision_preference_pair_count':0,'secret_scan_ok':True,'llm_task_intake_count':0,'llm_rubric_generation_count':0,'llm_evaluator_result_count':0,'llm_grader_result_count':0,'llm_verifier_result_count':0,'deterministic_precheck_count':0,'semantic_score_count':0,'artifact_contract_score_count':0,'deterministic_fallback_count':0,'llm_unavailable_count':0,'score_source_counts':{},'dependency_shadow_ok':True,'actual_outputs_raw_count':0,'actual_outputs_normalized_count':0,'actual_outputs_schema_valid_count':0,'actual_outputs_fallback_count':0,'actual_outputs_inferred_artifact_count':0,'actual_outputs_discarded_field_count':0,'artifact_contract_consistency_ok':True,'artifact_contract_consistency_issues':[],'scale_ready':False,'scale_blockers':[],'fallback_only_task_count':0,'rich_trace_task_count':0,'workflow_trace_rich_count':0,'llm_role_completeness_ok':False,'semantic_grading_grounded_count':0,'semantic_grading_logs_only_count':0,'semantic_grading_unavailable_count':0,'model_task_intake_count':0,'model_rubric_generation_count':0,'model_evaluator_result_count':0,'model_grader_result_count':0,'model_verifier_result_count':0,'model_role_completeness_ok':False,'model_score_count':0,'model_grading_grounded_count':0,'model_grading_logs_only_count':0,'model_grading_unavailable_count':0,'verifier_verified_count':0,'verifier_failed_count':0,'model_score_verified_count':0,'model_score_needs_review_count':0,'score_reliability_counts':{},'scale_warnings':[],'operation_other_count':0,'operation_mapped_count':0}
    repo_root=Path.cwd()
    counters['dependency_shadow_ok']=not (repo_root/'src/pydantic').exists() and not (repo_root/'src/typer').exists()
    if not counters['dependency_shadow_ok']:
        counters['release_valid']=False
    for f in REQUIRED:
        if not (root/f).exists(): counters['release_valid']=False
    try:
        tasks=read_jsonl(root/'full_task_records.jsonl'); counters['task_count']=len(tasks)
        intake_specs=read_jsonl(root/'task_intake_specs.jsonl'); rubrics_rows=read_jsonl(root/'rubrics.jsonl')
        counters['llm_task_intake_count']=sum(1 for s in intake_specs if (s.get('metadata_json') or {}).get('intake_source')=='llm')
        counters['llm_rubric_generation_count']=sum(1 for r in rubrics_rows if (r.get('metadata_json') or {}).get('rubric_source')=='llm')
        graders=read_jsonl(root/'grader_results.jsonl'); verifiers=read_jsonl(root/'verifier_results.jsonl'); evaluators=read_jsonl(root/'evaluator_feedback.jsonl'); roles=read_jsonl(root/'role_results_index.jsonl')
        counters['grader_result_count']=len(graders); counters['verifier_result_count']=len(verifiers)
        counters['llm_grader_result_count']=sum(1 for g in graders if g.get('provider')=='openai' and g.get('semantic_score') is not None)
        counters['llm_verifier_result_count']=sum(1 for v in verifiers if v.get('provider')=='openai')
        counters['llm_evaluator_result_count']=sum(1 for e in evaluators if e.get('provider')=='openai')
        counters['llm_task_intake_count']=max(counters['llm_task_intake_count'], sum(1 for r in roles if r.get('role')=='intake_agent' and r.get('live_call_ok') and r.get('structured_output_validation_ok')))
        counters['llm_rubric_generation_count']=max(counters['llm_rubric_generation_count'], sum(1 for r in roles if r.get('role')=='rubric_agent' and r.get('live_call_ok') and r.get('structured_output_validation_ok')))
        counters['llm_evaluator_result_count']=max(counters['llm_evaluator_result_count'], sum(1 for r in roles if r.get('role')=='evaluator_agent' and r.get('live_call_ok') and r.get('structured_output_validation_ok')))
        counters['llm_grader_result_count']=max(counters['llm_grader_result_count'], sum(1 for r in roles if r.get('role')=='grader_agent' and r.get('live_call_ok') and r.get('structured_output_validation_ok')))
        counters['llm_verifier_result_count']=max(counters['llm_verifier_result_count'], sum(1 for r in roles if r.get('role')=='verifier_agent' and r.get('live_call_ok') and r.get('structured_output_validation_ok')))
        counters['deterministic_precheck_count']=sum(1 for g in graders if g.get('artifact_contract_score') is not None)
        counters['semantic_score_count']=sum(1 for g in graders if (g.get('model_score') is not None or g.get('semantic_score') is not None))
        counters['artifact_contract_score_count']=sum(1 for g in graders if g.get('artifact_contract_score') is not None)
        counters['deterministic_fallback_count']=sum(1 for g in graders if g.get('score_source')=='deterministic_fallback' or (g.get('metadata_json') or {}).get('deterministic_fallback')) + sum(1 for r in roles if r.get('fallback_used'))
        counters['llm_unavailable_count']=sum(1 for g in graders if (g.get('metadata_json') or {}).get('llm_unavailable')) + sum(1 for r in roles if r.get('provider')=='unavailable')
        counters['verifier_verified_count']=sum(1 for v in verifiers if v.get('verification_status')=='verified')
        counters['verifier_failed_count']=sum(1 for v in verifiers if v.get('verification_status')=='failed')
        counters['semantic_grading_grounded_count']=sum(1 for v in verifiers if v.get('verification_status')=='verified' and (v.get('semantic_evidence_grounding_ok') or v.get('evidence_grounding_ok')))
        rel_counts={}
        for g in graders:
            rel=(g.get('score_reliability') or (g.get('metadata_json') or {}).get('score_reliability') or 'unverified')
            rel_counts[rel]=rel_counts.get(rel,0)+1
        counters['score_reliability_counts']=rel_counts
        counters['model_score_verified_count']=rel_counts.get('verified',0)
        counters['model_score_needs_review_count']=sum(v for k,v in rel_counts.items() if k != 'verified')
        counters['semantic_grading_logs_only_count']=sum(1 for g in graders if ((g.get('metadata_json') or {}).get('model_grading_basis') or (g.get('metadata_json') or {}).get('semantic_grading_basis'))=='logs_only')
        counters['semantic_grading_unavailable_count']=sum(1 for g in graders if ((g.get('metadata_json') or {}).get('model_grading_basis') or (g.get('metadata_json') or {}).get('semantic_grading_basis'))=='missing_outputs')
        sc={};
        for g in graders: sc[g.get('score_source')]=sc.get(g.get('score_source'),0)+1
        counters['score_source_counts']=sc; counters['hillclimb_result_count']=len(read_jsonl(root/'hillclimb_results.jsonl')); counters['process_supervision_count']=len(read_jsonl(root/'process_supervision.jsonl')); counters['reward_modeling_count']=len(read_jsonl(root/'reward_modeling.jsonl')); counters['revision_preference_pair_count']=len(read_jsonl(root/'revision_preference_pairs.jsonl'))
        traces=read_jsonl(root/'agent_traces.jsonl'); counters['trace_count']=len(traces)
        for t in traces:
            try: WorkflowTrace.model_validate(t); counters['trace_valid_count']+=1
            except Exception: counters['trace_invalid_count']+=1; counters['release_valid']=False
            steps=t.get('steps') or []; counters['trace_step_count_total'] += len(steps)
            counters['trace_steps_with_tool_count'] += sum(1 for s in steps if s.get('tool'))
            counters['trace_steps_with_input_count'] += sum(1 for s in steps if s.get('input'))
            counters['trace_steps_with_output_count'] += sum(1 for s in steps if s.get('output'))
            counters['trace_steps_with_state_change_count'] += sum(1 for s in steps if s.get('state_change'))
            counters['trace_steps_with_reasoning_count'] += sum(1 for s in steps if s.get('reasoning'))
            counters['operation_other_count'] += sum(1 for s in steps if s.get('operation') == 'other')
            counters['operation_mapped_count'] += sum(1 for s in steps if s.get('operation') and s.get('operation') != 'other')
        reports=read_jsonl(root/'trace_normalization_reports.jsonl')
        ao_reports=read_jsonl(root/'actual_outputs_normalization_reports.jsonl')
        counters['actual_outputs_raw_count']=sum(int(r.get('actual_outputs_raw_count') or 0) for r in ao_reports)
        counters['actual_outputs_normalized_count']=sum(int(r.get('actual_outputs_normalized_count') or (1 if r.get('actual_outputs_normalized') else 0)) for r in ao_reports)
        counters['actual_outputs_schema_valid_count']=sum(int(r.get('actual_outputs_schema_valid_count') or (1 if r.get('actual_outputs_schema_valid') else 0)) for r in ao_reports)
        counters['actual_outputs_fallback_count']=sum(int(r.get('actual_outputs_fallback_count') or (1 if r.get('actual_outputs_fallback') else 0)) for r in ao_reports)
        counters['actual_outputs_inferred_artifact_count']=sum(int(r.get('actual_outputs_inferred_artifact_count') or 0) for r in ao_reports)
        counters['actual_outputs_discarded_field_count']=sum(int(r.get('actual_outputs_discarded_field_count') or 0) for r in ao_reports)
        if counters['actual_outputs_discarded_field_count']:
            counters['release_valid']=False
        counters['raw_trace_count']=len(read_jsonl(root/'raw_agent_traces.jsonl'))
        counters['normalized_trace_count']=sum(1 for r in reports if r.get('trace_normalized') or r.get('normalized_trace_ref'))
        counters['fallback_trace_count']=sum(1 for r in reports if r.get('fallback_trace'))
        counters['fallback_trace_step_count']=sum(int(r.get('normalized_step_count') or 0) for r in reports if r.get('fallback_trace'))
        counters['discarded_step_count']=sum(int(r.get('discarded_step_count') or 0) for r in reports)
        counters['raw_trace_step_count']=sum(int(r.get('raw_step_count') or 0) for r in reports)
        counters['normalized_trace_step_count']=sum(int(r.get('normalized_step_count') or 0) for r in reports)
        counters['raw_trace_parse_error_count']=sum(1 for r in reports if r.get('raw_trace_parse_error'))
        counters['trace_normalization_error_count']=sum(1 for r in reports if r.get('trace_normalization_error'))
        counters['trace_normalization_partial_count']=sum(1 for r in reports if r.get('trace_normalization_partial'))
        counters['trace_lossless_count']=sum(1 for r in reports if r.get('trace_lossless'))
        counters['trace_lossless_failure_count']=sum(1 for r in reports if not r.get('trace_lossless'))
        arts=json.loads((root/'artifacts_index.json').read_text() or '[]'); counters['artifact_count']=len(arts); counters['artifact_missing_count']=sum(1 for a in arts if a.get('artifact_missing') or (a.get('package_relative_path') and not (root/'packages'/str(a.get('task_id',''))/a.get('package_relative_path')).exists()))
        actual_outputs=read_jsonl(root/'actual_outputs.jsonl')
        existing_refs=set()
        for a in arts:
            tid=str(a.get('task_id') or '')
            rel=a.get('package_relative_path')
            if rel:
                existing_refs.add(str(rel)); existing_refs.add(f'packages/{tid}/{rel}')
        for ao in actual_outputs:
            for ref in (ao.get('deliverable_refs') or []) + (ao.get('artifact_refs') or []) + (ao.get('files_created') or []):
                if ref:
                    existing_refs.add(str(ref))
        verifiers_by_attempt={v.get('attempt_id'): v for v in verifiers}
        issues=[]
        for g in graders:
            md=g.get('metadata_json') or {}
            task_id=g.get('task_id'); attempt_kind=g.get('attempt_kind'); attempt_id=g.get('attempt_id')
            missing=[m for m in (md.get('missing_artifacts') or []) if m]
            mismatches=[m for m in (md.get('deliverable_mismatches') or []) if m and m.get('status') not in {'acceptable_alias','resolved'}]
            for m in missing:
                issues.append({'task_id':task_id,'attempt_kind':attempt_kind,'file':m,'field':'metadata_json.missing_artifacts','reason':'required artifact missing','evidence_ref':None})
            for m in mismatches:
                issues.append({'task_id':task_id,'attempt_kind':attempt_kind,'file':m.get('expected'),'field':'metadata_json.deliverable_mismatches','reason':'unresolved deliverable mismatch','evidence_ref':None})
            for ref in g.get('evidence_refs') or []:
                raw_ref=str(ref)
                normalized_ref=_normalize_artifact_evidence_ref(raw_ref)
                # Log/prompt/metadata refs are not artifact-contract evidence.
                # Artifact refs may be decorated as preview refs and should resolve directly or by package suffix.
                if _is_artifact_evidence_ref(raw_ref) and not _artifact_ref_resolves(normalized_ref, existing_refs):
                    issues.append({'task_id':task_id,'attempt_kind':attempt_kind,'file':None,'field':'evidence_refs','reason':'evidence_ref points to missing artifact','evidence_ref':raw_ref,'normalized_evidence_ref':normalized_ref})
            verifier=verifiers_by_attempt.get(attempt_id)
            if verifier and verifier.get('artifact_contract_ok') is True and missing:
                issues.append({'task_id':task_id,'attempt_kind':attempt_kind,'file':None,'field':'verifier.artifact_contract_ok','reason':'verifier artifact_contract_ok=true while required artifacts are missing','evidence_ref':None})
        counters['artifact_contract_consistency_issues']=issues
        counters['artifact_contract_consistency_ok']=len(issues)==0
        if issues:
            counters['release_valid']=False
        packages=read_jsonl(root/'packages_index.jsonl')
        counters['incomplete_package_count']=sum(1 for p in packages if p.get('export_ready') is False)
        counters['trace_missing_count']=max(0, len(packages)*2 - counters['trace_count'])
        if counters['discarded_step_count'] > 0 or counters['normalized_trace_step_count'] < counters['raw_trace_step_count']:
            counters['release_valid']=False
        if counters['incomplete_package_count']: counters['release_valid']=False
        if counters['artifact_missing_count']: counters['release_valid']=False
        if len(packages)!=counters['task_count']: counters['release_valid']=False
        counters['fallback_only_task_count']=counters['task_count'] if counters['task_count'] and counters['trace_count'] and counters['fallback_trace_count'] == counters['trace_count'] else 0
        counters['rich_trace_task_count']=counters['task_count'] if counters['raw_trace_step_count'] > 0 and counters['fallback_only_task_count'] == 0 else 0
        counters['workflow_trace_rich_count']=sum(1 for r in reports if not r.get('fallback_trace') and int(r.get('normalized_step_count') or 0) > 2)
        counters['model_task_intake_count']=counters['llm_task_intake_count']; counters['model_rubric_generation_count']=counters['llm_rubric_generation_count']; counters['model_evaluator_result_count']=counters['llm_evaluator_result_count']; counters['model_grader_result_count']=counters['llm_grader_result_count']; counters['model_verifier_result_count']=counters['llm_verifier_result_count']; counters['model_score_count']=counters['semantic_score_count']; counters['model_grading_grounded_count']=counters['semantic_grading_grounded_count']; counters['model_score_verified_count']=counters.get('model_score_verified_count',0); counters['model_score_needs_review_count']=counters.get('model_score_needs_review_count',0); counters['model_grading_logs_only_count']=counters['semantic_grading_logs_only_count']; counters['model_grading_unavailable_count']=counters['semantic_grading_unavailable_count']
        counters['llm_role_completeness_ok']=counters['llm_task_intake_count']>=1 and counters['llm_rubric_generation_count']>=1 and counters['llm_evaluator_result_count']>=1 and counters['llm_grader_result_count']>=2 and counters['llm_verifier_result_count']>=2; counters['model_role_completeness_ok']=counters['llm_role_completeness_ok']
    except Exception:
        counters['release_valid']=False
    counters['secret_scan_ok']=scan_tree_for_secrets(root)
    if not counters['secret_scan_ok']: counters['release_valid']=False
    public=root/'public'
    if public.exists():
        counters['public_prompt_leak_ok']=not has_prompt_leak(public)
        counters['public_secret_scan_ok']=scan_tree_for_secrets(public)
        try:
            ptraces=read_jsonl(public/'agent_traces.jsonl')
            counters['public_trace_count']=len(ptraces)
            counters['public_system_prompt_redacted_count']=sum(1 for t in ptraces if t.get('system_prompt') is None and t.get('system_prompt_hash'))
            counters['public_prompt_metadata_count']=sum(1 for t in ptraces if (t.get('metadata_json') or {}).get('prompt_template_id') or (t.get('metadata_json') or {}).get('prompt_template_version') or (t.get('metadata_json') or {}).get('prompt_publication_status'))
        except Exception:
            counters['public_release_valid']=False
        if not counters.get('public_prompt_leak_ok') or not counters.get('public_secret_scan_ok'):
            counters['public_release_valid']=False
    else:
        counters['public_release_valid']=False
    if not counters['public_release_valid']:
        counters['release_valid']=False
    if counters['trace_invalid_count'] or counters['trace_valid_count'] != counters['trace_count']: counters['release_valid']=False
    warnings=[]
    if counters.get('model_score_needs_review_count', 0) > 0:
        warnings.append('Some model-judged scores failed verifier grounding/consistency checks.')
    if counters.get('operation_other_count', 0) > counters.get('operation_mapped_count', 0) and counters.get('workflow_trace_rich_count', 0) > 0:
        warnings.append('Many rich-trace operations are still mapped to other.')
    counters['scale_warnings']=warnings
    blockers=[]
    if not counters['release_valid']: blockers.append('release_valid=false')
    if not counters['public_release_valid']: blockers.append('public_release_valid=false')
    if counters['discarded_step_count'] != 0: blockers.append('discarded_step_count_nonzero')
    if counters['fallback_only_task_count'] != 0: blockers.append('fallback_only_task_count_nonzero')
    if counters['raw_trace_step_count'] <= 0: blockers.append('raw_trace_step_count_zero')
    if counters['normalized_trace_count'] < 1: blockers.append('normalized_trace_count_lt_1')
    if counters['process_supervision_count'] <= 4: blockers.append('process_supervision_count_lte_4')
    if counters['actual_outputs_schema_valid_count'] < 1: blockers.append('actual_outputs_schema_valid_count_lt_1')
    if not counters['artifact_contract_consistency_ok']: blockers.append('artifact_contract_consistency_not_ok')

    requires_model_role_completeness = bool(
        counters.get('score_source_counts', {}).get('model_judged', 0)
        or counters.get('model_task_intake_count', 0)
        or counters.get('model_rubric_generation_count', 0)
        or counters.get('model_evaluator_result_count', 0)
        or counters.get('model_grader_result_count', 0)
        or counters.get('model_verifier_result_count', 0)
    )

    if requires_model_role_completeness and not counters['model_role_completeness_ok']:
        blockers.append('model_role_completeness_not_ok')
    if requires_model_role_completeness and counters['semantic_score_count'] < 2:
        blockers.append('model_score_count_lt_2')
    if requires_model_role_completeness and counters.get('model_grading_logs_only_count', 0) > 0 and counters.get('artifact_count', 0) > 0:
        blockers.append('model_grading_logs_only_with_artifacts')
    if not counters.get('public_prompt_leak_ok', False): blockers.append('public_prompt_leak')
    counters['scale_blockers']=blockers
    counters['scale_ready']=not blockers
    return counters

def format_counters(c: dict[str,object]) -> str:
    def val(v): return str(v).lower() if isinstance(v,bool) else str(v)
    return '\n'.join(f'{k}={val(v)}' for k,v in c.items())
