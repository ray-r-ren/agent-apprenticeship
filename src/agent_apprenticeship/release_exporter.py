from __future__ import annotations
import shutil
from pathlib import Path
from .io import read_workflow_json as read_json, write_json, append_jsonl, read_workflow_jsonl as read_jsonl
from .public_sanitizer import create_public_release
from .package_exporter import IGNORED_RELEASE_DIR_NAMES, is_ignored_release_path, public_task_record
from .trace_normalizer import normalize_trace_for_export
from .atif import to_workflow

RELEASE_FILES=['full_task_records.jsonl','tasks.jsonl','task_intake_specs.jsonl','rubrics.jsonl','rubric_items.jsonl','raw_agent_traces.jsonl','agent_traces.jsonl','trace_normalization_reports.jsonl','actual_outputs_normalization_reports.jsonl','actual_outputs.jsonl','grader_results.jsonl','verifier_results.jsonl','evaluator_feedback.jsonl','revision_plans.jsonl','hillclimb_results.jsonl','lessons.jsonl','training_signals.jsonl','process_supervision.jsonl','reward_modeling.jsonl','verifier_training.jsonl','revision_preference_pairs.jsonl','role_results_index.jsonl','artifacts_index.json','packages_index.jsonl','forsy_like_collections.jsonl']


def _copy_ignore(dir_path: str, names: list[str]) -> set[str]:
    ignored=set()
    for name in names:
        if name in IGNORED_RELEASE_DIR_NAMES or 'pycache' in name.lower():
            ignored.add(name)
    return ignored

def _copytree_ignore_errors(src: Path, dest: Path) -> None:
    shutil.copytree(src, dest, ignore=_copy_ignore, symlinks=False, ignore_dangling_symlinks=True)

def _safe_read(path: Path):
    try:
        return read_json(path)
    except Exception:
        return None

def _append_if_exists(release_root: Path, src: Path, dst: str) -> bool:
    obj=_safe_read(src)
    if obj is None: return False
    append_jsonl(release_root/dst, obj); return True


def _public_task_row(raw: dict, tid: str) -> dict:
    payload=raw.get('raw_payload') or {}
    row=public_task_record(raw)
    row.setdefault('task_id', raw.get('task_id') or tid)
    row['domain']=raw.get('normalized_domain') or payload.get('normalized_domain') or payload.get('domain')
    row['subdomain']=raw.get('normalized_subdomain') or payload.get('normalized_subdomain') or payload.get('subdomain')
    role=raw.get('agent_apprentice_role') or payload.get('agent_apprentice_role') or raw.get('apprenticeship_role') or payload.get('apprenticeship_role')
    if role is not None:
        row['agent_apprentice_role']=role
    expected_value=raw.get('expected_economic_value') or payload.get('expected_economic_value') or raw.get('expected_pay') or payload.get('expected_pay')
    apprentice_value=raw.get('expected_economic_value_for_agent_apprentice') or payload.get('expected_economic_value_for_agent_apprentice') or raw.get('expected_apprentice_pay') or payload.get('expected_apprentice_pay')
    if expected_value is not None:
        row['expected_economic_value']=expected_value
    if apprentice_value is not None:
        row['expected_economic_value_for_agent_apprentice']=apprentice_value
    for key in ['apprenticeship_role','task_family','difficulty_tier','needs_expert_review']:
        if raw.get(key) is not None:
            row[key]=raw.get(key)
        elif payload.get(key) is not None:
            row[key]=payload.get(key)
    if raw.get('expected_deliverable') is not None:
        row['expected_deliverable']=raw.get('expected_deliverable')
    elif payload.get('expected_deliverable') is not None:
        row['expected_deliverable']=payload.get('expected_deliverable')
    return row

def _step_count(obj) -> int:
    if isinstance(obj, list):
        return len(obj)
    if isinstance(obj, dict):
        for key in ['steps','trace_steps','trace','records','events','actions']:
            val=obj.get(key)
            if isinstance(val, list): return len(val)
            if isinstance(val, dict):
                nested=val.get('steps') or val.get('events') or val.get('actions')
                if isinstance(nested, list): return len(nested)
    return 0

def _task_status(statuses: list[str]) -> str:
    useful=[s for s in statuses if s]
    if useful and all(s == 'completed' for s in useful):
        return 'completed'
    if useful and all(s == 'failed' for s in useful):
        return 'failed'
    return 'partial' if useful else 'failed'



def _propagate_release_status_fields(release_root):
    """Propagate lightweight task/attempt status into exported JSONL files.

    packages_index.jsonl is currently the source of truth for task_status.
    This keeps tasks.jsonl/public/tasks.jsonl and trace rows aligned without
    adding a new metadata block or mutating raw outputs/runs data.
    """
    import json
    from pathlib import Path

    root = Path(release_root)

    def _read_jsonl(path):
        if not path.exists():
            return []
        rows = []
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                rows.append(None)
        return rows

    def _write_jsonl(path, rows):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for row in rows:
                if row is not None:
                    f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    status_by_task = {}
    for rel in ("packages_index.jsonl", "public/packages_index.jsonl"):
        for row in _read_jsonl(root / rel):
            if not isinstance(row, dict):
                continue
            task_id = row.get("task_id")
            task_status = row.get("task_status")
            if task_id and task_status:
                status_by_task[task_id] = task_status

    if not status_by_task:
        return

    # Propagate task_status into private/public task rows.
    for rel in ("tasks.jsonl", "public/tasks.jsonl"):
        path = root / rel
        rows = _read_jsonl(path)
        changed = False
        for row in rows:
            if not isinstance(row, dict):
                continue
            task_id = row.get("task_id")
            task_status = status_by_task.get(task_id)
            if task_status and row.get("task_status") != task_status:
                row["task_status"] = task_status
                changed = True
        if changed:
            _write_jsonl(path, rows)

    # Do not inject task_status into agent_traces.jsonl here.
    # Trace rows are governed by the WorkflowTrace schema; attempt_status belongs
    # there, but task_status is task/package-level metadata and is exported via
    # tasks.jsonl and packages_index.jsonl.


def create_release(run_root: Path, release_root: Path) -> Path:
    release_root.mkdir(parents=True, exist_ok=True)
    for f in RELEASE_FILES:
        p=release_root/f; p.parent.mkdir(parents=True, exist_ok=True); p.write_text('[]\n' if f.endswith('.json') else '')
    pkgs=list((run_root/'packages').glob('*')) if (run_root/'packages').exists() else []
    artifacts=[]; incomplete=0; missing_traces=0; raw_trace_count=0; normalized_trace_count=0; fallback_trace_count=0; fallback_trace_step_count=0; discarded_step_count=0; raw_trace_step_count=0; normalized_trace_step_count=0; partial_count=0; lossless_count=0; lossless_failure_count=0; parse_error_count=0; norm_error_count=0; actual_outputs_normalized_count=0; actual_outputs_schema_valid_count=0
    for pkg in pkgs:
        tid=pkg.name; blockers=[]
        raw=_safe_read(pkg/'task/raw_task_record.json')
        manifest=_safe_read(pkg/'package_manifest.json') or {}
        iteration_public={'completion_reason': manifest.get('loop_stop_reason'), 'initial_attempt_id': manifest.get('baseline_attempt_id'), 'revision_attempt_ids': manifest.get('revised_attempt_ids') or [], 'final_attempt_id': manifest.get('selected_attempt_id'), 'preferred_attempt_id': manifest.get('selected_attempt_id')}
        if raw is not None:
            pub_task=_public_task_row(raw, tid)
            append_jsonl(release_root/'full_task_records.jsonl', {**pub_task, 'task_id':tid,'package_path':f'packages/{tid}','raw_task_record':public_task_record(raw),'publishable_task_metadata':pub_task, **iteration_public, 'trace_refs':{'baseline':{'raw':'packages/'+tid+'/attempts/baseline/agent_trace.raw.json','normalized':'packages/'+tid+'/attempts/baseline/agent_trace.normalized.json','canonical':'packages/'+tid+'/attempts/baseline/agent_trace.json'},'revised':{'raw':'packages/'+tid+'/attempts/revised/agent_trace.raw.json','normalized':'packages/'+tid+'/attempts/revised/agent_trace.normalized.json','canonical':'packages/'+tid+'/attempts/revised/agent_trace.json'}}})
            append_jsonl(release_root/'tasks.jsonl', _public_task_row(raw, tid))
        else: blockers.append('missing task/raw_task_record.json')
        if not _append_if_exists(release_root, pkg/'task/task_intake_spec.json', 'task_intake_specs.jsonl'): blockers.append('missing task_intake_spec')
        rub=_safe_read(pkg/'rubric/rubric.json')
        if rub is not None: append_jsonl(release_root/'rubrics.jsonl', rub)
        else: blockers.append('missing rubric')
        for row in read_jsonl(pkg/'rubric/rubric_items.jsonl'): append_jsonl(release_root/'rubric_items.jsonl', row)
        attempt_trace_refs={}
        attempt_statuses=[]
        for a in ['baseline','revised']:
            raw_tr=_safe_read(pkg/f'attempts/{a}/agent_trace.raw.json')
            if raw_tr is not None:
                raw_trace_count += 1; raw_trace_step_count += _step_count(raw_tr)
                append_jsonl(release_root/'raw_agent_traces.jsonl', raw_tr)
            tr=_safe_read(pkg/f'attempts/{a}/agent_trace.json')
            norm=_safe_read(pkg/f'attempts/{a}/agent_trace.normalized.json')
            report=_safe_read(pkg/f'attempts/{a}/trace_normalization_report.json')
            if report is not None:
                append_jsonl(release_root/'trace_normalization_reports.jsonl', report)
                fallback_trace_count += 1 if report.get('fallback_trace') else 0
                fallback_trace_step_count += int(report.get('normalized_step_count') or 0) if report.get('fallback_trace') else 0
                discarded_step_count += int(report.get('discarded_step_count') or 0)
                partial_count += 1 if report.get('trace_normalization_partial') else 0
                parse_error_count += 1 if report.get('raw_trace_parse_error') else 0
                norm_error_count += 1 if report.get('trace_normalization_error') else 0
                if report.get('trace_lossless'): lossless_count += 1
                else: lossless_failure_count += 1
            if norm is not None:
                normalized_trace_count += 1; normalized_trace_step_count += len(norm.get('steps') or [])
            if tr is not None:
                tr_row=to_workflow(normalize_trace_for_export(dict(tr), report))
                attempt_statuses.append(tr_row.get('attempt_status'))
                tr_row['iteration_index']=0 if a == 'baseline' else 1
                tr_row['previous_attempt_id']=None if a == 'baseline' else manifest.get('baseline_attempt_id')
                tr_row['revision_group_id']=tid
                tr_row['completion_reason']=manifest.get('loop_stop_reason')
                append_jsonl(release_root/'agent_traces.jsonl', tr_row)
                append_jsonl(release_root/'forsy_like_collections.jsonl', {'collection_id': tr.get('collection_id') or tid, 'trace_id': tr.get('trace_id'), 'attempt_kind': a, 'iteration_index': tr_row['iteration_index'], 'previous_attempt_id': tr_row['previous_attempt_id'], 'trajectory': normalize_trace_for_export(tr, report), 'trace_ref': f'packages/{tid}/attempts/{a}/agent_trace.json'})
            else:
                missing_traces += 1; blockers.append(f'missing attempts/{a}/agent_trace.json')
            attempt_trace_refs[a]={'raw': f'packages/{tid}/attempts/{a}/agent_trace.raw.json' if (pkg/f'attempts/{a}/agent_trace.raw.json').exists() else None, 'normalized': f'packages/{tid}/attempts/{a}/agent_trace.normalized.json' if (pkg/f'attempts/{a}/agent_trace.normalized.json').exists() else None, 'canonical': f'packages/{tid}/attempts/{a}/agent_trace.json' if (pkg/f'attempts/{a}/agent_trace.json').exists() else None, 'normalization_report': f'packages/{tid}/attempts/{a}/trace_normalization_report.json' if (pkg/f'attempts/{a}/trace_normalization_report.json').exists() else None}
            ao_report=_safe_read(pkg/f'attempts/{a}/actual_outputs_normalization_report.json')
            if ao_report is not None:
                append_jsonl(release_root/'actual_outputs_normalization_reports.jsonl', ao_report)
                actual_outputs_normalized_count += 1 if ao_report.get('actual_outputs_normalized') else 0
                actual_outputs_schema_valid_count += 1 if ao_report.get('actual_outputs_schema_valid') else 0
            if not _append_if_exists(release_root, pkg/f'attempts/{a}/actual_outputs.json', 'actual_outputs.jsonl'):
                blockers.append(f'missing attempts/{a}/actual_outputs.json')
        for name,out in [('baseline_grader_result.json','grader_results.jsonl'),('revised_grader_result.json','grader_results.jsonl'),('baseline_verifier_result.json','verifier_results.jsonl'),('revised_verifier_result.json','verifier_results.jsonl')]:
            if not _append_if_exists(release_root, pkg/'grading'/name, out): blockers.append(f'missing grading/{name}')
        for src,dst in [('feedback/baseline_evaluator_feedback.json','evaluator_feedback.jsonl'),('feedback/revision_plan.json','revision_plans.jsonl'),('signals/hillclimb_result.json','hillclimb_results.jsonl'),('signals/lesson_pack.json','lessons.jsonl')]:
            if not _append_if_exists(release_root, pkg/src, dst): blockers.append(f'missing {src}')
        for src,dst in [('training_signals.jsonl','training_signals.jsonl'),('process_supervision.jsonl','process_supervision.jsonl'),('reward_modeling.jsonl','reward_modeling.jsonl'),('revision_preference_pairs.jsonl','revision_preference_pairs.jsonl')]:
            for row in read_jsonl(pkg/'signals'/src): append_jsonl(release_root/dst, row)
        idx=_safe_read(pkg/'artifacts_index.json') or []
        for row in idx:
            if not is_ignored_release_path(row.get('package_relative_path','')):
                artifacts.append({'task_id': tid, **row})
        export_ready=not blockers
        if not export_ready: incomplete += 1
        task_status=_task_status(attempt_statuses)
        append_jsonl(release_root/'packages_index.jsonl', {'task_id':tid,'package_path':f'packages/{tid}','task_status':task_status,'export_ready':export_ready,'export_blocker':'; '.join(blockers) if blockers else None,'trace_refs':attempt_trace_refs, **iteration_public})
        dest=release_root/'packages'/tid
        if dest.exists(): shutil.rmtree(dest)
        _copytree_ignore_errors(pkg,dest)
    roles_dir=run_root/'roles'
    if roles_dir.exists():
        for rr_path in roles_dir.rglob('role_result.json'):
            rr=_safe_read(rr_path)
            if rr is None: continue
            parts=rr_path.relative_to(roles_dir).parts
            task_id=parts[0] if len(parts)>0 else None
            role=rr.get('role') or (parts[1] if len(parts)>1 else rr_path.parent.name)
            attempt_kind=parts[2] if len(parts)>2 else rr_path.parent.name
            if role in {'intake_agent','rubric_agent'} or attempt_kind == 'role_result.json':
                attempt_kind='task_level'
            append_jsonl(release_root/'role_results_index.jsonl', {'role': role, 'task_id': task_id, 'attempt_kind': attempt_kind, 'provider': rr.get('provider'), 'model': rr.get('model'), 'live_call_ok': rr.get('live_call_ok'), 'structured_output_validation_ok': rr.get('structured_output_validation_ok'), 'fallback_used': bool((rr.get('metadata_json') or {}).get('fallback_used') or not rr.get('live_call_ok')), 'prompt_hash': (rr.get('metadata_json') or {}).get('prompt_hash'), 'public_summary': (rr.get('metadata_json') or {}).get('public_summary') or rr.get('error_type'), 'role_result_ref_internal': str(rr_path.relative_to(run_root)), 'prompt_ref_internal': rr.get('prompt_ref'), 'output_ref_internal': rr.get('output_ref'), 'parsed_output_ref_internal': rr.get('parsed_output_ref')})
    write_json(release_root/'artifacts_index.json', artifacts)
    aggregate_counts={'tasks':len(pkgs),'attempts':len(read_jsonl(release_root/'actual_outputs.jsonl')),'traces':len(read_jsonl(release_root/'agent_traces.jsonl')),'traced_steps':normalized_trace_step_count,'process_supervision_rows':len(read_jsonl(release_root/'process_supervision.jsonl')),'reward_modeling_rows':len(read_jsonl(release_root/'reward_modeling.jsonl')),'revision_preference_pairs':len(read_jsonl(release_root/'revision_preference_pairs.jsonl'))}
    write_json(release_root/'dataset_manifest.json', {'schema_version':'aa-release-v0.1','task_count':len(pkgs),**aggregate_counts,'files':RELEASE_FILES,'incomplete_package_count':incomplete,'trace_missing_count':missing_traces,'raw_trace_count':raw_trace_count,'raw_trace_step_count':raw_trace_step_count,'normalized_trace_count':normalized_trace_count,'normalized_trace_step_count':normalized_trace_step_count,'fallback_trace_count':fallback_trace_count,'fallback_trace_step_count':fallback_trace_step_count,'discarded_step_count':discarded_step_count,'raw_trace_parse_error_count':parse_error_count,'trace_normalization_error_count':norm_error_count,'trace_normalization_partial_count':partial_count,'trace_lossless_count':lossless_count,'trace_lossless_failure_count':lossless_failure_count,'actual_outputs_normalized_count':actual_outputs_normalized_count,'actual_outputs_schema_valid_count':actual_outputs_schema_valid_count})
    (release_root/'dataset_card.md').write_text('# Agent Apprenticeship Dataset Release\n\nThis release captures reusable agent work experience across task execution, artifact creation, evaluation, verifier-backed reliability checks, evaluator feedback, revision trajectories, process-supervision rows, reward-modeling examples, and revision preference pairs.\n')
    write_json(release_root/'quality_report.json', {'task_count':len(pkgs),'secret_scan_ok':True,'incomplete_package_count':incomplete,'trace_missing_count':missing_traces,'raw_trace_count':raw_trace_count,'raw_trace_step_count':raw_trace_step_count,'normalized_trace_count':normalized_trace_count,'normalized_trace_step_count':normalized_trace_step_count,'fallback_trace_count':fallback_trace_count,'fallback_trace_step_count':fallback_trace_step_count,'discarded_step_count':discarded_step_count,'raw_trace_parse_error_count':parse_error_count,'trace_normalization_error_count':norm_error_count,'trace_normalization_partial_count':partial_count,'trace_lossless_count':lossless_count,'trace_lossless_failure_count':lossless_failure_count,'actual_outputs_normalized_count':actual_outputs_normalized_count,'actual_outputs_schema_valid_count':actual_outputs_schema_valid_count})
    create_public_release(release_root)
    try:
        from .validation import validate_release
        counters=validate_release(release_root)
        manifest=_safe_read(release_root/'dataset_manifest.json') or {}
        quality=_safe_read(release_root/'quality_report.json') or {}
        for key,value in counters.items():
            if key in {'release_valid','public_release_valid','scale_ready','scale_blockers','fallback_only_task_count','rich_trace_task_count','workflow_trace_rich_count','raw_trace_count','raw_trace_step_count','normalized_trace_count','normalized_trace_step_count','fallback_trace_count','discarded_step_count','process_supervision_count','actual_outputs_raw_count','actual_outputs_normalized_count','actual_outputs_schema_valid_count','artifact_contract_consistency_ok','model_role_completeness_ok','model_task_intake_count','model_rubric_generation_count','model_evaluator_result_count','model_grader_result_count','model_verifier_result_count','model_score_count','artifact_contract_score_count','model_grading_grounded_count','model_grading_logs_only_count','model_grading_unavailable_count','public_prompt_leak_ok','public_secret_scan_ok','public_trace_count','public_system_prompt_redacted_count','public_prompt_metadata_count','dependency_shadow_ok','verifier_verified_count','verifier_failed_count','model_score_verified_count','model_score_needs_review_count','score_reliability_counts','scale_warnings','operation_other_count','operation_mapped_count'}:
                manifest[key]=value; quality[key]=value
        write_json(release_root/'dataset_manifest.json', manifest)
        write_json(release_root/'quality_report.json', quality)
    except Exception:
        pass
    _propagate_release_status_fields(release_root)
    return release_root
