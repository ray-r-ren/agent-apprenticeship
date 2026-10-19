from __future__ import annotations
from pathlib import Path
from typing import Callable
from .schemas import RawTaskRecord, ActualOutputs, WorkflowTrace
from .config import get_settings
from .task_intake import task_intake
from .rubric_generation import generate_rubric
from .package_exporter import init_package, write_task_package, write_artifacts_index
from .apprentice_adapters import run_external_agent_attempt
from .codex_runner import deterministic_attempt, run_codex_attempt, run_custom_attempt
from .grader import grade_attempt, apply_score_reliability
from .verifier import verify_attempt
from .evaluator import evaluate_attempt
from .loop_review import reviewer_feedback_from_evaluator, write_loop_review_iteration
from .lesson_extractor import extract_lesson
from .training_signals import process_supervision_from_trace, reward_modeling, training_signals
from .revision import compute_hillclimb, preference_pair
from .io import write_json, append_jsonl, read_workflow_json as read_json


def _attempt(package_root, raw, spec, kind, runner, feedback=None):
    settings = get_settings()
    if runner == 'deterministic':
        return deterministic_attempt(package_root, raw, spec, kind, feedback)
    if runner == 'custom':
        return run_custom_attempt(package_root, raw, spec, kind, timeout=settings.task_timeout_seconds)
    if runner not in {'codex', 'deterministic', 'custom'}:
        return run_external_agent_attempt(package_root, raw, spec, kind, timeout=settings.task_timeout_seconds)
    return run_codex_attempt(package_root, raw, spec, kind, timeout=settings.task_timeout_seconds)


def _is_apprentice_operational_failure(actual: ActualOutputs) -> bool:
    metadata = actual.metadata_json or {}
    return bool(
        metadata.get("apprentice_agent_operational_error")
        or metadata.get("worker_agent_operational_error")
    )


def _write_baseline_only_signals(pkg: Path, b_actual: ActualOutputs, b_grade, b_ver, b_trace: WorkflowTrace, fb) -> None:
    lesson=extract_lesson(compute_hillclimb(b_grade, b_grade)); write_json(pkg/'signals/lesson_pack.json', lesson)
    for ex in process_supervision_from_trace(b_trace, b_grade, b_ver, fb): append_jsonl(pkg/'signals/process_supervision.jsonl', ex)
    append_jsonl(pkg/'signals/reward_modeling.jsonl', reward_modeling(b_actual,b_grade))


def _package_manifest(settings, **data):
    return {
        **data,
        "apprenticeship_mode": getattr(settings, "apprenticeship_mode", None),
        "mentor_mode": settings.mentor_mode,
    }


def _mode_label(settings) -> str:
    return getattr(settings, "apprenticeship_mode", None) or settings.mentor_mode


def run_task(
    raw: RawTaskRecord,
    run_root: Path,
    runner='deterministic',
    max_iterations: int | None=None,
    pre_attempt_callback: Callable[[Path], None] | None = None,
    attempt_complete_callback: Callable[[Path, str], None] | None = None,
    revision_decision_callback: Callable[[Path], bool] | None = None,
) -> Path:
    settings=get_settings()
    max_iter=max_iterations if max_iterations is not None else settings.max_iterations
    max_iter=max(1, int(max_iter))
    role_root=run_root/'roles'/raw.raw_task_id
    spec,q=task_intake(raw, role_root)
    rubric,rq=generate_rubric(spec, role_root)
    pkg=init_package(run_root, spec.task_id); write_task_package(pkg, raw, spec, q, rubric, rq)
    if pre_attempt_callback:
        pre_attempt_callback(pkg)
    actual_iterations=1
    baseline = _attempt(pkg, raw, spec, 'baseline', runner)
    b_actual=ActualOutputs.model_validate(read_json(pkg/'attempts/baseline/actual_outputs.json'))
    b_trace=WorkflowTrace.model_validate(read_json(pkg/'attempts/baseline/agent_trace.json'))
    if attempt_complete_callback:
        attempt_complete_callback(pkg, 'baseline')
    revised_attempt_ids=[]; selected_attempt_id=b_actual.attempt_id
    if _is_apprentice_operational_failure(b_actual):
        write_json(pkg/'package_manifest.json', _package_manifest(settings, max_iterations=max_iter,actual_iterations=1,loop_stop_reason='apprentice_agent_operational_error',baseline_attempt_id=b_actual.attempt_id,revised_attempt_ids=revised_attempt_ids,selected_attempt_id=selected_attempt_id))
        write_artifacts_index(pkg)
        return pkg
    b_grade=grade_attempt(rubric, b_actual, 'baseline', b_trace, role_root, pkg)
    b_ver=verify_attempt(b_grade, b_actual, b_trace, role_root, pkg)
    b_grade=apply_score_reliability(b_grade, b_ver)
    write_json(pkg/'grading/baseline_grader_result.json', b_grade); write_json(pkg/'grading/baseline_verifier_result.json', b_ver)
    write_loop_review_iteration(
        pkg,
        iteration=1,
        mode=_mode_label(settings),
        actual_outputs=b_actual,
        trace=b_trace,
        grader_result=b_grade,
        verifier_result=b_ver,
        stop_reason="awaiting_reviewer_feedback",
        continue_loop=False,
    )
    stop_reason='max_iterations_reached' if max_iter <= 1 else 'completed_baseline_and_revision'
    if max_iter <= 1:
        fb,rp=evaluate_attempt(b_grade, b_ver, b_actual, b_trace, f'{spec.task_id}_revised', role_root, pkg)
        write_json(pkg/'feedback/baseline_evaluator_feedback.json', fb); write_json(pkg/'feedback/revision_plan.json', rp)
        write_loop_review_iteration(
            pkg,
            iteration=1,
            mode=_mode_label(settings),
            actual_outputs=b_actual,
            trace=b_trace,
            grader_result=b_grade,
            verifier_result=b_ver,
            evaluator_feedback=fb,
            revision_plan=rp,
            reviewer_feedback=reviewer_feedback_from_evaluator(fb, b_grade, b_ver),
            stop_reason=stop_reason,
            continue_loop=False,
        )
        if revision_decision_callback:
            revision_decision_callback(pkg)
        _write_baseline_only_signals(pkg, b_actual, b_grade, b_ver, b_trace, fb)
        write_json(pkg/'package_manifest.json', _package_manifest(settings, max_iterations=max_iter,actual_iterations=actual_iterations,loop_stop_reason=stop_reason,baseline_attempt_id=b_actual.attempt_id,revised_attempt_ids=revised_attempt_ids,selected_attempt_id=selected_attempt_id))
        write_artifacts_index(pkg)
        return pkg
    fb,rp=evaluate_attempt(b_grade, b_ver, b_actual, b_trace, f'{spec.task_id}_revised', role_root, pkg)
    write_json(pkg/'feedback/baseline_evaluator_feedback.json', fb); write_json(pkg/'feedback/revision_plan.json', rp)
    baseline_reviewer_feedback=reviewer_feedback_from_evaluator(fb, b_grade, b_ver)
    write_loop_review_iteration(
        pkg,
        iteration=1,
        mode=_mode_label(settings),
        actual_outputs=b_actual,
        trace=b_trace,
        grader_result=b_grade,
        verifier_result=b_ver,
        evaluator_feedback=fb,
        revision_plan=rp,
        reviewer_feedback=baseline_reviewer_feedback,
        stop_reason="revision_requested",
        continue_loop=True,
    )
    if revision_decision_callback and not revision_decision_callback(pkg):
        stop_reason='mentor_decision_finish'
        existing_reviewer = read_json(pkg/'loops/iterations/001/reviewer_feedback.json') if (pkg/'loops/iterations/001/reviewer_feedback.json').exists() else {}
        if existing_reviewer.get('reviewer_type') not in {'human_expert','simulated_for_test','org_custom_reviewer'}:
            write_loop_review_iteration(
                pkg,
                iteration=1,
                mode=_mode_label(settings),
                actual_outputs=b_actual,
                trace=b_trace,
                grader_result=b_grade,
                verifier_result=b_ver,
                evaluator_feedback=fb,
                revision_plan=rp,
                reviewer_feedback=baseline_reviewer_feedback,
                stop_reason=stop_reason,
                continue_loop=False,
            )
        _write_baseline_only_signals(pkg, b_actual, b_grade, b_ver, b_trace, fb)
        write_json(pkg/'package_manifest.json', _package_manifest(settings, max_iterations=max_iter,actual_iterations=actual_iterations,loop_stop_reason=stop_reason,baseline_attempt_id=b_actual.attempt_id,revised_attempt_ids=revised_attempt_ids,selected_attempt_id=selected_attempt_id))
        write_artifacts_index(pkg)
        return pkg
    revised = _attempt(pkg, raw, spec, 'revised', runner, fb.feedback_summary)
    actual_iterations=2
    r_actual=ActualOutputs.model_validate(read_json(pkg/'attempts/revised/actual_outputs.json'))
    r_trace=WorkflowTrace.model_validate(read_json(pkg/'attempts/revised/agent_trace.json'))
    r_grade=grade_attempt(rubric, r_actual, 'revised', r_trace, role_root, pkg)
    r_ver=verify_attempt(r_grade, r_actual, r_trace, role_root, pkg)
    r_grade=apply_score_reliability(r_grade, r_ver)
    write_json(pkg/'grading/revised_grader_result.json', r_grade); write_json(pkg/'grading/revised_verifier_result.json', r_ver)
    write_loop_review_iteration(
        pkg,
        iteration=2,
        mode=_mode_label(settings),
        actual_outputs=r_actual,
        trace=r_trace,
        grader_result=r_grade,
        verifier_result=r_ver,
        stop_reason="awaiting_reviewer_feedback",
        continue_loop=False,
    )
    r_fb,r_rp=evaluate_attempt(r_grade, r_ver, r_actual, r_trace, f'{spec.task_id}_final', role_root, pkg)
    write_json(pkg/'feedback/revised_evaluator_feedback.json', r_fb); write_json(pkg/'feedback/revised_revision_plan.json', r_rp)
    revised_attempt_ids.append(r_actual.attempt_id)
    selected_attempt_id=r_actual.attempt_id if (r_grade.final_score or r_grade.score) >= (b_grade.final_score or b_grade.score) else b_actual.attempt_id
    write_loop_review_iteration(
        pkg,
        iteration=2,
        mode=_mode_label(settings),
        actual_outputs=r_actual,
        trace=r_trace,
        grader_result=r_grade,
        verifier_result=r_ver,
        evaluator_feedback=r_fb,
        revision_plan=r_rp,
        reviewer_feedback=reviewer_feedback_from_evaluator(r_fb, r_grade, r_ver),
        stop_reason=stop_reason,
        continue_loop=False,
    )
    hill=compute_hillclimb(b_grade, r_grade); write_json(pkg/'signals/hillclimb_result.json', hill)
    lesson=extract_lesson(hill); write_json(pkg/'signals/lesson_pack.json', lesson)
    for sig in training_signals(hill, ['attempts/baseline/agent_trace.json','attempts/revised/agent_trace.json'], ['grading/baseline_grader_result.json','grading/revised_grader_result.json'], ['grading/baseline_verifier_result.json','grading/revised_verifier_result.json']): append_jsonl(pkg/'signals/training_signals.jsonl', sig)
    for ex in process_supervision_from_trace(b_trace, b_grade, b_ver, fb)+process_supervision_from_trace(r_trace, r_grade, r_ver, fb): append_jsonl(pkg/'signals/process_supervision.jsonl', ex)
    append_jsonl(pkg/'signals/reward_modeling.jsonl', reward_modeling(b_actual,b_grade)); append_jsonl(pkg/'signals/reward_modeling.jsonl', reward_modeling(r_actual,r_grade))
    append_jsonl(pkg/'signals/revision_preference_pairs.jsonl', preference_pair(hill, 'rubric/rubric.json'))
    write_json(pkg/'package_manifest.json', _package_manifest(settings, max_iterations=max_iter,actual_iterations=actual_iterations,loop_stop_reason=stop_reason,baseline_attempt_id=b_actual.attempt_id,revised_attempt_ids=revised_attempt_ids,selected_attempt_id=selected_attempt_id))
    write_artifacts_index(pkg)
    return pkg
