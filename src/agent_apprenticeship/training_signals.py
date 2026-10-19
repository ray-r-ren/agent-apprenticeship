from __future__ import annotations
from .schemas import *
from .revision import preference_pair

def _current_evaluator_feedback_text(grader: GraderResult | None, evaluator: EvaluatorFeedback | None) -> str | None:
    if not evaluator:
        return None
    text=evaluator.feedback_summary
    md=(grader.metadata_json if grader else {}) or {}
    if md.get('artifact_contract_passed') is True and 'artifact_contract_score 0.0' in text:
        return 'Evaluator feedback omitted because it referenced stale artifact-contract state.'
    return text

def process_supervision_from_trace(trace: WorkflowTrace, grader: GraderResult | None=None, verifier: VerifierResult | None=None, evaluator: EvaluatorFeedback | None=None) -> list[ProcessSupervisionExample]:
    final=grader.score if grader else None
    evaluator_text=_current_evaluator_feedback_text(grader, evaluator)
    out=[]
    for s in trace.steps:
        if s.action == 'user_message': q='neutral'; reward=0.0; source='none'
        elif s.step_outcome in ['failed','blocked']: q='negative'; reward=-1.0; source='heuristic'
        elif s.step_outcome in ['progress','completed','corrected']: q='positive'; reward=1.0; source='heuristic'
        else: q='unknown'; reward=None; source='none'
        out.append(ProcessSupervisionExample(example_id=f'ps_{trace.trace_id}_{s.step}', task_id=trace.task_id, attempt_id=trace.attempt_id, attempt_kind=trace.attempt_kind, trace_id=trace.trace_id, step=s.step, actor=s.actor, action=s.action, operation=s.operation, tool=s.tool, observation=s.observation, input=s.input, output=s.output, state_change=s.state_change, reasoning=s.reasoning, caused_by=s.caused_by, causal_type=s.causal_type, success=s.success, step_outcome=s.step_outcome, step_quality_label=q, local_reward=reward, failure_mode=s.error_type, grader_feedback=grader.reasoning_summary if grader else None, verifier_feedback=verifier.verifier_notes if verifier else None, evaluator_feedback=evaluator_text, revision_reason=None, final_outcome_score=final, label_source=source, label=q, metadata_json={'raw_step': s.metadata_json.get('raw_step') if s.metadata_json else None}))
    return out

def reward_modeling(outputs: ActualOutputs, grader: GraderResult) -> RewardModelingExample:
    return RewardModelingExample(example_id=f'rm_{outputs.attempt_id}', task_id=outputs.task_id, attempt_id=outputs.attempt_id, rubric_ref=f'rubric/rubric.json', output_refs=outputs.deliverable_refs, attempt_summary=outputs.output_summary, rubric_item_scores=grader.rubric_item_scores, final_score=grader.final_score if grader.final_score is not None else grader.score, passed=grader.passed, failure_modes=grader.failed_criteria, grader_notes=grader.reasoning_summary, evidence_refs=grader.evidence_refs, score_source=grader.score_source, grader_kind=grader.grader_kind, confidence=grader.confidence, score_reliability=grader.score_reliability, verifier_status=grader.verifier_status, verifier_confidence=grader.verifier_confidence, verifier_issue_count=grader.verifier_issue_count, verifier_issues_summary=grader.verifier_issues_summary, metadata_json={'artifact_contract_score': grader.artifact_contract_score, 'model_score': grader.model_score if grader.model_score is not None else grader.semantic_score, 'legacy_semantic_score': grader.semantic_score, 'final_score': grader.final_score, 'verifier_confidence': grader.verifier_confidence, 'score_reliability': grader.score_reliability})

def training_signals(hill: HillclimbResult, traces: list[str], graders: list[str], verifiers: list[str]) -> list[TrainingSignal]:
    return [TrainingSignal(signal_id=f'sig_{hill.task_id}_rollout', task_id=hill.task_id, signal_type='rollout', source_attempt_ids=[hill.baseline_attempt_id,hill.revised_attempt_id], baseline_score=hill.baseline_score, revised_score=hill.revised_score, score_delta=hill.revision_score_delta, criteria_improved=hill.criteria_improved, criteria_regressed=hill.criteria_regressed, failed_criteria_before=hill.failed_criteria_before, failed_criteria_after=hill.failed_criteria_after, feedback_ref='feedback/baseline_evaluator_feedback.json', revision_plan_ref='feedback/revision_plan.json', grader_result_refs=graders, verifier_result_refs=verifiers, trace_refs=traces, artifact_refs=[], score_source='structural_guardrail', grader_kind='structural_guardrail', confidence=0.6, training_use_cases=['rollout','reward_modeling','revision_preference'], limitations=['Structural guardrails are package-integrity evidence only; Mentor/verifier/evaluator records are needed for semantic outcome labels.'], metadata_json={'structural_guardrail': True})]
