from __future__ import annotations
import json
from pathlib import Path
from pydantic import BaseModel
from .schemas import EvaluatorFeedback, GraderResult, VerifierResult, WorkflowTrace, ActualOutputs, RevisionPlan
from .config import get_settings
from .io import read_workflow_json as read_json
from .openai_structured import get_model_provider_status, run_structured_role
from .public_sanitizer import sha256_text
from .artifact_previews import build_artifact_previews

def _mentor_provider_can_attempt() -> bool:
    return bool(get_model_provider_status().get('provider_available'))

def _mentor_provider_id() -> str:
    settings=get_settings()
    return settings.model_provider or 'openai'

def _model_evaluator_active(settings) -> bool:
    return settings.mentor_mode in {'model_assisted', 'hybrid'} or settings.evaluation_mode in {'hybrid', 'llm_required'}

class LLMEvaluatorOutput(BaseModel):
    evaluator_feedback: EvaluatorFeedback
    revision_plan: RevisionPlan

def deterministic_feedback(grader: GraderResult) -> EvaluatorFeedback:
    failed = grader.failed_criteria if grader.score_source == 'structural_guardrail' else grader.failed_criteria
    summary = 'Mentor evaluator unavailable; no outcome judgment was produced. Structural guardrails suggest preserving valid artifacts and repairing missing deliverables.'
    return EvaluatorFeedback(feedback_id=f'feedback_{grader.attempt_id}', task_id=grader.task_id, attempt_id=grader.attempt_id, target_actor='apprentice', feedback_type='artifact_missing' if failed else 'other', failed_rubric_items=failed, evidence_refs=grader.evidence_refs, artifact_refs=grader.evidence_refs, feedback_summary=summary, actionable_feedback=['Preserve any valid generated artifacts.', 'Repair missing deliverables reported by structural guardrails.'] if failed else ['Preserve valid artifacts for any future revision.'], suggested_revision='Repair missing deliverables if the user or Mentor asks for a revision.', revision_priority='medium' if failed else 'low', confidence=0.35, hidden_reference_used=False, hidden_reference_leaked=False, failed_or_weak_rubric_items=failed, artifact_specific_comments=[], trace_specific_comments=[], revision_plan='Repair missing deliverables only if a revision is requested.', model=None, provider=None, metadata_json={'mentor_evaluator_status':'unavailable','structural_guidance':True})

def _revision_plan_from_feedback(fb: EvaluatorFeedback, target_attempt_id: str) -> RevisionPlan:
    return RevisionPlan(revision_plan_id=f'revision_plan_{fb.task_id}', task_id=fb.task_id, source_attempt_id=fb.attempt_id, target_attempt_id=target_attempt_id, revision_kind='local_fix', revision_reason=fb.feedback_summary, failed_rubric_items=fb.failed_rubric_items, planned_changes=fb.actionable_feedback, expected_score_improvement=0.1, risk_of_regression='low', uses_evaluator_feedback=True, metadata_json={})

def _latest_review_packet(package_root: Path | None, attempt_kind: str) -> dict:
    if not package_root:
        return {}
    packets = sorted((package_root / "loops" / "iterations").glob("*/review_packet.json"))
    if not packets:
        return {}
    preferred = []
    for path in packets:
        try:
            packet = read_json(path)
        except Exception:
            continue
        actual_ref = str(packet.get("actual_outputs_ref") or "")
        if f"attempts/{attempt_kind}/" in actual_ref:
            preferred.append((path, packet))
    if preferred:
        path, packet = preferred[-1]
        packet = dict(packet)
        packet["review_packet_ref"] = str(path.relative_to(package_root))
        return packet
    try:
        packet = read_json(packets[-1])
    except Exception:
        return {}
    packet = dict(packet)
    packet["review_packet_ref"] = str(packets[-1].relative_to(package_root))
    return packet


def _compact_trace_summary(trace: WorkflowTrace | None) -> dict:
    if trace is None:
        return {}
    from .atif import from_workflow
    return from_workflow(trace.model_dump())


def _compact_grader(grader: GraderResult) -> dict:
    data = grader.model_dump()
    return {
        "grader_result_id": data.get("grader_result_id"),
        "attempt_id": data.get("attempt_id"),
        "passed": data.get("passed"),
        "score": data.get("score"),
        "final_score": data.get("final_score"),
        "score_source": data.get("score_source"),
        "score_reliability": data.get("score_reliability"),
        "failed_criteria": data.get("failed_criteria") or [],
        "evidence_refs": (data.get("evidence_refs") or [])[:20],
    }


def _compact_verifier(verifier: VerifierResult) -> dict:
    data = verifier.model_dump()
    return {
        "verifier_result_id": data.get("verifier_result_id"),
        "attempt_id": data.get("attempt_id"),
        "passed": data.get("passed"),
        "verdict": data.get("verdict"),
        "failed_checks": data.get("failed_checks") or data.get("failed_criteria") or [],
        "evidence_refs": (data.get("evidence_refs") or [])[:20],
        "notes": str(data.get("notes") or data.get("verifier_notes") or "")[:1000],
    }


def _compact_actual(outputs: ActualOutputs) -> dict:
    data = outputs.model_dump()
    return {
        "task_id": data.get("task_id"),
        "attempt_id": data.get("attempt_id"),
        "attempt_kind": data.get("attempt_kind"),
        "status": data.get("status"),
        "output_summary": data.get("output_summary"),
        "primary_output_ref": data.get("primary_output_ref"),
        "deliverable_refs": data.get("deliverable_refs") or [],
        "artifact_refs": data.get("artifact_refs") or [],
        "files_created": data.get("files_created") or [],
        "error_type": data.get("error_type"),
        "error_message": data.get("error_message"),
    }


def _compact_artifact_previews(bundle: dict | None) -> dict:
    bundle = bundle or {}
    return {
        "artifact_content_refs": (bundle.get("artifact_content_refs") or [])[:12],
        "artifact_content_previews": (bundle.get("artifact_content_previews") or [])[:8],
        "artifact_content_hashes": (bundle.get("artifact_content_hashes") or {}) if len(str(bundle.get("artifact_content_hashes") or {})) < 2000 else {},
        "artifact_content_preview_truncated": bundle.get("artifact_content_preview_truncated"),
        "model_grading_basis": bundle.get("model_grading_basis"),
    }


def _evaluator_prompt(grader: GraderResult, verifier: VerifierResult, outputs: ActualOutputs, trace: WorkflowTrace | None, target_attempt_id: str, artifact_preview_bundle: dict | None=None, review_packet: dict | None=None) -> str:
    return """Return only valid JSON. Do not include markdown. Do not add extra top-level fields; place extras under metadata_json.extra_model_fields.
Required skeleton: {"evaluator_feedback":{"feedback_id":"feedback_<task_id>_<attempt_kind>","task_id":"...","attempt_id":"...","target_actor":"apprentice","feedback_type":"other","failed_rubric_items":[],"evidence_refs":[],"artifact_refs":[],"feedback_summary":"...","actionable_feedback":[],"suggested_revision":"...","revision_priority":"medium","confidence":0.7,"hidden_reference_used":false,"hidden_reference_leaked":false,"failed_or_weak_rubric_items":[],"artifact_specific_comments":[],"trace_specific_comments":[],"metadata_json":{}},"revision_plan":{"revision_plan_id":"revision_plan_<task_id>","task_id":"...","source_attempt_id":"...","target_attempt_id":"...","revision_kind":"local_fix","revision_reason":"...","failed_rubric_items":[],"planned_changes":[],"risk_of_regression":"medium","uses_evaluator_feedback":true,"metadata_json":{}}}.
Evaluate the attempt by first reading the loop review packet when provided, then checking compact Mentor grader/verifier outputs, actual_outputs.json, artifact refs, artifact content previews, and trace evidence. Treat the review packet as the current iteration dossier: it summarizes outputs, artifacts, trace refs, rubric/verifier status, known failures, diff evidence, and recommended review focus. Produce grounded actionable feedback and a concrete revision plan whose revision guidance can be mapped to feedback_type/feedback_content in future traces. Keep the response concise. Do not rely on Apprentice Agent self-assessment or self evaluation as outcome evidence. Do not use deterministic fallback judgement; structural guardrails are integrity checks only and are not outcome judgments.
Target revised attempt id: """ + target_attempt_id + "\nLoop Review Packet JSON:\n" + json.dumps(review_packet or {}, sort_keys=True)[:5000] + "\nCompact GraderResult JSON:\n" + json.dumps(_compact_grader(grader), sort_keys=True) + "\nCompact VerifierResult JSON:\n" + json.dumps(_compact_verifier(verifier), sort_keys=True) + "\nCompact ActualOutputs JSON:\n" + json.dumps(_compact_actual(outputs), sort_keys=True) + "\nArtifact content previews JSON:\n" + json.dumps(_compact_artifact_previews(artifact_preview_bundle), sort_keys=True)[:5000] + "\nCompact Trace summary JSON:\n" + json.dumps(_compact_trace_summary(trace), sort_keys=True)

def evaluate_attempt(grader: GraderResult, verifier: VerifierResult, outputs: ActualOutputs, trace: WorkflowTrace | None, target_attempt_id: str, role_root: Path | None=None, package_root: Path | None=None) -> tuple[EvaluatorFeedback, RevisionPlan]:
    settings=get_settings(); role_root=role_root or Path('outputs/roles')
    artifact_preview_bundle=build_artifact_previews(package_root, grader.evidence_refs or outputs.deliverable_refs or outputs.artifact_refs or outputs.files_created)
    review_packet=_latest_review_packet(package_root, outputs.attempt_kind)
    if _model_evaluator_active(settings) and settings.llm_evaluator_enabled:
        prompt=_evaluator_prompt(grader, verifier, outputs, trace, target_attempt_id, artifact_preview_bundle, review_packet)
        try:
            provider=_mentor_provider_id()
            model_override=settings.llm_evaluator_model if provider == 'openai' else None
            rr=run_structured_role('evaluator_agent', prompt, LLMEvaluatorOutput, role_root/'evaluator_agent'/outputs.attempt_kind, allow_fallback=settings.allow_deterministic_eval_fallback, model_override=model_override, normalizer_context={'task_id': outputs.task_id, 'attempt_id': outputs.attempt_id, 'attempt_kind': outputs.attempt_kind, 'target_attempt_id': target_attempt_id, 'grader_result_id': grader.grader_result_id, 'verifier_result_id': verifier.verifier_result_id, 'artifact_content_refs': artifact_preview_bundle.get('artifact_content_refs'), 'artifact_content_previews': artifact_preview_bundle.get('artifact_content_previews'), 'artifact_content_hashes': artifact_preview_bundle.get('artifact_content_hashes'), 'artifact_content_preview_truncated': artifact_preview_bundle.get('artifact_content_preview_truncated'), 'model_grading_basis': artifact_preview_bundle.get('model_grading_basis'), 'review_packet_ref': review_packet.get('review_packet_ref'), 'model': model_override or settings.model_provider_model, 'provider':provider})
            if rr.live_call_ok and rr.structured_output_validation_ok:
                parsed=read_json(role_root/'evaluator_agent'/outputs.attempt_kind/'parsed_output.json')
                fb=EvaluatorFeedback.model_validate(parsed['evaluator_feedback'])
                rp=RevisionPlan.model_validate(parsed['revision_plan'])
                fb.provider=rr.provider; fb.model=rr.model; fb.revision_plan=fb.revision_plan or rp.revision_reason
                fb.metadata_json.update({'llm_prompt_ref_internal':str(role_root/'evaluator_agent'/outputs.attempt_kind/'prompt.md'),'llm_response_ref_internal':str(role_root/'evaluator_agent'/outputs.attempt_kind/'raw_output.txt'),'prompt_hash':sha256_text(prompt),'public_response_summary':'Mentor Model evaluator generated grounded feedback and a revision plan from the loop review packet and package evidence.','review_source':'review_packet','review_packet_ref':review_packet.get('review_packet_ref'), **artifact_preview_bundle})
                rp.metadata_json.update({'provider':rr.provider,'model':rr.model,'review_source':'review_packet','review_packet_ref':review_packet.get('review_packet_ref'),'llm_prompt_ref_internal':str(role_root/'evaluator_agent'/outputs.attempt_kind/'prompt.md'),'llm_response_ref_internal':str(role_root/'evaluator_agent'/outputs.attempt_kind/'raw_output.txt')})
                return fb,rp
            if settings.evaluation_mode == 'llm_required' or settings.llm_fail_closed:
                raise RuntimeError(rr.error_message or 'Model evaluator failed')
        except Exception:
            if settings.evaluation_mode == 'llm_required' or settings.llm_fail_closed:
                raise
    fb=deterministic_feedback(grader)
    rp=_revision_plan_from_feedback(fb, target_attempt_id)
    fb.metadata_json.update({'review_source':'review_packet', 'review_packet_ref':review_packet.get('review_packet_ref')})
    rp.metadata_json.update({'review_source':'review_packet', 'review_packet_ref':review_packet.get('review_packet_ref')})
    if _mentor_provider_can_attempt() and settings.llm_evaluator_enabled and settings.evaluation_mode != 'deterministic_only':
        fb.metadata_json.update({'model_evaluator_enabled': True, 'llm_evaluator_enabled': True, 'llm_unavailable': True, 'mentor_evaluator_status': 'unavailable', 'review_source':'review_packet', 'review_packet_ref':review_packet.get('review_packet_ref'), **artifact_preview_bundle})
    return fb,rp
