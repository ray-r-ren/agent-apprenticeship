from __future__ import annotations
from .atif import from_workflow
import json
from pathlib import Path
from .schemas import VerifierResult, GraderResult, ActualOutputs, WorkflowTrace
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

def _model_verifier_active(settings) -> bool:
    return settings.mentor_mode in {'model_assisted', 'hybrid'} or settings.evaluation_mode in {'hybrid', 'llm_required'}

def deterministic_verify(grader: GraderResult, outputs: ActualOutputs) -> VerifierResult:
    artifact_ok=bool(grader.evidence_refs) and not grader.metadata_json.get('missing_artifacts') and outputs.status == 'success'
    score_ok=0 <= grader.score <= grader.max_score
    issues=[]
    if not artifact_ok:
        issues.append('missing deliverable evidence')
    if not score_ok:
        issues.append('score inconsistency')
    if grader.hidden_reference_leaked:
        issues.append('hidden reference leakage')
    return VerifierResult(verifier_result_id=f'verifier_{outputs.attempt_id}', task_id=outputs.task_id, attempt_id=outputs.attempt_id, attempt_kind=outputs.attempt_kind, grader_result_id=grader.grader_result_id, verification_status='not_run', artifact_contract_ok=artifact_ok, evidence_grounding_ok=bool(grader.evidence_refs), score_consistency_ok=score_ok, hidden_reference_leaked=grader.hidden_reference_leaked, issues=issues, confidence=0.6, verifier_notes='Structural guardrail checked artifact refs, score bounds, and leakage flags only; Mentor verification did not run.', semantic_evidence_grounding_ok=None, unsupported_claims=[], leakage_check_ok=not grader.hidden_reference_leaked, model=None, provider=None, metadata_json={'score_source': grader.score_source, 'structural_guardrail': True, 'role_status': 'mentor_verifier_unavailable'})

def _verifier_prompt(grader: GraderResult, outputs: ActualOutputs, trace: WorkflowTrace | None, artifact_preview_bundle: dict | None=None) -> str:
    return """Return only valid JSON matching VerifierResult. Do not include markdown.
Check whether the grader evidence is grounded in actual_outputs.json, artifact refs, trace evidence, and especially the artifact content previews. Check score consistency, unsupported claims, and hidden-reference leakage. Do not treat Apprentice Agent self-assessment or self evaluation as outcome evidence. Do not use deterministic fallback judgement; structural guardrails are package-integrity checks only and are not verifier judgments. Return exactly the VerifierResult schema fields. Do not add top-level fields not in schema. If you want to report extra checks such as reproducible, within_time_limit, uses_only_allowed_apis, passed_checks, or within_memory_limit, put them inside metadata_json.extra_model_fields.
GraderResult JSON:
""" + json.dumps(grader.model_dump(), sort_keys=True) + "\nActualOutputs JSON:\n" + json.dumps(outputs.model_dump(), sort_keys=True) + "\nArtifact content previews JSON:\n" + json.dumps(artifact_preview_bundle or {}, sort_keys=True)[:16000] + "\nTrace summary JSON:\n" + json.dumps((from_workflow(trace.model_dump()) if trace else {}), sort_keys=True)

def verify_attempt(grader: GraderResult, outputs: ActualOutputs, trace: WorkflowTrace | None=None, role_root: Path | None=None, package_root: Path | None=None) -> VerifierResult:
    settings=get_settings(); pre=deterministic_verify(grader, outputs); role_root=role_root or Path('outputs/roles')
    artifact_preview_bundle=build_artifact_previews(package_root, grader.evidence_refs or outputs.deliverable_refs or outputs.artifact_refs or outputs.files_created)
    if _model_verifier_active(settings) and settings.llm_verifier_enabled:
        prompt=_verifier_prompt(grader, outputs, trace, artifact_preview_bundle)
        try:
            provider=_mentor_provider_id()
            model_override=settings.llm_verifier_model if provider == 'openai' else None
            rr=run_structured_role('verifier_agent', prompt, VerifierResult, role_root/'verifier_agent'/outputs.attempt_kind, allow_fallback=settings.allow_deterministic_eval_fallback, model_override=model_override, normalizer_context={'task_id': outputs.task_id, 'attempt_id': outputs.attempt_id, 'attempt_kind': outputs.attempt_kind, 'grader_result_id': grader.grader_result_id, 'artifact_contract_score': grader.artifact_contract_score, 'evidence_refs': grader.evidence_refs, 'artifact_content_refs': artifact_preview_bundle.get('artifact_content_refs'), 'artifact_content_previews': artifact_preview_bundle.get('artifact_content_previews'), 'artifact_content_hashes': artifact_preview_bundle.get('artifact_content_hashes'), 'artifact_content_preview_truncated': artifact_preview_bundle.get('artifact_content_preview_truncated'), 'model_grading_basis': artifact_preview_bundle.get('model_grading_basis'), 'model': model_override or settings.model_provider_model, 'provider': provider})
            if rr.live_call_ok and rr.structured_output_validation_ok:
                parsed=read_json(role_root/'verifier_agent'/outputs.attempt_kind/'parsed_output.json')
                v=VerifierResult.model_validate(parsed)
                v.provider=rr.provider; v.model=rr.model; v.semantic_evidence_grounding_ok=v.semantic_evidence_grounding_ok if v.semantic_evidence_grounding_ok is not None else v.evidence_grounding_ok; v.leakage_check_ok=v.leakage_check_ok if v.leakage_check_ok is not None else not v.hidden_reference_leaked
                v.metadata_json.update({'llm_prompt_ref_internal':str(role_root/'verifier_agent'/outputs.attempt_kind/'prompt.md'),'llm_response_ref_internal':str(role_root/'verifier_agent'/outputs.attempt_kind/'raw_output.txt'),'prompt_hash':sha256_text(prompt),'public_response_summary':'Mentor Model verifier checked grounding and score consistency.', **artifact_preview_bundle})
                return v
            if settings.evaluation_mode == 'llm_required' or settings.llm_fail_closed:
                raise RuntimeError(rr.error_message or 'Model verifier failed')
        except Exception:
            if settings.evaluation_mode == 'llm_required' or settings.llm_fail_closed:
                raise
    if _mentor_provider_can_attempt() and settings.llm_verifier_enabled and settings.evaluation_mode != 'deterministic_only':
        pre.metadata_json.update({'model_verifier_enabled': True, 'llm_verifier_enabled': True, 'llm_unavailable': True, 'role_status': 'mentor_verifier_unavailable', **artifact_preview_bundle})
    return pre
