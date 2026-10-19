from __future__ import annotations
from .atif import from_workflow
from pathlib import Path
import json, re
from .schemas import GraderResult, RubricSpec, ActualOutputs, RubricItemScore
from .config import get_settings
from .schemas import WorkflowTrace
from .io import read_workflow_json as read_json, write_json
from .openai_structured import get_model_provider_status, run_structured_role
from .public_sanitizer import sha256_text
from .artifact_previews import build_artifact_previews

INPUT_ARTIFACT_NAMES = {
    'payments.csv', 'invoices.csv', 'fx_rates.csv', 'vendor_aliases.csv', 'reconciliation_policy.md'
}

def _mentor_provider_can_attempt() -> bool:
    return bool(get_model_provider_status().get('provider_available'))

def _mentor_provider_id() -> str:
    settings=get_settings()
    return settings.model_provider or 'openai'

def _model_grading_active(settings) -> bool:
    return settings.mentor_mode in {'model_assisted', 'hybrid'} or settings.evaluation_mode in {'hybrid', 'llm_required'}

def _is_output_artifact_name(name: str, output_names: set[str]) -> bool:
    base=Path(str(name)).name
    if base in INPUT_ARTIFACT_NAMES:
        return False
    return (not output_names) or base in output_names


def _expected_items_from_outputs(outputs: ActualOutputs) -> list[str]:
    md=outputs.metadata_json or {}
    items=md.get('expected_deliverable_items') or []
    return [Path(str(x)).name for x in items if x]

def _produced_items(outputs: ActualOutputs) -> list[str]:
    refs=list(outputs.deliverable_refs or []) + list(outputs.artifact_refs or []) + list(outputs.files_created or [])
    md=outputs.metadata_json or {}
    refs += [str(x) for x in (md.get('produced_deliverable_items') or [])]
    return list(dict.fromkeys(Path(str(x)).name for x in refs if x))

def _deliverable_match(outputs: ActualOutputs, evidence: list[str], missing: list[str]) -> dict[str, object]:
    expected=_expected_items_from_outputs(outputs)
    produced=_produced_items(outputs)
    expected_set=set(expected)
    produced_set=set(produced)
    if not expected:
        status='exact' if not missing else 'partial'
        mismatches=[]
    else:
        missing_expected=sorted(expected_set - produced_set)
        extra_produced=sorted(produced_set - expected_set)
        mismatches=[{'expected': m, 'status':'missing_or_substituted'} for m in missing_expected]
        status='exact' if not missing_expected else ('partial' if evidence else 'mismatch')
        if missing_expected and extra_produced:
            status='partial'
    return {'expected_deliverable_items': expected, 'produced_deliverable_items': produced, 'deliverable_match_status': status, 'deliverable_mismatches': mismatches}

def _candidate_refs(outputs: ActualOutputs, artifact_name: str, attempt_kind: str) -> list[str]:
    refs=[]
    refs.extend(outputs.deliverable_refs or [])
    refs.extend(outputs.artifact_refs or [])
    refs.extend(outputs.files_created or [])
    refs.append(f'attempts/{attempt_kind}/artifacts/{artifact_name}')
    refs.append(f'artifacts/{artifact_name}')
    out=[]
    for r in refs:
        if not r: continue
        r=str(r)
        if r.endswith('/'+artifact_name) or Path(r).name == artifact_name:
            if r.startswith('artifacts/'):
                r=f'attempts/{attempt_kind}/{r}'
            out.append(r)
    return list(dict.fromkeys(out))

def _resolve_required_artifacts(rubric: RubricSpec, outputs: ActualOutputs, attempt_kind: str, package_root: Path | None=None) -> tuple[list[str], list[str]]:
    output_refs=list(outputs.deliverable_refs or []) + list(outputs.artifact_refs or []) + list(outputs.files_created or [])
    output_names={Path(str(x)).name for x in output_refs if x and ('artifacts/' in str(x) or str(x).startswith(f'attempts/{attempt_kind}/'))}
    required=_expected_items_from_outputs(outputs)
    if not required:
        required=[]
    for item in ([] if required else rubric.rubric_items):
        required.extend(item.required_artifacts)

    if not required:
        required.extend(rubric.required_artifacts)
    required=[Path(x).name for x in required if x and _is_output_artifact_name(str(x), output_names)]
    if not required:
        required=[Path(x).name for x in output_refs if x and 'artifacts/' in str(x)]
    required=list(dict.fromkeys(required))
    evidence=[]; missing=[]
    refs=set(outputs.deliverable_refs + outputs.artifact_refs + outputs.files_created)
    for name in required:
        cands=_candidate_refs(outputs, name, attempt_kind)
        found=None
        for c in cands:
            path_ok = bool(package_root and (package_root/c).exists())
            if c in refs or path_ok:
                found=c; break
        if not found and package_root:
            c=f'attempts/{attempt_kind}/artifacts/{name}'
            if (package_root/c).exists():
                found=c
        if not found and not package_root:
            for c in cands:
                if Path(c).name == name and (c in refs or outputs.status == 'success'):
                    found=c; break
        if found: evidence.append(found)
        else: missing.append(name)
    return evidence, missing

def deterministic_grade(rubric: RubricSpec, outputs: ActualOutputs, attempt_kind: str, package_root: Path | None=None) -> GraderResult:
    evidence, missing = _resolve_required_artifacts(rubric, outputs, attempt_kind, package_root)
    match=_deliverable_match(outputs, evidence, missing)
    required_count=max(1, len(match.get('expected_deliverable_items') or missing) or len(evidence)+len(missing))
    artifact_contract_score = 1.0 if (outputs.status == 'success' and not missing and bool(evidence) and match['deliverable_match_status']=='exact') else max(0.0, min(1.0, len(evidence)/required_count))
    artifact_contract_passed = outputs.status == 'success' and artifact_contract_score >= 1.0 and not missing and match['deliverable_match_status']=='exact'
    item_scores=[]
    for item in rubric.rubric_items:
        output_names={Path(str(x)).name for x in (outputs.deliverable_refs or []) + (outputs.artifact_refs or []) + (outputs.files_created or []) if x and ('artifacts/' in str(x) or str(x).startswith(f'attempts/{attempt_kind}/'))}
        req=[Path(x).name for x in item.required_artifacts if _is_output_artifact_name(str(x), output_names)]
        item_evidence=[e for e in evidence if Path(e).name in req] or evidence
        item_missing=[m for m in missing if m in req]
        ok = outputs.status == 'success' and not item_missing and bool(item_evidence)
        item_scores.append(RubricItemScore(rubric_item_id=item.rubric_item_id, criterion_name=item.criterion_name, score=1.0 if ok else 0.0, max_score=1.0, passed=ok, evidence_refs=item_evidence, failure_mode=None if ok else 'missing_artifact', notes='Structural artifact-contract guardrail only; semantic correctness is not judged.', confidence=0.8, artifact_presence_ok=ok, semantic_correctness_score=None, reasoning_summary='Required artifact paths were resolved.' if ok else 'One or more required artifacts were not resolved.', improvement_suggestion=None if ok else 'Create the missing required artifact files.'))
    score=artifact_contract_score
    return GraderResult(grader_result_id=f'grader_{outputs.attempt_id}', task_id=outputs.task_id, attempt_id=outputs.attempt_id, attempt_kind=attempt_kind, rubric_id=rubric.rubric_id, grader_kind='structural_guardrail', score_source='structural_guardrail', score=score, max_score=1.0, passed=artifact_contract_passed, rubric_item_scores=item_scores, failed_criteria=[s.rubric_item_id for s in item_scores if not s.passed], passed_criteria=[s.rubric_item_id for s in item_scores if s.passed], evidence_refs=evidence, confidence=0.8, reasoning_summary='Structural guardrail checked artifact presence/readiness only; Mentor grading is unavailable unless a Mentor Model or human grader runs.', limitations=['semantic_quality_not_judged','mentor_grade_unavailable'], hidden_reference_used=False, hidden_reference_leaked=False, artifact_contract_score=artifact_contract_score, semantic_score=None, final_score=score, model=None, provider=None, deterministic_precheck_ref=None, public_response_summary='Structural artifact-contract guardrail completed.', score_reliability='needs_review', metadata_json={'structural_guardrail': True, 'role_status': 'mentor_grader_unavailable', 'artifact_contract_passed': artifact_contract_passed, 'semantic_quality_not_judged': True, 'missing_artifacts': missing, **match})

def _grader_prompt(rubric: RubricSpec, outputs: ActualOutputs, trace: WorkflowTrace | None, precheck: GraderResult, artifact_preview_bundle: dict | None=None) -> str:
    return """Return only valid JSON matching GraderResult. Do not include markdown.
You are the Mentor Model grader. Score each rubric item using the Apprentice-generated plus Mentor-audited task rubric, attempt trace, actual_outputs.json, artifact refs, and artifact content previews when available. Return a JSON object with exactly these required top-level fields: grader_result_id, task_id, attempt_id, attempt_kind, rubric_id, grader_kind, score_source, artifact_contract_score, model_score, semantic_score, final_score, score, max_score, passed, confidence, evidence_refs, rubric_item_scores, limitations, hidden_reference_used, hidden_reference_leaked, metadata_json. Each rubric_item_scores entry must include artifact_presence_ok, semantic_correctness_score, evidence_refs, reasoning_summary, confidence, failure_mode, and improvement_suggestion. Cite evidence_refs from actual_outputs/artifacts/trace, distinguish artifact presence from Mentor-judged correctness, and set score_source=model_judged and grader_kind=model. Do not rely on Apprentice Agent self-assessment or self evaluation. Do not use deterministic fallback judgement; structural artifact-contract guardrails are auxiliary only and are not grader judgments.
Rubric JSON:
""" + json.dumps(rubric.model_dump(), sort_keys=True) + "\nActualOutputs JSON:\n" + json.dumps(outputs.model_dump(), sort_keys=True) + "\nArtifact content previews JSON:\n" + json.dumps(artifact_preview_bundle or {}, sort_keys=True)[:16000] + "\nTrace summary JSON:\n" + json.dumps((from_workflow(trace.model_dump()) if trace else {}), sort_keys=True) + "\nStructural guardrail JSON:\n" + json.dumps(precheck.model_dump(), sort_keys=True)

def grade_attempt(rubric: RubricSpec, outputs: ActualOutputs, attempt_kind: str, trace: WorkflowTrace | None=None, role_root: Path | None=None, package_root: Path | None=None) -> GraderResult:
    settings=get_settings()
    pre=deterministic_grade(rubric, outputs, attempt_kind, package_root)
    role_root=role_root or Path('outputs/roles')
    artifact_preview_bundle=build_artifact_previews(package_root, pre.evidence_refs or outputs.deliverable_refs or outputs.artifact_refs or outputs.files_created)
    pre.metadata_json.update(artifact_preview_bundle)
    if _model_grading_active(settings) and settings.llm_grader_enabled:
        prompt=_grader_prompt(rubric, outputs, trace, pre, artifact_preview_bundle)
        (role_root/'grader_agent'/attempt_kind).mkdir(parents=True, exist_ok=True)
        structural_ref = role_root/'grader_agent'/attempt_kind/'structural_guardrail.json'
        write_json(structural_ref, pre)
        try:
            provider=_mentor_provider_id()
            model_override=settings.llm_grader_model if provider == 'openai' else None
            rr=run_structured_role('grader_agent', prompt, GraderResult, role_root/'grader_agent'/attempt_kind, allow_fallback=settings.allow_deterministic_eval_fallback, model_override=model_override, normalizer_context={'task_id': outputs.task_id, 'attempt_id': outputs.attempt_id, 'attempt_kind': attempt_kind, 'rubric_id': rubric.rubric_id, 'artifact_contract_score': pre.artifact_contract_score, 'evidence_refs': pre.evidence_refs, 'artifact_content_refs': artifact_preview_bundle.get('artifact_content_refs'), 'artifact_content_previews': artifact_preview_bundle.get('artifact_content_previews'), 'artifact_content_hashes': artifact_preview_bundle.get('artifact_content_hashes'), 'artifact_content_preview_truncated': artifact_preview_bundle.get('artifact_content_preview_truncated'), 'model_grading_basis': artifact_preview_bundle.get('model_grading_basis'), 'model': model_override or settings.model_provider_model, 'provider': provider})
            if rr.live_call_ok and rr.structured_output_validation_ok:
                parsed=read_json(role_root/'grader_agent'/attempt_kind/'parsed_output.json')
                g=GraderResult.model_validate(parsed)
                g.grader_kind='model'; g.score_source='model_judged'; g.legacy_score_source='llm_semantic'; g.artifact_contract_score=pre.artifact_contract_score; g.semantic_score=g.semantic_score if g.semantic_score is not None else g.score; g.model_score=g.model_score if g.model_score is not None else g.semantic_score; g.legacy_semantic_score=g.legacy_semantic_score if g.legacy_semantic_score is not None else g.semantic_score; g.final_score=g.final_score if g.final_score is not None else g.semantic_score; g.score=g.final_score; g.model=rr.model; g.provider=rr.provider; g.deterministic_precheck_ref=None; g.llm_prompt_ref_internal=str(role_root/'grader_agent'/attempt_kind/'prompt.md'); g.llm_response_ref_internal=str(role_root/'grader_agent'/attempt_kind/'raw_output.txt'); g.public_prompt_hash=sha256_text(prompt); g.public_response_summary=g.public_response_summary or 'Mentor Model grader scored rubric items with evidence refs and artifact previews when available.'
                g.metadata_json.update({'structural_guardrail_ref': str(structural_ref), 'structural_guardrail': pre.model_dump(), **artifact_preview_bundle, 'llm_role_result_ref_internal':str(role_root/'grader_agent'/attempt_kind/'role_result.json')})
                return g
            if settings.evaluation_mode == 'llm_required' or settings.llm_fail_closed:
                raise RuntimeError(rr.error_message or 'Model grader failed')
        except Exception:
            if settings.evaluation_mode == 'llm_required' or settings.llm_fail_closed:
                raise
    if settings.evaluation_mode == 'deterministic_only':
        pre.semantic_score=None; pre.final_score=pre.score
    elif _mentor_provider_can_attempt() and settings.llm_grader_enabled:
        pre.grader_kind='structural_guardrail'; pre.score_source='structural_guardrail'; pre.semantic_score=None; pre.final_score=pre.score
        provider=_mentor_provider_id()
        pre.model=settings.model_provider_model or settings.llm_grader_model; pre.provider=provider
        pre.limitations.append('Mentor Model grader unavailable; only structural guardrails ran.')
        pre.metadata_json.update({'llm_grader_enabled': True, 'llm_unavailable': True, 'role_status': 'mentor_grader_unavailable'})
    return pre


def apply_score_reliability(grader: GraderResult, verifier) -> GraderResult:
    status=getattr(verifier, 'verification_status', None) or 'not_run'
    issues=list(getattr(verifier, 'issues', None) or [])
    if status == 'verified':
        reliability='verified'
    elif status == 'failed':
        reliability='failed_verification'
    else:
        reliability='needs_review'
    grader.score_reliability=reliability
    grader.verifier_status=status
    grader.verifier_confidence=getattr(verifier, 'confidence', None)
    grader.verifier_issue_count=len(issues)
    grader.verifier_issues_summary='; '.join(str(x) for x in issues[:5]) if issues else None
    grader.metadata_json.update({
        'score_reliability': reliability,
        'verifier_status': status,
        'verifier_confidence': getattr(verifier, 'confidence', None),
        'verifier_issue_count': len(issues),
        'verifier_issues_summary': grader.verifier_issues_summary,
    })
    return grader
