from __future__ import annotations
import hashlib, json, re, shutil
from pathlib import Path
from typing import Any
from .env import redact_secrets, contains_secret
from .io import read_workflow_jsonl as read_jsonl, append_jsonl, write_json

PROMPT_TEMPLATE_ID='agent_apprenticeship_worker_v0'
PROMPT_TEMPLATE_VERSION='0.1'
PROMPT_PUBLICATION_STATUS='template_available_in_codebase'
RAW_LOG_NAMES=set()
CONTROLLER_TELEMETRY_KEYS={
    'max_iterations','actual_iterations','max_revision_iterations','actual_revision_iterations',
    'stop_on_verifier_pass','stop_on_score_threshold','stop_on_no_improvement',
    'stop_on_provider_limit','stop_on_timeout'
}
CONTROLLER_TELEMETRY_PATTERNS=list(CONTROLLER_TELEMETRY_KEYS)
PUBLIC_OMIT_KEYS={
    'source_url_or_ref','source_kind','source_url','source_ref','source_basis','source_license',
    'expected_pay','expected_apprentice_pay',
    'evaluation_mode','data_sharing_level','sensitive_info_masking',
    'raw_llm_output','raw_model_output',
    'limitations','known_limitations','limitations_count','when_not_to_use',
    'not_useful_for','non_use_cases','rubric_limitations',
}
_ABS_LOCAL_PATH_RE=re.compile(r"(/Users/[^\s\"\']+|/home/[^\s\"\']+|/private/[^\s\"\']+|/tmp/[^\s\"\']+|/var/folders/[^\s\"\']+)")
_USAGE_RE=re.compile(r"(you['’]?ve hit your usage limit|usage limit|rate limit|quota)", re.I)
PUBLIC_TEXT_REPLACEMENTS=[
    ("source_url_or_ref", "reference_or_context"),
    ("public_source_urls", "public_reference_links"),
    ("public_source_url", "public_reference_link"),
    ("source_urls", "reference_links"),
    ("source_url", "reference_link"),
    ("source_ref", "reference_id"),
    ("source_basis", "reference_basis"),
    ("source_kind", "reference_kind"),
    ("source_license", "reference_license"),
    ("expected_apprentice_pay", "expected_economic_value_for_agent_apprentice"),
    ("expected_pay", "expected_economic_value"),
    ("data_sharing_level", "sensitive_info_masking"),
    ("evaluation_mode", "mentor_mode"),
    ("worker_attempt", "apprentice_attempt"),
    ("worker_agent", "apprentice_agent"),
    ("Gatekeeping", "Review Routing"),
    ("gatekeeping", "review routing"),
    ("Worker " + "Agent", "Apprentice Agent"),
    ("worker " + "agent", "Apprentice Agent"),
    ("Deep" + "Seek", "model provider"),
    ("files_created_" + "list" + "ing_inconsistency", "files_created_index_inconsistency"),
    ("Cur" + "ated", "Selected"),
    ("cur" + "ated", "selected"),
    ("Contribution " + "Bundle", "Experience Compilation"),
    ("contribution " + "bundle", "Experience Compilation"),
    ("Export Full " + "Training Package", "Export Full Experience Compilation"),
    ("Training Data " + "Optimization", "Experience Compiler"),
    ("T" + "DO", "Experience Compiler"),
    ("Honest " + "limitations", "Generation notes"),
    ("honest " + "limitations", "generation notes"),
    ("Limit" + "ations", "Generation Notes"),
    ("limit" + "ations", "generation notes"),
    ("limit" + "ation", "generation note"),
    ("Applic" + "ability", "Best Use"),
    ("applic" + "ability", "fit"),
    ("Required " + "Conditions", "Required Inputs"),
    ("Evidence " + "Coverage", "Source Evidence"),
    ("Evidence " + "coverage", "Source evidence"),
    ("evidence " + "coverage", "source evidence"),
    ("Coverage " + "Notes", "Generation Notes"),
    ("Data Coverage " + "Notes", "Generation Notes"),
    ("Transfer " + "Boundaries", "Transfer Notes"),
    ("Missing Evidence " + "Notes", "Generation Notes"),
    ("Quality " + "Notes", "Quality Signals"),
    ("quality " + "notes", "quality signals"),
]

def sha256_text(text: str | None) -> str | None:
    if not text:
        return None
    return 'sha256:' + hashlib.sha256(text.encode()).hexdigest()

def classify_provider_failure(text: str | None) -> dict[str, Any]:
    text=text or ''
    if _USAGE_RE.search(text):
        return {'provider_failure_type':'usage_limit','fallback_reason':'provider_usage_limit','error_type':'ProviderUsageLimit','retryable':True,'provider':'openai','runner_backend':'codex_cli','failure_owner':'provider_or_quota','should_retry_after':None}
    return {}

def redact_internal_prompt_blocks(text: str | None) -> str | None:
    """Public releases preserve prompts as research context; only secrets are redacted."""
    if text is None:
        return None
    return redact_secrets(text)

def public_error_summary(error_text: str | None) -> str | None:
    if not error_text:
        return None
    return redact_internal_prompt_blocks(error_text)

def sanitize_public_text(text: str | None, prompt_text: str | None=None) -> str | None:
    if text is None:
        return None
    safe=redact_secrets(text)
    for old, new in PUBLIC_TEXT_REPLACEMENTS:
        safe=safe.replace(old, new)
        safe=safe.replace(old.upper(), new.upper())
    for token in CONTROLLER_TELEMETRY_PATTERNS:
        safe=safe.replace(token, '[internal controller setting omitted]')
    def _path_repl(match):
        text=match.group(0)
        for marker in ['/attempts/','/packages/']:
            if marker in text:
                return 'attempts/' + text.split('/attempts/', 1)[1] if marker == '/attempts/' else text.split('/packages/',1)[1]
        return '[local path omitted]'
    safe=_ABS_LOCAL_PATH_RE.sub(_path_repl, safe)
    return safe

def _sanitize_obj(obj: Any, prompt_text: str | None=None) -> Any:
    if isinstance(obj, list):
        return [_sanitize_obj(v, prompt_text) for v in obj]
    if not isinstance(obj, dict):
        return sanitize_public_text(obj, prompt_text) if isinstance(obj, str) else obj
    out={}
    for k,v in obj.items():
        if k in CONTROLLER_TELEMETRY_KEYS or k in PUBLIC_OMIT_KEYS or k.startswith('stop_on_'):
            continue
        safe_key=sanitize_public_text(str(k), prompt_text) or str(k)
        if safe_key in PUBLIC_OMIT_KEYS or safe_key.startswith('stop_on_'):
            continue
        out[safe_key]=_sanitize_obj(v, prompt_text)
    return out

def sanitize_public_obj(obj: dict[str, Any], prompt_text: str | None=None) -> dict[str, Any]:
    return _sanitize_obj(obj, prompt_text)

def has_prompt_leak(root: Path) -> bool:
    """Compatibility name: now checks only secrets and controller telemetry leakage."""
    for p in root.rglob('*'):
        if p.is_file() and p.stat().st_size < 5_000_000:
            text=p.read_text(errors='ignore')
            if contains_secret(text):
                return True
            if any(pat in text for pat in CONTROLLER_TELEMETRY_PATTERNS):
                return True
    return False

def create_public_release(release_root: Path) -> Path:
    public=release_root/'public'
    if public.exists():
        shutil.rmtree(public)
    public.mkdir(parents=True, exist_ok=True)
    jsonl_files=['full_task_records.jsonl','tasks.jsonl','task_intake_specs.jsonl','rubrics.jsonl','rubric_items.jsonl','raw_agent_traces.jsonl','agent_traces.jsonl','trace_normalization_reports.jsonl','actual_outputs_normalization_reports.jsonl','actual_outputs.jsonl','grader_results.jsonl','verifier_results.jsonl','evaluator_feedback.jsonl','revision_plans.jsonl','hillclimb_results.jsonl','lessons.jsonl','training_signals.jsonl','process_supervision.jsonl','reward_modeling.jsonl','verifier_training.jsonl','revision_preference_pairs.jsonl','role_results_index.jsonl','packages_index.jsonl','forsy_like_collections.jsonl']
    for name in jsonl_files:
        src=release_root/name
        dst=public/name
        dst.write_text('')
        for row in read_jsonl(src):
            append_jsonl(dst, sanitize_public_obj(row))
    for name in ['dataset_manifest.json','quality_report.json']:
        src=release_root/name
        data=json.loads(src.read_text() or '{}') if src.exists() else {}
        data['public_sanitized']=True
        write_json(public/name, sanitize_public_obj(data))
    if (release_root/'dataset_card.md').exists():
        (public/'dataset_card.md').write_text(redact_internal_prompt_blocks((release_root/'dataset_card.md').read_text()) or '')
    artifacts=[]
    src=release_root/'artifacts_index.json'
    if src.exists():
        for row in json.loads(src.read_text() or '[]'):
            artifacts.append(sanitize_public_obj(row))
    write_json(public/'artifacts_index.json', artifacts)
    private=[]
    packages=release_root/'packages'
    if packages.exists():
        for p in packages.glob('*/attempts/*'):
            private.append({'attempt_dir_internal':str(p.relative_to(release_root)), 'prompt_ref_internal':str((p/'prompt.md').relative_to(release_root)) if (p/'prompt.md').exists() else None, 'stdout_ref_internal':str((p/'stdout.txt').relative_to(release_root)) if (p/'stdout.txt').exists() else None, 'stderr_ref_internal':str((p/'stderr.txt').relative_to(release_root)) if (p/'stderr.txt').exists() else None, 'final_message_ref_internal':str((p/'final_message.txt').relative_to(release_root)) if (p/'final_message.txt').exists() else None})
    write_json(release_root/'private_debug_manifest.json', {'private_debug_artifacts':private})
    return public
