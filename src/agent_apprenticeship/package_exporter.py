from __future__ import annotations
from pathlib import Path
import hashlib, mimetypes, json, shutil
from .schemas import *
from .io import write_json, append_jsonl, read_workflow_json as read_json
from .rubric_generation import worker_visible_markdown
from .artifact_resolver import artifact_ref_candidates, normalize_artifact_ref


SOURCE_FIELD_KEYS={'source_url_or_ref','source_kind','source_url','source_ref','source_basis','source_license'}
IGNORED_RELEASE_DIR_NAMES={
    'node_modules','.venv','__pycache__','.pytest_cache','.mypy_cache','.ruff_cache','.git','.cache','dist','build'
}

def is_ignored_release_path(path: Path | str) -> bool:
    parts=Path(path).parts
    for part in parts:
        if part in IGNORED_RELEASE_DIR_NAMES or 'pycache' in part.lower():
            return True
    return False

FINANCE_INPUTS = {
    'invoices.csv': """invoice_id,vendor,invoice_date,due_date,currency,amount
INV-1001,Acme Incorporated,2026-01-05,2026-02-04,USD,1200.00
INV-1002,Globex LLC,2026-01-08,2026-02-07,EUR,950.00
INV-1002,Globex LLC,2026-01-08,2026-02-07,EUR,950.00
INV-1003,Soylent Corp,2026-01-12,2026-02-11,GBP,500.00
INV-1004,Initech,2026-01-15,2026-02-14,USD,750.00
INV-1005,Acme Inc,2026-01-20,2026-02-19,JPY,100000
""",
    'payments.csv': """payment_id,payment_date,vendor_name,invoice_ref,currency,amount
PAY-9001,2026-02-03,ACME Inc.,INV-1001,USD,1200.00
PAY-9002,2026-02-08,Globex Corporation,INV-1002,EUR,500.00
PAY-9003,2026-02-10,Globex LLC,INV-1002,EUR,450.00
PAY-9004,2026-02-15,Soylent,INV-1003,GBP,550.00
PAY-9005,2026-02-18,Initech LLC,INV-9999,USD,300.00
PAY-9006,2026-02-20,Acme Incorporated,INV-1005,JPY,100000
""",
    'vendor_aliases.csv': """canonical_vendor,alias
Acme Incorporated,ACME Inc.
Acme Incorporated,Acme Inc
Globex LLC,Globex Corporation
Soylent Corp,Soylent
Initech,Initech LLC
""",
    'fx_rates.csv': """currency,usd_rate,effective_date
USD,1.0000,2026-02-01
EUR,1.1000,2026-02-01
GBP,1.2800,2026-02-01
JPY,0.0068,2026-02-01
""",
    'reconciliation_policy.md': """# Reconciliation policy

- Convert every invoice and payment amount to USD using `fx_rates.csv`.
- Resolve vendor names through `vendor_aliases.csv` before matching.
- Flag duplicate invoice IDs as `duplicate_invoice_id`.
- A payment can be within 3 calendar days after due date and still be on time.
- Categorize partial payments as `partial_payment` when total paid is less than invoice amount.
- Categorize overpayments as `overpayment` when total paid exceeds invoice amount by more than 1 USD.
- Categorize payments with no matching invoice as `missing_invoice_reference`.
- Produce vendor totals in USD and an audit summary with assumptions and exception counts.
""",
}

def init_package(root: Path, task_id: str) -> Path:
    p=root/'packages'/task_id
    for sub in ['task','task/task_instruction_assets','rubric','input','hidden_reference','attempts/baseline/artifacts','attempts/revised/artifacts','grading','feedback','signals']:
        (p/sub).mkdir(parents=True, exist_ok=True)
    return p

def _drop_source_fields(obj):
    if isinstance(obj, list):
        return [_drop_source_fields(v) for v in obj]
    if isinstance(obj, dict):
        return {k:_drop_source_fields(v) for k,v in obj.items() if k not in SOURCE_FIELD_KEYS and v is not None}
    return obj

def public_task_record(raw: RawTaskRecord | dict) -> dict:
    data=raw.model_dump(mode='json') if hasattr(raw, 'model_dump') else dict(raw)
    payload=dict(data.get('raw_payload') or {})
    if data.get('expected_economic_value') is None:
        data['expected_economic_value']=data.get('expected_pay') or payload.get('expected_economic_value') or payload.get('expected_pay')
    if data.get('expected_economic_value_for_agent_apprentice') is None:
        data['expected_economic_value_for_agent_apprentice']=data.get('expected_apprentice_pay') or payload.get('expected_economic_value_for_agent_apprentice') or payload.get('expected_apprentice_pay')
    data.pop('expected_pay', None)
    data.pop('expected_apprentice_pay', None)
    if isinstance(data.get('raw_payload'), dict):
        data['raw_payload']={k:v for k,v in data['raw_payload'].items() if k not in {'expected_pay','expected_apprentice_pay'}}
    return _drop_source_fields(data)

def public_task_intake_spec(spec: TaskIntakeSpec | dict) -> dict:
    data=spec.model_dump(mode='json') if hasattr(spec, 'model_dump') else dict(spec)
    if data.get('expected_economic_value') is None:
        data['expected_economic_value']=data.get('expected_pay')
    if data.get('expected_economic_value_for_agent_apprentice') is None:
        data['expected_economic_value_for_agent_apprentice']=data.get('expected_apprentice_pay')
    data.pop('expected_pay', None)
    data.pop('expected_apprentice_pay', None)
    return _drop_source_fields(data)

def _copy_asset(src: Path, dst_dir: Path) -> Path:
    dst_dir.mkdir(parents=True, exist_ok=True)
    target=dst_dir/src.name
    if src.resolve() == target.resolve() if target.exists() else False:
        return target
    if target.exists():
        stem=target.stem; suffix=target.suffix
        for i in range(2,1000):
            candidate=dst_dir/f'{stem}-{i}{suffix}'
            if not candidate.exists():
                target=candidate; break
    if src.is_dir():
        shutil.copytree(src, target, dirs_exist_ok=True)
    else:
        shutil.copy2(src, target)
    return target

def _task_brief(raw: RawTaskRecord, spec: TaskIntakeSpec) -> str:
    lines=[
        f"# {spec.normalized_title}",
        "",
        "## Instruction",
        spec.normalized_instruction,
        "",
        "## Expected deliverable",
        raw.expected_deliverable or raw.raw_payload.get('expected_deliverable') or spec.expected_agent_deliverable,
        "",
        "## Publishable task metadata",
    ]
    for key, value in {
        'domain': spec.domain,
        'subdomain': spec.subdomain,
        'apprenticeship_role': spec.apprenticeship_role,
        'task_family': spec.task_family,
        'difficulty_tier': spec.difficulty_tier,
        'needs_expert_review': spec.needs_expert_review,
        'expected_economic_value': spec.expected_economic_value or spec.expected_pay,
        'expected_economic_value_for_agent_apprentice': spec.expected_economic_value_for_agent_apprentice or spec.expected_apprentice_pay,
    }.items():
        if value is not None:
            lines.append(f"- {key}: {value}")
    return "\n".join(lines).rstrip()+"\n"

def materialize_task_inputs(package_root: Path, raw: RawTaskRecord, spec: TaskIntakeSpec) -> list[str]:
    input_dir = package_root/'input'
    asset_dir = package_root/'task'/'task_instruction_assets'
    input_dir.mkdir(parents=True, exist_ok=True)
    created=[]
    (input_dir/'task_brief.md').write_text(_task_brief(raw, spec))
    created.append('task_brief.md')
    write_json(input_dir/'task.json', public_task_record(raw))
    created.append('task.json')

    # Only materialize real local attachment refs or built-in fixture files with exact known names.
    # Instruction fragments are never converted into invented input filenames.
    for ref in raw.input_artifact_refs or []:
        src=Path(ref)
        if src.exists() and (src.is_file() or src.is_dir()):
            _copy_asset(src, asset_dir)
            target=_copy_asset(src, input_dir)
            created.append(target.name)
    for name in (raw.raw_payload.get('input_requirements') or []):
        if name in FINANCE_INPUTS:
            target=input_dir/name
            target.write_text(FINANCE_INPUTS[name])
            created.append(name)
    return list(dict.fromkeys(created))

def write_task_package(package_root: Path, raw: RawTaskRecord, spec: TaskIntakeSpec, quality: TaskIntakeQualityReport, rubric: RubricSpec, rubric_quality: RubricQualityReport):
    write_json(package_root/'task/raw_task_record.json', public_task_record(raw)); write_json(package_root/'task/task_intake_spec.json', public_task_intake_spec(spec)); write_json(package_root/'task/task_intake_quality_report.json', quality)
    materialize_task_inputs(package_root, raw, spec)
    write_json(package_root/'rubric/rubric.json', rubric); write_json(package_root/'rubric/rubric_quality_report.json', rubric_quality); (package_root/'rubric/worker_visible_rubric.md').write_text(worker_visible_markdown(rubric)); write_json(package_root/'rubric/verifier_private_rubric.json', rubric)
    for item in rubric.rubric_items: append_jsonl(package_root/'rubric/rubric_items.jsonl', item)
    (package_root/'README.md').write_text(f'# Task package {spec.task_id}\n\nLocal agent apprenticeship trace package.\n')
    write_json(package_root/'manifest.json', {'task_id': spec.task_id, 'schema_version':'aa-package-v0.1'})

def _media_type(path: Path, mime: str | None) -> str:
    suffix=path.suffix.lower()
    if mime and mime.startswith('image/'): return 'image'
    if mime and mime.startswith('audio/'): return 'audio'
    if mime and mime.startswith('video/'): return 'video'
    if suffix in {'.py','.js','.ts','.sh','.rs','.go','.java','.sql','.html','.css'}: return 'code'
    if suffix in {'.csv','.json','.jsonl','.yaml','.yml','.xml'}: return 'data'
    if suffix in {'.md','.txt','.log'}: return 'text'
    if suffix in {'.pdf','.doc','.docx'}: return 'document'
    if suffix in {'.zip','.tar','.gz'}: return 'archive'
    return 'unknown'

def _sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def _artifact_kind(path: Path, media_type: str) -> str:
    rel=path.as_posix()
    if '/artifacts/' in rel:
        return 'worker_output'
    if path.name in {'agent_trace.json','agent_trace.raw.json','agent_trace.normalized.json'}:
        return 'trace'
    if path.name in {'stdout.txt','stderr.txt','final_message.txt'} or media_type == 'text' and path.suffix == '.log':
        return 'log'
    if rel.startswith('input/') or '/input/' in rel:
        return 'input'
    if rel.startswith('grading/') or '/grading/' in rel:
        return 'evaluation'
    if rel.startswith('feedback/') or '/feedback/' in rel:
        return 'feedback'
    return 'package_file'

def _attempt_from_package_rel(package_root: Path, rel: Path) -> tuple[str | None, str | None]:
    parts=rel.parts
    if len(parts) >= 2 and parts[0] == 'attempts':
        attempt_kind=parts[1]
        return f"{package_root.name}_{attempt_kind}", attempt_kind
    return None, None

def _trace_artifact_links(package_root: Path) -> dict[str, list[dict]]:
    links: dict[str, list[dict]] = {}
    for trace_path in package_root.glob('attempts/*/agent_trace.json'):
        try:
            trace=read_json(trace_path)
        except Exception:
            continue
        attempt_id=trace.get('attempt_id')
        attempt_kind=trace.get('attempt_kind') or trace_path.parent.name
        trace_id=trace.get('trace_id')
        for step in trace.get('steps') or []:
            link={
                'attempt_id': attempt_id,
                'attempt_kind': attempt_kind,
                'trace_id': trace_id,
                'step': step.get('step'),
                'actor': step.get('actor'),
                'operation': step.get('operation'),
            }
            for ref in step.get('artifact_refs') or []:
                normalized=normalize_artifact_ref(ref)
                keys=artifact_ref_candidates(normalized) | {normalized, f'packages/{package_root.name}/{normalized}'}
                for key in keys:
                    if key:
                        bucket=links.setdefault(key, [])
                        if link not in bucket:
                            bucket.append(link)
    return links

def write_artifacts_index(package_root: Path):
    paths=[]
    trace_links=_trace_artifact_links(package_root)
    for f in package_root.rglob('*'):
        rel=f.relative_to(package_root)
        if 'agent_env' in rel.parts or is_ignored_release_path(rel):
            continue
        if f.is_file() and f.name != 'artifacts_index.json':
            mime=mimetypes.guess_type(f.name)[0]
            media_type=_media_type(f, mime)
            attempt_id, attempt_kind=_attempt_from_package_rel(package_root, rel)
            link_keys=artifact_ref_candidates(rel.as_posix()) | {rel.as_posix(), f'packages/{package_root.name}/{rel.as_posix()}'}
            linked=[]
            for key in link_keys:
                for link in trace_links.get(key, []):
                    if link not in linked:
                        linked.append(link)
            paths.append({
                'package_relative_path': rel.as_posix(),
                'artifact_ref': rel.as_posix(),
                'artifact_kind': _artifact_kind(rel, media_type),
                'linked_trace_steps': linked,
                'produced_by_attempt_id': attempt_id,
                'produced_by_attempt_kind': attempt_kind,
                'size_bytes': f.stat().st_size,
                'content_hash': _sha256(f),
                'mime_type': mime,
                'media_type': media_type,
                'preview_available': media_type in {'text','code','data','document'},
                'preview_truncated': False,
                'artifact_missing': False,
            })
    counters={'raw_trace_count':0,'raw_trace_step_count':0,'normalized_trace_count':0,'normalized_trace_step_count':0,'fallback_trace_count':0,'fallback_trace_step_count':0,'discarded_step_count':0,'raw_trace_parse_error_count':0,'trace_normalization_error_count':0,'trace_normalization_partial_count':0,'trace_lossless_count':0,'trace_lossless_failure_count':0}
    import json
    for report_path in package_root.glob('attempts/*/trace_normalization_report.json'):
        try: r=json.loads(report_path.read_text())
        except Exception: continue
        counters['raw_trace_count'] += 1 if r.get('raw_trace_ref') else 0
        counters['raw_trace_step_count'] += int(r.get('raw_step_count') or 0)
        counters['normalized_trace_count'] += 1 if r.get('normalized_trace_ref') else 0
        counters['normalized_trace_step_count'] += int(r.get('normalized_step_count') or 0)
        counters['fallback_trace_count'] += 1 if r.get('fallback_trace') else 0
        counters['fallback_trace_step_count'] += int(r.get('normalized_step_count') or 0) if r.get('fallback_trace') else 0
        counters['discarded_step_count'] += int(r.get('discarded_step_count') or 0)
        counters['raw_trace_parse_error_count'] += 1 if r.get('raw_trace_parse_error') else 0
        counters['trace_normalization_error_count'] += 1 if r.get('trace_normalization_error') else 0
        counters['trace_normalization_partial_count'] += 1 if r.get('trace_normalization_partial') else 0
        counters['trace_lossless_count'] += 1 if r.get('trace_lossless') else 0
        counters['trace_lossless_failure_count'] += 0 if r.get('trace_lossless') else 1
    manifest_path=package_root/'manifest.json'
    try: manifest=json.loads(manifest_path.read_text())
    except Exception: manifest={'task_id': package_root.name, 'schema_version':'aa-package-v0.1'}
    manifest['trace_counters']=counters
    write_json(manifest_path, manifest)
    write_json(package_root/'artifacts_index.json', paths)
    from .agent_env_exporter import export_agent_env
    export_agent_env(package_root)
    return paths
