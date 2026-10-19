from __future__ import annotations
import time, traceback, json
from datetime import datetime, timezone
from pathlib import Path
from .schemas import RawTaskRecord
from .io import read_workflow_jsonl as read_jsonl, append_jsonl, write_json
from .loop import run_task
from .release_exporter import create_release
from .public_sanitizer import classify_provider_failure
from .validation import validate_release


def _task_id_for_row(row: dict) -> str:
    raw_id = str(row.get('raw_task_id') or row.get('task_id') or 'task_unknown')
    payload = row.get('raw_payload') if isinstance(row.get('raw_payload'), dict) else {}
    return str(payload.get('task_id') or row.get('task_id') or raw_id.replace('raw_', 'task_'))


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')


def _progress(run_root: Path, task_index: int, task_count: int, task_id: str, stage: str, status: str = 'ok', message: str | None = None) -> None:
    msg = message or stage.replace('_', ' ')
    print(f'[{task_index}/{task_count}] {msg}', flush=True)
    append_jsonl(run_root/'progress.jsonl', {
        'timestamp': _utc_now(),
        'task_index': task_index,
        'task_count': task_count,
        'task_id': task_id,
        'stage': stage,
        'status': status,
        'message': msg,
    })


def _task_scale_ready(package_path: Path) -> bool | None:
    manifest = package_path/'package_manifest.json'
    if not manifest.exists():
        return None
    try:
        data=json.loads(manifest.read_text())
        if 'scale_ready_task' in data:
            return bool(data['scale_ready_task'])
    except Exception:
        return None
    return None


def run_batch(input_path: Path, output_root: Path, limit: int | None=None, resume=False, max_parallel=1, retry_limit=0, task_timeout_seconds=900, runner='deterministic', release_id: str | None=None, max_iterations: int | None=None) -> Path:
    run_id=release_id or f'run_{int(time.time())}'
    release_root=output_root/'releases'/run_id
    if release_root.exists() and not resume:
        raise RuntimeError('Release already exists. Use --resume or delete the existing output directory.')

    rows=read_jsonl(input_path)[:limit]
    task_ids=[_task_id_for_row(row) for row in rows]
    duplicates=sorted({tid for tid in task_ids if task_ids.count(tid) > 1})
    if duplicates:
        raise RuntimeError('Duplicate task IDs found in input file: ' + ', '.join(duplicates))

    run_root=output_root/'runs'/run_id; run_root.mkdir(parents=True, exist_ok=True); (run_root/'quarantine').mkdir(exist_ok=True)
    checkpoint=run_root/'checkpoint.json'; done=set()
    if resume and checkpoint.exists(): done=set(__import__('json').loads(checkpoint.read_text()).get('completed',[]))

    completed=0; failed=0
    total=len(rows)
    for idx, row in enumerate(rows, start=1):
        raw=RawTaskRecord.model_validate(row); tid=_task_id_for_row(row)
        started=time.time()
        _progress(run_root, idx, total, tid, 'starting', message=f'starting task_id={tid}')
        if tid in done or raw.raw_task_id in done:
            _progress(run_root, idx, total, tid, 'skipped', status='skipped', message='complete status=skipped')
            append_jsonl(run_root/'batch_status.jsonl', {'task_id':tid,'status':'skipped','package_path':None,'error_type':None,'error_message':None,'duration_seconds':round(time.time()-started,3),'scale_ready_task':None})
            continue
        try:
            _progress(run_root, idx, total, tid, 'baseline_started', message='baseline started')
            pkg=run_task(raw, run_root, runner=runner, max_iterations=max_iterations)
            _progress(run_root, idx, total, tid, 'baseline_complete', message='baseline complete')
            _progress(run_root, idx, total, tid, 'evaluation_complete', message='evaluation complete')
            _progress(run_root, idx, total, tid, 'revised_started', message='revised started')
            _progress(run_root, idx, total, tid, 'revised_complete', message='revised complete')
            done.add(pkg.name); done.add(tid); done.add(raw.raw_task_id)
            error_type=None; error_message=None; final_status='completed'
            try:
                attempts=[json.loads((pkg/'attempts/baseline/actual_outputs.json').read_text()), json.loads((pkg/'attempts/revised/actual_outputs.json').read_text())]
                if any((a.get('metadata_json') or {}).get('provider_failure_type')=='usage_limit' for a in attempts):
                    final_status='failed'; error_type='ProviderUsageLimit'; error_message='Provider usage limit encountered during attempt.'
                elif any(a.get('status') in ['failed','timeout','error'] for a in attempts):
                    final_status='failed'; error_type='AttemptCompletedWithErrors'; error_message='One or more attempts completed with error status.'
            except Exception as exc:
                final_status='failed'; error_type=type(exc).__name__; error_message=str(exc)
            if final_status == 'completed': completed += 1
            else: failed += 1
            _progress(run_root, idx, total, tid, 'package_exported', message='package exported')
            _progress(run_root, idx, total, tid, 'complete', status=final_status, message=f'complete status={"ok" if final_status == "completed" else "failed"}')
            append_jsonl(run_root/'batch_status.jsonl', {'task_id':pkg.name,'status':final_status,'package_path':str(pkg),'error_type':error_type,'error_message':error_message,'duration_seconds':round(time.time()-started,3),'scale_ready_task':_task_scale_ready(pkg)})
        except Exception as e:
            failed += 1
            cls=classify_provider_failure(str(e))
            etype=cls.get('error_type') or type(e).__name__
            emsg=str(e)
            _progress(run_root, idx, total, tid, 'complete', status='failed', message='complete status=failed')
            append_jsonl(run_root/'batch_status.jsonl', {'task_id':tid,'status':'failed','package_path':None,'error_type':etype,'error_message':emsg,'duration_seconds':round(time.time()-started,3),'scale_ready_task':False})
            (run_root/'quarantine'/f'{tid}.txt').write_text(traceback.format_exc())
        write_json(checkpoint, {'completed':sorted(done)})
    create_release(run_root, release_root)
    counters=validate_release(release_root)
    print('Run complete:', flush=True)
    print(f'tasks_total={total}', flush=True)
    print(f'tasks_completed={completed}', flush=True)
    print(f'tasks_failed={failed}', flush=True)
    print(f'release_path={release_root}', flush=True)
    print(f'scale_ready={counters.get("scale_ready")}', flush=True)
    print(f'scale_blockers={counters.get("scale_blockers")}', flush=True)
    return run_root
