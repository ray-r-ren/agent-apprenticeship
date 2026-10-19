from __future__ import annotations
import json, hashlib
from pathlib import Path
from pydantic import BaseModel
from .atif import encode_traces, decode_traces, validate_trace, event_trace, NAMESPACE

TRACE_FILES = {'agent_trace.json', 'agent_trace.raw.json', 'agent_trace.normalized.json', 'trajectory.json', 'trace.json', 'trace.jsonl', 'agent_traces.jsonl', 'raw_agent_traces.jsonl'}
EVENT_FILES = {'tdo_trace.jsonl', 'compiler_trace.jsonl', 'session_events.jsonl'}


def _encode(path: Path, obj):
    data = obj.model_dump(mode='json') if isinstance(obj, BaseModel) else obj
    data = encode_traces(data)
    if path.name in EVENT_FILES and isinstance(data, dict) and 'agent' not in data:
        data = event_trace(data, 'experience-compiler' if path.name != 'session_events.jsonl' else 'agent-apprenticeship')
    if path.name in TRACE_FILES | EVENT_FILES:
        data = validate_trace(data)
    return data


def write_json(path: Path, obj):
    data = _encode(path, obj)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False)+'\n', encoding='utf-8')


def read_json(path: Path):
    data = json.loads(path.read_text(encoding='utf-8'))
    if path.name in TRACE_FILES | EVENT_FILES:
        return validate_trace(data)
    return data


def append_jsonl(path: Path, obj):
    data = _encode(path, obj)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as f:
        f.write(json.dumps(data, sort_keys=True, ensure_ascii=False)+'\n')


def read_jsonl(path: Path):
    if not path.exists(): return []
    rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
    if path.name in TRACE_FILES | EVENT_FILES:
        return [validate_trace(row) for row in rows]
    return rows


def _workflow(data, path):
    if path.name in EVENT_FILES:
        return data['extra'][NAMESPACE]['event']
    return decode_traces(data)


def read_workflow_json(path: Path):
    return _workflow(read_json(path), path)


def read_workflow_jsonl(path: Path):
    return [_workflow(row, path) for row in read_jsonl(path)]


def sha256_file(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_trace_tree(root: Path) -> int:
    """Validate trace documents and embedded trajectories before package exchange."""
    count = 0

    def check(value):
        nonlocal count
        if isinstance(value, list):
            for item in value:
                check(item)
        elif isinstance(value, dict):
            if isinstance(value.get('steps'), list) and ('agent' in value or 'trace_id' in value):
                validate_trace(value)
                count += 1
            else:
                for item in value.values():
                    check(item)

    for path in root.rglob('*'):
        if not path.is_file() or path.suffix not in {'.json', '.jsonl'}:
            continue
        try:
            if path.suffix == '.jsonl':
                for row in read_jsonl(path):
                    check(row)
            else:
                check(read_json(path))
        except (ValueError, TypeError) as exc:
            if path.name in TRACE_FILES | EVENT_FILES:
                raise ValueError(f'{path}: invalid ATIF v1.8 trace: {exc}') from exc
            # Non-trace artifacts can contain arbitrary or partial JSON. A parsed
            # envelope containing a recognizable invalid trace must still fail.
            try:
                rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.suffix == '.jsonl' else [json.loads(path.read_text())]
            except ValueError:
                continue
            for row in rows:
                check(row)
    return count
