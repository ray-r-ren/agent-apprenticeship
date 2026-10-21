import copy
import json
from pathlib import Path

import pytest

from agent_apprenticeship.atif import (
    NAMESPACE, dump_trace, from_workflow, load_trace, to_workflow, validate_trace,
)
from agent_apprenticeship.io import (
    append_jsonl, read_jsonl, read_workflow_jsonl, validate_trace_tree, write_json,
)


@pytest.fixture
def trajectory():
    return {
        'schema_version': 'ATIF-v1.8', 'session_id': 'run-1', 'trajectory_id': 'root',
        'agent': {'name': 'external-agent', 'version': '9', 'model_name': 'model-x'},
        'steps': [
            {'step_id': 1, 'source': 'user', 'message': [
                {'type': 'text', 'text': 'Inspect these inputs'},
                {'type': 'image', 'source': {'media_type': 'image/png', 'path': 'input/image.png'}},
                {'type': 'audio', 'source': {'media_type': 'audio/wav', 'path': 'input/clip.wav', 'duration_sec': 2.5}},
            ]},
            {'step_id': 2, 'source': 'agent', 'message': 'Parallel inspection',
             'tool_calls': [
                 {'tool_call_id': 'a', 'function_name': 'inspect', 'arguments': {'path': 'input/image.png'}},
                 {'tool_call_id': 'b', 'function_name': 'delegate', 'arguments': {'task': 'transcribe'}},
             ],
             'observation': {'results': [
                 {'source_call_id': 'a', 'content': 'image result'},
                 {'source_call_id': 'b', 'content': [{'type': 'audio', 'source': {'media_type': 'audio/flac', 'path': 'out.flac'}}],
                  'subagent_trajectory_ref': [{'trajectory_id': 'child'}]},
             ]}, 'metrics': {'prompt_tokens': 5, 'completion_tokens': 3},
             'extra': {'vendor_field': [1, 2], NAMESPACE: {'custom_extension': 'keep'}}},
        ],
        'subagent_trajectories': [{
            'schema_version': 'ATIF-v1.8', 'trajectory_id': 'child', 'session_id': 'run-1',
            'agent': {'name': 'transcriber', 'version': '1'},
            'steps': [{'step_id': 1, 'source': 'agent', 'message': 'Transcript'}],
        }],
        'continued_trajectory_ref': 'next.json',
        'extra': {'vendor': {'keep': True}, NAMESPACE: {'custom_root': 123}},
    }


def test_native_roundtrip_and_workflow_preserves_harbor_fields(tmp_path, trajectory):
    path = tmp_path / 'trajectory.json'
    dump_trace(path, trajectory)
    assert load_trace(path) == trajectory
    converted = from_workflow(to_workflow(trajectory))
    for key in ('agent', 'subagent_trajectories', 'continued_trajectory_ref'):
        assert converted[key] == trajectory[key]
    for original, step in zip(trajectory['steps'], converted['steps']):
        for key, value in original.items():
            if key != 'extra':
                assert step[key] == value
    assert converted['steps'][1]['extra'][NAMESPACE]['custom_extension'] == 'keep'
    assert converted['extra'][NAMESPACE]['custom_root'] == 123
    assert '_atif_original' not in json.dumps(converted)


@pytest.mark.parametrize('mutation', [
    lambda t: t.update(schema_version='ATIF-v1.7'),
    lambda t: t['steps'][1].update(step_id=4),
    lambda t: t['steps'][1]['observation']['results'][0].update(source_call_id='missing'),
    lambda t: t['steps'][0].update(model_name='agent-only'),
    lambda t: t['steps'][1].update(llm_call_count=0),
    lambda t: t['steps'][0]['message'][2]['source'].update(media_type='image/png'),
    lambda t: t['subagent_trajectories'].append(copy.deepcopy(t['subagent_trajectories'][0])),
    lambda t: t.update(trace_id='legacy-top-level'),
])
def test_invalid_atif_rejected(trajectory, mutation):
    mutation(trajectory)
    with pytest.raises(ValueError):
        validate_trace(trajectory)


def test_jsonl_contains_whole_trajectories_and_events(tmp_path, trajectory):
    path = tmp_path / 'agent_traces.jsonl'
    append_jsonl(path, trajectory)
    assert read_jsonl(path) == [trajectory]
    assert read_workflow_jsonl(path)[0]['trace_id'] == 'root'
    event_path = tmp_path / 'compiler_trace.jsonl'
    event = {'event_type': 'judge', 'round': 1, 'message': 'Accepted'}
    append_jsonl(event_path, event)
    assert read_jsonl(event_path)[0]['schema_version'] == 'ATIF-v1.8'
    assert read_workflow_jsonl(event_path) == [event]
    assert validate_trace_tree(tmp_path) == 2


def test_reject_legacy_without_overwriting_output(tmp_path):
    path = tmp_path / 'trajectory.json'
    path.write_text('unchanged')
    with pytest.raises(ValueError):
        dump_trace(path, {'trace_id': 'legacy', 'steps': []})
    assert path.read_text() == 'unchanged'
    path.write_text(json.dumps({'trace_id': 'legacy', 'steps': []}))
    with pytest.raises(ValueError):
        validate_trace_tree(tmp_path)


def test_nested_trace_export(tmp_path, trajectory):
    path = tmp_path / 'evaluation.json'
    write_json(path, {'trace': to_workflow(trajectory)})
    assert json.loads(path.read_text())['trace']['schema_version'] == 'ATIF-v1.8'
    assert validate_trace_tree(tmp_path) == 1


def test_schemas():
    root = Path(__file__).resolve().parents[1]
    from agent_apprenticeship.atif_models import Trajectory, Step
    for name, model in [('trace', Trajectory), ('trace_step', Step)]:
        schema = json.loads((root / f'schemas/{name}.schema.json').read_text())
        schema.pop('$schema')
        assert schema == model.model_json_schema()


def test_external_worker_trace_is_ingested_without_flattening(tmp_path, monkeypatch, trajectory):
    monkeypatch.setenv('AA_HOME', str(tmp_path / 'home'))
    from agent_apprenticeship.codex_runner import deterministic_attempt, ensure_attempt_outputs
    from agent_apprenticeship.package_exporter import init_package
    from agent_apprenticeship.schemas import RawTaskRecord
    from agent_apprenticeship.task_intake import task_intake

    raw = RawTaskRecord(raw_task_id='external', source_kind='synthetic', raw_title='Inspect inputs',
                        raw_description='Inspect the image and audio.', raw_payload={'deliverables': ['note.md']})
    spec, _ = task_intake(raw, tmp_path / 'roles')
    package = init_package(tmp_path / 'run', spec.task_id)
    deterministic_attempt(package, raw, spec)
    attempt = package / 'attempts/baseline'
    (attempt / 'agent_trace.raw.json').unlink()
    dump_trace(attempt / 'agent_trace.json', trajectory)
    _, _, valid = ensure_attempt_outputs(package, spec, 'baseline', 'task', 'workspace-write')
    assert valid
    result = load_trace(attempt / 'agent_trace.json')
    assert result['steps'][0]['message'] == trajectory['steps'][0]['message']
    assert result['steps'][1]['tool_calls'] == trajectory['steps'][1]['tool_calls']
    assert result['steps'][1]['observation'] == trajectory['steps'][1]['observation']
    assert result['subagent_trajectories'] == trajectory['subagent_trajectories']
    assert result['extra'][NAMESPACE]['task_id'] == spec.task_id
    assert load_trace(attempt / 'agent_trace.raw.json') == trajectory
