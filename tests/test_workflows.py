import json
from pathlib import Path

from agent_apprenticeship.atif import validate_trace
from agent_apprenticeship.io import validate_trace_tree


def test_run_release_compiler_bundle_and_fallback(tmp_path, monkeypatch):
    monkeypatch.setenv('AA_HOME', str(tmp_path / 'home'))
    from agent_apprenticeship.schemas import RawTaskRecord, TaskIntakeSpec
    from agent_apprenticeship.loop import run_task
    from agent_apprenticeship.release_exporter import create_release
    from agent_apprenticeship.bundle_exporter import create_contribution_bundle
    from agent_apprenticeship.config import get_settings
    from agent_apprenticeship.tdo import run_tdo_for_package
    from agent_apprenticeship.codex_runner import ensure_attempt_outputs
    from agent_apprenticeship.session_events import append_session_event, backfill_session_event_task_ids, read_session_events

    run = tmp_path / 'run'
    raw = RawTaskRecord(raw_task_id='test', source_kind='synthetic', raw_title='Create note',
                        raw_description='Write a planning note.', raw_payload={'deliverables': ['note.md']})
    package = run_task(raw, run, runner='deterministic', max_iterations=2)
    from agent_apprenticeship.package_exporter import _trace_artifact_links
    assert _trace_artifact_links(package)
    run_tdo_for_package(package, get_settings(), run_root=run)
    append_session_event(run, event_type='session_finished', run_id='run')
    backfill_session_event_task_ids(run, 'test')
    assert read_session_events(run)[0]['task_id'] == 'test'
    release = create_release(run, tmp_path / 'release')
    release_rows = [json.loads(line) for line in (release / 'agent_traces.jsonl').read_text().splitlines()]
    assert all(row['extra']['agent_apprenticeship']['attempt_status'] == 'completed' for row in release_rows)
    bundle = create_contribution_bundle(run, tmp_path / 'bundle', include_debug=True)
    from agent_apprenticeship.agent_env_verify import verify
    for root in (package, release, bundle):
        assert verify(root / "agent_env/artifacts/experience")
    for root in (run, release, bundle):
        assert validate_trace_tree(root) >= 6
        for path in root.rglob('agent_trace.json'):
            data = validate_trace(json.loads(path.read_text()))
            assert data['extra']['agent_apprenticeship']['task_id'] == 'test'
    assert list(package.rglob('compiler_trace.jsonl'))
    compiler = next(package.rglob('compiler_trace.jsonl'))
    assert all(validate_trace(json.loads(line)) for line in compiler.read_text().splitlines())

    spec = TaskIntakeSpec.model_validate(json.loads((package / 'task/task_intake_spec.json').read_text()))
    attempt = package / 'attempts/baseline'
    (attempt / 'agent_trace.raw.json').unlink()
    (attempt / 'agent_trace.json').write_text('{"steps": [{"step": 1}]}')
    _, _, valid = ensure_attempt_outputs(package, spec, 'baseline', 'task', 'workspace-write')
    assert not valid
    assert (attempt / 'rejected_agent_response.txt').exists()
    assert validate_trace_tree(package) >= 6
