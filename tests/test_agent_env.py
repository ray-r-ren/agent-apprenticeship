import json
from pathlib import Path
import subprocess
import sys

import pytest

from agent_apprenticeship.agent_env_exporter import export_agent_env
from agent_apprenticeship.agent_env_verify import verify


def test_portable_bundle_and_integrity(tmp_path):
    source = tmp_path / 'source'
    (source / 'task').mkdir(parents=True)
    (source / 'task/task.json').write_text('{"task_id":"test"}')
    (source / 'input').mkdir()
    (source / 'input/media.bin').write_bytes(bytes(range(256)))
    outside = tmp_path / 'private.txt'
    outside.write_text('private')
    (source / 'input/outside').symlink_to(outside)
    bundle = export_agent_env(source)
    artifact = bundle / 'artifacts/experience'
    assert verify(artifact)
    assert not (artifact / 'snapshot/input/outside').exists()
    task = json.loads((bundle / 'tasks/experience.json').read_text())
    assert [step['type'] for step in task] == ['deploy_sandbox', 'load_artifact', 'verify_sandbox']
    assert task[1]['artifact_id'] == 'experience'
    assert task[2]['criteria'][0]['bash_cmd'] == 'python3 verify.py'
    result = subprocess.run([sys.executable, str(artifact / 'verify.py')], cwd=tmp_path)
    assert result.returncode == 0
    first = (artifact / 'snapshot.json').read_bytes()
    export_agent_env(source)
    assert (artifact / 'snapshot.json').read_bytes() == first
    assert not list((artifact / 'snapshot').rglob('agent_env'))
    (artifact / 'snapshot/input/media.bin').write_bytes(b'changed')
    assert not verify(artifact)
    assert subprocess.run([sys.executable, str(artifact / 'verify.py')]).returncode == 1


@pytest.mark.parametrize('path', ['../escape', '/tmp/escape'])
def test_verifier_rejects_escaping_paths(tmp_path, path):
    (tmp_path / 'snapshot').mkdir()
    (tmp_path / 'snapshot.json').write_text(json.dumps({'files': [{'path': path, 'sha256': 'bad'}]}))
    assert not verify(tmp_path)


def test_exports_rebuild_agent_env_for_older_packages(tmp_path):
    from agent_apprenticeship.cli import _copy_or_zip_package
    import zipfile
    source = tmp_path / 'source'
    source.mkdir()
    (source / 'package_manifest.json').write_text('{}')
    output = _copy_or_zip_package(source, tmp_path / 'export', item_id='test', source='test')
    assert verify(output / 'package/agent_env/artifacts/experience')
    archive = _copy_or_zip_package(source, tmp_path / 'export.zip', item_id='test', source='test')
    with zipfile.ZipFile(archive) as zf:
        assert any(name.endswith('/package/agent_env/tasks/experience.json') for name in zf.namelist())


def test_compiler_disabled_and_experience_pack(tmp_path, monkeypatch):
    monkeypatch.setenv('AA_HOME', str(tmp_path / 'home'))
    monkeypatch.setenv('AA_TDO_DISABLED', '1')
    from agent_apprenticeship.config import get_settings
    from agent_apprenticeship.tdo import run_tdo_for_package
    from agent_apprenticeship.learning import compile_experience_pack
    package = tmp_path / 'package'
    package.mkdir()
    run_tdo_for_package(package, get_settings())
    assert verify(package / 'agent_env/artifacts/experience')
    pack = compile_experience_pack([{'task_id': 'test', 'title': 'test'}])
    assert verify(pack / 'agent_env/artifacts/experience')


def test_regeneration_removes_deleted_artifacts(tmp_path):
    (tmp_path / 'retained.json').write_text('{}')
    (tmp_path / 'deleted.json').write_text('{}')
    export_agent_env(tmp_path)
    (tmp_path / 'deleted.json').unlink()
    bundle = export_agent_env(tmp_path)
    assert not (bundle / 'artifacts/experience/snapshot/deleted.json').exists()
    assert verify(bundle / 'artifacts/experience')
