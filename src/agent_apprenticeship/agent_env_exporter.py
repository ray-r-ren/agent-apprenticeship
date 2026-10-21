from __future__ import annotations

import hashlib
import os
from pathlib import Path
import shutil
import tempfile

from .io import validate_trace_tree, write_json

_EXCLUDED = {'agent_env', '__pycache__', 'node_modules', 'build', 'dist', 'roles'}


def export_agent_env(source: Path) -> Path:
    source = source.resolve()
    destination = source / 'agent_env'
    validate_trace_tree(source)
    with tempfile.TemporaryDirectory(prefix='.agentenv-', dir=source) as temp:
        staging = Path(temp)
        artifact = staging / 'artifacts' / 'experience'
        snapshot = artifact / 'snapshot'
        snapshot.mkdir(parents=True)
        files = []
        for directory, dirs, names in os.walk(source, followlinks=False):
            dirs[:] = sorted(d for d in dirs if not d.startswith('.') and d not in _EXCLUDED
                             and not (Path(directory) / d).is_symlink())
            for name in sorted(names):
                path = Path(directory) / name
                if name.startswith('.') or path.is_symlink() or not path.is_file():
                    continue
                if name.endswith(('.pyc', '.log')) or name in {'stdout.txt', 'stderr.txt', 'rejected_agent_response.txt'}:
                    continue
                relative = path.relative_to(source)
                target = snapshot / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, target)
                files.append({'path': relative.as_posix(), 'sha256': hashlib.sha256(target.read_bytes()).hexdigest()})
        write_json(artifact / 'snapshot.json', {'files': files})
        shutil.copyfile(Path(__file__).with_name('agent_env_verify.py'), artifact / 'verify.py')
        write_json(staging / 'tasks' / 'experience.json', [
            {'id': 'sandbox', 'type': 'deploy_sandbox', 'sandbox_name': 'experience',
             'sandbox_mode': 'vm', 'sandbox_type': 'local'},
            {'id': 'restore', 'type': 'load_artifact', 'sandbox_name': 'experience',
             'artifact_id': 'experience', 'destination_path': '/app/experience', 'depends_on': ['sandbox']},
            {'id': 'verify', 'type': 'verify_sandbox', 'sandbox_name': 'experience',
             'base_dir': '/app/experience', 'verifier_id': 'experience_integrity', 'depends_on': ['restore'],
             'criteria': [{'type': 'bash_cmd_succeeds', 'criterion': 'snapshot_integrity',
                           'bash_cmd': 'python3 verify.py'}]},
        ])
        if destination.exists():
            shutil.rmtree(destination)
        staging.rename(destination)
    return destination
