from __future__ import annotations
import mimetypes, shutil
from pathlib import Path
from .schemas import ArtifactRef
from .io import sha256_file
from .env import contains_secret

def media_type_for(path: Path) -> str:
    suf=path.suffix.lower()
    if suf in ['.txt','.md','.log','.json','.jsonl','.csv','.yaml','.yml']: return 'text' if suf in ['.txt','.md','.log'] else 'data'
    if suf in ['.py','.js','.ts','.sh','.html','.css']: return 'code'
    return 'unknown'

def artifact_ref(path: Path, package_root: Path, task_id: str, attempt_id: str | None, kind='output', role='other') -> ArtifactRef:
    rel = path.relative_to(package_root).as_posix() if path.is_relative_to(package_root) else path.name
    text = path.read_text(errors='ignore') if path.exists() and path.stat().st_size < 2_000_000 else ''
    return ArtifactRef(artifact_id=rel.replace('/','__'), task_id=task_id, attempt_id=attempt_id, artifact_kind=kind, artifact_role=role, workspace_path=str(path), package_relative_path=rel, release_relative_path=None, mime_type=mimetypes.guess_type(path.name)[0], media_type=media_type_for(path), size_bytes=path.stat().st_size if path.exists() else None, content_hash=sha256_file(path) if path.exists() else None, secret_scan_ok=not contains_secret(text), metadata_json={})

def copy_inputs(input_refs: list[str], dest: Path):
    dest.mkdir(parents=True, exist_ok=True)
    for ref in input_refs:
        p=Path(ref)
        if p.exists() and p.is_file(): shutil.copy2(p, dest/p.name)
