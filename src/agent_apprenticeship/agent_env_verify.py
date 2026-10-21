import hashlib
import json
from pathlib import Path


def verify(root):
    root = Path(root)
    snapshot = (root / 'snapshot').resolve()
    manifest = json.loads((root / 'snapshot.json').read_text(encoding='utf-8'))
    files = manifest['files']
    if not isinstance(files, list) or not files:
        return False
    seen = set()
    for entry in files:
        relative = Path(entry['path'])
        path = snapshot / relative
        if relative.is_absolute() or '..' in relative.parts or path.is_symlink():
            return False
        if not path.resolve().is_relative_to(snapshot) or not path.is_file() or relative.as_posix() in seen:
            return False
        seen.add(relative.as_posix())
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry['sha256']:
            return False
    return True


if __name__ == '__main__':
    try:
        passed = verify(Path(__file__).resolve().parent)
    except (OSError, ValueError, KeyError, TypeError):
        passed = False
    raise SystemExit(0 if passed else 1)
